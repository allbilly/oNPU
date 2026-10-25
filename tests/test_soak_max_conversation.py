"""Exercise cold/cached classification with a fake tokenizer and HTTP worker."""
import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from scripts import soak_context, soak_max_conversation


def run_fixture(tmp_path, monkeypatch, *, cached_tokens, max_tokens=8):
    class Tokenizer:
        eos_id = 0

        @classmethod
        def from_gguf_kv(cls, _kv):
            return cls()

        def encode(self, text):
            return text.split()

        def decode(self, tokens):
            return " ".join(map(str, tokens))

    def render(*, messages, **_kwargs):
        return " ".join(f"{message['role']} {message['content']} end" for message in messages) + " assistant"

    class Environment:
        def __init__(self):
            self.filters, self.globals = {}, {}

        def from_string(self, _text):
            return SimpleNamespace(render=render)

    for name in ("tinygrad", "tinygrad.llm", "tinygrad.llm.cli", "tinygrad.llm.gguf"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["tinygrad.llm.cli"].SimpleTokenizer = Tokenizer
    sys.modules["tinygrad.llm.gguf"].gguf_parse = lambda _path: ({
        "general.architecture": "test", "test.context_length": 512, "tokenizer.chat_template": "fixture"}, {})
    monkeypatch.setattr("jinja2.sandbox.ImmutableSandboxedEnvironment", Environment)
    metrics = None
    requests = []

    def handle(request):
        nonlocal metrics
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/status":
            return httpx.Response(200, json={"hardware": {"memory": {}, "temperature": None}, "runtime": {
                "selected": "other-model", "sessions": [
                    {"model_id": "other-model", "state": "ready", "context": 1024, "active_request": None,
                     "rss_bytes": 1, "last_metrics": {"cached_tokens": 500}},
                    {"model_id": "target-model", "state": "ready", "context": 512, "active_request": None,
                     "rss_bytes": 1, "last_metrics": metrics}]}})
        if request.url.path == "/api/chat/stop":
            assert request.url.params["model"] == "target-model"
            return httpx.Response(200, json={})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "target-model"
        prompt_tokens = len(render(messages=body["messages"]).split())
        if prompt_tokens + body["max_tokens"] > 512:
            return httpx.Response(400, json={"error": "Context capacity exceeded"})
        first = metrics is None
        metrics = {"cached_tokens": cached_tokens if first else 383, "prompt_tokens": prompt_tokens,
                   "ttft_ms": 10, "prefill_tps": 100}
        answer = "NEPTUNE" if len(body["messages"]) > 1 else "ORCHID"
        event = {"choices": [{"delta": {"content": answer}}],
                 "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")

    original_client = httpx.AsyncClient
    monkeypatch.setattr(soak_max_conversation.httpx, "AsyncClient", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["soak_max_conversation", "--model", "target-model", "--checkpoint",
                                     str(tmp_path / "fake.gguf"), "--report", str(report_path), "--max-tokens", str(max_tokens)])
    return report_path, requests


def test_preseeded_prefix_cannot_be_reported_as_cold(tmp_path, monkeypatch):
    path, requests = run_fixture(tmp_path, monkeypatch, cached_tokens=383)
    with pytest.raises(RuntimeError, match="Unload and reload the target model"):
        asyncio.run(soak_max_conversation.main())
    report = json.loads(path.read_text())
    assert not report["complete"]
    assert "not an uncached maximum-context measurement" in report["error"]
    cold, = report["cases"]
    assert cold["metrics"]["cached_tokens"] == 383 and not cold["cold_pass"] and not cold["api_pass"]
    assert requests.count(("POST", "/v1/chat/completions")) == 1
    assert requests.count(("POST", "/api/chat/stop")) == 1
    assert all("load" not in path for _, path in requests)


def test_zero_cache_and_cached_followups_complete_for_the_requested_model(tmp_path, monkeypatch):
    path, _requests = run_fixture(tmp_path, monkeypatch, cached_tokens=0)
    asyncio.run(soak_max_conversation.main())
    report = json.loads(path.read_text())
    assert report["complete"] and "error" not in report
    cold, repeated, growing = report["cases"]
    assert cold["api_pass"] and cold["cold_pass"] and cold["metrics"]["cached_tokens"] == 0
    assert repeated["api_pass"] and repeated["matches_cold"] and repeated["metrics"]["cached_tokens"] == 383
    assert growing["api_pass"] and growing["correct"] and growing["prompt_tokens"] == 504


def test_growing_prompt_reserves_the_configured_reply_budget(tmp_path, monkeypatch):
    path, _requests = run_fixture(tmp_path, monkeypatch, cached_tokens=0, max_tokens=16)
    asyncio.run(soak_max_conversation.main())
    report = json.loads(path.read_text())
    assert report["complete"]
    assert report["cases"][-1]["prompt_tokens"] == 512 - 16


@pytest.mark.parametrize("empty_stage,no_op_stop", [("stop", False), ("disconnect", False), (None, False), (None, True)])
def test_context_soak_rejects_empty_cancellation_streams(tmp_path, monkeypatch, empty_stage, no_op_stop):
    original_client = httpx.AsyncClient
    run_fixture(tmp_path, monkeypatch, cached_tokens=0)  # Install the fixture tokenizer/template.
    streams = 0
    active = False
    current_stream = None
    closed = []

    class CancellationStream(httpx.AsyncByteStream):
        def __init__(self, stage):
            self.stage = stage
            self.cancelled = asyncio.Event()

        async def __aiter__(self):
            nonlocal active
            if self.stage == empty_stage:
                return
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            await self.cancelled.wait()
            active = False
            yield b'data: {"error":{"message":"Generation stopped"}}\n\ndata: [DONE]\n\n'

        async def aclose(self):
            nonlocal active
            closed.append((self.stage, active))
            active = False

    def handle(request):
        nonlocal streams, active, current_stream
        if request.url.path == "/api/status":
            return httpx.Response(200, json={"hardware": {"memory": {"available": 1}, "temperature": None},
                "runtime": {"sessions": [{"model_id": "target", "state": "generating" if active else "ready", "context": 8192,
                                          "active_request": "target-request-1" if active else None, "rss_bytes": 1}]}})
        if request.url.path == "/api/chat/stop":
            assert dict(request.url.params) == {"model": "target", "request_id": "target-request-1"}
            if not no_op_stop:
                current_stream.cancelled.set()
            return httpx.Response(200, json={})
        if not json.loads(request.content).get("stream"):
            return httpx.Response(409, json={"error": "busy"})
        streams += 1
        stage = "stop" if streams == 1 else "disconnect"
        active = True
        current_stream = CancellationStream(stage)
        return httpx.Response(200, headers={"content-type": "text/event-stream", "X-Request-ID": f"target-request-{streams}"},
                              stream=current_stream)

    monkeypatch.setattr(soak_context.httpx, "AsyncClient", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(soak_context, "STOP_TIMEOUT", .02)
    path = tmp_path / "soak.json"
    monkeypatch.setattr(sys, "argv", ["soak_context", "--model", "target", "--checkpoint", str(tmp_path / "fake.gguf"),
                                     "--context", "8192", "--report", str(path), "--labels", "not-selected", "--repeats", "0"])
    if empty_stage or no_op_stop:
        message = "before cancellation" if empty_stage else "Stop did not cancel the connected request"
        with pytest.raises(AssertionError, match=message):
            asyncio.run(soak_context.main())
    else:
        asyncio.run(soak_context.main())
    report = json.loads(path.read_text())
    if empty_stage or no_op_stop:
        assert not report["complete"]
        failed_stage = empty_stage or "stop"
        assert not any(row["label"] == failed_stage and row["api_pass"] for row in report["cases"])
        if no_op_stop:
            assert closed == [("stop", True)]
    else:
        assert report["complete"] and report["api_failures"] == 0
        assert closed == [("stop", False), ("disconnect", True)]


def test_context_soak_keeps_short_conversation_growth_inside_capacity(tmp_path, monkeypatch):
    original_client = httpx.AsyncClient
    run_fixture(tmp_path, monkeypatch, cached_tokens=0)

    def handle(request):
        assert request.url.path == "/api/status", "Fixture generation must not send chat requests"
        return httpx.Response(200, json={"hardware": {"memory": {"available": 1}, "temperature": None},
            "runtime": {"sessions": [{"model_id": "target", "state": "ready", "context": 512,
                                      "active_request": None, "rss_bytes": 1}]}})

    monkeypatch.setattr(soak_context.httpx, "AsyncClient", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    path, fixtures = tmp_path / "soak.json", tmp_path / "fixtures"
    monkeypatch.setattr(sys, "argv", ["soak_context", "--model", "target", "--checkpoint", str(tmp_path / "fake.gguf"),
                                     "--context", "512", "--report", str(path), "--labels", "conversation-turn-1,conversation-turn-2",
                                     "--repeats", "0", "--fixtures-only", "--fixtures", str(fixtures)])
    asyncio.run(soak_context.main())
    assert json.loads(path.read_text())["complete"]
    inputs = sorted(fixtures.glob("*.json"))
    assert [file.stem for file in inputs] == ["conversation-turn-1", "conversation-turn-2"]
    for file, target in zip(inputs, [256, 504], strict=True):
        body = json.loads(file.read_text())
        tokens = sum(len(message["content"].split()) + 2 for message in body["messages"]) + 1
        assert tokens == target
        assert tokens + body["max_tokens"] <= 512
