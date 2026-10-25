"""Verification reports must follow the requested worker and fail their gates."""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from scripts import (
    benchmark_placement,
    smoke,
    smoke_vlm,
    soak_context,
    soak_max_conversation,
)

NODE = shutil.which("node")
BROWSER_VLM = Path(__file__).resolve().parent / "browser_vlm.mjs"
CDP_HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const scenario = process.argv[2], exception = scenario !== 'clean';
const reports = [];
let savedReport = { complete: true, previousRun: true };
const runtime = { state: 'ready', supports_images: true, actual_device: 'ROCKCHIP + CL',
  placement: { vision: { gpu_vision: true, prepared_mlp: 12 } } };
const context = vm.createContext({
  process: { argv: [] }, Buffer, console: { log() {} }, setTimeout, clearTimeout,
  fs: { writeFileSync(path, contents) { if (path.endsWith('.json')) { savedReport = JSON.parse(contents); reports.push(savedReport); } } },
  fetch: async (url) => {
    if (scenario === 'connection-error') throw Error('CDP unavailable');
    return { json: async () => url.endsWith('/json/list')
      ? [{ type: 'page', webSocketDebuggerUrl: 'ws://mock-browser' }] : { runtime } };
  },
  WebSocket: class {
    constructor() { queueMicrotask(() => this.onopen()); }
    close() {}
    send(raw) {
      const command = JSON.parse(raw);
      let result = {};
      if (command.method === 'DOM.getDocument') result = { root: { nodeId: 1 } };
      if (command.method === 'DOM.querySelector') result = { nodeId: 2 };
      if (command.method === 'Page.captureScreenshot') result = { data: '' };
      if (command.method === 'Runtime.evaluate') result = { result: { value:
        command.params.expression.startsWith('({text:') ? { text: 'An astronaut.', error: [], images: 1 } : true } };
      queueMicrotask(() => {
        if (exception && command.method === 'Runtime.enable') this.onmessage({ data: JSON.stringify({
          method: 'Runtime.exceptionThrown', params: { exceptionDetails: { text: 'Uncaught UI error' } } }) });
        this.onmessage({ data: JSON.stringify({ id: command.id, result }) });
      });
    }
  },
});
const source = fs.readFileSync(process.argv[1], 'utf8').replace("import fs from 'node:fs';", '');
vm.runInContext(`(async () => { ${source}\n })()`, context).then(() => {
  assert.equal(exception, false, 'A browser exception must prevent successful verification');
  assert.equal(reports[0].complete, false, 'Reset previous verification before starting');
  assert.equal(savedReport.complete, true);
}).catch((error) => {
  if (!exception || !/Uncaught UI error|CDP unavailable/.test(error.message)) throw error;
  assert.equal(savedReport.complete, false, 'A failed rerun must overwrite previous success');
}).catch((error) => { console.error(error); process.exitCode = 1; });
"""


def run_placement_fixture(tmp_path, monkeypatch, *, selected_other=False, diverge=False, fail_load=False):
    mode = "off"
    metrics = {"decode_tps": 100, "ttft_ms": 10}

    def handle(request):
        nonlocal mode
        if request.url.path == "/api/models/target/load":
            if fail_load:
                assert not json.loads(path.read_text())["complete"], "Previous success must reset before a new request"
                return httpx.Response(409, json={"error": "Worker busy"})
            mode = json.loads(request.content)["npu_decode"]
            return httpx.Response(202, json={})
        if request.url.path == "/api/status":
            target = {"model_id": "target", "state": "ready", "actual_device": "CL", "placement": {},
                      "last_metrics": metrics}
            other = {**target, "model_id": "other", "last_metrics": {"decode_tps": 999, "ttft_ms": 999}}
            return httpx.Response(200, json={"runtime": {**(other if selected_other else target),
                                                        "sessions": [other, target]}})
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["model"] == "target"
        answer = "changed output" if diverge and mode == "auto" else "baseline output"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(benchmark_placement.httpx, "AsyncClient", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    path = tmp_path / "placement.json"
    monkeypatch.setattr(sys, "argv", ["benchmark_placement", "--model", "target", "--report", str(path)])
    return path


def test_placement_report_uses_requested_models_metrics(tmp_path, monkeypatch):
    path = run_placement_fixture(tmp_path, monkeypatch, selected_other=True)
    asyncio.run(benchmark_placement.main())
    report = json.loads(path.read_text())
    assert report["complete"]
    assert all(row["median_decode_tps"] == 100 and row["median_ttft_ms"] == 10 for row in report["summary"])


def test_placement_output_mismatch_cannot_report_completion(tmp_path, monkeypatch):
    path = run_placement_fixture(tmp_path, monkeypatch, diverge=True)
    with pytest.raises(AssertionError, match="Placement changed"):
        asyncio.run(benchmark_placement.main())
    report = json.loads(path.read_text())
    assert not report["complete"]
    assert report["response_matches_gpu_baseline"] == [True, False, True]
    monkeypatch.setattr(sys, "argv", ["soak_max_conversation", "--model", "target", "--checkpoint", "unused.gguf",
                                     "--report", str(tmp_path / "following.json"), "--wait-report", str(path)])

    async def wait_for_failure():
        with pytest.raises(RuntimeError, match="Preceding benchmark failed.*Placement changed"):
            await asyncio.wait_for(soak_max_conversation.main(), timeout=.1)

    asyncio.run(wait_for_failure())


def test_placement_rerun_clears_success_and_records_early_failure(tmp_path, monkeypatch):
    path = run_placement_fixture(tmp_path, monkeypatch, fail_load=True)
    path.write_text(json.dumps({"complete": True, "previous_run": True}))
    with pytest.raises(AssertionError, match="Worker busy"):
        asyncio.run(benchmark_placement.main())
    report = json.loads(path.read_text())
    assert not report["complete"] and "Worker busy" in report["error"]
    assert report["elapsed_seconds"] >= 0


@pytest.mark.skipif(NODE is None, reason="Node.js is required for browser verification tests")
@pytest.mark.parametrize("scenario", ["clean", "exception", "connection-error"])
def test_vlm_browser_report_rejects_uncaught_exceptions(scenario):
    result = subprocess.run([NODE, "-e", CDP_HARNESS, str(BROWSER_VLM), scenario],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", [smoke, smoke_vlm], ids=["text", "vision"])
@pytest.mark.parametrize("empty_stage,no_op_stop", [("stop", False), ("disconnect", False), (None, False), (None, True)])
def test_smoke_requires_started_cancellation_streams(tmp_path, monkeypatch, module, empty_stage, no_op_stop):
    long_requests = 0
    stopped = False
    active = False
    current_stream = None
    closed = []
    metrics = {"prompt_tokens": 4, "completion_tokens": 1, "npu_submissions": 1}

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
            # Disconnect works independently, even when the Stop endpoint is a no-op.
            active = False

    def handle(request):
        nonlocal long_requests, stopped, active, current_stream
        if request.url.path == "/api/status":
            state = {"model_id": "target", "selected_profile": "CL", "actual_device": "CL" if module is smoke else "ROCKCHIP",
                     "state": "generating" if active else "ready", "context": 512, "supports_images": True, "last_metrics": metrics}
            return httpx.Response(200, json={"runtime": {**state, "sessions": [state]}})
        if request.url.path == "/api/chat/stop":
            assert dict(request.url.params) == {"model": "target", "request_id": "target-request-1"}
            stopped = True
            if not no_op_stop:
                current_stream.cancelled.set()
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        if body["max_tokens"] == 512:
            return httpx.Response(400, json={"error": "capacity"})
        if body.get("stream"):
            if body["max_tokens"] == 128:
                long_requests += 1
                stage = "stop" if long_requests == 1 else "disconnect"
                active = True
                current_stream = CancellationStream(stage)
                return httpx.Response(200, headers={"content-type": "text/event-stream", "X-Request-ID": f"target-request-{long_requests}"},
                                      stream=current_stream)
            else:
                text = 'data: ' + json.dumps({"choices": [{"delta": {"content": "hello"}}], "usage": metrics}) + '\n\ndata: [DONE]\n\n'
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)
        if active:
            return httpx.Response(409, json={"error": "busy"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}], "usage": metrics})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(module, "STOP_TIMEOUT", .02)
    argv = ["smoke", "--model", "target"]
    if module is smoke_vlm:
        image = tmp_path / "image.jpg"
        image.write_bytes(b"fixture image")
        argv += ["--image", str(image), "--report", str(tmp_path / "vision.json")]
    monkeypatch.setattr(sys, "argv", argv)
    if empty_stage:
        with pytest.raises(AssertionError, match="before cancellation"):
            asyncio.run(module.main())
    elif no_op_stop:
        with pytest.raises(AssertionError, match="Stop did not cancel the connected request"):
            asyncio.run(module.main())
        assert stopped and closed == [("stop", True)]
    else:
        asyncio.run(module.main())
        assert stopped and long_requests == 2
        assert closed == [("stop", False), ("disconnect", True)]


@pytest.mark.parametrize("module", [smoke_vlm, soak_context, soak_max_conversation])
def test_reporting_harness_resets_previous_success_before_preflight(tmp_path, monkeypatch, module):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"complete": True, "previous_run": True}))

    async def fail_preflight(*_args):
        assert not json.loads(path.read_text())["complete"]
        raise RuntimeError("Preflight failed after resetting the report")

    monkeypatch.setattr(module, "verify", fail_preflight)
    argv = ["verification", "--model", "target", "--report", str(path)]
    argv += ["--image", str(tmp_path / "missing.jpg")] if module is smoke_vlm else [
        "--checkpoint", str(tmp_path / "missing.gguf")]
    if module is soak_context:
        argv += ["--context", "512"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="Preflight failed"):
        asyncio.run(module.main())
    report = json.loads(path.read_text())
    assert not report["complete"] and "previous_run" not in report
    assert "Preflight failed" in report["error"] and report["elapsed_seconds"] >= 0
