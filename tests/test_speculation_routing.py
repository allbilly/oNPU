"""Pure request-level speculation cost routing contracts."""
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from onpu.speculation import MODE, route_speculation, use_speculation


def decoder(*, observation_rate=0.001, trace_seconds=0.01, ordinary_rate=0.1):
    return SimpleNamespace(
        ordinary_decode_seconds=ordinary_rate,
        serving_setup={"prepared_warmup": {"prompt_observation_calibration": {
            "prompt_tokens": 128,
            "samples_seconds": [0.2, 0.16, 0.18],
            "seconds_per_token": observation_rate,
            "trace_samples_seconds": [0.04, 0.01, 0.02],
            "trace_seconds": trace_seconds,
        }}},
    )


def route(target, *, mode="auto", temperature=0.0, prompt=20, reply=5, context=512, cached=0):
    return route_speculation(target, temperature, mode, prompt_tokens=prompt, max_tokens=reply,
                             max_context=context, cached_tokens=cached)


def test_route_auto_clamps_reply_budget_and_selects_when_setup_amortizes():
    selected = route(decoder(), prompt=20, reply=100, context=24)
    assert selected == {
        "requested": "auto", "selected": MODE, "reason": "setup_amortizable", "prompt_tokens": 20,
        "reply_budget": 4, "cached_tokens": 0, "estimated_setup_seconds": 0.03,
        "estimated_ordinary_decode_seconds": 0.30000000000000004,
    }


def test_route_auto_prefers_ordinary_for_long_prompt_or_cached_prefix():
    long_prompt = route(decoder(), prompt=400, reply=5, context=512)
    assert long_prompt["selected"] == "ordinary"
    assert long_prompt["reason"] == "draft_setup_cost"
    assert long_prompt["estimated_setup_seconds"] > long_prompt["estimated_ordinary_decode_seconds"]

    cached = route(decoder(), prompt=20, reply=5, cached=4)
    assert cached["selected"] == "ordinary" and cached["reason"] == "cached_prefix"
    assert cached["cached_tokens"] == 4


def test_route_budget_one_and_force_are_explicit():
    no_budget = route(decoder(), prompt=20, reply=8, context=21)
    assert no_budget["selected"] == "ordinary" and no_budget["reason"] == "no_decode_budget"
    forced = route(decoder(), mode="force", prompt=400, reply=1, context=401, cached=100)
    assert forced["selected"] == MODE and forced["reason"] == "forced_experiment"


@pytest.mark.parametrize("kwargs", [
    {"prompt": -1}, {"reply": -1}, {"prompt": 513}, {"cached": -1}, {"cached": 21, "prompt": 20},
    {"context": 0}, {"prompt": True},
])
def test_route_rejects_invalid_token_budgets(kwargs):
    with pytest.raises(ValueError, match="Invalid speculative routing token budget"):
        route(decoder(), **kwargs)


@pytest.mark.parametrize("target,temperature,mode", [
    (None, 0.0, "force"), (decoder(), -0.1, "force"), (decoder(), float("nan"), "force"),
    (decoder(), float("inf"), "force"),
])
def test_forced_route_rejects_missing_decoder_or_unsupported_temperature(target, temperature, mode):
    with pytest.raises(ValueError, match="Forced speculation requires"):
        route(target, temperature=temperature, mode=mode)


@pytest.mark.parametrize("calibration,ordinary", [
    (None, 0.1), ({"seconds_per_token": float("nan"), "trace_seconds": 0.01}, 0.1),
    ({"seconds_per_token": float("inf"), "trace_seconds": 0.01}, 0.1),
    ({"seconds_per_token": 0.001, "trace_seconds": float("inf")}, 0.1),
    ({"seconds_per_token": 0.001, "trace_seconds": 0.01}, float("nan")),
    ({"seconds_per_token": 0.001, "trace_seconds": 0.01}, float("inf")),
])
def test_route_falls_back_to_ordinary_when_calibration_is_missing_or_nonfinite(calibration, ordinary):
    target = decoder(ordinary_rate=ordinary)
    if calibration is None:
        target.serving_setup = {}
    else:
        target.serving_setup = {"prepared_warmup": {"prompt_observation_calibration": calibration}}
    result = route(target)
    assert result["selected"] == "ordinary"
    assert result["reason"] == "calibration_unavailable"


def test_route_uses_strict_cost_equality_as_ordinary():
    # prompt * rate + trace == (reply budget - 1) * ordinary rate
    target = decoder(observation_rate=0.015, trace_seconds=0.1, ordinary_rate=0.1)
    result = route(target, prompt=20, reply=5)
    assert result["estimated_setup_seconds"] == result["estimated_ordinary_decode_seconds"]
    assert result["selected"] == "ordinary" and result["reason"] == "draft_setup_cost"


def test_use_speculation_force_requires_eligible_loaded_decoder():
    assert use_speculation(decoder(), 0.0, "force")
    with pytest.raises(ValueError):
        use_speculation(None, 0.0, "force")
    with pytest.raises(ValueError):
        use_speculation(decoder(), -1.0, "force")


def test_worker_auto_rejects_unamortized_long_prompt_before_decoder_call(monkeypatch, tmp_path):
    import tinygrad
    from tinygrad import nn
    from tinygrad.llm import cli
    from tinygrad.llm.kernels import mali
    from tinygrad.llm.model import Transformer

    from onpu import worker
    from onpu.speculation import TARGET_SHA256

    events, done = [], threading.Event()
    ordinary_calls = []
    prompt_ids = list(range(8176))
    model = SimpleNamespace(max_context=8192, _cached_tokens=[], warmup=lambda: None,
                            get_start_pos=lambda _ids: 0)

    def ordinary_generate(ids, **kwargs):
        if ids == prompt_ids:
            ordinary_calls.append((list(ids), kwargs.get("temperature", 0.0)))
        return iter((1, 2, 91))

    model.generate = ordinary_generate

    class Tokenizer:
        bos_id, eos_id, eot_id, eom_id = 0, 91, 90, None
        preset = "qwen2"

        def decode(self, ids):
            return "".join({0: "", 1: "hello", 2: " world", 90: "<eot>", 91: "<eos>"}[token] for token in ids)

        def stream_decoder(self):
            return lambda token=None: "" if token is None else self.decode([token])

        def is_end(self, token):
            return token in (self.eos_id, self.eot_id, self.eom_id)

    class Protocol:
        def write(self, line):
            event = json.loads(line)
            events.append(event)
            if event.get("event") == "done":
                done.set()

        def flush(self):
            pass

    class Input:
        def __iter__(self):
            yield json.dumps({"op": "generate", "id": "long-auto", "body": {
                "model": "target", "max_tokens": 16, "temperature": 0.0, "messages": [], "speculation": "auto"}})
            assert done.wait(3), "Worker did not finish fake request"
            yield json.dumps({"op": "shutdown"})

    class Draft:
        ordinary_decode_seconds = 0.3

        def __init__(self):
            self.serving_setup = {"prepared_warmup": {"prompt_observation_calibration": {
                "prompt_tokens": 128, "samples_seconds": [0.3, 0.256, 0.28],
                "seconds_per_token": 0.002, "trace_samples_seconds": [0.02, 0.01, 0.03], "trace_seconds": 0.01}}}

        def generate(self, *_args, **_kwargs):
            raise AssertionError("Unamortized auto request must not enter speculative generation")

    monkeypatch.setattr(sys, "stdout", Protocol())
    monkeypatch.setattr(sys, "stdin", Input())
    monkeypatch.setattr(sys, "argv", ["worker", "--model", str(tmp_path / "target.gguf"), "--context", "8192",
                                      "--speculation", "dflash2-experimental", "--draft-path", str(tmp_path)])
    monkeypatch.setenv("ONPU_PROFILE", "CL")
    monkeypatch.setenv("ONPU_NPU_PREFILL", "off")
    monkeypatch.setenv("ONPU_NPU_DECODE", "off")
    monkeypatch.setattr(tinygrad, "Device", SimpleNamespace(DEFAULT="CL"))
    monkeypatch.setattr(Transformer, "from_gguf", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr(nn.state, "get_parameters", lambda _model: [SimpleNamespace(device="CL")])
    monkeypatch.setattr(cli.SimpleTokenizer, "from_gguf_kv", lambda _metadata: Tokenizer())
    monkeypatch.setattr(mali, "configure", lambda *args, **kwargs: {"linear_layers": 0, "prefill_chunk": 128})
    monkeypatch.setattr(worker, "mali_profile", lambda *args, **kwargs: (TARGET_SHA256, {"recurrent_scan": True, "pack_weights": True}))
    monkeypatch.setattr(worker, "validate_speculation", lambda *args: None)
    monkeypatch.setattr(worker, "prepare_target_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "load_decoder", lambda *args: Draft())
    monkeypatch.setattr(worker, "prepare_prompt", lambda *_args: ("prompt", prompt_ids))

    worker.main()

    started = next(event for event in events if event.get("event") == "started")
    terminal = next(event for event in events if event.get("event") == "done")
    routing = started["speculation_routing"]
    assert routing["selected"] == "ordinary" and routing["reason"] == "draft_setup_cost"
    assert routing["estimated_setup_seconds"] > routing["estimated_ordinary_decode_seconds"]
    assert terminal["metrics"]["speculation_routing"] == routing
    assert started["generation_mode"] == terminal["metrics"]["generation_mode"] == "ordinary"
    assert "speculation" not in terminal["metrics"]
    assert ordinary_calls == [(prompt_ids, 0.0)]
    text = "".join(choice.get("delta", {}).get("content", "")
                   for event in events if event.get("event") == "chunk"
                   for choice in event["data"].get("choices", []))
    assert text == "hello world"
