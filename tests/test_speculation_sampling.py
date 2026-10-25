"""CPU fake coverage for sampled speculative routing and iterator ownership."""
from __future__ import annotations

import json
import sys
import threading
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

from onpu.speculation import TARGET_SHA256, speculative_chunks, use_speculation


class FakeTokenizer:
    bos_id, eos_id, eot_id, eom_id = 0, 91, 90, None
    preset = "qwen2"

    def decode(self, ids):
        return "".join({0: "", 1: "hello", 2: " world", 90: "<eot>", 91: "<eos>"}[token] for token in ids)

    def stream_decoder(self):
        return lambda token=None: "" if token is None else self.decode([token])

    def is_end(self, token):
        return token in (self.eos_id, self.eot_id, self.eom_id)


class FakeSource:
    def __init__(self, values, events=None):
        self.values, self.events, self.closed = list(values), events if events is not None else [], False

    def __iter__(self):
        return iter(self.values)

    def close(self):
        self.closed = True
        self.events.append("source_closed")


class FakeChunks:
    def __init__(self, rows=(), error=None):
        self.rows, self.error, self.closed = list(rows), error, False
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        if self.index >= len(self.rows):
            raise StopIteration
        result = self.rows[self.index]
        self.index += 1
        return result

    def close(self):
        self.closed = True


class FakeFormatHandler:
    def __init__(self, rows=(), error=None):
        self.server = SimpleNamespace(tok=FakeTokenizer())
        self.rows, self.error, self.temperatures, self.chunks = rows, error, [], []

    def run_model(self, ids, model_name, **kwargs):
        self.temperatures.append(kwargs["temperature"])
        chunks = FakeChunks(self.rows, self.error)
        self.chunks.append(chunks)
        return chunks


class FakeSampledDecoder:
    ordinary_decode_seconds = 0.4

    def __init__(self, values=(1, 2, 91)):
        self.values, self.samples, self.generations, self.sources = values, [], [], []

    def sample(self, ids, *, temperature, **kwargs):
        self.samples.append((list(ids), temperature, dict(kwargs)))
        source = FakeSource(self.values)
        self.sources.append(source)
        return source

    def generate(self, ids, **kwargs):
        self.generations.append((list(ids), dict(kwargs)))
        source = FakeSource(self.values)
        self.sources.append(source)
        return source


@dataclass
class StatsFixture:
    proposed_tokens: int = 3
    accepted_tokens: int = 2


class DraftMetricsFixture:
    def proposal_preparation_metadata(self):
        return {"enabled": True, "graphs": 2}

    def observation_preparation_metadata(self):
        return {"enabled": True, "graphs": 3, "native_copy_fallbacks": 1}


class SamplingRouteTests(unittest.TestCase):
    def test_use_speculation_selects_sampled_only_for_supported_auto_requests(self):
        sampled = FakeSampledDecoder()
        greedy_only = SimpleNamespace(generate=lambda *args, **kwargs: iter(()))
        self.assertTrue(use_speculation(sampled, 0.0, "auto"))
        self.assertTrue(use_speculation(sampled, 0.6, "auto"))
        self.assertFalse(use_speculation(greedy_only, 0.6, "auto"))
        self.assertFalse(use_speculation(sampled, 0.6, "off"))
        self.assertFalse(use_speculation(sampled, 0.0, "off"))
        self.assertFalse(use_speculation(None, 0.6, "auto"))
        self.assertFalse(use_speculation(sampled, -0.1, "auto"))
        self.assertFalse(use_speculation(sampled, float("nan"), "auto"))
        with self.assertRaises(ValueError):
            use_speculation(sampled, 0.6, "dflash2-experimental")

    def test_positive_temperature_uses_sampled_source_and_forwards_formatter_temperature(self):
        decoder = FakeSampledDecoder()
        handler = FakeFormatHandler(rows=({"choices": [{"delta": {"content": "hello"}}]},))
        stream = speculative_chunks(handler, decoder, [7, 8], "target", max_tokens=4, reasoning=False,
                                    cancelled=lambda: False, cancel_error=RuntimeError, temperature=0.6)
        chunks = list(stream)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(decoder.generations, [])
        self.assertEqual(decoder.samples[0][0:2], ([7, 8], 0.6))
        self.assertEqual(decoder.samples[0][2], {
            "max_new_tokens": 4, "eos_token_ids": {90, 91}, "cancelled": mock.ANY,
            "prefill_chunk_size": 128, "ordinary_decode_seconds": 0.4,
        })
        self.assertEqual(handler.temperatures, [0.6])
        self.assertTrue(decoder.sources[0].closed)
        self.assertTrue(handler.chunks[0].closed)

    def test_off_request_uses_ordinary_generation_baseline(self):
        decoder = FakeSampledDecoder()
        self.assertFalse(use_speculation(decoder, 0.6, "off"))
        handler = FakeFormatHandler()
        chunks = list(handler.run_model([7, 8], "target", temperature=0.6))
        self.assertEqual(chunks, [])
        self.assertEqual(handler.temperatures, [0.6])
        self.assertEqual(decoder.samples, [])

    def test_cancel_and_formatter_errors_close_sampled_source(self):
        decoder = FakeSampledDecoder()
        handler = FakeFormatHandler(rows=({"delta": "chunk"},))
        stream = speculative_chunks(handler, decoder, [1], "target", max_tokens=2, reasoning=False,
                                    cancelled=lambda: True, cancel_error=RuntimeError, temperature=0.6)
        with self.assertRaises(RuntimeError):
            next(stream)
        self.assertTrue(decoder.sources[0].closed)
        self.assertTrue(handler.chunks[0].closed)

        decoder = FakeSampledDecoder()
        handler = FakeFormatHandler(error=KeyError("formatter"))
        stream = speculative_chunks(handler, decoder, [1], "target", max_tokens=2, reasoning=False,
                                    cancelled=lambda: False, cancel_error=RuntimeError, temperature=0.6)
        with self.assertRaises(KeyError):
            next(stream)
        self.assertTrue(decoder.sources[0].closed)
        self.assertTrue(handler.chunks[0].closed)

def test_worker_mock_routes_sampled_temperature_and_off_baseline(monkeypatch, tmp_path):
    check = unittest.TestCase()
    import tinygrad
    from tinygrad import nn
    from tinygrad.llm import cli
    from tinygrad.llm.kernels import mali
    from tinygrad.llm.model import Transformer

    from onpu import worker

    events, terminals = [], threading.Event()
    ordinary_calls, startup_order = [], []
    model = SimpleNamespace(max_context=512, _cached_tokens=[], warmup=lambda: None,
                            get_start_pos=lambda ids: 0)

    def ordinary_generate(ids, **kwargs):
        startup_order.append("generate")
        ordinary_calls.append((list(ids), kwargs.get("temperature", 0.0)))
        return iter((1, 2, 91))

    model.generate = ordinary_generate

    class Protocol:
        def write(self, line):
            event = json.loads(line)
            events.append(event)
            if event.get("event") == "done":
                terminals.set()

        def flush(self):
            pass

    class Input:
        def __iter__(self):
            requests = (
                {"id": "sampled", "body": {"model": "target", "max_tokens": 3, "temperature": 0.6,
                                               "messages": [], "speculation": "auto"}},
                {"id": "ordinary", "body": {"model": "target", "max_tokens": 3, "temperature": 0.0,
                                                "messages": [], "speculation": "off"}},
            )
            for request in requests:
                yield json.dumps({"op": "generate", **request})
                if not terminals.wait(3):
                    raise AssertionError("Worker did not finish fake request")
                terminals.clear()
            yield json.dumps({"op": "shutdown"})

    class WorkerTokenizer(FakeTokenizer):
        def encode(self, text):
            return [7, 8]

    class WorkerDecoder(FakeSampledDecoder):
        def __init__(self, values):
            super().__init__(values)
            self.stats, self.draft = StatsFixture(), DraftMetricsFixture()

        def sample(self, ids, *, temperature, **kwargs):
            self.samples.append((list(ids), temperature, dict(kwargs)))
            source = FakeSource(self.values)
            self.sources.append(source)
            return source

    sampled_decoder = WorkerDecoder((1, 2, 91))
    sampled_decoder.serving_setup = {"prepared_warmup": {"prompt_observation_calibration": {
        "seconds_per_token": .001, "trace_seconds": 0.0}}}
    monkeypatch.setattr(sys, "stdout", Protocol())
    monkeypatch.setattr(sys, "stdin", Input())
    monkeypatch.setattr(sys, "argv", ["worker", "--model", str(tmp_path / "target.gguf"), "--context", "512",
                                      "--speculation", "dflash2-experimental", "--draft-path", str(tmp_path)])
    monkeypatch.setenv("ONPU_PROFILE", "CL")
    monkeypatch.setenv("ONPU_NPU_PREFILL", "off")
    monkeypatch.setenv("ONPU_NPU_DECODE", "off")
    monkeypatch.setattr(tinygrad, "Device", SimpleNamespace(DEFAULT="CL"))
    monkeypatch.setattr(Transformer, "from_gguf", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr(nn.state, "get_parameters", lambda _model: [SimpleNamespace(device="CL")])
    monkeypatch.setattr(cli.SimpleTokenizer, "from_gguf_kv", lambda _metadata: WorkerTokenizer())
    monkeypatch.setattr(mali, "configure", lambda *args, **kwargs: startup_order.append("configure") or {"linear_layers": 0, "prefill_chunk": 128})
    monkeypatch.setattr(worker, "mali_profile", lambda *args, **kwargs: (TARGET_SHA256, {"recurrent_scan": True, "pack_weights": True}))
    monkeypatch.setattr(worker, "validate_speculation", lambda *args: None)
    monkeypatch.setattr(worker, "prepare_target_table", lambda *args, **kwargs: startup_order.append("table"))
    monkeypatch.setattr(worker, "load_decoder", lambda *args: sampled_decoder)
    monkeypatch.setattr(worker, "prepare_prompt", lambda *_args: ("prompt", [7, 8]))
    worker.main()
    check.assertEqual(startup_order[:2], ["configure", "table"])
    check.assertLess(startup_order.index("table"), startup_order.index("generate"))

    ready = next(event for event in events if event.get("event") == "ready")
    check.assertTrue(ready["speculation"]["sampled"])
    check.assertFalse(ready["speculation"]["greedy_only"])
    sampled_event = next(event for event in events if event.get("event") == "done" and event.get("id") == "sampled")
    ordinary_event = next(event for event in events if event.get("event") == "done" and event.get("id") == "ordinary")
    check.assertEqual(sampled_event["metrics"]["generation_mode"], worker.SPECULATIVE_MODE)
    check.assertEqual(sampled_event["metrics"]["speculation_routing"]["reason"], "setup_amortizable")
    check.assertEqual(ordinary_event["metrics"]["generation_mode"], "ordinary")
    check.assertEqual(ordinary_event["metrics"]["speculation_routing"]["reason"], "request_off")
    check.assertEqual(sampled_decoder.samples[0][0:2], ([7, 8], 0.6))
    check.assertTrue(sampled_decoder.sources[0].closed)
    sampled_text = "".join(choice.get("delta", {}).get("content", "")
                           for event in events if event.get("event") == "chunk" and event.get("id") == "sampled"
                           for choice in event["data"].get("choices", []))
    check.assertEqual(sampled_text, "hello world")
    check.assertIn(([7, 8], 0.0), ordinary_calls)
    check.assertEqual(len(sampled_decoder.samples), 1)
