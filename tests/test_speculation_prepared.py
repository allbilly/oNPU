"""CPU-only contracts for prepared speculative target options and metrics."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from onpu import speculation
from onpu.app import create_app
from onpu.speculation import (
    MODE,
    calibrate_ordinary_decode,
    decoder_metrics,
    prepared_target_options,
    speculative_chunks,
    use_speculation,
    warm_prepared_decoder,
)


@dataclass
class StatsFixture:
    proposed_tokens: int = 7
    accepted_tokens: int = 4
    verify_calls: int = 2


class DraftMetricsFixture:
    def proposal_preparation_metadata(self):
        return {"enabled": True, "graphs": 2, "captured_graphs": 2, "calls": 7, "weight_copies": 0}

    def observation_preparation_metadata(self):
        return {
            "enabled": True,
            "graphs": 3,
            "max_graphs": 5,
            "native_cache_copy": True,
            "native_copy_appends": 4,
            "native_copy_commands": 6,
            "native_copy_bytes": 768,
            "native_copy_fallbacks": 2,
        }


class FakeTokens:
    def __init__(self, target, count: int, events=None):
        self.target, self.count, self.yielded, self.closed = target, count, 0, False
        self.events = events if events is not None else []

    def __iter__(self):
        return self

    def __next__(self):
        if self.yielded >= self.count:
            raise StopIteration
        self.yielded += 1
        self.target._cached_tokens.append(self.yielded)
        self.events.append(("yield", self.yielded))
        return self.yielded

    def close(self):
        self.closed = True
        self.events.append(("close", self.yielded))


class FakeOrdinaryModel:
    def __init__(self, max_context=512, yields_per_request=2):
        self.max_context = max_context
        self._cached_tokens = [999]
        self.yields_per_request = yields_per_request
        self.prompts, self.cache_at_generate, self.iterators, self.events = [], [], [], []

    def generate(self, prompt, *, temperature):
        self.prompts.append((list(prompt), temperature))
        self.cache_at_generate.append(list(self._cached_tokens))
        self.events.append(("generate", len(prompt)))
        tokens = FakeTokens(self, self.yields_per_request, self.events)
        self.iterators.append(tokens)
        return tokens


class FakePreparedDecoder:
    def __init__(self, max_context=512, yields_per_request=24):
        self.target = SimpleNamespace(max_context=max_context, _cached_tokens=[999])
        self.yields_per_request = yields_per_request
        self.calls, self.iterators = [], []
        self.stats = SimpleNamespace(prompt_observe_seconds=0.16, trace_prepare_seconds=0.04)

    def generate(self, prompt, **kwargs):
        self.calls.append((list(prompt), dict(kwargs)))
        tokens = FakeTokens(self.target, self.yields_per_request)
        self.iterators.append(tokens)
        return tokens


class CloseableChunks:
    def __init__(self):
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def close(self):
        self.closed = True


class PreparedTargetOptionsTests(unittest.TestCase):
    def test_off_returns_independent_unchanged_copy(self):
        base = {"recurrent_scan": True, "pack_weights": True, "custom_option": "kept"}
        result = prepared_target_options(base, "off")
        self.assertEqual(result, base)
        self.assertIsNot(result, base)
        result["extra"] = True
        self.assertNotIn("extra", base)
        self.assertEqual(base, {"recurrent_scan": True, "pack_weights": True, "custom_option": "kept"})

    def test_optimized_mode_adds_packing_and_consistent_decode_before_capture(self):
        base = {"recurrent_scan": True, "pack_weights": True}
        result = prepared_target_options(base, MODE)
        self.assertEqual(result, {
            "recurrent_scan": True,
            "pack_weights": True,
            "pack_output": False,
            "pack_gdn_gate_up": True,
            "consistent_decode": True,
            "native_state_snapshots": True,
            "gated_attention": True,
            "bounded_half_prefill": True,
        })
        self.assertIsNot(result, base)
        self.assertEqual(base, {"recurrent_scan": True, "pack_weights": True})

    def test_unknown_mode_and_incompatible_base_are_rejected(self):
        compatible = {"recurrent_scan": True, "pack_weights": True}
        with self.assertRaises(ValueError):
            prepared_target_options(compatible, "unknown-mode")
        incompatible = (
            {},
            {"recurrent_scan": False, "pack_weights": True},
            {"recurrent_scan": True, "pack_weights": False},
            {"recurrent_scan": True, "pack_weights": True, "unmeasured": True},
        )
        for base in incompatible:
            original = base.copy()
            with self.subTest(base=base), self.assertRaises(ValueError):
                prepared_target_options(base, MODE)
            self.assertEqual(base, original)

    def test_decoder_metrics_keeps_proposal_and_observation_metadata_separate(self):
        stats = StatsFixture()
        draft = DraftMetricsFixture()
        result = decoder_metrics(SimpleNamespace(stats=stats, draft=draft))
        self.assertEqual(result["mode"], MODE)
        self.assertEqual(result["stats"], {"proposed_tokens": 7, "accepted_tokens": 4, "verify_calls": 2})
        self.assertEqual(result["prepared_draft"], {
            "enabled": True, "graphs": 2, "captured_graphs": 2, "calls": 7, "weight_copies": 0,
        })
        self.assertEqual(result["prepared_observation"], {
            "enabled": True, "graphs": 3, "max_graphs": 5, "native_cache_copy": True,
            "native_copy_appends": 4, "native_copy_commands": 6, "native_copy_bytes": 768,
            "native_copy_fallbacks": 2,
        })
        self.assertNotEqual(result["prepared_draft"]["graphs"], result["prepared_observation"]["graphs"])
        self.assertEqual(result["prepared_observation"]["native_copy_fallbacks"], 2)
        json.dumps(result)

    def test_ordinary_decode_calibration_excludes_prompt_prefill_and_uses_minimum(self):
        model = FakeOrdinaryModel(max_context=512)
        times = iter((10.0, 10.4, 20.0, 20.2, 30.0, 30.3))

        def clock():
            model.events.append(("clock", None))
            return next(times)

        with patch.object(speculation.time, "perf_counter", side_effect=clock):
            result = calibrate_ordinary_decode(model)
        self.assertAlmostEqual(result["seconds_per_token"], min(result["samples"]))
        self.assertEqual(len(result["samples"]), 3)
        for actual, expected in zip(result["samples"], (0.4, 0.2, 0.3)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["prompt_tokens"], 128)
        self.assertEqual(model.cache_at_generate, [[], [], []])
        self.assertEqual(model._cached_tokens, [])
        self.assertEqual([len(prompt) for prompt, _ in model.prompts], [128, 128, 128])
        self.assertEqual([temperature for _, temperature in model.prompts], [0.0, 0.0, 0.0])
        for tokens in model.iterators:
            self.assertTrue(tokens.closed)
            self.assertEqual(tokens.yielded, 2)
        first_yield = model.events.index(("yield", 1))
        first_clock = model.events.index(("clock", None))
        self.assertLess(first_yield, first_clock)

    def test_ordinary_decode_calibration_closes_and_resets_on_short_or_invalid_timing(self):
        short = FakeOrdinaryModel(yields_per_request=1)
        with patch.object(speculation.time, "perf_counter", side_effect=(1.0, 1.1)), self.assertRaises(StopIteration):
            calibrate_ordinary_decode(short)
        self.assertEqual(short._cached_tokens, [])
        self.assertTrue(short.iterators[0].closed)

        invalid = FakeOrdinaryModel()
        with patch.object(speculation.time, "perf_counter", side_effect=(1.0, 1.0)), self.assertRaises(ValueError):
            calibrate_ordinary_decode(invalid)
        self.assertEqual(invalid._cached_tokens, [])
        self.assertTrue(invalid.iterators[0].closed)

    def test_prepared_decoder_warmup_forces_24_tokens_three_times_and_cleans_cache(self):
        decoder = FakePreparedDecoder(max_context=160)
        result = warm_prepared_decoder(decoder)
        self.assertEqual(result, {"requests": 3, "output_tokens_per_request": 24, "forced_fallback_for_setup": True,
                                  "prompt_observation_calibration": {"prompt_tokens": 128,
                                      "samples_seconds": [0.16] * 3, "seconds_per_token": 0.16 / 128,
                                      "trace_samples_seconds": [0.04] * 3, "trace_seconds": 0.04}})
        self.assertEqual(len(decoder.calls), 3)
        self.assertTrue(all(len(prompt) == 128 for prompt, _ in decoder.calls))
        self.assertTrue(all(kwargs == {"max_new_tokens": 24, "ordinary_decode_seconds": 1e-12, "prefill_chunk_size": 128}
                            for _, kwargs in decoder.calls))
        self.assertTrue(all(tokens.closed and tokens.yielded == 24 for tokens in decoder.iterators))
        self.assertEqual(decoder.target._cached_tokens, [])

    def test_prepared_decoder_warmup_rejects_invalid_observation_calibration_and_cleans_cache(self):
        for field, value in (("prompt_observe_seconds", 0.0), ("prompt_observe_seconds", float("nan")),
                             ("trace_prepare_seconds", float("inf")), ("trace_prepare_seconds", -1.0)):
            with self.subTest(field=field, value=value):
                decoder = FakePreparedDecoder()
                decoder.stats = SimpleNamespace(prompt_observe_seconds=0.1, trace_prepare_seconds=0.02)
                setattr(decoder.stats, field, value)
                with self.assertRaises(ValueError):
                    warm_prepared_decoder(decoder)
                self.assertEqual(len(decoder.calls), 3)
                self.assertTrue(all(tokens.closed and tokens.yielded == 24 for tokens in decoder.iterators))
                self.assertEqual(decoder.target._cached_tokens, [])

    def test_prepared_decoder_warmup_closes_and_cleans_after_short_iterator(self):
        decoder = FakePreparedDecoder(max_context=64, yields_per_request=23)
        with self.assertRaises(ValueError):
            warm_prepared_decoder(decoder)
        self.assertEqual(len(decoder.calls), 1)
        self.assertEqual(len(decoder.calls[0][0]), 40)
        self.assertEqual(decoder.calls[0][1]["max_new_tokens"], 24)
        self.assertTrue(decoder.iterators[0].closed)
        self.assertEqual(decoder.target._cached_tokens, [])

    def test_speculative_stream_forwards_the_calibrated_ordinary_decode_cost(self):
        class Tokenizer:
            eos_id, eot_id, eom_id = 9, None, None

        class Decoder:
            ordinary_decode_seconds = 0.4

            def __init__(self):
                self.calls = []
                self.tokens = FakeTokens(SimpleNamespace(_cached_tokens=[]), 0)

            def generate(self, ids, **kwargs):
                self.calls.append((list(ids), dict(kwargs)))
                return self.tokens

        class Handler:
            def __init__(self):
                self.server = SimpleNamespace(tok=Tokenizer())
                self.produced = []

            def run_model(self, *args, **kwargs):
                chunks = CloseableChunks()
                self.produced.append(chunks)
                return chunks

        decoder, handler = Decoder(), Handler()
        chunks = list(speculative_chunks(handler, decoder, [1, 2], "target", max_tokens=5,
                                         reasoning=False, cancelled=lambda: False,
                                         cancel_error=RuntimeError))
        self.assertEqual(chunks, [])
        self.assertEqual(decoder.calls[0][1]["ordinary_decode_seconds"], 0.4)
        self.assertTrue(decoder.tokens.closed)
        self.assertTrue(handler.produced[0].closed)

    def test_per_request_auto_off_and_sampled_selection(self):
        decoder = object()
        self.assertTrue(use_speculation(decoder, 0.0, "auto"))
        self.assertFalse(use_speculation(decoder, 0.6, "auto"))
        self.assertFalse(use_speculation(decoder, 0.0, "off"))
        self.assertTrue(use_speculation(decoder, 0.0, "force"))
        with self.assertRaises(ValueError):
            use_speculation(None, 0.0, "force")
        self.assertFalse(use_speculation(None, 0.0, "auto"))
        with self.assertRaises(ValueError):
            use_speculation(decoder, 0.0, "dflash2-experimental")

    def test_chat_route_forwards_per_request_controls_to_worker(self):
        with tempfile.TemporaryDirectory(prefix="speculation-route-") as directory:
            root = Path(directory)
            app = create_app(root / "missing-backend", root / "data")
            pool = app.state.runtime
            pool.sessions["target"] = SimpleNamespace(model={"id": "target", "name": "Target"})
            forwarded = []

            async def generate(body):
                forwarded.append(dict(body))
                channel = asyncio.Queue()
                await channel.put({"event": "chunk", "data": {
                    "id": "completion", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                }})
                await channel.put({"event": "done"})
                return f"request-{len(forwarded)}", channel, {"event": "started"}

            async def release(_key):
                return None

            pool.generate = generate
            pool.release = release
            client = TestClient(app)
            try:
                base = {"model": "target", "messages": [{"role": "user", "content": "compare"}],
                        "max_tokens": 4, "temperature": 0.0}
                response = client.post("/v1/chat/completions", json={**base, "speculation": "off", "cache_prompt": False})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(forwarded[-1]["speculation"], "off")
                self.assertIs(forwarded[-1]["cache_prompt"], False)

                response = client.post("/v1/chat/completions", json=base)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(forwarded[-1]["speculation"], "auto")
                self.assertIs(forwarded[-1]["cache_prompt"], True)

                response = client.post("/v1/chat/completions", json={**base, "speculation": "force"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(forwarded[-1]["speculation"], "force")

                response = client.post("/v1/chat/completions", json={**base, "speculation": "dflash2-experimental"})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(len(forwarded), 3)
            finally:
                client.close()
                app.state.store.close()


if __name__ == "__main__":
    unittest.main()
