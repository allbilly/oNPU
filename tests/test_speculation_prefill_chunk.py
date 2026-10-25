"""CPU-only contracts for the prepared decoder's prefill chunk setting."""
from __future__ import annotations

import builtins
import unittest
from pathlib import Path
from unittest.mock import patch

from test_speculation_prepared import FakePreparedDecoder
from test_speculation_sampling import FakeFormatHandler, FakeSampledDecoder

from onpu import speculation


class PrefillChunkTests(unittest.TestCase):
    def test_64_chunk_warmup_calibrates_prompt_and_forwards_chunk_each_run(self):
        decoder = FakePreparedDecoder(max_context=160)
        decoder.prefill_chunk_size = 64

        result = speculation.warm_prepared_decoder(decoder)

        self.assertEqual(result["prompt_observation_calibration"], {
            "prompt_tokens": 64,
            "samples_seconds": [0.16] * 3,
            "seconds_per_token": 0.16 / 64,
            "trace_samples_seconds": [0.04] * 3,
            "trace_seconds": 0.04,
        })
        self.assertEqual(len(decoder.calls), 3)
        self.assertTrue(all(len(prompt) == 64 for prompt, _ in decoder.calls))
        self.assertTrue(all(kwargs == {"max_new_tokens": 24, "ordinary_decode_seconds": 1e-12,
                                       "prefill_chunk_size": 64} for _, kwargs in decoder.calls))
        self.assertTrue(all(tokens.closed and tokens.yielded == 24 for tokens in decoder.iterators))
        self.assertEqual(decoder.target._cached_tokens, [])

    def test_greedy_and_sampled_runtime_receive_selected_chunk(self):
        for temperature in (0.0, 0.6):
            with self.subTest(temperature=temperature):
                decoder = FakeSampledDecoder()
                decoder.prefill_chunk_size = 64
                handler = FakeFormatHandler(rows=({"choices": [{"delta": {"content": "hello"}}]},))
                list(speculation.speculative_chunks(handler, decoder, [7, 8], "target", max_tokens=4,
                                                    reasoning=False, cancelled=lambda: False,
                                                    cancel_error=RuntimeError, temperature=temperature))
                call = decoder.generations[0] if temperature == 0.0 else decoder.samples[0]
                self.assertEqual(call[-1]["prefill_chunk_size"], 64)
                self.assertEqual(call[-1]["max_new_tokens"], 4)

    def test_invalid_load_chunk_is_rejected_before_tinygrad_import(self):
        real_import = builtins.__import__
        attempted = []

        def guarded_import(name, *args, **kwargs):
            if name == "tinygrad" or name.startswith("tinygrad."):
                attempted.append(name)
                raise AssertionError("tinygrad import should not be reached for invalid chunk")
            return real_import(name, *args, **kwargs)

        for value in (0, 32, 96, 64.0, True):
            with (
                self.subTest(value=value),
                patch("builtins.__import__", side_effect=guarded_import),
                self.assertRaises(ValueError),
            ):
                speculation.load_decoder(None, Path("unused"), "unused", value)
        self.assertEqual(attempted, [])


if __name__ == "__main__":
    unittest.main()
