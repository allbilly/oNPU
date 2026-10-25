"""CPU contracts for checkpoint-qualified experimental context selection."""
from __future__ import annotations

import unittest

from onpu.speculation import MODE, TARGET_SHA256, prepared_target_options


class PreparedTargetContextProfileTests(unittest.TestCase):
    def assert_profile(self, result):
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

    def test_known_layout_accepts_context_512(self):
        base = {"recurrent_scan": True, "pack_weights": True}
        result = prepared_target_options(
            base, MODE, checkpoint_sha256=TARGET_SHA256, context=512,
        )
        self.assert_profile(result)
        self.assertEqual(result["recurrent_scan"], True)
        self.assertEqual(result["pack_weights"], True)
        self.assertEqual(base, {"recurrent_scan": True, "pack_weights": True})
        self.assertIsNot(result, base)

    def test_pinned_identity_allows_empty_layout_through_context_8192(self):
        base = {}
        result = prepared_target_options(
            base, MODE, checkpoint_sha256=TARGET_SHA256, context=8192,
        )
        self.assert_profile(result)
        self.assertEqual(base, {})
        self.assertIsNot(result, base)

    def test_context_513_is_accepted_for_known_layout(self):
        result = prepared_target_options(
            {"recurrent_scan": True, "pack_weights": True}, MODE,
            checkpoint_sha256=TARGET_SHA256, context=513,
        )
        self.assert_profile(result)

    def test_invalid_contexts_are_rejected(self):
        for context in (63, 8193, 512.0, True, False, "512"):
            with self.subTest(context=context), self.assertRaises(ValueError):
                prepared_target_options(
                    {"recurrent_scan": True, "pack_weights": True}, MODE,
                    checkpoint_sha256=TARGET_SHA256, context=context,
                )

    def test_wrong_checksum_is_rejected_for_known_layout(self):
        with self.assertRaises(ValueError):
            prepared_target_options(
                {"recurrent_scan": True, "pack_weights": True}, MODE,
                checkpoint_sha256="0" * 64, context=512,
            )

    def test_empty_layout_requires_exact_pinned_identity(self):
        for checksum in (None, "unknown-checkpoint", "0" * 64):
            with self.subTest(checksum=checksum), self.assertRaises(ValueError):
                prepared_target_options(
                    {}, MODE, checkpoint_sha256=checksum, context=8192,
                )

    def test_unknown_layout_and_extra_keys_are_rejected(self):
        for base in (
            {"recurrent_scan": False, "pack_weights": True},
            {"recurrent_scan": True, "pack_weights": False},
            {"recurrent_scan": True, "pack_weights": True, "extra": True},
            {"unexpected": True},
        ):
            with self.subTest(base=base), self.assertRaises(ValueError):
                prepared_target_options(
                    base, MODE, checkpoint_sha256=TARGET_SHA256, context=512,
                )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            prepared_target_options(
                {"recurrent_scan": True, "pack_weights": True}, "unknown-mode",
                checkpoint_sha256=TARGET_SHA256, context=512,
            )

    def test_off_returns_unchanged_independent_copy_without_profile_override(self):
        base = {"custom_option": "kept", "context": 8192}
        result = prepared_target_options(
            base, "off", checkpoint_sha256="unknown-checkpoint", context=8192,
        )
        self.assertEqual(result, base)
        self.assertIsNot(result, base)
        result["added"] = True
        self.assertNotIn("added", base)
        self.assertEqual(base, {"custom_option": "kept", "context": 8192})


if __name__ == "__main__":
    unittest.main()
