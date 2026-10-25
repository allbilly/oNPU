"""CPU-only contracts for the pinned tied Q8 speculative target helper."""
from __future__ import annotations

import gc
import json
import sys
import unittest
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from onpu.speculation import (
    MODE,
    TARGET_SHA256,
    decoder_metrics,
    prepare_target_table,
    target_table_metadata,
)


class SharedQ8TargetTests(unittest.TestCase):
    def test_off_mode_returns_before_device_or_checkpoint_import(self):
        with patch("tinygrad.Device", side_effect=AssertionError("Device accessed")), \
             patch.dict(sys.modules, {"tinygrad.llm.kernels.mali_q8_checkpoint": None}):
            self.assertIsNone(prepare_target_table(object(), Path("missing"), "off", checkpoint_sha256=None))

    def test_wrong_pinned_hash_is_rejected_before_device_access(self):
        with patch("tinygrad.Device", side_effect=AssertionError("Device accessed")), \
             self.assertRaisesRegex(ValueError, "pinned target checkpoint"):
            prepare_target_table(object(), Path("missing"), MODE, checkpoint_sha256="bad")

    def test_unqualified_device_and_driver_skip_table_loading(self):
        for device_info in ({"device_name": "Other GPU", "driver_version": "2.1"},
                            {"device_name": "Mali-LODX r0p0", "driver_version": "other"}):
            model = SimpleNamespace(token_embd=SimpleNamespace(weight=SimpleNamespace(device="CL:0", shape=(248320, 2048))))
            with self.subTest(device_info=device_info), patch("tinygrad.Device", {"CL:0": SimpleNamespace(**device_info)}):
                self.assertIsNone(prepare_target_table(model, Path("checkpoint"), MODE, checkpoint_sha256=TARGET_SHA256))

    def test_qualified_device_loads_raw_then_replaces_and_does_not_keep_dense_weight(self):
        events = []
        class Weight:
            device, shape = "CL:0", (248320, 2048)
        old_weight = Weight()
        old_ref = weakref.ref(old_weight)
        model = SimpleNamespace(token_embd=SimpleNamespace(weight=old_weight))
        del old_weight
        metadata = {"enabled": True, "shared_table": True, "raw_realized": False}

        class Checkpoint:
            def __init__(self, path):
                events.append(("checkpoint", path))
                self.entries = {"token_embd.weight": object()}
            def weight(self, key, shape, device):
                events.append(("weight", key, shape, device))
                return "raw-table"

        def replace(actual_model, raw):
            events.append(("replace", raw))
            actual_model.token_embd = SimpleNamespace(table="replacement")
            gc.collect()
            self.assertIsNone(old_ref(), "helper retained the dense matrix across replacement")
            return metadata

        checkpoint_module = ModuleType("tinygrad.llm.kernels.mali_q8_checkpoint")
        checkpoint_module.Q8Checkpoint = Checkpoint
        shared_module = ModuleType("tinygrad.llm.kernels.mali_q8_shared")
        shared_module.replace_tied_q8_table = replace
        device_lookup = {"CL:0": SimpleNamespace(device_name="Mali-LODX r0p0", driver_version="2.1")}
        with patch("tinygrad.Device", device_lookup), patch.dict(sys.modules, {
            checkpoint_module.__name__: checkpoint_module, shared_module.__name__: shared_module,
        }):
            result = prepare_target_table(model, Path("checkpoint"), MODE, checkpoint_sha256=TARGET_SHA256)
        self.assertEqual(events, [("checkpoint", Path("checkpoint")),
                                  ("weight", "token_embd.weight", (248320, 2048), "CL:0"),
                                  ("replace", "raw-table")])
        self.assertEqual(result, {"enabled": True, "helper_metadata": metadata})

    def test_output_checkpoint_entry_or_wrong_shape_is_rejected(self):
        class Checkpoint:
            entries: ClassVar[dict[str, object]] = {"output.weight": object()}
            def __init__(self, path): pass
            def weight(self, *args): raise AssertionError("weight should not load")
        checkpoint_module = ModuleType("tinygrad.llm.kernels.mali_q8_checkpoint")
        checkpoint_module.Q8Checkpoint = Checkpoint
        shared_module = ModuleType("tinygrad.llm.kernels.mali_q8_shared")
        shared_module.replace_tied_q8_table = lambda *args: self.fail("replacement should not run")
        device_lookup = {"CL:0": SimpleNamespace(device_name="Mali-LODX r0p0", driver_version="2.1")}
        with patch("tinygrad.Device", device_lookup), patch.dict(sys.modules, {
            checkpoint_module.__name__: checkpoint_module, shared_module.__name__: shared_module,
        }), self.assertRaisesRegex(ValueError, "tied vocabulary"):
            prepare_target_table(SimpleNamespace(token_embd=SimpleNamespace(weight=SimpleNamespace(
                    device="CL:0", shape=(248320, 2048)))), Path("checkpoint"), MODE,
                    checkpoint_sha256=TARGET_SHA256)
        # The target matrix shape is independently checked when there is no separate output tensor.
        Checkpoint.entries = {}
        with patch("tinygrad.Device", device_lookup), patch.dict(sys.modules, {
            checkpoint_module.__name__: checkpoint_module, shared_module.__name__: shared_module,
        }), self.assertRaisesRegex(ValueError, "tied vocabulary"):
            prepare_target_table(SimpleNamespace(token_embd=SimpleNamespace(weight=SimpleNamespace(
                    device="CL:0", shape=(7, 32)))), Path("checkpoint"), MODE, checkpoint_sha256=TARGET_SHA256)

    def test_table_metadata_requires_shared_backend_types_and_metrics_include_it_conditionally(self):
        class Table:
            def metadata(self): return {"retained_raw_bytes": 952, "raw_realized": False}
        class Embedding:
            def __init__(self, table): self.table = table
        class Head:
            def __init__(self, table): self.table = table
        shared_module = ModuleType("tinygrad.llm.kernels.mali_q8_shared")
        shared_module.MaliQ8Embedding, shared_module.MaliQ8Head = Embedding, Head
        table = Table()
        model = SimpleNamespace(token_embd=Embedding(table), output=Head(table))
        @dataclass
        class Stats:
            proposed_tokens: int = 3
        stats = Stats()
        draft = SimpleNamespace(proposal_preparation_metadata=dict, observation_preparation_metadata=dict)
        with patch.dict(sys.modules, {shared_module.__name__: shared_module}):
            self.assertIsNone(target_table_metadata(SimpleNamespace(token_embd=SimpleNamespace(weight=object()))))
            with self.assertRaisesRegex(RuntimeError, "must share"):
                target_table_metadata(SimpleNamespace(token_embd=Embedding(table), output=Head(Table())))
            metadata = target_table_metadata(model)
            self.assertEqual(metadata, {"enabled": True, "shared_table": True, "retained_raw_bytes": 952,
                                        "raw_realized": False})
            result = decoder_metrics(SimpleNamespace(stats=stats, draft=draft, target=model))
            self.assertEqual(result["target_table"], metadata)
            json.dumps(result)
        dense_metrics = decoder_metrics(SimpleNamespace(stats=stats, draft=draft))
        self.assertNotIn("target_table", dense_metrics)


if __name__ == "__main__":
    unittest.main()
