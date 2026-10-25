import asyncio
import hashlib
import json
import os
import struct

import httpx
import pytest

from onpu.library import (
    Downloads,
    estimated_memory,
    inspect_gguf,
    inspect_hf,
    model_present,
    register,
)
from onpu.store import Store


@pytest.fixture
def unpacked_checkpoint(tmp_path, monkeypatch):
    from onpu import execution, library

    path = tmp_path / "qualified.gguf"
    path.write_bytes(b"qualified unpacked checkpoint")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    measured = next(iter(library._UNPACKED_MALI_MEMORY.values()))
    monkeypatch.setitem(execution.QUALIFIED_MALI_RECURRENT_CHECKPOINTS, checksum, (512, False))
    monkeypatch.setitem(library._UNPACKED_MALI_MEMORY, checksum, measured)
    return {"path": str(path), "architecture": "qwen35", "quantization": "Q8_0", "size": measured[0], "parameters": measured[1],
            "layers": 32, "kv_heads": 4, "head_dim": 256}, measured[2]


def test_metadata_and_reference_import(gguf, tmp_path):
    store = Store(tmp_path / "data")
    try:
        model = register(store, gguf)
        assert model["name"] == "Test model"
        assert model["parameters"] == 4
        assert model["quantization"] == "F16"
        assert model["context_limit"] == 1024
        assert register(store, gguf)["id"] == model["id"]
        assert len(store.list("models")) == 1
        store.delete("models", model["id"])
        assert gguf.exists()
    finally:
        store.close()


@pytest.mark.parametrize("data", [b"", b"invalid", b"GGUF" + bytes(24), b"GGUF\x03\x00\x00\x00" + b"\xff" * 16])
def test_rejects_malformed_files(data, tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        inspect_gguf(path)


def test_rejects_metadata_without_weights(gguf):
    gguf.write_bytes(gguf.read_bytes()[:-8])
    with pytest.raises(ValueError, match="weight data"):
        inspect_gguf(gguf)


def test_memory_includes_fp32_kv():
    model = {"size": 1000, "parameters": 1000, "layers": 28, "kv_heads": 8, "head_dim": 128}
    assert estimated_memory(model, 1024, True) - estimated_memory(model, 512, True) == 2 * 28 * 8 * 128 * 512 * 4


@pytest.mark.parametrize("architecture", ["qwen3", "llama", "idefics3"])
def test_standard_transformer_memory_uses_its_fp16_kv_storage(architecture):
    model = {"architecture": architecture, "size": 1000, "parameters": 1000, "layers": 28, "kv_heads": 8, "head_dim": 128}
    assert estimated_memory(model, 40960, True) - estimated_memory(model, 8192, True) == 2 * 28 * 8 * 128 * 32768 * 2


def test_measured_unpacked_memory_requires_matching_worker_layout(unpacked_checkpoint):
    from pathlib import Path

    from onpu.worker import mali_options

    model, measured = unpacked_checkpoint
    options = {"profile": "CL", "npu_prefill": "off", "npu_decode": "off"}
    assert mali_options(Path(model["path"]), 512, "CL", "off", "off") == {"recurrent_scan": True, "pack_weights": False}
    assert estimated_memory(model, 512, True, options) == measured
    assert measured > (9830.328125 + 1536) * 1024**2
    assert measured < 16739618816 * 0.8 < estimated_memory(model, 512, True)
    # The measured peak already includes the complete 512-position state. A
    # smaller context keeps that conservative budget rather than extrapolating.
    assert estimated_memory(model, 128, True, options) == measured


@pytest.mark.parametrize("context,realize,changes", [
    (513, True, {}), (0, True, {}), (512, False, {}),
    (512, True, {"profile": "CPU"}), (512, True, {"profile": "ROCKCHIP"}),
    (512, True, {"npu_prefill": "auto"}), (512, True, {"npu_decode": "on"}),
])
def test_unqualified_execution_retains_conservative_memory(unpacked_checkpoint, context, realize, changes):
    model, _ = unpacked_checkpoint
    options = {"profile": "CL", "npu_prefill": "off", "npu_decode": "off", **changes}
    assert estimated_memory(model, context, realize, options) == estimated_memory(model, context, realize)


def test_changed_checkpoint_or_metadata_cannot_use_measured_memory(unpacked_checkpoint, monkeypatch):
    from pathlib import Path

    from onpu import execution

    model, _ = unpacked_checkpoint
    options = {"profile": "CL"}
    conservative = estimated_memory(model, 512, True)
    assert estimated_memory({**model, "parameters": model["parameters"] - 1}, 512, True, options) == conservative - 4
    checksum = execution.checkpoint_checksum(Path(model["path"]))
    # Even qualified smaller profiles cannot borrow the fully unpacked budget.
    monkeypatch.setitem(execution.QUALIFIED_MALI_RECURRENT_CHECKPOINTS, checksum, (512, True))
    assert estimated_memory(model, 512, True, options) == conservative
    monkeypatch.setitem(execution.QUALIFIED_MALI_RECURRENT_CHECKPOINTS, checksum, (512, False))
    Path(model["path"]).write_bytes(b"unqualified different weights")
    assert estimated_memory({**model, "sha256": checksum}, 512, True, options) == conservative
    Path(model["path"]).unlink()
    assert estimated_memory(model, 512, True, options) == conservative


def test_checksum_cache_rechecks_same_size_file_with_restored_mtime(tmp_path, monkeypatch):
    from onpu import execution

    path = tmp_path / "weights.gguf"
    path.write_bytes(b"first weights")
    before = path.stat()
    calls, file_digest = [], hashlib.file_digest

    def count_reads(*args):
        calls.append(1)
        return file_digest(*args)

    monkeypatch.setattr(execution.hashlib, "file_digest", count_reads)
    # Display estimates may reuse an advisory cache. Actual trust decisions use
    # the default fresh read, including when stat timestamps happen to collide.
    first = execution.checkpoint_checksum(path, refresh=False)
    assert execution.checkpoint_checksum(path, refresh=False) == first
    assert len(calls) == 1
    path.write_bytes(b"other weights")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert execution.checkpoint_checksum(path) != first
    assert len(calls) == 2


def test_worker_and_admission_reverify_content_even_when_stat_signatures_collide(unpacked_checkpoint, monkeypatch):
    from pathlib import Path

    from onpu import execution, worker

    model, measured = unpacked_checkpoint
    path = Path(model["path"])
    options = {"profile": "CL"}
    # Force the collision observed on this filesystem without any timing
    # assumptions. The old signature-keyed trust decision fails this case.
    signature = execution._signature(path.stat())
    monkeypatch.setattr(execution, "_signature", lambda stat: signature)
    assert estimated_memory(model, 512, True, options, refresh=False) == measured
    assert worker.mali_options(path, 512, "CL", "off", "off") == {"recurrent_scan": True, "pack_weights": False}
    path.write_bytes(b"X" * path.stat().st_size)
    assert estimated_memory(model, 512, True, options) == estimated_memory(model, 512, True)
    # Fresh verification also replaces any advisory entry for the same key.
    assert estimated_memory(model, 512, True, options, refresh=False) == estimated_memory(model, 512, True)
    assert worker.mali_options(path, 512, "CL", "off", "off") == {}


def test_admission_uses_one_fresh_read_for_policy_and_memory_budget(unpacked_checkpoint, monkeypatch):
    model, measured = unpacked_checkpoint
    calls, file_digest = [], hashlib.file_digest

    def count_reads(*args):
        calls.append(1)
        return file_digest(*args)

    monkeypatch.setattr(hashlib, "file_digest", count_reads)
    assert estimated_memory(model, 512, True, {"profile": "CL"}) == measured
    assert len(calls) == 1
    assert estimated_memory(model, 512, True, {"profile": "CL"}) == measured
    assert len(calls) == 2


@pytest.fixture
def vision_directory(tmp_path):
    path = tmp_path / "vision"
    path.mkdir()
    config = {"model_type": "idefics3", "scale_factor": 4,
              "vision_config": {"image_size": 512, "patch_size": 16},
              "text_config": {"max_position_embeddings": 8192, "num_hidden_layers": 30, "num_key_value_heads": 3, "head_dim": 64}}
    (path / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "generation_config.json"):
        (path / name).write_text("{}")
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    weights = path / "model.safetensors"
    weights.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    return path


def test_vision_directory_import_and_missing_weights(tmp_path, vision_directory):
    path = vision_directory
    weights = path / "model.safetensors"
    store = Store(tmp_path / "data")
    try:
        model = register(store, path)
        assert model["format"] == "HF" and model["supports_images"]
        assert model["parameters"] == 4 and model["image_tokens"] == 64
        assert model_present(model)
        weights.write_bytes(weights.read_bytes()[:-1])
        with pytest.raises(ValueError, match="weight data"):
            inspect_hf(path)
        weights.unlink()
        assert not model_present(model)
    finally:
        store.close()


@pytest.mark.parametrize("field,value", [
    ((), []), (("text_config",), None), (("vision_config",), []),
    (("text_config", "head_dim"), None), (("text_config", "num_hidden_layers"), -1),
    (("text_config", "num_key_value_heads"), True), (("text_config", "max_position_embeddings"), "8192"),
    (("vision_config", "patch_size"), 0), (("scale_factor",), 1.5), (("scale_factor",), 3),
])
def test_rejects_malformed_vision_configuration(vision_directory, field, value):
    config_path = vision_directory / "config.json"
    config = json.loads(config_path.read_text())
    if not field:
        config = value
    elif len(field) == 1:
        config[field[0]] = value
    else:
        config[field[0]][field[1]] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration"):
        inspect_hf(vision_directory)


@pytest.mark.parametrize("header", [
    [], {"weight": None}, {"weight": {"dtype": "BF16"}},
    {"weight": {"dtype": "BF16", "shape": [True, 4], "data_offsets": [0, 8]}},
    {"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0]}},
    {"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0.0, 8.0]}},
])
def test_rejects_malformed_safetensors_metadata(vision_directory, header):
    encoded = json.dumps(header).encode()
    (vision_directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(8))
    with pytest.raises(ValueError):
        inspect_hf(vision_directory)


def test_partial_download_is_resumed_from_same_file(gguf, tmp_path, monkeypatch):
    """Resume survives resolving a branch to an immutable revision."""
    async def run():
        store = Store(tmp_path / "data")
        payload = gguf.read_bytes()
        partial = store.root / "models/job-test.gguf.part"
        partial.write_bytes(payload[:64])
        calls = []
        def handler(request):
            calls.append(request)
            if "/api/models/" in str(request.url):
                return httpx.Response(200, json={"sha": "immutable"})
            assert request.headers["range"] == "bytes=64-"
            return httpx.Response(206, content=payload[64:], headers={"content-range": f"bytes 64-{len(payload)-1}/{len(payload)}"})
        real_client = httpx.AsyncClient
        monkeypatch.setattr("onpu.library.httpx.AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
        downloads = Downloads(store)
        try:
            job = downloads.start({"repo": "test/model", "file": "test.gguf", "revision": "immutable"}, "job")
            await downloads.tasks[job["id"]]
            result = store.get("downloads", "job")
            assert result["state"] == "complete", result
            assert result["received"] == len(payload)
            assert (store.root / "models/job-test.gguf").read_bytes() == payload
            assert len(result["sha256"]) == 64
            assert len(store.list("models")) == 1
        finally:
            await downloads.close()
            store.close()
    asyncio.run(run())


def test_rejects_path_traversal_download(tmp_path):
    store = Store(tmp_path / "data")
    try:
        downloads = Downloads(store)
        with pytest.raises(ValueError):
            downloads.start({"repo": "test/model", "file": "../escape.gguf"})
    finally:
        store.close()
