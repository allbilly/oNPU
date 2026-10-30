"""Deferred/shared CPU INT8 admission is pinned and rechecked in the worker."""
import asyncio
import hashlib

import pytest

from onpu import library, rockchip_options
from onpu.runtime import EngineError, Runtime
from onpu.store import Store


def candidate(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"CPU INT8 admission layout fixture")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(rockchip_options.TEXT_INT8_CONTEXTS, checksum, 512)
    model = {"id": "candidate", "name": "Candidate", "path": str(path), "architecture": "gemma2", "size": 2784495456,
             "parameters": 2614341888, "layers": 26, "kv_heads": 4, "head_dim": 256}
    return model, path, checksum


def test_memory_estimate_uses_verified_layout_and_rejects_stale_identity(tmp_path, monkeypatch):
    model, path, checksum = candidate(tmp_path, monkeypatch)
    options = {"profile": "ROCKCHIP", "npu_prefill": "auto"}
    conservative = library.estimated_memory(model, 512, True)
    reduced = library.estimated_memory(model, 512, True, options)
    assert reduced == conservative - 2 * model["parameters"]
    assert library.estimated_memory(model, 513, True, options) == library.estimated_memory(model, 513, True)
    assert library.estimated_memory(model, 512, True, {**options, "npu_prefill": "off"}) == conservative
    path.write_bytes(b"different checkpoint")
    assert library.estimated_memory({**model, "sha256": checksum}, 512, True, options) == conservative


@pytest.mark.parametrize("available,admitted", [(14 * 1024**3, True), (6 * 1024**3, False)])
def test_admission_binds_worker_identity_and_keeps_available_memory_guard(tmp_path, monkeypatch, available, admitted):
    from onpu import runtime as serving

    async def run():
        model, _, checksum = candidate(tmp_path, monkeypatch)
        store = Store(tmp_path / "data")
        try:
            runtime = Runtime(store, tmp_path)
            options = {**store.settings(), "profile": "ROCKCHIP", "context": 512, "realize": True,
                       "npu_prefill": "auto", "npu_decode": "off"}
            monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 16739618816, "available": available})
            monkeypatch.setattr(serving.hardware, "reclaimable_rss", lambda pid: 0)
            calls = []

            async def stop():
                calls.append("stop")

            async def launch(*args, **kwargs):
                calls.append(kwargs["env"])
                raise OSError("launch admitted")

            monkeypatch.setattr(runtime, "_stop", stop)
            monkeypatch.setattr(serving.asyncio, "create_subprocess_exec", launch)
            with pytest.raises(EngineError, match="launch admitted" if admitted else "available RAM"):
                await runtime.load(model, options)
            if admitted:
                assert calls[1]["ONPU_CPU_INT8_CHECKPOINT"] == checksum
                assert calls[1]["ONPU_NPU_PREFILL"] == "auto"
                assert calls[1]["ONPU_NPU_DECODE"] == "on"
            else:
                assert not calls
        finally:
            store.close()

    asyncio.run(run())


def test_inherited_worker_admission_identity_cannot_leak_to_another_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("ONPU_CPU_INT8_CHECKPOINT", "unrelated")
    store = Store(tmp_path / "data")
    try:
        assert "ONPU_CPU_INT8_CHECKPOINT" not in Runtime(store, tmp_path).environment("CL")
    finally:
        store.close()


def test_recurrent_admission_passes_checked_worker_execution_environment(tmp_path, monkeypatch):
    from onpu import runtime as serving

    async def run():
        model, _, checksum = candidate(tmp_path, monkeypatch)
        model["architecture"] = "rwkv7"
        monkeypatch.setattr(rockchip_options, "RWKV7_CHECKPOINT", checksum)
        store = Store(tmp_path / "data")
        try:
            runtime = Runtime(store, tmp_path)
            options = {**store.settings(), "profile": "ROCKCHIP", "context": 512, "realize": True,
                       "npu_prefill": "auto", "npu_decode": "off"}
            monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 16739618816, "available": 14 * 1024**3})
            monkeypatch.setattr(serving.hardware, "reclaimable_rss", lambda pid: 0)
            captured = []

            async def stop():
                return None

            async def launch(*args, **kwargs):
                captured.append(kwargs["env"])
                raise OSError("recurrent launch admitted")

            monkeypatch.setattr(runtime, "_stop", stop)
            monkeypatch.setattr(serving.asyncio, "create_subprocess_exec", launch)
            with pytest.raises(EngineError, match="recurrent launch admitted"):
                await runtime.load(model, options)
            layout = rockchip_options.text_execution_profile(checksum, 512, "ROCKCHIP", "auto")
            assert captured[0]["ONPU_CPU_INT8_CHECKPOINT"] == checksum
            assert all(captured[0][key] == value for key, value in layout["environment"].items())
        finally:
            store.close()

    asyncio.run(run())


def test_missing_files_keep_conservative_memory_estimate(tmp_path, monkeypatch):
    model, path, _ = candidate(tmp_path, monkeypatch)
    path.unlink()
    assert library.estimated_memory(model, 512, True, {"profile": "ROCKCHIP", "npu_prefill": "auto"}) == library.estimated_memory(model, 512, True)


@pytest.mark.parametrize("profile", ["ROCKCHIP", "CL", "CPU"])
@pytest.mark.parametrize("path", [42, [], {}, "", None])
def test_invalid_path_keeps_conservative_memory_estimate(profile, path):
    size, parameters, _ = next(iter(library._UNPACKED_MALI_MEMORY.values()))
    model = {"path": path, "architecture": "qwen35", "size": size, "parameters": parameters}
    options = {"profile": profile, "npu_prefill": "auto", "npu_decode": "off"}
    assert library.estimated_memory(model, 512, True, options) == library.estimated_memory(model, 512, True)
