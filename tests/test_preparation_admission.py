import asyncio
import hashlib

import pytest

from onpu.runtime import EngineError, Runtime, RuntimePool
from onpu.store import Store


@pytest.mark.parametrize("available,expected", [(14 * 1024**3, "launch"), (10 * 1024**3, "available RAM")])
def test_qualified_unpacked_model_uses_layout_budget_and_keeps_available_ram_guard(tmp_path, monkeypatch, available, expected):
    from onpu import execution, library
    from onpu import runtime as serving

    async def run():
        path = tmp_path / "qualified.gguf"
        path.write_bytes(b"qualified unpacked admission weights")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        measured = next(iter(library._UNPACKED_MALI_MEMORY.values()))
        monkeypatch.setitem(execution.QUALIFIED_MALI_RECURRENT_CHECKPOINTS, checksum, (512, False))
        monkeypatch.setitem(library._UNPACKED_MALI_MEMORY, checksum, measured)
        model = {"id": "qualified", "path": str(path), "architecture": "qwen35", "size": measured[0], "parameters": measured[1],
                 "layers": 32, "kv_heads": 4, "head_dim": 256}
        store = Store(tmp_path / "data")
        try:
            instance = Runtime(store, tmp_path)
            options = {**store.settings(), "profile": "CL", "context": 512, "realize": True, "npu_prefill": "off", "npu_decode": "off"}
            monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 16739618816, "available": available})
            monkeypatch.setattr(serving.hardware, "rss", lambda pid: 0)
            stopped = []

            async def stop():
                stopped.append(True)

            async def launch(*args, **kwargs):
                # Reaching launch proves admission without creating a worker or
                # allocating any checkpoint/device storage during this test.
                raise OSError("launch admitted")

            monkeypatch.setattr(instance, "_stop", stop)
            monkeypatch.setattr(serving.asyncio, "create_subprocess_exec", launch)
            with pytest.raises(EngineError, match=expected) as error:
                await instance.load(model, options)
            assert error.value.status == (500 if expected == "launch" else 400)
            assert bool(stopped) == (expected == "launch")
        finally:
            store.close()

    asyncio.run(run())


@pytest.mark.parametrize("state", ["loading", "preparing"])
def test_another_preparing_model_rejects_before_allocating_or_stopping_workers(tmp_path, monkeypatch, state):
    async def run():
        store = Store(tmp_path / "data")
        try:
            pool = RuntimePool(store, tmp_path)
            existing = Runtime(store, tmp_path)
            existing.model = {"id": "gpu", "name": "GPU model"}
            existing.options = {"profile": "CL", "npu_prefill": "off", "npu_decode": "off"}
            existing.profile, existing.state = "CL", state
            pool.sessions["gpu"], pool.selected = existing, "gpu"

            async def must_not_load(*args, **kwargs):
                raise AssertionError("Do not start or stop any worker before preparation admission")

            monkeypatch.setattr(Runtime, "load", must_not_load)
            with pytest.raises(EngineError, match="still preparing") as error:
                await pool.load({"id": "npu", "name": "NPU model"}, {"profile": "ROCKCHIP"})
            assert error.value.status == 409
            assert pool.sessions == {"gpu": existing} and pool.selected == "gpu" and existing.state == state
        finally:
            store.close()
    asyncio.run(run())


def test_ready_models_allow_another_profile_and_same_session_can_restart(tmp_path, monkeypatch):
    async def run():
        store = Store(tmp_path / "data")
        try:
            pool = RuntimePool(store, tmp_path)
            existing = Runtime(store, tmp_path)
            existing.model = {"id": "gpu", "name": "GPU model"}
            existing.options = {"profile": "CL", "npu_prefill": "off", "npu_decode": "off"}
            existing.profile, existing.state = "CL", "ready"
            pool.sessions["gpu"] = existing
            loaded = []

            async def load(self, model, options):
                self.model, self.options, self.profile, self.state = model, options, options["profile"], "loading"
                loaded.append(model["id"])

            monkeypatch.setattr(Runtime, "load", load)
            model, options = {"id": "npu", "name": "NPU model"}, {"profile": "ROCKCHIP"}
            await pool.load(model, options)
            await pool.load(model, options)
            assert loaded == ["npu", "npu"] and pool.sessions["gpu"] is existing and existing.state == "ready"
        finally:
            store.close()
    asyncio.run(run())
