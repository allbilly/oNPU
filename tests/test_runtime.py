import asyncio
import time

import pytest

from onpu.runtime import EngineError, Runtime
from onpu.store import Store


def test_single_request_admission_and_cancellation(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        runtime.state = "ready"
        commands = []
        async def send(command):
            commands.append(command)
            if command["op"] == "generate":
                await runtime.requests[command["id"]].put({"event": "started", "prompt_tokens": 5})
        runtime.send = send
        try:
            key, channel, first = await runtime.generate({"model": "test"})
            assert first["prompt_tokens"] == 5
            with pytest.raises(EngineError, match="busy"):
                await runtime.generate({"model": "test"})
            await runtime.cancel(key)
            assert commands[-1] == {"op": "cancel", "id": key}
            runtime.active = None
            await runtime.release(key)
            assert key not in runtime.requests
        finally:
            store.close()
    asyncio.run(run())


def test_stalled_startup_becomes_recoverable_error(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        runtime.state = "preparing"
        runtime.load_started = time.monotonic() - 500
        try:
            await runtime.check_startup_timeout()
            assert runtime.state == "error"
            assert "timed out" in runtime.message
            await runtime.unload()
            assert runtime.state == "idle"
        finally:
            store.close()
    asyncio.run(run())


def test_context_error_is_reported_before_streaming(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        runtime.state = "ready"
        async def send(command):
            if command["op"] == "generate":
                await runtime.requests[command["id"]].put({"event": "error", "message": "Context exceeded", "status": 400})
                runtime.active = None
                runtime.state = "ready"
        runtime.send = send
        try:
            with pytest.raises(EngineError) as exc:
                await runtime.generate({"model": "test"})
            assert exc.value.status == 400
            assert runtime.requests == {}
        finally:
            store.close()
    asyncio.run(run())
