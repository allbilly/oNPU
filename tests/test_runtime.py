import asyncio
import contextlib
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from onpu.runtime import EngineError, Runtime, RuntimePool
from onpu.store import Store


def test_vision_gpu_option_reaches_only_worker_configuration(tmp_path):
    store = Store(tmp_path / "data")
    try:
        runtime = Runtime(store, tmp_path)
        env = runtime.environment("ROCKCHIP", vision_gpu=False)
        assert env["DEV"] == "CPU"
        assert env["ONPU_VISION_GPU"] == "0"
        env = runtime.environment("CL", vision_gpu=True)
        assert env["DEV"] == "CL"
        assert env["ONPU_VISION_GPU"] == "1"
    finally:
        store.close()


def test_gpu_npu_weight_reservation_rejects_before_stopping_existing_model(tmp_path, monkeypatch):
    import onpu.runtime as serving

    store = Store(tmp_path / "data")
    try:
        runtime = Runtime(store, tmp_path)
        previous = {"id": "previous"}
        runtime.model, runtime.state = previous, "ready"
        options = {**store.settings(), "profile": "CL", "context": 8192, "realize": True, "npu_prefill": "auto", "npu_decode": "off"}
        monkeypatch.setattr(serving, "estimated_memory", lambda *args: 1024**3)
        monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 8 * 1024**3, "available": 2 * 1024**3})
        monkeypatch.setattr(serving.hardware, "reclaimable_rss", lambda pid: 0)

        async def must_not_stop(*args, **kwargs):
            raise AssertionError("Admission must reserve NPU weights before stopping the current worker")

        monkeypatch.setattr(runtime, "_stop", must_not_stop)
        with pytest.raises(EngineError, match="available RAM") as error:
            asyncio.run(runtime.load({"id": "candidate", "parameters": 350_000_000}, options))
        assert error.value.status == 400 and runtime.model is previous
    finally:
        store.close()


def test_packed_vlm_cache_reservation_rejects_before_stopping_existing_model(tmp_path, monkeypatch):
    import onpu.runtime as serving

    store = Store(tmp_path / "data")
    try:
        runtime = Runtime(store, tmp_path)
        previous = {"id": "previous"}
        runtime.model, runtime.state = previous, "ready"
        options = {**store.settings(), "profile": "ROCKCHIP", "context": 8192, "realize": True}
        monkeypatch.setattr(serving, "estimated_memory", lambda *args: 1024**3)
        monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 8 * 1024**3, "available": 1600 * 1024**2})
        monkeypatch.setattr(serving.hardware, "reclaimable_rss", lambda pid: 0)

        async def must_not_stop(*args, **kwargs):
            raise AssertionError("Admission must reserve packed KV before stopping the current worker")

        monkeypatch.setattr(runtime, "_stop", must_not_stop)
        candidate = {"id": "candidate", "architecture": "idefics3", "layers": 30, "kv_heads": 3, "head_dim": 64}
        with pytest.raises(EngineError, match="available RAM"):
            asyncio.run(runtime.load(candidate, options))
        assert runtime.model is previous
    finally:
        store.close()


@pytest.mark.parametrize("file_gib,private_gib,available_after_gib,stopped,loaded", [
    pytest.param(2, 3, 5, False, False, id="file-cache-keeps-current-worker"),
    pytest.param(0, 1, 5, False, False, id="shared-pages-keep-current-worker"),
    pytest.param(0, 3, 7, True, True, id="private-anonymous-and-shmem-can-be-reclaimed"),
    pytest.param(0, 3, 5, True, False, id="post-stop-memory-is-still-rechecked"),
])
def test_reload_admission_credits_only_memory_stopping_the_worker_can_free(
        tmp_path, monkeypatch, file_gib, private_gib, available_after_gib, stopped, loaded):
    import onpu.runtime as serving

    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        previous = {"id": "previous", "name": "Previous"}
        old_process = SimpleNamespace(pid=123)
        runtime.model, runtime.state, runtime.process = previous, "ready", old_process
        memory_reads, stops, launches = [], [], []
        snapshots = iter([4, available_after_gib])

        def memory():
            available = next(snapshots) * 1024**3
            memory_reads.append(available)
            return {"total": 16 * 1024**3, "available": available}

        def proc_read(path):
            if path == "/proc/123/status":
                return f"VmRSS: {3 * 1024**2} kB\nRssFile: {file_gib * 1024**2} kB\n"
            assert path == "/proc/123/smaps_rollup"
            return (f"Rss: {3 * 1024**2} kB\nPrivate_Clean: {file_gib * 1024**2} kB\n"
                    f"Private_Dirty: {(private_gib - file_gib) * 1024**2} kB\n")

        async def stop():
            stops.append(runtime.process)
            runtime.process, runtime.model, runtime.state = None, None, "idle"

        async def launch(*args, **kwargs):
            launches.append(args)
            return SimpleNamespace(pid=456)

        async def reader(process):
            pass

        monkeypatch.setattr(serving, "estimated_memory", lambda *args: 6 * 1024**3)
        monkeypatch.setattr(serving.hardware, "memory", memory)
        monkeypatch.setattr(serving.hardware, "read", proc_read)
        monkeypatch.setattr(runtime, "_stop", stop)
        monkeypatch.setattr(runtime, "_events", reader)
        monkeypatch.setattr(runtime, "_logs", reader)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
        candidate = {"id": "candidate", "name": "Candidate", "path": "/candidate.gguf"}
        try:
            if loaded:
                await runtime.load(candidate, store.settings())
                assert runtime.model is candidate and runtime.state == "loading"
            else:
                with pytest.raises(EngineError, match="available RAM") as error:
                    await runtime.load(candidate, store.settings())
                assert error.value.status == 400
                if not stopped:
                    assert runtime.model is previous and runtime.state == "ready"
                    assert runtime.process is old_process
            assert stops == ([old_process] if stopped else [])
            assert len(launches) == int(loaded)
            assert len(memory_reads) == (2 if stopped else 1)
        finally:
            await asyncio.gather(*runtime.readers)
            store.close()

    asyncio.run(run())


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
            key, _channel, first = await runtime.generate({"model": "test"})
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
        runtime.model = {"id": "long-model", "name": "Long model"}
        runtime.options = {"context": 8192}
        runtime.load_started = time.monotonic() - 500
        try:
            await runtime.check_startup_timeout()
            assert runtime.state == "error"
            assert "timed out" in runtime.message
            assert runtime.status()["model_id"] == "long-model"
            assert runtime.status()["context"] == 8192
            await runtime.unload()
            assert runtime.state == "idle"
        finally:
            store.close()
    asyncio.run(run())


def test_active_startup_progress_survives_total_300_seconds_and_still_has_a_total_limit(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        runtime.state = "preparing"
        runtime.model = {"id": "long-model", "name": "Long model"}
        runtime.load_started = time.monotonic() - 500
        runtime.startup_progress = time.monotonic() - 2
        try:
            await runtime.check_startup_timeout()
            assert runtime.state == "preparing"
            runtime.load_started = time.monotonic() - runtime.startup_total_timeout - 1
            await runtime.check_startup_timeout()
            assert runtime.state == "error" and runtime.status()["model_id"] == "long-model"
        finally:
            store.close()
    asyncio.run(run())


def test_failed_model_does_not_disable_hardware_profile_recovery(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        failed = Runtime(store, tmp_path)
        failed.model, failed.profile, failed.state = {"id": "failed", "name": "failed"}, "CL", "error"
        pool.sessions["failed"] = failed
        checked = []
        async def probe(profile):
            checked.append(profile)
            return {"available": True, "device": profile}
        pool.idle.probe = probe
        try:
            assert (await pool.probe("CL"))["available"]
            assert checked == ["CL"]
        finally:
            await pool._stop()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["device", "backend"])
def test_cancelled_discovery_reaps_its_subprocess_and_pipes(tmp_path, monkeypatch, operation):
    from onpu.hardware import backend_info
    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        create = asyncio.create_subprocess_exec
        children = []
        async def launch(*args, **kwargs):
            proc = await create(sys.executable, "-c", "import time; time.sleep(60)",
                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                start_new_session=kwargs.get("start_new_session", False))
            children.append(proc)
            return proc
        monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
        task = asyncio.create_task(runtime.probe("CPU") if operation == "device" else backend_info(tmp_path))
        try:
            while not children:
                await asyncio.sleep(.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert children[0].returncode is not None
            assert children[0].stdout.at_eof() and children[0].stderr.at_eof()
        finally:
            for proc in children:
                if proc.returncode is None:
                    proc.kill()
                    await proc.communicate()
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


@pytest.mark.parametrize("active", ["stalled", "completed", "replacement"])
def test_cancellation_timeout_rechecks_request_after_waiting_for_lifecycle(tmp_path, monkeypatch, active):
    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        runtime.model = {"id": "model", "name": "Model"}
        runtime.state, runtime.active = "generating", "stalled"
        runtime.requests["stalled"] = asyncio.Queue()
        started = asyncio.Event()
        timeout = asyncio.timeout

        def immediate_timeout(delay):
            assert delay == 45
            started.set()
            return timeout(0)

        async def send(command):
            if command["op"] == "generate":
                await runtime.requests[command["id"]].put({"event": "started"})

        monkeypatch.setattr(asyncio, "timeout", immediate_timeout)
        runtime.send = send
        await runtime.lifecycle.acquire()
        before = asyncio.all_tasks()
        await runtime.release("stalled")
        draining, = asyncio.all_tasks() - before
        try:
            await started.wait()
            # Let the cancellation deadline expire while another lifecycle
            # operation still holds the lock, then finish the old request.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not draining.done()
            if active != "stalled":
                runtime.active, runtime.state = None, "ready"
            if active == "replacement":
                new_key, _, _ = await runtime.generate({"model": "model"})
            runtime.lifecycle.release()
            await asyncio.wait_for(draining, 1)
            if active == "stalled":
                assert runtime.state == "error" and "timed out" in runtime.message
            elif active == "replacement":
                assert runtime.state == "generating" and runtime.active == new_key
                assert new_key in runtime.requests
            else:
                assert runtime.state == "ready" and runtime.active is None
            assert "stalled" not in runtime.requests
        finally:
            if runtime.lifecycle.locked():
                runtime.lifecycle.release()
            await draining
            await runtime._shutdown()
            store.close()

    asyncio.run(run())


def test_gpu_and_npu_sessions_route_and_cancel_independently(tmp_path, monkeypatch):
    async def load(self, model, options):
        self.model, self.profile, self.options, self.state = model, options["profile"], options, "ready"
    monkeypatch.setattr(Runtime, "load", load)

    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        try:
            await pool.load({"id": "gpu", "name": "GPU model"}, {"profile": "CL"})
            gpu = pool.sessions["gpu"]
            await pool.load({"id": "npu", "name": "NPU model"}, {"profile": "ROCKCHIP"})
            npu = pool.sessions["npu"]
            commands = {"gpu": [], "npu": []}
            for name, session in pool.sessions.items():
                async def send(command, name=name, session=session):
                    commands[name].append(command)
                    if command["op"] == "generate":
                        await session.requests[command["id"]].put({"event": "started", "prompt_tokens": 5})
                session.send = send
            gpu_key, _, _ = await pool.generate({"model": "gpu"})
            npu_key, _, _ = await pool.generate({"model": "npu"})
            assert gpu.active == gpu_key and npu.active == npu_key
            with pytest.raises(EngineError, match="busy"):
                await pool.generate({"model": "gpu"})
            await pool.cancel(npu_key)
            assert commands["npu"][-1] == {"op": "cancel", "id": npu_key}
            assert len(commands["gpu"]) == 1
            gpu.active = npu.active = None
            await pool.release(gpu_key)
            await pool.release(npu_key)
            assert not gpu.requests and not npu.requests
            gpu_state = gpu.state
            await pool.unload("npu")
            assert pool.sessions == {"gpu": gpu}
            assert gpu.state == gpu_state
        finally:
            await pool._stop()
            store.close()
    asyncio.run(run())


def test_npu_owner_conflict_preserves_loaded_models(tmp_path, monkeypatch):
    async def load(self, model, options):
        self.model, self.profile, self.options, self.state = model, options["profile"], options, "ready"
    monkeypatch.setattr(Runtime, "load", load)

    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        try:
            await pool.load({"id": "npu", "name": "Small model"}, {"profile": "ROCKCHIP"})
            with pytest.raises(EngineError, match="already assigned"):
                await pool.load({"id": "gpu", "name": "GPU model"}, {"profile": "CL", "npu_decode": "auto"})
            assert list(pool.sessions) == ["npu"]
            assert pool.sessions["npu"].state == "ready"
        finally:
            await pool._stop()
            store.close()
    asyncio.run(run())


def test_unused_npu_auto_placement_does_not_reserve_second_model(tmp_path, monkeypatch):
    async def load(self, model, options):
        self.model, self.profile, self.options, self.state = model, options["profile"], options, "ready"
    monkeypatch.setattr(Runtime, "load", load)

    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        try:
            await pool.load({"id": "gpu", "name": "GPU model"}, {"profile": "CL", "npu_decode": "auto"})
            pool.sessions["gpu"].info["placement"] = {"decode_layers": 0, "prefill_layers": 0}
            await pool.load({"id": "npu", "name": "Small model"}, {"profile": "ROCKCHIP"})
            assert set(pool.sessions) == {"gpu", "npu"}
        finally:
            await pool._stop()
            store.close()
    asyncio.run(run())


def test_explicit_model_id_takes_precedence_over_another_models_name(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        commands = []
        try:
            for key, name in (("gpu", "npu"), ("npu", "NPU model")):
                session = Runtime(store, tmp_path)
                session.model, session.state = {"id": key, "name": name}, "ready"
                pool.sessions[key] = session

                async def send(command, key=key, session=session):
                    commands.append((key, command["op"]))
                    if command["op"] == "generate":
                        await session.requests[command["id"]].put({"event": "started"})

                session.send = send
            key, _, _ = await pool.generate({"model": "npu"})
            await pool.cancel(key, model="npu")
            assert commands == [("npu", "generate"), ("npu", "cancel")]
            assert pool.for_model("NPU model") is pool.sessions["npu"]
            assert pool.sessions["gpu"].active is None
        finally:
            await pool._stop()
            store.close()

    asyncio.run(run())


@pytest.mark.parametrize("unloaded", ["cpu", "missing"])
def test_unloading_another_model_preserves_chat_selection(tmp_path, unloaded):
    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        try:
            for key in ("gpu", "npu", "cpu"):
                session = Runtime(store, tmp_path)
                session.model, session.state = {"id": key, "name": key}, "ready"
                pool.sessions[key] = session
            pool.selected = "gpu"
            selected = pool.current
            await pool.unload(unloaded)
            assert pool.selected == "gpu" and pool.current is selected
            assert selected.state == "ready"
        finally:
            await pool._stop()
            store.close()

    asyncio.run(run())


@pytest.mark.parametrize("profile", ["CPU", "CL"])
@pytest.mark.parametrize("placed", [0, 1])
def test_npu_probe_respects_assisted_workers_ownership(tmp_path, profile, placed):
    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        checked = []

        async def probe(profile):
            checked.append(profile)
            return {"available": True, "device": profile}

        pool.idle.probe = probe
        session = Runtime(store, tmp_path)
        session.model, session.profile, session.state = {"id": "assisted", "name": "Assisted"}, profile, "ready"
        session.options = {"profile": profile, "npu_prefill": "auto", "npu_decode": "off"}
        session.info = {"device": f"{profile}+ROCKCHIP", "placement": {"prefill_layers": placed, "decode_layers": 0}}
        pool.sessions["assisted"] = session
        try:
            assert (await pool.probe("ROCKCHIP"))["available"]
            assert checked == ([] if placed else ["ROCKCHIP"])
        finally:
            await pool._stop()
            store.close()

    asyncio.run(run())


def test_pool_shutdown_reaps_active_workers_after_their_lifecycle_finishes(tmp_path, monkeypatch):
    import onpu.runtime as serving

    class Pipe:
        closed = waited = False

        def close(self):
            self.closed = True

        async def wait_closed(self):
            assert self.closed
            self.waited = True

    class Process:
        def __init__(self, pid):
            self.pid, self.returncode, self.stdin = pid, None, Pipe()
            self.waited = False

        async def wait(self):
            self.waited = True
            assert self.returncode == 0
            return self.returncode

    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        processes, channels, reader_cancelled = {}, {}, []
        stopped = []
        npu_stopped = asyncio.Event()

        def terminate(pid, sig):
            assert sig == serving.signal.SIGTERM
            processes[pid].returncode = 0
            stopped.append(pid)
            if pid == 10002:
                npu_stopped.set()

        monkeypatch.setattr(serving.os, "killpg", terminate)

        async def reader(key):
            try:
                await asyncio.Future()
            finally:
                reader_cancelled.append(key)

        for key, pid in (("gpu", 10001), ("npu", 10002)):
            session = Runtime(store, tmp_path)
            session.model = {"id": key, "name": key}
            session.process = processes[pid] = Process(pid)
            session.state, session.active = "generating", key
            session.requests[key] = channels[key] = asyncio.Queue(maxsize=1)
            channels[key].put_nowait({"event": "chunk", "id": key})
            session.readers = [asyncio.create_task(reader(key))]
            pool.sessions[key] = session
        pool.selected = "gpu"
        await asyncio.sleep(0)
        gpu = pool.sessions["gpu"]
        with pytest.raises(EngineError, match="Stop generation"):
            await gpu.unload()
        await gpu.lifecycle.acquire()
        stopping = None
        try:
            # Match app.lifespan: the pool lock is held by the shutdown caller.
            async with pool.lifecycle:
                stopping = asyncio.create_task(pool._stop())
                await asyncio.wait_for(npu_stopped.wait(), 1)
                assert 10002 in stopped and 10001 not in stopped
                assert not stopping.done()
                gpu.lifecycle.release()
                await asyncio.wait_for(stopping, 1)
            assert not pool.sessions and pool.selected is None
            assert sorted(reader_cancelled) == ["gpu", "npu"]
            for process in processes.values():
                assert process.returncode == 0 and process.waited
                assert process.stdin.closed and process.stdin.waited
            for key, channel in channels.items():
                assert channel.get_nowait() == {"event": "error", "id": key, "message": "Inference worker stopped", "status": 503}
                assert channel.empty()
            assert gpu.requests == {} and gpu.active is None
            store.close()
            # Reader tasks have finished before the lifespan can close its store.
            await asyncio.sleep(0)
        finally:
            if gpu.lifecycle.locked():
                gpu.lifecycle.release()
            if stopping is not None:
                await stopping
            store.close()

    asyncio.run(run())


def test_pool_timeout_snapshot_survives_a_concurrent_load(tmp_path, monkeypatch):
    async def load(self, model, options):
        self.model, self.profile, self.options, self.state = model, options["profile"], options, "ready"

    monkeypatch.setattr(Runtime, "load", load)

    async def run():
        store = Store(tmp_path / "data")
        pool = RuntimePool(store, tmp_path)
        entered, finish = asyncio.Event(), asyncio.Event()
        checked = []
        await pool.load({"id": "old", "name": "old"}, {"profile": "CL"})
        old = pool.sessions["old"]

        async def old_timeout():
            checked.append("old")
            old.state = "stopping"
            entered.set()
            await finish.wait()
            old.state = "error"

        async def new_timeout():
            checked.append("new")

        old.check_startup_timeout = old_timeout
        task = asyncio.create_task(pool.check_startup_timeout())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await pool.load({"id": "new", "name": "new"}, {"profile": "CPU"})
            pool.sessions["new"].check_startup_timeout = new_timeout
            finish.set()
            await asyncio.wait_for(task, 1)
            assert checked == ["old"]
            await pool.check_startup_timeout()
            assert checked == ["old", "old", "new"]
        finally:
            finish.set()
            await task
            await pool._stop()
            store.close()

    asyncio.run(run())


def test_startup_deadline_is_rechecked_after_waiting_for_lifecycle(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        session = Runtime(store, tmp_path)
        session.state = "preparing"
        session.load_started = time.monotonic() - session.startup_total_timeout - 1
        await session.lifecycle.acquire()
        task = asyncio.create_task(session.check_startup_timeout())
        try:
            await asyncio.sleep(0)
            assert not task.done()
            # A load already holding the lock has just started a fresh worker.
            session.load_started = session.startup_progress = time.monotonic()
            session.lifecycle.release()
            await asyncio.wait_for(task, 1)
            assert session.state == "preparing"
        finally:
            if session.lifecycle.locked():
                session.lifecycle.release()
            await task
            store.close()

    asyncio.run(run())


def test_shutdown_preserves_already_completed_request_events(tmp_path):
    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        channel = asyncio.Queue()
        done = {"event": "done", "id": "completed"}
        channel.put_nowait(done)
        runtime.requests["completed"] = channel
        runtime.state = "ready"
        try:
            await runtime._shutdown()
            assert channel.get_nowait() is done
            assert channel.empty() and runtime.requests == {}
        finally:
            store.close()

    asyncio.run(run())


def test_generation_during_memory_estimation_preserves_the_existing_worker(tmp_path, monkeypatch):
    import onpu.runtime as serving

    async def run():
        store = Store(tmp_path / "data")
        runtime = Runtime(store, tmp_path)
        previous = {"id": "previous", "name": "Previous model"}
        runtime.model, runtime.state = previous, "ready"
        estimating, finish = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        commands = []

        def estimate(*args):
            loop.call_soon_threadsafe(estimating.set)
            assert finish.wait(5)
            return 1024**3

        async def must_not_stop(*args, **kwargs):
            raise AssertionError("A generation admitted while estimating must retain its worker")

        async def send(command):
            commands.append(command)
            if command["op"] == "generate":
                await runtime.requests[command["id"]].put({"event": "started", "prompt_tokens": 5})

        monkeypatch.setattr(serving, "estimated_memory", estimate)
        monkeypatch.setattr(runtime, "_stop", must_not_stop)
        runtime.send = send
        loading = asyncio.create_task(runtime.load({"id": "candidate"}, store.settings()))
        key = None
        try:
            await asyncio.wait_for(estimating.wait(), 1)
            key, channel, _ = await runtime.generate({"model": "previous"})
            finish.set()
            with pytest.raises(EngineError, match="Stop the current request"):
                await asyncio.wait_for(loading, 1)
            assert runtime.model is previous and runtime.state == "generating"
            assert runtime.active == key and runtime.requests[key] is channel
            assert [command["op"] for command in commands] == ["generate"]
        finally:
            finish.set()
            with contextlib.suppress(EngineError):
                await loading
            runtime.active = None
            if key is not None:
                await runtime.release(key)
            store.close()

    asyncio.run(run())
