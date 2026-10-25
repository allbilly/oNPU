from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
import uuid
from collections import deque
from pathlib import Path

from . import hardware
from .library import estimated_memory
from .speculation import validate_speculation
from .store import Store


class EngineError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


class Runtime:
    def __init__(self, store: Store, backend: Path):
        self.store, self.backend = store, backend
        self.process: asyncio.subprocess.Process | None = None
        self.state = "idle"
        self.message = "Choose a model to get started"
        self.model: dict | None = None
        self.profile = "CL"
        self.info: dict = {}
        self.logs = deque(maxlen=160)
        self.requests: dict[str, asyncio.Queue] = {}
        self.active: str | None = None
        self.lifecycle = asyncio.Lock()
        self.readers: list[asyncio.Task] = []
        self.last_used = time.monotonic()
        self.last_metrics: dict | None = None
        self.options: dict = {}
        self.load_started = 0.0
        self.startup_progress = 0.0
        self.startup_timeout = max(30, float(os.environ.get("ONPU_STARTUP_TIMEOUT", "300")))
        self.startup_total_timeout = max(self.startup_timeout, float(os.environ.get("ONPU_STARTUP_TOTAL_TIMEOUT", "1800")))

    def environment(self, profile: str, realize=True, npu_prefill="off", npu_decode="off", vision_gpu=True) -> dict:
        env = dict(os.environ)
        env.update(DEV="CPU" if profile == "ROCKCHIP" else profile,
                   PYTHONPATH=os.pathsep.join([str(self.backend), str(Path(__file__).resolve().parent.parent)]),
                   DEBUG="0", REALIZE=str(int(realize)), HALF="1", PYTHONUNBUFFERED="1", BEAM="0",
                   ONPU_PROFILE=profile, ONPU_NPU_PREFILL=npu_prefill,
                   ONPU_NPU_DECODE="on" if profile == "ROCKCHIP" else npu_decode,
                   ONPU_VISION_GPU=str(int(vision_gpu)))
        return env

    @property
    def python(self) -> str:
        candidate = self.backend / ".venv/bin/python"
        return str(candidate) if candidate.exists() else sys.executable

    async def probe(self, profile: str) -> dict:
        if profile not in ("CPU", "CL", "ROCKCHIP"):
            return {"available": False, "name": profile, "reason": "Unknown device profile"}
        proc = await asyncio.create_subprocess_exec(self.python, "-m", "onpu.worker", "--model", "probe", "--context", "64", "--probe",
                                                    env=self.environment(profile), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                    cwd=str(self.backend), start_new_session=True)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), 45)
        except TimeoutError:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            await proc.communicate()
            return {"available": False, "name": profile, "reason": "Device check timed out"}
        except BaseException:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            await asyncio.shield(proc.communicate())
            raise
        for line in out.decode(errors="replace").splitlines():
            try:
                data = json.loads(line)
                if data.get("event") == "probe":
                    return data
            except json.JSONDecodeError:
                pass
        return {"available": False, "name": profile, "reason": err.decode(errors="replace")[-1200:] or "Backend probe failed"}

    async def load(self, model: dict, options: dict):
        async with self.lifecycle:
            if self.active or self.state in ("loading", "preparing"):
                raise EngineError("Stop the current request or finish loading first")
            if options["profile"] not in ("CPU", "CL", "ROCKCHIP"):
                raise EngineError("Unknown device profile", 400)
            if options["profile"] == "ROCKCHIP" and not options["realize"]:
                raise EngineError("NPU inference requires prepared FP16 weights", 400)
            if not options["realize"] and any(options.get(k, "off") != "off" for k in ("npu_prefill", "npu_decode")):
                raise EngineError("NPU assistance requires prepared FP16 weights", 400)
            try:
                draft_path = await asyncio.to_thread(validate_speculation, model, options)
            except (ValueError, OSError) as exc:
                raise EngineError(str(exc), 400) from exc
            if draft_path is not None:
                options = {**options, "draft_path": draft_path}
            need = await asyncio.to_thread(estimated_memory, model, options["context"], options["realize"], options)
            # Estimation can hash a checkpoint while the current worker remains
            # ready. A generation admitted during that await owns the worker.
            if self.active or self.state in ("loading", "preparing"):
                raise EngineError("Stop the current request or finish loading first")
            available = hardware.memory()
            if options["profile"] == "CL" and any(options.get(k, "off") != "off" for k in ("npu_prefill", "npu_decode")):
                need += model.get("parameters", 0) * 2
            if (options["profile"] == "ROCKCHIP" and model.get("architecture") == "idefics3"
                    and model.get("kv_heads") == 3 and model.get("head_dim") == 64 and 4096 <= options["context"] <= 8192):
                padded = (options["context"] + 3) // 4 * 4
                need += model.get("layers", 0) * (64 + 3 * 64 * (padded * 2 + (padded + 16) * 4))
            if available["total"] and need > available["total"] * options["memory_fraction"]:
                raise EngineError("Estimated model memory exceeds the configured RAM budget", 400)
            reclaimable = hardware.reclaimable_rss(self.process.pid if self.process else None)
            if available["available"] and need > available["available"] + reclaimable - 512 * 1024**2:
                raise EngineError("Not enough available RAM to load this model safely", 400)
            await self._stop()
            if available["available"] and need > hardware.memory()["available"] - 512 * 1024**2:
                raise EngineError("Not enough available RAM to load this model safely", 400)
            self.model, self.profile, self.options = model, options["profile"], dict(options)
            self.info = {}
            self.state, self.message = "loading", "Starting tinygrad worker"
            self.load_started = time.monotonic()
            self.startup_progress = self.load_started
            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.python, "-m", "onpu.worker", "--model", model["path"], "--context", str(options["context"]),
                    "--realize", str(int(options["realize"])),
                    "--speculation", options.get("speculation", "off"), "--draft-path", options.get("draft_path") or "",
                    env=self.environment(self.profile, options["realize"], options.get("npu_prefill", "off"), options.get("npu_decode", "off"),
                                         options.get("vision_gpu", True)),
                    cwd=str(self.backend), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    limit=2 * 1024 * 1024, start_new_session=True)
                self.readers = [asyncio.create_task(self._events(self.process)), asyncio.create_task(self._logs(self.process))]
            except Exception as exc:
                self.state, self.message = "error", str(exc)
                raise EngineError(str(exc), 500) from exc

    async def _events(self, process):
        try:
            while line := await process.stdout.readline():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self.logs.append(line.decode(errors="replace").strip())
                    continue
                kind = event.get("event")
                if kind == "state":
                    self.state, self.message = event["state"], event["message"]
                    if self.state in ("loading", "preparing"):
                        self.startup_progress = time.monotonic()
                elif kind == "ready":
                    self.state, self.message, self.info = "ready", "Ready for a conversation", event
                    self.last_used = time.monotonic()
                elif kind == "fatal":
                    self.state, self.message = "error", event["message"]
                elif key := event.get("id"):
                    if kind == "chunk" and any(choice.get("delta", {}).get(field) for choice in event.get("data", {}).get("choices", [])
                                                for field in ("content", "reasoning_content")):
                        self.message = "Generating reply"
                    if channel := self.requests.get(key):
                        await channel.put(event)
                    if kind in ("done", "cancelled", "error"):
                        if kind == "done":
                            self.last_metrics = {**event["metrics"], "id": key, "model": self.model["id"], "profile": self.profile,
                                                 "context": self.info.get("context"), "created": time.time(), "model_path": self.model["path"]}
                            self.store.put("metrics", self.last_metrics)
                        self.active = None
                        self.state, self.message = "ready", "Ready for a conversation"
                        self.last_used = time.monotonic()
        finally:
            code = await process.wait()
            if process is self.process and self.state not in ("idle", "stopping"):
                had_error = self.state == "error"
                self.state = "error"
                if not had_error:
                    self.message = f"Worker exited ({code}). Reload the model to recover."
                for key, channel in list(self.requests.items()):
                    await channel.put({"event": "error", "id": key, "message": self.message, "status": 500})
                self.active = None

    async def _logs(self, process):
        while line := await process.stderr.readline():
            self.logs.append(line.decode(errors="replace").strip())

    async def send(self, command: dict):
        if not self.process or self.process.returncode is not None:
            raise EngineError("Inference worker is not running", 503)
        try:
            self.process.stdin.write((json.dumps(command) + "\n").encode())
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise EngineError("Inference worker exited", 503) from exc

    async def generate(self, body: dict) -> tuple[str, asyncio.Queue, dict]:
        # No await occurs between admission checks and reservation: one model owner.
        if self.state != "ready" or self.active:
            raise EngineError("The model is busy or not ready. Try again when it is ready.")
        key, channel = uuid.uuid4().hex, asyncio.Queue(maxsize=512)
        self.active, self.state, self.message = key, "generating", "Processing prompt"
        self.requests[key] = channel
        try:
            await self.send({"op": "generate", "id": key, "body": body})
            first = await asyncio.wait_for(channel.get(), 30)
            if first["event"] == "error":
                raise EngineError(first["message"], first.get("status", 500))
            if first["event"] != "started":
                raise EngineError("Request was cancelled", 409)
            return key, channel, first
        except BaseException:
            with contextlib.suppress(EngineError):
                await asyncio.shield(self.cancel(key))
            self.requests.pop(key, None)
            raise

    async def cancel(self, key: str | None = None):
        key = key or self.active
        if key and key == self.active:
            self.message = "Stopping after the current device operation"
            await self.send({"op": "cancel", "id": key})

    async def release(self, key: str):
        if self.active == key:
            with contextlib.suppress(EngineError):
                await self.cancel(key)
        # Drain events in the background after an HTTP disconnect so backpressure
        # cannot prevent the worker from acknowledging cancellation.
        channel = self.requests.get(key)
        if self.active == key and channel:
            async def drain():
                try:
                    async with asyncio.timeout(45):
                        while (await channel.get())["event"] not in ("done", "cancelled", "error"):
                            pass
                except TimeoutError:
                    async with self.lifecycle:
                        if self.active == key:
                            await self._stop(preserve_model=True)
                            self.state, self.message = "error", "Cancellation timed out; worker stopped. Reload the model."
                finally:
                    self.requests.pop(key, None)
            asyncio.create_task(drain())
        else:
            self.requests.pop(key, None)

    async def unload(self):
        async with self.lifecycle:
            if self.active:
                raise EngineError("Stop generation before unloading the model")
            await self._stop()

    async def check_startup_timeout(self):
        if self.state in ("loading", "preparing"):
            async with self.lifecycle:
                now = time.monotonic()
                stalled = now - (self.startup_progress or self.load_started) > self.startup_timeout
                expired = now - self.load_started > self.startup_total_timeout
                if self.state in ("loading", "preparing") and (stalled or expired):
                    await self._stop(preserve_model=True)
                    self.state = "error"
                    self.message = "Model preparation timed out. Worker stopped; try CPU or a smaller context."

    async def _shutdown(self):
        # Service shutdown must reap active workers too, after any load/unload.
        async with self.lifecycle:
            await self._stop()

    async def _stop(self, preserve_model=False):
        self.state = "stopping"
        if self.process and self.process.stdin:
            self.process.stdin.close()
        if self.process and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 8)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
        for task in self.readers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.readers, return_exceptions=True)
        if self.process and self.process.stdin:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await self.process.stdin.wait_closed()
        self.readers = []
        if (key := self.active) is not None and (channel := self.requests.get(key)) is not None:
            # A disconnected consumer may have left a full queue. Teardown must
            # neither block on it nor leave surviving consumers waiting forever.
            while not channel.empty():
                channel.get_nowait()
            channel.put_nowait({"event": "error", "id": key, "message": "Inference worker stopped", "status": 503})
        self.requests.clear()
        self.process, self.active = None, None
        if not preserve_model:
            self.model, self.info = None, {}
        self.state, self.message = "idle", "Choose a model to get started"

    def status(self) -> dict:
        return {"state": self.state, "message": self.message, "model_id": self.model["id"] if self.model else None,
                "model_name": self.model["name"] if self.model else None, "selected_profile": self.profile,
                "actual_device": self.info.get("device"), "context": self.info.get("context", self.options.get("context")),
                "load_seconds": self.info.get("load_seconds"),
                "base_device": self.info.get("base_device"), "placement": self.info.get("placement"),
                "gpu_kernels": self.info.get("gpu_kernels"),
                "speculation": self.info.get("speculation", {"mode": "off"}),
                "backend_path": self.info.get("backend_path"),
                "supports_images": self.info.get("supports_images", False),
                "rss_bytes": hardware.rss(self.process.pid if self.process else None), "active_request": self.active,
                "last_metrics": self.last_metrics, "options": self.options}


class RuntimePool:
    """Independent model owners, retaining the selected session for the existing UI.

    Requests route by model ID. A GPU-only model and an NPU-assisted model can run
    simultaneously; two workers never compete for the same NPU owner lock.
    """
    def __init__(self, store: Store, backend: Path):
        self.store, self.backend = store, backend
        self.sessions: dict[str, Runtime] = {}
        self.selected: str | None = None
        self.lifecycle = asyncio.Lock()
        self.idle = Runtime(store, backend)

    @property
    def current(self):
        return self.sessions.get(self.selected, self.idle)

    def __getattr__(self, name):
        return getattr(self.current, name)

    @property
    def active(self):
        return next((r.active for r in self.sessions.values() if r.active), None)

    def for_model(self, key):
        if (session := self.sessions.get(key)) is not None and session.model:
            return session
        return next((r for r in self.sessions.values() if r.model and key == r.model["name"]), None)

    async def probe(self, profile):
        existing = next((r for r in self.sessions.values() if r.state not in ("idle", "error")
                         and (r.profile == profile or profile == "ROCKCHIP" and self.uses_npu(r.options, r.info.get("placement")))), None)
        if existing:
            return {"available": existing.state != "error", "name": profile, "device": existing.info.get("device"), "loaded": True}
        return await self.idle.probe(profile)

    @staticmethod
    def uses_npu(options, placement=None):
        if options["profile"] == "ROCKCHIP":
            return True
        if placement is not None:
            return bool(placement["decode_layers"] or placement["prefill_layers"])
        return any(options.get(k, "off") != "off" for k in ("npu_prefill", "npu_decode"))

    async def load(self, model, options):
        async with self.lifecycle:
            key = model["id"]
            for other_key, other in self.sessions.items():
                if other_key == key:
                    continue
                if other.state in ("idle", "error"):
                    continue
                if other.state in ("loading", "preparing"):
                    raise EngineError("Another model is still preparing. Wait until it is ready before loading this model.", 409)
                if other.profile == options["profile"]:
                    raise EngineError("A model is already loaded on this profile. Unload it first.")
                if self.uses_npu(options) and self.uses_npu(other.options, other.info.get("placement")):
                    raise EngineError("The NPU is already assigned to another model. Use GPU-only mode or unload that model.")
            session = self.sessions.get(key) or Runtime(self.store, self.backend)
            # Runtime's admission check also accounts for memory held by other workers.
            await session.load(model, options)
            self.sessions[key], self.selected = session, key

    async def generate(self, body):
        session = self.for_model(body["model"])
        if session is None:
            raise EngineError("Load the requested model before sending a chat request", 404)
        return await session.generate(body)

    async def release(self, key):
        for session in self.sessions.values():
            if key in session.requests:
                await session.release(key)
                return

    async def cancel(self, key=None, model=None):
        session = self.for_model(model) if model else next((r for r in self.sessions.values() if key and r.active == key), self.current)
        if session is not None:
            await session.cancel(key)

    async def unload(self, model=None):
        async with self.lifecycle:
            key = model or self.selected
            if key in self.sessions:
                await self.sessions[key].unload()
                del self.sessions[key]
            if self.selected not in self.sessions:
                self.selected = next(reversed(self.sessions), None)

    async def check_startup_timeout(self):
        for session in list(self.sessions.values()):
            await session.check_startup_timeout()

    async def _stop(self):
        await asyncio.gather(*(session._shutdown() for session in self.sessions.values()))
        self.sessions.clear()
        self.selected = None

    def status(self):
        return {**self.current.status(), "sessions": [session.status() for session in self.sessions.values()]}
