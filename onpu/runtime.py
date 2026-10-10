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
        self.profile = "CPU"
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
        self.startup_timeout = max(30, float(os.environ.get("ONPU_STARTUP_TIMEOUT", "300")))

    def environment(self, profile: str, realize=True) -> dict:
        env = dict(os.environ)
        env.update(DEV=profile, PYTHONPATH=os.pathsep.join([str(self.backend), str(Path(__file__).resolve().parent.parent)]),
                   DEBUG="0", REALIZE=str(int(realize)), HALF="1", PYTHONUNBUFFERED="1", BEAM="0")
        return env

    @property
    def python(self) -> str:
        candidate = self.backend / ".venv/bin/python"
        return str(candidate) if candidate.exists() else sys.executable

    async def probe(self, profile: str) -> dict:
        if profile not in ("CPU", "CL"):
            return {"available": False, "name": "Rockchip NPU", "reason": "Selective LLM offload is not qualified for this checkout."}
        proc = await asyncio.create_subprocess_exec(self.python, "-m", "onpu.worker", "--model", "probe", "--context", "64", "--probe",
                                                    env=self.environment(profile), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                    cwd=str(self.backend))
        try:
            out, err = await asyncio.wait_for(proc.communicate(), 45)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"available": False, "name": profile, "reason": "Device check timed out"}
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
            if options["profile"] not in ("CPU", "CL"):
                raise EngineError("NPU-assisted profiles require model-level qualification and are not enabled", 400)
            available = hardware.memory()
            need = estimated_memory(model, options["context"], options["realize"])
            if available["total"] and need > available["total"] * options["memory_fraction"]:
                raise EngineError("Estimated model memory exceeds the configured RAM budget", 400)
            await self._stop()
            if available["available"] and need > hardware.memory()["available"] - 512 * 1024**2:
                raise EngineError("Not enough available RAM to load this model safely", 400)
            self.model, self.profile, self.options = model, options["profile"], dict(options)
            self.info = {}
            self.state, self.message = "loading", "Starting tinygrad worker"
            self.load_started = time.monotonic()
            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.python, "-m", "onpu.worker", "--model", model["path"], "--context", str(options["context"]),
                    "--realize", str(int(options["realize"])), env=self.environment(self.profile, options["realize"]),
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
                        await self._stop()
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
        if self.state in ("loading", "preparing") and time.monotonic() - self.load_started > self.startup_timeout:
            async with self.lifecycle:
                if self.state in ("loading", "preparing"):
                    await self._stop()
                    self.state = "error"
                    self.message = "Model preparation timed out. Worker stopped; try CPU or a smaller context."

    async def _stop(self):
        self.state = "stopping"
        if self.process and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 8)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
        for task in self.readers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.readers, return_exceptions=True)
        self.readers = []
        self.process, self.model, self.active, self.info = None, None, None, {}
        self.state, self.message = "idle", "Choose a model to get started"

    def status(self) -> dict:
        return {"state": self.state, "message": self.message, "model_id": self.model["id"] if self.model else None,
                "model_name": self.model["name"] if self.model else None, "selected_profile": self.profile,
                "actual_device": self.info.get("device"), "context": self.info.get("context"), "load_seconds": self.info.get("load_seconds"),
                "rss_bytes": hardware.rss(self.process.pid if self.process else None), "active_request": self.active,
                "last_metrics": self.last_metrics, "options": self.options}
