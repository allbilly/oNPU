from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, hardware
from .hardware import backend_info
from .library import CATALOG, Downloads, estimated_memory, register
from .runtime import EngineError, Runtime
from .store import Store


class LoadOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: str = Field(default="CPU", pattern="^(CPU|CL)$")
    context: int = Field(default=1024, ge=64, le=32768)
    realize: bool = True


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: int = Field(default=1024, ge=64, le=32768)
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.6, ge=0, le=2)
    profile: str = Field(default="CPU", pattern="^(CPU|CL)$")
    realize: bool = True
    idle_timeout: int = Field(default=600, ge=0, le=86400)
    memory_fraction: float = Field(default=0.8, ge=0.2, le=0.9)


class ImportModel(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=300)
    messages: list[dict] = Field(min_length=1, max_length=256)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float = Field(default=0.6, ge=0, le=2)
    stream_options: dict | None = None
    tools: list[dict] | None = None
    enable_thinking: bool = False


class DownloadBody(BaseModel):
    catalog_id: str | None = None
    repo: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    file: str | None = None
    revision: str = Field(default="main", pattern=r"^[\w.-]+$")


class BenchmarkBody(BaseModel):
    prompt: str = Field(default="Explain why Rockchip single board computers are useful for local AI.", min_length=1, max_length=100000)
    max_tokens: int = Field(default=32, ge=2, le=1024)
    repeats: int = Field(default=3, ge=1, le=10)


def create_app(backend: Path | None = None, data: Path | None = None, api_key: str | None = None) -> FastAPI:
    backend = (backend or Path(os.environ.get("ONPU_BACKEND", "~/tinygrad"))).expanduser().resolve()
    data = (data or Path(os.environ.get("ONPU_DATA_DIR", ".data"))).expanduser().resolve()
    api_key = api_key if api_key is not None else os.environ.get("ONPU_API_KEY", "")
    store = Store(data)
    runtime = Runtime(store, backend)
    downloads = Downloads(store)
    profiles = {"CPU": {"available": None, "name": "CPU", "reason": "Checking device"},
                "CL": {"available": None, "name": "Mali OpenCL", "reason": "Checking device"},
                "ROCKCHIP": {"available": False, "name": "Rockchip NPU assist", "reason": "Experimental: selective LLM offload is not yet qualified."}}
    bench_tasks: dict[str, asyncio.Task] = {}
    started = time.monotonic()
    identity = {"path": str(backend), "available": (backend / "tinygrad/llm/model.py").is_file()}

    async def discover():
        identity.update(await backend_info(backend))
        if identity["available"]:
            # Serial checks avoid competing with a device during initial discovery.
            for profile in ("CPU", "CL"):
                profiles[profile] = await runtime.probe(profile)
        else:
            for profile in ("CPU", "CL"):
                profiles[profile] = {"available": False, "name": profile, "reason": "tinygrad checkout not found"}

    async def housekeeping():
        while True:
            await asyncio.sleep(5)
            await runtime.check_startup_timeout()
            timeout = store.settings()["idle_timeout"]
            if timeout and runtime.state == "ready" and not runtime.active and time.monotonic() - runtime.last_used > timeout:
                with contextlib.suppress(EngineError):
                    await runtime.unload()

    @asynccontextmanager
    async def lifespan(app):
        checks, timer = asyncio.create_task(discover()), asyncio.create_task(housekeeping())
        for item in store.list("benchmarks"):
            if item["state"] == "running":
                store.put("benchmarks", {**item, "state": "interrupted", "error": "Service restarted"})
        try:
            yield
        finally:
            for task in [checks, timer, *bench_tasks.values()]:
                task.cancel()
            await asyncio.gather(checks, timer, *bench_tasks.values(), return_exceptions=True)
            await downloads.close()
            async with runtime.lifecycle:
                await runtime._stop()
            store.close()

    app = FastAPI(title="oNPU", version=__version__, lifespan=lifespan)
    app.state.runtime, app.state.store = runtime, store
    static = Path(__file__).parent / "static"

    @app.middleware("http")
    async def access(request: Request, call_next):
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"error": {"message": "Invalid content length"}}, status_code=400)
        if length > 2 * 1024**2:
            return JSONResponse({"error": {"message": "Request body exceeds 2 MiB"}}, status_code=413)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"error": {"message": "Cross-origin changes are not allowed"}}, status_code=403)
        public = request.url.path in ("/", "/health", "/api/login") or request.url.path.startswith("/static/")
        if api_key and not public:
            token = request.headers.get("authorization", "").removeprefix("Bearer ") or request.cookies.get("onpu_session", "")
            if not secrets.compare_digest(token, api_key):
                return JSONResponse({"error": {"message": "Sign in with the server API key"}}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.exception_handler(EngineError)
    async def engine_error(request, exc):
        return JSONResponse({"error": {"message": str(exc), "type": "engine_error"}}, status_code=exc.status)

    @app.exception_handler(httpx.HTTPError)
    async def remote_error(request, exc):
        return JSONResponse({"error": {"message": f"Model host request failed: {exc}"}}, status_code=502)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/")
    async def home():
        return FileResponse(static / "index.html")

    @app.post("/api/login")
    async def login(body: dict, request: Request):
        if api_key and not secrets.compare_digest(str(body.get("key", "")), api_key):
            raise HTTPException(401, "Invalid API key")
        response = JSONResponse({"ok": True})
        if api_key:
            response.set_cookie("onpu_session", api_key, httponly=True, samesite="strict", secure=request.url.scheme == "https", max_age=86400)
        return response

    @app.post("/api/logout")
    async def logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie("onpu_session")
        return response

    @app.get("/api/status")
    async def status():
        metrics = store.list("metrics", 200)
        return {"version": __version__, "uptime": time.monotonic() - started, "runtime": runtime.status(),
                "hardware": hardware.snapshot(data), "backend": identity, "profiles": profiles,
                "totals": {"requests": len(metrics), "tokens": sum(m["completion_tokens"] for m in metrics)}}

    @app.get("/api/logs")
    async def logs():
        return {"lines": list(runtime.logs)}

    @app.post("/api/profiles/check")
    async def check_profiles():
        if runtime.active or runtime.state in ("loading", "preparing"):
            raise EngineError("Wait for inference to finish before checking devices")
        await discover()
        return profiles

    @app.get("/api/models")
    async def models():
        settings = store.settings()
        return {"models": [{**item, "state": runtime.state if runtime.model and runtime.model["id"] == item["id"] else "installed",
                            "estimated_bytes": estimated_memory(item, settings["context"], settings["realize"]),
                            "present": Path(item["path"]).is_file()} for item in store.list("models")], "catalog": CATALOG}

    @app.post("/api/models/import")
    async def import_model(body: ImportModel):
        try:
            return await asyncio.to_thread(register, store, Path(body.path))
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/models/{key}/load", status_code=202)
    async def load(key: str, body: LoadOptions):
        model = store.get("models", key)
        if not model:
            raise HTTPException(404, "Model not found")
        if not Path(model["path"]).is_file():
            raise HTTPException(400, "Model file is missing")
        if profiles[body.profile]["available"] is False:
            raise EngineError(profiles[body.profile].get("reason", "Device is unavailable"), 400)
        if body.context > model.get("context_limit", body.context):
            raise HTTPException(400, "Requested context exceeds the model's declared capacity")
        await runtime.load(model, {**store.settings(), **body.model_dump()})
        return runtime.status()

    @app.post("/api/models/unload")
    async def unload():
        await runtime.unload()
        return runtime.status()

    @app.delete("/api/models/{key}")
    async def remove(key: str):
        if runtime.model and runtime.model["id"] == key:
            raise EngineError("Unload the model before removing it")
        if not store.get("models", key):
            raise HTTPException(404, "Model not found")
        # Removing a library reference preserves the original model file.
        store.delete("models", key)
        return {"ok": True}

    @app.get("/api/settings")
    async def get_settings():
        return {"settings": {k: v for k, v in store.settings().items() if k != "id"}, "auth_enabled": bool(api_key),
                "data_dir": str(data), "backend_path": str(backend)}

    @app.put("/api/settings")
    async def save_settings(body: Settings):
        if body.max_tokens >= body.context:
            raise HTTPException(400, "Reply limit must be smaller than context size")
        store.put("settings", {"id": "global", **body.model_dump()})
        return body.model_dump()

    @app.get("/api/downloads")
    async def download_jobs():
        return {"jobs": store.list("downloads")}

    @app.post("/api/downloads", status_code=202)
    async def download(body: DownloadBody):
        if body.catalog_id:
            spec = next((x for x in CATALOG if x["id"] == body.catalog_id), None)
            if not spec:
                raise HTTPException(404, "Catalog model not found")
        else:
            if not body.repo or not body.file:
                raise HTTPException(400, "Choose a repository and an exact GGUF file")
            spec = body.model_dump()
        try:
            return downloads.start(spec)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/downloads/{key}/pause")
    async def pause(key: str):
        await downloads.cancel(key)
        return store.get("downloads", key)

    @app.post("/api/downloads/{key}/resume", status_code=202)
    async def resume(key: str):
        job = store.get("downloads", key)
        if not job:
            raise HTTPException(404, "Download not found")
        if job["state"] not in ("paused", "error"):
            raise HTTPException(409, "Only paused or failed downloads can be resumed")
        return downloads.start(job, key)

    @app.get("/api/huggingface/search")
    async def hf_search(q: str):
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            response = await client.get("https://huggingface.co/api/models", params={"search": q[:100], "filter": "gguf", "limit": 12})
            response.raise_for_status()
            return {"repos": [{"id": x["id"], "downloads": x.get("downloads", 0)} for x in response.json()]}

    @app.get("/api/huggingface/files")
    async def hf_files(repo: str):
        if len(repo.split("/")) != 2 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./" for c in repo):
            raise HTTPException(400, "Invalid repository name")
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            response = await client.get(f"https://huggingface.co/api/models/{repo}")
            response.raise_for_status()
            result = response.json()
            return {"revision": result["sha"], "files": [f["rfilename"] for f in result.get("siblings", [])
                    if f["rfilename"].endswith(".gguf") and "/" not in f["rfilename"]]}

    @app.get("/v1/models")
    async def openai_models():
        return {"object": "list", "data": [{"id": item["id"], "object": "model", "created": int(item["imported_at"]),
                                               "owned_by": "local", "name": item["name"]} for item in store.list("models")]}

    async def start_request(body: ChatBody):
        if not runtime.model or body.model not in (runtime.model["id"], runtime.model["name"]):
            raise EngineError("Load the requested model before sending a chat request", 404)
        for message in body.messages:
            if message.get("role") not in ("system", "user", "assistant", "tool"):
                raise HTTPException(400, "Unsupported message role")
            if not isinstance(message.get("content", ""), (str, list, type(None))):
                raise HTTPException(400, "Message content must be text")
            if len(json.dumps(message)) > 500000:
                raise HTTPException(400, "Message is too large")
        payload = body.model_dump()
        payload["max_tokens"] = body.max_completion_tokens or body.max_tokens or store.settings()["max_tokens"]
        return await runtime.generate(payload)

    @app.post("/v1/chat/completions")
    async def chat(body: ChatBody, request: Request):
        key, channel, prep = await start_request(body)

        async def events():
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(channel.get(), 10)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if event["event"] == "chunk":
                        chunk = event["data"]
                        if chunk.get("usage") and not (body.stream_options or {}).get("include_usage"):
                            continue
                        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                    elif event["event"] == "done":
                        yield "data: [DONE]\n\n"
                        break
                    elif event["event"] in ("error", "cancelled"):
                        yield "data: " + json.dumps({"error": {"message": event.get("message", "Generation stopped")}}) + "\n\n"
                        yield "data: [DONE]\n\n"
                        break
            finally:
                await asyncio.shield(runtime.release(key))

        if body.stream:
            return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": key})
        output, reasoning, tool_calls, usage, finish, base = [], [], [], {}, "stop", {}
        try:
            while True:
                try:
                    event = await asyncio.wait_for(channel.get(), 0.5)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        raise EngineError("Client disconnected", 499)
                    continue
                if event["event"] == "chunk":
                    base = event["data"]
                    usage = base.get("usage", usage)
                    for choice in base.get("choices", []):
                        delta = choice.get("delta", {})
                        output.append(delta.get("content", ""))
                        reasoning.append(delta.get("reasoning_content", ""))
                        tool_calls.extend({k: v for k, v in tc.items() if k != "index"} for tc in delta.get("tool_calls", []))
                        finish = choice.get("finish_reason") or finish
                elif event["event"] == "done":
                    message = {"role": "assistant", "content": "".join(output) or None}
                    if any(reasoning):
                        message["reasoning_content"] = "".join(reasoning)
                    if tool_calls:
                        message["tool_calls"] = tool_calls
                    return {"id": base.get("id", key), "object": "chat.completion", "created": int(time.time()), "model": body.model,
                            "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
                elif event["event"] in ("error", "cancelled"):
                    raise EngineError(event.get("message", "Generation stopped"), event.get("status", 409))
        finally:
            await asyncio.shield(runtime.release(key))

    @app.post("/api/chat/stop")
    async def stop():
        await runtime.cancel()
        return {"ok": True}

    @app.get("/api/metrics")
    async def metrics():
        return {"metrics": store.list("metrics", 200)}

    async def benchmark(job: dict, body: BenchmarkBody):
        key = None
        try:
            for index in range(body.repeats):
                # Unique suffix prevents reused prefixes from masquerading as uncached prefill.
                request = ChatBody(model=runtime.model["id"], messages=[{"role": "user", "content": f"Trial {uuid.uuid4().hex[:8]}: {body.prompt}"}],
                                   temperature=0, max_tokens=body.max_tokens)
                key, channel, _ = await start_request(request)
                while True:
                    event = await channel.get()
                    if event["event"] == "done":
                        job["results"].append(event["metrics"])
                        break
                    if event["event"] in ("error", "cancelled"):
                        raise EngineError(event.get("message", "Benchmark stopped"))
                await runtime.release(key)
                key = None
                job["completed"] = index + 1
                store.put("benchmarks", job)
            job["state"] = "complete"
        except asyncio.CancelledError:
            job["state"] = "cancelled"
        except Exception as exc:
            job.update(state="error", error=str(exc))
        finally:
            if key:
                await runtime.release(key)
            store.put("benchmarks", job)

    @app.get("/api/benchmarks")
    async def benchmarks():
        return {"jobs": store.list("benchmarks")}

    @app.post("/api/benchmarks", status_code=202)
    async def run_benchmark(body: BenchmarkBody):
        if runtime.state != "ready" or any(not t.done() for t in bench_tasks.values()):
            raise EngineError("Load a model and wait for it to be ready before benchmarking")
        job = {"id": uuid.uuid4().hex[:16], "state": "running", "model": runtime.model["name"], "model_id": runtime.model["id"],
               "profile": runtime.profile, "context": runtime.info.get("context"), "revision": identity.get("revision"),
               "kernel": hardware.snapshot(data)["kernel"], "options": runtime.options, "created": time.time(),
               "repeats": body.repeats, "completed": 0, "results": [], "error": None, "qualification": "Performance only · quality not evaluated"}
        store.put("benchmarks", job)
        bench_tasks[job["id"]] = asyncio.create_task(benchmark(job, body))
        return job

    @app.post("/api/benchmarks/{key}/cancel")
    async def cancel_benchmark(key: str):
        if (task := bench_tasks.get(key)) and not task.done():
            task.cancel()
            await task
        return store.get("benchmarks", key)

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
