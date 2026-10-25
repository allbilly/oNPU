import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from onpu.app import create_app
from onpu.runtime import Runtime


def client(tmp_path, base_url="http://127.0.0.1", **kwargs):
    return TestClient(create_app(tmp_path / "missing-backend", tmp_path / "data", **kwargs), base_url=base_url)


def test_model_import_remove_and_validation(tmp_path, gguf):
    with client(tmp_path) as c:
        assert c.get("/").status_code == 200
        assert c.get("/static/app.js").status_code == 200
        result = c.post("/api/models/import", json={"path": str(gguf)})
        assert result.status_code == 200
        key = result.json()["id"]
        assert c.get("/v1/models").json()["data"][0]["id"] == key
        # NPU is a recognized profile; this fixture has no backend/device to run it.
        assert c.post(f"/api/models/{key}/load", json={"profile": "ROCKCHIP"}).status_code == 400
        assert c.post(f"/api/models/{key}/load", json={"profile": "UNKNOWN"}).status_code == 422
        assert c.post(f"/api/models/{key}/load", json={"context": 0}).status_code == 422
        assert c.post("/v1/chat/completions", json={"model": key, "messages": [{"role": "user", "content": "Hello"}]}).status_code == 404
        assert c.delete(f"/api/models/{key}").status_code == 200
        assert gguf.exists()
        assert c.get("/api/models").json()["models"] == []


def test_settings_survive_restart(tmp_path):
    with client(tmp_path) as c:
        data = c.get("/api/settings").json()["settings"]
        data.update(context=512, max_tokens=64, idle_timeout=0)
        assert c.put("/api/settings", json=data).status_code == 200
        assert c.put("/api/settings", json={**data, "max_tokens": 600}).status_code == 400
    with client(tmp_path) as c:
        settings = c.get("/api/settings").json()["settings"]
        assert settings["context"] == 512
        assert settings["idle_timeout"] == 0


def test_malformed_vision_import_is_a_client_error(tmp_path):
    path = tmp_path / "vision"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "idefics3"}))
    for name in ("model.safetensors", "tokenizer.json", "generation_config.json"):
        (path / name).write_bytes(b"{}")
    with client(tmp_path) as c:
        response = c.post("/api/models/import", json={"path": str(path)})
        assert response.status_code == 400, response.text
        assert c.get("/api/models").json()["models"] == []


@pytest.mark.parametrize("state", ["loading", "preparing"])
def test_device_checks_wait_for_unselected_model_preparation(tmp_path, monkeypatch, state):
    probes = []

    async def probe(self, profile):
        probes.append(profile)
        return {"available": True}

    monkeypatch.setattr(Runtime, "probe", probe)
    with client(tmp_path) as c:
        pool = c.app.state.runtime
        for key, profile, session_state in [("gpu", "CL", "ready"), ("npu", "ROCKCHIP", state)]:
            session = Runtime(pool.store, pool.backend)
            session.model, session.profile, session.state = {"id": key, "name": key}, profile, session_state
            pool.sessions[key] = session
        pool.selected = "gpu"
        response = c.post("/api/profiles/check")
        assert response.status_code == 409, response.text
        assert probes == []


def test_model_memory_estimate_uses_effective_execution_options(tmp_path, gguf, monkeypatch):
    estimates = []

    def estimate(model, context, realize, options, *, refresh=True):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Checkpoint verification must run outside the API event loop")
        assert refresh is False
        estimates.append((context, realize, dict(options)))
        return 12345

    monkeypatch.setattr("onpu.app.estimated_memory", estimate)
    with client(tmp_path) as c:
        model = c.post("/api/models/import", json={"path": str(gguf)}).json()
        response = c.get("/api/models").json()["models"][0]
        assert response["estimated_bytes"] == 12345
        assert estimates[-1][2]["profile"] == "CL"
        pool = c.app.state.runtime
        session = Runtime(pool.store, pool.backend)
        session.model, session.profile, session.state = model, "ROCKCHIP", "ready"
        session.options = {**pool.store.settings(), "profile": "ROCKCHIP", "context": 8192, "npu_prefill": "auto"}
        pool.sessions[model["id"]] = session
        response = c.get("/api/models").json()["models"][0]
        assert response["profile"] == "ROCKCHIP"
        assert estimates[-1][0:2] == (8192, True)
        assert estimates[-1][2]["npu_prefill"] == "auto"


def test_load_uses_the_models_declared_capacity_above_32768(tmp_path, gguf, monkeypatch):
    async def probe(self, profile):
        return {"available": True, "device": profile}
    loaded = []
    async def load(self, model, options):
        loaded.append(options["context"])
        self.model, self.options, self.profile, self.state = model, options, options["profile"], "ready"
        self.info = {"context": options["context"], "device": "CL"}
    monkeypatch.setattr(Runtime, "probe", probe)
    monkeypatch.setattr(Runtime, "load", load)
    backend = tmp_path / "missing-backend" / "tinygrad" / "llm"
    backend.mkdir(parents=True)
    (backend / "model.py").write_text("")
    with client(tmp_path) as c:
        model = c.post("/api/models/import", json={"path": str(gguf)}).json()
        model["context_limit"] = 40960
        c.app.state.store.put("models", model)
        response = c.post(f"/api/models/{model['id']}/load", json={"context": 40960})
        assert response.status_code == 202, response.text
        assert c.post(f"/api/models/{model['id']}/load", json={"context": 40961}).status_code == 400
        assert loaded == [40960]
        assert c.get("/api/status").json()["runtime"]["context"] == 40960


def test_gpu_is_default_and_npu_placement_is_validated(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/api/settings").json()["settings"]["profile"] == "CL"
        assert c.get("/api/status").json()["features"]["vision_gpu"] is True
        assert c.post("/api/models/missing/load", json={"npu_decode": "always"}).status_code == 422
        assert c.post("/api/models/missing/load", json={"vision_gpu": "invalid"}).status_code == 422


def test_selection_by_model_name_uses_the_loaded_session(tmp_path):
    with client(tmp_path) as c:
        pool = c.app.state.runtime
        session = Runtime(pool.store, pool.backend)
        session.model = {"id": "gpu-model", "name": "Friendly model name"}
        session.profile, session.state = "CL", "ready"
        pool.sessions[session.model["id"]] = session
        response = c.post("/api/models/Friendly%20model%20name/select")
        assert response.status_code == 200, response.text
        assert pool.selected == "gpu-model"
        assert response.json()["state"] == "ready"
        assert response.json()["model_id"] == "gpu-model"


def test_benchmark_keeps_its_model_when_chat_selection_changes(tmp_path, monkeypatch):
    called = []
    async def generate(self, body):
        called.append(body["model"])
        pool.selected = "npu"
        channel = asyncio.Queue()
        await channel.put({"event": "done", "metrics": {"model": body["model"]}})
        return "request", channel, {"event": "started"}
    monkeypatch.setattr(Runtime, "generate", generate)
    with client(tmp_path) as c:
        pool = c.app.state.runtime
        for key, profile in [("gpu", "CL"), ("npu", "ROCKCHIP")]:
            session = Runtime(pool.store, pool.backend)
            session.model, session.profile, session.state = {"id": key, "name": key}, profile, "ready"
            session.info = {"context": 512}
            pool.sessions[key] = session
        pool.selected = "gpu"
        response = c.post("/api/benchmarks", json={"max_tokens": 8, "repeats": 2})
        assert response.status_code == 202
        for _ in range(30):
            jobs = c.get("/api/benchmarks").json()["jobs"]
            if jobs[0]["state"] != "running":
                break
        assert jobs[0]["state"] == "complete", jobs
        assert called == ["gpu", "gpu"]
        assert [r["model"] for r in jobs[0]["results"]] == ["gpu", "gpu"]


def test_stop_scopes_request_after_the_same_model_has_started_a_new_reply(tmp_path, monkeypatch):
    commands = []

    async def send(self, command):
        commands.append((self.model["id"], command))

    monkeypatch.setattr(Runtime, "send", send)
    with client(tmp_path) as c:
        pool = c.app.state.runtime
        for key, profile in [("gpu", "CL"), ("npu", "ROCKCHIP")]:
            session = Runtime(pool.store, pool.backend)
            session.model, session.profile, session.state = {"id": key, "name": key}, profile, "generating"
            session.active = key + "-new"
            pool.sessions[key] = session
        pool.selected = "npu"
        assert c.post("/api/chat/stop", params={"model": "gpu", "request_id": "gpu-old"}).status_code == 200
        assert commands == []
        assert c.post("/api/chat/stop", params={"model": "gpu", "request_id": "gpu-new"}).status_code == 200
        assert commands == [("gpu", {"op": "cancel", "id": "gpu-new"})]
        assert pool.sessions["npu"].active == "npu-new"


def test_unsupported_sampling_is_rejected(tmp_path):
    with client(tmp_path) as c:
        payload = {"model": "test", "messages": [{"role": "user", "content": "Hello"}], "top_p": 0.9}
        assert c.post("/v1/chat/completions", json=payload).status_code == 422
        payload.pop("top_p")
        payload["max_tokens"] = 0
        assert c.post("/v1/chat/completions", json=payload).status_code == 422


def test_authentication_and_csrf(tmp_path):
    with client(tmp_path, base_url="http://192.168.1.50:8080", api_key="test-key") as c:
        assert c.get("/health").status_code == 200
        assert c.get("/api/status").status_code == 401
        assert c.post("/api/login", json={"key": "bad"}).status_code == 401
        result = c.post("/api/login", json={"key": "test-key"})
        assert result.status_code == 200
        assert "HttpOnly" in result.headers["set-cookie"]
        assert c.get("/api/status").status_code == 200
        assert c.post("/api/models/unload", headers={"Origin": "https://evil.example"}).status_code == 403
        assert c.post("/api/logout").status_code == 200
        assert c.get("/api/status").status_code == 401
        assert c.get("/api/status", headers={"Authorization": "Bearer test-key"}).status_code == 200


def test_lan_access_without_api_key(tmp_path):
    base_url = "http://192.168.1.50:8080"
    with client(tmp_path, base_url=base_url, api_key="") as c:
        assert c.get("/").status_code == 200
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/api/status").status_code == 200
        settings = c.get("/api/settings")
        assert settings.status_code == 200
        assert settings.json()["auth_enabled"] is False
        assert c.get("/v1/models").status_code == 200
        assert c.post("/api/models/unload", headers={"Origin": base_url}).status_code == 200
        assert c.post("/api/models/unload", headers={"Origin": "https://evil.example"}).status_code == 403
