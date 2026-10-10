from pathlib import Path

from fastapi.testclient import TestClient

from onpu.app import create_app


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
        assert c.post(f"/api/models/{key}/load", json={"profile": "ROCKCHIP"}).status_code == 422
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
