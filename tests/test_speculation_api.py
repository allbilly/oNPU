import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from onpu.app import LoadOptions, create_app
from onpu.runtime import Runtime


def test_load_schema_requires_explicit_experimental_mode():
    assert LoadOptions().speculation == "off"
    assert LoadOptions().draft_path is None
    for options in ({"speculation": "auto"}, {"speculation": "eagle4"}, {"draft_path": ""}):
        with pytest.raises(ValidationError):
            LoadOptions(**options)


def test_load_api_passes_explicit_mode_and_local_path_to_model_owner(tmp_path, gguf, monkeypatch):
    async def backend_info(path):
        return {"available": True, "path": str(path)}

    async def probe(self, profile):
        return {"available": True, "device": profile}

    loaded = []

    async def load(self, model, options):
        loaded.append(dict(options))
        self.model, self.options, self.profile, self.state = model, options, options["profile"], "ready"
        self.info = {"context": options["context"], "device": "CL", "speculation": {"mode": options["speculation"], "greedy_only": True}}

    monkeypatch.setattr("onpu.app.backend_info", backend_info)
    monkeypatch.setattr(Runtime, "probe", probe)
    monkeypatch.setattr(Runtime, "load", load)
    backend = tmp_path / "backend"
    (backend / "tinygrad/llm").mkdir(parents=True)
    (backend / "tinygrad/llm/model.py").write_text("")
    with TestClient(create_app(backend, tmp_path / "data")) as client:
        model = client.post("/api/models/import", json={"path": str(gguf)}).json()
        options = {"profile": "CL", "context": 512, "speculation": "dflash2-experimental", "draft_path": str(tmp_path / "draft")}
        response = client.post(f"/api/models/{model['id']}/load", json=options)
        assert response.status_code == 202, response.text
        assert loaded[0]["speculation"] == options["speculation"]
        assert loaded[0]["draft_path"] == options["draft_path"]
        assert response.json()["speculation"] == {"mode": "dflash2-experimental", "greedy_only": True}
        assert client.get("/api/status").json()["runtime"]["options"]["speculation"] == "dflash2-experimental"
