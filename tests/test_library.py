import asyncio
from pathlib import Path

import httpx
import pytest

from onpu.library import Downloads, estimated_memory, inspect_gguf, register
from onpu.store import Store


def test_metadata_and_reference_import(gguf, tmp_path):
    store = Store(tmp_path / "data")
    try:
        model = register(store, gguf)
        assert model["name"] == "Test model"
        assert model["parameters"] == 4
        assert model["quantization"] == "F16"
        assert model["context_limit"] == 1024
        assert register(store, gguf)["id"] == model["id"]
        assert len(store.list("models")) == 1
        store.delete("models", model["id"])
        assert gguf.exists()
    finally:
        store.close()


@pytest.mark.parametrize("data", [b"", b"invalid", b"GGUF" + bytes(24), b"GGUF\x03\x00\x00\x00" + b"\xff" * 16])
def test_rejects_malformed_files(data, tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        inspect_gguf(path)


def test_rejects_metadata_without_weights(gguf):
    gguf.write_bytes(gguf.read_bytes()[:-8])
    with pytest.raises(ValueError, match="weight data"):
        inspect_gguf(gguf)


def test_memory_includes_fp32_kv():
    model = {"size": 1000, "parameters": 1000, "layers": 28, "kv_heads": 8, "head_dim": 128}
    assert estimated_memory(model, 1024, True) - estimated_memory(model, 512, True) == 2 * 28 * 8 * 128 * 512 * 4


def test_partial_download_is_resumed_from_same_file(gguf, tmp_path, monkeypatch):
    """Resume survives resolving a branch to an immutable revision."""
    async def run():
        store = Store(tmp_path / "data")
        payload = gguf.read_bytes()
        partial = store.root / "models/job-test.gguf.part"
        partial.write_bytes(payload[:64])
        calls = []
        def handler(request):
            calls.append(request)
            if "/api/models/" in str(request.url):
                return httpx.Response(200, json={"sha": "immutable"})
            assert request.headers["range"] == "bytes=64-"
            return httpx.Response(206, content=payload[64:], headers={"content-range": f"bytes 64-{len(payload)-1}/{len(payload)}"})
        real_client = httpx.AsyncClient
        monkeypatch.setattr("onpu.library.httpx.AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
        downloads = Downloads(store)
        try:
            job = downloads.start({"repo": "test/model", "file": "test.gguf", "revision": "immutable"}, "job")
            await downloads.tasks[job["id"]]
            result = store.get("downloads", "job")
            assert result["state"] == "complete", result
            assert result["received"] == len(payload)
            assert (store.root / "models/job-test.gguf").read_bytes() == payload
            assert len(result["sha256"]) == 64
            assert len(store.list("models")) == 1
        finally:
            await downloads.close()
            store.close()
    asyncio.run(run())


def test_rejects_path_traversal_download(tmp_path):
    store = Store(tmp_path / "data")
    try:
        downloads = Downloads(store)
        with pytest.raises(ValueError):
            downloads.start({"repo": "test/model", "file": "../escape.gguf"})
    finally:
        store.close()
