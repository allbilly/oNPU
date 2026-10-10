# oNPU

oNPU is a local web app for managing and chatting with GGUF language models on Rockchip boards. It uses an existing [tinygrad](https://github.com/tinygrad/tinygrad) checkout as its inference backend. CPU inference is supported; GPU and NPU acceleration are experimental.

## Run

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a working tinygrad checkout.

```bash
cd ~/oNPU
uv sync --python ~/tinygrad/.venv/bin/python
./run.sh --backend ~/tinygrad
```

Open <http://127.0.0.1:8080>. Import a local GGUF model or download one from the app. Project data is stored in `.data` by default; use `--data-dir` to choose another location.

The API provides OpenAI-compatible chat completions at `/v1/chat/completions`. See <http://127.0.0.1:8080/docs> while the app is running for the API reference.

## Development

```bash
uv sync --extra test --python ~/tinygrad/.venv/bin/python
.venv/bin/python -m pytest -q
```

See [VERIFICATION.md](VERIFICATION.md) for current device and backend limitations, and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for attribution.
