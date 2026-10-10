# oNPU verification · 2026-10-10

The application works on CPU through the user's existing `~/tinygrad` checkout. OpenCL device access works, but full-model GPU generation failed on this board image. NPU assistance remains disabled and unqualified.

## Environment actually tested

| Item | Observed value |
| --- | --- |
| Board | Orange Pi 5, RK3588S, aarch64 |
| Memory | 16,739,618,816 bytes reported by Linux |
| Kernel | `6.1.99-rockchip-rk3588` |
| Backend | `/home/orangepi/tinygrad` |
| Backend HEAD | `805a8d5ec710e872cdd8866e9d60f2bbb444a461` |
| Backend Python | Existing tinygrad virtual environment, Python 3.12.12 |
| Backend worktree | Existing uncommitted changes; oNPU did not edit or reset it |
| App | Python 3.12, locked FastAPI/Uvicorn dependencies, buildless HTML/CSS/JS |
| Mali device | `Mali-LODX r0p0`, OpenCL driver version `2.1` |

The checkout differs from the GitHub default branch revision `078eee94d8f1912bb63ce786c250d2c24984844a`. Tests deliberately used the user's local checkout as requested.

## Checks that passed

- `python -m pytest -q`: **17 passed**. Coverage includes malformed and truncated GGUF metadata, model references, memory estimates, immutable-revision download resumption, traversal rejection, persisted settings, invalid generation settings, authentication, cross-origin protection, single-request admission, context errors and recoverable startup timeout. The suite uses fixtures and mock HTTP where appropriate; these are not substitutes for real inference.
- `node --check onpu/static/app.js`, `bash -n run.sh`, and Python compilation passed.
- `uv sync --extra test` installed the local package successfully. `uv build --wheel` produced `dist/onpu-0.1.0-py3-none-any.whl`. The wheel was compared byte-for-byte with all 13 current application files and includes its UI assets, third-party notices and service template.
- Both CPU and OpenCL executed a real tinygrad tensor operation yielding `[2.0, 3.0, 4.0]`. The OpenCL device name was obtained from the actual initialized device.
- `scripts/smoke.py` passed on CPU with an existing matched FP16 Qwen3-0.6B artifact and with the freshly downloaded Q8 artifact. It exercised real non-streaming and SSE generation, usage accounting, normal EOS, a one-token cap, HTTP 400 context errors, HTTP 409 overlapping-request rejection, Stop and disconnect cancellation. The deterministic test prompt returned `Hello!`.
- The official `Qwen/Qwen3-0.6B-GGUF` Q8 file was downloaded, paused after approximately 229 MB, resumed, checked for complete length and valid GGUF weight data, hashed, and registered in the library. The pinned revision was `23749fefcc72300e3a2ad315e1317431b06b590a`; file length was **639,446,688 bytes** and local SHA-256 was `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`. This is a computed local hash, not a claimed publisher checksum comparison.
- A real CPU benchmark completed and persisted its request measurements, profile, context, backend revision, kernel and settings. Results are explicitly marked performance-only.
- `scripts/browser_check.py` passed against real inference in Chromium. It clicked all navigation items; saved settings and checked them after reload; sent streaming chat and checked persisted history; used the Stop button; imported a model; unloaded it and observed zero worker RSS; loaded it through the actual form; and checked desktop/mobile layouts and themes without console errors. The last CSS adjustment was separately checked on all five mobile views, including hidden sidebar text and horizontal overflow.

Screenshots of the real app are in `docs/screenshots/`: `overview.png`, `chat.png`, `dark.png` and `mobile.png`.

## Observed performance and GPU failure

The CPU short-request samples were around **0.8 decode tokens/s**, with approximately 7.7 seconds to first emitted text on a fresh short prefix. These are diagnostic observations on the current shared board, not a controlled performance comparison or a quality certification. The Q8 model's worker RSS was around 2 GB after preparation. The application makes these costs visible; it does not imply tinygrad currently achieves the native qwen3.c engine's speeds.

Full-model Mali testing failed. The FP16 model with context 512 and Q8 model with context 128 stalled during warmup; a process trace placed the wait inside OpenCL buffer copy. The kernel logged `CS_CONFIG_FAULT` (`0x40`) for the tested Mali contexts. A bounded alternate run with `REALIZE=0`, `JIT=0`, and `PARALLEL=0` also failed to establish serving readiness and produced a corresponding GPU driver fault.

The app stayed responsive and could unload the failed GPU workers and return to CPU. OpenCL remains selectable as **experimental**, with **Tensor OK** indicating the scope of the successful check. Startup has a five-minute bound, and cancellation has a separate 45-second bound; the startup watchdog's recovery behavior is covered by the automated suite.

GPU repair was not attempted in the user's dirty tinygrad checkout or in system drivers. The source and driver fault require a separate investigation before GPU model serving can be qualified. No NPU submissions or acceleration claims are made by this implementation.

## Limits

Only the two Qwen3 artifacts and bounded context-512 CPU serving were exercised end to end. Long-running stress tests, numerical logit comparisons, other architectures, larger contexts, quantized GPU qualification and hybrid NPU placement remain unverified. The optional systemd unit is supplied for review and installation; it was not enabled during QA. The preview server is stopped after verification, and the saved library/download data remains in `.data` for the documented launch command.

Owned temporary research snapshots, diagnostic logs, duplicate build output and initial QA screenshots were removed. The intended wheel, final screenshots, source, environment and saved model library were retained.

## Fixes checked during implementation

The worker passes a copy of prompt IDs to tinygrad's streaming handler so that generated tokens do not inflate recorded prompt counts. The browser click delegate preserves native navigation and form submission; only explicit form/keyboard handlers prevent defaults. Disconnect cleanup is shielded from HTTP task cancellation. The model worker runs in its own process group so unload can also terminate compiler children.

INTENT: oNPU is empty; the task expects a working local model appliance; tinygrad’s CLI documents GGUF loading, CPU/OpenCL execution, and streaming chat as the integration points.

TWINS: searched `len(ids)`/`run_model(ids)` and document click handlers using `preventDefault()` - found 0 other incorrect sites: none.
