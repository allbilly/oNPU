# oNPU verification

**Historical follow-up (before the October 25 checkpoint):** The exact Qwen Q8_0 checkpoint now enables combined GPU projections and packed Q8 output-head reads with assistance Off. Six measured trials per variant give median prefill88.38→94.18 tok/s and decode6.73→7.88 tok/s versus the prior GPU path. Prefill ranges overlap; decode ranges separate. All256 complete held-out logit dumps and full-model synthetic32K/40K cache-boundary checks are bitwise equal. This is a GPU improvement, not an RKLLM win.

The historical serving check rejected overlapping model preparation with HTTP409 before creating a second worker. NPU hardware detection reads the parent driver correctly. Both original40,960 GPU/8,192 NPU sessions, GPU selection and saved settings are restored. Live tests pass cold/cached equivalence, both overflow forms preserving21 cached positions, and a0.136-second NPU reply during GPU prefill followed by Stop/recovery. There are58 passing focused tests,241 mypy source files and passing lint/whitespace checks. Full-model level0 references remain incomplete: the first SmolVLM export hit a read-only SDK workspace; the corrected exporter passes a real synthetic SDK export, and a retry is queued behind a3GiB memory threshold without changing API sessions. Qwen conversion was stopped to recover serving memory. Current evidence: `~/tinygrad/docs/rockchip-rkllm-parity-results.json`.

**2026-10-21 reference correction:** Rockchip's current published benchmark requires W8A8 models exported with `optimization_level=0`, 128 input positions, 64 outputs and maximum CPU/NPU clocks. Earlier local references below use a different recipe or lack complete recorded build settings; their measurements remain local historical comparisons. They do not establish wins against the published speed-priority recipe. Fresh level0 exports and repeated quality-matched comparisons are in progress. The 21-row coverage matrix is `~/tinygrad/docs/rockchip-rkllm-coverage.json`; unmeasured entries remain unqualified. Qwen's model card documents a 32,768-token window; the 40,960-token tests below exercise configured capacity.

**2026-10-21 context update:** Current workers pass capacity, cache and recovery checks at their declared limits: Qwen3-0.6B at 40,960 on OpenCL and SmolVLM at 8,192 on CPU/NPU. Both remain ready with GPU selected and saved settings unchanged. These API checks do not establish reliable answer quality at the model maximum.

A live regression found that rejecting an oversized prompt cleared valid KV history before inference started. The worker now preserves that history for validation errors; errors during inference still invalidate it. The current GPU matrix passes all eight checks, including 40,959 input + one output token, identical cached replies, both overflow forms and 40,958 cached positions retained afterward. The current NPU matrix passes all 11 capacity/cache/Stop/disconnect/recovery checks. The initial short GPU regression also preserves 544 of 545 cached positions after both rejected forms.

| Current Qwen 40K API measurement | Result |
|---|---:|
| Cold prompt / prefill | 40,704 tokens / 12.20 tok/s |
| Cold first text | 55.6 minutes |
| Identical cached first text | 1.58 seconds |
| Growing follow-up first text | 21.04 / 16.05 seconds |
| Cached nonstream full reply | 5.58 seconds |
| Stop / disconnect recovery | 0.272 / 0.394 seconds |
| Independent NPU reply during GPU prefill | 0.118 seconds |
| Status latency, p95 / maximum | 40.91 / 80.66 ms |

The fixed three-position retrieval probe scores **1/3 for each model**: beginning and middle fail, end succeeds. Qwen receives a 32-token reply budget, so these are untruncated answers. RKLLM also fails the identical VLM beginning/middle inputs, with different wording. No matched 40K Qwen reference is available, so its failures are not attributed to either the checkpoint or the backend. Cold 40K use remains slow, and general maximum-context answer quality is unqualified.

Qwen's GPU-only path passes 256 held-out late-16K positions at perplexity ratio 1.0020264066 versus RKLLM W8A8 (unchanged gate: 1.02). An exact-arithmetic workgroup-order change passes six tests, including independent NumPy checks near 40K and byte-identical output against the prior dispatch. All 256 full-vocabulary logit files match the preceding qualified GPU path byte for byte. Prepared weights remain FP16 from the Q8_0 GGUF; the two engines use different quantization formats.

| Qwen 16K cold workload | oNPU prefill tok/s | RKLLM prefill tok/s | oNPU decode tok/s | RKLLM decode tok/s |
|---|---:|---:|---:|---:|
| Repeated records | 23.19 | 19.79 | 3.81 | 0.57 |
| WikiText | 22.38 | 20.00 | 3.95 | 0.69 |

These are **single measured pilots per engine/workload**, with three 64-token warmups, 16,376 identical input IDs and eight raw output IDs. GPU and SDK run exclusively; SDK workload order is reversed. Observed prefill gains are 17.2%/11.9%; decode gains are 6.69x/5.70x. They are not repeated medians or a universal speed guarantee. SmolVLM's three-run 8K comparison and four limited quality gates are detailed below.

The installed RKLLM artifact/toolkit stops at 16,384, so a matched **40,960-token RKLLM speed/quality comparison is unavailable**. The full model-max usability target remains unqualified. The 40K timing above is an API diagnostic with both workers resident; different prompts and thermal histories prevent attribution of changes from earlier runs to a single optimization.

Twenty-five focused app tests and 31 backend/Mali tests pass with `-n12`; mypy checks 241 source files, and backend/app lint and whitespace checks pass. All measured numerical source files remain unchanged. Reversing only the new validation guard reconstructs the qualified worker hash exactly. Both workers have zero swap use; a two-second idle sample records 0% CPU across 20 model processes. Current evidence: `/home/orangepi/vlm-bench/qwen-final-assessment.json` and `tinygrad/docs/rockchip-long-context-benchmark.json`.

The earlier FP32 packed-attention oNPU API measures median124.07 prefill and18.41 visible decode tokens/s versus the old RKLLM160.17/17.68 baseline. It does not meet the prefill target. Its trials decline alongside rising temperatures and reduced clocks. The old SDK pixel fixture also differs from the API's CPU tensor normalization by one FP32 ULP; the new comparison below fixes that input mismatch. No clock or thermal policy is changed.

The qualified long-context NPU path now uses per-row INT8 query/key score dots with FP32 softmax/value accumulation and ordinary FP16 KV. Native profiling attributed53 of63 seconds to CPU attention. The new primitive, including incremental packing, measures37.19ms versus54.17ms. All25 backend runtime/placement tests and25 app regressions pass; mypy checks241 source files and whole-backend Ruff passes. Tests include a quantized reference, the final physical cache position, causal NaN masking, divergence/reset/gaps, actual pool dispatch and bit-identical last-row output across query counts and pooled decode.

Fresh matched API/RKLLM tests use the same checkpoint, bit-identical API-normalized pixels, exactly8184 prompt positions plus8 generated tokens, three128-position warmups andthree cold measurements per workload. API trials require zero KV reuse, uncached vision andlength termination; SDK trials verify full prefill and8191 retained KV positions. Caption-style prompts avoid unequal early-EOS output counts. Both engines run exclusively, with a60-second quiet period before their measured sequence; workload order is reversed for SDK. Starting/ongoing clocks and temperatures are recorded.

| 8K workload | oNPU prefill tok/s | RKLLM prefill tok/s | oNPU decode tok/s | RKLLM decode tok/s |
|---|---:|---:|---:|---:|
| Repeated records | 173.82 | 157.77 | 31.72 | 17.11 |
| WikiText | 165.80 | 158.36 | 30.50 | 16.46 |

These three-run medians are10.2%/4.7% faster in prefill andabout1.85x faster in decode. First text/token medians are48.53/52.71 seconds (records) and50.79/52.52 seconds (WikiText); API first text andSDK first raw token remain slightly different timing boundaries. Every measured oNPU prefill rate exceeds every SDK rate on its matching workload. The API sequence runs both workloads back to back and includes thermal slowdown; these measurements are specific to this board and inputs.

All four unchanged quality gates pass before activation:256 late-context positions at perplexity ratio0.999586 (CPU vision) /1.006175 (GPU vision);20 image cases at0.993698 /1.007301, with9/7 correct versus SDK7. These are limited quality proxies. Both engines describe the astronaut image as a song under the long WikiText distraction, so faster inference does not resolve the checkpoint's weak long-context image grounding.

The isolated API passes10 capacity/cache/overflow/recovery checks. The normal API then loads the new path by default, completes an actual8192-total-position request andan identical cached repeat, andserves NPU HELLO in0.106s during Qwen40960 GPU prefill. GPU Stop recovery is0.706s. Both models finish ready with GPU selected andsaved settings unchanged; a two-second idle sample records0% CPU across both workers andtheir descendant processes. Raw reports: /home/orangepi/vlm-bench/continue-preview-*-timing.json, continue-sdk-*-timing.json, continue-main-verification.json. Qwen 16K now has passing quality and two matched speed pilots; full model-max qualification remains limited as described above.

The experimental FP16 QK dot passes the numerical gates:256 late-context positions at perplexity ratio1.00282 (CPU vision) /0.99496 (GPU vision), and20 image cases at1.01304 /0.99238 with8/7 correct versus SDK7. It measures median131.42 prefill tokens/s in three uncached API trials, still below RKLLM. **It is disabled in serving:** a final audit finds different greedy answers for identical cold/cached boundary input (Secret word:44 versus13). The14 transport checks and actual8192-token usage pass but do not qualify this candidate. API visible output tokens and SDK raw generated IDs differ, so the experimental decode figures also do not establish a matched win.

Before the INT8 score-dot qualification, the FP32 rollback passed10 live API checks with0 API failures and0 health errors. It established cache equality, actual8192-total usage andindependent GPU/NPU recovery, while still missing the speed target. Its wrong O retrieval answer and3/5 correctness are retained as limitations. Historical raw reports remain under /home/orangepi/vlm-bench/fix-context-final-fp32-*; they are superseded for current serving by the continue-* reports above.

The [backend progress record](../tinygrad/docs/rockchip-long-context-progress.md)
tracks the current fixes and remaining checks. The 2026-10-18 comparison below
records its original source and conditions; it is not a timing of the new Mali
kernels or proof of long-context performance.

## 2026-10-18 short-context comparison

The selective inference port is verified on the current tinygrad `oNPU` branch.
OpenCL remains the default. CPU/NPU serving uses measured per-phase assistance,
adaptive prefill chunks and guarded native replay. A GPU model and a different
small NPU model serve real requests concurrently.

| Matched Qwen3-0.6B FP16 input | tinygrad prefill | RKLLM prefill | tinygrad decode | RKLLM decode |
| ---: | ---: | ---: | ---: | ---: |
| 128 tokens | 280.353 ms | 326.310 ms | 18.7022 tokens/s | 16.6705 tokens/s |
| 256 tokens | 631.830 ms | 655.672 ms | 17.0492 tokens/s | 16.5525 tokens/s |

The final comparison uses immutable current-source copies, four complete warmups,
three measurements per fresh process, bracketing/reversed engine order, identical
raw IDs and 32 greedy output IDs. Every generated sequence agrees. Source hashes
and clock samples match the recorded conditions. Current-source full-vocabulary
logit checks pass 128/256 prefill and a stateful decode step; the SDK EOS mask is
recorded separately. This is matched FP16 evidence, not W8A8 or universal coverage.

- Eighteen backend tests, 24 generation/parser/tokenizer checks and 23 app unit
  tests pass. Backend mypy (219 files), Ruff, changed-app Python lint, JavaScript
  syntax and whitespace checks pass. Optional SDK fixtures cannot start because
  the `openai` client package is absent; no dependency or test was weakened.
- Fresh GPU and NPU HTTP generation, SSE, limits, context/busy errors, Stop,
  disconnect and recovery pass. Short/long prompt switching works in greedy and
  sampled modes. Both workers report `/home/orangepi/tinygrad`.
- Real concurrent GPU/NPU requests overlap, return nonempty text and record NPU
  submissions. GPU-only sessions use CL with assistance Off.
- Chromium passes real chat/history, Stop, all navigation, import, per-model
  unload/reload, settings, themes and desktop/mobile layouts without console
  errors. The corrected Rockchip form exposes prefill Auto/Off and performs an
  actual Auto load while preserving the GPU session.
- Both final sessions are left ready at <http://127.0.0.1:8091>. Warming four
  prompt sizes and both sampling modes takes about 150 seconds on cold NPU load.
- Runtime code has no reference-worktree dependency. Both reference folders stay
  intact, including their uncommitted work. No commits, pushes, driver changes or
  kernel-module insertions were performed.

The summary is [verification-20261018.json](docs/verification-20261018.json).
[Exported evidence](docs/rk3588-verification-20261018/completion-audit.json) includes
raw logs, compressed benchmark/quality reports, screenshots, source checksums and
a reviewable source archive. Earlier studies below are historical and should not
be mixed with this matched FP16 comparison.

---

# Historical placement study · 2026-10-18

The app now uses `/home/orangepi/tinygrad`, branch `oNPU`, for CPU, OpenCL and
optional NPU-assisted inference. OpenCL is the default profile. The earlier
isolated `tinygrad-opencl` setup described below is historical; it is no longer
required by the current application.

## Current integration

Raw HTTP, concurrent-session, browser-control, hardware and source-hash evidence
is retained in [verification-20261018.json](docs/verification-20261018.json).

- Qwen3-0.6B Q8 GGUF, prepared into FP16 weights, serves on Mali OpenCL.
- SmolLM2-135M F16 serves with dense decode projections on the Rockchip NPU and
  surrounding operations on CPU. Both models remain loaded independently and
  real requests were observed active concurrently.
- Each worker reports `/home/orangepi/tinygrad` as its imported backend. No
  retained execution path reads `tinygrad-1500` or `tinygrad-1500-clean`.
- GPU and NPU HTTP checks pass real non-streaming generation, SSE, usage counts,
  one-token limits, context rejection, per-model request admission, Stop,
  disconnect cancellation, and generation after cancellation.
- Chromium checks pass model selection, chat and persisted history, Stop,
  per-model unload/reload while preserving the other session, saved settings,
  themes, and desktop/mobile navigation without console errors.
- The app unit suite passes **22 tests** with `-n12`. Tinygrad's tokenizer/native
  tests and existing custom-kernel/JIT regressions pass; mypy and Ruff pass in
  the backend. Changed app Python files pass Ruff with the backend configuration.

## Observed placement comparison

The same Qwen3 model at context 512 ran three deterministic 16-token chat
requests per mode, GPU-only / auto GPU+NPU / GPU-only. Median decode rates were
**1.644 / 1.816 / 1.766 tokens/s**; median first-text latencies were
**4730 / 4619 / 4629 ms**. All three responses matched exactly across modes.
These are modest diagnostic gains on this board, with drift between the two GPU
controls. They do not establish a general speed or model-quality guarantee.

Auto compares numerical outputs and measures each candidate including transfers.
In this run it selected 141 decode layers and the final vocabulary projection
for prefill. Bulk prefill stayed on the GPU. Both assistance controls default to
Off. The detailed retained implementation, limits and raw benchmark are in
`~/tinygrad/docs/rockchip-onpu.md` and `rockchip-onpu-benchmark.json`.

The SmolLM2 eight-token greedy response matched the CPU reference. Its observed
NPU-assisted decode sample was 6.26 tokens/s, versus 4.29 tokens/s for CPU; these
were diagnostic samples, not a paired throughput study.

The smaller model was downloaded through the application's existing downloader:
`bartowski/SmolLM2-135M-Instruct-GGUF`, revision
`09816acd5d99df7be770d85ea30822623dab342c`, file
`SmolLM2-135M-Instruct-f16.gguf`, 270,885,952 bytes, locally computed SHA-256
`f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57`.
The file and its registered model remain in `.data`.

NPU support is the bounded finite-FP16 dense matrix path, accumulating in FP32.
It does not qualify arbitrary NPU operators, all models, larger contexts or native
GGUF quantized arithmetic. GPU assistance occupies the same NPU resource as a
separate NPU-assisted model; GPU-only serving can coexist with that second model.
No kernel modules were inserted. The two reference worktrees remain intact.

---

# oNPU verification · 2026-10-11

Qwen3-0.6B model serving now passes on Mali OpenCL through the isolated `~/tinygrad-opencl` checkout with a local queue workaround. The default 1024-token context, FP16 weight preparation, real generation, streaming and cancellation were exercised on 2026-10-11. CPU checks through the user's existing `~/tinygrad` checkout passed on 2026-10-10. NPU assistance remains disabled and unqualified.

## Environment actually tested

| Item | Observed value |
| --- | --- |
| Board | Orange Pi 5, RK3588S, aarch64 |
| Memory | 16,739,618,816 bytes reported by Linux |
| Kernel | `6.1.99-rockchip-rk3588` |
| GPU backend | `/home/orangepi/tinygrad-opencl` |
| GPU backend HEAD | `8a4e8c237b9b744a8bddf213d1a5b874d0e531a0` |
| GPU backend Python | oNPU virtual environment, Python 3.12; no backend virtual environment |
| GPU backend worktree | Detached upstream master plus a local seven-line `ops_cl.py` patch |
| App | Python 3.12, locked FastAPI/Uvicorn dependencies, buildless HTML/CSS/JS |
| Mali device | `Mali-LODX r0p0`, OpenCL driver version `2.1` |

The latest upstream master fetched on 2026-10-11 was still `8a4e8c2`, committed on 2026-10-10. The original CPU validation used `/home/orangepi/tinygrad` at `805a8d5ec710e872cdd8866e9d60f2bbb444a461`; its existing NPU work remains in that checkout. The GPU patch is also retained in [patches/tinygrad-mali-opencl.patch](patches/tinygrad-mali-opencl.patch), and `git apply --check` succeeded against a fresh worktree at the tested master commit.

## GPU repair checks · 2026-10-11

- An ordinary `./run.sh --backend ~/tinygrad-opencl` server used oNPU's own Python environment. A load request containing only `{"profile":"CL"}` reached `ready` in **29.37 seconds**, with actual device `CL`, context **1024** and `realize=true`. No diagnostic monkey patches, altered JIT flags or per-kernel waits were used.
- `scripts/smoke.py --profile CL --context 1024` passed real non-streaming and SSE generation, usage accounting, normal EOS, a one-token cap, HTTP 400 context errors, HTTP 409 overlapping-request rejection, Stop, disconnect cancellation and generation after cancellation. The deterministic short prompt returned `Hello!`. The verifier now checks the actual execution device and the requested context.
- The worker was unloaded, zero worker RSS was observed, and a new default-options GPU load reached readiness in **28.81 seconds**. A new chat request after that reload succeeded.
- Kernel-log snapshots before and after the default load, chat checks and reload showed **zero new `CS_CONFIG_FAULT` reports** (10 historical reports before and after). The driver still emitted two GPU activity timing warnings; the serving checks completed successfully.
- Twelve tinygrad tests passed on the actual CL device: compile caching, invalid-kernel errors, unaligned buffer copies, FP16 and small matrix products against reference results, JIT input copies, JIT assignment and JIT cloning. Tests ran with `-n12` against the isolated patched checkout.
- `python -m mypy tinygrad/` passed for **213 source files**, and `python -m ruff check .` passed in the isolated backend. The oNPU suite was rerun: **17 passed**.

## Earlier application checks · 2026-10-10

- `python -m pytest -q`: **17 passed**. Coverage includes malformed and truncated GGUF metadata, model references, memory estimates, immutable-revision download resumption, traversal rejection, persisted settings, invalid generation settings, authentication, cross-origin protection, single-request admission, context errors and recoverable startup timeout. The suite uses fixtures and mock HTTP where appropriate; these are not substitutes for real inference.
- `node --check onpu/static/app.js`, `bash -n run.sh`, and Python compilation passed.
- `uv sync --extra test` installed the local package successfully. `uv build --wheel` produced `dist/onpu-0.1.0-py3-none-any.whl`. The wheel was compared byte-for-byte with all 13 current application files and includes its UI assets, third-party notices and service template.
- Both CPU and OpenCL executed a real tinygrad tensor operation yielding `[2.0, 3.0, 4.0]`. The OpenCL device name was obtained from the actual initialized device.
- `scripts/smoke.py` passed on CPU with an existing matched FP16 Qwen3-0.6B artifact and with the freshly downloaded Q8 artifact. It exercised real non-streaming and SSE generation, usage accounting, normal EOS, a one-token cap, HTTP 400 context errors, HTTP 409 overlapping-request rejection, Stop and disconnect cancellation. The deterministic test prompt returned `Hello!`.
- The official `Qwen/Qwen3-0.6B-GGUF` Q8 file was downloaded, paused after approximately 229 MB, resumed, checked for complete length and valid GGUF weight data, hashed, and registered in the library. The pinned revision was `23749fefcc72300e3a2ad315e1317431b06b590a`; file length was **639,446,688 bytes** and local SHA-256 was `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`. This is a computed local hash, not a claimed publisher checksum comparison.
- A real CPU benchmark completed and persisted its request measurements, profile, context, backend revision, kernel and settings. Results are explicitly marked performance-only.
- `scripts/browser_check.py` passed against real inference in Chromium. It clicked all navigation items; saved settings and checked them after reload; sent streaming chat and checked persisted history; used the Stop button; imported a model; unloaded it and observed zero worker RSS; loaded it through the actual form; and checked desktop/mobile layouts and themes without console errors. The last CSS adjustment was separately checked on all five mobile views, including hidden sidebar text and horizontal overflow.

Screenshots of the real app are in `docs/screenshots/`: `overview.png`, `chat.png`, `dark.png` and `mobile.png`.

## GPU workaround and observed performance

The CPU short-request samples were around **0.8 decode tokens/s**, with approximately 7.7 seconds to first emitted text on a fresh short prefix. These are diagnostic observations on the current shared board, not a controlled performance comparison or a quality certification. The Q8 model's worker RSS was around 2 GB after preparation. The application makes these costs visible; it does not imply tinygrad currently achieves the native qwen3.c engine's speeds.

Before the repair, the FP16 model with context 512 and Q8 model with context 128 stalled during warmup; a process trace placed the wait inside OpenCL buffer copy. The kernel logged `CS_CONFIG_FAULT` (`0x40`) for the tested Mali contexts. A bounded alternate run with `REALIZE=0`, `JIT=0`, and `PARALLEL=0` also failed to establish serving readiness and produced a corresponding GPU driver fault.

Unmodified latest master reproduced the fault: weights loaded, but the process segfaulted inside `clEnqueueReadBuffer` during warmup. Finishing the queue before that read also crashed. Diagnostic runs with waits after every kernel and after batches of 8, 16 and 32 kernels completed warmup. Deferring buffer releases did not fix the failure.

The local workaround drains the OpenCL queue after each batch of 32 kernel dispatches, only for the observed `Mali-LODX r0p0` / driver `2.1` pair. It resets the dispatch count on ordinary synchronization. This avoids the observed native driver crash while preserving asynchronous execution within each batch. The exact defect inside the closed driver has not been established; 32 is a tested batch size, not a proven hardware limit.

The fresh GPU short-request sample took **5.57 seconds** to first emitted text and measured approximately **1.78 decode tokens/s**. Prepared model allocation was approximately **1.58 GB**, and worker RSS after warmup was approximately **2.66 GB**. These are diagnostic measurements on the shared board, not a controlled CPU/GPU comparison. The Q8 GGUF was prepared into FP16 GPU weights, so this does not claim native Q8 arithmetic.

OpenCL remains labelled **experimental** for general hardware/model coverage. The profile's **Tensor OK** indicator still describes its startup probe, while the separate real-model checks above establish the tested model's serving behavior. Startup has a five-minute bound, and cancellation has a separate 45-second bound. No system drivers were changed, and no NPU submissions were made by this session.

## Limits

CPU serving was exercised with two Qwen3 artifacts at context 512. GPU serving was exercised with Qwen3-0.6B Q8 GGUF and FP16 weight preparation at contexts 128 and 1024. Long-running stress tests, full-model numerical logit comparisons, other architectures, larger contexts, on-the-fly GPU dequantization (`realize=false`) and hybrid NPU placement remain unverified. The optional systemd unit is supplied for review and installation; it was not enabled during QA. The preview server is stopped after verification, and the saved library/download data remains in `.data` for the documented launch command.

Owned temporary research snapshots, diagnostic logs, duplicate build output and initial QA screenshots were removed. The intended wheel, final screenshots, source, environment, saved model library and persistent GPU backend worktree were retained.

## Fixes checked during implementation

The worker passes a copy of prompt IDs to tinygrad's streaming handler so that generated tokens do not inflate recorded prompt counts. The browser click delegate preserves native navigation and form submission; only explicit form/keyboard handlers prevent defaults. Disconnect cleanup is shielded from HTTP task cancellation. The model worker runs in its own process group so unload can also terminate compiler children.

INTENT: model warmup crashes with an unbounded Mali OpenCL queue; the task expects usable GPU chat; tinygrad’s runtime documentation specifies CL acceleration on OpenCL 2.0 compatible devices.

TWINS: searched clEnqueueNDRangeKernel/clEnqueueReadBuffer - found 0 other implementation sites: none. The only additional matches were autogenerated API declarations.
## Checkpoint validation — October 25, 2026

The cleanup checkpoint records the exact backend SHA in `BACKEND_REVISION`.
The scoped backend regression set passed 328 methods and 136 subtests, with two
existing expected failures. The app suite passed 321 methods and 30 subtests.
Ruff passed in both repositories, and mypy passed over 264 backend source files.

The default LLVM host fails the existing NULL reduction test with a typed-pointer
IR error on both the committed baseline and the working tree. The NULL suite
passes with the supported `HCQ_RUNTIME_DEV=CPU:CLANG` host. Runtime defaults were
left unchanged. The GGUF memory test now collects unreachable tensors around its
counter measurement; its resident-weight lower bound and scratch upper bound
remain unchanged.

These checks save the current experimental state. They do not establish broad
model quality, RKLLM speed parity or model-maximum context support. See
[the validation record](docs/checkpoint-validation-20261025.json) for scope and
the existing [benchmark](benchmark.md) for measured and pending work.
