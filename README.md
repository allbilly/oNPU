# oNPU

oNPU is a local web app for managing and chatting with GGUF language models and selected vision-language models on Rockchip boards. It uses an existing [tinygrad](https://github.com/tinygrad/tinygrad) checkout as its inference backend. Mali OpenCL is the default execution profile. Optional NPU assistance is measured per inference phase; CPU inference remains available.

## Run

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a working tinygrad checkout.

```bash
cd ~/oNPU
uv sync --python ~/tinygrad/.venv/bin/python
./run.sh --backend ~/tinygrad
```

Open <http://127.0.0.1:8080>. Import a local GGUF model or download one from the app. Project data is stored in `.data` by default; use `--data-dir` to choose another location.

The API provides OpenAI-compatible chat completions at `/v1/chat/completions`. See <http://127.0.0.1:8080/docs> while the app is running for the API reference.

See [benchmark.md](benchmark.md) for the RK3588 model comparison, measured local baselines, and remaining qualification work.

## GPU-first serving and optional NPU assistance

The exact backend commit paired with this checkpoint is recorded in
[BACKEND_REVISION](BACKEND_REVISION). Use the fork's `tinygrad-oNPU` branch
(`oNPU` locally here) or that recorded commit. It contains the Mali queue
workaround and retained Rockchip kernels; no separate reference worktree is needed.

Choose **Mali GPU · OpenCL** for normal loading. The prefill and decode NPU
controls default to **Off**. **Auto** compares the original layer with the NPU
path, including transfers, verifies numerical agreement, and selects an NPU
candidate only when its measured median is at least 10% faster. Most surrounding
operations remain on the GPU. Bulk prefill, single-row decode, and the final
vocabulary projection are measured separately.

The qualified Qwen3-0.6B Q8_0 checkpoint uses combined GPU projections and a
packed Q8 output-head read when both assistance phases are Off. On this board,
six short-context measurements per variant improve median prefill by 6.6% and
decode by 17.1%, with identical generated IDs. Full logits match bit for bit on
256 held-out positions and synthetic 32K/40K cache-boundary checks. These are
improvements over the prior GPU path; they do not establish an RKLLM win.

The **Rockchip NPU + CPU** profile prepares FP16 weights and places supported
dense decode projections on the NPU. Its prefill control defaults to Auto in
the load form; Auto selects supported prefill work only when it wins. Other
operations use the CPU. It is an
inference profile, not a claim that the complete model runs on the NPU or that
quantized GGUF weights execute with native quantized arithmetic.

A GPU-only model and a different NPU-assisted model can remain loaded and serve
requests concurrently. Each model has its own worker, request admission, KV
state and cancellation. Use its **Chat** button to select it; API requests route
by `model` ID. Unloading one model preserves the other. NPU assistance on a GPU
model reserves the same physical NPU, so disable assistance when assigning that
NPU to a separate model.

Prepare one model at a time. Loading a second model while another is preparing
returns HTTP 409 to prevent simultaneous startup allocations from exhausting RAM.
Once both are ready, their inference requests can run concurrently.

For explicit API loading:

```bash
curl http://127.0.0.1:8080/api/models/GPU_MODEL_ID/load \
  -H 'Content-Type: application/json' \
  -d '{"profile":"CL","context":512,"realize":true}'
curl http://127.0.0.1:8080/api/models/SMALL_MODEL_ID/load \
  -H 'Content-Type: application/json' \
  -d '{"profile":"ROCKCHIP","context":512,"realize":true,"npu_prefill":"auto"}'
```

For measured assistance on one GPU model, add `"npu_prefill":"auto"` and/or
`"npu_decode":"auto"` to its load options. Actual execution devices and placement
measurements are reported in `/api/status`. Saved user profile preferences remain
in effect; select OpenCL to change an existing CPU preference.

CPU/NPU serving selects a suitable prefill chunk and reuses its prepared graph.
Cold preparation currently takes about 150 seconds while compiling prompt sizes
and greedy/sampled paths; steady-state latency excludes that one-time load.
Earlier Qwen3-0.6B tests beat local FP16 RKLLM references at 128 and
256 input tokens. Two GPU-only 16K cold pilots also beat a local W8A8 reference,
with a passing long-context quality gate; these are single runs per workload.
Those references do not certify the published speed-priority export recipe.
The official benchmark numbers are valid speed targets; comparisons report
workload, dtype and clock scope. Matched local SDK/export reproduction is
optional when useful and currently deferred. Output quality is qualified
independently, and a faster-than-published timing does not certify a matched
local comparison.
The installed SDK stops at 16K, so no 40K speed comparison is available.
The 40K API passes capacity/cache/recovery checks, but cold first text takes
55.6 minutes and the three-position retrieval probe scores only 1/3.
An identical cached request reaches first text in 1.58 seconds. See the
[verification record](VERIFICATION.md) for measurements and limitations.

## Image conversations

Import a local Hugging Face `SmolVLM-256M-Instruct` directory containing
`config.json`, `model.safetensors`, `tokenizer.json` and `generation_config.json`.
The directory is referenced in place. Load the vision model, then use **Image**
in the chat composer to attach one PNG or JPEG per conversation. Start a new chat
to use another image. The worker uses one square 512-pixel image and 64 image
tokens. Prompt, image positions and reply share the loaded context capacity.
The exact qualified SmolVLM checkpoint uses INT8 text projections through its
declared 8,192 positions on the NPU profile. The GPU profile uses FP16 weights.
The small model's long-context retrieval is not consistently correct. The qualified long-context
NPU path uses per-row INT8 query/key score dots, FP32 softmax/value accumulation,
and pooled CPU decode. Matched 8K tests beat the earlier local W8A8 reference in
both phases. Reproducing a recorded level0 SDK export is optional and currently
deferred; output-quality checks remain separate from the published speed target.
The experimental FP16 attention dot remains disabled after a cache-output
mismatch. See the [verification limits](VERIFICATION.md).

The Rockchip hybrid profile uses FP16 vision operations and, for the qualified
checkpoint, INT8 text projections with CPU attention and sampling. **Use GPU for
images when available** allows the GPU to handle image operations that benefit
from sharing buffers with the NPU. Turn it off to keep the GPU free for another
model. Missing GPU support falls back to CPU/NPU vision. GPU vision is restricted
to explicitly qualified checkpoint hashes; ordinary text model profiles retain
their existing placement behavior.

Preparation warms the encoder, adaptive embedding-input prefill variants and both
greedy and sampled generation before reporting ready. Sampling stays inside the
prepared decoder graph, including the first image token. Vision time, prefill tokens/s and image-to-answer time
appear in API metrics, and the actual device reports GPU use. Backend comparisons
and qualification limits are tracked in `~/tinygrad/docs/rockchip-vlm-progress.md`.
For the accelerated vision profile, load with:

```bash
curl http://127.0.0.1:8080/api/models/MODEL_ID/load \
  -H 'Content-Type: application/json' \
  -d '{"profile":"ROCKCHIP","context":512,"realize":true,"npu_prefill":"auto","vision_gpu":true}'
```

With a ready vision worker, check image requests and cancellation:

```bash
.venv/bin/python scripts/smoke_vlm.py --url http://127.0.0.1:8092 \
  --model MODEL_ID --image /path/to/image.jpg --report /tmp/vlm-check.json
```

## Feature comparison with oMLX

The [current comparison summary](docs/omlx-comparison.md) separates the latest
frozen API evidence from subsequent CPU-only prefill verification and lists
the remaining batching, cache, model support and context gaps.

This is a source audit, not a speed comparison between Apple Silicon and RK3588.
The [pinned oMLX documentation](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/README.md)
and [local evidence](docs/omlx-feature-comparison.json) define the comparison.
The original comparison is pinned to oMLX `d762927` and EAGLE-4 `ea02a3b`.
The [October 25 freshness check](docs/omlx-eagle-freshness-20261025.json)
inspected oMLX `98de148` and EAGLE-4 `60bab6b`. The inspected
[oMLX DFlash page](https://github.com/jundot/omlx/blob/98de1482882fe22edd1310a0c428bddd94b9f006/docs/experimental/dflash_mlx_integration.md)
documents scheduler integration and shared paged RAM/SSD prefix caching, which
remain oNPU gaps. EAGLE-4 added a TokenSpeed TP8 measurement; its inspected
[support matrix](https://github.com/eagle-hq/EAGLE-4/blob/60bab6b469e795d47bda82b16243b82bb7b6206f/docs/support-matrix.md)
names no Qwen3.5-2B-compatible draft. This check covers selected official pages,
not a full audit of either new head.

| Capability | oNPU now | Remaining gap |
| --- | --- | --- |
| Independent models | GPU-only and NPU-assisted workers | Measured GPU-target/NPU-draft coordination |
| Chat API and streaming | Present, with reasoning and tool-call deltas | Additional API families |
| Tool calling | JSON and supported XML formats | Broader formats, MCP and schema validation |
| Concurrent requests | One active request per worker | Continuous batching within a model |
| Prefix cache | Recent RAM prefix per worker | Shared paged cache and SSD persistence |
| KV compression | No served compressed-KV mode | Optional lossy [Affine4](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/docs/affine4.md)/[Affine8](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/docs/affine8.md) for full attention; recurrent/sliding caches stay native |
| Model management | Per-load options, RAM admission, automatic idle unload | Persisted per-model presets and memory-pressure LRU swapping |
| Multimodal | SmolVLM image path | Broader VLM/OCR image frontends, audio and video |
| Responses API/state | Missing | [Create/get/delete and previous-response context](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/omlx/server.py#L8249) |
| Decision/tokenizer APIs | Missing | [System One](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/omlx/server.py#L4189), [tokenize/detokenize aliases](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/omlx/server.py#L8112) |
| Speech APIs | Missing | Optional `mlx_audio` [ASR/TTS, processing and voices](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/omlx/api/audio_routes.py#L474), plus [custom transcription WebSocket](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/omlx/api/audio_routes.py#L647) |
| Speculative decoding | Pinned Qwen3.5-2B DFlash2: current-source short/near-8K API checks passed; explicit force uses batched fallback retention, automatic routing unchanged | Model-maximum context, broader pairs, NPU draft coordination and unprofitable-case costs |
| Dashboard and metrics | Present | Further qualification as model coverage expands |

Responses storage is optional; the source audit does not establish persistence
across restarts. Audio routes require `mlx_audio` and compatible engines. The
custom WebSocket is `/v1/audio/transcriptions/realtime`; it does not establish
general `/v1/realtime` compatibility. This inventory informs later priorities;
the active work remains inference speed and trained speculation.

oMLX has model-dependent [DFlash/MTP integration](https://github.com/jundot/omlx/blob/d76292753cd61faab831cc5677d1262764d1bed3/docs/experimental/dflash_mlx_integration.md).

The opt-in CPU+NPU recurrent path supports parked head workers and pooled
prefill sigmoid. A matched Qwen3.5-2B test measured **79.87 prefill / 10.73 decode
tok/s**, versus 71.26 / 10.69 with pooled scan and generic sigmoid. Prefill
improved by 12.1%; decode ranges overlap. All twelve output sequences matched
the earlier INT8 baseline. This backend profile is not automatically enabled
by the app worker. Its output quality versus FP16/RKLLM remains unqualified.
[Model evidence](docs/benchmark-cpu-sigmoid-backend-model-v1.json).

### Trained DFlash2 (experimental)

Selective FP16 vector loads are integrated for measured Mali shapes and row
capacities. The [V33 trained-target experiment](docs/speculative-half-decode-target-v33.json)
measured 1.33–1.34× faster decode with exact logits, captured features and state.
[V37 GPU-profile regression checks](docs/speculative-routing-preflight-v37.json) passed
336 methods plus 101 subtests, Ruff and mypy over 262 backend source files.

The new two-row FP32 GEMV candidate passed exact operator and integration checks
(315 methods plus 125 subtests). Its selected 8-row head/down operators were
about 1.87×/1.82× faster. An isolated whole-model comparison then reduced median
speculative request latency from 8.86 → 7.95 seconds for counting, 8.58 → 7.80
for code, and 14.18 → 13.70 for explanation. Counting and code also beat fresh
ordinary decoding; explanation still favors ordinary. All 54 frozen output
sequences matched. The [V38 API cohort](docs/speculative-api-tiled-v38.json) then
qualified the integrated tiled source within its tested scope: 36 short off/force
greedy requests, 27 fresh probability comparisons, the 8,176-input plus 16-output
boundary in off/force/auto modes, sampled temperature-0.6 cancellation and exact
recovery passed. V38 regression checks totaled 351 methods and 156 subtests.
Auto selected speculation for a short counting request and ordinary decoding at
the long boundary because estimated draft setup cost exceeded its reply budget.
Both original models were restored and answered fresh requests. The normal server
was not restarted. Model-maximum context 262,144, RKLLM quality and general speed
remain unqualified. [Whole-model comparison](docs/speculative-vocabulary-row-tiling-model-v1.json),
[Integration checks](docs/speculative-vocabulary-row-tiling-preflight-v2.json).

V38 short complete-request medians (ordinary/forced speculative) were
10.40/7.41 seconds for counting, 8.23/7.65 for code and 10.97/13.34 for
explanation. Forced speculation was slower for explanation. At the 8,176-token
boundary, single functional trials took 452.06 seconds with `off`, 484.53 with
`force` and 463.65 with `auto` selecting ordinary. These are minutes-long
functional checks, not a speed qualification. See [benchmark.md](benchmark.md)
for the full table and routing checks.

[Prior-source API cohort (V37)](docs/speculative-api-routing-v37.json) remains a
separate dated receipt. Its timings and source qualification are not transferred
to V38.

A newer opt-in decoder change is testing strict-extension reuse of matching
target and DFlash2 prefix caches. It also invalidates token cache markers on
direct target forwards. CPU checks passed (75 methods and 52 subtests; four skips).
The first cold-replay comparison failed a strict recurrent-state check; that
[failed gate](docs/speculative-prefix-reuse-core-v1.json) is preserved. A later
[matched-seed GPU diagnostic](docs/speculative-prefix-reuse-matched-v4.json)
repeated one short continuation six times. All 42 state buffers and common probe
logits were bit-identical between ordinary cached and speculative cached
continuation; all 18 temperature comparisons had zero probability error and
total variation. Cold replay differed equally from both cached paths.
This verifies the core cache in that fixed case, not broader quality or model
maximum context.

The [V40 private API check](docs/speculative-prefix-reuse-api-v40.json) then
verified the pinned experimental worker's real chat path: a forced follow-up
reused 31 of its 55 prompt tokens and matched ordinary cached output. Automatic
cached requests stayed ordinary; `cache_prompt=false` reported zero reuse.
Cache-enabled sampled cancellation and greedy recovery passed. The formatter
now completes normal decoder bookkeeping after its last output, and this pinned
template preserves the generated empty think header in assistant history.
The experimental loader enables reuse; the general backend option remains off
by default. App checks passed 301 methods and 21 subtests, plus full Ruff.
Both original workers were restored and answered fresh requests.

The original run controller rejected an obsolete validator summary field.
Separate corrected validation and an independent audit verified the recorded
requests without changing the raw run or original failure record. This is a
short functional qualification, not a model-maximum, RKLLM-quality or speed
qualification.

The later [V41 long-conversation check](docs/speculative-prefix-reuse-long-api-v41.json)
passed on its frozen source at context 8,192. An 8,081-token forced follow-up
reused 8,049 tokens and returned first output in 1.98 seconds. An automatic cached
request reached the exact 8,191-input plus one-output budget boundary, reusing
8,115 tokens and returning first output in 5.23 seconds. Cache-enabled sampled
cancellation, cold recovery and an over-budget HTTP 400 rejection passed.
The cold requests still took over seven minutes to first output. These are
single-output functional cases; sustained decode, complete sampled generation,
the declared 262,144-token model maximum and RKLLM parity remain unqualified.
See [benchmark.md](benchmark.md) for prefill proxies and complete request times.

An optional backend memory layout now avoids retaining the draft's full
projected-feature history when prepared proposals only need K/V and position
metadata. CPU checks and a [trained-draft GPU component check](docs/speculative-draft-feature-history-component-v1.json)
passed exact K/V, logits, candidate-ID and sampled-probability comparisons through
8,192 synthetic feature rows. Retained history was 32 MiB in the control and zero
in the candidate. This measures that history tensor, not total RSS or model-max
fit. The [V42 private API check](docs/speculative-draft-feature-history-api-v42.json)
then enabled the new layout in the pinned experimental loader. All ten short
request controls and sampled cancellation/recovery passed; telemetry reported
zero retained projected-history bytes and preserved cached-prefix counts.
Generic backend callers still retain features by default for eager execution
and deep rollback. The new layout has 8K component parity plus short API
qualification; V41's long-conversation timings describe the previous retaining
layout. Model-maximum context and speed qualification remain open.
V38 remains a receipt for its frozen source, rather than qualification of this
newer code.

A newer backend option, `prepare_proposals(reuse_prefix_inputs=True)`, reuses
owned padded K/V inputs and updates only changed full-attention rows. Sliding
windows refresh, and rollback or failures clear stale padding. The first tensor
assignment variant was slower and was kept disabled. The
[native-copy component check](docs/speculative-draft-reusable-inputs-component-v2.json)
then preserved trained-draft K/V, padded inputs, logits, IDs and sampled
probabilities through 8,192 synthetic feature rows. Repeated-prefix input
staging took 1.45–2.87 ms versus 14.62–16.55 ms for the control. These are
component timings, with three warmups and six measured pairs per prefix;
they do not establish a complete proposal or generation speedup. CPU checks
passed 61 methods and 56 subtests, plus three Mali fixture tests, Ruff and mypy.
Both original workers were restored and answered fresh requests. The option
defaults off and is not enabled by the app. A later
[54-request full-target comparison](docs/speculative-draft-reusable-inputs-model-v1.json)
matched all frozen output IDs but found no complete-request gain: candidate
medians were 0.3–1.1% higher than the current path. The app therefore keeps
pad/clone inputs. That native-input option remains disabled. Shared-table API qualification is
recorded separately below; model-maximum context remains pending.

A separate [short-row Q8 operator](docs/benchmark-mali-q8-short-head-operator-v4.json)
now preserves the dense FP32 reduction order with FP16-rounded Q8 weights.
The direct real-weight test reduced one-row head latency from 53.70 to 34.77 ms;
three/eight-row paired cases remained slightly slower than dense. All output
bits matched, including actual public-kernel routing. That standalone operator cohort preceded the shared-table model/API integration
below. Its timings establish no inference-wide speed claim.

The [shared-table model comparison](docs/benchmark-mali-q8-shared-table-model-v1.json)
then released the actual dense tie and kept one raw Q8 allocation, while leaving
its real logical FP16 view unrealized through all 72 requests. Every output
matched its frozen reference. Observed ordinary-request medians were 4.4–6.2%
shorter; speculative deltas were smaller and ranges overlap in two workloads.
The [V44 private API check](docs/speculative-shared-q8-table-api-v44.json) now
qualifies this worker hook for the pinned experimental pair on the measured
Mali device/driver. All ten short cache/ordinary/automatic/cold controls and
sampled cancellation/recovery passed; table telemetry confirmed raw sharing
without dense-view or projected-history materialization. Other profiles/devices
retain their existing path. App checks passed 307 methods plus 23 subtests and
Ruff; original workers were restored with fresh replies. Context cap/admission
are unchanged; the new layout's full-8K API boundary, model maximum and RKLLM
quality/speed qualification remain pending. Benchmark records retain the
clock/process-order and memory-measurement limits.

The [V46 long attempt](docs/speculative-shared-q8-long-api-v46-attempt.json)
proposed seven tokens at an 8,081-token prompt with 8,049 cached tokens, then
stopped after one visible token and EOS. Its exact-16-output harness check
failed before matched ordinary/cancellation/recovery/boundary controls. Current
long API qualification remains open; this is not sustained decode evidence.

The [V47 attempt](docs/speculative-shared-q8-long-api-v47-attempt.json) used a
one-digit seed and produced all 16 requested follow-up tokens at an 8,081-token
prompt: 14 proposed, 8 accepted, then 6 ordinary-fallback tokens. Fallback left
the draft behind and cleared its reusable prefix, so the full long API cohort
stopped before ordinary comparison and recovery controls.

The backend now has an opt-in `retain_fallback_prefix` option, requiring
`reuse_prefix=True`. It observes target features already computed during
fallback, without adding target forward calls. A deferred final token stays
unprocessed and outside both cached prefixes. The option defaults off and is
not enabled by the app. CPU verification and the short trained-model study are separate from pending
API, full-8K and model-maximum qualification.

[V36](docs/speculative-api-half-decode-v36.json) remains a prior frozen-source
cohort with its own timings and probability audit.

The pinned Qwen3.5-2B/DFlash2 pair supports greedy and positive-temperature
speculation on the OpenCL GPU. Its implemented sampler uses conditional draft
probabilities, target acceptance tests and residual correction after rejection.

[Prior API cohort (V31)](docs/speculative-api-half-prefill-v31.json)
covers its frozen bounded-FP16 source: 36 short greedy requests with three
warmups and three measured trials per engine and prompt; all responses matched
frozen references. All 27 probability comparisons were freshly executed with
zero measured difference. The exact 8,192-token API boundary passed with 8,176
input tokens plus 16 output tokens. Ordinary, speculative and post-cancellation
greedy responses matched the frozen boundary response. Sampled SSE at
temperature 0.6 passed wrong-ID preservation, matching-ID stop and stream
completion. Both original models were restored and answered fresh requests.

That cohort’s V29 preflight passed 38 CPU tests plus 24 subtests, four native GPU tests plus 15
subtests, and 255 app tests plus 17 subtests, Ruff and mypy over 262 source files.
Median complete-request times (ordinary/speculative) were 13.38/9.56 seconds
for counting, 10.98/9.61 for code and 14.15/17.23 for explanation. Single uncached
long requests took 466.02/490.27 seconds. Speculation still loses on explanation
and that short-output long-prompt case; general speed/quality, model maximum
262,144 and RKLLM parity remain unqualified. See [benchmark.md](benchmark.md)
for prefill and decode rates. The [V25 API cohort](docs/speculative-api-deferred-v25.json)
retains its historical timings and explicitly reused audit provenance.

The profile enables optional
[bounded FP16 prefill](docs/speculative-half-prefill-target-v30.json). The separate
V30 paired trained-target core experiment measured 2.01× faster prefill for
counting, 1.36× for code and 1.97× for explanation, with identical prefill
logits, features and persistent state. Those option-off/on measurements are
not the V31 ordinary/speculative API comparison. The core option defaults off.

This profile uses packed GPU state snapshots and avoids the unused 970 MiB
packed output-head copy. Selective draft input-projection packing adds 48 MiB
to memory admission. The request cohort does not isolate its causal speed
effect. Target weights run in FP16 while the target decoder starts with FP32
activations. Measured peak RSS covers all owned test processes; it is not a
GPU-only memory measurement. Deferred fallback delivers the selected token before
computing its successor and can leave the last delivered input pending, following
ordinary generation's valid-prefix cache contract. This option defaults off in
the core; the pinned experimental app profile enables it. The measured explanation
requests used 27 fallback forwards for 28 fallback tokens and only two state snapshots.

Load the exact paired target with these options and a local directory containing
its [pinned trained draft](https://huggingface.co/rasyosef/Qwen3.5-2B-DFlash2/tree/c9aefeb3bccf819f3575521801d95b3ace87f8e7):

```json
{
  "profile": "CL",
  "context": 8192,
  "realize": true,
  "npu_prefill": "off",
  "npu_decode": "off",
  "speculation": "dflash2-experimental",
  "draft_path": "/path/to/DFlash2-Qwen3.5-2B"
}
```

Experimental loading is opt-in and currently accepts contexts 64–8,192. After
loading, per-request `"speculation":"auto"` uses startup observation-cost estimates
and the available reply budget to choose ordinary or speculative decoding; an
available cached prefix selects ordinary decoding. `"off"` forces ordinary decoding.
`"force"` explicitly exercises speculation regardless of the upfront estimate,
while retaining measured per-block fallback. The selected mode and routing reason
are recorded in worker metrics as `generation_mode` and `speculation_routing`.
V37 verified these routing controls on its isolated server.
[Routing validation status](docs/speculation-cost-routing-status.json).

An optional backend extra-cost threshold passed CPU/App checks and a 54-request
trained greedy comparison. Explanation measured 14.24 → 13.02 seconds, counting
was similar, and code measured 8.52 → 8.70 seconds. Clock readings varied during
trials. This option remains disabled in the app; it establishes no general speed
gain or new API qualification.
[Complete comparison](docs/speculative-extra-budget-model-v2.json).

Temperature zero uses greedy speculation and positive
temperatures use the trained sampler. The core supports an optional seed, but
the API exposes no seed field. `cache_prompt:false` bypasses ordinary text-prefix
reuse; speculative requests currently prefill afresh.

Both original GPU/NPU workers were restored and checked with fresh replies after
verification. The normal service on port 8091 was not restarted. Earlier
[V13/V14](docs/speculative-api-sampled-v14.json),
[V3](docs/speculative-api-prepared-v3.json),
[V4](docs/speculative-api-adaptive-v4.json), and
[V5](docs/speculative-api-validrows-v5.json) records describe historical source
cohorts, rather than qualifying later changes.

### Remaining speculation work

Model-maximum context, broader trained pairs, GPU-target/NPU-draft coordination,
and unprofitable-request overhead remain open. The comparison table also records
continuous batching, shared paged/SSD KV caching, persisted per-model presets,
and broader API/multimodal coverage as gaps.

The backend now bounds sliding-layer draft buffers by their window and uses the
checkpoint's context limit in the core. An opt-in bounded attention path also
preserves the model's output gate. [Operator checks](docs/speculative-long-context-operators-v3.json)
passed on a small CPU fixture with a 600-token prefix and directly on Mali.
The [real-checkpoint core check](docs/speculative-long-context-core-v16.json)
also passed at configured context 8,192 with an actual 2,061-token prompt:
16 greedy IDs matched the ordinary reference, sampled cancellation and recovery
passed, and all 27 probability comparisons matched exactly. Speculation was
slower in this single long-prefix trial. The historical V19 run qualified the
API boundary at 8,192; V21 rechecked that boundary after selective packing.
The model maximum of 262,144 remains unqualified. V15, V16 and V19 describe
their separate pinned builds.

The refreshed [EAGLE-4 0.1.2 matrix](https://github.com/eagle-hq/EAGLE-4/blob/4cfc53cd88e588b9d86bd9119e7a3903709d588d/docs/support-matrix.md)
uses Qwen3.8-27B as its default measured target and includes released
language-only Nemotron 120B-A12B rows. Upstream accepts the latter's reported
divergences under its engine-numerics rule. The inspected matrix names no
matched small EAGLE-4 target/draft pair; this is a scoped finding, not a global absence claim.
The separately documented Qwen3.5-2B/DFlash2 pair uses a different trained draft.
EAGLE-4 support on RK3588 remains unqualified. Source links, dated refreshes
and checkpoint details are in the [comparison record](docs/omlx-feature-comparison.json).

## Development

```bash
uv sync --extra test --python ~/tinygrad/.venv/bin/python
.venv/bin/python -m pytest -q
```

With a ready model, exercise GPU serving, streaming and cancellation through the real API:

```bash
.venv/bin/python scripts/smoke.py --model MODEL_ID --profile CL --context 1024
```

See [VERIFICATION.md](VERIFICATION.md) for current device and backend limitations, and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for attribution.

[Fallback-prefix candidate CPU evidence](docs/speculative-fallback-prefix-retention-cpu-v1.json):
28 targeted methods plus 31 subtests passed (two GPU-only skips), all 330 app
JUnit cases passed, mypy checked 264 sources and full Ruff passed. The option
remains disabled in the app; the trained-model study measured mixed costs.

The [trained fallback-retention study](docs/speculative-fallback-prefix-retention-model-v1.json)
passed 72 requests and all cached ordinary-output comparisons. Combined time
improved 8.30% for counting versus the discarded forced path, but regressed
5.05% for explanation. Cached ordinary decoding had the lowest combined cost
on both, so the app keeps the option off. The frozen owner retained an obsolete
two-process completion gate and exited 2; the child passed and a separate
one-process data revalidation is recorded. Both original workers were restored
and freshly checked. This is short-prompt core evidence, not an 8K API or RKLLM win.

[Fallback feature batching](docs/speculative-fallback-prefix-retention-model-v2.json)
now reduces repeated observation work using bounded owned feature copies. On the
trained pair, combined seed+follow-up time improved 8.00% for counting and 12.00%
for explanation versus scalar retention, with the same output and target-call
counts. Cached ordinary decoding remained fastest overall, so the app option
stays off. CPU/App checks and the measured scope are documented; full 8K/API and
model-maximum qualification remain open. The original validator failure is
preserved with separate corrected data verification.

A [selected official-document refresh](docs/omlx-eagle-freshness-20261025-v48.json)
checked oMLX `78fa08b` and unchanged EAGLE-4 `60bab6b`. Current oMLX docs still
describe standard batched-engine scheduling and shared paged hot RAM/SSD prefix
caching, which remain oNPU gaps. The inspected EAGLE-4 matrix names no matched
small Qwen3.5-2B pair; this is a matrix-scoped finding, not a global absence claim.

The [frozen V48 API qualification](docs/speculative-forced-batched-retention-api-v48.json)
now covers explicit forced-request batched retention on the measured shared-table
profile. Automatic/off defaults keep their existing paths. All 13 short controls
and six long requests passed, including exact forced/ordinary 16-token output
matching, sampled scoped cancellation, cold recovery, an 8,191+1 token budget,
and rejection of an over-budget request before generation. The cold automatic
request exercised the scalar/non-retaining reset after force. App checks passed
341 JUnit cases; both original workers were restored and answered fresh requests.

This qualifies the pinned greedy API controls through the tested 8K range, plus
sampled interruption controls. It does not establish broad sampled quality, an
RKLLM speed win or model maximum 262,144. Uncached 8K prefill still took roughly
7–8 minutes in these single-request observations; improving that latency remains
necessary for the requested usability target.

[Configured-capacity memory probes](docs/speculative-context-capacity-allocation-v1.json)
now cover 8K/16K/32K/64K allocations with 16 matching short ordinary/speculative
requests. At 64K the process peak was 9.156 GiB and minimum available RAM 3.237 GiB.
These used only 30/27-token prompts and are not full-prefix qualification; the
experimental DFlash2 API cap remains 8K. The 262K maximum is still unqualified.
The capacity-probe owners restored both original serving models afterward.

[Context-aware prefill checks](docs/speculative-context-aware-prefill-cpu-v1.json)
cover consistent selected 64/128-token grouping in warmup and requests, plus
prepared 64-row draft observation math and cache ownership. A separate [actual
16K-prefix core diagnostic](docs/speculative-actual16k-core-v1.json) completed
one 16,256-token OpenCL prompt at context 16,384 with chunk 64. All seven cold,
cached, cancellation and recovery cases passed, along with an independent audit;
both original workers were restored and answered fresh requests.

This is bounded greedy core evidence. The speculative path was slower in both
single-request comparisons (1,111.006 s vs 973.890 s cold; 9.787 s vs 7.202 s cached).
The experimental DFlash2 API cap remains 8K; the 16K API, sampled 16K execution, broad quality, model maximum 262,144,
and benchmark speed qualification remain unqualified. The separate allocation
probes still used only short prompts.

The backend's `extra/rockchip/shared_prefill/` directory contains the standalone
shared-buffer projection experiment, a portable benchmark CLI and its four
historical measurement records. It remains outside model serving and is not
TinyJit-safe. The packaged helper is byte-identical to the measured version;
the portable CLI has device-free checks but has not been rerun on hardware.
