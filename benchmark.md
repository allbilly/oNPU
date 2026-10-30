# Model Performance Benchmark

## Qwen2 0.5B — October 30 SDK/GQA API result

Ten uncached API requests use 128 input / 64 actual output tokens, context 512, CPU4–7 and unchanged adaptive governors, without sudo. Cells are published RKLLM / oNPU; bold marks the better reported median. This is a published-reference comparison with different operating conditions, not a local RKLLM rerun. TTFT ends at the worker's first emitted text; prompt preparation is excluded.

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2 | 0.5B | W8A8 / INT8 projections + FP32 CPU helpers | 128 | 64 | 145.90 / **142.06** | 41.58 / **43.98** | 669.56 / — |

| Phase | RKLLM tok/s | oNPU API median tok/s | Reference wins / measured requests |
|---|---:|---:|---|
| Prefill proxy (input / TTFT) | 877.31 | **901.11** | 7/10; range 725.32–968.74. |
| Decode | 41.58 | **43.98** | 10/10; range 42.21–44.12. |

The median prefill lead is 2.7%, while request variation is much larger. **This is a measured median lead, not a consistent every-request win or independent quality certification.** The previous API median was 741.77 tok/s with the same chat fixture; that is an unpaired historical comparison.

The app selects the exact checkpoint/context recipe: SDK blocked gate/up execution with direct SwiGLU consumption, one packed K-group upload, safe simultaneous input/output borrowing, and grouped-query CPU prefill with token-major queries. Quantization boundaries, scales, rounding and per-query FP32 reduction order are retained. Both new backend processes match all seven 64-token sequences and 32 full-vocabulary teacher vectors bitwise. Decode's original graph stays in use.

All 18 API checks passed, including cached one-token continuation, streaming, 448+64 context budget, cancellation and recovery. Both original GPU/NPU sessions were restored with fresh replies; the original GPU remained resident. Model-maximum context and independent RKLLM quality remain unqualified. A post-measurement guard tightens disjointness to cache-line boundaries; it is equivalent for this graph's 64-aligned buffer ranges, and the exact source-hash distinction is recorded in the receipt.

The same-checkpoint upstream llama.cpp CPU reference remains 291.95 CLI prompt-eval / 43.48 decode tok/s; its separate synthetic pp128/tg64 means are 289.37/39.22. Those are different timing boundaries from API input/TTFT. [Current evidence, every trial and source scope](docs/benchmark-qwen2-sdk-gqa-api-v1.json) · [Implementation details](../tinygrad/extra/rockchip/qwen2_prefill.md) · [Earlier llama runs](docs/benchmark-qwen2-dense-software-v1.json).

## Historical Qwen2 0.5B — October 30 prior API result

Five uncached API requests used 128 input / 64 actual output tokens at context 512, CPU4–7, and unchanged adaptive governors. No sudo or board-setting changes. Cells are published RKLLM / oNPU; bold identifies the better reported metric. These are worker first-emitted-text timings, with prompt preparation excluded, and are separate from raw backend first-token timings.

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2 | 0.5B | W8A8 / INT8 projections + FP32 CPU helpers | 128 | 64 | **145.90** / 172.56 | 41.58 / **43.94** | 669.56 / — |

| Phase | RKLLM tok/s | oNPU tok/s | Five-request result |
|---|---:|---:|---|
| Prefill proxy (input / TTFT) | **877.31** | 741.77 | 0/5 beat the reference; range 695.46–803.63. |
| Decode | 41.58 | **43.94** | 5/5 beat the reference; range 43.75–44.03. |

The app now selects prepared QKV, shared persistent host workers, token-major CPU prefill output, exact normal-range exp2, two-row raw SwiGLU, natural K896, native output synchronization/borrowing, and the two-KV-head decode pool for this exact checkpoint/context. Backend tests retain bitwise equality of all seven 64-token sequences and 32 full-vocabulary teacher vectors to the previous local recipe. This is a local arithmetic regression gate, not independent RKLLM quality qualification.

All 13 API checks passed: warmups and measured requests, a cached one-token follow-up, streaming, 448+64 context budget, cancellation and recovery. Original GPU/NPU sessions, options and selection were restored and both answered fresh requests. The original GPU stayed resident. This covers context 512; model-maximum context remains unqualified. **Decode wins the published-reference comparison; prefill still needs improvement.**

| CPU-only llama.cpp reference | CLI prompt-eval tok/s | CLI decode tok/s | Synthetic pp128 / tg64 tok/s |
|---|---:|---:|---:|
| Qwen2 Q8_0, four big cores, ctx512 | 291.95 | 43.48 | 289.37 / 39.22 |

The upstream llama.cpp binary matches checkout `0893ec9`; three CLI requests used the same checkpoint and exact 128 raw backend prompt IDs. CLI prompt evaluation excludes first-answer work. Its separate synthetic pp128/tg64 tests use different timing/context boundaries from the API. CPU kernel inspection informed reuse of the existing exact exp2 shortcut; no approximate llama polynomial was copied into this recipe.

[Current API, raw trials, numerical scope and profiler/llama evidence](docs/benchmark-qwen2-dense-software-v1.json).

## Qwen2.5 1.5B — October 30 current API result

Five measured API requests used the Qwen2.5 Q8_0 checkpoint at context 512, with 128 prompt and 64 generated tokens. Measurements are worker-reported TTFT and prefill proxy, not separate HTTP transport timings. No sudo or clock changes were used. Table cells are RKLLM / oNPU; bold marks the better reported value.

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | 1.5B | W8A8 / INT8 projections + FP32 CPU attention | 128 | 64 | 378.31 / **370.74** | 16.69 / **16.76** | 1689.21 / — |

| Engine / measurement | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|
| Published RKLLM | 338.35 | 16.69 |
| oNPU API worker, five measured requests | **345.25** | **16.76** |

The controller recorded 13 requests: three warmups, five measured requests, cached-prefix, streaming, a 448-token boundary prompt, cancellation, and post-cancel recovery. Two of five measured requests beat both published rates; one missed the prefill rate and two missed the decode rate. Median decode is only about 0.4% above RKLLM, so this is not a consistent or independently quality-qualified speed win. Measured ranges were 357.18–384.91 ms TTFT, 332.54–358.37 prefill-proxy tok/s, and 16.22–16.95 decode tok/s.

The cached follow-up reused 127 of 128 prompt tokens; the boundary request completed 64 tokens at exactly 512 total. Streaming completed, cancellation returned `Generation stopped`, post-cancel recovery succeeded, and the original sessions were restored with two fresh replies. The worker ran on CPU4–7 at ready and final checks. No replay or scale-bit environment overrides were set: the active source defaults included scalar copy, deduplicated guards, and `ONPU_EXP2_SCALE_BITS=1`; the Qwen profile enabled the existing exp2 fast path. Output quality and model-maximum context remain unqualified.

The exact-scale paired worker screen measured 327.24 / 15.99 tok/s control and 343.64 / 16.09 with scale bits, a 5.0% prefill-proxy and 0.6% decode median increase. It is worker IPC evidence, not a separate API treatment. The current API includes scalar copy, deduplicated guards, and exact scale bits, so it does not isolate the cause of its small decode lead. The rejected split-KV head experiment and wide-prefill prototype are absent from production; neither supplied a clear whole-model decode win.

The llama.cpp CPU reference retained below reports 82.00 CLI prompt-evaluation / 12.75 decode tok/s with a different prompt fixture and timing boundary from this chat-fixture API request. Keep it as a separate descriptive reference. [Exact-scale API receipt](docs/benchmark-qwen25-api-exact-scale-v1.json) · [Exact-scale worker pair](docs/benchmark-qwen25-exact-scale-paired-v1.json) · [Split-KV rejection](docs/benchmark-qwen25-split-kv-v1.json) · [Host-overhead evidence](docs/benchmark-qwen25-host-overhead-v1.json).

### Historical: paired worker and CPU diagnostics — October 30

The ordinary-worker control used the same 128-input/64-output request shape as the published comparison. Cells in the first table are RKLLM / oNPU; bold marks the lower TTFT or higher decode rate. Worker memory was not measured in this comparison.

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | 1.5B | W8A8 / INT8 projections + FP32 CPU attention | 128 | 64 | 378.31 / **376.43897** | **16.69** / 15.75714 | 1689.21 / — |

| Engine / measurement | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|
| Published RKLLM | 338.35 | 16.69 |
| Ordinary oNPU worker control, six measured requests | 340.05 | 15.76 |
| Rejected wide-prefill worker, six measured requests | 310.58 | 15.78 |
| Current llama.cpp CPU, exact 128-ID CLI requests | 82.00 | 12.75 |

The worker's prefill rate is the 128 input tokens divided by worker TTFT, while llama.cpp reports pure prompt evaluation. The llama CLI's exact 128 prompt IDs match the backend reference fixture; the worker uses the chat fixture. These boundaries and prompts differ. The ordinary worker's small median prefill lead over RKLLM is not a repeatable or qualified win; its decode median is about 5.6% behind. The wide-prefill arm was slower on prefill and is rejected. Its original validator failure remains recorded: only the two wide-arm warmups lacked wide-kernel capture; the separate measured-row audit confirms all six measured rows selected the wide kernel. It does not change the raw failure or qualify a win.

The llama.cpp CPU-only build is fresh at `0893ec9` (HEAD), uses the same Q8_0 GGUF, and runs on four big CPU cores. A separate standard `llama-bench` reported 70.759666 pp128 and 10.101472 tg64 tok/s; these synthetic measurements are not the same workload as the worker or CLI request. The paired lease is restored, and the shared restoration receipt records two fresh replies for the original sessions. Those worker screens did not establish a default speed gain; the wide production prototype and related scheduling experiments were removed, with no selectable wide flag remaining. At the time of this worker comparison, HTTP/API v2 at 323.32 / 15.82 was the latest API cohort. The completed API result above is a separate, unpaired cohort and does not establish a causal change.

[Paired worker comparison and measured-row audit](docs/benchmark-qwen25-wide-paired-v1.json) · [Historical wide-model candidate](docs/benchmark-qwen25-wide-model-v1.json) · [Current llama.cpp CPU receipt](docs/benchmark-llama-cpu-current-v2.json) · [Prior API v2 result](docs/benchmark-qwen25-api-fast-prefill-v2.json) · [Shared worker restoration receipt](docs/benchmark-qwen25-idle-phase-v1.json).

### Historical: Qwen2.5 1.5B matched adaptive sweep and upstream CPU reference

Updated October 30; oNPU measurements completed October 29. No sudo, governor, IRQ or voltage changes.

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | 1.5B | w8a8 / INT8 projections + FP32 CPU attention | 128 | 64 | 378.31 / **371.51** | 16.69 / **16.80** | 1689.21 / 4688.60* |

| Engine / measurement | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|
| Published RKLLM W8A8 | 338.35 (input/TTFT proxy) | 16.69 |
| oNPU candidate, six measured requests | **344.54** (input/TTFT proxy) | **16.80** |
| oNPU matched control, six measured requests | 344.63 (input/TTFT proxy) | 16.70 |
| Main upstream llama.cpp CPU, exact prompt, context 512 | 85.89 (internal prompt-eval) | 14.73 (internal eval,63steps) |
| Main upstream llama.cpp CPU, standard llama-bench | 82.33 (pure pp128) | 12.19 (separate tg64) |

The matched oNPU order was control/candidate/candidate/control, with three full warmups and three measured 128-input/64-output requests per process. All four processes passed exact generated-ID and 32-target full-vocabulary logit regression checks. The candidate adds an exact normal-range exp2 shortcut inspired by llama.cpp's finite-input fast path; it ties control prefill and improves decode by about 0.6%. Both candidate medians are slightly above the published target, while individual trial ranges cross it. This is median parity/slight lead under measured conditions, not a guarantee for every request or independent RKLLM quality qualification.

The main upstream checkout was `b92761a` at `~/rkllama/ggml-org_llama.cpp`. Its CPU-only binaries generated a successful smoke reply, then ran three context 512 CLI requests using the **same 128 prompt IDs**, 64 output tokens and four big CPU cores. The generation text differs between engines; local exact regression is not an independent quality-equivalence test. Standard `llama-bench` uses synthetic pp128/tg64 tests separately, with context derived from each test; its prefill rate excludes the oNPU TTFT boundary and its tg64 does not share the 128-token prompt. Those reference columns are labeled accordingly. Both programs used the same Qwen2.5 Q8_0 GGUF, with no GPU layers in llama.cpp. oNPU used CPU4–7 compute workers plus its existing CPU0 IRQ-wake helper and the NPU; the resident GPU model remained loaded and idle.

*The oNPU memory value is median sampled process lifetime peak RSS in MiB, including preparation/validation. RKLLM's MB accounting differs, so memory figures are not directly comparable. Both original serving sessions were restored and answered fresh requests after the tests.

[Raw trials, source provenance, reference runs and restoration scope](docs/benchmark-qwen25-exp2-abba-v1.json) · [CPU kernel review](docs/llama-cpu-kernel-review.md).

### Historical: Qwen2.5 1.5B API qualification v2 — context 512

This earlier API cohort is retained as historical; the October 30 API result above is the current cohort, and the comparison between them is unpaired.

| Serving path | Input / output / context | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|---:|
| Published RKLLM reference | 128 / 64 / — | **378.31** | **338.35** | **16.69** |
| oNPU API, five measured requests | 128 / 64 / 512 | 395.89 | 323.32 | 15.82 |

The no-sudo API worker completed three warmups and five measured requests, each producing 64 tokens. Measured ranges were 377.47–442.78 ms worker-reported TTFT, 289.08–339.10 prefill-proxy tok/s and 15.48–16.29 decode tok/s. The cached-prefix request, streaming request, 448-token prompt near the 512-token context limit, cancellation and post-cancel recovery checks all passed. All eleven selected worker features were active; the worker reported CPU4–7 both when ready and at the last affinity check. Both original serving sessions were restored and returned fresh replies. The API median is below the published prefill and decode rates and slower on TTFT, so this is a functional serving qualification, not a serving speed win. API output quality and model-maximum context remain unqualified.

The first API attempt is preserved separately because the restarted API parent inherited CPU0–3 from its monitoring controller. That run did not record the target worker's affinity mask directly. The corrected run required and recorded CPU4–7 before measurement and on its final affinity check. [Corrected API results, ranges, request checks and receipts](docs/benchmark-qwen25-api-fast-prefill-v2.json) · [Affinity-confounded v1 diagnostic](docs/benchmark-qwen25-api-affinity-diagnostic-v1.json).

### Historical: earlier exp2 scale experiment before exact-scale qualification

| Process | Median TTFT (ms) | Median prefill proxy (tok/s) | Median decode (tok/s) |
|---|---:|---:|---:|
| Scale screen, first process | 358.99 | 356.56 | 16.85 |
| Repeat process | 382.92 | 334.28 | 16.76 |
| Six-request pooled summary | 362.47 | 353.13* | 16.83 |

The isolated CPU exp2 component was 14.274% faster, with bitwise-equal output across its finite and edge fixtures. This did not establish a whole-model win: the first model process exceeded published prefill, the repeat did not, and the pooled rate is a summary across six requests rather than a matched control/candidate treatment. All six 64-token sequences and all 32 local full-vocabulary teacher rows matched the prior INT8 recipe exactly; this is local regression evidence, not independent FP16/RKLLM quality. Both model owners restored the original sessions, which returned fresh replies. *The pooled prefill proxy is 128 tokens divided by median TTFT; the median of per-request prefill rates is 353.16 tok/s.

At the time of this earlier report, the helper gated normal-range exponent-bit scaling behind `ONPU_EXP2_SCALE_BITS`, default off. The saved model reports record `ONPU_EXP2_FAST_PATH=1` but do not record the scale-bits setting, so they are not a measured `SCALE_BITS=0`/`1` comparison. The subsequent exact-scale worker pair and API follow-up are recorded above. The earlier component/model report remains historical and does not provide an independent output-quality check. [Earlier component and model runs](docs/benchmark-qwen25-exp2-scale-v1.json).


### Latest TinyLLAMA result — adaptive settings, no sudo, October 29

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TinyLLAMA | 1.1B | w8a8 / INT8 projections + FP32 CPU attention | 128 | 64 | **243.93** / 257.96 | 24.43 / **25.33** | 1093.66 / 3502.47* |

| Model | RKLLM prefill proxy (tok/s) | oNPU prefill proxy (tok/s) |
|---|---:|---:|
| TinyLLAMA 1.1B | **524.74** | 496.20 |

Six measured requests across two separately loaded processes, with three warmups per process, context 512 and unchanged adaptive governors. Process prefill medians: 531.52 and 472.07 tok/s. The recipe uses prepared QKV, shared CPU workers, token-major CPU attention output, eight-query attention tiles, zero-tail storage reuse and native transfer/cache optimizations. All 32 checked full-vocabulary teacher vectors remain bitwise equal to the prior CPU-attention recipe in both processes.

The first process exceeded the published prefill reference, but the repeat did not. The combined prefill median remains below RKLLM; a consistent prefill win is unachieved. Decode exceeds the published reference. Trial ranges overlap the prefill target. Independent RKLLM quality equivalence and API throughput remain pending. *The oNPU memory figure is median process lifetime peak RSS in MiB; RKLLM reports MB with different accounting, so it is not a directly comparable memory measurement. The GPU model stayed resident; both original sessions were restored with fresh replies after the TinyLLaMA development lease. The FP16 NPU-attention experiment failed its numerical gate and is not selected. [Raw trials, provenance and scope](docs/benchmark-tinyllama-adaptive-layout-v1.json).


### Latest Qwen3 result — fixed stock settings, October 29

| Qwen3 0.6B, 128 input / 64 output, context 512 | Published RKLLM | oNPU | oNPU gain |
|---|---:|---:|---:|
| TTFT (ms, lower is better) | 199.09 | **149.50** | 24.9% lower latency |
| Prefill proxy (tok/s) | 642.93 | **856.18** | **33.2%** |
| Decode (tok/s) | 32.91 | **38.58** | **17.2%** |

All 12 measured requests across four separately loaded processes beat both published speed targets. Prefill range: **825.32–862.76 tok/s**; decode range: **37.85–38.76 tok/s**. Each process had three warmups followed by three measured requests. Generated IDs and 32-target full-vocabulary teacher arrays match across all four blocks.

This is a repeatable **published-reference speed win for this workload**, conditional on supported stock CPU/NPU/DDR performance governors, NPU IRQs on CPU6 and CPU6 deep-idle disabled. It uses the existing numerical recipe, not the rejected score-vector kernel. The changes were authorized as temporary; these figures do not describe ordinary adaptive-governor serving. The four blocks have identical settings, so their inherited control/candidate report labels do not denote different treatments.

RKLLM states maximum CPU/NPU clocks; its table does not establish matching DDR/IRQ settings, checkpoint arithmetic or workload contents. Independent RKLLM quality equivalence, model-maximum context and a new API throughput run remain unqualified. Primary timing/numerical validation passed; ancillary CPU-source-export bookkeeping did not, and no new 1,024-target corpus run is claimed. Restoration receipts are recorded in the [complete result record](docs/benchmark-qwen3-stock-tuning-v1.json).

[Official reference](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md), rechecked October 29 · [TinyLLAMA next-step plan](docs/tinyllama-prefill-plan.md).


### Latest Qwen3 prefill check — October 29, score-vector experiment

| Model / current backend | RKLLM prefill proxy (tok/s) | oNPU prefill proxy (tok/s) |
|---|---:|---:|
| Qwen3 0.6B, existing recipe remeasurement | 642.93 | **667.79** |
| Qwen3 0.6B, rejected vector-score candidate | 642.93 | **655.35** |

Six measured 128-input/64-output requests per arm, context 512, balanced control/candidate/candidate/control process order. Bold denotes only the higher measured median against the published reference. The candidate loses to the existing recipe, so it is not promoted. Individual trials cross the reference threshold; this is **not a consistent or matched quality-qualified RKLLM win**. The 1,024-target worker failed before loading due to a missing staged helper import; completed timing/short-teacher checks remain separate valid evidence. Prior 642.22 tok/s results below remain their original cohort, not overwritten. [Raw trial summary](docs/benchmark-qwen3-score-vector-v1.json).

TinyLLAMA next: [prefill plan](docs/tinyllama-prefill-plan.md), starting from the newer focused 358.06 tok/s recipe rather than the older 235.05 sweep.


Updated: 2026-10-29 (Asia/Shanghai). Platform: Orange Pi 5, RK3588S, 16 GB RAM.

Reference: [Rockchip rknn-llm benchmark.md](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md).
The reference table uses maximum CPU/NPU frequencies and W8A8 exports made with `optimization_level=0`.
The allocated context is not stated in the upstream table. Local pilots use context 512 and existing host clock policies.

Performance cells below show **RKLLM / oNPU**. **Bold identifies the better reported number for the same model**,
not a certified win under matched conditions. A dash means a matching local measurement is unavailable.
The main table records GPU-first OpenCL baselines. Dtype cells show reference / local precision;
FP16 GPU operands currently come from Q8_0 GGUF checkpoints. Experimental CPU+NPU results are listed separately.
No row is currently qualified as faster in both prefill and decode at comparable output quality.
The 21 RK3588 baseline rows remain FP16 GPU measurements. New CPU+NPU INT8 measurements are recorded separately in the W8A8 sweep section below.

Qwen3-0.6B now has a separate CPU+NPU fast-profile result: the 64-token backend candidate reports 38.39 decode tok/s versus the published RKLLM 32.91. The API profile's prefill-proxy median is 650.49 versus 642.93 tok/s, but its 627.42–711.39 range crosses the reference and it stops at 59 of 64 requested tokens. This is a promising phase-specific result, not a consistent matched-quality speed win. [Measurements and API evidence](docs/benchmark-qwen3-cpu-token-inputs-v1.json) · [API qualification](docs/benchmark-qwen3-api-fast-prefill-v1.json).

Indexed sweep checked on October 28: [CPU+NPU candidate timings](#onpu-w8a8-sweep-cpunpu-candidates)
cover all 21 reference models plus Spark-X2.6, with 204 valid timing variants.
The latest [Qwen2 four-row prefill comparison](#qwen205b-four-row-swiglu-prefill-v1)
records 24 matched sequences and four bitwise-equal teacher arrays; its
prefill NPU/native-replay qualification passed. Quality and the all-model speed
goal remain unqualified. [Full verification](docs/benchmark-qwen2-row-batch-models-verification-v1.json).

Current preferred validated MiniCPM3 result: **15.72 prefill proxy tok/s and 4.61 decode tok/s**.
The CPU latent-projection candidate preserves all 24 generated sequences and
all four sets of 32 full-vocabulary teacher vectors exactly. Comparable RKLLM
quality remains unqualified. [Results and verification](docs/benchmark-cpu-mla-projections-models-verification-v1.json).

DeepSeekOCR grouped CPU experts passed the exact-bit and local numerical gates. The candidate becomes the preferred timing variant. This is text-only decoder inference; RKLLM and held-out quality remain unqualified. [Verification](docs/benchmark-deepseek-cpu-experts-verification-v1.json).

## Qwen3-0.6B prefill follow-up — October 29

The fast recipe is now selected by the oNPU worker for the verified F16 checkpoint with **Rockchip NPU + CPU**, context **512**, and prefill **Auto/On**. It includes prepared NPU attention, shared CPU workers, packing/view/copy optimizations and ready CPU token inputs. Other checkpoint/context profiles retain their existing behavior.

| Measurement | Input tokens | Actual output tokens | Median TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|---:|---:|
| Published RKLLM W8A8 | 128 | 64 | 199.09 | 642.93 | 32.91 |
| Matched backend control | 128 | 64 | 204.24 | 626.72 | 38.27 |
| Backend with ready CPU token inputs | 128 | 64 | 199.31 | 642.22 | **38.39** |
| oNPU API fast profile | 128 | 59 (64-token budget) | **196.77** | **650.49** | 37.36 |

The API median is 1.18% above the published prefill-proxy target, but its five measured requests range from 627.42 to 711.39 tok/s. **A consistent RKLLM win is not established.** The API measures first emitted text with a different chat prompt; the raw backend measures first generated token. Its naturally stopped 59-token decode rate is not a matched 64-output comparison. Bold marks better reported values, not a matched-engine certification.

The backend candidate preserves all 1,024 held-out full-vocabulary logit vectors bitwise. The app passes uncached generation, a one-token cached-prefix follow-up, streaming, the 448-token prompt + 64-token context boundary, cancellation and recovery. Both original models were restored and answered fresh requests. The one-token JIT input guard was added after the backend batch and is covered by CPU regression and the API test; source cohorts are recorded separately.

[Backend measurements](docs/benchmark-qwen3-cpu-token-inputs-v1.json) · [API qualification and source provenance](docs/benchmark-qwen3-api-fast-prefill-v1.json) · [Earlier transfer optimization](docs/benchmark-qwen3-batch-borrow-v1.json).

## RK3588

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :---------- | :--------: | :---: | :----: | :--------: | :------: | :------: | :--------: |
| Qwen2 | 0.5B | w8a8 / FP16 | 128 | 64 | **145.90** / 1280.52 | **41.58** / 10.21 | 669.56 / 2753.29* |
| MiniCPM4 | 0.5B | w8a8 / FP16 | 128 | 64 | **135.29** / 1262.78 | **45.34** / 8.34 | 534.82 / 2554.70* |
| Qwen3 | 0.6B | w8a8 / FP16 | 128 | 64 | **199.09** / 1406.79 | **32.91** / 7.96 | 791.11 / 3568.17* |
| TinyLLAMA | 1.1B | w8a8 / FP16 | 128 | 64 | **243.93** / 2736.21 | **24.43** / 5.87 | 1093.66 / 5798.29* |
| Qwen2.5 | 1.5B | w8a8 / FP16 | 128 | 64 | **378.31** / 4027.07 | **16.69** / 4.71 | 1689.21 / 8713.62* |
| RWKV7 | 1.5B | w8a8 / FP16 | 128 | 64 | **634.34** / 160077.04 | **13.82** / 0.80 | 1486.19 / 6501.57* |
| InternLM2 | 1.8B | w8a8 / FP16 | 128 | 64 | **380.14** / 4176.50 | **15.45** / 4.04 | 1775.56 / 9340.63* |
| Gemma2 | 2B | w8a8 / FP16 | 128 | 64 | **598.41** / 5833.00 | **10.37** / 3.19 | 2779.22 / 10090.47* |
| Gemma3n | 2B | w8a8 / FP16 | 128 | 64 | **1124.51** / 71678.78 | **9.91** / 2.22 | 2702.59 / 12978.16* |
| TeleChat2 | 3B | w8a8 / FP16 | 128 | 64 | **618.48** / 6211.90 | **10.18** / 3.00 | 2788.39 / 13416.96* |
| Phi3 | 3.8B | w8a8 / FP16 | 128 | 64 | **1017.28** / 161489.96 | **7.45** / 2.15 | 3758.34 / 12159.94* |
| MiniCPM3 | 4B | w8a8 / FP16 | 128 | 64 | **1408.49** / 388796.27 | **5.94** / 0.19 | 4384.02 / 12899.05* |
| ChatGLM3 | 6B | w8a8 / FP16 | 128 | 64 | **1352.94** / 300759.88 | **4.98** / 1.51 | 5985.99 / 12611.86* |
| Qwen3-VL | 2B | w8a8 / FP16 | 128 | 64 | **383.62** / 3945.27 | **14.98** / 3.77 | 1868.98 / 7008.97* |
| DeepSeekOCR | 3B(A570M) | w8a8 / FP16 | 128 | 64 | **701.21** / 28090.65 | **31.60** / 3.94 | 3066.66 / 10203.73* |
| Qwen3.5 | 0.8B | w8a8 / FP16 | 128 | 64 | **587.74** / 2735.69 | **27.05** / 4.54 | 1039.66 / 2466.36* |
| Qwen3.5 | 2B | w8a8 / FP16 | 128 | 64 | **776.95** / 7096.23 | **13.59** / 2.40 | 2121.91 / 5208.04* |
| Qwen3.5 | 4B | w8a8 / FP16 | 128 | 64 | **2068.88** / 23436.57 | **6.22** / 0.60 | 4575.78 / 9830.33* |
| Gemma4 | E2B | w8a8 / FP16 | 128 | 64 | **599.06** / 63315.40 | **11.12** / 2.88 | 2498.70 / 12862.52* |
| LFM2.5-VL | 450M | w8a8 / FP16 | 128 | 64 | **102.74** / 14427.52 | **62.16** / 7.18 | 436.94 / 1743.02* |
| Spark-X2.5 | 1.7B | w8a8 / FP16 | 128 | 64 | **386.23** / 4075.64 | **15.61** / 3.91 | 1830.32 / 7123.13* |

\* Local memory is the worker's lifetime peak RSS in MiB, including loading and compilation.
The reference does not specify an equivalent RSS measurement; memory numbers are recorded without declaring a memory winner.
Local timings exclude loading and warmup. Checkpoint and result hashes, device options and allocated contexts are in
[the measured baseline record](docs/benchmark-results.json). Downloads may overlap these initial baselines;
exclusive paired measurements are required for final speed qualification.

All 21 main-table models now have complete measured baselines, plus Spark-X2.6. Each uses three full warmups and three measured 128-input/64-output runs. Full-model output quality and matched RKLLM performance remain unqualified.
The Qwen3.5 rows use `--gpu-recurrent-scan`, actual 128-token prefill chunks and row-major recurrent projections without packed copies.
TTFT is 2.74, 7.10 and 23.44 seconds for 0.8B, 2B and 4B, down from 41.22, 101.53 and 480.83 seconds.
The worker enables this path for the exact checked checkpoint hashes at context 512 or below, with realized weights and GPU-only inference.
The generic RAM admission estimate can still reject larger models before worker startup; that estimate has not been requalified here.
The RWKV7 FP16 GPU baseline uses one-token generic recurrent execution: its measured TTFT is 160.08 seconds (0.80 prefill proxy tok/s), and decode is 0.80 tok/s. Its full-model output quality is not qualified.
Qwen3-VL-2B measures its language decoder with text input only; image encoder/connector performance is not included.
DeepSeekOCR also measures text-only language decoding; its image encoder and connector are excluded.

<!-- onpu-w8a8-sweep:start -->
## oNPU W8A8 sweep: CPU+NPU candidates

These are actual CPU+NPU INT8 projection runs, shown as **published RKLLM / oNPU**.
Candidate timings now cover all 21 main models plus Spark-X2.6; quality and speed qualification remain open. A dash is pending or failed, never an inferred result.
Each timing row requires three full warmups and three measured 128-input/64-output requests at context 512,
stable measured IDs and positive NPU submissions in both phases. **Bold marks only the better reported number.**
Whole-model INT8 coverage and comparable output quality remain unqualified; no row is a certified RKLLM win.

A subsequent storage-view lifetime correction changes the current runtime. The
recorded timings below remain tied to their captured source cohorts; numerical
and performance requalification after that correction is pending. The fix
passes 144 tests, with 11 skips, plus mypy and Ruff.
[Regression, source and restoration evidence](docs/benchmark-memory-view-cpu-verification-v1.json).

A controlled Qwen3.5-4B follow-up captured an errno-14 failure for a 20 MiB
INT8 projection buffer (8,192×2,560) while more than 5 GiB remained available.
The running kernel explicitly reported an IOVA allocation failure. The tracked
DMA ranges span almost 4 GiB; intervals between tracked buffers are not proven
free. This is allocation evidence with deferred weight realization, not a new
successful timing row or a cause claim for every failed model. Both original
models were restored and answered fresh requests.
[Controlled allocation and kernel-log evidence](docs/benchmark-w8a8-allocation-context-v3.json).

An October 26 follow-up now completes Qwen3.5-4B with two memory domains and
deferred CPU weight realization. Three full warmups and three measured requests
record **32.20 prefill proxy tok/s** and **4.84 decode tok/s**, below the recorded
RKLLM reference of 61.87 and 6.22. Both phases execute NPU work in the second
domain and use native replay; all six generated ID sequences match each other.
This adds one timing candidate; output quality, full learned-matrix coverage,
and API support for the opt-in configuration remain unqualified.
[Measured run](docs/benchmark-npu-domains-qwen35-4b-v3.json) ·
[Capacity, hardware, cleanup and serving-restoration evidence](docs/benchmark-npu-domains-verification-v1.json).

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :---------- | :--------: | :---: | :----: | :--------: | :------: | :------: | :--------: |
| Qwen2 | 0.5B | w8a8 / INT8† | 128 | 64 | 145.90 / **142.06** | 41.58 / **43.98** | 669.56 / — |
| MiniCPM4 | 0.5B | w8a8 / INT8† | 128 | 64 | **135.29** / 177.75 | **45.34** / 42.73 | 534.82 / 1977.30* |
| Qwen3 | 0.6B | w8a8 / INT8 projections + FP16 attention‡ | 128 | 64 | 199.09 / **149.50** | 32.91 / **38.58** | 791.11 / — |
| TinyLLAMA | 1.1B | w8a8 / INT8† | 128 | 64 | **243.93** / 257.96 | 24.43 / **25.33** | 1093.66 / 3502.47* |
| Qwen2.5 | 1.5B | w8a8 / INT8† | 128 | 64 | 378.31 / **371.51** | 16.69 / **16.80** | 1689.21 / 4688.60* |
| RWKV7 | 1.5B | w8a8 / INT8† | 128 | 64 | **634.34** / 1963.26 | 13.82 / **14.51** | 1486.19 / 3711.14* |
| InternLM2 | 1.8B | w8a8 / INT8† | 128 | 64 | **380.14** / 533.32 | **15.45** / 15.11 | 1775.56 / 6220.91* |
| Gemma2 | 2B | w8a8 / INT8† | 128 | 64 | **598.41** / 713.58 | **10.37** / 9.93 | 2779.22 / 6247.94* |
| Gemma3n | 2B | w8a8 / INT8† | 128 | 64 | 1124.51 / **1091.49** | **9.91** / 8.57 | 2702.59 / 5957.46* |
| TeleChat2 | 3B | w8a8 / INT8† | 128 | 64 | **618.48** / 696.42 | **10.18** / 9.81 | 2788.39 / 11106.87* |
| Phi3 | 3.8B | w8a8 / INT8† | 128 | 64 | **1017.28** / 1137.48 | **7.45** / 7.22 | 3758.34 / 8922.37* |
| MiniCPM3 | 4B | w8a8 / INT8† | 128 | 64 | **1408.49** / 8140.42 | **5.94** / 4.61 | 4384.02 / 9778.67* |
| ChatGLM3 | 6B | w8a8 / INT8† | 128 | 64 | **1352.94** / 2298.50 | **4.98** / 4.32 | 5985.99 / 12262.25* |
| Qwen3-VL | 2B | w8a8 / INT8† | 128 | 64 | **383.62** / 732.18 | **14.98** / 14.39 | 1868.98 / 5752.71* |
| DeepSeekOCR | 3B(A570M) | w8a8 / INT8† | 128 | 64 | **701.21** / 1345.75 | **31.60** / 23.29 | 3066.66 / 8856.39* |
| Qwen3.5 | 0.8B | w8a8 / INT8† | 128 | 64 | **587.74** / 882.05 | **27.05** / 21.02 | 1039.66 / 2031.64* |
| Qwen3.5 | 2B | w8a8 / INT8† | 128 | 64 | **776.95** / 1122.42 | **13.59** / 11.13 | 2121.91 / 4647.27* |
| Qwen3.5 | 4B | w8a8 / INT8† | 128 | 64 | **2068.88** / 2677.79 | **6.22** / 5.29 | 4575.78 / 9737.73* |
| Gemma4 | E2B | w8a8 / INT8† | 128 | 64 | **599.06** / 873.51 | **11.12** / 9.58 | 2498.70 / 5589.87* |
| LFM2.5-VL | 450M | w8a8 / INT8† | 128 | 64 | **102.74** / 161.75 | **62.16** / 60.37 | 436.94 / 1505.80* |
| Spark-X2.5 | 1.7B | w8a8 / INT8† | 128 | 64 | **386.23** / 574.79 | **15.61** / 14.65 | 1830.32 / 5808.43* |

Additional candidate with no RKLLM reference row:

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :---------- | :--------: | :---: | :----: | :--------: | :------: | :------: | :--------: |
| Spark-X2.6 | 4B | — / INT8† | 128 | 64 | — / 1506.13 | — / 6.14 | — / 9341.85* |
Spark-X2.6 now has a measured INT8 candidate; no RKLLM reference row is available.

The latest gated-attention follow-up improves Qwen3.5-4B from 32.20 to
**41.74 prefill proxy tok/s**, and from 4.84 to **5.23 decode tok/s**.
Spark-X2.5 improves from 53.17 to **159.37 prefill proxy tok/s**, and from 9.10
to **13.10 decode tok/s**. The CPU kernel now handles output/head gates without
bypassing them. Spark's 512-token window covers its full allocated 512-token
context, so all 28 attention blocks use the same causal kernel. Smaller windows
retain their fallback. Both candidates remain below the recorded RKLLM rates.
All six requests per model have stable outputs within that candidate, but the
outputs differ from the previous oNPU path; comparable quality remains pending.
[Timing, profile, failed-attempt and restoration evidence](docs/benchmark-gated-attention-verification-v1.json).
The latest raw runs sample big cores at 1.8 GHz, NPU at 1 GHz and the memory
controller at 2.112 GHz. A post-run policy snapshot shows big-core minimum and
maximum limits pinned to 1.8 GHz. The reference uses maximum clock settings;
these local settings were not changed.

A later three-domain/deferred-loading sweep adds valid timings for Gemma2,
TeleChat2, Phi3, MiniCPM3, Gemma3n and Gemma4, plus the additional Spark-X2.6.
Every successful row has three full warmups and three measured requests with
stable measured 64-token outputs, native replay and positive NPU work in both
phases. Three domains were configured; actual use was one or two. These are
CPU/NPU candidates, which can be slower than the separate GPU baseline.
That initial ChatGLM3 attempt hit the 2 GiB RAM reserve before timing; the
private file-backed follow-up below supersedes it and preserves the failure. All attempts finished and both original serving
sessions were restored and answered fresh requests.
[Raw runs, validators, memory failure and restoration proof](docs/benchmark-w8a8-missing-domains-verification-v1.json).

The final ChatGLM3-6B gap is now measured using private file-backed packed CPU
weights: **55.69 prefill proxy tok/s** and **4.32 decode tok/s**, below the
recorded RKLLM reference of 94.61 and 4.98. All six requests have identical
64-token outputs. Observed minimum available RAM was about 5.0 GiB; the earlier
loader stopped at its 2 GiB reserve. File-backed pages remain resident as used,
and RSS is not exclusive physical memory. This completes timing coverage, not
whole-model INT8 coverage, comparable quality, API qualification or speed wins.
[Raw run](docs/benchmark-chatglm-mapped-weights-v1.json) ·
[Loader tests, source cohorts, memory and restoration evidence](docs/benchmark-chatglm-mapped-verification-v1.json).

**Recorded main-table models: 21/21.** Original sweep snapshot: 2026-10-23T09:48:47.724248+00:00.
TinyLLAMA and Qwen2.5 now use the October 29 focused process cohorts shown at the top of this page, rather than their older sweep variants. TinyLLAMA still misses the combined prefill target; Qwen2.5 has a small median lead with trial ranges crossing the target. These backend results do not establish independent RKLLM quality equivalence. [TinyLLAMA](docs/benchmark-tinyllama-adaptive-layout-v1.json) · [Qwen2.5](docs/benchmark-qwen25-exp2-abba-v1.json).
The Qwen3.5-2B preferred row now uses the October 25 CPU sigmoid/pool follow-up.
Its 128-row CPU scan, parked head workers and pooled prefill sigmoid preserve
the prior INT8 output sequence across all twelve control/candidate requests.
[Updated model evidence](docs/benchmark-cpu-sigmoid-backend-model-v1.json).
† Dynamic per-token activation/per-channel weight INT8 projections, INT32 accumulation and FP32 host arithmetic. ‡ Qwen3 uses the October 29 fixed-stock profile: 12 measured requests across four processes, with all requests above the published prefill and decode targets. This profile requires the documented CPU/NPU performance governors, NPU IRQ placement and CPU6 idle setting; output quality versus RKLLM remains unqualified. Its local memory figure is omitted because this profile's table does not report one. [Profile and conditions](docs/benchmark-qwen3-stock-tuning-v1.json). Earlier Qwen3 CPU-token-input results remain in [their original cohort](docs/benchmark-qwen3-cpu-token-inputs-v1.json).
Qwen3 uses the preferred matched FP16 source where selected. Other checkpoint provenance is in the result record.
The learned-matrix coverage audit is separate; INT8 placement alone does not establish that every learned projection was quantized.
* Local memory is lifetime peak process RSS in MiB and is not the reference memory metric; no memory winner is declared.

Prefill proxies are `128000 / TTFT(ms)` and include first-token work.

| Model | Model Size | Prefill proxy(tok/s), RKLLM / oNPU INT8 |
| :--- | :---: | ---: |
| Qwen2 | 0.5B | 877.31 / **901.11** |
| MiniCPM4 | 0.5B | **946.12** / 720.13 |
| Qwen3 | 0.6B | 642.93 / **856.18‡** |
| TinyLLAMA | 1.1B | **524.74** / 496.20 |
| Qwen2.5 | 1.5B | 338.35 / **344.54** |
| RWKV7 | 1.5B | **201.78** / 65.20 |
| InternLM2 | 1.8B | **336.72** / 240.01 |
| Gemma2 | 2B | **213.90** / 179.38 |
| Gemma3n | 2B | 113.83 / **117.27** |
| TeleChat2 | 3B | **206.96** / 183.80 |
| Phi3 | 3.8B | **125.83** / 112.53 |
| MiniCPM3 | 4B | **90.88** / 15.72 |
| ChatGLM3 | 6B | **94.61** / 55.69 |
| Qwen3-VL | 2B | **333.66** / 174.82 |
| DeepSeekOCR | 3B(A570M) | **182.54** / 95.11 |
| Qwen3.5 | 0.8B | **217.78** / 145.12 |
| Qwen3.5 | 2B | **164.75** / 114.04 |
| Qwen3.5 | 4B | **61.87** / 47.80 |
| Gemma4 | E2B | **213.67** / 146.53 |
| LFM2.5-VL | 450M | **1245.86** / 791.37 |
| Spark-X2.5 | 1.7B | **331.41** / 222.69 |
| Spark-X2.6 | 4B | — / 84.99 |

† Qwen3 uses the October 29 existing-recipe prefill remeasurement: six requests per arm in balanced control/candidate/candidate/control order. Trial variability crosses the RKLLM reference. Its 38.39 tok/s decode value in the main table comes from the separate ready-CPU-token-input cohort, not this prefill-only remeasurement.
‡ Qwen3 uses the newer fixed-stock profile described above. The 667.79 tok/s existing-recipe and 655.35 tok/s rejected candidate measurements remain documented in the score-vector experiment section.

Pending timing rows: none. Candidate timing coverage is complete; comparable quality and qualified RKLLM wins remain open.
[Per-model runs, variants, ranges, failures and hashes](docs/benchmark-w8a8-results.json).
Vision/connector performance is excluded for text-only VLM decoder cases.

The October 26 CPU recurrent-scan follow-up updates the preferred Qwen3.5 timing variants to Qwen3.5/0.8B: 145.12 prefill proxy tok/s and 21.02 decode tok/s, Qwen3.5/2B: 114.04 prefill proxy tok/s and 11.13 decode tok/s, Qwen3.5/4B: 47.80 prefill proxy tok/s and 5.29 decode tok/s. Each uses the 128/64/512 protocol with three warmups and three measured requests. These candidates also include mapped CPU loading and current CPU attention placement; earlier-run differences do not isolate recurrent-scan unrolling. Output quality and full-model W8A8 coverage remain pending. [Run reports and restoration verification](docs/benchmark-cpu-gdn-models-v1-verification.json).

A diagnostic-only CPU-attention profile measured Gemma4/E2B at 6.46s TTFT without the hook and 4.11s with 35 CPU attention blocks; the attention pilot decoded at 0.36 tok/s. Gemma3n/2B used 30 CPU attention blocks and measured 9.83s TTFT and 0.33 tok/s decode. Each report contains only 8 generated tokens. Both attention candidates were below the 1.5 tok/s gate and honestly skipped the 64-token full sweep; these are not timing candidates. The native profile and selected generated C show whole-weight Q8 unpacking dominates CPU decode time. [Raw diagnostic profiles and source verification](docs/benchmark-gemma-attention-profile-verification-v1.json).

Gemma CPU attention v3 added full W8A8 timing candidates only for rows that completed 64 outputs and passed the candidate gate. Accepted: Gemma4/E2B TTFT 1.43s, prefill proxy 89.59 tok/s, decode 9.03 tok/s, Gemma3n/2B TTFT 6.93s, prefill proxy 18.47 tok/s, decode 6.20 tok/s. The 8-token stock/control and pilot profiles are diagnostics; the V2 token-changing run is rejected. Output quality and full-model coverage remain unqualified. [Run, pilot, rejection and pruning evidence](docs/benchmark-gemma-pruned-verification-v3.json).

Gemma host-kernel v4 reports are accepted only for candidates that completed 64 output tokens under the full 128/64/512 protocol. Accepted: Gemma4/E2B TTFT 0.87s, prefill proxy 146.95 tok/s, decode 9.05 tok/s, Gemma3n/2B TTFT 1.10s, prefill proxy 116.56 tok/s, decode 8.26 tok/s. The CPU GELU hook emitted one kernel per FFN block during 128-token prefill and none during decode. The CBUF command-buffer measurements are primitive-only and are not model speedups. Model output quality and full learned-matrix coverage remain unqualified; no general RKLLM win is claimed. [Full reports, stock-8 gate profiles, CBUF shapes and CPU evidence](docs/benchmark-gemma-host-kernels-verification-v4.json).
<!-- onpu-w8a8-sweep:end -->

Gemma decode v5 completed both full 128/64/512 candidates with three warmups and three measured requests: Gemma4/E2B: 146.53 prefill proxy tok/s, 9.58 decode tok/s; Gemma3n/2B: 117.27 prefill proxy tok/s, 8.57 decode tok/s. All twelve 64-token sequences match v4. Each phase uses the vector CPU softcap; the narrow K=4096 NPU decode case also passed signed-integer native replay regression. Historical comparisons do not isolate the two changes or clock drift. Comparable output quality remains unqualified. [Full reports and restoration verification](docs/benchmark-gemma-decode-verification-v5.json).

The SDK-derived INT8 prefill layout was also measured as a complete projection pipeline, including CPU quantization, transfers, NPU execution and FP32 output conversion. All 126 outputs across nine cases matched the independent reference bit for bit. One original shape improved by 6.9%; two were near parity, and six expanded widths were slower in every paired block. The option remains off by default and these operator measurements change no model timing row. [Full pipeline results and restoration evidence](../tinygrad/extra/rockchip/int8_sdk_prefill.md).

The SDK pipeline v2 repeat adds four-row CPU conversion and preserves 140 bitwise outputs across ten projection cases. Its 64/3072/2048 case is 9.7% faster than the current control; expanded widths and a two-call row split still lose overall. Broader dispatch remains disabled, and no model timing row changes. [V2 samples and profiles](../tinygrad/extra/rockchip/int8_sdk_prefill_v2_results.json).

Gemma2 host-v1 completes the full 128/64/512 comparison: current control 42.34/7.27 prefill-proxy/decode tok/s; capped CPU attention, vector GELU and final-logit softcap candidate 138.74/9.24. Both use three full warmups and three measured requests. Candidate sequences are stable across all six requests but first differ from control at output token 14 (29 of 64 positions differ). Comparable quality remains unqualified. CPU/DMC clocks vary and later candidate NPU samples end at 800 MHz versus 1 GHz for control; no fixed-clock win is claimed. Actual counts are 26 CPU attention calls per phase, 26 GELU calls in prefill and one final softcap per phase. Both normal workers were restored and answered fresh requests. [Full reports and evidence](docs/benchmark-gemma2-host-verification-v1.json).

Gemma2 fixed-prefix quality diagnostic: the throughput benchmark first emits EOS at output 8, before the free-sequence divergence at output 14. Under identical teacher-forced prefixes, the control reproduces all 64 benchmark predictions and the candidate reproduces the first divergence. On 128 held-out natural-text targets across 16 prefixes, sampled perplexity is 28.64 for control and 27.67 for candidate (ratio 0.96620), passing the declared local 1.02 regression limit. Distributions still differ: natural-target argmax agreement is 90.6%, mean probability total variation 0.09149 and maximum logit difference 6.0789. This is a local host-kernel regression check against existing INT8, not FP16 or RKLLM quality parity. A separate direct-logit profile attributes 31.9% of native prefill time and 89.7% of native decode time to NPU matmuls. Timing cells remain unchanged. [Raw-backed quality and profile evidence](docs/benchmark-gemma2-host-quality-v1.json).

Gemma2 natural-depth v1 removes zero padding from 2304 to 2560 input channels. Full 128/64/512 runs measure 140.29/9.79 prefill-proxy/decode tok/s versus 136.94/9.30 for the identically configured padded control. All twelve 64-token sequences match the prior host candidate. Packed weights fall from 2,829,582,336 to 2,614,099,968 bytes (205.5 MiB saved). Independent integer/FP32 checks and vocabulary-tail/partial-row native replay passed. Quality qualification against RKLLM and all-options API qualification remain open; clocks vary. [Full reports and restoration evidence](docs/benchmark-gemma2-natural-verification-v1.json).

Depth 2304 is now enabled by default after the natural-depth qualification above. The measured explicit value 1 and new unset default select the same branch; `ONPU_INT8_NATURAL2304=0` retains the padded control. The proof records the default-only source change and post-promotion CPU/static checks. [Implementation notes](../tinygrad/extra/rockchip/int8_natural2304.md).

Gemma2 fused-GELU v1 combines raw INT8 projection scaling, GELU and independent down-projection-group quantization. Full 128/64/512 runs measure 183.86/9.93 prefill-proxy/decode tok/s versus 140.37/9.93 for the current unfused control. All twelve 64-token sequences match the prior candidate; the native graph executes 26 fused pack kernels during prefill and none during decode. The option remains off by default pending API qualification. RKLLM quality and fixed-clock qualification remain open; timing cells are candidates only. [Full reports, exact payload/native checks and restoration](docs/benchmark-gemma2-gelu-verification-v1.json).

Gemma2 static-layout follow-up resolves raw GELU fragment locations before the channel loops. Complete pack timings improve 6.8% at the full Gemma2 shape with identical payload and scale bits. Full requests measure 179.38/9.93 prefill-proxy/decode tok/s versus 169.93/9.86 for this run's control; all twelve output sequences match the previous candidate. The table now records this latest candidate. Earlier 183.86 tok/s measurements are retained above; adaptive clocks and process state vary, so those rates are not adjusted by the new ratio. The static layout defaults to the measured option within the optional fused-GELU path. RKLLM quality, fixed-clock performance and full serving API qualification remain open. Both original sessions were restored and answered fresh requests. [Full reports, compiled-kernel selection and checks](docs/benchmark-gemma2-gelu-static-verification-v1.json).

RWKV7 native CPU scan v1 permits actual 128-token prefill chunks while preserving FP32 recurrent state. Full requests measure 67.20/12.73 prefill-proxy/decode tok/s versus 9.12/10.25 for the current one-token control. Prompt NPU submissions fall from 46,208 to 409; the candidate executes 24 native scan kernels in each captured phase. Each variant is stable across all six runs, but two output positions differ between them (outputs 23 and 57, token 629 becomes 630). The 20-test CPU suite includes independent matrix equations, complete small-model PyTorch references, cache/reset/generation and pooled state checks. Held-out quality, full learned-matrix coverage, model-maximum context and serving API remain unqualified; adaptive clocks vary. Both normal sessions were restored and answered fresh requests. [Raw reports, checks, differences and restoration](docs/benchmark-rwkv7-scan-verification-v1.json).

RWKV7 native-scan quality follow-up **fails the predeclared local 2% regression limit**: sampled perplexity on 128 held-out targets rises from 8.8876 to 9.1603 (ratio 1.03069, +3.07%). Natural-target argmax agreement is 93.75%, mean probability total variation 0.05772 and maximum absolute logit difference 4.58. The faster row remains an experimental timing candidate, not a quality-qualified result. The direct native decode recurrence matches the existing tinygrad FP32 recurrence bit for bit in a separate trained-geometry primitive check; batching and the surrounding model pipeline need further isolation. Profiling attributes only 12.5% of candidate prefill time to NPU calls; the CPU scan takes 416.5 ms and three large CPU kernels take 338.3, 288.0 and 165.0 ms. NPU calls account for 72.5% of native decode time. These component profiles do not change timing cells. Both normal sessions were restored and answered fresh requests. [Raw-vector comparison, failed gate and profiles](docs/benchmark-rwkv7-scan-quality-v1.json).

RWKV7 single-token isolation retains the native scan while disabling prefix batching. All 192 native logit-vector hashes match the batched native run exactly; the 3.07% local regression persists, so batching alone is not its cause. Generated generic recurrence code shows a different FP32 order for normalized-key projection. An ordered-normalization fix now passes the focused bitwise test, but full-model quality and timing are pending. The earlier quality-failing cohort is retained as historical evidence; the corrected follow-up now supplies the measured row. [Isolation, source identities and restored serving](docs/benchmark-rwkv7-single-quality-v1.json).

RWKV7 ordered-normalization and small-depth v2 supersede the earlier experimental row. Preserving the generic fused FP32 normalized-key reduction order restores exact agreement on all 192 fixed-prefix full logit vectors, including 128 held-out targets: sampled perplexity is 8.8876 on both paths, ratio 1.00000. This passes the unchanged local regression gate against existing INT8, rather than qualifying FP16 or RKLLM output quality. Matched corrected-scan timing control/candidate measure 66.89/66.75 prefill proxy tok/s and 12.65/13.10 decode tok/s. All twelve 64-token timing sequences match. Opt-in physical depths 64/128/256 reduce packed weights by 68.125 MiB; the planner default remains off. The candidate still trails RKLLM 201.78/13.82 tok/s. Both original workers were restored and answered fresh requests; the GPU worker stayed loaded. [Full reports, exact local quality and restoration](docs/benchmark-rwkv7-small-depth-verification-v2.json).

RWKV7 decode-worker follow-up keeps CPU workers awake between single-row scan calls and parks them at the end of each inference call. Matched control/candidate measure 13.05/14.51 decode tok/s and 65.77/65.20 prefill proxy tok/s. The captured scan sources confirm parked prefill in both variants, parked control decode and spinning candidate decode. All twelve 64-token timing sequences match. All 192 fixed-prefix logit dumps match each other and the prior local INT8 cohort byte for byte. The current decode rate exceeds the published RKLLM 13.82 tok/s; prefill remains below 201.78. This is a reported speed comparison with adaptive local clocks, rather than matched RKLLM output-quality or both-phase qualification. `ONPU_RWKV_DECODE_SPIN=1` remains opt-in/default-off. The earlier CPU-only proxy used substantially more CPU time for spinning workers; that is not a full-model power measurement. Both original serving sessions were restored and answered fresh requests; the GPU session stayed loaded. [Raw timings, callback sources, exact local quality and restoration](docs/benchmark-rwkv7-pool-spin-verification-v1.json).

A fresh API worker now selects the pinned RWKV7 INT8 recurrent configuration with 128-row prefill, smaller physical depths and spinning decode workers. Eighteen uncached greedy `/v1/chat/completions` requests (three warmups and three measured requests per workload) reproduce the independent backend reference, including usage, finish reason and NPU submission count. Each prompt has 128 tokens; replies stop naturally or reach their 64-token budget. The following API metrics use first visible text, so they remain separate from the canonical backend row.

| API workload | Prompt tokens | Completion tokens | Prefill proxy tok/s | Decode tok/s |
| :--- | ---: | ---: | ---: | ---: |
| counting_continuation | 128 | 39 | 65.06 | 14.44 |
| long_story | 128 | 64 | 65.94 | 14.30 |
| code | 128 | 64 | 65.23 | 14.03 |

This qualifies the checked local serving configuration at context 512, not maximum context or comparable RKLLM output quality. GPU-first defaults remain in place; both original serving sessions were restored and answered fresh requests. [HTTP responses, references, timings and restoration](docs/benchmark-rwkv7-api-verification-v1.json).

Component profiling identifies CPU work behind the two largest prefill gaps. NPU kernels account for 4.6% of MiniCPM3's recorded native prefill time and 0.3% of DeepSeekOCR's. MiniCPM3 spends 11.93 seconds in the generic per-head query projection, 10.23 seconds in latent QK dots, 4.03 seconds in value projection and 2.02 seconds in weighted-value reduction. Its query projection decodes Q8 weights inside dot products; static weight preparation and MLA attention kernels are direct experiment targets. DeepSeekOCR spends 19.63 seconds in expert gate/up matmuls and 13.17 seconds in expert down matmuls—98.8% of its recorded native prefill time. Those expert matmuls also account for 87.2% of native decode time. These are one-request diagnostic kernel timings after three full warmups, not new throughput rows or quality-qualified wins. The captured CPU sources and bindings support the attribution. Both normal serving models were restored and answered fresh requests. [Profiles, exact sources, output comparisons and restoration](docs/benchmark-mla-moe-profile-verification-v1.json).

MiniCPM3's opt-in CPU MLA table preparation measures 3.94/5.63 prefill proxy tok/s and 2.19/2.89 decode tok/s for matched control/candidate. Preparing 124 contiguous FP16 tables (155 MiB logical size) removes all 124 Q8-reading per-head projection programs in each captured phase; NPU packed weights stay identical. All twelve 64-token sequences match each other and the previous reference. CPU tests preserve decoded weight bits, projection outputs and small-model cached/reset/generation behavior. Full-model logits, held-out perplexity and comparable RKLLM quality remain unqualified. A stale packed-byte validation constant initially stopped the candidate; the unchanged control was independently revalidated against the current profile, and the candidate resumed under the same numerical sources in a separate lease. Original failed validation, corrected audit, raw timings, source evidence and both serving restorations are retained. [Matched results and correction](docs/benchmark-minicpm3-mla-weights-verification-v2.json).

MiniCPM3 CPU MLA attention v1 compares the matched weights-only control with an opt-in QK/PV candidate: 5.63/10.62 prefill proxy tok/s and 2.89/3.86 decode tok/s. Both variants prepare 124 CPU MLA FP16 tables and retain identical packed NPU weights; the candidate adds 62 QK and 62 PV CPU kernels in each phase while keeping generic softmax and the existing small/static fallback. All twelve generated 64-token sequences match across the pair, with CPU checks recorded separately. These are local regression checks, not model-logit, held-out, or RKLLM quality qualification. The candidate improves both measured phases against this control. The optional helper remains off by default. [Reports, CPU receipt and restored-serving evidence](docs/benchmark-minicpm3-mla-attention-verification-v1.json).

A fresh real RKLLM Qwen3-0.6B W8A8 GDB trace now covers the complete bounded 128-input/2-output pass: **2,657 submissions, 3,751 tasks and 367,308 register words**. Prefill contains 896 FP16-family matrix tasks and 896 DPU conversion tasks; decode contains 594 INT8-family matrix tasks and no FP16-family matrix tasks. The capture includes active per-core task ranges and matrix-address buffer bindings. This is register/scheduling evidence, with no debugger throughput or comparable-quality claim; existing model timing cells stay unchanged. Both original serving sessions were restored and answered fresh requests. [Full trace inventory and verification](docs/benchmark-rkllm-full-register-capture-v2.json).

### Grouped NPU MLA projections

The latest completed NPU experiment runs forty distinct head projections in one
submission across three cores. These are **CPU / grouped NPU milliseconds**;
lower is better. They include FP16 input conversion, layout, transfers, native
replay and output assembly, with immutable weights packed before measurement.

| Projection, all 40 heads | 1 row CPU / NPU (ms) | 17 rows CPU / NPU (ms) | 128 rows CPU / NPU (ms) |
|---|---:|---:|---:|
| Query latent, K64/N256 | 0.597 / **0.354** | 7.534 / **1.008** | 34.493 / **7.560** |
| Value, K256/N64 | 0.593 / **0.313** | 7.534 / **0.802** | 36.971 / **7.084** |

The grouped three-core path is 1.69–1.89× faster than CPU at one row and
4.56–5.22× at 128 rows. It wins all eight paired timing blocks for every tested
shape, and grouped outputs match separate NPU calls bit for bit. Clocks were
not fixed. These gains do not establish full-model throughput or RKLLM wins.
[Raw measurements](docs/benchmark-minicpm3-grouped-npu-operator-v1.json) ·
[Source, test and restoration verification](docs/benchmark-minicpm3-grouped-npu-operator-verification-v1.json).

The optional model hook now supports independent prefill/decode selection and
symbolic token counts. Its matched MiniCPM3 128-input/64-output/context-512 run
completed the control's three warmups and three measurements at 10.73 prefill
proxy tok/s and 4.03 decode tok/s. The following fixed-token logit diagnostic
failed with `cycle detected in assign graph`; the candidate was correctly not
run. A small NPU reproducer also fails for fixed-shape full-logit calls with a
fresh cache, while generation-matching symbolic token inputs pass. The trained
comparison still needs that diagnostic change and revalidation, so those control
timings do not replace the validated model row. The candidate combines INT8 dense projections
with FP16 MLA projections; comparable FP16/RKLLM output quality remains
unqualified. The model timing tables retain the completed CPU-MLA result above
until the grouped candidate completes inference and validation.

The v2 repeat fixes the diagnostic input form and validates the control's six
requests plus 32 fixed-token full-vocabulary logit vectors: **10.62 prefill proxy
tok/s and 3.98 decode tok/s**. The candidate prepares all 62 grouped FP16 hooks
but stops during its first warmup with `Rockchip matmul requires finite FP16
activations` in captured decode. No candidate timing request completes. A short
trained diagnostic subsequently found non-finite query output despite finite
inputs and a finite independent reference. Whole-buffer synchronization,
separate FP16 allocations and one-core execution did not resolve it; none is
accepted as a fix. The preferred timing cells remain unchanged.

The subsequent **grouped INT8 MLA v1** run completes three full warmups and
three measured requests per arm with the same 128-input/64-output/context-512
workload. Both variants retain CPU MLA attention and prepared FP16 source tables;
the candidate adds 62 grouped INT8 projection hooks, with 124 grouped programs
in each captured phase. CPU surrounding arithmetic remains FP32.

| MiniCPM3 variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) | Local numerical gate |
|---|---:|---:|---:|---|
| Matched CPU-MLA control | 12049.75 | 10.62 | 3.97 | Reference |
| Grouped INT8 candidate | 6450.59 | 19.84 | 4.81 | **Failed; rejected for promotion** |
| Published RKLLM reference | 1408.49 | 90.88 | 5.94 | Comparable quality unqualified |

The candidate is 1.87× faster in prefill and 1.21× faster in decode than this
control, but fails two unchanged local limits over 32 repeated-prompt teacher
targets: relative RMS logit error is **0.16412 versus a 0.01 limit**, and mean
probability total variation is **0.06455 versus a 0.02 limit**. The sampled
perplexity ratio is 0.96179 and passes its 1.02 limit; this does not override
the failed limits. Argmax agreement is 96.875%. This small diagnostic does not
qualify held-out, FP16 or RKLLM output quality. Candidate timing cells therefore
do not replace the preferred row. The INT8 hook remains opt-in. Both normal
serving models were restored with their options and answered fresh requests.
[Raw reports, unchanged thresholds and restored-serving evidence](docs/benchmark-minicpm3-mla-int8-verification-v1.json).

### Grouped INT8 residual v2

A matched 128-input/64-output/context-512 run completes three warmups and three
measured requests per arm. The candidate computes three scaled INT8 products,
`A0W0 + A1W0 + A0W1`, in one grouped submission per projection. Original source
tables remain FP16; CPU attention and surrounding FP32 arithmetic remain.

| MiniCPM3 variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Matched CPU-MLA control | 12027.73 | 10.64 | 3.98 |
| Residual INT8 candidate | 7901.40 | 16.20 | 4.41 |
| Published RKLLM reference | 1408.49 | 90.88 | 5.94 |

The local numerical gate **fails**: relative RMS logit error is
0.159280 (limit 0.01), mean probability total variation is
0.063834 (limit 0.02), and sampled perplexity ratio is
1.028491 (limit 1.02). Argmax agreement is
93.750%. These 32 repeated-prompt teacher targets are a local
regression check, not held-out, FP16 or RKLLM quality qualification. The preferred timing row is retained.

Candidate packed NPU weights are 4,236,083,200 bytes: 3,992,289,280 dense INT8
bytes plus 243,793,920 bytes for grouped residual products. Another 9,523,200
bytes hold CPU dequantization scales; these are outside the NPU weight count.
The option remains experimental and opt-in. Before the trained runs, CPU
checks, NPU changed-input/native-replay tests and complete small-model generation
passed. All candidate timing requests used the original 10 GiB soft and 11 GiB
hard memory limits. During the later non-greedy quality diagnostic, repeated
soft-limit throttling delayed compilation; the soft limit was raised to 10.5 GiB
while the hard limit and global 2 GiB reserve stayed unchanged. Saved timing
trials and medians were verified unchanged. Both original serving models were
restored and answered fresh requests.
[Raw reports, quality vectors, gates and serving evidence](docs/benchmark-minicpm3-mla-int8-residual-verification-v2.json).

A subsequent bitwise-invariant trace of all 124 query/value projections at the
last fixed-teacher decode position finds exact agreement with the independent
scaled INT8 recipe, including independently reconstructed packed weights and
scales. The largest RMS difference from the original CPU projection on identical
inputs is 0.2254%. This single-position check does not establish prefill or
earlier-cache correctness and does not override the failed whole-model gates.
[Actual projection snapshots and independent analysis](docs/benchmark-minicpm3-projection-trace-verification-v1.json).

### Query-only NPU MLA v1

A matched 128-input/64-output/context-512 run completes three warmups and three
measured requests per arm. The candidate computes three scaled INT8 products,
`A0W0 + A1W0 + A0W1`, only for query projections. Value projections keep the
original CPU dot with FP16 source tables and FP32 arithmetic. CPU attention
and other surrounding arithmetic remain FP32.

| MiniCPM3 variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Matched CPU-MLA control | 11918.53 | 10.74 | 4.03 |
| Query-only NPU candidate | 10082.89 | 12.69 | 4.27 |
| Published RKLLM reference | 1408.49 | 90.88 | 5.94 |

The local numerical gate **fails**: relative RMS logit error is
0.164209 (limit 0.01), mean probability total variation is
0.063534 (limit 0.02), and sampled perplexity ratio is
1.021769 (limit 1.02). Argmax agreement is
93.750%. These 32 repeated-prompt teacher targets are a local
regression check, not held-out, FP16 or RKLLM quality qualification. The preferred timing row is retained.

Candidate packed NPU weights are 4,114,186,240 bytes: 3,992,289,280 dense INT8
bytes plus 121,896,960 bytes for query residual products. Another 7,618,560
bytes hold CPU dequantization scales; these are outside the NPU weight count.
The option remains experimental and opt-in. Before the trained runs, CPU
checks, NPU changed-input/native-replay tests and complete small-model generation
passed. The run used 10 GiB soft/11 GiB hard limits and a 2 GiB board reserve. Both original serving models were restored and answered
fresh requests.
[Raw reports, quality vectors, gates and serving evidence](docs/benchmark-minicpm3-mla-query-only-verification-v1.json).

The historical residual raw reports have a stale FP16 suffix in their precision
description. Their `grouped_mla_precision=int8_residual`, integer ABI and checked
packed codes establish INT8 execution. New reports correct the suffix; prior
reports and their hashes are retained.

### CPU MLA causal work v1

A matched 128-input/64-output/context-512 run completes three warmups and three
measured requests per arm. Both arms use CPU MLA query/value projections,
prepared FP16 source tables, FP32 attention and generic softmax. The candidate
skips bounded masked QK outputs and zero-probability PV tails; unsafe inputs
retain the original full calculation. Single-row decode avoids the new scan.

| MiniCPM3 variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Matched CPU-MLA control | 11971.85 | 10.69 | 4.01 |
| Causal-work candidate | 10960.11 | 11.68 | 4.01 |
| Published RKLLM reference | 1408.49 | 90.88 | 5.94 |

The local numerical gate **passes**: relative RMS logit error is
0.000000 (limit 0.01), mean probability total variation is
0.000000 (limit 0.02), and sampled perplexity ratio is
1.000000 (limit 1.02). Argmax agreement is
100.000%. These 32 repeated-prompt teacher targets are a local
regression check, not held-out, FP16 or RKLLM quality qualification. The preferred timing row is retained.

Both arms retain the same 3,992,289,280 packed NPU bytes for dense INT8
projections, 124 CPU FP16 projection tables and 62 QK/PV kernels per phase.
No grouped NPU MLA projection is enabled. Exact bitwise final-logit agreement
is **required and passed**, in addition to the original numerical limits.
The control sets `ONPU_MLA_CAUSAL_WORK=0`, the candidate sets it to `1`.
The run used 10 GiB soft/11 GiB hard limits and a 2 GiB board reserve. Both original serving models were restored and answered fresh
requests. These local checks do not qualify RKLLM or held-out output quality.
[Raw reports, quality vectors, gates and serving evidence](docs/benchmark-minicpm3-cpu-mla-causal-verification-v1.json).

### DeepSeekOCR grouped CPU experts v1

Both arms use identical dense INT8 NPU options, deferred CPU weight realization,
128 input/64 output tokens, context 512, three warmups and three measured requests.
The earlier published 3.43/3.15 tok/s candidate used realized weights.
This new matched pair defers weight realization in both arms for the packed-weight
memory experiment; its original-expert control must not be presented as the
earlier baseline. Saved source shows raw Q8 blocks decoded inside the deferred
control expert loops; the candidate prepares packed FP16 weights once. Its gains
therefore include preparation as well as grouped execution. Only the candidate
enables grouped CPU expert projections. It groups selected
rows by expert, reuses packed FP16 weights and computes independent output
channels with the original sequential FP32 feature accumulation. Routing,
activations, shared experts and the final top-k weighted sum stay in the model.

| DeepSeekOCR variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Original CPU-expert control | 81829.83 | 1.56 | 1.47 |
| Grouped CPU-expert candidate | 1345.75 | 95.11 | 23.29 |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 |

The local qualification **passes**: exact bitwise agreement of 32 fixed-teacher full-vocabulary
vectors is **passed**, relative logit RMS is 0.000000 (limit 0.01),
mean probability total variation is 0.000000 (limit 0.02),
and sampled perplexity ratio is 1.000000 (limit 1.02).
Targets repeat the prompt; this is local regression against existing INT8 inference,
not held-out, FP16 or RKLLM quality qualification. The candidate becomes the preferred timing variant.

The checkpoint has one leading dense block and eleven MoE blocks. The candidate
executes all 33 grouped expert kernels in each captured phase; the control uses
the original expert matrices. Dense NPU weights remain 505,446,400 bytes in both
arms. Original expert tensors/layouts are retained, with a separate CPU packed
copy; local RSS includes preparation up to each measured request.
The run used 10 GiB soft/11 GiB hard limits, no swap and a global 2 GiB RAM reserve. Both original serving models were restored with their options and
answered fresh requests. The image encoder and connector are excluded.
[Raw reports, bitwise comparison, source hashes and restoration](docs/benchmark-deepseek-cpu-experts-verification-v1.json).

A subsequent diagnostic of the same configuration attributes 81.17% of recorded
prefill kernel time and 50.42% of decode time to grouped CPU experts. NPU
execution alone accounts for 4.46% and 39.24%; copies are separate. All four
profile-run sequences match the candidate. Instrumented component timings do
not replace the throughput row above. A CPU tile prototype improves the isolated
gate/up and down kernels by 8.9% and 8.5%, with twelve exact-bit routing/length
checks. Its subsequent integration passed exact model-logit checks but lost in
complete-model timing, as recorded below; the runtime option was removed.
[Current profile and restored serving](docs/benchmark-deepseek-cpu-experts-profile-verification-v1.json)
· [Prototype samples and generated sources](docs/benchmark-deepseek-cpu-experts-tile-prototype-v2.json).

### DeepSeekOCR eight-row expert tile v1

Both arms use grouped CPU experts with the same N16 packed FP16 tables,
dense INT8 NPU options, deferred weight realization and 505,446,400 dense NPU
weight bytes. Only `ONPU_EXPERT_TILE8=0/1` changes. The candidate reuses each
weight across up to eight prefill rows, computes eight independent outputs at a
time and separates full row tiles from tails. Each output retains sequential
FP32 feature accumulation. Capacity-one decode retains the original kernel.

| DeepSeekOCR tile variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Four-row control | 1295.03 | 98.84 | 24.71 |
| Eight-row candidate | 1320.75 | 96.91 | 23.83 |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 |

Each arm uses 128 input/64 output tokens, context 512, three complete warmups
and three measurements. All twelve generated sequences match the prior grouped
candidate. Local numerical regression **passes**: bitwise equality of 32
fixed-teacher full-vocabulary vectors is **passed**,
relative logit RMS is 0.000000, mean probability total
variation is 0.000000 and sampled perplexity ratio is
1.000000. The original 0.01/0.02/1.02 numerical
limits and the exact-bit requirement remain unchanged. Targets repeat the
prompt; held-out, FP16 and comparable RKLLM quality remain unqualified.

Preferred-row promotion requires local regression and both measured throughput
medians to improve over this matched control. The prior preferred timing variant is retained. CPU-only qualification
passes 49 tests with one hardware skip, mypy on 274 files and Ruff. A separate
native NPU model test passes changed-prefix and cached-decode equality before
either trained arm runs. The candidate captures 33 eight-row expert programs in
prefill and none in decode; both arms execute 33 expert programs in each phase.
The slower experimental tile option was removed after the comparison; the
runtime and tests were restored byte for byte to their previously qualified
four-row versions. Both original serving models were
restored with their options/settings and answered fresh requests. The image
encoder and connector are excluded. These measurements do not establish a
full VLM or certified RKLLM win.
[Reports, exact logits, source hashes and restoration](docs/benchmark-deepseek-expert-tile-verification-v1.json).

### Selected NPU experts and physical depths

These are operator experiments, not new model rows. The selected-expert INT8
runtime keeps an immutable atlas resident and validates routing IDs before
patching task weight addresses in ordinary and native replay. Raw integer
outputs match independent references, including changed IDs, duplicates and
signed extremes. It remains unselected by the app/model placement configuration.
Separate projections trail the prepared CPU kernels. A partial fused gate/up
trial measures 0.713 ms on NPU versus 0.767 ms on CPU (paired 1.076x), but its
subsequent prefill harness checked native execution too early. The corrected
warmup diagnosis and the completed prefill tests below supersede that harness
failure; no backend admission fix was needed.

`ONPU_INT8_EXPERT_DEPTHS=1` removes zero padding at depths 896 and 1280.
It remains off by default. The hardware tests cover exact raw integer results
with grouped 17/128-row execution and routed single-token execution. Padded
and natural complete decode projections match bit for bit for all six changed
inputs. Timing includes quantization, copies, routing/command cleaning,
submission, output scaling and native replay.

| Selected projection | Padded NPU median ms | Natural NPU median ms | Paired speedup |
|---|---:|---:|---:|
| E8, N1792, K1280, six routes | 0.685 | 0.628 | 1.134x |
| E8, N1280, K896, six routes | 0.435 | 0.403 | 1.081x |

Speedups are medians of paired block ratios, not ratios of separate medians.
The prefill probe uses 128 tokens, six selected experts per token and balanced
12-row groups across 64 experts, with combined gate/up width 1792 and depth
1280. Route-map creation is outside timing. The NPU side includes input gather,
quantization, padded work, copies and dequantization/scatter; the prepared CPU
kernel includes bucketing and uses a favorable combined-width layout.

| Rows allocated per expert | CPU median ms | NPU median ms | Paired NPU speedup |
|---|---:|---:|---:|
| 12 | 51.89 | 10.34 | 5.090x |
| 128 | 51.72 | 41.82 | 1.225x |

The 128-row case includes allocation/computation overhead for worst-case unique
top-k occupancy, but routing in this test is still balanced. All six routing/input
changes pass sampled independent integer/scaling checks and native execution.
Operator differences from original FP16 weights are about 1.1% relative RMS;
that is not a trained-model or RKLLM quality qualification. Efficient dynamic
expert row counts, runtime route-map creation, complete-model quality and
repeated full-model timing remain open. Both serving models were restored and
answered fresh requests. The preferred model rows and GPU baseline stay unchanged.
[Raw samples, raw integer tests, source hashes and restoration](docs/benchmark-expert-depths-prefill-verification-v1.json).

### Dynamic expert row counts

An experimental ABI-4 path accepts per-expert counts, validates all counts before
command mutation, skips empty experts, and builds three-core task chains from
qualified row-count templates. Ordinary and native replay update task counts and
subcore ranges. Exact integer tests cover zero/one/full and changing row counts,
sparse experts, invalid-count rejection and native-to-ordinary recovery. Twelve
changed balanced/skewed/mixed projection cases match fixed-NPU outputs bit for bit.

| Routing | Prepared CPU ms | Fixed NPU ms | Dynamic-count NPU ms | Paired speedup vs fixed NPU |
|---|---:|---:|---:|---:|
| balanced | 51.87 | 40.72 | 44.23 | 0.925x |
| skewed | 51.97 | 42.93 | 35.36 | 1.188x |
| mixed | 55.99 | 39.86 | 42.43 | 0.852x |

These 128-token/top-6/E64 operators still gather, quantize and copy full 128-row
buffers per expert. Dynamic NPU work alone does not consistently improve the
complete operator pipeline. Route-map/count construction is outside timing.
Compact active-route buffers remain necessary before model integration; no
preferred model row changes. All original serving options/settings were restored
and both models answered fresh requests.
[Raw samples, exact checks and restoration](docs/benchmark-bucketed-experts-verification-v1.json).

### Compact expert buffers

The experimental ABI-5 path packs expert input and output rows consecutively,
using prefix sums of validated counts. This 128-token/top-6 workload stores 768
routed rows rather than 8,192 fixed expert slots. Native scale guards and buffer
bounds use the compact row count; weights and task descriptors remain resident.
Twelve changed balanced/skewed/mixed projection cases match padded-NPU outputs
bit for bit. Raw hardware tests cover exact integers/task counts, aggregate
count bounds, recovery, and ordinary/native replay at depths 896 and 1280.
The fixed-slot ABI remains covered by its hardware suite.

| Routing | Prepared CPU ms | Fixed NPU ms | Dynamic slots NPU ms | Compact NPU ms | Paired compact/fixed speedup |
|---|---:|---:|---:|---:|---:|
| balanced | 48.91 | 39.34 | 40.98 | 10.77 | 3.575x |
| skewed | 50.79 | 49.68 | 39.12 | 6.54 | 7.491x |
| mixed | 56.40 | 40.35 | 42.46 | 11.03 | 3.660x |

Speedups are medians of paired block ratios, not ratios of separate medians.
The NPU timings include input gather, format-specific quantization/transfers,
command updates, matrix execution, expert-scale selection and output scatter.
Route-map/count construction is excluded. The prepared CPU combined-width
reference includes bucketing. All cases use E64, width 1792, depth 1280 and
128 tokens/top-6, with balanced, skewed and mixed maps. There are no steady NPU
allocations. Model placement, complete-model quality and repeated trained-model
timing still need qualification; the preferred model rows are unchanged.
Both serving models were restored and answered fresh requests.
[Raw samples, exact checks and restoration](docs/benchmark-compact-experts-verification-v1.json).

### Trained compact NPU experts v2

This paired DeepSeekOCR text-decoder run integrates runtime routing maps,
fused gate/up projections and down projections. Prefill builds stable compact
expert buckets inside inference; decode selects expert weight addresses at
runtime. The original expert tensors/state remain, with no duplicate packed
CPU expert copies on the candidate. Both arms use identical dense INT8 options,
natural depths 896/1280, deferred weight realization, context 512, 128 input/64
output tokens, three full warmups and three measured requests.

| Trained DeepSeekOCR path | TTFT ms | Prefill proxy tok/s | Decode tok/s | Peak RSS MiB |
|---|---:|---:|---:|---:|
| CPU FP16 expert control | 1303.63 | 98.19 | 25.60 | 8751.12 |
| Compact NPU INT8 expert candidate | 516.35 | 247.89 | 37.25 | 8073.47 |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 | 3066.66* |

The local numerical gate **fails**: relative logit RMS is
0.065740 (limit 0.01), mean probability total variation is
0.017152 (limit 0.02), and sampled perplexity ratio is
1.000309 (limit 1.02). All vectors must be finite.
This new expert path changes FP16 expert arithmetic to INT8, so bitwise equality
and equality to old CPU greedy IDs are diagnostics, not admission gates.
Measured candidate greedy sequences must be stable. The fixed 32 teacher targets
repeat the prompt; these are local regression checks, not held-out, FP16-source
or matched RKLLM quality qualification. The prior preferred model timing variant is retained.

Before either trained arm, a synthetic full model matches an independent CPU
INT8 expert recipe across changed prefixes, cache decode and resets. Expert
state keys/IDs are retained. Wide vocabulary tile/tail integer goldens at
N24576/N6400, K1280 pass six changed inputs each through native replay.
Candidate captures contain 22 NPU expert programs per phase and 22 runtime route
maps in prefill, none in decode; CPU expert kernels are replaced. The image
encoder/connector, application placement and broad API qualification are outside
this experiment. The feature remains off by default. Both serving models were
restored and answered fresh requests. Memory metrics differ from the RKLLM
reference methodology; no memory winner is declared.
[Full raw results, quality gates and source/restoration evidence](docs/benchmark-npu-experts-verification-v2.json).

### Precision-protected NPU experts v1

This paired DeepSeekOCR text-decoder run tests trace-guided precision protection.
Transformer blocks 1 and 2 retain CPU FP16 experts; block 3 retains its CPU
FP16 down projection. Other expert projections use compact/routed INT8 NPU
execution. Block 0 is the leading dense block. Prefill builds stable compact
expert buckets inside inference; decode selects expert weight addresses at
runtime. Original expert tensors/state remain. CPU packed weights are prepared only
for the preserved CPU parts; NPU-owned projections avoid duplicate CPU packs. Both arms use identical dense INT8 options,
natural depths 896/1280, deferred weight realization, context 512, 128 input/64
output tokens, three full warmups and three measured requests.

| Trained DeepSeekOCR path | TTFT ms | Prefill proxy tok/s | Decode tok/s | Peak RSS MiB |
|---|---:|---:|---:|---:|
| CPU FP16 expert control | 1274.64 | 100.42 | 25.78 | 8751.28 |
| Hybrid CPU/NPU expert candidate | 651.78 | 196.38 | 34.31 | 8423.65 |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 | 3066.66* |

The local numerical gate **fails**: relative logit RMS is
0.055437 (limit 0.01), mean probability total variation is
0.012442 (limit 0.02), and sampled perplexity ratio is
1.017578 (limit 1.02). All vectors must be finite.
This new expert path changes FP16 expert arithmetic to INT8, so bitwise equality
and equality to old CPU greedy IDs are diagnostics, not admission gates.
Measured candidate greedy sequences must be stable. The fixed 32 teacher targets
repeat the prompt; these are local regression checks, not held-out, FP16-source
or matched RKLLM quality qualification. The prior preferred model timing variant is retained.

Before either trained arm, two synthetic full-model modes match independent CPU
expert recipes (all INT8, and INT8 gate/up plus FP16 CPU down) across changed prefixes, cache decode and resets. Expert
state keys/IDs are retained. Wide vocabulary tile/tail integer goldens at
N24576/N6400, K1280 pass six changed inputs each through native replay.
Candidate captures contain 17 NPU expert programs per phase and 17 runtime route
maps in prefill, none in decode. Seven CPU expert kernels remain per phase.
The hybrid choice was fixed from the earlier layer trace; the finite/RMS/TV/PPL
gates are unchanged. The image
encoder/connector, application placement and broad API qualification are outside
this experiment. The feature remains off by default. Both serving models were
restored and answered fresh requests. Memory metrics differ from the RKLLM
reference methodology; no memory winner is declared.
[Full raw results, quality gates and source/restoration evidence](docs/benchmark-npu-experts-hybrid-verification-v1.json).

### Activation-residual NPU experts v5

This paired DeepSeekOCR text-decoder run adds a second activation product to recover INT8 activation precision. Each routed row shares its packed weight matrix across A0W0 and A1W0. Explicit fused dequantization matches the independent CPU recipe; no test tolerance or quality gate changes. The same trace-guided hybrid mask remains in place.
Transformer blocks 1 and 2 retain CPU FP16 experts; block 3 retains its CPU
FP16 down projection. Other expert projections use compact/routed INT8 NPU
execution. Block 0 is the leading dense block. Prefill builds stable compact
expert buckets inside inference; decode selects expert weight addresses at
runtime. Original expert tensors/state remain. CPU packed weights are prepared only
for the preserved CPU parts; NPU-owned projections avoid duplicate CPU packs. Both arms use identical dense INT8 options,
natural depths 896/1280, deferred weight realization, context 512, 128 input/64
output tokens, three full warmups and three measured requests.

| Trained DeepSeekOCR path | TTFT ms | Prefill proxy tok/s | Decode tok/s | Peak RSS MiB |
|---|---:|---:|---:|---:|
| CPU FP16 expert control | 1337.79 | 95.68 | 25.26 | 8749.05 |
| Activation-residual CPU/NPU expert candidate | 771.40 | 165.93 | 33.28 | 8458.89 |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 | 3066.66* |

The local numerical gate **fails**: relative logit RMS is
0.054358 (limit 0.01), mean probability total variation is
0.014354 (limit 0.02), and sampled perplexity ratio is
1.015918 (limit 1.02). All vectors must be finite.
This new expert path changes FP16 expert arithmetic to INT8, so bitwise equality
and equality to old CPU greedy IDs are diagnostics, not admission gates.
Measured candidate greedy sequences must be stable. The fixed 32 teacher targets
repeat the prompt; these are local regression checks, not held-out, FP16-source
or matched RKLLM quality qualification. The prior preferred model timing variant is retained.

Before either trained arm, four synthetic full-model modes match independent CPU
expert recipes (all INT8 and CPU-down hybrid, each with and without activation residuals) across changed prefixes, cache decode and resets. Two operator tests and nine subtests cover routed row pairs, compact chunking, exact integer products, changed routing/counts and native replay. Expert
state keys/IDs are retained. Wide vocabulary tile/tail integer goldens at
N24576/N6400, K1280 pass six changed inputs each through native replay.
Candidate captures contain 17 NPU expert programs per phase and 17 runtime route
maps in prefill, none in decode. Seven CPU expert kernels remain per phase.
The hybrid choice was fixed from the earlier layer trace; the finite/RMS/TV/PPL
gates are unchanged. The image
encoder/connector, application placement and broad API qualification are outside
this experiment. The feature remains off by default. Both serving models were
restored and answered fresh requests. Memory metrics differ from the RKLLM
reference methodology; no memory winner is declared.
[Full raw results, quality gates and source/restoration evidence](docs/benchmark-npu-experts-activation-verification-v5.json).

### Weight-residual NPU experts v1

This paired DeepSeekOCR text-decoder run computes A0W0+A1W0+A0W1 for
all eleven MoE blocks. The activation pair shares one W0 atlas; a separate W1
atlas stores quantized FP32 weight remainders. This uses two packed INT8 weight
representations, not the same storage cost as ordinary W8A8. Each projection
uses two NPU submissions and one ordered fused CPU dequantization. Original
expert tensors/state remain and CPU packed expert copies are skipped. Both arms
use identical dense INT8 options, natural depths 896/1280, deferred loading,
context 512, 128 input/64 output tokens, three warmups and three measured requests.

| Trained DeepSeekOCR path | TTFT ms | Prefill proxy tok/s | Decode tok/s | Peak RSS MiB |
|---|---:|---:|---:|---:|
| CPU FP16 expert control | 1272.10 | 100.62 | 25.63 | 8749.66 |
| Three-product NPU expert candidate | 791.17 | 161.79 | 27.10 | 11254.34 |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 | 3066.66* |

The local numerical gate **fails**: relative logit RMS is
0.055134 (limit 0.01), mean probability total variation is
0.016913 (limit 0.02), and sampled perplexity ratio is
0.996463 (limit 1.02). All vectors must be finite.
This new expert path changes FP16 expert arithmetic to INT8, so bitwise equality
and equality to old CPU greedy IDs are diagnostics, not admission gates.
Measured candidate greedy sequences must be stable. The fixed 32 teacher targets
repeat the prompt; these are local regression checks, not held-out, FP16-source
or matched RKLLM quality qualification. The prior preferred model timing variant is retained.

Before either trained arm, six synthetic model modes match independent CPU
integer recipes, including activation and weight residuals with/without CPU
down projections. Two operator tests and twelve subtests cover changed IDs/counts,
compact chunking, signed integer outputs and native replay. Wide vocabulary
integer checks cover N24576/N6400 at K1280 with six inputs each. Candidate captures
contain 44 NPU expert programs and 22 fused three-product dequantization kernels
per phase, with 22 runtime route maps in prefill and none in decode. CPU expert
kernels are replaced. Numerical gates and original synthetic tolerances remain
unchanged. The image encoder/connector, application placement and broad API
qualification are outside this experiment. The feature remains off by default.
Both original serving models were restored and answered fresh requests. Memory
metrics differ from the RKLLM reference methodology; no memory winner is declared.
The candidate initially stalled at its 11 GiB hard limit during warmup. The
recorded runtime adjustment raised it to 11.75 GiB before any measured request,
with a 2 GiB global reserve retained. The control retained its initial limit. The original run admission failed only
because its observer omitted indexed-device expert labels. A separate audit
recomputed 44 expert programs per phase from the preserved full labels and
re-ran the original validator with that derived field corrected. Raw reports
and the original failed admission remain unchanged in the evidence.
A separate CPU-only diagnostic evaluates blocks 3 and 4 using captured control
activations. Smoothing calibration uses prefix positions 0..63; evaluation uses
positions 64..127 and captured decode, not held-out text. Activation correction
approaches the W8 weight-error floor (roughly 0.5–0.8% projection RMS). Three
products reduce the tested projection errors to about 0.007–0.025%, but this does
not carry through to the trained logit gate above. The tested smoothing option
also worsens later-position errors and is not integrated. The discrepancy now
requires a trained intermediate trace; operator precision alone does not explain
it. [Diagnostic inputs, formulas, results and serving restoration](docs/benchmark-expert-quant-error-diagnostics-v2.json).

[Full raw results, quality gates and source/restoration evidence](docs/benchmark-npu-experts-weight-verification-v1.json).

### DeepSeekOCR six-row prefill tile v1

Both arms use grouped CPU experts with the same N16 packed FP16 tables,
dense INT8 NPU options, natural depths, deferred loading and 427,540,480 dense
NPU weight bytes. Only `--cpu-expert-prefill-rows 4/6` changes. The candidate
reuses each weight across six prefill rows, keeps sixteen independent output
channels and separates full six-row tiles from tails. Each output retains
sequential FP32 feature accumulation. Capacity-one decode keeps the original
four-row/sixteen-channel kernel; no extra packed weight copy is created.

| DeepSeekOCR tile variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Four-row control | 1202.03 | 106.49 | 25.84 |
| Six-row candidate | 1133.94 | 112.88 | 25.21 |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 |

Each arm uses 128 input/64 output tokens, context 512, three complete warmups
and three measurements. All twelve generated sequences match the prior grouped
candidate. Local numerical regression **passes**: bitwise equality of 32
fixed-teacher full-vocabulary vectors is **passed**,
relative logit RMS is 0.000000, mean probability total
variation is 0.000000 and sampled perplexity ratio is
1.000000. The original 0.01/0.02/1.02 numerical
limits and the exact-bit requirement remain unchanged. Targets repeat the
prompt; held-out, FP16 and comparable RKLLM quality remain unqualified.

Preferred-row promotion requires local regression and both measured throughput
medians to improve over this matched control. The prior preferred timing variant is retained. CPU-only qualification
passes 100 tests and 89 subtests, with twenty hardware skips, mypy on 278 files
and Ruff. Two actual-NPU model tests pass changed-prefix and cached-decode equality before
either trained arm runs. The candidate captures 33 six-row expert programs in
prefill and none in decode; both arms execute 33 expert programs in each phase.
The default remains four rows; six rows are opt-in. Both original serving models were
restored with their options/settings and answered fresh requests. The image
encoder and connector are excluded. These measurements do not establish a
full VLM or certified RKLLM win.
[Reports, exact logits, source hashes and restoration](docs/benchmark-deepseek-cpu-prefill-r6-verification-v1.json).

### DeepSeekOCR natural-text quality sample v1

This matched run uses 128 input/64 output tokens, context 512, three warmups
and three measured requests. Both arms have identical dense INT8 NPU options
and four-row CPU expert settings. The control uses FP16 CPU expert weights;
the candidate uses single-product INT8 NPU experts across all eleven MoE blocks.
These are measured candidate speeds, not certified RKLLM quality wins.

| DeepSeekOCR path | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| CPU expert control | 1233.78 | 103.75 | 26.61 |
| Fast NPU expert candidate | 478.80 | **267.33** | **37.38** |
| Published RKLLM reference | 701.21 | 182.54 | 31.60 |

Sixteen separate WikiText-2 test prefixes contribute 512 teacher-forced English
targets. The local point perplexity ratio is 0.996349;
the empirical 95% sequence-cluster bootstrap upper estimate is
1.011648, against the predeclared 1.02 limit.
These local PPL checks **pass**. Natural-vector relative RMS is
0.062065, mean probability TV is 0.046617, and
next-token argmax agreement is 94.73%; the distributions
and predicted token sequences are therefore not identical.
Every saved vector is independently checked finite, hashed and rescored. Repeating
the first sequence after the others reproduces all its logit bits in both arms.
The corpus region was not used for this path's calibration; pretraining overlap
is unknown, and this English-only sample does not qualify vision/OCR tasks.

The original 32-target numerical gate **still fails**:
relative logit RMS is 0.065740 (limit 0.01), mean TV
0.017152 (limit 0.02), and perplexity ratio
1.000309 (limit 1.02). Those gates were not changed.
No automatic promotion occurs; the preferred row and 68 accepted timing variants
remain unchanged. The control is the current mixed-precision decoder, not a full
FP16 model or a matched RKLLM export. The image encoder/connector is excluded.
Both original serving models were restored and answered fresh requests.
[Raw runs, independently rescored logits, uncertainty and restoration](docs/benchmark-deepseek-natural-quality-verification-v1.json).

### DeepSeekOCR decoded Q8_0 GGUF reference v1

This quality-only comparison scores 512 teacher-forced English targets from
16 separate WikiText-2 test prefixes against logits from the existing Q8_0
GGUF decoded to half/native-F32 values with CPU F32 expert projections. It is
not an original HF FP16 checkpoint, an RKLLM export, or a vision/OCR reference.

| Expert path vs decoded GGUF | Point PPL ratio | 95% sequence-bootstrap upper | Local 1.02 PPL gate |
|---|---:|---:|---|
| Mixed CPU FP16 experts control | 1.012232 | 1.023470 | fails |
| Single-product INT8 NPU experts candidate | 1.008536 | 1.018805 | passes |

The candidate passes this narrow PPL comparison; the control's bootstrap upper
estimate exceeds 1.02. Candidate vectors still differ from the decoded-GGUF
reference (relative RMS 0.063864, mean TV
0.046421). The unchanged fixed-32 gate also still fails on
relative RMS 0.065740 (limit 0.01). This English-only
sample has unknown pretraining overlap and does not certify HF FP16, RKLLM,
vision/OCR, or broad output quality. It causes no promotion or timing-row change;
the preferred timing variant remains `deepseek-cpu-experts-v1-candidate` and the canonical
inventory remains at 68 valid timing variants.
[Comparison and provenance](docs/benchmark-deepseek-gguf-reference-verification-v1.json).

### Small-row INT8 buffer-register experiment v2

This isolated test changes both ordered CNA `CBUF_CON0` writes in each task
to the captured SDK value `0xb1`; every other command-body word, input, weight
and integer operation is identical. All 66 checks across 11 three-core shapes
match independent INT32 matrix products exactly, including changed random,
all-127 and alternating -128/127 inputs. No timed call allocates NPU buffers.

| Path | Groups | Rows | Full output N | K | Paired control/candidate time |
|---|---:|---:|---:|---:|---:|
| Grouped | 6 | 1 | 1792 | 1280 | 0.999× |
| Grouped | 6 | 2 | 1792 | 1280 | 1.008× |
| Grouped | 64 | 12 | 1792 | 1280 | 1.001× |
| Grouped | 64 | 24 | 1792 | 1280 | 1.001× |
| Grouped | 6 | 1 | 1280 | 896 | 0.993× |
| Grouped | 6 | 2 | 1280 | 896 | 1.002× |
| Grouped | 64 | 12 | 1280 | 896 | 1.000× |
| Grouped | 64 | 24 | 1280 | 896 | 1.011× |
| Dense | 1 | 1 | 3072 | 3072 | 0.994× |
| Dense | 1 | 1 | 6144 | 3072 | 0.997× |
| Dense | 1 | 1 | 24576 | 3072 | 1.002× |

The paired medians range from 0.993× to 1.011×, with no material speedup established.
Ratios above 1 favor the candidate. Each case has three warmups and twelve
alternating ABBA/BAAB blocks (24 timed calls per arm). Timing includes the
prepared Python runtime call and blocking submission, excluding uploads,
output copies and reference calculation. These are primitive results, not
model timings or RKLLM speed wins; no product guard/default is changed. Both
original serving models were restored and answered fresh requests. The first
attempt stopped before matrix submission because its harness expected one
register write; v2 handles the two ordered writes in the actual recipe.
[Exact outputs, paired samples and restoration evidence](docs/benchmark-smallrow-cbuf-verification-v2.json).

### Qwen3 prefill worker-pool comparison v1

Four separate processes run in control/candidate/candidate/control order, each
with 128 input/64 output tokens, context 512, three full warmups and three
measured requests. The only option change is `--cpu-prefill-pool`.

| Process | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| 1-control | 275.90 | 463.93 | 37.43 |
| 2-candidate | 280.11 | 456.96 | 37.78 |
| 3-candidate | 285.08 | 449.00 | 37.90 |
| 4-control | 299.28 | 427.70 | 36.97 |

The candidate actually dispatches through the persistent pool. All 32 saved
full-vocabulary teacher vectors are bitwise equal across all four processes,
and all 24 generated sequences match. This is local numerical regression,
not HF FP16 or matched RKLLM quality qualification. The no-pool attention
kernel already creates temporary threads; this option replaces that dispatch
with persistent workers. The results do not establish a useful prefill gain
relative to the run-to-run variation, and the older preferred prefill result
is faster. The existing preferred row is retained; four valid measurement
variants are added (72 total). Source identities are recorded for each run,
so differences from the older source cohort are not attributed to this flag.
Both original serving models were restored and answered fresh API requests.
[Exact comparisons, samples and restoration](docs/benchmark-qwen3-prefill-pool-verification-v1.json).

### Qwen3 native component profile v1

This diagnostic profiles measured request index 3 after three warmups: one
128-token prefill and 63 decode transitions. Summed native profile rows are
280.696 ms for prefill (CPU 174.629 ms / 62.21%, NPU 80.177 ms / 28.56%, copies
25.889 ms / 9.22%) and 1548.924 ms for decode (CPU 196.768 ms / 12.70%, NPU
1329.906 ms / 85.86%, copies 22.250 ms / 1.44%). Prefill includes 28 CPU
attention calls (70.211 ms) and 28 FFN-pack calls (39.565 ms).

These instrumented component sums exclude Python dispatch and KV reset; they are
not whole-request wall time, throughput, or causal attribution. The profile is
diagnostic only and does not qualify timing or quality. All six generated
sequences match the existing Qwen3 reference. The original source hashes, native
call inventory and generated program sources are retained in the profile report.
Both original serving models were restored and answered fresh requests.
[Profile and restoration evidence](docs/benchmark-qwen3-native-profile-verification-v1.json).

### Qwen3 CPU kernel microbenchmarks v1

Separate CPU-only tests retained the current attention register tile (8 query
rows × 2 channel vectors). Six alternative tiles were slower on the tested
128-row prefill and partial-row cases. All 63 finite-input comparisons were
bitwise equal, and all 21 poisoned-future masking checks passed. Compiling the
identical captured FFN packing source with `-O3` instead of `-O2` passed three
bitwise input-pattern checks but measured 0.995× paired speedup. Neither change
is promoted. These are small CPU kernel experiments, not model-speed or quality
results; the model timing tables and 72 canonical variants remain unchanged.
[Samples and exact-output checks](docs/benchmark-qwen3-cpu-micro-verification-v1.json).

### Cache-from-DMA output collection v1

The six private, aligned output shapes (12 KiB–3 MiB) compare driver
FROM_DEVICE with EL0 CIVAC after a fresh blocking NPU submit. Initial CPU
zeroes are synchronized to the device before its first write. For each of six
changed inputs, the candidate CIVAC result is copied and checked against the
exact INT32 golden before the driver FROM_DEVICE cross-check; all 72 arm/input
checks pass. The CPU reads the previous output before each overwrite.

Median driver/CIVAC ratios by M×N×K are: 1×3072×1024 1.333×, 17×3072×1024 1.106×, 64×3072×1024 1.081×, 128×4096×1024 1.071×, 128×6144×1024 1.078×, 1×24576×1024 1.178×.
These times cover output synchronization plus CPU copying only. They do not
establish arbitrary-alias or dirty-mapping behavior, a general cache policy, or
model-level speed or quality. No canonical timing row or preferred variant is
changed. Both original serving models were restored and answered fresh requests.
[Raw probe and restoration evidence](docs/benchmark-cache-from-dma-verification-v1.json).

### Qwen3 native FROM_DEVICE invalidation ABBA v1

Four control/candidate/candidate/control processes vary only `ONPU_REPLAY_NATIVE_FROM=0/1`. Each uses 128 input/64 output tokens, context 512, three warmups and three measurements; the CPU prefill pool stays off.

| Process | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| 1-control | 278.73 | 459.22 | 38.10 |
| 2-candidate | 286.20 | 447.23 | 38.15 |
| 3-candidate | 273.34 | 468.28 | 38.02 |
| 4-control | 289.55 | 442.07 | 38.01 |

Aggregating the six measured requests per role gives:

| Role | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| Control | 284.14 | 450.48 | 38.0650 |
| Candidate | 277.10 | 461.93 | 38.0656 |

Prefill proxy here is 128000 divided by the aggregate median TTFT. These
measurements improve prefill versus the matched control, while decode is nearly
unchanged. They do not establish a fixed-clock or comparable-quality RKLLM win.

The candidate invalidates 84 prefill and 112 decode native output sites, leaving 63 and 7 driver FROM_DEVICE sites; control retains 147 and 119 driver sites. All four reports preserve the 32 teacher vectors bitwise and all 24 generated sequences. This is local regression, not HF FP16 or matched RKLLM quality qualification. Native FROM_DEVICE remains opt-in/default-off. The older preferred prefill result remains faster, so the preferred row is retained; four valid timing variants are added (76 total). Native invalidation and source records are descriptive evidence, not a general speed or correctness claim. Both original serving models were restored and answered fresh requests.
[Exact comparisons, per-run sources and restoration](docs/benchmark-qwen3-native-from-verification-v1.json).

### Qwen3 FP32 KV staging microbenchmark v1

A separate CPU-only prototype staged visible FP16 keys/values into FP32 once
per head before the existing query-block loops, retaining the scalar tail and
dot reduction order. Twelve changed-input/future-NaN comparisons were bitwise
equal. Paired control/candidate ratios were 1.003× at 128 rows/end 128,
0.992× at 128 rows/end 256, and 0.816× at 17 rows/end 62. This did not establish
a speed benefit, so no backend attention change is made. These are isolated
CPU kernel measurements, not model throughput or quality certification.
[Raw checks and paired samples](docs/benchmark-qwen3-attention-float-kv-v1.json).

### NPU prefill attention operator revisions v1–v4

These are synthetic complete-attention operator measurements. Each paired
median ratio is computed within its own revision from the raw ABBA blocks;
packing, uploads, grouped QK, causal softmax, grouped PV, output invalidation,
and unpack are included. Revisions used different clock, affinity, IRQ, and
measurement protocols, so ratios are not causal cross-revision comparisons.

| Revision | Layout / mode | Paired median control/candidate ratio |
|---|---|---:|
| v1 | original / single | 0.763× |
| v1 | original / residual | 0.476× |
| v2 | original / single | 0.560× |
| v2 | folded / single | 0.355× |
| v2 | folded / residual | 0.291× |
| v2 | original / residual | 0.301× |
| v3 | original / single | 0.785× |
| v3 | folded / single | 0.938× |
| v3 | folded_parallel / single | 1.102× |
| v3 | folded_parallel / residual | 0.776× |
| v3 | folded / residual | 0.459× |
| v3 | original / residual | 0.465× |
| v4 | blocked / single | 1.040× |
| v4 | sdk / single | 0.676× |
| v4 | sdk / residual | 0.479× |
| v4 | blocked / residual | 0.755× |

Revision 3's fastest ordinary case is folded-parallel single precision at
1.102×; its folded-parallel residual case is 0.776×. Under revision
4's common protocol, blocked/single is 1.040× while
SDK/single is 0.676×; residual blocked/SDK are
0.755×/0.479×. Three
captured SDK row-major FP16→FP32 matrix goldens (source task 28640) are finite;
maximum absolute error is 1.907e-6. This operator result does not add backend
API support for row-128 attention.

Dynamic RHS remains unsupported by current NativeReplay. These synthetic
operator results are not model timing, model quality, or an RKLLM win; they do
not change numerical backend code, defaults, or model timing rows. [Reports, validators,
and restoration evidence](docs/benchmark-npu-prefill-attention-op-series-verification-v1.json).

### Direct NPU/OpenCL FFN chain v1

This synthetic chain uses the Qwen3 projection shapes: NPU INT8 gate/up
(128×6144×1024), activation/quantization, and two NPU down projections
(64×1024×3072). DMA-buffer handles pass the large tensors directly between
NPU and OpenCL; only normal CPU allocations use host import. Blocking NPU
completion, explicit visibility synchronization, GPU completion, input uploads
and output reads are included. Six changed-input cases reproduce every FFN
payload byte and scale bit, and downstream NPU INT32 outputs match independent
CPU integer references.

Twenty-four timed calls per arm in twelve alternating ABBA blocks measure a
paired 1.499× speed ratio: separate median times are 6.916 ms for CPU packing
and 4.702 ms for the DMA-buffer GPU route. These are whole-chain operator
measurements, not trained-model throughput, task quality, or an RKLLM win.
No model timing row or default changes from this probe. Both original serving
models were restored and answered fresh API requests.
[Data visibility, exact checks and complete timing samples](docs/benchmark-npu-opencl-ffn-chain-verification-v1.json).

### DPU layout discovery and pipeline v1/v2

The discovery probe relocates only the source and destination addresses from the
captured 69-word DPU task. Two dtype hypotheses use two patterns each; FP16 reconstructs all
16,384 source elements in the native N16K32 order; FP32 is unsupported. Source
and destination buffers are independently guarded at 8 MiB.

Pipeline v1 preserved all 63,488 finite FP16 encodings but timed out in the consumer
chain before consumer validation or timings. Pipeline v2 uses the corrected PC
successor tail links: the CPU link decoder rejects the old chain and validates
1/2/16/32-task chains. The [Rockchip driver](https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c)
starts the first command, so hardware successor links are required. The completed v2 report passes finite packing checks and
12 consumer checks; its group-1/group-16 paired median CPU-control/pipeline ratios
are 0.899× and 0.948×.

Both ratios are below one: DPU packing loses to CPU packing for the measured
CPU-resident inputs, including finite checks, uploads, matmul, output invalidation
and CPU copying. Group 16 uses four persistent CPU workers for the control;
group 1 uses a single CPU. No softmax or complete attention operation is measured.
A device-resident input could have different costs, but that benefit is untested.
This recipe is not enabled in a model. No model timing rows or preferred variants
change. [Discovery, retained v1 failure, corrected v2 and
restoration proof](docs/benchmark-dpu-layout-series-verification-v1.json).

## Prefill throughput

These values are `128000 / TTFT(ms)` for 128 input tokens. They include the first output token's overhead;
they are proxies, not separately timed prefill kernels. The original eight-column table is unchanged.

<!-- onpu-prefill:start -->
| Model | Model Size | Prefill proxy(tok/s), RKLLM / oNPU |
| :--- | :---: | ---: |
| Qwen2 | 0.5B | **877.31** / 99.96 |
| MiniCPM4 | 0.5B | **946.12** / 101.36 |
| Qwen3 | 0.6B | **642.93** / 90.99 |
| TinyLLAMA | 1.1B | **524.74** / 46.78 |
| Qwen2.5 | 1.5B | **338.35** / 31.78 |
| RWKV7 | 1.5B | **201.78** / 0.80 |
| InternLM2 | 1.8B | **336.72** / 30.65 |
| Gemma2 | 2B | **213.90** / 21.94 |
| Gemma3n | 2B | **113.83** / 1.79 |
| TeleChat2 | 3B | **206.96** / 20.61 |
| Phi3 | 3.8B | **125.83** / 0.79 |
| MiniCPM3 | 4B | **90.88** / 0.33 |
| ChatGLM3 | 6B | **94.61** / 0.43 |
| Qwen3-VL | 2B | **333.66** / 32.44 |
| DeepSeekOCR | 3B(A570M) | **182.54** / 4.56 |
| Qwen3.5 | 0.8B | **217.78** / 46.79 |
| Qwen3.5 | 2B | **164.75** / 18.04 |
| Qwen3.5 | 4B | **61.87** / 5.46 |
| Gemma4 | E2B | **213.67** / 2.02 |
| LFM2.5-VL | 450M | **1245.86** / 8.87 |
| Spark-X2.5 | 1.7B | **331.41** / 31.41 |
| Spark-X2.6 | 4B | — / 0.79 |
<!-- onpu-prefill:end -->

## Additional models

[Spark-X2.6-4B](https://github.com/XHToken/Spark-X2.6) is included at the same 128-input / 64-output workload.
It uses the Spark2_5 architecture shared with X2.5. The checkpoint catalog pins the official Q8_0 GGUF revision and SHA256.
There is no published X2.6 row in the referenced RKLLM table; its reference cells remain unavailable.
Decoder checks pass against an independent PyTorch calculation on CPU and Mali OpenCL.
The tokenizer matches the official tokenizer on 92 test cases. The full X2.6 baseline is recorded below; its first four greedy tokens also match an independent streaming PyTorch decoder.
No winner is declared without a reference.

The runner supports `--gpu-no-weight-packing` to avoid retaining a second FP16 copy of the body weights.
This option is needed for the 4B FP16 candidate on a 16 GB board and is recorded in the result.
It preserves the same mathematical operations; the small-decoder checks cover both packing settings.

<!-- onpu-additional-benchmarks:start -->
| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :---------- | :--------: | :---: | :----: | :--------: | :------: | :------: | :--------: |
| Spark-X2.6 | 4B | — / FP16 | 128 | 64 | — / 161422.06 | — / 2.07 | — / 13190.66* |
| Spark-X2.6 | 1.7B | — / INT8† (experimental) | 128 | 64 | — / 540.37 | — / 14.55 | — / 5805.91* |
<!-- onpu-additional-benchmarks:end -->

### LFM3 candidates (unmeasured)

Liquid AI has released [LFM3](https://huggingface.co/blog/LiquidAI/lfm3).
The following additions are pending loader support and measurement, checked on 2026-10-28.
They are separate from the 21 RKLLM reference models and the measured Spark-X2.6 candidate.

| Model | Intended workload | Configured maximum context (tokens) | oNPU status | Prefill / decode (tok/s) |
| :--- | :--- | ---: | :--- | :---: |
| [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | Text generation | 262144 | Planned; loader compatibility unqualified | — / — |
| [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) | Image/video to text | 128000 | Planned; text and vision compatibility unqualified | — / — |

The official [text config](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/config.json)
has 24 layers: 13 gated short convolutions, five sliding-window attention layers and six global attention layers.
It specifies a seven-element convolution kernel, a 256-token sliding window, sandwich normalization,
and different RoPE settings for sliding and global attention. Existing LFM2 support does not establish LFM3 compatibility.
The [VL config](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/config.json)
also introduces a hybrid vision encoder and projector that need their own correctness checks.
The context values above describe the checkpoints; full-context operation on this board is untested.
The initial benchmark target remains 128 input / 64 output tokens on the GPU-first path, with NPU acceleration evaluated separately.
No RKLLM comparison or oNPU speed claim is available for these additions yet.

### Spark-X2.6-1.7B candidate (measured)

The official [Spark-X2.6-1.7B model card](https://huggingface.co/XHToken/Spark-X2.6-1.7B)
identifies this as a separate release following the 4B model. Its complete
[config at f18c032](https://huggingface.co/XHToken/Spark-X2.6-1.7B/blob/f18c032c7675f807c0fdffe281dbe124edba2605/config.json)
matches [Spark-X2.5-1.7B at 14d6e83](https://huggingface.co/XHToken/Spark-X2.5-1.7B/blob/14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8/config.json):
`spark2_5`, 28 layers, hidden size 2048, FFN size 6656, eight attention heads,
two KV heads, head dimension 256, GELU and a 131072-token vocabulary.
The official Q8_0 checkpoint is now downloaded and hash-verified at publisher
GGUF revision `cdc3d8893fba4ce0687f1b49ce1e18b29cae96d0`. Its tensor shapes,
quantization types, token vocabulary, merges and BOS/EOS IDs match the supported
X2.5-1.7B GGUF. Full-model inference and fixed-prefix logits were measured; comparable quality and serving behavior remain unqualified.
[Checkpoint receipt](docs/benchmark-spark-x26-17b-checkpoint-v1.json) ·
[GGUF metadata checks](docs/benchmark-spark-x26-17b-metadata-v1.json).

| Model | Intended workload | Configured maximum context (tokens) | oNPU status | Prefill / decode (tok/s) |
| :--- | :--- | ---: | :--- | :---: |
| [Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B) | Text generation | 1048576 | Measured; preferred additional variant | — / 236.87 |

This row does not reuse the measured 4B timings or the X2.5-1.7B results.
The initial target is 128 input / 64 output tokens on the GPU-first path, with
NPU acceleration measured separately. Full configured context on this board
and comparable RKLLM quality are untested; no RKLLM reference timing is available.


Four ABBA processes use the official X2.6-1.7B checkpoint, 128 inputs, 64 outputs and context 512. All 24 outputs match the fresh control reference, and four teacher arrays match bitwise. This checkpoint has no historical preferred timing or RKLLM reference.

| Role | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 741.393 | 172.65 | 14.392 |
| candidate | 6 | 539.846 | 237.10 | 14.557 |

The slower-TTFT candidate is raw-erf-gelu-models-v1-spark-x26-17b-2-candidate. Both candidate processes beat both matched controls in TTFT and decode; it is the preferred additional variant.

[Four reports and validation evidence](docs/benchmark-raw-erf-gelu-models-verification-v1.json).

## GPU tuning measurements

These compare **initial / optimized oNPU GPU runs**, with three full warmups and three measured runs each.
All 64 measured greedy IDs match their initial baselines. The main RKLLM table retains the initial baselines
for Gemma4, Gemma3n and ChatGLM3, whose candidates have faster prefill without a demonstrated both-phase improvement.
All three Qwen3.5 sizes improve both phases over their initial runs; their main-table rows use the latest memory-bounded bulk path.

| Model | Prefill proxy(tok/s), initial / optimized | Decode(tok/s), initial / optimized |
| :--- | ---: | ---: |
| Gemma4-E2B | 2.02 / **10.31** | **2.88** / 2.87 |
| Gemma3n-2B | 1.79 / **8.26** | **2.22** / 2.22 |
| ChatGLM3-6B | 0.43 / **3.61** | **1.51** / 1.39 |
| Qwen3.5-0.8B | 3.11 / **46.79** | 3.14 / **4.54** |
| Qwen3.5-2B | 1.26 / **18.04** | 1.27 / **2.40** |
| Qwen3.5-4B | 0.27 / **5.46** | 0.27 / **0.60** |

ChatGLM3 GPU clock snapshots varied between 800 MHz and 1 GHz in the optimized runs. Its decode change
is therefore not isolated from clock variation. [Full measurements and comparisons](docs/benchmark-gpu-tuning.json).

The Qwen3.5 bulk path processes each recurrent scan in one OpenCL kernel and prepares eligible projections.
For every size, the three measured TTFT and decode ranges are separated from the initial baseline at the same CPU affinity.
All six warmup/measured requests per model reproduce the initial 64 generated IDs. These are local GPU improvements;
comparable-quality RKLLM qualification and maximum-context validation remain pending.
The 0.8B worker also passes streaming, context-limit rejection, cancellation and identical greedy recovery checks.

The earlier packed 0.8B experiment reached 56.79 prefill tok/s at 2910.21 MiB steady allocation.
Removing recurrent packed copies gives 46.79 tok/s at 2169.96 MiB, with decode still 4.54 tok/s.
This is a measured memory/speed tradeoff. Seven recurrent/GEMM tests and serving-policy tests pass;
an import/order-sensitive existing attention-suite failure is recorded separately with its controlled runs.
[Current family measurements and checks](docs/benchmark-recurrent-family.json),
[earlier packed 0.8B measurements](docs/benchmark-recurrent-gpu.json).

## oNPU W8A8 CPU+NPU measurements (partial)

This earlier Qwen3 pilot is retained for provenance; see the new sweep section for current CPU+NPU measurements.
SmolVLM is listed separately because its vision path remains FP16. The [W8A8 coverage record](docs/benchmark-w8a8-coverage.json)
keeps the precision, workload, timing and quality evidence explicit. No FP16 GPU result is counted as a local W8A8 measurement.

128 input tokens, 64 generated tokens. Prefill proxy is `128000 / TTFT(ms)`; it includes first-token overhead.
This proxy is not an independently timed pure prefill kernel.

| Engine | TTFT(ms) ↓ | Prefill proxy(tok/s) ↑ | Decode(tok/s) ↑ |
| :--- | ---: | ---: | ---: |
| RKLLM W8A8, published | **199.09** | **642.93** | 32.91 |
| oNPU CPU+NPU W8A8, experimental† | 223.77 | 572.02 | **34.25†** |
| oNPU OpenCL GPU, latest baseline | 1406.79 | 90.99 | 7.96 |

Latest GPU results use three warmups and three measured runs at context 512.
The previous six-run GPU comparison measured a 17.1% decode improvement with bitwise-equal held-out logits;
neither comparison establishes an RKLLM win.

† The CPU+NPU pilot uses dynamic per-token activation/per-channel weight INT8, INT32 accumulation and FP32 host arithmetic.
Three measured runs after three warmups. Its decode median is 4.1% above the published RKLLM number, but TTFT is slower.
This experimental path remains disabled in serving; its matched level-0 SDK speed/output-quality comparison is unfinished.

A separate 256-position quality experiment measured perplexity ratios **1.04464 unscaled** and **1.02194 scaled**
against the GPU FP16 path; both exceed the existing 1.02 limit. This is not an SDK quality comparison,
and its configuration has not been tied to the timing pilot above. [Evidence and scope](docs/benchmark-w8a8-coverage.json).

A native output-copy borrowing experiment passed six replay tests, including bitwise comparisons, input rebinding,
returned tensors, multiple readers, partial writes and scratch release. Its paired full-model runs used five warmups
and three measured runs per setting, with all 64 generated IDs matching. No eligible Qwen3 output-copy sites were
eliminated, so the small timing differences are treated as variation. The experiment defaults to disabled.
[CPU/NPU tuning measurements and limits](docs/benchmark-cpu-tuning.json).

## Multimodal: SmolVLM-256M

| model | Stage | RK3588 RKLLM(w8a8) | oNPU |
| :--- | :---: | :---: | :---: |
| SmolVLM-256M | Img-encoder(512*512) | **842ms** | 1230.36ms† |
| | Prefill(len=128) | **77.3ms (1655.89 tok/s)** | 99.10ms (1291.58 tok/s)† |
| | Decode | 78 tokens/s | **105.74 tokens/s†** |

† Local CPU+NPU baseline: FP16 vision, INT8 text projections, 512-position allocation, three warmups and three measured runs, with 64 generated tokens.
The native C generation loop is enabled. Numerical leaders remain provisional until a matched SDK quality check passes.
[Multimodal measurement details](docs/benchmark-mm-results.json). Earlier long-context results use a different workload/reference recipe.

## Qualification and work order

- [x] Record all 21 published RK3588 rows without substituting measurements from different workloads.
- [x] Check access to the 21 candidate checkpoint configurations and record immutable revisions.
- [x] Benchmark the additional Spark-X2.6-4B checkpoint with the same workload.
- [x] Download the candidate checkpoints, implement the required text-decoder paths, and fill all 21 local baseline rows from complete model runs.
- [ ] Complete reproducible level-0 SDK exports for the available Qwen3 and SmolVLM checkpoints.
- [ ] Compare identical token inputs, context allocation, output count and host clock policies, with repeated warm runs.
- [ ] Check comparable output quality against the actual W8A8 reference before promoting an INT8 path.
- [ ] Measure and publish oNPU CPU+NPU W8A8 results for all 21 models; the FP16 GPU table does not complete this sweep.
- [ ] Optimize and measure both phases, starting with Qwen3-0.6B; qualify each additional model separately.
- [x] Record the [oMLX feature comparison](docs/omlx-feature-comparison.json) from current source and served interfaces.
- [x] Verify the earlier frozen V31/V38 DFlash2 API cohorts at context 8,192 with an actual 8,176-token prompt, sampled cancellation and exact recovery.
- [ ] Requalify the current shared-Q8/no-feature-history layout near 8K with actual proposals, matched ordinary decoding, cancellation/recovery and boundary rejection (V47 produced 16 tokens, then stopped on draft-cache invalidation after adaptive fallback; the optional retention path remains unqualified).
- [ ] Qualify model-maximum context, improve unprofitable speculation, and extend trained checkpoint/EAGLE4 coverage.

A qualified speed win requires at least three warmups and three measured runs per engine, matched provenance,
and the held-out quality gate. The current pair runner additionally requires separated timing ranges in both phases.
Model access or architecture gaps remain explicit rather than being counted as performance wins.

## Evidence

- [Checkpoint catalog and access results](docs/benchmark-catalog.json)
- [Spark-X2.5/X2.6 support and verification](docs/benchmark-spark-support.json)
- [RWKV7 support, tests and remaining limits](docs/benchmark-rwkv-support.json)
- [TeleChat2 source, conversion and tokenizer checks](docs/benchmark-telechat-support.json)
- [MiniCPM3 decoder, scaling and layout checks](docs/benchmark-minicpm3-support.json)
- [ChatGLM3 projections, tokenizer and schema checks](docs/benchmark-chatglm3-support.json)
- [Gemma4-E2B decoder, tokenizer and schema checks](docs/benchmark-gemma4-support.json)
- [Gemma3n-E2B AltUp/LAUREL decoder and schema checks](docs/benchmark-gemma3n-support.json)
- [Streamed GGUF loading checks](docs/benchmark-loading-support.json)
- [DeepSeekOCR text decoder and tokenizer checks](docs/benchmark-deepseek-ocr-support.json)
- [Independent full-model Spark reference checker](../tinygrad/extra/rockchip/check_spark_reference.py)
- [Measured local baselines](docs/benchmark-results.json)
- [Baseline table updater](../tinygrad/extra/rockchip/benchmark_report.py)
- [Coverage and model-specific gaps](../tinygrad/docs/rockchip-rkllm-coverage.json)
- [Recorded results and verification](../tinygrad/docs/rockchip-rkllm-parity-results.json)
- [Matched comparison runner](../tinygrad/extra/rockchip/benchmark_rkllm_pair.py)
- Local raw pilot: `/home/orangepi/vlm-bench/parity/qwen512-cpu-npu-pilot.json`
- [API and long-context verification](VERIFICATION.md)

<!-- onpu-cpu-scan-quality:start -->
## Qwen3.5-0.8B CPU scan: local quality check

On a fixed 256-position WikiText-2 test subset at context 512, conditional perplexity was
10.58231 for local GPU FP16, 10.55392 for serial INT8, and 10.67492 for INT8 with CPU scan.
The ratios versus local FP16 are 0.99732 and 1.00875; both pass the unchanged 1.02 limit.
This reference uses FP16 inference from the same Q8_0 checkpoint. RKLLM quality parity remains unqualified.

The scan still fails the prior INT8 token-identity check, so its timing is reported separately
and does not replace the established main-table row. Timings use three full warmups and three
128-input/64-output measured requests at context 512. All measured samples are retained.

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms), RKLLM / scan | Tokens/s, RKLLM / scan | memory(MB), RKLLM / scan |
| :--- | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| Qwen3.5 | 0.8B | w8a8 / INT8 projections | 128 | 64 | **587.74** / 1551.13 | **27.05** / 17.32 | 1039.66 / 2702.41* |

Prefill proxy: **217.78** / 82.52 tok/s. Candidate measured TTFT ranged from 1545.99 to 6424.62 ms.
Local memory is peak process RSS in MiB and differs from the reference memory metric.
The scan remains an opt-in backend experiment; the normal API worker keeps it off.
[Hashes, matched configuration, quality results and failed identity gate](docs/benchmark-qwen35-cpu-scan-quality.json).
<!-- onpu-cpu-scan-quality:end -->

<!-- onpu-gpu-q8-experiment:start -->
## Qwen3.5-0.8B GPU Q8 storage comparison

All three GPU-only variants passed the same prior 64-token GPU output check, with three full warmups and three measured 128-input/64-output requests at context 512.
Q8_0 weights are decoded to FP16 operands; activations remain FP16/FP32. These are not W8A8 activation-quantized measurements.

| GPU variant | TTFT(ms) | Prefill proxy(tok/s) | Decode(tok/s) | Peak RSS(MiB) |
| :--- | ---: | ---: | ---: | ---: |
| fp16-control | 2737.97 | 46.75 | 4.56 | 2440.53 |
| q8-head | 2721.83 | 47.03 | 4.78 | 2711.59 |
| q8-body-head | 2446.49 | 52.32 | 8.49 | 2263.01 |

The Q8 body/head variant improves decode over the FP16 control, but remains below the published RKLLM 27.05 tok/s and the experimental CPU-scan result of 17.32 tok/s.
Defaults are unchanged. [Complete metrics, dispatch counts and hashes](docs/benchmark-qwen35-gpu-q8.json).
<!-- onpu-gpu-q8-experiment:end -->

<!-- onpu-gpu-q8-tile-followup:start -->
## Qwen3.5-0.8B GPU tile follow-up

This is a separate GPU-only follow-up cohort. The previous Q8 body/head row below is retained from the earlier cohort;
the new Q8 body/head run and fresh FP16 control are from this tile cohort. These are separate cohorts, not a paired before/after run.
Q8_0 weights use FP16 runtime operands. This is a storage-format experiment, not W8A8 activation quantization or a W8A8 quality claim.
The tile settings remain opt-in; normal API defaults are unchanged. No API or SDK quality qualification is claimed.

| GPU variant | Cohort | TTFT(ms) | Prefill proxy(tok/s) | Decode(tok/s) | Peak RSS(MiB) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Q8 body/head (previous) | earlier GPU-Q8 cohort | 2446.49 | 52.32 | 8.4874 | 2263.01 |
| FP16 recurrent control (fresh) | tile follow-up | 2733.67 | 46.82 | 4.5407 | 2446.98 |
| Q8 body/head (tuned) | tile follow-up | **2019.20** | **63.39** | **8.5327** | **2262.86** |

The new rows use three full warmups and three measured 128-input/64-output requests at context 512.
Per-run values, ranges, report hashes, numerical source hashes, and serving receipt are recorded in
[the GPU Q8 experiment record](docs/benchmark-qwen35-gpu-q8.json).
<!-- onpu-gpu-q8-tile-followup:end -->

<!-- onpu-speculative-api-tiled-v38:start -->
## Current tiled DFlash2 API qualification (V38)

The two-row tiled FP32 GEMV source is now qualified within this pinned
Qwen3.5-2B/DFlash2 OpenCL API scope. V38 passed 36 short off/force greedy
requests with three warmups and three measured runs per workload and mode, plus
27 freshly executed probability comparisons with zero measured error and total
variation. The actual 8,176-input plus 16-output boundary passed in off, force
and auto modes. Sampled temperature-0.6 cancellation and exact forced greedy
recovery passed. Both original workers were restored and answered fresh replies.
The normal server was not restarted.

| Workload | Off wall median (s) | Forced speculative wall median (s) |
|---|---:|---:|
| counting | 10.40 | **7.41** |
| short_code | 8.23 | **7.65** |
| explanation | **10.97** | 13.34 |

Forced speculation was slower on explanation. These are scoped same-cohort
request timings, not a general speed claim. The long boundary requests are
single functional trials measured in minutes; forced speculation was slower.

V38 regression accounting is 351 methods plus 156 subtests. The context tested
was 8,192; model maximum 262,144, general speed/quality and RKLLM quality remain
unqualified. See [README](README.md) for the long boundary route outcomes.

[V38 API, tiled-source pins, probability audit and restoration evidence](docs/speculative-api-tiled-v38.json).

A subsequent exact-FP32 prefill operator experiment compared five tile variants
with the existing kernel on real FFN gate/down weights, both packed and row-major,
at 27, 36 and 128 valid rows. Two GPU tests and 150 subtests passed. Most variants
were slower; swapping row/column workgroup axes changed row-major medians by only
about 1–2%. One packed 36-row gate projection fell from 16.97 to 15.63 ms, while
the 128-row cases showed no broad gain. The production kernels remain unchanged.
[Complete operator timings and restoration evidence](docs/speculative-prefill-fp32-tiles-v1.json).

The newer opt-in cross-request prefix cache has no qualified API or general-speed
timing row. Its first cold-replay comparison failed a strict recurrent-state
check; the [failed gate](docs/speculative-prefix-reuse-core-v1.json) is preserved.
A [matched-seed diagnostic](docs/speculative-prefix-reuse-matched-v4.json) then
repeated one short continuation six times: all 42 state buffers and common probe
logits matched ordinary cached continuation bit for bit, and 18 temperature
comparisons had zero probability error and total variation. Cold replay differed
equally from both cached paths. The diagnostic timing summaries are in that
record; ordinary state alignment is excluded from its timer, so the three rows
do not establish a like-for-like speed comparison.

The [V40 API bridge](docs/speculative-prefix-reuse-api-v40.json) verified ten
short requests plus cache-enabled sampled cancellation and greedy recovery.
Forced continuation reused 31 of 55 prompt tokens and matched ordinary cached
output; cached automatic requests stayed ordinary, and cache disabling reported
zero reuse. The experimental loader enables this cache. App checks passed 301
methods and 21 subtests, plus full Ruff. The original controller failed on an
obsolete validator alias; separate corrected validation and an independent
audit verified the unchanged raw evidence. Both original workers answered
fresh requests after restoration. These functional cases add no RKLLM timing
win or new full-context qualification. V38 above records its earlier frozen source.
<!-- onpu-speculative-api-tiled-v38:end -->

<!-- onpu-long-prefix-api-v41:start -->
## Long cached conversation with retained draft features (V41)

Pinned Qwen3.5-2B/DFlash2 OpenCL worker, configured context 8,192. Each completed
request below emitted one token; these are functional cases from one conversation.
The prefill proxy is uncached input tokens divided by time to first output and
includes first-output work. Sustained decode throughput was not measured.

| Request | Prompt tokens | Cached tokens | New input tokens | Prefill proxy (new tok/s) | First output (s) | Request wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| Cold seed | 8,048 | 0 | 8,048 | 19.23 | 418.58 | 422.91 |
| Forced cached follow-up | 8,081 | 8,049 | 32 | 16.13 | 1.98 | 5.27 |
| Cold recovery after cancellation | 8,114 | 0 | 8,114 | 18.03 | 450.07 | 452.03 |
| Automatic cached boundary | 8,191 | 8,115 | 76 | 14.52 | 5.23 | 5.29 |

The forced follow-up returned the expected next number. Sampled cancellation at
temperature 0.6 preserved the request on a wrong-ID stop, stopped on the matching
ID, emitted `[DONE]` and returned the worker to ready. Cold greedy recovery
returned the expected answer. The final automatic request used ordinary cached
decoding and reached the exact 8,191-input plus one-output budget; a subsequent
8,191+2 request was rejected with HTTP 400 before generation, preserving its
completed metrics. Both original workers were restored and answered fresh replies.

This qualifies the tested conversation and cached ordinary boundary. Exact
forced 8,191+1 execution, complete long sampled generation, model maximum
262,144, broad quality and matched RKLLM performance remain unqualified. Cold
first-answer latency remains the main usability problem.
[Complete evidence, per-request metrics and source/restoration provenance](docs/speculative-prefix-reuse-long-api-v41.json).
<!-- onpu-long-prefix-api-v41:end -->

<!-- onpu-draft-feature-history-component-v1:start -->
## Prepared draft feature-history memory option

This component experiment used the actual pinned DFlash2 weights and real shared
target embedding/LM head, with six synthetic target-feature taps. It did not run
the full target model or API. Across 67 appends totaling 8,192 rows, K/V matched
bit for bit. Proposal logits, hidden outputs and candidate IDs matched at five
prefix positions; sampled proposals matched at temperatures 0.6, 1 and 2.

| Component metric | Feature-retaining control | Append-only candidate |
|---|---:|---:|
| Retained full projected history at 8,192 rows | 33,554,432 B | 0 B |
| Native observation-copy bytes across the stream | 2,406,746,112 | 1,284,069,376 |
| Native observation-copy commands | 1,152 | 1,024 |

The byte counts cover retained history and native copies only. Observation
scratch still exists; total RSS, complete-model speed and model-maximum fit
were not qualified. CPU checks covered 24 unique backend methods plus two
subtests, 301 app methods plus 21 subtests, mypy and Ruff. Both original workers
answered fresh replies after the GPU-only lease; the NPU worker stayed loaded.
The later [V42 API check](docs/speculative-draft-feature-history-api-v42.json)
enabled the append-only layout in the pinned experimental loader. Ten short
requests plus sampled cancellation/recovery passed with zero retained-history
bytes reported and correct cached-prefix counts. The V41 long-context timing
table above predates activation; V42 adds no standardized speed result or new
long-context/model-maximum qualification.
[Component records, source pins and restoration evidence](docs/speculative-draft-feature-history-component-v1.json).
<!-- onpu-draft-feature-history-component-v1:end -->

<!-- onpu-speculative-api-routing-v37:start -->
## Prior-source DFlash2 request cost routing (V37)

The pinned Qwen3.5-2B/DFlash2 OpenCL profile passed 36 short `off`/`force` requests,
27 fresh target-probability comparisons with zero measured difference, the actual
8,176-input plus 16-output boundary, sampled scoped cancellation and exact recovery.
Q8_0 checkpoint storage supplies FP16 runtime weights; these are GPU measurements.
Each short workload uses three warmups and three measured trials per mode.
`force` bypasses the upfront router and retains measured per-block fallback.
Bold identifies the better observed median independently for each metric.

| Workload | Engine | n (warmup/measured) | Wall (s) | TTFT (ms) | Prefill (tokens/s, proxy) | Decode (tokens/s) | Total (s) |
|---|---|---|---|---|---|---|---|
| counting | ordinary | 3 / 3 | 10.266 | **1716.888** | **17.473** | 3.397 | 10.253 |
| counting | forced speculative | 3 / 3 | **8.532** | 1776.696 | 16.885 | **4.155** | **8.519** |
| short_code | ordinary | 3 / 3 | **8.126** | **3133.180** | **11.490** | 3.414 | **8.113** |
| short_code | forced speculative | 3 / 3 | 8.417 | 3200.879 | 11.247 | **4.585** | 8.401 |
| explanation | ordinary | 3 / 3 | **10.812** | **1712.109** | **15.770** | **3.412** | **10.799** |
| explanation | forced speculative | 3 / 3 | 13.817 | 1779.055 | 15.177 | 2.582 | 13.803 |

Scoped short trials only. This table makes no RKLLM comparison, general speedup claim,
or causal packed-versus-unpacked full-model claim.

At the actual 8,176-input plus 16-output boundary, single functional request
wall times were 452.06 seconds for `off`, 484.53 for `force`, and 463.65 for
`auto` selecting ordinary. Forced cancellation took 480.52 seconds and exact
forced greedy recovery took 483.23 seconds. These minutes-long trials verify
functionality and routing; they are not repeated speed comparisons.

Separate `auto` checks selected speculation for short counting and ordinary for
the 8,176-token prompt, both with exact reference replies. Estimated long draft
setup was 8.907 seconds versus 4.373 seconds of calibrated ordinary decoding
after the first token. These estimates are heuristics, not runtime bounds.

Long rows below are single functional trials with 8,176 input and 16 output tokens.

| Request | Actual mode | Wall (s) | TTFT (s) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---|---:|---:|---:|---:|
| off | ordinary | 456.46 | 451.54 | 18.11 | 3.08 |
| force | dflash2-experimental | 486.90 | 478.87 | 17.07 | 1.88 |
| auto | ordinary | 465.91 | 461.04 | 17.73 | 3.11 |
| recovery (force) | dflash2-experimental | 484.62 | 478.22 | 17.10 | 2.36 |

The auto request reported `draft_setup_cost` and used ordinary decoding. It did
not beat the explicit ordinary control in this single trial. The sampled force
request ignored the wrong cancellation ID, stopped for the matching ID and
returned to ready state; its subsequent forced greedy recovery matched exactly.
Both original normal models were restored and answered fresh requests. The normal
server was not restarted. General speed, model maximum 262,144 and RKLLM quality
remain unqualified.

[V37 API, routing, probability and restoration evidence](docs/speculative-api-routing-v37.json).
<!-- onpu-speculative-api-routing-v37:end -->

<!-- onpu-speculative-api-half-decode-v36:start -->
## Prior trained DFlash2 API with selective FP16 decode loads (V36)

The frozen V36 Qwen3.5-2B/DFlash2 OpenCL profile passed 36 short greedy
requests: three warmups and three measured trials per engine and prompt, with
alternating measured order. All response views matched the frozen references.
Both engines use the same bounded FP16 prefill and selective FP16 decode loads.
The Q8_0 checkpoint executes with FP16 weights; these are not W8A8 results.

Prefill is input tokens/first-text time. Decode is the server-reported
(completion tokens minus one)/(last-text time minus first-text time), which
can exclude computation after the last visible text. Complete-request wall time
includes that work. Bold marks the better observed median per metric.

| Prompt | Input tokens | API output tokens | Ordinary prefill (tok/s) | DFlash2 prefill (tok/s) | Ordinary decode (tok/s) | DFlash2 decode (tok/s) | Ordinary request (s) | DFlash2 request (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 30 | 29 | 17.31 | **17.34** | 3.39 | **4.19** | 10.30 | **8.45** |
| short_code | 36 | 17 | **11.45** | 11.45 | 3.39 | **4.65** | **8.17** | 8.29 |
| explanation | 27 | 32 | **15.77** | 15.63 | **3.38** | 2.58 | **10.89** | 13.76 |

Counting favors speculation on complete-request time. Code has a higher reported
decode rate but is slightly slower as a complete request; explanation also loses.
These are same-cohort ordinary/speculative measurements, not a causal speed
comparison with V31 or the V33 target-core experiment.

All 27 probability comparisons were freshly executed on this source/profile
with zero measured probability error and total variation. The exact 8,176-input
+ 16-output API boundary passed for ordinary and speculative execution; both
matched the frozen response.

| Mode | Prefill proxy (tok/s) | Decode (tok/s) | Request (s) |
| :--- | ---: | ---: | ---: |
| ordinary | **20.04** | **3.26** | **412.54** |
| speculative | 19.04 | 1.93 | 437.28 |

The long rows are single uncached functional trials. Speculation was slower;
cold first-answer latency remains minutes and does not meet the speed goal.
Sampled SSE at temperature 0.6 passed wrong-ID preservation, matching-ID stop,
stream completion and exact greedy recovery. Recovery took 496.96 seconds.

V34 regression checks passed 297 methods plus 77 subtests, whole-repository
Ruff and mypy over 262 source files. A short-lived CPU subprocess prepared the
real tokenizer references; the existing memory admission threshold was unchanged.
The runner terminated successfully, restored both original models with exact
options/settings/selection, and verified fresh replies. The normal server was
not restarted. Model maximum 262,144, broader pairs, general speed/quality and
RKLLM parity remain unqualified.

[Frozen V36 API, fresh probability audit and restoration evidence](docs/speculative-api-half-decode-v36.json).
<!-- onpu-speculative-api-half-decode-v36:end -->

<!-- onpu-half-decode-target-v33:start -->
## Trained target core: selective FP16 decode loads (V33)

A paired Qwen3.5-2B experiment at configured context 8,192 compared the existing
FP16 projection path with wider loads that preserve every half product and FP32
sum order. The same resident weights served 36 requests: three warmups and three
alternating measured trials per mode and prompt, with eight generated tokens.
The decode rate below is seven synchronized continuation forwards divided by
their summed duration; between-step fingerprint readbacks are excluded. There
is no draft or HTTP request in this experiment. These are not W8A8 results.

| Prompt | Input tokens | Legacy prefill (tok/s) | Vector-load prefill (tok/s) | Legacy decode (tok/s) | Vector-load decode (tok/s) | Decode ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 30 | 17.23 | 17.25 | 2.30 | **3.08** | 1.34× |
| short_code | 36 | 11.21 | 11.27 | 2.31 | **3.06** | 1.33× |
| explanation | 27 | 15.30 | 15.07 | 2.32 | **3.11** | 1.34× |

Prefill uses the same path in both modes and is a control; no prefill gain is
claimed. Tokens, full logits, hidden output, six feature taps, and all 42 valid
recurrent/KV state tensors matched exactly. Full and partial verifier cases
passed at captured capacities 3 and 8.

The integrated dispatcher uses only the measured complete projection shapes,
row capacities and Mali-LODX r0p0 / driver 2.1. The eight-row wide projection
keeps its legacy path because wider loads regressed it. Other shapes and
devices retain the existing fallback. The integrated native GPU tests compare
against original scalar projections, including dynamic rows and NaN padding.

[V34 regression checks](docs/speculative-half-decode-preflight-v34.json) passed
297 test methods plus 77 subtests, whole-repository Ruff and mypy over 262 source
files. Both normal models were restored and answered fresh requests.

V38 above qualifies the integrated tiled source within its tested API scope.
V37 below remains a separate historical routing/API cohort for its frozen source;
V31 remains historical for its own frozen source.
Model maximum, general speculative speed and RKLLM speed/quality remain unqualified.

[Paired trained-target measurements and provenance](docs/speculative-half-decode-target-v33.json).
<!-- onpu-half-decode-target-v33:end -->

<!-- onpu-speculative-api-half-prefill-v31:start -->
## Prior-source DFlash2 API with bounded FP16 prefill (V31)

The frozen V31 Qwen3.5-2B/DFlash2 OpenCL profile passed 36 short greedy
requests using three warmups and three alternating measured trials per engine
and prompt. All responses matched frozen references. Both engines use bounded
FP16 prefill. The Q8_0 checkpoint runs with FP16 weights; embeddings enter FP32
and recurrent projections preserve FP16 products with FP32 sums. These are
not W8A8 results. Prefill is input tokens/TTFT, including first-token work;
decode is the server-reported rate. Bold marks the better observed median
independently per metric.

| Prompt | Input tokens | API output tokens | Ordinary prefill (tok/s) | DFlash2 prefill (tok/s) | Ordinary decode (tok/s) | DFlash2 decode (tok/s) | Ordinary request (s) | DFlash2 request (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 30 | 29 | 17.01 | **17.31** | 2.50 | **3.59** | 13.38 | **9.56** |
| short_code | 36 | 17 | 11.33 | **11.44** | 2.50 | **4.00** | 10.98 | **9.61** |
| explanation | 27 | 32 | **15.76** | 14.92 | **2.50** | 2.01 | **14.15** | 17.23 |

Counting and code favor speculation; explanation remains slower. This is a
same-cohort ordinary/speculative comparison, not a causal timing comparison
with V25 or the separate option-off/on V30 target-core experiment.

All 27 probability comparisons were freshly executed on the frozen V31 source
and worker profile, with zero measured difference. The run also passed the
exact 8,176-input + 16-output API boundary. The new ordinary response matched
the frozen previously qualified response; speculation and post-cancellation
recovery matched it exactly.

| Mode | Prefill proxy (tok/s) | Decode (tok/s) | Request (s) |
| :--- | ---: | ---: | ---: |
| ordinary | **17.79** | **2.32** | **466.02** |
| speculative | 16.99 | 1.66 | 490.27 |

The long rows are single uncached functional trials; speculation was slower.
Sampled SSE at temperature 0.6 passed wrong-ID preservation, matching-ID stop,
stream completion, return to ready and greedy recovery. The recovery request
took 490.29 seconds. No previous probability data or timing was transferred;
the old boundary response was used only as a regression gold.

V29 preflight passed 38 CPU methods plus 24 subtests, four native GPU methods
plus 15 subtests, and 255 app methods plus 17 subtests, whole-repository Ruff,
and mypy over 262 source files. Both original models were restored with exact
options, selection and settings and answered fresh requests. The normal server
was not restarted. Model-maximum context 262,144, general speed/quality and
RKLLM parity remain unqualified.

[V31 frozen-source API evidence, fresh audit and restoration proof](docs/speculative-api-half-prefill-v31.json).
<!-- onpu-speculative-api-half-prefill-v31:end -->

<!-- onpu-bounded-half-prefill-v30:start -->
## Trained target core: bounded FP16 prefill

The pinned Qwen3.5-2B target now has an optional prepared FP16 projection path
that limits work to valid prompt rows. It regenerates the current compiler's
shader during setup, requires qualified source fingerprints and preserves the
shader body and arithmetic. The core default is off; the pinned experimental
DFlash2 app profile enables it. It adds no packed weight copy.

This paired core test used context 8,192, actual prompts of 27–36 tokens and
eight greedy output tokens, with three warmups and three alternating measured
trials per mode and workload. Weights run in FP16 after loading the Q8_0
checkpoint; these are not W8A8 results. Prefill includes the first greedy token.
Continuation is seven one-token calls after that anchor. Bold marks the better
observed median independently per metric.

| Workload | Input tokens | Output tokens | Legacy prefill (s) | Bounded prefill (s) | Legacy prefill tok/s | Bounded prefill tok/s | Legacy continuation tok/s | Bounded continuation tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counting | 30 | 8 | 3.595 | **1.792** | 8.345 | **16.739** | **2.331** | 2.308 |
| short_code | 36 | 8 | 4.404 | **3.239** | 8.174 | **11.114** | 2.317 | **2.324** |
| explanation | 27 | 8 | 3.589 | **1.818** | 7.523 | **14.850** | **2.316** | 2.301 |

Prefill medians improved 2.01× for counting, 1.36× for code and 1.97× for
explanation. All frozen eight-token sequences matched. Prefill logits, hidden
states and six target taps were bit-identical, and 42 persistent cache/state
buffers matched after prefill and after seven continuations. Full 128-row,
partial 112-row and position-128 append checks also passed. Every measured
optimized prefill used 54 bounded projections; continuation used none.
The decode kernels are unchanged; their small timing differences do not
establish a decode optimization.

Preflight passed 38 CPU methods and 24 subtests, four native GPU methods and
15 subtests, and 255 app methods and 17 subtests, plus whole-repository Ruff
and mypy over 262 source files. The NPU model stayed resident; observed
available memory stayed above 2.19 GiB. Both original models were restored
with exact options/settings/selection and answered fresh requests.

This target-only cohort has no draft or HTTP layer and remains the separate
causal option-off/on result. V38 above qualifies the integrated tiled API within
its tested scope; V37 and V31 remain historical source cohorts. V30 itself does
not establish model-maximum context or RKLLM speed/quality parity.

[Trained-target measurements, fingerprints and restoration](docs/speculative-half-prefill-target-v30.json),
[integrated regression checks](docs/speculative-half-prefill-preflight-v29.json),
[FP16 operator/source proof](docs/speculative-half-prefill-grid-v1.json).
<!-- onpu-bounded-half-prefill-v30:end -->

<!-- onpu-target-prefill-profile-v28:start -->
## DFlash2 target prefill: bottleneck and kernel experiments

The pinned Qwen3.5-2B target processed the frozen 8,176-token prompt at context
8,192 in 64 prepared chunks. The sum of chunk durations was 416.78 seconds.
This diagnostic excludes draft loading and observation, profiles five chunks,
and does not measure complete API latency or a speed improvement.

| Prompt position | Input rows | Chunk wall (s) | Named linear kernels (GPU s) | Attention (GPU s) | Recurrent scan (GPU s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 128 | 6.01 | 3.18 | 0.019 | 0.063 |
| 3,968 | 128 | 6.41 | 3.19 | 0.409 | 0.065 |
| 7,936 | 128 | 7.26 | 3.27 | 0.865 | 0.067 |

Named linear kernels account for 46–54% of GPU event time in the four full
128-row samples. Generated/other kernels remain separately classified. The
largest named group is the packed FFN projection, with 48 calls per chunk;
the recurrent scan accounts for about 1%. The target starts in FP32, while
recurrent blocks cast their projection inputs to FP16.

Two isolated experiments used the same trained 6,144 × 2,048 FP16 weight and
128-row FP32 activations. These are not W8A8 measurements. Each cohort rotates
timing order, with three measured appearances at every position per mode.
Bold marks the lowest observed median within that cohort.

| Local-memory experiment: 4 warmup + 12 measured trial sets | Median kernel latency (ms) |
| :--- | ---: |
| Current kernel | **32.73** |
| Activation tile 32 | 42.61 |
| Activation tile 64 | 42.22 |
| Activation tile 128 | 43.16 |

All 48 candidate outputs matched baseline FP32 bits. These slower variants
were rejected.

| Register experiment: 5 warmup + 15 measured trial sets | Median kernel latency (ms) |
| :--- | ---: |
| Current kernel | 32.78 |
| 8 rows/thread, 32 columns/group | **29.84** |
| 16 rows/thread, 32 columns/group | 651.35 |
| 4 rows/thread, 64 columns/group | 33.28 |
| 8 rows/thread, 64 columns/group | 29.86 |

All 80 candidate outputs matched baseline FP32 bits. The selected register
variant reduced this isolated kernel's median latency by 8.97%, with no extra
packed weight storage. Follow-up symbolic-row experiments confirmed short and
remainder-row regressions; adaptive branches erased the full-chunk advantage.
The register variant was rejected for default use. The separate bounded FP16
path above preserves the current compiler body and changes only the launch.
Both original models answered fresh requests
after each owned run; the NPU stayed resident during the operator experiments.

[Target diagnostic and provenance](docs/speculative-target-prefill-v28.json),
[rejected local-memory experiment](docs/speculative-packed-prefill-x-tile-v1.json),
[register experiment](docs/speculative-packed-prefill-register-v1.json),
[symbolic/partial-row rejection](docs/speculative-packed-prefill-adaptive-v3.json).
<!-- onpu-target-prefill-profile-v28:end -->

<!-- onpu-speculative-api-deferred-v25:start -->
## Historical DFlash2 API: deferred fallback delivery (V25)

These V25 measurements refer to their frozen source cohort, including its
explicitly reused V24 probability component. V31 above independently qualifies
its frozen source with fresh probability, boundary, cancellation and recovery
checks. The timings below remain historical.

The pinned Qwen3.5-2B/DFlash2 profile passed 36 short greedy requests at configured context 8,192, using three warmups and three alternating measured trials per mode and prompt. Responses matched frozen references. Weights run in FP16 and the target decoder starts with FP32 activations; these are not W8A8 results. Bold marks the better observed median independently per metric. Prefill is input tokens/TTFT, including first-token work; decode is the server-reported rate.

| Prompt | Input tokens | API output tokens | Ordinary prefill (tok/s) | DFlash2 prefill (tok/s) | Ordinary decode (tok/s) | DFlash2 decode (tok/s) | Ordinary request (s) | DFlash2 request (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 30 | 29 | **8.49** | 8.34 | 2.50 | **3.51** | 15.15 | **11.61** |
| short_code | 36 | 17 | 8.17 | **8.33** | 2.50 | **3.99** | 11.21 | **10.32** |
| explanation | 27 | 32 | 7.52 | **7.64** | **2.50** | 2.01 | **16.02** | 18.97 |

The explanation trials used 28 fallback tokens, 27 fallback target calls and two native state snapshots across two verification blocks. Deferred delivery skips the unused final forward and fallback state copies. Its final token can remain pending, while the cache marker names exactly the processed valid prefix. This option defaults off in the core and is enabled in the pinned experimental app profile. Explanation remains slower than ordinary decoding.

The run also passed the exact 8,176-input + 16-output boundary, sampled SSE cancellation at temperature 0.6 with wrong-ID preservation and matching-ID stop, and exact greedy recovery.

| Mode | Prefill proxy (tok/s) | Decode (tok/s) | Request (s) |
| :--- | ---: | ---: | ---: |
| Ordinary | **19.15** | **2.34** | **433.36** |
| DFlash2 | 17.78 | 1.67 | 468.97 |

These long rows are single uncached functional trials; speculation was slower. No general speed, model-maximum 262,144, RKLLM parity or causal timing improvement over an earlier source cohort is qualified.

The 27 zero-difference probability comparisons are explicitly reused from V24's successful subtest after matching numerical, worker and speculation sources. V24's subsequent API load returned HTTP 400 before worker startup; its API failure remains recorded. V25 independently completed the HTTP cohort. No earlier timing values were transferred. Both original models were restored with exact options, selection and settings and answered fresh requests; the normal server was not restarted.

[Scoped API qualification, explicit evidence composition, source fingerprints, work counters and restoration proof](docs/speculative-api-deferred-v25.json).
<!-- onpu-speculative-api-deferred-v25:end -->

<!-- onpu-speculative-api-fc-v21:start -->
## Historical DFlash2 API with selectively packed draft input projection (V21)

The pinned Qwen3.5-2B/DFlash2 pair passed 36 short greedy requests at configured context 8,192. Each prompt used three warmups and three alternating measured trials per mode; responses matched frozen references. Target weights run in FP16, the target decoder starts with FP32 activations, and the trained draft selectively packs its input projection. These are not W8A8 measurements.

Bold marks the better measured median independently for each metric. Prefill is input tokens/TTFT, including first-token work; decode is the server-reported rate. API output-token accounting can differ from raw generated IDs at EOS.

| Prompt | Input tokens | API output tokens | Ordinary prefill (tok/s) | DFlash2 prefill (tok/s) | Ordinary decode (tok/s) | DFlash2 decode (tok/s) | Ordinary request (s) | DFlash2 request (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 30 | 29 | **8.45** | 8.44 | 2.51 | **3.55** | 15.13 | **11.45** |
| short_code | 36 | 17 | **8.36** | 8.30 | 2.51 | **4.04** | 11.10 | **10.28** |
| explanation | 27 | 32 | **7.67** | 7.60 | **2.51** | 1.94 | **15.89** | 19.55 |

Counting and short code were faster with speculation; explanation was slower despite switching to ordinary continuation after two unprofitable blocks. The experiment does not isolate a causal packed-versus-unpacked complete-request speedup.

The same run passed the exact 8,176-input + 16-output boundary, sampled SSE at temperature 0.6 with wrong-ID preservation and matching-ID cancellation, and exact greedy recovery. All 27 worker-shaped probability comparisons matched with zero measured difference.

| Mode | Prefill proxy (tok/s) | Decode (tok/s) | Request (s) |
| :--- | ---: | ---: | ---: |
| Ordinary | **18.69** | **2.35** | **443.86** |
| DFlash2 | 17.71 | 1.60 | 471.17 |

These long rows are single uncached functional trials, not repeated performance qualification. Speculation was about 6% slower on this request. Selective draft packing adds 48 MiB to admission. Both original serving models were restored with their exact options, selection and settings and answered fresh requests; the normal server was not restarted. Model-maximum context 262,144, broader checkpoint quality and RKLLM parity remain unqualified.

[Current scoped API qualification, source fingerprints, regression checks and restoration proof](docs/speculative-api-fc-v21.json).
<!-- onpu-speculative-api-fc-v21:end -->

<!-- onpu-fp32-workgroup-operator-v1:start -->
## FP32 projection workgroup experiment

A trained 6,144 × 2,048 FP16-weight projection was tested with FP32 inputs and outputs, matching the verifier's activation path. Each row count used four warmup trial sets and twelve measured sets. Four modes rotated equally through every execution position; timings include the operator call and OpenCL synchronization.

| Input rows | Current 256 threads (ms) | 64 threads (ms) | 128 threads (ms) | 256-thread control (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.928 | 1.926 | **1.901** | 1.922 |
| 3 | **1.947** | 2.014 | 1.986 | 1.983 |
| 8 | 5.578 | **5.532** | 5.576 | 5.581 |

Differences were small and mixed, so the application retains its existing kernel. All 48 trial sets matched the original output bits across the candidate modes; first and last sets also matched independent scalar GEMV calls. Three GPU test methods and 72 subtests passed, together with Ruff and mypy. No extra weight storage was used.

This is an isolated projection result; the full-vocabulary head and complete requests were not benchmarked here. Both original models were restored with fresh replies, and the NPU stayed loaded throughout. [Measurements and source fingerprints](docs/speculative-fp32-workgroup-operator-v1.json).
<!-- onpu-fp32-workgroup-operator-v1:end -->

<!-- onpu-speculative-api-context-v19:start -->
## Historical DFlash2 API boundary (V19)

The pinned Qwen3.5-2B FP16-runtime target and trained DFlash2 pair passed the isolated API at configured context 8,192 with 8,176 prompt tokens and 16 new tokens. Ordinary/speculative greedy responses matched; sampled SSE at temperature 0.6 passed wrong-ID preservation, matching-ID cancellation and exact greedy recovery. Both original serving models were restored and answered fresh requests. All 27 worker-shaped probability comparisons matched with zero measured difference.

These are single uncached functional trials, not a repeated performance qualification or an RKLLM comparison. Bold marks the better observed number. Prefill is input tokens/TTFT; decode is the server-reported rate.

| Mode | Prefill proxy (tok/s) | Decode (tok/s) | Request (s) |
| :--- | ---: | ---: | ---: |
| Ordinary | **19.52** | **2.32** | **425.37** |
| DFlash2 | 17.92 | 1.34 | 467.45 |

Speculation was slower on this long request. Growing feature/full-attention KV buffers bypass the allocator size cache; fixed sliding buffers retain reuse. This is not model-maximum-context, general quality or speed qualification.

[Exact-boundary qualification, source fingerprints, CPU checks and restoration proof](docs/speculative-api-context-v19.json).
<!-- onpu-speculative-api-context-v19:end -->

<!-- onpu-draft-fc-operator-v1:start -->
## DFlash2 input projection: isolated packed-weight experiment

The trained draft's 2,048 × 12,288 input projection was tested with the existing Mali packed and unpacked GEMM kernels. Each size used three warmup pairs and ten measured pairs, alternating execution order. All 65 pairs produced identical FP32 output bits; independent NumPy checks also passed the existing GEMM tolerances.

| Input rows | Unpacked median (ms) | Packed median (ms) | Operator speedup |
| ---: | ---: | ---: | ---: |
| 8 | 25.67 | **16.20** | 1.58× |
| 13 | 27.71 | **16.45** | 1.68× |
| 30 | 33.08 | **16.47** | 2.01× |
| 112 | 125.02 | **63.96** | 1.95× |
| 128 | 134.28 | **64.73** | 2.07× |

Packing added 48 MiB and took 143.51 ms once. Timings cover the operator call and OpenCL synchronization; readback and reference checks are outside the timed region. Selective packing is now wired into the local experimental candidate, with 48 MiB added to admission. The V21 cohort passed scoped complete-request API checks for this source, including the 8,192-token boundary and sampled cancellation/recovery. These operator results do not establish a complete-request or RKLLM win. Both original models answered fresh requests after GPU restoration; the NPU remained loaded throughout. [Evidence](docs/speculative-draft-fc-operator-v1.json).

<!-- onpu-draft-fc-operator-v1:end -->

<!-- onpu-speculative-api-state-v15:start -->
## Historical short-prompt DFlash2 API: greedy and sampled

The pinned Qwen3.5-2B Q8_0 checkpoint runs with FP16 runtime weights and the trained DFlash2 draft on OpenCL GPU. Configured context is 512; prompts contain 27–36 tokens and request at most 32 new tokens. Model-maximum context and RKLLM output-quality parity remain unqualified.

One full run passed 36 greedy and 38 sampled nonstream requests, greedy/sampled SSE, wrong-ID preservation, matching-ID cancellation, and exact recovery. All 27 worker-startup-shaped probability comparisons passed with zero measured difference. Both original GPU/NPU models were restored and answered fresh requests.

Three warmups and three alternating measured trials per engine were run for each greedy prompt, with uncached input and identical outputs. Bold marks the better median: higher throughput or lower wall time. Prefill is the API prompt-token/TTFT metric; decode is the server-reported rate, whose EOS accounting can differ from visible tokens.

| Prompt | Input tokens | Visible tokens | Ordinary prefill (tok/s) | DFlash2 prefill (tok/s) | Ordinary decode (tok/s) | DFlash2 decode (tok/s) | Ordinary wall (s) | DFlash2 wall (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 30 | 29 | **8.55** | 8.48 | 2.50 | **3.66** | 15.13 | **11.21** |
| short_code | 36 | 17 | **8.38** | 8.33 | 2.50 | **4.08** | 11.12 | **10.19** |
| explanation | 27 | 32 | **7.70** | 7.64 | **2.50** | 1.95 | **15.94** | 19.48 |

Counting was 1.35× faster and code 1.09× faster end to end. Explanation was 22% slower despite switching to ordinary continuation after two unprofitable blocks. Packed state snapshots were used by every recorded speculative snapshot call. This profile avoids the unused 970 MiB packed output head; the observed peak RSS of all owned test processes was 9288.30 MiB, not GPU-only memory.

Sampled tests cover temperatures 0.6, 1, and 2. Their output lengths and descriptive timings are recorded without paired sampled speedup claims.

[Full-run qualification, source fingerprints, restoration proof and measured summaries](docs/speculative-api-state-v15.json). The [V13/V14 composite](docs/speculative-api-sampled-v14.json) remains historical evidence; its original failed full-run record is retained.
<!-- onpu-speculative-api-state-v15:end -->

<!-- onpu-trained-long-context-v16:start -->
### Long-prefix backend check

Configured context was 8,192 with an actual 2,061-token prompt. The trained pair matched all 16 ordinary-reference IDs, exercised the 2,048-row sliding caches, and passed sampled cancellation plus exact greedy recovery. All 27 probability comparisons passed with zero measured difference.

These are single functional trials, not a repeated performance qualification or an RKLLM comparison. Prefill is input tokens divided by first-token delay.

| Mode | Prefill proxy (tok/s) | Request (s) |
| :--- | ---: | ---: |
| Ordinary | **20.48** | **106.71** |
| DFlash2 | 19.62 | 115.39 |

This trial was slower with speculation. At the time of this core check the API limit was 512; the later V19 API run tested the 8,192 boundary. The model maximum of 262,144 remains unqualified.

[Core-run evidence and restoration proof](docs/speculative-long-context-core-v16.json).
<!-- onpu-trained-long-context-v16:end -->

<!-- onpu-speculative-http-v3:start -->
## Historical trained DFlash2 through HTTP (V3)

The pinned Qwen3.5-2B pair passed 36 uncached ordinary/speculative requests at context 512, with three warmups and three measurements per engine for each prompt.
Visible text, usage and finish reasons matched frozen references. Streaming, sampled ordinary fallback, scoped cancellation and post-cancel recovery passed.
The following medians include the complete HTTP request. Output counts exclude EOS, whose computation and cleanup remain in the timing.

| Prompt | Visible output tokens | Ordinary request(s) | Speculative request(s) | Ordinary/speculative |
| :--- | ---: | ---: | ---: | ---: |
| counting | 29 | 17.88 | **13.17** | 1.36x |
| short_code | 17 | 12.91 | **11.34** | 1.14x |
| explanation | 32 | **18.72** | 35.74 | 0.52x |

First-text latency remains higher with this speculative configuration. Gains depend on the prompt; the explanation workload is slower.
These observations do not establish an RKLLM win or model-maximum context qualification. The normal GPU/NPU sessions were restored and answered fresh requests.
[Per-run values, TTFT, full-cost decode definitions, source hashes and restoration proof](docs/speculative-api-prepared-v3.json).
<!-- onpu-speculative-http-v3:end -->

<!-- onpu-speculative-http-adaptive-v4:start -->
## Historical adaptive speculative HTTP V4

V4 passed the same three frozen greedy prompts as fixed V3 at context 512: 36 requests total, with three warmups and three measurements per engine/workload. V3 is retained as a separate fixed-pipeline cohort.
The table reports complete HTTP request medians after loading and warming. Setup times for both cohorts are recorded separately below.

| Prompt | Visible | V3 ordinary (s) | V3 fixed (s) | V4 ordinary (s) | V4 adaptive (s) | V4 fallback tokens/request |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 29 | 17.88 | 13.17 | 17.90 | **12.91** | 0/0/0 |
| short_code | 17 | 12.91 | 11.34 | 12.92 | **11.34** | 0/0/0 |
| explanation | 32 | 18.72 | 35.74 | **18.71** | 22.95 | 28/28/28 |

| Prompt | V3 ordinary TTFT (ms) | V3 fixed TTFT (ms) | V4 ordinary TTFT (ms) | V4 adaptive TTFT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| counting | 5837.89 | 7696.76 | **5835.95** | 5849.37 |
| short_code | 5845.63 | 7785.74 | **5845.63** | 5857.92 |
| explanation | 5832.54 | 7759.26 | **5830.50** | 5845.12 |

V3 load-request-to-first-Ready: 139.50s; V4: 226.63s. These are separate cohorts.
V4 setup used three ordinary decode calibration samples (128 prompt tokens) and 3 prepared-graph warmups × 24 output tokens before Ready.
V4 improves the counting and short-code request medians; explanation remains slower than ordinary. Per-round fallback is measured, while full-request routing/probe overhead remains open.
This is an isolated greedy HTTP result, not an RKLLM comparison, sampled speculation, model-maximum context, or general speed qualification.
[Per-run medians, TTFT, fallback counts, setup timing and source/job/receipt hashes](docs/speculative-api-adaptive-v4.json).
<!-- onpu-speculative-http-adaptive-v4:end -->

<!-- onpu-speculative-http-validrows-v5:start -->
## Historical valid-row GEMM speculative HTTP V5

V5 passed the same three frozen greedy prompts as V4 at context 512. V4 and V5 are distinct source cohorts; the comparison does not transfer V4 evidence to V5.
The table uses full HTTP request medians. Per-request fallback counts retain all three measured values for each workload.

| Prompt | Visible | V4 ordinary (s) | V4 adaptive (s) | V5 ordinary (s) | V5 valid-row (s) | V5 fallback tokens/request |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 29 | 17.90 | 12.91 | 15.57 | **10.65** | 0/0/0 |
| short_code | 17 | 12.92 | 11.34 | 11.38 | **9.69** | 0/0/0 |
| explanation | 32 | 18.71 | 22.95 | **16.41** | 20.66 | 28/28/28 |

| Prompt | V4 ordinary/adaptive TTFT (ms) | V5 ordinary/valid-row TTFT (ms) |
| :--- | ---: | ---: |
| counting | 5835.95 / 5849.37 | 3509.81 / 3516.98 |
| short_code | 5845.63 / 5857.92 | 4314.12 / 4299.63 |
| explanation | 5830.50 / 5845.12 | 3505.85 / 3512.27 |

| Setup cohort | Load request to first Ready (s) | Worker load (s) | Ordinary calibration | Prepared graph warmup |
| :--- | ---: | ---: | :--- | :--- |
| V4 adaptive | 226.63 | 179.01 | 3 single-token samples after 128 -token prefills | 3 × 24 tokens |
| V5 valid-row | 223.05 | 174.61 | 3 single-token samples after 128 -token prefills | 3 × 24 tokens |

These are separate source/setup cohorts; the V5 readiness interval includes its configured setup cost.
V5 passed the isolated HTTP gate for this workload. It does not establish RKLLM parity, sampled speculation, model-maximum context, or general serving performance.
[Per-run metrics, fallback vectors, setup timing and V4/V5 source/job/receipt hashes](docs/speculative-api-validrows-v5.json).
<!-- onpu-speculative-http-validrows-v5:end -->



<!-- onpu-cpu-gdn-head-parallel-operator-v1:start -->
## CPU GDN head-parallel correctness prototype

A staged C scan partitions independent batch/head slices across the existing CPU
worker pool. Twelve shapes matched the serial C baseline exactly for output,
state and history; null-dispatch fallback and partial/completed dispatch errors
were checked. Four real-pool cases covered both `parallel` and `parallel_parked`,
including parked workers and restored caller affinity. Both original models
were restored and answered fresh requests. The initial operator proof preceded the opt-in backend binding and model
measurements below. Its own scope does not establish model speed or quality.

[CPU operator correctness and restoration evidence](docs/benchmark-cpu-gdn-head-parallel-operator-v1.json).
<!-- onpu-cpu-gdn-head-parallel-operator-v1:end -->

<!-- onpu-cpu-gdn-head-parallel-model-v1:start -->
## Qwen3.5-2B CPU scan with parked head workers

The optional CPU scan/pool binding is integrated in tinygrad. Both new CPU+NPU
INT8 variants use 128 input tokens, 64 output tokens and context 512, with three
full warmups and three measured requests. All twelve sequences match each other
and the six earlier preferred INT8 sequences. Pool counters record 18 prefill
and 1,134 decode dispatches per pooled request; both phases submit NPU work.
Only pooling differs between the two new variants.

| Variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| CPU scan, serial | 2002.41 | 63.92 | 10.00 |
| CPU scan, parked pool | **1798.98** | **71.15** | **10.80** |

This historical pooled candidate first replaced the original Qwen3.5-2B row. Its
71.15/10.80 tok/s remain below the recorded RKLLM 164.75/13.59 tok/s reference.
Output quality versus GPU FP16/RKLLM and full learned-matrix coverage remain
unqualified. The app worker does not automatically enable this model profile.
Both original models were restored and answered fresh requests.

[Matched model candidates and provenance](docs/benchmark-cpu-gdn-head-parallel-model-v1.json),
[CPU tensor/JIT/configuration and App checks](docs/benchmark-cpu-gdn-head-parallel-preflight-v1.json).

The separate C microbenchmark used four warmups and twelve measured trials per
mode/profile, caller CPU4, workers CPU5-7, no state-history output and normalized
query/key inputs. Reset, allocation and hashing were outside timers; timings
include repeated C calls through ctypes. All 512 warmup/measured trials had
identical output/state bits. These are scan timings, not full-model speedups.

| Valid rows | Original serial (ms) | Candidate serial (ms) | Parallel (ms) | Parked parallel (ms) |
|---:|---:|---:|---:|---:|
| 1 | 0.940 | 0.894 | **0.224** | 0.250 |
| 3 | 1.807 | 1.657 | **0.419** | 0.474 |
| 8 | 3.967 | 3.565 | **0.903** | 0.960 |
| 128 | 56.409 | 50.749 | **12.723** | 12.740 |

The 16-head/key128/value128 shape matches the recorded 2B recurrent geometry;
the artifact also contains synthetic 32-head coverage. Parking workers avoids
leaving them spinning between NPU calls.

[C scan timing and restoration evidence](docs/benchmark-cpu-gdn-head-parallel-perf-v1.json).
<!-- onpu-cpu-gdn-head-parallel-model-v1:end -->

<!-- onpu-native-replay-warmup-profile-v1:start -->
## Native replay profile and warmup preparation

The pooled Qwen3.5-2B diagnostic attributes about 88% of captured native prefill
wall time to CPU kernels, and about 78% of native decode wall time to Rockchip
calls. Rockchip labels include validation/submission/driver time as well as NPU
execution; these are label wall times, not pure hardware utilization. Measured
deltas subtract the counters after warmup trial2 from those after measured trial5.

The first diagnostic exposed native prefill compilation inside measured trial3:
its TTFTs were 11,647.53 / 1,799.08 / 1,799.94 ms. Replay preparation now happens
after the existing ordinary warmup update, without executing the graph twice.
The verification run constructed decode replay in warmup0 and prefill replay in
warmup2; all three measured TTFTs were 1,799.97 / 1,801.71 / 1,801.79 ms, with
matching INT8 output IDs. Captured native work accounted for 99.2% of the measured
prefill interval after preparation moved into warmup.

Both original models were restored and answered fresh requests. This diagnostic
did not replace the unprofiled timing entry; the subsequent pooled-sigmoid
comparison below supplies the current entry. Profiling and source capture do
not establish a new SDK win, output quality or API scope.

[Initial diagnostic](docs/benchmark-cpu-npu-native-profile-v1.json),
[Warmup fix and generated CPU source](docs/benchmark-native-replay-warmup-model-v1.json),
[CPU/App regressions](docs/benchmark-native-replay-warmup-preflight-v1.json).
<!-- onpu-native-replay-warmup-profile-v1:end -->

<!-- onpu-cpu-sigmoid-backend-model-v1:start -->
## Qwen3.5-2B pooled prefill sigmoid

The opt-in CPU+NPU profile dispatches the gate-half sigmoid over parked CPU
workers during prefill. It retains the captured scalar arithmetic and the
existing split-down projection path. Decode keeps its existing path.

Both variants use pooled recurrent CPU scan, 128 input tokens, 64 output tokens,
context 512, three full warmups and three measured requests. Only the candidate
enables the FFN flag that selects the pooled float sigmoid for this model.

| Variant | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Pooled scan, generic sigmoid | 1796.29 | 71.26 | 10.69 |
| Pooled scan and sigmoid | **1602.53** | **79.87** | 10.73 |

Prefill improved by 12.1%. Decode ranges overlap (10.683–10.756 versus
10.676–10.758 tok/s), so no decode improvement is established. The main table
uses the complete latest candidate, including its 6326.94 MiB lifetime process
peak RSS. It does not combine the new prefill timing with an older decode run.
All twelve generated sequences equal the prior INT8 control. Actual pool
dispatches increase from 18 to 42 per prefill request; decode remains 1,134.

The recorded RKLLM reference remains faster at 164.75/13.59 tok/s. FP16/RKLLM
output quality, full learned-matrix W8A8 coverage, API use and model-maximum
context remain unqualified. The app worker does not automatically enable this
profile. Both original serving models were restored and answered fresh requests.

[Whole-model measurements and provenance](docs/benchmark-cpu-sigmoid-backend-model-v1.json),
[CPU/App checks](docs/benchmark-cpu-sigmoid-backend-preflight-v4.json),
[Isolated sigmoid correctness and timings](docs/benchmark-cpu-prefill-sigmoid-pool-v1.json).
<!-- onpu-cpu-sigmoid-backend-model-v1:end -->

<!-- onpu-speculative-extra-budget-model-v2:start -->
## Optional early fallback after an expensive speculative block

The backend accepts `max_speculative_extra_seconds` alongside an ordinary
decode reference. After each completed block, it compares accumulated measured
compute time with the ordinary cost of the same committed token count. An
explicit threshold can trigger fallback before the existing two-block rate
check. Consumer pauses are excluded. This is a decision threshold, with possible
overshoot by a complete block; it does not interrupt GPU kernels. The default
remains `None`, and the app does not enable this option.

The trained Qwen3.5-2B/DFlash2 core comparison used the same resident pair in
three modes: ordinary, existing policy, and a threshold equal to one calibrated
ordinary token time. Each workload/mode had three full warmups and three measured
requests, with balanced execution positions. All 54 output sequences matched the
frozen references, including natural EOS. Timers include iterator cleanup and
final GPU synchronization; cache reset and loading are outside the timers.

| Workload | Ordinary (s) | Existing policy (s) | Extra-cost threshold (s) |
|---|---:|---:|---:|
| Counting | 10.28 | **8.86** | 8.89 |
| Short code | **8.19** | 8.52 | 8.70 |
| Explanation | **10.91** | 14.24 | 13.02 |

The threshold reduced the observed explanation median by 8.6%, switching after
one speculative block instead of two. Counting ranges overlap. Code measured
2.03% slower, although both policies used three blocks, accepted 15 proposals
and generated zero fallback tokens. Frequency endpoints changed during trials;
these timings do not isolate a causal policy or kernel effect. Every sample is
retained, and the app/default policy stays unchanged.

These are short greedy core requests with context 8192 configured, not a new
HTTP, long-context, model-maximum, sampled GPU, or RKLLM qualification. The
ordinary mode shared draft residency, so its memory is not an ordinary-only
serving footprint. Both original serving models were restored with their exact
settings and answered fresh requests. The first harness attempt failed before
any trial because it omitted the worker's fixed-prefill generator wrapper;
that attempt and its restoration receipt are preserved separately.

[Measurements, ranges, policy counters and provenance](docs/speculative-extra-budget-model-v2.json),
[CPU/App regression checks](docs/speculative-extra-budget-preflight-v1.json).
<!-- onpu-speculative-extra-budget-model-v2:end -->

<!-- onpu-trained-gpu-kernel-profile-v2:start -->
## Current trained GPU kernel costs

The current Qwen3.5-2B/DFlash2 profile collected OpenCL event timestamps after
three unprofiled warmups per workload. Each diagnostic replays the last warmup's
fallback decisions so the added waits do not change the execution path. All
twelve requests retained the frozen IDs, and block/acceptance/fallback counts
matched the successful unprofiled control.

| Workload | Prefill GPU kernels (s) | Draft GPU kernels (s) | Verify GPU kernels (s) | Fallback GPU kernels (s) | All GPU kernels (s) | Vocabulary-head share |
|---|---:|---:|---:|---:|---:|---:|
| Counting | 1.659 | 1.760 | 3.577 | 0 | 7.173 | 23.25% |
| Short code | 3.053 | 1.400 | 2.737 | 0 | 7.352 | 17.20% |
| Explanation | 1.629 | 0.882 | 1.789 | 6.399 | 10.815 | 20.72% |

Totals also include observation, snapshot and restore kernels.

The vocabulary head uses FP32 activations and FP16 stored weights. Its full
248,320×2,048 shape accounts for 1.668 / 1.265 / 2.241 seconds of recorded GPU
kernel work across the three workloads. Counting also spent 0.965 seconds in
8-row 6,144×2,048 projections and 0.500 seconds in 8-row 2,048×6,144 projections
during verification. Matrix dimensions come from recorded buffer extents and
the kernel interfaces; no per-layer weight identity is inferred.

These serialized kernel times exclude copies, CPU work and gaps. They are not
complete-request latency measurements, and adding host wait time to GPU event
time would double-count work. Frequency readings varied. The table supplies
optimization priorities without changing the inference timing tables above.
The earlier workgroup sweep covered only the 6,144×2,048 shape and showed mixed
small differences; it did not qualify the full vocabulary head.

The collector passed a real GPU event/output check before model loading. Its
first attempt failed before inference because the collector omitted the name
field from the runtime's four-field argument signature. Both attempts retain
their own evidence; the successful run restored both original workers and
verified fresh replies.

[Kernel sources, dimensions, phase totals and provenance](docs/speculative-trained-gpu-profile-v2.json),
[Earlier limited workgroup sweep](docs/speculative-fp32-workgroup-operator-v1.json).
<!-- onpu-trained-gpu-kernel-profile-v2:end -->

<!-- onpu-vocabulary-row-tiling-model-v1:start -->
## Qualified two-row GPU GEMV and trained-model timings

The new kernel keeps two output rows live per work item and interleaves row
tiles within each column group. Each output retains its original 16 K lanes,
FP32 FMA sequence, component additions and ordered lane reduction. Dispatch is
limited to FP16 stored weights on the qualified Mali device/driver: the 8-row
248,320×2,048 head and 3/8-row 2,048×6,144 down-projection shapes. Other cases
retain their existing path. No extra packed weight copy is introduced.

The actual-checkpoint operator grid measured the selected 8-row head at
203.58 → 109.14 ms and down projection at 5.65 → 3.10 ms, with exact output and
weight bits. Six warmup sets and twelve measured sets used balanced forward
and reverse mode orders. Earlier workgroup/unrolling variants gave no material
gain for the currently used 1/8-row head cases; aggressive 8-row unrolling
regressed and was not integrated.

The whole-model comparison runs legacy and integrated kernels in separate
processes, forcing the legacy factory before capture in the control. Both use
the same trained Qwen3.5-2B/DFlash2 pair, configured context 8192, the three frozen
short prompts, and three full warmups plus three measured requests per mode.
The candidate process also measures ordinary decoding on the same resident pair.

| Workload | Ordinary (s) | Legacy speculation (s) | Two-row speculation (s) |
|---|---:|---:|---:|
| Counting | 10.39 | 8.86 | **7.95** |
| Short code | 8.21 | 8.58 | **7.80** |
| Explanation | **10.92** | 14.18 | 13.70 |

The observed speculative latency reductions versus the legacy kernels were
10.3%, 9.1% and 3.4%. The new counting and code medians were 23.5% and 5.1%
shorter than ordinary decoding. Explanation remains slower than ordinary.
Every one of the 54 warmup/measured sequences matched its frozen reference.
Runtime counters recorded 104 / 78 / 52 tiled-kernel calls per candidate
speculative request and zero in legacy/ordinary requests.

Timers include iterator cleanup and final GPU synchronization. State reset,
loading and graph warmup are excluded. The ordinary mode shares draft residency;
its memory is not an ordinary-only serving footprint. The control process ran
before the candidate process. Clock endpoints varied, and every measured sample
is retained. These are scoped core-model results, with natural EOS or a 32-token
budget; they do not change the canonical 128-input/64-output W8A8 table.

The integrated shader source matches the qualified prototype except for its
entry-point name. CPU/GPU/App checks passed 315 methods plus 125 subtests, Ruff
and mypy. Both original serving workers were restored with their exact options,
selection and settings and answered fresh requests. New-source HTTP, sampled
GPU, long-context/model-maximum and RKLLM quality/speed qualification remain open.

[Whole-model records, ranges and provenance](docs/speculative-vocabulary-row-tiling-model-v1.json),
[Operator grid](docs/speculative-vocabulary-row-tiles-v2.json),
[Integration checks](docs/speculative-vocabulary-row-tiling-preflight-v2.json),
[Rejected workgroup/unrolling variants](docs/speculative-vocabulary-fp32-operator-v1.json).
<!-- onpu-vocabulary-row-tiling-model-v1:end -->

<!-- onpu-draft-reusable-inputs-component-v2:start -->
## Reusable draft K/V input staging (component only)

The opt-in backend layout retains owned fixed-capacity proposal inputs. It
updates changed full-attention rows and refreshes sliding windows; reset,
rollback and failed writes invalidate reuse and clear stale padding. Native
rectangular copies use the existing in-order OpenCL queue. Unsupported storage,
profiling and capture use the Tensor fallback.

The trained Qwen3.5-2B DFlash2/shared-head component comparison streamed 67
synthetic six-tap feature appends through 8,192 rows. All K/V bits matched;
five proposal positions matched logits, hidden states and candidate IDs.
Thirteen proposal-input comparisons matched all padded bits and zero tails;
three sampled temperatures matched tokens and conditional probabilities.
The target body and HTTP API were not executed in this comparison.

Each row below uses three warmup pairs and six measured pairs, alternating
control/candidate order. Timers cover repeated unchanged-prefix `_inputs`
staging plus GPU synchronization. They exclude observation and complete
proposal/generation work.

| Committed prefix tokens | Pad/clone control (ms) | Native reusable inputs (ms) |
|---:|---:|---:|
| 128 | 16.55 | **1.45** |
| 2,048 | 16.10 | **2.87** |
| 4,096 | 14.62 | **2.84** |
| 8,086 | 16.07 | **2.84** |

The first Tensor-assignment variant passed parity but was 11.6–14.6% slower in
these staging-only measurements; it was not enabled. The native follow-up
recorded 304 copy commands and 503,361,536 logical input-copy bytes. Fixed
proposal-input capacity is still 29,360,128 bytes alongside committed K/V.
These counters exclude observation copies, attention scratch, model traffic
and total RSS; they do not establish a memory-fit or generation-speed claim.

CPU checks passed 61 methods plus 56 subtests; eight existing explicit-hardware
cases were skipped in the CPU run. Three Mali fixtures passed, including partial
Tensor-write recovery and native enqueue-failure/retry. Full Ruff and mypy over
263 backend sources passed. Both original serving workers were restored with
fresh replies, while the NPU worker remained loaded throughout each GPU lease.

The option defaults off and the app does not enable it. Whole-model and current
HTTP qualification, model-maximum context and RKLLM quality/performance remain
open. The canonical W8A8 table is unchanged.

[Native-copy results and provenance](docs/speculative-draft-reusable-inputs-component-v2.json),
[Slower Tensor-assignment result](docs/speculative-draft-reusable-inputs-component-v1.json).
<!-- onpu-draft-reusable-inputs-component-v2:end -->

<!-- onpu-draft-reusable-inputs-model-v1:start -->
## Native draft-input copies: full-target comparison

A single resident Qwen3.5-2B/DFlash2 pair compared ordinary decoding, the
current pad/clone proposal inputs and native reusable inputs. Both proposal
graphs were kept resident. Each workload/mode used three warmups and three
measurements with balanced execution orders, configured context 8192 and a
32-token reply budget. Actual prompts contain 30/36/27 tokens; natural EOS
ends counting/code at 30/18 outputs and explanation reaches 32. These are
FP16-runtime core measurements, separate from the W8A8 table.

All 54 warmup/measured sequences matched their frozen IDs. Both speculative
modes used the same adaptive-fallback reference as the serving worker. Timers
include iterator exhaustion/cleanup and final synchronization; loading and
state reset are excluded.

| Workload | Ordinary wall (s) | Current speculation (s) | Native-input speculation (s) |
|---|---:|---:|---:|
| counting | 10.27 | **7.37** | 7.45 |
| short_code | 8.13 | **7.46** | 7.50 |
| explanation | **10.80** | 13.16 | 13.20 |

Prefill proxies include first-token work. Decode rates include cleanup after
the first output; values below are current speculation / native-input candidate.

| Workload | Prefill proxy (tok/s) | Decode full cost (tok/s) |
|---|---:|---:|
| counting | 17.34 / 17.01 | 5.14 / 5.12 |
| short_code | 11.36 / 11.45 | 3.96 / 3.92 |
| explanation | 15.65 / 15.66 | 2.71 / 2.70 |

The native candidate's complete-request medians were 1.12%, 0.54% and 0.34%
higher than control. Code/explanation sample ranges overlap; counting ranges
are separate in these three measurements. GPU clock observations span 300 MHz
to 1 GHz, so the small differences do not establish their cause. Faster isolated
input staging did not yield a demonstrated generation gain. The app continues
to leave `reuse_prefix_inputs` off. Model-maximum context, current-source HTTP
qualification and RKLLM quality/performance remain open.

The child exited successfully and every strict data-validator gate passed.
The original controller exited 2 because its 269-entry package scan was compared
with the report's 270-entry map, which additionally pins the timing/reset helper.
The original job/failure remains unchanged. Independent data audit and separate
full-map revalidation verify the complete map, helper hash and all 314 frozen
pins. Both original workers were restored and answered fresh requests; the
normal server was not restarted. Two earlier harness attempts are preserved:
one prefill-bound mismatch and one canceled before measurement to restore the
worker's adaptive policy. Neither supplies published timing results.

[Full-target records, ranges and corrected source validation](docs/speculative-draft-reusable-inputs-model-v1.json),
[Separate faster staging component](docs/speculative-draft-reusable-inputs-component-v2.json).
<!-- onpu-draft-reusable-inputs-model-v1:end -->

<!-- onpu-q8-short-head-operator-v4:start -->
## Short Q8 vocabulary projections (operator qualification)

The new opt-in `gemv_q8_rows` reconstructs FP16-rounded Q8_0 weights and
retains the dense FP32 GEMV's 16 K lanes, component additions and ordered lane
reduction for capacities 1–8. It rejects FP16 activations, whose existing
consistent-decode path rounds each product differently. Existing model and app
projection dispatch is unchanged.

The public operator selects the measured g2/t8 geometry only for eight rows,
the 248,320×2,048 vocabulary matrix and the qualified Mali device/driver. Other
shapes retain its original geometry. The direct paired test below uses the
actual tied Qwen3.5-2B embedding/output weight, six warmup pairs and twelve
measured pairs per capacity, with alternating order. Timers cover the complete
operator call plus GPU synchronization.

| Row capacity | Dense FP16 storage (ms) | Q8 storage, FP16-rounded values (ms) |
|---:|---:|---:|
| 1 | 53.70 | **34.77** |
| 3 | **55.39** | 57.03 |
| 8 | **113.43** | 119.06 |

All 54 pairs had identical FP32 output bits and argmaxes, with finite outputs.
Runtime kernel-name counters confirm the public path used the expected t1,
t3 and tuned t8/g2 kernels. Three Mali fixtures also passed, including symbolic
row rebinding, NaN padding, partial columns and all experimental geometries.
CPU checks passed seven methods plus 24 subtests; Ruff and mypy over 263
backend sources passed.

The one-row case reduced latency by 35.2%. Three- and eight-row paired medians
were 3.0% and 5.0% higher than dense, so no general projection or inference win
is declared. Earlier balanced six-variant grids found an eight-row Q8 median
below their dense control; that apparent win did not reproduce in this direct
paired public-operator cohort. Each cohort remains separately recorded; clock
variation and workload mixing prevent a cross-cohort before/after claim.

Raw matrix storage is 540,344,320 bytes versus 1,017,118,720 bytes for dense
FP16, but both matrices are resident in this comparison. Current head attachment
also retains the dense embedding/output wrappers. Releasing that dense storage
requires separate shared raw-embedding/head integration; these byte counts do
not establish a whole-model RAM saving or model-maximum fit.

This is Q8 storage with FP16-rounded weights and FP32 arithmetic, not W8A8.
No target body, draft, API, long-context request, SDK quality comparison or
whole-model speedup was qualified here. The normal NPU worker stayed loaded;
the original GPU worker was restored with its exact options, and both workers
answered fresh requests. The normal server was not restarted.

[Direct public-operator results and activation proof](docs/benchmark-mali-q8-short-head-operator-v4.json),
[First untuned operator cohort](docs/benchmark-mali-q8-short-head-operator-v1.json),
[First geometry grid](docs/benchmark-mali-q8-short-head-operator-v2.json),
[Final bounded geometry grid](docs/benchmark-mali-q8-short-head-operator-v3.json).
<!-- onpu-q8-short-head-operator-v4:end -->

<!-- onpu-shared-q8-table-model-api44:start -->
## Shared Q8 embedding/head: trained model and API

The opt-in replacement shares one owned Q8_0 allocation between native token
lookup and the bounded FP32 vocabulary head. Its `weight` is a real lazy FP16
dequantization view. Native requests leave that view unrealized; reading and
explicitly realizing it still produces the correct dense values. Replacement
runs after Mali configuration and before warmup/capture, with tied/unpacked
FP16, residency, state and graph guards. Default ordinary profiles are unchanged.

The real target comparison ran dense control then shared-table candidate in
separate processes. Both kept the same dense body kernels and trained draft,
configured context 8192, actual prompts 30/36/27 and natural EOS or a 32-token
budget. Each workload/mode/variant used three warmups and three measurements.
All 72 output sequences matched frozen references. Speculation retained the
worker's 128-token prefill and adaptive fallback calibration.

| Workload | Dense ordinary (s) | Shared ordinary (s) | Dense speculative (s) | Shared speculative (s) |
|---|---:|---:|---:|---:|
| counting | 10.36 | 9.71 | 7.91 | **7.88** |
| short_code | 8.21 | 7.85 | 7.83 | **7.74** |
| explanation | 10.96 | **10.28** | 13.71 | 13.02 |

The observed ordinary-request medians were 4.4–6.2% shorter and speculative
medians 0.5–5.0% shorter. The counting/code speculative sample ranges overlap.
Clock variability and control-before-candidate ordering limit causal/general
speed claims. Explanation remains faster with ordinary decoding.

Prefill proxies include first-token work; decode includes cleanup after the
first output. The shared-table candidate measured:

| Workload | Ordinary prefill proxy (tok/s) | Ordinary decode (tok/s) | Speculative prefill proxy (tok/s) | Speculative decode (tok/s) |
|---|---:|---:|---:|---:|
| counting | 17.62 | 3.62 | 16.97 | 4.77 |
| short_code | 11.43 | 3.62 | 11.43 | 3.69 |
| explanation | 15.43 | 3.63 | 15.72 | 2.74 |

In the actual model the old tied allocation's weakrefs died before inference,
and lookup/head retained 540,344,320 raw bytes instead of the 1,017,118,720-byte
dense matrix. The lazy dense view stayed unrealized through calibration and all
requests. Lifetime peak RSS is recorded per process; loading, mappings and draft
residency are included. These observations do not establish whole-service RAM
savings or model-maximum fit. Admission reserve and the 8192 context cap remain
unchanged. Arithmetic/precision is not W8A8; the canonical SDK table is unchanged.

The V44 private HTTP run then qualified the current worker hook on the pinned
experimental target/SHA and Mali-LODX r0p0 driver 2.1. Other profiles/devices keep
their existing path. All ten short controls passed: forced cache reuse 31/55,
matched off/auto cached responses, cache disabling, repeated cold reference,
sampled temperature-0.6 scoped stop, wrong-ID preservation, DONE/ready and exact
greedy recovery. Every forced request reported one shared raw table, resident
raw storage, an unrealized logical dense view and zero projected-feature history.
App checks passed 307 methods plus 23 subtests (330 JUnit cases), with no errors,
failures or skips, and full Ruff. Both original workers were restored with their
exact options/selection/settings and answered fresh requests; normal server not
restarted.

This qualifies short functional HTTP serving. New-layout full-8192 API boundary,
model maximum 262144, broad sampled-distribution/output-quality coverage and
RKLLM parity remain open. Prior V41 long timings belong to a different frozen
layout. Core times above are not transferred into API or SDK speed claims.

[Trained dense/shared target comparison](docs/benchmark-mali-q8-shared-table-model-v1.json),
[Ownership and numeric component proof](docs/benchmark-mali-q8-shared-table-component-v1.json),
[Current-source V44 API proof](docs/speculative-shared-q8-table-api-v44.json).
<!-- onpu-shared-q8-table-model-api44:end -->

<!-- onpu-shared-q8-long-api46-attempt:start -->
## Current shared-Q8 long API attempt

The [V46 attempt](docs/speculative-shared-q8-long-api-v46-attempt.json) reached an
8,081-token forced request, reused 8,049 tokens and executed one seven-token
proposal block. It accepted zero draft tokens and stopped after visible `2` plus
EOS, despite a requested upper budget of 16. The harness required exactly 16
visible tokens and stopped before matched ordinary decoding, cancellation,
recovery or boundary controls. This is **not a long API qualification**.

The cold 8,048-token seed took 424.36 seconds to first output: **18.96 prefill
proxy tok/s**. The cached request processed 32 new prompt tokens at a **16.31
tok/s prefill proxy** and took 7.21 seconds overall. Each returned one visible
token, so neither establishes sustained decode throughput or a speed win.
The shared raw table remained resident, its logical dense view unrealized, and
projected feature history zero. The 8,192 cap and admission reserve were unchanged.

The next comparison must honor EOS as well as the output cap, compare exact
ordinary output from the same seed, and account for EOS in the processed cache.
It must also avoid interleaving ordinary work before testing the speculative
cache, since that deliberately invalidates the draft's cache identity.
Both original workers were [restored and freshly verified](docs/speculative-shared-q8-long-api-v46-closure.json); the normal server was not restarted.
<!-- onpu-shared-q8-long-api46-attempt:end -->

<!-- onpu-fallback-prefix-retention-v1:start -->
## Long-context fallback prefix retention candidate

The [V47 run](docs/speculative-shared-q8-long-api-v47-attempt.json) changed the
8,048-token seed to explicitly request one digit. Its cached 8,081-token follow-up
then produced all 16 visible tokens, proposed 14 draft tokens and accepted 8.
After two rounds the measured-cost policy switched to ordinary decoding for
6 tokens. The draft did not observe those fallback inputs, so finalization
cleared its reusable prefix. The probe stopped on its incorrect assumption
that every completed forced request retains a full draft prefix. Matched ordinary,
cancellation, recovery and boundary controls did not run; full qualification is false.

| Observed request | New prompt tokens | Prefill proxy (tok/s) | Decode (tok/s) | Visible output tokens |
|---|---:|---:|---:|---:|
| Cold seed | 8,048 | 18.77 | — | 1 |
| Cached forced follow-up | 32 | 15.91 | 2.24 | 16 |

These are single-request FP16-arithmetic observations with Q8 weight storage,
not W8A8 results, a repeated timing protocol or an RKLLM win. Both original
workers were restored and answered fresh requests; the normal server was not restarted.

The new backend-only `retain_fallback_prefix=False` option can observe existing
fallback features and retain the processed target/draft prefix. It requires
`reuse_prefix=True`, preserves the unprocessed final token in deferred mode,
and defaults off. The app does not enable it. CPU checks cover both fallback
modes, strict continuation, EOS, cancellation, close, observation errors,
unchanged target-forward counts and early anchors. The trained-model study below measures overhead and combined cost for these
short prompts. API and long-context qualification remain pending.
<!-- onpu-fallback-prefix-retention-v1:end -->

[Fallback-prefix candidate CPU evidence](docs/speculative-fallback-prefix-retention-cpu-v1.json):
28 targeted methods plus 31 subtests passed (two GPU-only skips), all 330 app
JUnit cases passed, mypy checked 264 sources and full Ruff passed. The option
remains disabled in the app; the trained-model study below measured mixed costs.

<!-- onpu-fallback-prefix-retention-trained-v1:start -->
## Trained fallback-retention comparison

The [trained-pair experiment](docs/speculative-fallback-prefix-retention-model-v1.json)
completed 72 requests on the shared-Q8 Qwen3.5-2B/DFlash2 pair. Each of two
workloads ran three warmup and three measured rounds across three rotating arms.
Every seed matched its frozen IDs, and every retained forced follow-up matched
ordinary decoding from the same cached target prefix. Retention preserved 59
processed tokens for counting and 58 for explanation, including the deferred
pending-token contract, without adding seed target-forward calls.

Median combined seed + follow-up latency, seconds (lower is better):

| Workload | Discard draft, forced follow-up | Retain draft, forced follow-up | Discard draft, cached ordinary follow-up |
|---|---:|---:|---:|
| counting | 15.079 | 13.828 | **12.391** |
| explanation | 22.431 | 23.563 | **16.761** |

Retention reduced combined counting time by 8.30% versus the old forced path,
but increased explanation time by 5.05%. Cached ordinary follow-ups had the
lowest combined cost in both workloads. The option stays **off in the app**.
The added seed observation cost is included; a faster follow-up alone is not
counted as an end-to-end win.

The seed fallback reference was deliberately 1e-12 seconds/token to force the
path under test; this is not the calibrated API policy or an RKLLM benchmark.
Raw generated ID counts include EOS. Actual seeds were 30/27 prompt tokens and
follow-ups 90/82; allocated context was 8192. This does not qualify actual 8K or
model-maximum execution, API serving, broader quality or general speed.

The child exited 0 and passed strict validation. The historical owner exited 2:
its copied completion gate still expected two processes although this protocol
uses one process with three internal arms. That failure is preserved. A separate
single-process revalidation (path and hash in the proof)
verified all required data; the proof links exact paths and hashes. Both original
workers were restored and answered fresh requests; the normal server was not restarted.
Follow-up prefill proxies in tok/s (new prompt tokens / raw first-token latency):

| Workload | Discarded forced | Retained forced | Cached ordinary |
|---|---:|---:|---:|
| counting | 19.64 | 17.89 | 18.16 |
| explanation | 17.93 | 13.94 | 14.14 |

Cold forced follow-ups process all 90/82 prompt tokens; cached follow-ups process
31/24 new prompt tokens. These rates cover different amounts of prefill work and
are not a general throughput ranking.
<!-- onpu-fallback-prefix-retention-trained-v1:end -->

<!-- onpu-fallback-feature-batching-v2:start -->
## Batched fallback feature observations

The backend can now batch up to eight completed fallback feature rows with
`retain_fallback_prefix=True, fallback_observation_batch_size=8`. Default size
remains one and the app does not enable retention. Only captured feature rows
are cloned immediately from expiring target graph buffers; logits/hidden outputs
are not copied or fabricated. Feature-only DFlash observation uses contiguous
rows, flushes the final tail before reuse, and leaves the final deferred input
unprocessed. Cancellation, close and errors discard the queue and invalidate
reuse. Legacy reset-failure behavior is preserved.

CPU checks passed 305 methods plus 293 subtests, with 35 skips; all 330 app JUnit
cases passed. Mypy checked 264 sources and full Ruff passed. App checks used a
GPU-only temporary lease after the broad CPU suite briefly left little free RAM;
the NPU stayed loaded, and original serving was restored and freshly checked.
[CPU/source and lifecycle proof](docs/speculative-fallback-prefix-batching-cpu-v1.json).

The [trained v2 comparison](docs/speculative-fallback-prefix-retention-model-v2.json)
completed 72 requests: two workloads, three rotating arms, three warmups and
three measured rounds. Seeds retained frozen EOS-aware IDs, all retained
follow-ups matched the cached ordinary oracle, and target/fallback call counts
were unchanged. Batching retained the same processed prefix with at most eight
owned rows. The queue peaked at eight rows; total copied bytes per seed were
638,976 for counting and 1,327,104 for explanation (total traffic, not peak RAM).

Median combined seed + follow-up seconds, lower is better:

| Workload | Scalar retention | Batch of eight | Cached ordinary follow-up |
|---|---:|---:|---:|
| counting | 13.670 | 12.577 | **12.406** |
| explanation | 23.646 | 20.809 | **16.819** |

| Workload | Observation calls scalar / batch | Seed observation seconds scalar / batch | Combined reduction vs scalar |
|---|---:|---:|---:|
| counting | 13 / 2 | 2.019 / 0.913 | 8.00% |
| explanation | 27 / 4 | 3.809 / 1.432 | 12.00% |

All follow-ups reused the same 59/58-token target prefixes and processed 31/24
new prompt tokens. Follow-up prefill proxies, tok/s (new tokens / raw first-token
latency):

| Workload | Scalar retention | Batch of eight | Cached ordinary |
|---|---:|---:|---:|
| counting | 17.89 | 17.89 | 18.14 |
| explanation | 13.93 | 13.92 | 14.14 |

Batching improved both scalar-retention workloads, but cached ordinary decoding
still had the lowest combined cost. The app keeps retention off. This experiment
forces seed fallback with a 1e-12 seconds/token reference; it is not calibrated
API routing, an RKLLM/W8A8 benchmark, full-8K/model-maximum execution or broad
quality qualification. Seeds were 30/27 prompt tokens; context allocation was 8192.

The original frozen v2 validator incorrectly required the *post-follow-up* draft
prefix to equal the seed prefix. Actual valid lengths grew from 59/58 to 98/89.
The child exited 0; strict parent validation failed and owner exited 1. Those
failures and files remain preserved. A corrected standalone validator checks
`follow_input + generated_outputs - pending_last_token`, rejects stale/missing/
incorrect counters and altered oracle IDs, and passes every scoped gate. An
independent audit verifies all 327 pins and the corrected contract. Exact paths
and hashes are in the proof; no raw result was rewritten. Both original workers
were restored and answered fresh requests; the normal server was not restarted.
<!-- onpu-fallback-feature-batching-v2:end -->

<!-- onpu-forced-batched-retention-api48:start -->
## Current forced-retention API: short and near-8K qualification

This V48 status supersedes earlier app-off/pending notes only for explicitly
forced requests on the measured profile. Earlier cohorts remain historical;
automatic/off behavior and the default loaded-model setting are unchanged.

The [V48 source-bound API proof](docs/speculative-forced-batched-retention-api-v48.json)
passed the original validator without correction: 19 gates, 327 pins, 13 short
non-stream controls, six long non-stream requests, two scoped sampled cancellations
and one budget rejection. All 341 App JUnit cases passed. Explicit `force` on the
measured shared-table profile enables fallback-prefix retention with batch size 8;
backends without the required capabilities retain their previous configuration.
Automatic speculative cold requests reset to non-retaining scalar observations;
cached automatic requests remain ordinary, as in the measured cost policy.

Single-request long API observations (prefill proxy = new prompt tokens / TTFT):

| Case | Prompt tokens | Cached tokens | Visible outputs | Prefill proxy (tok/s) | Decode (tok/s) | API wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| Cold forced seed | 8,048 | 0 | 1 | 18.43 | — | 438.850 |
| Cached forced decode | 8,081 | 8,049 | 16 | 16.12 | 2.31 | 9.252 |
| Cold recovery after cancellation | 8,129 | 0 | 16 | 17.06 | 2.39 | 482.826 |
| Cached automatic boundary | 8,191 | 8,144 | 1 | 12.46 | — | 3.825 |
| Cold forced seed for ordinary control | 8,048 | 0 | 1 | 16.91 | — | 476.794 |
| Cached ordinary control | 8,081 | 8,049 | 16 | 14.82 | 3.22 | 6.871 |

The forced cached request proposed 14 and accepted 8 tokens. Five fallback inputs
were observed in one batch, retaining 8096 processed rows while leaving the final
generated token pending. After sampled cancellation the recovery correctly began
with zero cached tokens and produced 16 outputs, retaining 8144 processed rows.
The exact 8191+1 automatic boundary reused those 8144 rows; requesting two outputs
instead returned 400 without replacing completed metrics. Recreated seed history
then gave exactly the same 16-token response under ordinary decoding.

These are functional single-request observations, not repeated speed measurements
or an RKLLM comparison. Context allocation was 8192; model maximum 262144 remains
unqualified. Broad sampled output/probability quality and forced exact/full-all-mode
8192 boundary execution are separate pending scopes. The interrupted stream's exact
cached-token count is not exposed, although strict extension and scoped stop were
checked. Both original workers were restored with exact options/selection/settings
and answered fresh requests; the normal server was not restarted.
<!-- onpu-forced-batched-retention-api48:end -->

<!-- onpu-context-capacity-small-request-v1:start -->
## Context capacity allocations: small-request characterization

The [source-bound capacity experiment](docs/speculative-context-capacity-allocation-v1.json)
completed four frozen small requests per configured capacity: counting at 30 input
tokens and explanation at 27, each ordinary and speculative. All 16 output sequences
matched their frozen EOS-aware references. Target KV, prepared attention workspace
and draft proposal capacities were checked against source formulas. These are
allocation and early-prefix execution checks, **not full 16K/32K/64K prompt tests**.

| Configured capacity | Target KV capacity (MiB) | Draft proposal input capacity (MiB) | Lifetime process peak RSS (GiB) | Minimum available RAM (GiB) |
|---|---:|---:|---:|---:|
| 8,192 | 96 | 28 | 7.788 | 4.803 |
| 16,384 | 192 | 44 | 7.908 | 4.506 |
| 32,768 | 384 | 76 | 8.236 | 4.156 |
| 65,536 | 768 | 140 | 9.156 | 3.237 |

The 2 GiB reserve was preserved. Best-effort buffer inventories, process RSS,
GlobalCounters and system available RAM are distinct metrics; the listed tensor
capacities do not sum to total peak RAM. At 16K and above, `mali.configure` also
prepares a 64-query workspace. This probe explicitly wrapped ordinary generation
to 128 to match speculative prefill; the 64 slot is reported unrealized. Its logical
capacity must not be presented as measured resident memory.

The initial 16K child exited 0 and all four requests matched, but its frozen checker
incorrectly expected only 1/8/128 workspaces. That failed validation and parent
exit 1 are preserved. The source-derived corrected checker requires 1/8/64/128,
rejects a missing 64 slot, and passes 8K/16K data. A second lease tested only the
remaining 32K/64K cases with that contract and exited 0. Both leases restored the
original workers, options, selection and settings, with fresh serving replies.

The served API remains capped at 8192 for this experimental pair. The declared
model maximum 262144 still needs total-peak accounting, scaled admission and actual
long-prefix quality/latency checks. The enumerated logical capacities alone grow
to 4.883 GiB at model maximum (4.716 GiB above 8K), but exclude weights, graph arenas,
allocator/driver memory and other scratch. They cannot prove fit. Before any API
extension, ordinary/speculative prefill grouping must be aligned and the 64-row
observation path qualified if using the larger-context default.
<!-- onpu-context-capacity-small-request-v1:end -->

## Actual 16K-prefix core diagnostic

The selected 64-token path completed a 16,256-token prompt at actual context
16,384. Cold ordinary/speculative and cached ordinary/speculative cases each
emitted 16 identical output IDs within their pair. The cached control restored
the same 16,271-token target prefix and recurrent state. Core cancellation after
3 outputs released the lock; both short recovery paths matched frozen IDs. All
seven cases passed the original validator and an independent source/artifact
audit.

| Case | Prompt tokens | Cached tokens | TTFT(s) | Uncached prefill proxy(tok/s) | Decode(tok/s) | Request(s) |
|---|---:|---:|---:|---:|---:|---:|
| Cold ordinary | 16,256 | 0 | **968.662** | **16.78** | **2.87** | **973.890** |
| Cold speculative | 16,256 | 0 | 1102.276 | 14.75 | 1.72 | 1111.006 |
| Cached ordinary | 16,274 | 16,271 | **2.360** | **1.27** | **3.10** | **7.202** |
| Cached speculative | 16,274 | 16,271 | 2.392 | 1.25 | 2.03 | 9.787 |

Lower TTFT and request time are better; higher throughput is better. The ordinary
case was faster in both single-request comparisons, so speculation is not a
speed win here and should not be promoted as an automatic speed basis. The
cached rows have only 3 uncached suffix tokens; their prefill proxy includes
first-token work and is not comparable to a full-prompt prefill rate. Decode
includes iterator cleanup and synchronization. These are single diagnostic
requests with short startup warmup, not repeated benchmark results.

The 64-row observation graph ran 257 times including startup; retained projected
feature history stayed at zero. Child lifetime peak RSS was 7.578 GiB; the parent
monitor's minimum available RAM was 5.055 GiB, with the 2 GiB reserve preserved.
These values do not prove model-maximum fit. Both original workers were restored
and answered fresh requests.

The experimental DFlash2 API cap remains 8K. HTTP/API16K, sampled16K, a fresh
probability audit, broad quality, model maximum 262,144, benchmark qualification
and RKLLM speed/quality remain unqualified. See the [raw-backed diagnostic and
lifecycle evidence](docs/speculative-actual16k-core-v1.json).

## GPU/NPU shared-buffer projection operators

An experimental projection path packs GPU activations directly into imported
Rockchip buffers, runs the existing blocking NPU matrix runtime, and unpacks the
result into GPU memory. Each shape used three distinct inputs, three warmups and
nine rotated measured calls per implementation. All 144 numerical comparisons,
20 rejection checks and post-guard recoveries passed. Shared-path output bits
matched the existing NPU copy path, with zero measured NPU allocations.

| Shape (M × K × N) | GPU, FP32 activations(ms) | NPU copy, FP16 activations(ms) | NPU shared, FP16 activations(ms) | Shared speedup over NPU copy |
|---|---:|---:|---:|---:|
| 64 × 2048 × 2048 | 16.854 | 6.958 | **2.354** | 2.96× |
| 64 × 2048 × 6144 | 23.430 | 8.978 | **3.455** | 2.60× |
| 64 × 2048 × 12288 | 33.177 | 11.296 | **5.391** | 2.10× |
| 64 × 4096 × 2048 | 11.771 | 13.114 | **4.273** | 3.07× |

All paths use FP16 weights and FP32 outputs. The two NPU paths use the same
FP16-rounded activations; the GPU comparison uses FP32 activations. The latter
is a different arithmetic contract, so its ratio is not a precision-matched
model speedup. The bit-identical NPU comparison shows a 2.1–3.1× reduction in
steady-state call latency.

Times include per-call packing, cache/queue synchronization, finite-value
validation, NPU execution and GPU unpacking/completion. One-time weight
preparation, imports, allocation, compilation and validation readback are
excluded. These random operator inputs do not establish model output quality.

The helper is experimental and imperative; it is not installed in serving or
recorded by TinyJit. A JIT-visible bridge plus full-model quality/performance
checks is required before model integration. Both original workers were restored
and answered fresh requests. [Raw records, source pins and audit](docs/benchmark-gpu-npu-shared-prefill-operator-v1.json).

Portable source, validators and the four historical records are now included in
the backend at `extra/rockchip/shared_prefill/`. The helper matches the measured
source byte for byte. Packaging and CLI checks used no devices; the portable
wrapper has not been rerun on hardware.

<!-- onpu-expert-activation-residual-status-v1:start -->
### NPU expert activation-residual qualification status

The initial failed qualification is superseded by the completed v5 run above.
The compact fixture now prepares its sliced atlas on CPU before upload. A
bitwise-verified intermediate trace identified separate-product versus fused
rounding in residual dequantization; an explicit FMA now matches the independent
CPU recipe. No tolerances or admission gates changed. Qualification passes
95 CPU tests and 86 subtests, mypy, Ruff, two hardware operator tests with nine
subtests, all four synthetic model modes, and the wide vocabulary integer checks.

The trained residual candidate measures 165.93 prefill proxy tok/s and 33.28
decode tok/s, but fails the local relative-logit RMS gate (0.054358 versus 0.01).
It remains off by default and the prior preferred model row is retained. Both
original serving models were restored and answered fresh requests.
[Current trained result and restoration](docs/benchmark-npu-experts-activation-verification-v5.json).
[Historical failed checks](docs/benchmark-npu-expert-activation-residual-status-v1.json).
<!-- onpu-expert-activation-residual-status-v1:end -->

<!-- onpu-expert-firstdiff-v2:start -->
### Expert-to-attention numerical boundary

Two diagnostic traces preserve all 32 teacher logit vectors bit for bit against
their respective untraced runs. First-MoE outputs differ by about 0.01–0.02%,
while the next MoE input differs by about 0.69%. The boundary trace locates
amplification at the intervening dynamic INT8 quantizers and attention path.
A 0.0105% attention-input difference flips 131 of 163,840 prefill codes by one;
the QKV output difference becomes 0.0807%, attention output 0.2751%, attention
output projection 1.0279%, and the following normalized input 0.6866%.

Independent CPU reconstruction of the layer-2 QKV weights predicts every changed
integer sum exactly: 491,520 prefill sums and 3,840 decode sums. Decode changes
only one of 1,280 input codes at that first quantizer. Static weight scales and
FP32 dequantization also match exactly. This rules out incorrect NPU integer
products at this boundary; small expert arithmetic differences are amplified
through later quantization. The original logit gate remains failed.

These captures use shortened diagnostic generation and provide no new throughput
or comparable-quality claim. Preferred timing rows remain unchanged. Both original
serving models were restored and answered fresh requests.
[Early expert boundaries](docs/benchmark-npu-expert-firstdiff-v1-verification.json).
[Quantizer/attention boundary and independent integer verification](docs/benchmark-npu-boundary-firstdiff-v2-verification.json).
<!-- onpu-expert-firstdiff-v2:end -->

<!-- onpu-gpu-ffn-integration-status-v1:start -->
### NPU/OpenCL FFN integration: historical qualification failures

The optional `ONPU_FFN_GPU=1` integration remains disabled by default. The earlier
standalone **1.50× FFN-chain speedup** is an operator result, not a whole-model win.
These early attempts did not pass hardware numerical qualification, so no new
Qwen3 control/candidate timing was accepted at that point. The later v5 comparison
below resolves the numerical and dispatch failures; preferred rows remain unchanged
because the integrated candidate is slower in prefill.

| Integration attempt | Observed result |
|---|---|
| CPU scale arguments | Stopped at unsupported CPU-to-Rockchip buffer mapping. |
| All-device arguments | Stopped when an NPU output slice required an unavailable generic renderer. |
| Direct tile outputs | Reached GPU execution, then failed exact CPU packed-payload comparison; the first reported mismatch is one byte in a scale header (index 56: 35 versus 36). |

The six-buffer implementation uses DMA-buffer imports and writes each NPU input
tile directly. CPU checks pass: 41 tests, 9 hardware skips, 66 subtests; type checking
passes for 280 source files and Ruff passes. Native replay, trained-model equality,
and model speed were unqualified in these early attempts. The later v5 result
resolves numerical equality and replay for its checked workload; no exact-output
check was relaxed. Both original serving
models were restored and answered fresh requests after each attempt.
[Qualification failures, source hashes and restoration evidence](docs/benchmark-gpu-ffn-integration-status-v1.json).
<!-- onpu-gpu-ffn-integration-status-v1:end -->

### GPU FFN rounding arithmetic trace v5

The failing synthetic CPU/OpenCL trace exposed quotient/activation rounding
differences; the corrected v5 division trace has zero bit mismatches across
gate, up, exp argument, exp2, denominator, quotient and activation for eight
128×3072 cases. Its activation-derived scale bits also match the compiled CPU
FFN scale path in all eight cases. CPU checks pass 41 tests, 9 skips and 66
subtests; mypy passes 280 files and Ruff passes.

This is operator arithmetic evidence only: no NPU is used and no timing is
qualified. The historical GPU integration-v1 failure remains retained; this
trace does not supersede it. The v4 hardware qualification did not pass: its
changed payload/scale/down integer and native replay checks passed on eight
inputs, but its final captured GPU-call count was 3 instead of the required 1;
no model timing arms ran.
The separate CPU probe shows three separately realized outputs capture three
calls, while grouped realization captures one with identical outputs. The separately validated v5 model comparison below supersedes these integration failures: it passes the hardware and local exact-output gates, but is slower in prefill.
The feature remains off by default. No model timing, quality or RKLLM win is
claimed.
[Traces, leases, CPU checks, historical failure and v4 status](docs/benchmark-gpu-ffn-rounding-status-v2.json).


### Qwen3 direct NPU/OpenCL FFN ABBA v5

Residual-corrected GPU division and grouped output realization now pass the
full eight-input NPU/OpenCL/NPU test: both packed tiles, scale bits, downstream
integer references, input rebinding, native replay and exactly one GPU dispatch.
The four-process control/candidate/candidate/control model comparison also
completes all three warmups and three measured 128-input/64-output requests
per process at context 512.

| Aggregate role (six measured requests) | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| Control | 284.24 | 450.32 | 37.5766 |
| Candidate | 309.32 | 413.81 | 37.8200 |

All four processes preserve 32 full-vocabulary teacher vectors bit for bit and
all 24 generated sequences. Candidate prefill is slower than its matched control;
decode uses the unchanged CPU/NPU path and is nearly unchanged. The candidate
records 28 GPU FFN calls in prefill and zero in decode. Native FROM_DEVICE
invalidation stays disabled in both roles. The earlier standalone 1.50× chain
result does not predict this integrated full-model result.

`ONPU_FFN_GPU` remains off by default and the older preferred Qwen3 row is retained.
The arithmetic and dispatch failures above are historical; they are resolved
for this qualified workload, while performance remains unsuccessful. Four valid
timing variants are added (80 total). Local numerical equality does not establish
held-out, HF FP16, or comparable RKLLM quality. Both original serving sessions
were restored and answered fresh requests.
[Raw reports, comparisons, source hashes and restoration](docs/benchmark-qwen3-gpu-ffn-verification-v5.json).

A read-only follow-up on the existing GDB capture verifies a DPU conversion
source at BO offset 0x420000 and destination at 0x840000. That destination
matches the next FP16 task's CNA register used as RHS by the local renderer.
This establishes address linkage only: buffer bytes, tensor semantics and
packing equivalence require a fresh bounded payload capture before any port
or speed claim.
[Captured addresses and source hash](docs/benchmark-dpu-conversion-address-inventory-v1.json).


### CPU/OpenCL host-attention diagnostics v1–v6

Six FP32 OpenCL prototypes directly import owned, page-aligned CPU Tensor mmaps.
All 288 numerical cases pass the same relative-RMS limit of 1e-5, including partial
prompts, longer prefixes and masked future NaNs. The largest observed error is
below 6e-7. These tests do not establish bitwise equality or comparable model quality.

Query/GQA reuse improves some larger-prefix operator cases: the four-query version
records paired CPU/GPU ratios of about 1.25 at end 256 and 1.67 at end 512. The
initial 128-token case does not show a robust benefit, and 17-row partial prompts
lose. Early GPU samples often differ substantially from later samples; CPU/GPU
frequencies vary and memory stays at 528 MHz in these isolated tests, whereas
mixed NPU model runs record 2.112 GHz. These measurements are not substitutes for
full-model request timing. These CPU-backed adaptations are not enabled in serving.

Independent checks found:

- The old and current CPU attention generators compile to byte-identical ELF for
  the checked one-row and 128-row Qwen3 shapes on the current toolchain.
- Native quantizer input writes into owned NPU memory are 7–12% faster than normal
  CPU packing plus copying across six tested shape/sync combinations, with identical
  payload and scale bits. This rules out that mechanism as a demonstrated regression
  in those isolated cases; it is not a model speed claim.
- Idle-priority CPU wake workers, with or without explicit attention-worker placement,
  did not establish a useful speed benefit in the simulated prefill sequence and remain unselected.

A controlled maximum-clock comparison needs an administrator-held frequency lease.
The reviewable helper changes only supported minimum-frequency requests and restores
them afterward; the agent has not applied it. Model rows and all 80 valid model
variants remain unchanged. Both original serving sessions were restored and answered
fresh requests after each completed GPU lease.
[Raw prototypes, diagnostics, clock conditions and restoration evidence](docs/benchmark-gpu-host-attention-series-verification-v1.json).

### Qwen3 static SwiGLU fragment loads v1

The CPU SwiGLU packer can resolve three-core gate/up fragment addresses before
entering the channel loops. The complete operator is 1.056x faster at Qwen3's
128x3072 geometry by the median paired ratio; all 54 cases preserve payload
and scale bits. The full-model comparison below uses four ABBA processes,
three warmups and three measured 128-input/64-output requests per process,
context 512, identical packed weights and native replay, with only
`ONPU_SWIGLU_PACK_STATIC_LAYOUT` changed.

| Aggregate role (six measured requests) | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| Control | 300.61 | 425.81 | 37.7555 |
| Candidate | 285.04 | 449.06 | 37.7783 |

All four processes preserve 32 full-vocabulary teacher vectors bit for bit
and all 24 generated sequences. Captured source inspection confirms 28 pack
calls in each phase and the intended fragment-load implementation in each
role. These are adaptive-clock measurements; the operator ratio must not be
applied to whole-model throughput. The option remains off by default and
the older preferred Qwen3 row is retained. Comparable RKLLM quality and an
RKLLM prefill win remain unqualified. CPU verification passed 80 tests and
74 subtests (nine explicit hardware skips), mypy on 280 files, and Ruff.
The initial 700 MiB CPU test attempt exhausted its cap during worker startup;
the successful retry used 2 GiB. Both original serving models were restored
and answered fresh requests after CPU checks and after the model comparison.
The original model owner encountered a stdout BrokenPipeError while recording
restoration; its failed job is preserved and its final exit code is unknown.
A separate successful recovery check verified both original options, selection,
settings, no remaining benchmark processes and fresh API replies. Completed
model measurements were revalidated without rerunning them.
[Raw measurements, exact local checks and restoration](docs/benchmark-qwen3-swiglu-static-verification-v1.json).

### QKV preparation without head normalization: three-model ABBA v1

The new explicit `--fused-qkv-without-norm` option skips Q/K normalization
when the checkpoint has none. Qwen2 keeps dense bias addition before fused
RoPE/cache preparation; MiniCPM4 and TinyLlama consume raw INT8 projection
tiles directly. Existing geometry, full-head RoPE, context and attention-feature
guards remain. Four ABBA processes per model use the same checkpoint/input,
128 inputs, 64 outputs, context 512, three warmups and three measured requests.
Only this QKV option changes between each matched control/candidate pair.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate |
|---|---:|---:|
| Qwen2/0.5B | 252.30 / 255.74 | 37.244 / 37.618 |
| MiniCPM4/0.5B | 539.71 / 607.20 | 44.807 / 45.982 |
| TinyLLAMA/1.1B | 183.06 / 189.40 | 24.216 / 24.338 |

All 24 generated sequences per model match each other and the previous
preferred sequence. All four sets of 32 full-vocabulary teacher vectors per
model match bit for bit. Captured native source confirms prepared QKV in
every candidate attention block during prefill and zero such calls in decode;
the candidate source has no normalization reduction. The independent audit
and final comparison recheck report/NPZ hashes and full-vector equality.
These are adaptive-clock measurements. They do not exceed the older preferred
prefill rates, so preferred main-table rows stay unchanged and the option
remains opt-in. Comparable HF/RKLLM quality remains unqualified.

CPU checks passed 77 tests plus 35 subtests, with seven explicit hardware skips;
mypy checked 280 files and Ruff passed. Two failed test-fixture attempts are
preserved; the final fixture uses the existing QKV test's precompiled function
boundary and fixed-capacity NumPy readback. No numerical assertion was relaxed.
All twelve model processes exited successfully and cleaned up. Both original
serving sessions were restored with their original settings/selection and
answered fresh requests. Twelve actual timing variants bring the inventory to 96.
[Full runs, source hashes, numerical checks and restoration](docs/benchmark-qkv-without-norm-models-verification-v1.json).

### Explicit CPU prefill pooling: wide-FFN ABBA v1

Explicit `--cpu-prefill-pool` now creates the requested parked-worker pool even
when two KV heads and context 512 do not qualify for decode attention pooling.
The existing wide-FFN sigmoid path becomes available during prefill. Both arms
request `--cpu-pool`; only the prefill-pool switch changes. New QKV preparation,
static SwiGLU packing, GPU FFN and native FROM invalidation remain disabled.
Four ABBA processes per model use 128 inputs, 64 outputs, context 512,
three full warmups and three measured requests per process.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate |
|---|---:|---:|
| Qwen2/0.5B | 262.83 / 373.38 | 37.429 / 37.934 |
| TinyLLAMA/1.1B | 183.97 / 267.10 | 24.353 / 24.418 |
| Qwen2.5/1.5B | 109.27 / 158.52 | 15.493 / 15.866 |

All 24 generated sequences per model match each other and the previous
preferred sequence; all four sets of 32 full-vocabulary teacher vectors per
model match bit for bit. Captured source verifies nonzero parked-worker
callbacks in 24, 22 and 28 sigmoid kernels respectively, only during prefill.
Each teacher check records positive pool dispatches. Creating a pool also
enables its existing decode affinity policy; decode is measured rather than
attributed solely to the prefill sigmoid change.

The main CPU+NPU rows select the slower-TTFT candidate process per model.
Both candidates individually improve prefill and decode over the previous
preferred row. Each selected row is an actual three-warmup/three-measurement
run; the table above aggregates six measured requests per role. Existing
adaptive clocks remain unchanged. Comparable HF/RKLLM quality and RKLLM
prefill wins remain unqualified; the option is still explicit.

CPU verification passed 82 tests and 35 subtests, with seven explicit hardware
skips; mypy checked 280 files and Ruff passed. A fixture-only formatting edit
was verified to have identical ASTs under the benchmark interpreter. All
twelve model processes exited successfully and cleaned up. Both original
serving models were restored with original options/settings/selection and
answered fresh requests. The timing inventory now contains 108 valid variants.
[Full reports, selection policy and restoration](docs/benchmark-explicit-prefill-pool-models-verification-v1.json).


A separate CPU-only prototype folds raw INT32 dequantization, SwiGLU and
independent split-K quantization into one pack function. Eight cases preserve
payload and scale bits across widths 4,864, 5,632 and 8,960, partial rows and
fragmented raw projections. This prototype is not integrated or timed.
[Numerical prototype](docs/benchmark-grouped-swiglu-pack-operator-v1.json).

### Grouped SwiGLU v1: arithmetic mismatch detected

The optional raw SwiGLU packer passed its CPU and full-FFN hardware test,
but the full-model gate stopped at TinyLlama because generated IDs differed.
Six model processes completed; remaining cases were not run. These attempts
do not replace any canonical timing or preferred row; the inventory stays at 108.
A CPU trace against the serving sigmoid helper identifies a rounding difference:
`gate / denominator` does not preserve materialized sigmoid followed by
gate/up multiplication. Direct division changes payload and scale bits in
all four traced geometries; explicit multiplication matches exactly. The
original tensor-only test reference allowed sigmoid algebraic fusion and
did not cover this serving boundary. The option remains off by default.
Both original serving models were restored and answered fresh requests.
[Failed runs, arithmetic diagnosis and restoration](docs/benchmark-grouped-swiglu-integration-status-v1.json).

### Grouped SwiGLU v2: materialized-sigmoid arithmetic

The corrected optional `--fused-swiglu-ffn` combines raw INT32 dequantization,
SwiGLU and independently scaled split-K packing. It preserves the serving
order: sigmoid reciprocal is rounded before gate and up multiplication.
The failed division-form v1 and the stricter serving-reference diagnosis
remain recorded above. Both roles request CPU prefill pooling; only the new
raw-FFN switch changes. Four ABBA processes per model use 128 inputs, 64
outputs, context 512, three warmups and three measured requests per process.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate |
|---|---:|---:|
| Qwen2/0.5B | 392.70 / 507.22 | 37.995 / 37.978 |
| TinyLLAMA/1.1B | 280.40 / 337.78 | 24.054 / 24.112 |
| Qwen2.5/1.5B | 158.72 / 230.75 | 15.905 / 15.857 |

All 24 generated sequences per model match the previous preferred sequence,
and all four sets of 32 teacher vectors per model match bit for bit. The
exclusive full-FFN NPU/native-replay test also passed with the serving sigmoid
helper as reference. Captured source confirms the new packer in every
candidate FFN prefill block and no such calls in decode. Controls retain the
separate pooled sigmoid. Clocks remain adaptive; comparable HF/RKLLM quality
and all-model RKLLM wins remain unqualified.

Preferred rows change only where both candidate processes individually improve
both phases over the previous preferred timing; selection uses the slower
candidate TTFT. Other preferred rows remain unchanged. Each selected row is
one actual three-warmup/three-measurement process, while the table above
aggregates six measured requests per role.

CPU verification passed 91 tests and 35 subtests, with 9 explicit hardware skips;
mypy checked 280 files and Ruff passed. The initial typecheck failure was a
reused optional local binding; it was renamed before the successful checks.
The qualification and twelve model processes exited successfully and cleaned
up. Both original serving models were restored and answered fresh requests.
The timing inventory now contains 120 valid variants.
[Full runs, arithmetic checks, selection and restoration](docs/benchmark-grouped-swiglu-models-verification-v2.json).

### Narrow INT8 decode CBUF: numerical qualification v2

A bounded single-core probe verifies M=1,K=4096 at per-core output widths
320,352,640,704,832 and864. Changing both writes of CNA_CBUF_CON0 from
0x48 to0xb1 preserves all60 independent INT32 checks across zeros, extrema,
alternating signs and random inputs. Output buffers are poisoned before every
arm to reject stale results. These are correctness checks, not timing or
model wins; production settings remain unchanged. The GPU model stayed
resident, and both original models answered fresh requests after NPU restoration.
[Commands, golden outputs, source hashes and restoration](docs/benchmark-narrow-decode-cbuf-verification-v2.json).

### Narrow INT8 decode CBUF: paired primitive measurements v3

A new single-core M=1,K=4096 sweep changes only the two CBUF register
writes. It passes 180 independent integer checks and rechecks every retained
timed output (2,880 checks). Three warmups precede 40 alternating ABBA/BAAB
blocks per shape, 80 calls per arm. Timing covers blocking submission;
command preparation, uploads and output checks are outside the interval.
Existing adaptive clocks remain unchanged. N=512 and N=1024 are unchanged
controls, already using one input/eleven weight banks.

| Per-core N | Baseline ms | One-input-bank ms | Median paired speedup | Faster blocks |
|---:|---:|---:|---:|---:|
| 128 | 0.144373 | 0.135624 | 1.064x | 38/40 |
| 192 | 0.138248 | 0.123373 | 1.120x | 40/40 |
| 256 | 0.180685 | 0.160123 | 1.120x | 37/40 |
| 320 | 0.210873 | 0.184914 | 1.140x | 40/40 |
| 352 | 0.176748 | 0.150790 | 1.173x | 40/40 |
| 416 | 0.270664 | 0.238434 | 1.126x | 39/40 |
| 448 | 0.296913 | 0.259726 | 1.145x | 40/40 |
| 512 | 0.337600 | 0.337308 | 1.001x | 21/40 |
| 576 | 0.308871 | 0.270663 | 1.141x | 39/40 |
| 608 | 0.438077 | 0.385870 | 1.136x | 40/40 |
| 640 | 0.336579 | 0.294434 | 1.143x | 40/40 |
| 704 | 0.497723 | 0.437640 | 1.140x | 40/40 |
| 768 | 0.533452 | 0.472494 | 1.130x | 39/40 |
| 832 | 0.383099 | 0.327684 | 1.168x | 39/40 |
| 864 | 0.433411 | 0.375370 | 1.168x | 40/40 |
| 896 | 0.410515 | 0.350141 | 1.171x | 40/40 |
| 960 | 0.437932 | 0.372745 | 1.175x | 40/40 |
| 1024 | 0.396078 | 0.396078 | 1.000x | 21/40 |

These are primitive measurements; no full-model timing is replaced and no
production default changes. The GPU model stayed resident. Both original
sessions were restored and answered fresh requests.
[Full samples, register hashes, exact outputs and restoration](docs/benchmark-narrow-decode-cbuf-verification-v3.json).

### Narrow CBUF combined v2: INT8 decode input and weight banks

The decode-only narrow-CBUF option changes captured INT8 decode commands from
CBUF 0x48 to 0xb1. Both roles use CPU pooling, CPU prefill pooling, and fused
SwiGLU; only `ONPU_INT8_DECODE_NARROW_CBUF` changes (0 control, 1 candidate).
QKV-without-norm, static SwiGLU, GPU FFN, and native FROM remain off. Four ABBA
processes per model use 128 inputs, 64 outputs, context 512, three warmups and
three measured requests per process.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate |
|---|---:|---:|
| Qwen2/0.5B | 493.03 / 498.15 | 37.771 / 38.687 |
| TinyLLAMA/1.1B | 342.42 / 339.64 | 24.044 / 25.004 |
| InternLM2/1.8B | 259.66 / 248.03 | 14.849 / 15.367 |

For each model all 24 generated sequences match. Each of four processes emitted
32 full-vocabulary FP32 teacher vectors, and the four arrays match bitwise.
These matched-output checks establish local consistency; HF/RKLLM quality and
the all-model speed goal remain unqualified. Clocks were adaptive.

The earlier narrow-CBUF primitive tests showed a 6–17% decode improvement;
that is a primitive-level result, not a full-model claim. Preferred rows change
only when both candidate processes individually improve TTFT and decode over
the previous preferred row; selection uses the slower-TTFT candidate. The
qualified timing inventory contains 132 variants. The v1 TinyLLAMA control
setup interruption was a monitor status-read timeout with zero measured trials,
not a numerical failure; its artifact is retained with this report.
[Combined runs, evidence, interruption, and selection](docs/benchmark-narrow-cbuf-models-combined-verification-v2.json).

### FFN prefill pool v2: pooled raw SwiGLU and GELU packing

All four models use the raw FFN packer in both roles and CPU attention pooling.
The control passes no raw-FFN pool callback; the candidate enables pooled
workers for raw FFN packing. Both roles use narrow-CBUF decode and keep GPU FFN
and native FROM off. Gemma2 retains file-backed CPU weights, deferred weight
realization, CPU GELU, CPU softcap, and three NPU domains in both roles. Four
ABBA processes per model use 128 input IDs, 64 outputs, context 512, three
warmups, and three measured requests per process.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate | Measured requests, control / candidate |
|---|---:|---:|---:|
| Qwen2/0.5B | 481.74 / 503.99 | 38.676 / 38.617 | 6 / 6 |
| TinyLLAMA/1.1B | 335.23 / 347.20 | 24.954 / 25.038 | 6 / 6 |
| InternLM2/1.8B | 252.74 / 252.51 | 15.149 / 14.746 | 6 / 6 |
| Gemma2/2B | 174.11 / 174.21 | 9.654 / 9.678 | 6 / 6 |

Each model has 32 full-vocabulary FP32 teacher vectors in each process; all four
arrays match bitwise, and all 24 generated sequences match. NPU/native-replay
qualification passed for SiLU and GELU at 17/128 rows with seven changed inputs.
The earlier CPU prototype passed eight tests, 800 exact timed output checks,
and 46 CPU tests with two hardware skips; mypy checked 280 files and Ruff
passed. Comparable HF/RKLLM quality and all-model speed qualification remain
open. Clocks were adaptive.

Preferred rows change only when both candidate processes individually improve
TTFT and decode over the previous preferred row; choose the slower-TTFT
candidate. The FFN study contributes 16 valid timing reports and brings the
valid timing inventory to 148. The earlier v1 qualification failure was a
lazy native-plan test-order issue; it ran no model timings and its job, report,
log, failed test source, and correction diagnosis are retained in the evidence.
[Full runs, qualification, source evidence, prior failure and restoration](docs/benchmark-ffn-prefill-pool-models-verification-v2.json).

### CPU MLA latent projections: MiniCPM3 full-model v1

The candidate adds only `--cpu-mla-projections` to the established CPU MLA
weights/attention baseline. Both arms use 124 prepared CPU weights, 62 CPU MLA
attention blocks, three NPU domains, deferred realization, no file-backed
weights, and zero GPU native calls. The candidate prepares 62 CPU MLA
projection blocks (162,529,280 packed bytes); the control prepares none.
Both prefill and decode call the serial native projection helper 124 times in
the candidate; neither arm profiles native calls or invokes native GPU work.
Four ABBA processes use 128 input IDs, 64 output tokens, context 512, three
warmups and three measured requests per process.

| Role | Aggregate median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| control | 10944.79 | 11.70 | 3.942 |
| candidate | 8092.56 | 15.82 | 4.578 |

All 24 warmup/measured 64-token sequences match the previous preferred sequence. Four
32-row full-vocabulary FP32 teacher arrays match bitwise across all four processes,
and their float64 NLL values were independently recomputed. This validates the
fixed-prefix local numerical gate, not comparable model quality or a speed
goal. Adaptive clocks were observed; process RSS is not exclusive physical
memory. The unrelated MLA projection primitive's speedup is not a full-model
claim.
[Full reports, numerical evidence, CPU checks and restoration](docs/benchmark-cpu-mla-projections-models-verification-v1.json).

### LFM2.5-VL/450M: CPU SwiGLU prefill worker options v1


The candidate enables only the existing CPU SwiGLU prefill worker options:
`--cpu-prefill-pool`, `--share-projection-weights`, and
`--fused-swiglu-ffn`. Control keeps those three options off. Both roles retain
CPU INT8 math, one NPU domain, 16 fused FFN blocks, six attention blocks, and
zero native GPU calls. LFM2.5-VL configuration is hidden size 1024 and FFN
intermediate size 4608 (not 6656), with ten short-convolution and six
attention layers. Four ABBA processes use 128 input IDs, 64 generated tokens,
context 512, three warmups and three measured requests each.

| Candidate process | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| lfm450-options-models-v1-lfm25-vl-450m-2-candidate selected | 161.745 | 791.37 | 60.368 |
| lfm450-options-models-v1-lfm25-vl-450m-3-candidate | 139.638 | 916.66 | 60.844 |

| Role aggregate | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 333.442 | 383.88 | 59.218 |
| candidate | 6 | 156.697 | 816.86 | 60.529 |

The table above reports both actual candidate processes. Preferred selection is
conservative: both candidates must individually beat the previous preferred
report in TTFT and decode, then the slower-TTFT candidate is selected. The
role-level summary uses all six measured requests per role. All 24 warmup and
measured 64-token sequences match; all four 32-row full-vocabulary FP32 teacher
arrays match bitwise, with float64 NLL recomputed independently. The new timing
remains below the RKLLM reference in both prefill and decode. Comparable output
quality and the all-model speed goal remain unqualified. Clocks were adaptive;
RSS is lifetime process peak, not exclusive physical memory.
[Full reports, numerical audit, source pins and restoration](docs/benchmark-lfm450-options-models-v1-verification.json).

### LFM2.5-VL/450M: narrow decode CBUF v1


Both roles retain `--cpu-prefill-pool`, `--share-projection-weights`, and
`--fused-swiglu-ffn`; only narrow decode CBUF changes (control 0, candidate
1). Both roles use CPU INT8, one NPU domain, 16 fused FFN blocks, six attention
blocks, and no native GPU calls. The 128-input/64-output, context-512 ABBA
run uses three warmups and three measured requests per process.

| Candidate process | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| lfm450-narrow-cbuf-models-v1-lfm25-vl-450m-2-candidate | 189.150 | 676.71 | 61.546 |
| lfm450-narrow-cbuf-models-v1-lfm25-vl-450m-3-candidate | 183.639 | 697.02 | 61.170 |

| Role aggregate | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 159.998 | 800.01 | 59.382 |
| candidate | 6 | 187.335 | 683.27 | 61.290 |

Prepared decode commands have M=1, K=4096 and per-core N=320 or 352. The two
CBUF values are 0x48/0x48 for control and 0xb1/0xb1 for candidate; no prefill
commands use the narrow buffer. All 24 warmup and measured 64-token sequences
match the prior preferred output. Four full-vocabulary FP32 teacher arrays are
bitwise identical and their float64 NLL was independently recomputed.

The candidates do not both improve TTFT and decode over the current preferred options process. Keep the existing preferred row and report both measured candidates without promotion. The candidate-role aggregate is slower than RKLLM in both phases. Quality and the all-model speed goal
remain unqualified. Clocks were adaptive; process RSS is not exclusive physical
memory.
[Reports, comparison, numerical audit and restoration](docs/benchmark-lfm450-narrow-cbuf-models-v1-verification.json).

### Qwen2/0.5B: natural K896 full-model v1

Both roles keep CPU prefill pooling and raw SwiGLU FFN pooling enabled, with
24 fused FFN blocks, 24 attention blocks, 49 CPU norms, no prepared QKV, and
shared projection packing disabled. The only role change is
`ONPU_INT8_EXPERT_DEPTHS=0/1`. Control uses physical input K=1024; candidate
uses natural K=896. Both remain CPU INT8 with NPU work enabled, one domain,
and native GPU calls disabled. Four ABBA runs use 128 inputs, 64 outputs,
context 512, three warmups and three measured requests per process.

| Candidate process | TTFT ms | Prefill proxy tok/s | Decode tok/s | Role |
|---|---:|---:|---:|---|
| qwen2-natural896-models-v1-qwen2-05b-2-candidate | 268.055 | 477.51 | 40.668 | candidate |
| qwen2-natural896-models-v1-qwen2-05b-3-candidate | 248.047 | 516.03 | 40.510 | candidate |

| Role aggregate | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 248.801 | 514.47 | 37.895 |
| candidate | 6 | 258.051 | 496.03 | 40.625 |

Each phase records 237 prepared input-width commands. Natural candidates use
K=896 and controls K=1024, with the actual width fields, register values,
three-core splits, and invalidation evidence retained. The separate golden
probe checks 48 shape/arm/input combinations against int32 references with
zero max error and no poison remaining. All 24 full-model generated sequences
match the prior preferred output; four full-vocabulary FP32 teacher arrays are
bitwise equal and float64 NLL was recomputed.

The candidate pair does not satisfy the two-process promotion gate; keep the existing preferred variant and publish these measurements without promotion. Candidate aggregate is slower than RKLLM in both timing phases. Model quality and the all-model speed goal remain unqualified. Clocks were adaptive; process RSS is not exclusive physical memory.
[Model reports, comparison, golden probe, numerical audit and restoration](docs/benchmark-qwen2-natural896-models-verification-v1.json).

### Qwen2/0.5B: current native phase profile v3

This diagnostic profiles one warmed request after four warmups, with 128 input
tokens, 64 outputs and context 512. Native counts confirm one prefill and 63
decode transitions; no warmup counts are included. Both processes reproduce
all five prior 64-token sequences. Control uses padded K1024, candidate K896.

| Recipe | Phase | CPU call ms | NPU call ms | Copy call ms |
|---|---|---:|---:|---:|
| control | prefill | 144.104 | 66.869 | 23.960 |
| control | decode | 3.947 | 20.181 | 0.342 |
| candidate | prefill | 144.429 | 64.929 | 24.528 |
| candidate | decode | 3.954 | 18.461 | 0.355 |

Decode values are per transition. These are instrumented call wall times,
including driver and cache work, rather than pure hardware busy time. The
largest prefill labels are CPU raw SwiGLU packing (55.54/59.64 ms), CPU
attention (38.94/35.68 ms), and the combined gate/up NPU projection
(30.23/29.04 ms), shown control/candidate. CPU work totals about 144 ms in
both prefill profiles. A separate materialized sigmoid costs about 1.4 ms
per decode transition; the current raw FFN path excludes one-token decode.

The initial diagnostic stopped at its native-run assertion after three
warmups. Its restoration and fresh-request receipts are preserved. The
corrected protocol retains that assertion and uses four warmups. Both
original serving sessions were restored and answered fresh requests.
These profiles do not change the 164 valid timing variants or preferred
model rankings. Comparable RKLLM quality and the all-model speed goal remain
unqualified.
[Profiles, source inventory, independent audit and restoration](docs/benchmark-qwen2-phase-profile-verification-v3.json).

### Experimental raw SwiGLU decode integration v1

An opt-in `--fused-swiglu-decode` path now reuses the raw FFN packing helper
for one-row CPU FP32 decode, retaining per-K-group quantization and materialized
sigmoid rounding. The CPU gate passed 41 tests (two hardware skips), mypy on
281 files and repository Ruff. NPU/native replay and full-model speed checks
are pending. The option defaults off and does not promote a benchmark row.
[CPU evidence and source changes](docs/benchmark-raw-swiglu-decode-cpu-verification-v1.json).

### Raw SwiGLU decode placement qualification v2

The decode option and reported block count now require all gate/up and down
projections to select NPU decode. Mixed placement retains the ordinary path
and reports zero eligible decode blocks. Updated CPU checks passed 44 tests
with three hardware skips, mypy on 281 files and repository Ruff. Both serving
sessions answered fresh requests after restoration. NPU/native replay and
full-model performance remain pending; no preferred timing row changes.
[Current CPU qualification](docs/benchmark-raw-swiglu-decode-cpu-verification-v2.json).

### Qwen2/0.5B: raw SwiGLU decode v1

Both roles retain raw FFN prefill pooling, CPU prefill pooling, 24 fused SwiGLU
blocks, and 24 attention blocks. The candidate enables the serial raw SwiGLU
decode path for 24 blocks; the control keeps that decode path disabled. The
only varied model flag is `--fused-swiglu-decode`; expert depths stay enabled
and narrow CBUF stays off in both roles. Four ABBA runs use 128 input IDs, 64
outputs, context 512, three warmups and three measured requests per process.

| Candidate process | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| qwen2-raw-decode-models-v1-qwen2-05b-2-candidate | 278.487 | 459.63 | 41.796 |
| qwen2-raw-decode-models-v1-qwen2-05b-3-candidate | 262.430 | 487.75 | 41.700 |

| Role aggregate | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 253.359 | 505.21 | 40.563 |
| candidate | 6 | 270.458 | 473.27 | 41.748 |

Both roles use 237 prepared input-width commands per phase with physical K=896.
Raw FFN helper calls remain at 24 in prefill for both arms; decode helper calls
are 0 in control and 24 in candidate. All 24 warmup and measured 64-token
sequences match; four full-vocabulary FP32 teacher arrays are bitwise equal
and their float64 NLL was independently recomputed. The one-row materialized
INT8/native-replay hardware qualification passed separately.

The candidate pair does not meet the two-process promotion gate; the previous preferred variant is retained. The candidate aggregate measures 473.27 prefill proxy tok/s and 41.748 decode tok/s, versus RKLLM's published 877.31 and 41.58. This does not establish comparable-quality performance. Full-model quality and the all-model speed goal remain unqualified. Clocks were adaptive; process RSS is not exclusive physical memory.
[Timing, qualification, numerical audit and restoration](docs/benchmark-qwen2-raw-decode-models-verification-v1.json).

### CPU raw FFN row-batching prototype v1

This CPU-only prototype interleaves two or four rows in the existing raw
SwiGLU packer. All 174 payload/scale comparisons and 640 timed output checks
were exact; two dense Tensor references also matched in the probe. Both
serving models stayed resident and answered fresh requests. No accelerator
was opened, and no model timing row changes.

| Hidden width | Row batch | Baseline median ms | Candidate median ms | Paired speedup | Faster blocks |
|---:|---:|---:|---:|---:|---:|
| 4864 | 2 | 3.103 | 2.954 | 1.114x | 29/40 |
| 4864 | 4 | 3.077 | 2.530 | 1.219x | 33/40 |
| 4608 | 2 | 2.955 | 2.602 | 1.143x | 38/40 |
| 4608 | 4 | 2.972 | 2.496 | 1.198x | 34/40 |

Each arm has 80 timed calls in 40 alternating ABBA/BAAB blocks. Speedups
are medians of paired block ratios. Clocks were adaptive; output checks and
poisoning were outside timing. Production integration and full-model gains
remain unqualified.
[Operator samples, generated C, source pins and independent audit](docs/benchmark-ffn-row-batch-cpu-verification-v1.json).

### Direct raw FFN row-batching generator v2

The proposed production generator emits row batches directly rather than
rewriting C text. It passed the same 174 exact payload/scale comparisons,
two dense-reference assertions and 640 checked timing calls. Single-row code
remains the original program. Both serving models remained loaded and
answered fresh requests; no accelerator opened.

| Hidden width | Row batch | Baseline median ms | Candidate median ms | Paired speedup | Faster blocks |
|---:|---:|---:|---:|---:|---:|
| 4864 | 2 | 3.484 | 3.196 | 1.095x | 23/40 |
| 4864 | 4 | 3.085 | 2.521 | 1.229x | 36/40 |
| 4608 | 2 | 3.332 | 3.060 | 1.034x | 20/40 |
| 4608 | 4 | 3.337 | 3.150 | 1.118x | 21/40 |

The four-row Qwen-width result is more consistent than the liquid-model
width result. These are operator measurements under adaptive clocks, with
checks outside timing. Full-model benefits remain unqualified and the
168 valid LLM timing variants are unchanged.
[Direct generator, samples and independent audit](docs/benchmark-ffn-row-batch-cpu-verification-v2.json).

### Experimental raw FFN row-batching integration v1

The direct row-batching generator is now available behind
`ONPU_SWIGLU_PREFILL_ROWS=2/4`; its default remains 1. CPU integration checks
passed 51 tests with three hardware skips, mypy on 281 files and repository
Ruff. Dynamic partial rows, exact payload/scales, inactive data and unchanged
one-row/GELU selection are covered. NPU and model measurements remain
pending; the 168 valid timing variants are unchanged.
[Integrated CPU qualification](docs/benchmark-ffn-row-batch-integration-cpu-verification-v1.json).

### Qwen2/0.5B: four-row SwiGLU prefill v1

Both roles retain raw FFN prefill pooling, CPU prefill pooling, 24 fused SwiGLU
blocks, and 24 attention blocks. Both roles keep serial raw SwiGLU decode enabled for 24 blocks. The only
varied setting is `ONPU_SWIGLU_PREFILL_ROWS=1/4`; expert depths stay enabled
and narrow CBUF stays off in both roles. Four ABBA runs use 128 input IDs, 64
outputs, context 512, three warmups and three measured requests per process.

| Candidate process | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| qwen2-row-batch-models-v1-qwen2-05b-2-candidate selected | 242.799 | 527.19 | 41.919 |
| qwen2-row-batch-models-v1-qwen2-05b-3-candidate | 238.218 | 537.32 | 42.097 |

| Role aggregate | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 248.232 | 515.65 | 41.992 |
| candidate | 6 | 240.508 | 532.21 | 42.002 |

Both roles use 237 prepared input-width commands per phase with physical K=896.
Raw FFN helper calls remain at 24 in prefill for both arms; decode helper calls
are 24 in both roles. Candidate prefill source contains four-row batching;
control prefill and both serial decode sources do not. All 24 warmup and measured 64-token
sequences match; four full-vocabulary FP32 teacher arrays are bitwise equal
and their float64 NLL was independently recomputed. The SiLU/GELU full-FFN
INT8/native-replay qualification passed at 17 and 128 rows with seven changed
inputs each.

Both candidates individually beat the prior preferred report in TTFT and decode; the slower-TTFT candidate is selected. The candidate aggregate measures 532.21 prefill proxy tok/s and 42.002 decode tok/s, versus RKLLM's published 877.31 and 41.58. This does not establish comparable-quality performance. Full-model quality and the all-model speed goal remain unqualified. Clocks were adaptive; process RSS is not exclusive physical memory.
[Timing, qualification, numerical audit and restoration](docs/benchmark-qwen2-row-batch-models-verification-v1.json).

### Qwen2/0.5B: combined current-recipe QKV v1

Both roles retain four-row raw SwiGLU prefill, fused decode and natural K896.
Only the candidate adds `--fused-qkv-without-norm`; actual prefill QKV helper
calls are 24 versus zero in control, with zero in decode for both. All 24
generated sequences and four full-vocabulary FP32 teacher arrays match the
current preferred output bitwise.

| Role | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 240.482 | 532.26 | 42.014 |
| candidate | 6 | 243.072 | 526.59 | 41.999 |

The candidate pair did not both improve prior preferred TTFT and decode.
The preferred row remains the four-row SwiGLU recipe. Local clocks were
adaptive; comparable RKLLM quality and the all-model goal remain unqualified.
[Raw reports, independent audit and restoration](docs/benchmark-qwen2-combined-qkv-models-verification-v1.json).

### CPU core affinity and attention query tiles v1

Device-tree and runtime checks identify cores 0–3 as Cortex-A55 efficiency
cores and 4–7 as Cortex-A76 performance cores. Benchmark processes use
`taskset -c 4-7`; monitors use 0–3. Persistent worker masks are 5, 6 and 7,
with pooled caller work on 4. Frequencies remain dynamic; affinity does not
lock either A76 pair to its maximum frequency.

The CPU attention probe tests existing query/PV tile options for Qwen2
geometry (14 query heads, 2 KV heads, head dimension 64, context 512).
All 96 output comparisons against the 4x2 baseline and 1600 timed checks
were bitwise exact. Both serving sessions remained resident and answered
fresh requests; only CPU/NPY devices opened.

| Total KV tokens | Query x channel tile | Baseline median ms | Candidate median ms | Paired speedup | Faster blocks |
|---:|---:|---:|---:|---:|---:|
| 128 | 8x1 | 2.140 | 2.053 | 1.039x | 33/40 |
| 128 | 8x2 | 2.150 | 1.972 | 1.090x | 37/40 |
| 128 | 8x4 | 2.130 | 2.464 | 0.861x | 0/40 |
| 128 | 16x1 | 2.132 | 2.151 | 0.992x | 18/40 |
| 128 | 16x2 | 2.152 | 2.138 | 1.007x | 23/40 |
| 512 | 8x1 | 7.718 | 7.278 | 1.069x | 36/40 |
| 512 | 8x2 | 7.990 | 6.712 | 1.187x | 39/40 |
| 512 | 8x4 | 7.792 | 8.799 | 0.886x | 2/40 |
| 512 | 16x1 | 7.728 | 7.463 | 1.030x | 30/40 |
| 512 | 16x2 | 7.770 | 6.750 | 1.162x | 39/40 |

Each arm has 80 timed calls in 40 alternating ABBA/BAAB blocks. Readback,
checks and output poisoning are outside timing. The 8x2 candidate is selected
for full-model qualification; no model timing row changes. There is no
independent Tensor oracle in this probe, and comparable RKLLM quality is
unqualified.
[Core masks, samples, generated C and independent audit](docs/benchmark-attention-query-tiles-cpu-verification-v1.json).

### MiniCPM4/0.5B: CPU pooling and QKV preparation v1

Four ABBA processes used 128 input tokens, 64 output tokens and context 512.
All 24 generated sequences and four full-vocabulary FP32 teacher arrays
match the pinned prior output bitwise.

| Role | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 226.033 | 566.29 | 42.207 |
| candidate | 6 | 220.115 | 581.51 | 48.896 |

Both candidate processes did not improve the prior preferred result in both
phases. The preferred row stays unchanged. Clocks were adaptive; comparable
RKLLM quality and all-model performance remain unqualified.
[Reports, exact-output comparison and restoration](docs/benchmark-minicpm4-pooled-qkv-models-verification-v1.json).

### TinyLLAMA/1.1B: four-row prefill and fused SwiGLU decode v2

Four ABBA processes used 128 input tokens, 64 output tokens and context 512.
All 24 generated sequences and four full-vocabulary FP32 teacher arrays
match the pinned prior output bitwise.

| Role | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| control | 6 | 368.661 | 347.20 | 24.997 |
| candidate | 6 | 357.479 | 358.06 | 25.295 |

Both candidate processes improved the prior preferred TTFT and decode.
The primary row selects the candidate with the slower process median TTFT.
Clocks were adaptive; comparable RKLLM quality and all-model performance
remain unqualified.
[Reports, exact-output comparison and restoration](docs/benchmark-tinyllama-row-batch-models-verification-v2.json).

## Spark phase profiles v1 (diagnostic only)

These are diagnostic per-call native profiles captured at request index 4. CPU, ROCKCHIP, and COPY times are sums of labeled calls; decode values divide the 63 measured transitions. Instrumentation includes driver/cache and supporting work, so these figures are not throughput or end-to-end qualification. The erf/GELU subset includes only CPU labels whose every matched native source contains both approximation constants; mixed source labels are marked ambiguous.

| Model | Prefill CPU ms | erf subset ms (prefill CPU) | NPU calls ms (prefill) | Decode CPU ms/transition | NPU ms/transition |
|---|---:|---:|---:|---:|---:|
| Spark-X2.5/1.7B | 658.906 | 313.067 | 290.289 | 8.427 | 58.495 |
| Spark-X2.6/4B | 1392.051 | 667.206 | 481.026 | 13.424 | 144.730 |

Timing, quality, and goal qualification remain false. Full raw per-phase labels and provenance are in [the verification artifact](docs/benchmark-spark-phase-profiles-verification-v1.json).

### CPU erf-GELU prefill vector qualification

The opt-in `--cpu-erf-gelu` kernel preserves the tensor erf formula and is
limited to CPU prefill; one-row decode stays generic. In 40 paired blocks
per geometry, finite output bits match in every timed call.

| Width, 128 rows | Tensor baseline ms | Serial vector ms | Paired speedup | Faster blocks |
|---|---:|---:|---:|---:|
| 6,656 | 11.48 | 5.93 | 1.934× | 40/40 |
| 10,240 | 17.41 | 8.78 | 1.982× | 40/40 |

The operator study checked 960 timed outputs, 22 finite datasets, two
nonfinite-classification datasets and four dynamic-row datasets. The port
passed 80 repository tests with three hardware skips, mypy on 282 files and
Ruff. Full-model speed and comparable RKLLM quality remain pending.
[Operator evidence](docs/benchmark-cpu-erf-gelu-operator-verification-v1.json) ·
[Integration checks](docs/benchmark-cpu-erf-gelu-integration-cpu-verification-v1.json).

### Spark: vector erf-GELU model comparison v2

Candidate adds only `--cpu-erf-gelu` to the current pooled recipe. Four ABBA
processes per model use 128 inputs, 64 outputs and context 512. All 24
generated sequences and four full-vocabulary FP32 teacher arrays per model
match exactly. The helper executes only in prefill; decode stays generic.

| Model | Role | Measured requests | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|
| Spark-X2.5/1.7B | control | 6 | 958.569 | 133.53 | 14.358 |
| Spark-X2.5/1.7B | candidate | 6 | 735.179 | 174.11 | 14.250 |
| Spark-X2.6/4B | control | 6 | 2010.554 | 63.66 | 6.118 |
| Spark-X2.6/4B | candidate | 6 | 1699.716 | 75.31 | 6.056 |

Preferred rows select the slower-TTFT candidate only when both candidate
processes improve the previous preferred TTFT and decode. Improvements over
the older preferred recipe do not isolate the GELU option: current controls
also use the current pool placement. Clocks remain adaptive, and comparable
RKLLM quality and the all-model goal remain unqualified.
[Reports, comparison and serving restoration](docs/benchmark-spark-erf-gelu-models-verification-v2.json).

### Raw erf-GELU fusion operator qualification v3

This private CPU prototype fuses raw INT32 scaling, erf GELU and independent
per-K-group quantization. The baseline is the existing multi-stage path with
the vector erf helper. Both use the corrected storage-view memory planner.
All 40 payload cases, two dynamic-row cases and 960 timed checks match the
independent oracle exactly. Timings below are operator timings, not LLM rates.

| Rows | Hidden width | Fused mode | Baseline ms | Fused ms | Median paired speedup | Faster blocks |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 6,656 | serial | 16.518 | 9.453 | 1.745× | 40/40 |
| 128 | 6,656 | pooled | 16.514 | 8.966 | 2.073× | 40/40 |
| 1 | 6,656 | serial | 0.903 | 0.347 | 2.584× | 40/40 |
| 128 | 10,240 | serial | 26.004 | 14.212 | 1.840× | 40/40 |
| 128 | 10,240 | pooled | 26.039 | 13.488 | 1.971× | 40/40 |
| 1 | 10,240 | serial | 1.138 | 0.393 | 2.892× | 40/40 |

The prototype is not installed in inference. NPU/full-model qualification and
comparable RKLLM quality remain pending. Both serving models stayed resident
and answered fresh requests.
[Payloads, timings, sources and restoration](docs/benchmark-raw-erf-gelu-operator-verification-v3.json).

### Raw erf-GELU full-model comparison v1

The first case passed its opt-in hardware qualification. Twelve model reports then measured three Spark checkpoints in per-model ABBA order. Both arms use the same pooled CPU recipe and standalone CPU erf helper; candidates add fused raw erf-GELU prefill and decode. The official X2.6-1.7B checkpoint has no historical timing prior or RKLLM row.

| Model | Role | Measured requests | Median TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|
| Spark-X2.5/1.7B | control | 6 | 730.576 | 175.20 | 14.415 |
| Spark-X2.5/1.7B | candidate | 6 | 554.414 | 230.87 | 14.659 |
| Spark-X2.6/4B | control | 6 | 1723.674 | 74.26 | 6.054 |
| Spark-X2.6/4B | candidate | 6 | 1427.393 | 89.67 | 6.164 |
| Spark-X2.6/1.7B | control | 6 | 741.393 | 172.65 | 14.392 |
| Spark-X2.6/1.7B | candidate | 6 | 539.846 | 237.10 | 14.557 |

For the two previously timed checkpoints, promotion requires both candidate processes to improve that checkpoint’s indexed preferred TTFT and decode. The new X2.6-1.7B candidate is considered only if both candidate processes improve both matched controls in both phases. Where a gate passes, the selected candidate is the one with the slower TTFT. A missing promotion does not invalidate the timing comparison. These are local INT8 timings and fixed-prefix regressions; no HF quality or RKLLM speed claim is made.

[Twelve model reports, comparison, qualification, current-source proof and serving restoration](docs/benchmark-raw-erf-gelu-models-verification-v1.json).


### Current-source Qwen3 baseline, October 28

Qwen3-0.6B was remeasured after the storage-view lifetime fix using the same
matched checkpoint, 128 input tokens, 64 outputs, context 512, three warmups
and three measured requests. The main W8A8 row now uses this current baseline:
**292.91 ms TTFT**, **436.99 prefill-proxy tok/s**, and
**37.80 decode tok/s**. The historical 227.34 ms / 34.24 tok/s source
cohort remains in the run index; this replacement is a requalification, not an
optimization win. Comparable HF/RKLLM output quality remains unqualified.

The separately instrumented request accounts for CPU attention at 61.12 ms
and raw FFN packing at 40.74 ms across 28 layers. NPU projections dominate
decode. Its call timers are disjoint; residual request time is reported rather
than assigned to an unmeasured cause. This profile is excluded from throughput
selection. Both processes produced six identical local sequences, identical to
each other and the recorded historical IDs. Affinity was CPUs 4–7; clocks were
adaptive and are retained in each trial.

A synthetic dynamic FP16 QK/PV prototype passed twelve changed-input oracle
checks, including poisoned future cache. Its first NumPy pipeline was slower
than CPU attention and was not integrated. Operator experiments do not count as
full-model results. Both original serving models were restored and answered
fresh requests with the original options and selection.
[Raw timing, separate profile, probe and restoration evidence](docs/benchmark-qwen3-current-baseline-v1-verification.json).

## October 29: bounded Qwen3 prefill screens

The post-reboot CPU+NPU screen uses the same source cohort, checkpoint, input
IDs, context 512, three full warmups and three measured 128-input/64-output
requests per process. CPU affinity is 4–7 and clocks are adaptive. The API and
its model sessions were absent in both arms. These are two screening processes,
not a final alternating-process speed qualification.

| Configuration | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Control: existing FFN threads | **285.66** | **448.09** | 36.57 |
| Candidate: reuse parked workers only for FFN packing | 291.77 | 438.70 | **37.76** |

All twelve generated sequences match the prior IDs. Both runs use native replay
in both phases, and all 32 full-vocabulary teacher vectors match bit for bit.
The pooled worker path executes in prefill only; decode keeps its existing
serial FFN kernel. The candidate does not improve measured prefill, so it is
rejected as a standalone optimization and has not changed production defaults.
Bold marks better reported samples, not RKLLM-qualified wins.

A separate NPU prototype streams two 64-row INT8 tiles per core under one
submission. All 36 independent integer-output checks pass, including partial
rows and one-core execution. Its paired matmul timings are slower for all six
tested geometries, including the three main Qwen3 projections; the full-model
test is therefore skipped. Copying the reference's row count alone is not a
speed improvement for this command/layout configuration.

The experimental mixed INT8/FP16 attention path also remains off: after reboot,
standalone FP16 attention passes, while one INT8 vocabulary projection followed
by FP16 attention still fails score validation. None of these results qualifies
an all-model or comparable-quality RKLLM win. The preferred comparison rows
remain unchanged.
[Raw screen statistics, integer oracles and source-linked evidence](docs/benchmark-qwen3-bounded-prefill-screen-v1.json).

## October 29: attention pipeline qualification

The complete captured DPU producer/FP16 matrix pair passes its initial constant
oracle, then fails after a valid INT8 projection. The producer still writes
the exact FP16 weights, while 160 matrix output values are nonfinite. Adding
the producer alone does not resolve the mixed-operation failure.

An experimental entirely INT8 attention runtime passes 21 changed-input cases
after INT8 work, independent integer QK/PV oracles, native replay, causal future
invariance, cached-tail equality and invalid-input rejection. Frozen balancing
and per-token scales avoid future-dependent quantization. It remains off:
operator correctness does not establish model quality or request speed.

| Separate paired operator screen | CPU control (ms) | Candidate with frontend/transfers (ms) |
|---|---:|---:|
| INT8 NPU attention | **5.48** | 10.56 |
| Online OpenCL attention, subgroups off | **5.13** | 8.60 |
| Online OpenCL attention, subgroups on | **6.76** | 9.93 |
| Dense Tensor OpenCL attention | **5.95** | 7.69 |

Each row is its own comparison; adaptive clocks prevent treating controls from
different runs as interchangeable. GPU checks include numerical, causal and
cached-tail comparisons and preserve both API sessions. No row is a full-model
result or an RKLLM win. Separate direct-call phase profiles attribute roughly
2 ms to host packing/softmax/unpack and 1.1 ms to NPU execution; the IRQ helper
has little effect there. Those instrumented direct-call timings use a different
protocol and are not substituted for the paired operator timings above.
No production setting or preferred full-model number is promoted.
[Qualification, profiles and original artifact hashes](docs/benchmark-qwen3-attention-paths-v1.json).


### October 29: mixed-precision tile-width boundary

The existing INT8 decode recipe now passes 25 changed-input mixed head/QKV/
FP16-attention cycles at **24,384 total columns (8,128 per core)**, followed by
the native cached-attention and invalid-input test. The 8,192-column per-core
case fails even with one core, despite correct INT32 projection outputs.
`configure` now rejects that larger tile when INT8 projections and experimental
NPU prefill attention are selected together. Normal serving defaults stay unchanged.

A narrow GDB capture preserved the four missing command-tail words for the
first 12 prefill and 6 decode submissions; the normal SDK run finished.
A separate ten-shape, 30-oracle INT8 qualification passed. The five-setting
SDK-style decode recipe alone measured 292.70 ms TTFT / 37.65 decode tok/s
versus 284.49 ms / 37.96 for its old-recipe control, with bitwise-equal teacher
vectors. It was rejected as a standalone speed improvement and was not used
in the width-only model comparison below.

| Configuration | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) | Qualification |
|---|---:|---:|---:|---|
| Published RKLLM W8A8 | **199.09** | **642.93** | 32.91 | Published reference |
| oNPU CPU attention, 24,384-column tiles | 305.69 | 418.72 | 37.51 | Local screening control |
| oNPU FP16 NPU attention, same tiles/INT8 recipe | 283.74 | 451.11 | **37.87** | Rejected by local numerical gate |

Both arms complete three full warmups and three measured 128-input/64-output
requests at context 512, with native replay and adaptive clocks on CPUs 4–7.
The candidate executes 28 prepared NPU attention calls per prefill. Decode
keeps the existing CPU-attention/INT8-projection path; rates vary with adaptive
clocks. Each arm's six generated sequences are stable, but they first differ at output index 2
and differ at 19 of 64 positions. Independent recomputation of 32 fixed-prefix
full-vocabulary teacher vectors gives maximum logit error **2.91887**, relative
L2 error **12.303%**, and top-1 agreement **93.75%**, failing the predeclared
0.05 / 0.2% / 96.875% limits. The repeated-prefix NLL decreases by 0.01645;
that does not establish held-out quality.

The configuration remains experimental and the preferred W8A8 rows are
unchanged. The next gate is numerical diagnosis and comparison with the original
FP16 checkpoint on frozen held-out text. Both original serving sessions were
restored and answered fresh requests. CPU regressions pass (57 passed, one
hardware skip); the final guard-scope check passes 11 tests with one hardware
skip, plus mypy and Ruff. Bold marks reported numerical samples only.
[Raw reports, independent audits, failed gate and serving receipts](docs/benchmark-qwen3-mixed-width-v1.json).


### October 29: original-checkpoint quality and output borrowing

A four-prefix live diagnostic now checks all 28 NPU attention layers against
independent CPU emulation of the declared FP16 query/probability casts. Worst
relative output error is 0.00338%; exported output and raw PV output agree
exactly after head reordering. Ordinary and native-replay logits match bit for
bit and reproduce the prior candidate's first teacher vector. A further 120
mixed cycles verify exact full INT32 matrices after FP16 attention across all
six required projection shapes, including changed signed inputs.

The gate uncovered a static-call scheduling bug before NPU submission:
constant length/count tensors reached a pointer-input kernel as `CONST`.
The helper now creates CPU integer buffers for constants while retaining its
existing symbolic path. CPU checks pass: 12 tests and 3 subtests, one hardware
skip; replay checks pass 19 tests and 27 subtests, two hardware skips; mypy and
Ruff pass. The original failed attempt and all source records are retained.

The separate original-checkpoint quality protocol was frozen before evaluation:
16 disjoint WikiText-2 prefixes, 64 following targets each, 1,024 targets total,
context 512, identical token IDs and the existing matched FP16 checkpoint.
Every full-vocabulary logit vector is retained and the NLL was independently
recomputed. Repeating the first sequence reproduces all 64 vectors bitwise.

| Arithmetic/configuration | Sampled perplexity | Change from original FP16 | Predeclared 2% mean-ratio screen |
|---|---:|---:|---|
| Original FP16 checkpoint, OpenCL | **30.61051** | Reference | Reference |
| INT8 projections, FP32 CPU attention | 31.45853 | +2.770% | Fail |
| INT8 projections, FP16 NPU prefill attention | 30.94500 | +1.093% | Pass on this sample |

The NPU-attention candidate also passes the predeclared comparison with the
CPU-INT8 control (perplexity ratio 0.98368). Its sequence-bootstrap 95% interval
against original FP16 is 0.98705–1.03584; uncertainty remains beyond the mean
screen. The earlier fixed-prompt FP32 logit gate remains failed and unchanged.
These are separate arithmetic modes, and the new original-checkpoint protocol
was not chosen after observing its quality results. This is sampled text
quality evidence, not broad VLM or model-maximum-context qualification.

An independent matched timing pair exercises existing mapped-output borrowing
with **CPU attention in both arms**, the original INT8 recipe and 24,384-column
tiles. Both use three full warmups and three measured 128-input/64-output
requests at context 512, native replay on CPUs 4–7 and adaptive clocks.

| CPU-attention configuration | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Output borrowing off | 328.78 | 389.32 | 37.03 |
| Output borrowing on | **263.79** | **485.23** | **37.78** |

Borrowing removes 84 prefill copies and 28 decode copies at qualified sites.
All twelve generated sequences and all 32 full-vocabulary teacher vectors
match bit for bit. This is one process pair with clock drift, not a final ABBA
or RKLLM speed qualification. Borrowing remains opt-in; its 1,024-target quality
run was not repeated. The quality-passing NPU-attention run and this timing
screen use different attention paths. Their combination is the next measured
candidate. Preferred comparison rows remain unchanged, and the all-model goal
is still incomplete. Both original serving models were restored and answered
fresh requests.
[Full arithmetic, quality, borrowing, source and serving evidence](docs/benchmark-qwen3-original-quality-and-borrow-v1.json).

### October 29: RKLLM CPU ISA capture

The existing Qwen3-0.6B W8A8 SDK model was sampled on performance cores 4–7,
with separate prefill/decode windows and bounded GDB entry observations.
The capture completed with 8,650 user-CPU samples, zero reported lost samples,
16 hot-range disassemblies and 24 entry records. Debugger/profiler timings are
excluded from speed tables; preferred comparison rows remain unchanged.

Observed SDK CPU paths include eight-lane FP16 reductions, a five-output FP32
reduction reusing its input, min/max-based asymmetric scale calculation, and
INT32-to-float output conversion. These give specific comparison targets for
our generated CPU code. FP16 accumulation and quantization policy changes
require quality checks. The SDK's hottest OpenMP locations are polling loops;
its 58.94% decode libgomp CPU-sample share is not a request-latency saving.

Inspection also corrects the prior claim that packed INT8 FFN preparation is
serial: the 128×3072 kernel creates three pthreads and joins them on each call.
A CPU-only check confirms these threads inherit the caller's affinity. The
unexecuted combined-host stage's single-core prefill restriction was corrected
to CPUs 4–7; no speed result is claimed for that correction. Both original
serving models were restored and answered fresh requests.
[Findings and next actions](docs/rkllm-cpu-isa.md) ·
[Machine-readable capture and verification evidence](docs/benchmark-rkllm-cpu-isa-v1.json).

### October 29: combined host optimizations and preserved sampled quality

Four unprofiled Qwen3-0.6B processes compare NPU FP16 prefill attention with
existing mapped output borrowing, native cache invalidation and static FFN
layout off/on/on/off. Each uses three complete warmups and three measured
128-input/64-output requests, context 512, CPUs 4–7, adaptive clocks and the
original INT8 recipe. All 24 generated sequences match; all subsequent arms'
32 full-vocabulary teacher rows match the first control bitwise.

| Process / reference | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Published RKLLM W8A8 | **199.09** | **642.93** | 32.91 |
| Control 1 | 261.39 | 489.69 | 37.96 |
| Candidate 1 | 253.19 | 505.54 | **38.04** |
| Candidate 2 | 247.64 | 516.88 | 37.68 |
| Control 2 | 271.64 | 471.22 | 37.92 |

Bold marks the best reported number in this table, not a qualified matched
RKLLM win. The candidate remains roughly 50 ms short on TTFT. Its lower TTFT
repeats across both processes, while decode shows little consistent change.
It borrows 84 prefill and 28 decode outputs; the earlier CPU-attention borrowing
pair's larger gain does not transfer unchanged to this attention configuration.

The first held-out quality worker stopped before model loading because of an
unescaped quote in its harness. That failure is retained. A corrected retry
preserves **all 1,024 full-vocabulary vectors bitwise** against the prior
NPU-attention candidate, including the repeated first sequence. Sampled
perplexity remains 30.94500, versus 30.61051 original FP16 (+1.093%). The prior
bootstrap uncertainty remains unchanged; this does not qualify all models,
VLM images, the API or maximum context. Production defaults remain unchanged.

Diagnostic clocks inside the quality retry split the prepared attention call
across all 28 layers. Median CPU packing/softmax/unpacking is 30.90 ms; NPU
QK/PV submit and wait is 15.85 ms; synchronization is 3.65 ms. Total median is
50.27 ms across 16 prefixes. These clocks are excluded from the unprofiled
speed measurements above, and category medians need not sum exactly. CPU
softmax alone has a median of 18.56 ms, identifying a concrete host-side cost.
Both benchmark owners have exited; original options, settings and selection
were restored, and both original serving models answered fresh requests.
[Raw timing arms, quality preservation, phase clocks and failed attempt](docs/benchmark-qwen3-combined-host-v1.json).

### October 29: shared CPU workers during prefill

A staged prototype replaces per-call thread creation across large RMSnorm,
quantization, dequantization, QKV preparation and FFN packing. It pins the main
thread once during prefill, uses the existing workers on CPUs 5–7, and parks
them/restores affinity afterward. Arithmetic, four work partitions, NPU recipe,
FP16 attention and the three previously measured host flags remain unchanged.
This is experimental code; production defaults and the historical preferred
cohort are unchanged.

Each process uses three full warmups and three measured 128-input/64-output
requests at context 512 with adaptive clocks. All 24 generated sequences
match, and the later processes' 32 teacher vectors match the first control
bitwise. Both candidate medians improve on both control medians.

| Process / reference | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Published RKLLM W8A8 | **199.09** | **642.93** | 32.91 |
| Control 1 | 253.47 | 504.98 | 37.93 |
| Shared workers 1 | 230.00 | 556.52 | **38.06** |
| Shared workers 2 | 221.84 | 576.99 | 37.98 |
| Control 2 | 267.57 | 478.38 | 38.05 |

Bold denotes the best reported number, not a qualified matched RKLLM win.
Prefill still needs another 22.75–30.91 ms reduction to meet the published
target. The quality process preserved all **1,024 full-vocabulary vectors
bitwise**, including the repeated first sequence: sampled perplexity remains
30.94500 (+1.093% versus original FP16). The same sampling/uncertainty and
API/VLM/model-maximum-context limits still apply.

All five workers exited successfully. The final quality report validator
raised a missing-helper error after inference; a corrected copy revalidated
all saved arrays and 28 checks without rerunning the model or altering any
numerical gate. The original failed validator/job remain recorded. Serving
options, settings and selection were restored, and both original models
answered fresh requests.

Six standalone CPU operators also have native tinygrad VIZ CLI JSON traces.
The reader runs with tracing disabled, and raw profile/rewrite hashes are
verified unchanged. Those graph traces are separate from model timing because
whole-model PROFILE currently bypasses optimized Rockchip native replay.
[Model timings and quality](docs/benchmark-qwen3-shared-host-pool-v1.json) ·
[Operator comparisons and agent-readable VIZ traces](docs/benchmark-shared-host-pool-operators-v1.json).

### October 29: worker reuse integrated into tinygrad

The repository now contains the opt-in shared dispatcher and guarded prefill
scope. The default remains off. Current-source inference was measured with
all five worker processes succeeding, without prototype monkeypatches. Each
process uses three warmups and three measured 128-input/64-output requests at
context 512, CPUs 4–7 and adaptive clocks, with the existing host flags enabled
in both arms.

| Process / reference | TTFT (ms) | Prefill proxy (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Published RKLLM W8A8 | **199.09** | **642.93** | 32.91 |
| Integrated control 1 | 252.09 | 507.76 | 37.94 |
| Integrated shared workers 1 | 211.12 | 606.29 | 38.08 |
| Integrated shared workers 2 | 207.47 | 616.96 | 38.03 |
| Integrated control 2 | 265.69 | 481.77 | **38.11** |

Bold marks the best reported number, not a matched RKLLM qualification.
Prefill remains 8.38–12.03 ms short of the published target. All 24 generated
sequences, all 32 teacher rows and all 1,024 held-out full-vocabulary vectors
match bitwise. Sampled perplexity remains 30.94500; prior quality uncertainty,
other-model and API/VLM/maximum-context limits remain explicit.

The subsequent serving reload exposed parameter discovery recursively entering
new JIT/runtime metadata. Both metadata attributes now hold weak references;
the model-state getter dereferences explicitly. Two new regressions cover a
warmed JIT and cyclic runtime state. The final CPU checks include 108 tests
and 8 subtests, 6 explicit hardware skips, mypy on 285 files and repository
Ruff. Generated C for all six operator fixtures and the native dispatcher
expression remain identical after this metadata-only fix. Exact timing-source
files are preserved with hashes matching the frozen measurement manifest;
post-fix throughput has not been rerun.

The original GPU worker stayed loaded. Only the failed NPU session was reloaded
after the fix, and both original models are ready with their original options,
settings and selection. Both answered fresh requests. The all-model speed goal
remains incomplete.
[Integrated timings, quality, source provenance and serving recovery](docs/benchmark-qwen3-shared-host-pool-integrated-v1.json) ·
[Shared-worker implementation](/home/orangepi/tinygrad/tinygrad/llm/kernels/rockchip_shared_pool.py).

### Qwen2.5 1.5B adaptive software screen — October 29

| Model / candidate | Prefill proxy, RKLLM / oNPU (tok/s) | Decode, RKLLM / oNPU (tok/s) | Decision |
|---|---:|---:|---|
| Qwen2.5 1.5B, prepared QKV + shared CPU + fused SwiGLU | **338.35** / 284.96 | **16.69** / 12.24 | Rejected; decode regressed. |

Three measured 128-input/64-output requests at context 512, adaptive governors and no sudo. The worker failed during report finalization after writing timing and 32-target teacher arrays. Independent comparison of those saved arrays confirms bitwise equality to the prior preferred recipe; the failed worker status is preserved. This does not qualify independent RKLLM quality equivalence. A separate opt-in decode pool for two KV heads is being tested next. [Measured trials and independent numerical verification](docs/benchmark-qwen25-adaptive-screen-v1.json).

### Qwen2.5 1.5B: opt-in two-KV-head decode pool — October 29

| Model / candidate | Prefill proxy, RKLLM / oNPU (tok/s) | Decode, RKLLM / oNPU (tok/s) |
|---|---:|---:|
| Qwen2.5 1.5B, CPU decode pool | **338.35** / 303.94 | 16.69 / **16.73** |

Three unprofiled measured requests after three warmups, 128 input/64 actual output, context 512. All 32 checked full-vocabulary teacher vectors remain bitwise equal to the prior local recipe. No sudo or governor/IRQ changes. Prefill is still below the reference; decode's small lead needs process-level repetition. The worker completed successfully, but the original validator incorrectly required 28 distinct source hashes for 28 same-shaped CPU kernel calls; independent revalidation preserved that ancillary failure and verified primary timing/numerical gates and actual shared source hashes. [Trials, scope and verification](docs/benchmark-qwen25-decode-pool-v1.json).
