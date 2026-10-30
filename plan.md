# oNPU performance plan

Updated October 30 (Asia/Shanghai). The all-model parity goal remains unfinished.

## Qwen2 0.5B — current measured result

API medians: **901.11 prefill-proxy / 43.98 decode tok/s**, TTFT 142.06ms, versus published RKLLM877.31/41.58 and145.90ms. Ten requests have128 uncached input and64 actual output tokens at context512;7/10 prefill and10/10 decode requests beat the reference. All18 API functional checks pass and both original sessions are restored. No sudo or board-setting changes. [Current receipts and source scope](docs/benchmark-qwen2-sdk-gqa-api-v1.json).

Retained: captured SDK A16/C4 gate/up layout, direct raw SwiGLU consumption, a single packed K-group upload with bounded header offsets, cache-line-safe simultaneous borrowing, grouped-query CPU prefill, and token-major QKV queries. All backend sequences/32-target full-vocabulary arrays remain bitwise equal to the prior local recipe. Decode keeps the original graph. A stricter cache-line guard was added after API timing; the receipt retains both hashes and its qualified-shape equivalence proof.

The published-reference median goal is reached for this workload, but the modest median margin does not establish a consistent every-request win. Independent RKLLM quality and model-max context remain unqualified. Next: reduce frequency/wakeup variance under ordinary adaptive serving, then broaden quality/context coverage. Do not repeat the rejected NPU-attention, fetch-grain-only, SDK down, temporary-padding, approximate-exp or O3 experiments without new evidence. No all-model or speculative-decoding goal is complete.

## Current Qwen2.5 status

The completed exact-scale API cohort used 128 prompt / 64 output at context 512 and ordinary adaptive settings. Five-request medians were 345.25 prefill-proxy / 16.76 decode tok/s and 370.74 ms TTFT against published RKLLM 338.35 / 16.69 and 378.31 ms. Only two of five measured requests exceeded both published rates; one missed prefill and two missed decode. The decode median lead is about 0.4%, narrow and not consistent enough to claim an independent quality-qualified serving win. The output-quality and model-maximum-context checks remain unqualified. [Current API receipt](docs/benchmark-qwen25-api-exact-scale-v1.json) · [Benchmark summary](benchmark.md).

The API worker used CPU4–7, had no replay/scale-bit environment overrides, and restored the original GPU/NPU sessions with two fresh replies. It ran scalar-copy, deduplicated guards and exact scale bits together, so the API result does not isolate the effect of any one change. In a separate paired worker screen, exact scale bits increased median prefill proxy by 5.0% and decode by 0.6%; this was worker IPC evidence, not an API treatment. The rejected split-KV head comparison changed decode by only about 0.2% and had unequal reported NPU frequencies across control/candidate blocks. Wide prefill and head-count prototypes remain removed. [Scale pair](docs/benchmark-qwen25-exact-scale-paired-v1.json) · [Host-overhead evidence](docs/benchmark-qwen25-host-overhead-v1.json) · [Split-KV rejection](docs/benchmark-qwen25-split-kv-v1.json).

The current defaults are `ONPU_REPLAY_SCALAR_COPY=1`, `ONPU_REPLAY_DEDUP_GUARDS=1`, and `ONPU_EXP2_SCALE_BITS=1`. Scale bits only affect paths where the existing `ONPU_EXP2_FAST_PATH` context is enabled; the Qwen profile enabled it. Other profiles with that path disabled are unchanged. Avoid attributing the unpaired API difference to these defaults; focus the next Qwen work on repeatability and the narrow decode margin.

The retained runtime passed 34 CPU tests and 27 subtests, with two hardware cases skipped. Mypy passed 286 source files and Ruff passed. After reverting the rejected experiments, all 297 runtime source hashes match the API-qualified snapshot.

## Next prefill candidate: TinyLLAMA

TinyLLAMA remains at 496.20 prefill-proxy / 25.33 decode tok/s versus published 524.74 / 24.43. The existing exact exp2 fast-path context is off in the Tiny profile; evaluate that prefill candidate before implementing new changes. No TinyLLAMA run of this candidate is claimed. The broader all-model goal remains open.

## Outcome and scope

Beat the published RKLLM W8A8 figures in both prefill and decode at comparable
output quality for all 21 reference models in [benchmark.md](benchmark.md).
Retain Spark-X2.6 1.7B/4B and LFM3 as requested additions; missing reference or
checkpoint details remain explicit, never guessed.

Keep OpenCL GPU-first serving, optional NPU acceleration where it improves
complete requests, and simultaneous serving of a GPU model and a separate NPU
model. Qualify API stability and model-maximum context. Resume speculative
decoding after the base inference milestone. Select the best usable placement
per model; every backend does not independently need to win.

The official RKLLM table is the speed target. Exporting new RKLLM models is not
a prerequisite for every comparison. Match checkpoint identity, precision,
request lengths and quality as far as evidence permits; disclose differences.

## Immediate follow-up: Qwen3, then TinyLLAMA

Qwen3: completed the SIMD score-preparation screen. Existing recipe: 667.79 tok/s median; candidate: 655.35 tok/s. Candidate rejected and removed; the CPU microbenchmark improvement did not translate into complete requests. All four timing workers passed their short checks; the 1,024-target worker failed at import, so no full-quality completion is claimed. Both original serving models were restored and answered fresh requests. Variation still crosses the published reference. See [result](docs/benchmark-qwen3-score-vector-v1.json).

The user explicitly authorized temporary stock CPU/NPU/DDR governors, CPU6 NPU IRQ placement and CPU6 idle adjustment, with restoration. Authorization is no longer pending. The user launched the privileged helper and the fixed-stock cohort completed. Original serving sessions and fresh replies are verified; board-setting restoration is being checked. Prepared helper: `scripts/temporary_stock_tuning.py` (read-only preview by default; `--apply --seconds 900` applies then restores on timeout or SIGINT/SIGTERM/SIGHUP). The completed fixed-stock cohort uses the existing recipe; comparison with the earlier adaptive cohort is descriptive, not paired ABBA.

TinyLLAMA: follow [the evidence-based prefill plan](docs/tinyllama-prefill-plan.md). The older 235.05 tok/s sweep is superseded for optimization planning by a focused 358.06 tok/s candidate. Target 524.74 tok/s requires reducing roughly 357.48 ms TTFT below 243.93 ms. Profile the current recipe, then choose placement or the largest demonstrated CPU/NPU cost; do not simply copy Qwen3's shape-specific attention path.

## Fixed-stock Qwen3 outcome

Four same-recipe processes completed twelve measured 128-input/64-output requests: 149.50 ms median TTFT, 856.18 prefill-proxy tok/s and 38.58 decode tok/s. Every request exceeds the published RKLLM 642.93/32.91 targets; prefill spans 825.32–862.76. This is a repeatable published-reference speed win for context 512 under the temporary authorized stock-clock/IRQ/idle profile. It does not establish independent RKLLM quality equivalence or ordinary adaptive-governor performance. Original numerical code remains selected; no losing micro-optimization was retained. [Results and restoration](docs/benchmark-qwen3-stock-tuning-v1.json).

Next: TinyLLAMA current four-row recipe under the same recorded operating conditions, followed by a phase profile if still below 524.74 tok/s. Measure its own gap before choosing GPU placement, generalized NPU attention, or CPU helper work. [Detailed plan](docs/tinyllama-prefill-plan.md).

## Earlier adaptive-host results (historical)

The earlier adaptive-host Qwen3 measurements below remain separate cohorts. The all-model goal remains unachieved.

| Qwen3-0.6B at context 512 | Published RKLLM | Backend token-input candidate | oNPU API fast profile |
|---|---:|---:|---:|
| Prompt tokens | 128 | 128 | 128 |
| Generated tokens | 64 | 64 | 59 (64-token budget) |
| Median TTFT, ms | 199.09 | 199.31 | 196.77 |
| Prefill proxy, tok/s | 642.93 | 642.22 | 650.49 |
| Decode, tok/s | 32.91 | 38.39 | 37.36 |

The backend input change reduces paired median TTFT from 204.24 to 199.31 ms and preserves all 1,024 held-out logit vectors bitwise. The worker now selects the measured recipe for the exact F16 checkpoint at context 512 in the Rockchip NPU+CPU profile. A cached-one-token input-contract regression was caught and fixed before API testing.

The API passes uncached, cached, streaming, 448-prompt+64-reply-budget, cancellation and recovery checks. Both original serving sessions were restored and answered fresh requests. The API's five measured prefill rates span 627.42–711.39 tok/s. Its median exceeds the published target by 1.18%, but this small, variable lead does not establish a consistent matched win. API TTFT is first emitted text; backend TTFT is first generated token. Its natural 59-token stopping also makes API decode a different workload from the reference.

These historical cohorts used adaptive governors. The authorized fixed-stock follow-up above temporarily changed governor, IRQ and CPU6 idle settings; it did not change voltage, drivers or boot settings. Larger-context acceleration and independent RKLLM quality equivalence remain unqualified.

[Backend cohort](docs/benchmark-qwen3-cpu-token-inputs-v1.json) · [API qualification](docs/benchmark-qwen3-api-fast-prefill-v1.json) · [Host tuning audit](docs/benchmark-rk3588-host-tuning-audit-v1.json).

### The register dump has now identified a specific command difference

- [x] Capture real RKLLM execution: 2,657 submissions, 3,751 tasks and 367,308
  register words. Another broad dump is unnecessary.
- [x] Reproduce the mixed failure: our current INT8 projection can make the
  following FP16 matrix operation return invalid values. Adding the captured
  DPU producer alone did not fix it.
- [x] Replay the captured SDK INT8 decode command followed by the captured
  DPU/FP16 pair: one baseline plus 20 changed-input cycles pass. A separate
  control without the DPU producer also passes one baseline plus 20 cycles.
  These are restricted constant-input oracles, not full attention/model tests.
- [x] Derive an M=1 INT8 prototype changing five register settings, with seven
  occurrences. At captured M=1, N=672, K=1024, all 108 ordered command words
  match the SDK body after address relocation. M>1 bodies remain unchanged.
- [x] Qualify ten INT8 shapes with 30 exact independent integer checks,
  including zero, random and extreme signed inputs with real buffer rebinding.
- [x] Isolate the mixed failure by width: 8,128 columns per core passes 20
  changed-input cycles; 8,192 fails even with one core. Three cores with 8,128
  columns each also pass. This is bounded evidence at K=1024, not a general
  hardware-limit claim.
- [x] Capture the omitted four-word SDK command tails for 18 selected submits.
  The normal SDK run completes; no broad repeat dump or model export is needed.
- [x] Measure the five-setting decode prototype alone: 292.70 ms TTFT and
  37.65 decode tok/s versus 284.49 ms and 37.96 for the existing recipe. All
  twelve sequences and 32 full-vocabulary teacher rows match exactly. Reject
  it as a standalone speed improvement; keep the original recipe selected.
- [x] Add a configuration guard against projection tiles above 24,384 when
  experimental NPU prefill attention is enabled. It prevents the known failing
  per-core width; it does not enable the experimental path by default.
- [x] Qualify the original recipe at 24,384 total columns: all 25 mixed
  head/QKV/FP16 cycles and the exact native cached/invalid-input test pass.
- [x] Complete six full requests per arm with original decode in both arms.
  The NPU-attention screen records 283.74 ms / 451.11 prefill-proxy tok/s /
  37.87 decode tok/s, versus 305.69 ms / 418.72 / 37.51 control. It fails the
  declared local logit gate; this is not a promoted or qualified RKLLM win.

The width comparison supplies a concrete next configuration: 24,384 total
projection columns, or 8,128 per core. It does not establish that NPU attention
improves whole-model latency. Earlier literal replay used captured
task bodies with relocated addresses and reconstructed link/tail words; it is
not a complete byte-for-byte reconstruction of SDK submissions.

Primary evidence: [SDK pair replay](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-sdk-int8-fp16-chain-v1/results/sdk-chain.json),
[bare FP16 control](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-sdk-int8-fp16-chain-v1/results/sdk-chain.bare.json),
[prototype CPU checks](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/int8-native-decode-cpu-v1.json).
At the literal replay, all 295 numerical sources matched its manifest. Later
changes add configuration validation and static metadata buffers; timed source
cohorts remain explicitly recorded.
The hardware owner exited successfully, restored serving, and both original
models answered fresh requests with their original settings and selection.
[Restoration and replies](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-sdk-int8-fp16-chain-v1-fresh.json).

## Why this took too long, and what changes now

Too much work went into small kernel experiments, repeated harness revisions
and short hardware batches before isolating the command transition. Serving
restoration cost roughly five minutes per batch. Some test/report mistakes
caused avoidable reruns. Those are execution problems, not evidence that parity
is impossible; more agents cannot run independent benchmarks on the same NPU.

The dump supplies the device commands. Matching one command does not match
packing, memory layout, preceding device state, synchronization, CPU attention,
FFN preparation or complete request latency. A separate Qwen3 request profile
measured 61.12 ms in CPU attention and 40.74 ms in FFN packing. Those figures
identify targets; they are not guaranteed additive savings against the control
above. The safe-width screen now completes, and the arithmetic and sampled
original-checkpoint quality checks below resolve its next decision.

## Current decision after arithmetic and original-checkpoint checks

All 28 live attention layers match independent emulation of the intended FP16
arithmetic (worst relative output error 0.00338%). Ordinary/native logits match
bitwise. All 120 subsequent FP16/INT8 mixed cycles produce exact integer
matrices across six shapes. Static metadata now has real CPU buffers; the
failed pre-fix run is preserved. Those dependencies no longer justify more
reset/register guesses.

The predeclared 1,024-target original-checkpoint quality screen passes for NPU
attention: perplexity 30.94500 versus 30.61051 original FP16 (+1.093%). The
CPU-INT8 control is 31.45853 (+2.770%) and misses the same 2% criterion.
Bootstrap uncertainty and the older failed FP32-logit gate remain recorded;
this is sampled text evidence, not universal quality certification.
[Raw arithmetic, quality and timing evidence](docs/benchmark-qwen3-original-quality-and-borrow-v1.json).

Existing output borrowing now has a matched CPU-attention timing screen:
328.78→263.79 ms TTFT, 389.32→485.23 prefill-proxy tok/s and
37.03→37.78 decode tok/s, with exact 32-row teacher vectors and all 12 sequences.
It removes 84/28 qualified prefill/decode copies. This one pair is not final
speed qualification, and its CPU attention differs from the quality-passing
NPU-attention configuration. Keep default promotion pending their combination.

- [x] Measure NPU FP16 attention with output borrowing, preserving the matched
  precision and checkpoint. Compare existing static FFN layout/native FROM
  options as one bounded configuration. Allow prefill CPUs 4–7 so newly created
  packing threads inherit all four performance cores; retain scoped decode
  pinning in the attention pool. Validate actual affinity before launch.
  Require exact same-arithmetic teacher vectors and repeatable whole requests.
- [x] Preserve all 1,024 frozen natural-target vectors bitwise with the selected
  combination. A quoting error prevented the first quality worker from loading;
  the corrected separate retry passed all 16 sequences and the repeated first
  sequence. The four completed speed arms were retained without rerunning them.
- [ ] Qualify real API/cached/streamed use before defaults. Do not combine
  throughput and quality numbers from different configurations into a claimed win.
- [x] Split the prepared attention call into CPU helpers, NPU submit/wait and
  synchronization during the quality retry. Across 16 prefixes and 28 layers,
  category medians are 30.90 ms, 15.85 ms and 3.65 ms respectively; total median
  is 50.27 ms. These instrumented diagnostic clocks are excluded from speed
  comparisons; independent category medians need not add exactly.
- [ ] Target the remaining host preparation cost. Large
  packed INT8 FFN kernels already create/join three pthreads per call; the
  global CPU prefill-pool flag does not replace that lifecycle. The earlier
  description of this path as serial was inaccurate. Its older global-pool
  ABBA run was drifted, not a win.
  Avoid repeating the rejected parked-FFN-only experiment without a new premise.

Prioritize prefill: current pilot decode already exceeds the published target.
Verify repeated thread creation and memory passes across quantization,
normalization, QKV preparation and FFN packing before selecting one shared
improvement. Reusing workers across these paths is broader than the earlier
FFN-only attempt and requires its own measured comparison. Synchronization
is only a few milliseconds; layout/conversion work remains a separate target.

The current source/count audit predicts 756 pthread-create attempts per
128-token prefill, including output dequantization; this is not a measured
syscall count. A staged dispatch-only prototype covers those five families.
Fifteen numerical cases, changed/nonfinite inputs, the unchanged small serial
path, stopped-pool fallback and scoped cleanup pass. Parked-worker variants
give mixed timings; a variant keeping workers active only during prefill wins
all six operator comparisons in both candidate blocks. The full-model ABBA
now measures 221.84–230.00 ms TTFT against 253.47–267.57 ms controls. All
32-row teacher vectors and all 1,024 held-out full-vocabulary vectors match
bitwise. The final validator lacked an integer helper; a corrected copy
revalidated the saved output without a model rerun. The original failed job
and validator are preserved. Serving was restored and both models answered
fresh requests. This prototype result is retained as its own historical cohort.
[Operator and trace evidence](docs/benchmark-shared-host-pool-operators-v1.json).
[Full-model evidence](docs/benchmark-qwen3-shared-host-pool-v1.json).

The path is now integrated behind `ONPU_SHARED_HOST_POOL=1`, with admission
guards and an unchanged default. The 296-source model run records
207.47–211.12 ms candidate TTFT versus 252.09–265.69 ms controls, with all
32 teacher rows and 1,024 held-out vectors bitwise unchanged. CPU checks pass
108 tests in total, plus 8 subtests, mypy on 285 files and repository Ruff;
6 hardware tests are explicitly skipped. New parameter-discovery regressions
cover warmed JITs and cyclic runtime metadata.

Serving reload exposed a recursion bug in strong runtime metadata handles.
Both the JIT marker and model-state handle now use weak references, and the
explicit getter dereferences the latter. The measured source files are saved
with exact frozen hashes; generated C and dispatcher expressions remain
identical after the metadata fix. The original GPU stayed loaded, and both
models were restored and answered fresh requests. Exact post-fix throughput
has not been rerun; retain this provenance distinction.
[Integrated inference and serving evidence](docs/benchmark-qwen3-shared-host-pool-integrated-v1.json).

Six CPU-kernel traces now use tinygrad's native VIZ CLI JSON export. The reader
must run with PROFILE=0, VIZ=0 and TRACK_MATCH_STATS=0: inheriting writer flags
can overwrite the input PKLs and append non-JSON exit logs. The corrected
exports verify raw profile/rewrite hashes unchanged before/after reading.
Whole-model PROFILE currently bypasses optimized Rockchip native replay, so
these graph diagnostics remain separate from unprofiled model timing.

The same instrumented quality prefixes now measure 19.53 ms in attention CPU
helpers, 17.12 ms in NPU submit/wait and 3.60 ms in synchronization. Treat those
as diagnostic categories, not additive speed promises. The separate model
profile leaves the wide gate/up NPU projection and FFN preparation as major
costs. Inspect existing SDK prefill templates before selecting one next
implementation. The register inventory contains M64/M1 INT8 tasks and separate
M128 SDK templates; it does not provide an exact captured M128 command for
every full Qwen3 projection. Avoid claiming otherwise or repeating a broad dump.
Before any next hardware launch, compile and run undefined-name lint on all
actual drivers and validators, then perform CPU-only path/argument smoke checks.

The SDK-prefill eligibility audit found that none of the pilot projection
shapes selects the current SDK whitelist. Historical alternatives at M128,
K1024 and N2048/4096/6144 were correct but slower. Do not reopen them by simply
flipping the flag. The generic CPU cast/residual labels remain a possible
fusion target, but saved same-name C variants came from a different model;
their association to this pilot is unverified. Capture the pilot's actual
CPU program sources before choosing a rewrite, or use a narrowly matched
frontend pattern with exact independent numerical checks.

### CPU ISA evidence now complements the NPU register capture

- [x] Sample the existing SDK on CPUs 4–7 and dump the hot CPU ranges, with
  separate prefill/decode windows and bounded GDB entry observations. Capture
  completed with zero reported lost samples; serving was restored and checked.
- [x] Compare generated oNPU CPU C/ELF with observed SDK FP16 reductions,
  multi-output reuse, range/scale calculation and INT32 output conversion.
  Correct the unexecuted single-core prefill test setup before measuring it.
- [ ] Use the selected combined-host profile to choose between thread-lifecycle,
  layout/copy and reduction work. FP16 accumulation and asymmetric quantization
  change arithmetic; require separate quality gates before adopting either.

The SDK's hottest OpenMP locations are polling loops. Its 58.94% decode
libgomp CPU-sample share is not 58.94% removable request latency. No throughput
number comes from the debugger capture. [CPU ISA findings and raw evidence](docs/rkllm-cpu-isa.md).

## Next execution: one candidate, explicit gates

| Order | Work | Pass condition | Failure disposition |
|---|---|---|---|
| 1 (done) | Qualify the SDK-style M=1 recipe on physical tile shapes emitted by Qwen3's actual decode plan, including vocabulary and wide-K projections. | Exact independent INT32 oracles for zero, random signed and extreme inputs; changed calls; correct multi-core assembly and submission counts. M>1 unchanged. | Preserve the first failing input/commands; restrict the recipe to passing shapes or reject it. Never apply the five constants globally on faith. |
| 2 (passed) | Alternate qualified decode, existing INT8 prefill/QKV and FP16 attention under one device lifetime, without resets. | At least 20 changed-input mixed cycles; causal future invariance, cached-prefix/tail checks and native replay pass. Include the full model's actual command order. | One follow-up only if the failure supplies a new, testable fact. Otherwise leave FP16 attention off and record the unresolved dependency. |
| 3 (guard added) | Integrate only the qualified recipe and attention path behind explicit shape/option guards. | Focused regression tests with -n12, mypy and Ruff pass; commands, packing, invalid-input handling and existing paths remain correct. | Fix the specific regression before timing; keep original defaults until qualification. |
| 4 (ran; precision checked separately) | Run matched full Qwen3 control/candidate processes, CPU attention versus candidate attention, with the same decode recipe where supported. | Three full warmups and at least three measured 128/64/512 requests per process; source/checkpoint/ID pins; native replay; stable generated outputs and predeclared fixed-prefix numerical checks. Include packing, copies, sync and sampling. | A clear request-level loss ends the candidate. One additional paired batch is allowed only if spread prevents a decision. |
| 5 | Verify the selected pilot under alternating control/candidate/candidate/control processes and frozen held-out quality inputs. | TTFT below 199.09 ms, decode above 32.91 tok/s, repeatable gains and comparable quality against the original checkpoint. | Report the measured remaining gap; do not claim success or start an all-model sweep. |

Batch related diagnostic cases under one hardware lease and restore once.
Order 4 is conditional on orders 1–3; do not spend a full-model run on a failed
primitive. Freeze quality inputs and tolerances before candidate evaluation.
Repeated greedy IDs and 32 teacher vectors are local regression checks, not
comparable-quality qualification.

If the candidate is correct but still slow, use its measured component/subphase
profile to select the largest remaining cost, then test one implementation
against unprofiled complete requests. With roughly 8–12 ms left, select a
repeatable optimization covering a material part of that gap. Do not derive a promised model
speedup by subtracting unrelated operator medians or profile totals.

## Closed experiments and reopening rules

Reset placements, submission ping-pong changes and pointer clearing did not
qualify. Reused FFN workers measured 291.77 ms versus 285.66 ms control and were
rejected for prefill. Streaming M64 INT8 tiles passed integer checks but lost
all six operator comparisons. Reopen only with a new fact that changes the
failed premise.

All-INT8 attention passed its bounded correctness checks but measured 10.56 ms
versus 5.48 ms CPU in its paired screen. Three OpenCL bridges also lost their
paired screens after frontend/transfers. Keep them out of the immediate path;
these losses do not rule out a different, demonstrated placement.
[Attention evidence](docs/benchmark-qwen3-attention-paths-v1.json).

## After the pilot: shared improvements, then complete coverage

- [ ] Qualify fresh and cached API requests, streaming, simultaneous GPU/NPU
  models and increasing context to model maximum with output reserve and no
  silent truncation. Restored short replies are not a maximum-context test.
- [ ] Extend proven shared changes by architecture. Profile exceptions before
  adding specialized kernels; a Qwen3 win does not establish other-model wins.
- [ ] Keep all requested families: dense Qwen/MiniCPM4/TinyLLAMA/InternLM2/
  TeleChat2/Phi3/ChatGLM3, Gemma2/3n/4, RWKV7, Qwen3.5, Spark-X2.5/2.6,
  MiniCPM3 MLA, DeepSeekOCR experts, Qwen3-VL, LFM2.5-VL and LFM3.
- [ ] Measure VLM image encoder and connector latency explicitly. Text-only
  decoder measurements cannot establish complete VLM superiority.
- [ ] Requalify current sources and update the [model ledger](docs/benchmark-model-status.json)
  after each accepted model result. Refresh its stale active-study status;
  retain historical source cohorts rather than implying they all ran today.
- [ ] Run the full sweep after shared code stabilizes. Update benchmark.md in
  the original reference format; report prefill tok/s, TTFT and decode tok/s.
  Keep true prefill throughput distinct from input_tokens/TTFT. Bold reported
  numerical winners and label quality/clock/context limitations explicitly.
- [ ] Resume speculative decoding after base speed and quality qualification;
  count draft, verification and rejection costs in complete request latency.

## Ownership and completion

The primary owns architecture, register interpretation, integration and
performance decisions. Luna agents handle bounded extraction, mechanical
staging and review. One hardware owner; inference on CPUs 4–7, preparation on
CPUs 1–3, with adequate free memory and the GPU session preserved. Keep the
working API available between batches. No commits or pushes during this plan
review; no ANE work is part of this goal.

Completion requires every requested model to satisfy speed, comparable quality
and usability requirements. No defensible all-model finish date follows from
the register dump. The next milestone is one combined configuration with repeatable RKLLM-beating
request timings and the measured original-checkpoint quality requirement.
