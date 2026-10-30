# TinyLLAMA 1.1B prefill plan

Current execution status: the `tinyllama-stock-host-v1` comparison is prepared and source-pinned. Both arms use the preferred four-row SwiGLU prefill plus fused decode, with deferred/file-backed CPU weights. The candidate enables owned CPU token inputs, output borrowing, batched borrowing, native invalidation and zero-tail views. Both arms preserve the resident GPU model, use a 6 GiB process cap and enforce a 2 GiB free-memory reserve. The runner requires an active root-owned stock-tuning receipt; the earlier Qwen3 receipt is restored and inactive. Sudo requires local authentication, so no new hardware timing has started.

The app now selects the existing TinyLLAMA SwiGLU recipe for its exact checkpoint at context 512. Thirty targeted policy/worker tests and changed-file lint pass. API qualification and candidate promotion remain pending the actual measurements.


## Goal and comparison

Beat RKLLM's published TinyLLAMA 1.1B prefill reference of **524.74 input
tokens/s** on the 128 input / 64 output request. This is the 128,000 / TTFT
proxy, so the reference corresponds to about **243.93 ms TTFT**. The requested
current result, 235.05 tok/s, corresponds to about **544.56 ms**. Its
machine-readable provenance identifies the older `w8a8-sweep-v1` CPU + NPU
candidate with INT8 projections, prefill/decode on, CPU attention and fused
FFN. Reaching the reference from that cohort needs 2.23x throughput, or roughly
55% less TTFT. It is not the latest preferred result: later focused TinyLLAMA
cohorts select the four-row recipe at 358.06 tok/s. Use the historical 235.05 record for continuity and measure a fresh current-preferred control; do not spend a hardware batch reproducing an obsolete recipe.

The reference is a published comparison, not a matched local RKLLM run. Preserve
the exact TinyLLAMA Q8_0 GGUF, tokenizer, 128 prompt IDs, 64 generated IDs,
context 512, greedy sampling, uncached prefix, three warmups and three timed
requests per process. Keep all trials, source hashes, actual phase placement,
and host state. Report TTFT and the proxy separately from true prefill timing;
the proxy includes first-token work.

## What the local evidence says

- The old CPU + NPU INT8 sweep candidate measured 235.05 tok/s; the latest
  preferred local focused recipe is 358.06 tok/s. These are distinct cohorts
  and should not be blended. Decode is a separate metric and cannot be used to
  infer prefill headroom.
- Four-row SwiGLU prefill plus fused SwiGLU decode is the current preferred
  local recipe. Its 4-process ABBA study measured 347.20 tok/s control and
  358.06 tok/s candidate. All generated sequences and FP32 teacher arrays
  matched bitwise. This roughly 3% gain is useful, but far short of the 2.23x
  needed from the requested baseline. Evidence: [TinyLLAMA row-batch v2
  verification](benchmark-tinyllama-row-batch-models-verification-v2.json).
- The earlier raw SwiGLU prefill pool candidate improved 335.23 to
  347.20 tok/s. Narrow-CBUF and other decode focused experiments do not supply
  a credible large prefill lever. See [FFN pool v2
  verification](benchmark-ffn-prefill-pool-models-verification-v2.json) and
  [narrow CBUF comparison](benchmark-narrow-cbuf-models-combined-verification-v2.json).
- The QKV-without-norm experiment, grouped SwiGLU variants, and explicit pool
  work are already represented in the benchmark evidence. Recheck their exact
  TinyLLAMA result rows before revisiting; do not repeat these settings as a
  fresh hypothesis. The published [W8A8 coverage inventory](benchmark-w8a8-results.json)
  links the older variants and their source reports.
- Current placement code supports independent CPU/NPU prefill and decode
  selection, NPU projection tiling, pooled CPU work and optional attention
  placements in [rockchip.py in tinygrad](/home/orangepi/tinygrad/tinygrad/llm/kernels/rockchip.py).
  TinyLLAMA is an ordinary dense decoder here; do not assume a Qwen-specific
  attention or token-input fast path admits this checkpoint.

## Concrete TinyLLAMA differences to investigate

The retained candidate report `benchmark-tinyllama-row-batch-models-v2-tinyllama-11b-2-candidate.json` records 22 blocks, width 2048, FFN width 5632, 32 query heads, 4 KV heads, head dimension 64 and vocabulary 32000. It uses 22 CPU attention blocks, 22 fused SwiGLU blocks, 45 CPU norms and no prepared QKV blocks. Projection weight sharing was off, with about 1.66 GB of packed projection storage. These are inspection targets, not proof that each is a bottleneck.

Qwen3's prepared NPU attention currently admits only 16 query / 8 KV heads and dimension 128. TinyLLAMA needs an 8:1 query/KV ratio rather than 2:1, plus scaling by 1/sqrt(64) rather than 1/sqrt(128). Merely enabling the Qwen flag will fall back. If attention dominates the fresh profile, parameterize the helper and NPU plan geometry, share each K/V operand across eight query heads, and qualify QK/PV, causal masks, scale, dynamic tails and mixed INT8/FP16 transitions before model timing. This is the specific NPU alternative to the GPU placement screen. If attention is small, skip this work.

Check whether prepared QKV without normalization and shared packed projections have already been rejected for these shapes. Reuse the existing evidence before changing them; reduced memory alone is not a throughput win.

## Prioritized work

### 0. Match operating conditions before changing kernels

The Qwen3 fixed-stock experiment measured 856.18 prefill tok/s (all 12 requests above the published target) with supported CPU/NPU/DDR performance governors plus CPU6 NPU IRQ affinity and its idle setting. Use the same explicitly recorded conditions for a fresh TinyLLAMA baseline; record a new bounded tuning window when executing this plan; the completed Qwen3 window must first be restored. Do not predict TinyLLAMA speed by multiplying its 358.06 tok/s adaptive-host result by Qwen3's gain. Recompute TinyLLAMA's remaining latency gap from its own measurement before selecting a kernel or placement change. Preserve an ordinary adaptive-host figure alongside the fixed-stock reference comparison; these answer different deployment questions.

### 1. Re-establish and profile the exact 128-token baseline

Before another optimization, locate the report that produced 235.05 tok/s and
confirm checkpoint hash, input token IDs, flags, context, measurement boundary,
and actual placement. Run a fresh ABBA control against the current preferred
recipe only after reviewing the one-owner lease/restore runner. Capture a
separate diagnostic profile of one warmed prefill, with native replay verified.
Attribute wall time to projection compute, quantization/packing, NPU submit and
wait, attention, SwiGLU activation/packing, copies/cache maintenance, output
head, and Python/JIT input preparation. Keep the instrumented profile out of
throughput figures.

**Decision gate:** select the largest measured component that can plausibly
remove a substantial part of the roughly 114 ms gap from the newer 358.06 tok/s recipe (357.48 ms TTFT). If the profile is dominated by projection work, test
placement or batching. If host helpers dominate, optimize only the named helper.
If the diagnostic path falls back from native replay or does not cover the same
128-token work, repair the observation before forming a performance hypothesis.

### 2. Test GPU-first / hybrid prefill placement against the CPU + NPU recipe

Use the existing OpenCL Mali GPU linear path as the first placement candidate,
since the current CPU + NPU INT8 recipe has a large gap and existing product
policy prefers GPU-first execution. Compare (a) current CPU + NPU, (b) GPU-only
prefill and decode, and (c) GPU prefill with NPU only on projection shapes
where measured transfer plus synchronization costs still leave a gain. Keep
decode placement fixed while screening prefill. Verify layer/operation counts
and device submissions; a selected `GPU` profile is not proof that the intended
work ran there.

Reject any variant whose complete uncached request is slower, whose output
fails its declared quality gate (exact regression for arithmetic-preserving changes; independent quality evaluation for GPU/NPU arithmetic changes), or whose apparent gain comes from changed
input, precision, warmup, or token accounting. Retain the prior preferred
recipe until a repeated full-model candidate wins.

### 3. Improve the dominant projection path only if profiling supports it

If bulk projections dominate, screen existing supported NPU tile/row settings
and prepared-command reuse with TinyLLAMA's actual matrix shapes. Include
quantization, layout conversion, submission and synchronization in the timing.
If GPU GEMM dominates, profile the prepared Mali linear path and its packed
weight/layout choices. Do not import a register recipe from another model or
shape without TinyLLAMA numerical and command qualification. If SwiGLU packing
or CPU helpers dominate instead, test row batching/pool sizing around the
already measured four-row configuration; require an end-to-end saving, not a
faster isolated pack kernel.

For each candidate, write a predicted latency saving and falsification test
before the hardware run. Screen one change at a time, then evaluate only the
best justified combination. Stop a line of work once its measured complete
request fails to improve.

### 4. Qualification and win criterion

For arithmetic-preserving changes, require identical generated token IDs and
bitwise-equal full-vocabulary FP32 teacher vectors on the frozen prompts. If
arithmetic changes, predeclare a TinyLLAMA-appropriate quality gate and use an
independent checkpoint/logit or held-out language-model quality reference;
local self-equality is insufficient. Repeat unprofiled paired ABBA process
blocks, retain losses, and compare process-level distributions. A win requires
the candidate's stable TTFT distribution below 243.93 ms (equivalently proxy
above 524.74 tok/s), with its confidence margin clearing the observed run
variation. Then validate the selected path through the oNPU serving API at the
same uncached 128-token workload and check cached follow-up behavior separately.

If no candidate clears that threshold, report the measured improvement and the
remaining gap. Do not label the target reached from an operator benchmark,
single best request, or decode gain.

## Evidence consulted

- `benchmark.md`: RKLLM reference table, TinyLLAMA preferred rows, experiment
  comparisons and qualification limits.
- `docs/benchmark-w8a8-results.json`: machine-readable sweep inventory,
  including the 235.0528 tok/s variant, exact flags and linked raw report.
- `docs/benchmark-tinyllama-row-batch-models-verification-v2.json`: latest
  focused TinyLLAMA row-batching/fused-decode comparison.
- `docs/benchmark-ffn-prefill-pool-models-verification-v2.json` and
  `docs/benchmark-narrow-cbuf-models-combined-verification-v2.json`: preceding
  retained comparisons and selected recipe history.
- `plan.md`: project-level goal and qualification procedure.
- `/home/orangepi/.codex/skills/onpu-rkllm-performance/references/measurement.md`
  and `local-evidence.md`: measurement contract and local experiment lessons.
