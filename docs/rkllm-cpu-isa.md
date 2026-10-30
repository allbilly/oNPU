# RKLLM CPU kernel and ISA inspection

Captured 2026-10-29 on this RK3588 board, using the existing Qwen3-0.6B W8A8
RKLLM model and runtime 1.3.1. This is diagnostic evidence; it does not add a
throughput result or establish an all-model speed win.

## What was captured

User-space CPU-clock sampling at 997 Hz on performance cores 4–7, with separate
prefill/decode windows, followed by bounded GDB observations at hot function
entries. Three 128-input/32-output SDK runs produced 8,650 samples, with zero
reported lost samples. Of these, 1,848 fall inside prefill and 5,602 inside
decode; startup samples are excluded from those phase totals. A separate
two-output-token run captured 24 entry records, including integer arguments,
readable pointer previews and callers. Both SDK processes exited normally.

The stripped SDK's internal semantic names are unavailable. Addresses below
are ELF virtual addresses, resolved through the actual process mappings and
unwind ranges. The nearest public symbol printed by objdump is not the
internal function's name. Debugger/profiler timings are excluded from benchmark
comparisons. Sampling excludes kernel execution and does not measure blocked
wall time, NPU time or a kernel's fraction of request latency.

## Findings and implications

| Observed CPU path | Evidence | Comparison with oNPU / action |
|---|---|---|
| FP16 vector reduction, ELF `0x2e47a0` | Four independent `fmla ... .8h` accumulators process 32 half elements per loop. Sampled entry length is 160. This range accounts for 7.46% of decode user-CPU samples. | Our inspected CPU attention uses four-lane FP32 arithmetic with FP16 cache loads. FP16 accumulation is a concrete alternative to evaluate where quality permits; these are not identical numerical operations. The exact SDK graph operation is not assigned from this routine alone. |
| Five-output reduction, ELF `0x317380` | Five FP16 vectors are widened to FP32; one FP32 vector is reused by five FMA accumulators. This range accounts for 7.12% of decode user-CPU samples. | Reuse across outputs is worth comparing with our cache/reduction loops. Instruction counts alone do not establish a speedup or prove this is the same graph operation. |
| Activation range reduction, ELF `0x325290` | SIMD FP32 min/max. Observed counts are 65,536 in prefill and 1,024 in decode. Captured return address `0x325384` reaches a caller computing `(max-min)/255` and rounding `-128-min/scale`. | The observed SDK path derives an asymmetric scale/zero point over the supplied range. Our inspected quantizer uses symmetric per-row `max(abs(x))/127` with round-to-even. Changing this is a quantization change requiring independent quality checks, not a register-only speed patch. Do not generalize one observed call to every SDK layer. |
| Integer output conversion, ELF `0x323c94` | Converts INT32 vectors to FP32, applies a subtraction using a column table, multiplies by scales, and places output columns. Observed row counts are 64 and 1. Prefill share is 12.93% of user-CPU samples. | CPU conversion and output layout are material parts of the SDK too. Our FFN path already fuses conversion, SwiGLU and packing; mapped output borrowing separately removes qualified copies. Compare complete data flow rather than copying this routine wholesale. |
| OpenMP synchronization | libgomp has 32.41% of prefill and 58.94% of decode user-CPU samples. Almost all of these hit a tight shared-state polling loop. | These percentages are largely waiting/synchronization activity, not useful matrix arithmetic and not equivalent latency savings. SDK worker persistence is visible, but blindly reproducing its spinning policy is unjustified. |

## Correction to our threading diagnosis

Our dumped packing kernels already contain FP32 NEON arithmetic and a vector
polynomial for the activation function. The opportunity is not simply to
enable SIMD or use a different CPU ISA. The candidate differences concern
precision, reuse, layout, memory traffic and dispatch.

The earlier plan described current packed INT8 FFN preparation as serial. That
was inaccurate for the pilot's large prefill shapes. At `rows*depth >= 32768`,
`rockchip_int8._ffn_pack` creates three pthreads, works on a fourth partition in
the caller, and joins the threads on every invocation. The inspected 128×3072
kernel takes this path. The activation quantizer also creates threads at its
large-shape threshold. These kernels are separate from the persistent CPU
attention pool; enabling that pool does not replace their thread lifecycle.

This exposed a problem in an **unexecuted** combined-host benchmark stage:
its process-wide `taskset -c 4` would make newly created packing threads inherit
CPU 4. The stage must allow CPUs 4–7 for prefill and retain the attention pool's
own scoped decode pinning. Existing measurements are not retrospectively
changed. Four later timing processes validated the corrected CPU mask; no
isolated speed gain is attributed to affinity alone. A CPU-only pthread check confirmed three
simultaneous workers inherit `[4]` from a CPU-4-only caller, and `[4,5,6,7]`
from a caller allowed all four performance cores.

## Next decisions

1. The matched combined-host comparison now completes with 247.64–253.19 ms
   candidate TTFT and 37.68–38.04 decode tok/s. All 1,024 natural-target vectors
   preserve the prior sampled-quality result bitwise. Prefill is still below
   RKLLM's published target; production defaults remain unchanged.
2. Its attention subphase profile measures 30.90 ms in CPU helpers, 15.85 ms in
   NPU submit/wait and 3.65 ms in synchronization across 28 layers. Use this and
   the full-request FFN preparation profile to choose the next CPU change. Persistent
   packing workers are a hypothesis, not an automatic win: an earlier parked
   FFN implementation lost its model comparison. A new attempt needs a new
   measured premise and an explicit affinity check.
3. If CPU reductions remain dominant, prototype FP16 accumulation or reuse
   across multiple outputs against an independent numerical oracle, then test
   original-checkpoint quality and unprofiled request speed. Keep quantization
   changes separate so their quality effect is measurable.

The combined-host timing and quality workers both restored serving, and both
original models answered fresh requests. [Timing, quality and phase-clock evidence](benchmark-qwen3-combined-host-v1.json).

No inference arithmetic or production defaults were changed for this capture.
All 295 numerical source hashes, the SDK library and model hashes remained
unchanged during capture. The original GPU and NPU serving models were restored
and both answered fresh requests.

## Evidence

- [Capture, phase counts, hot ranges and binary hashes](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-cpu-isa-v1/results/capture.json)
- [Bounded SDK disassemblies](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-cpu-isa-v1/results/sdk-asm)
- [Entry registers, pointer previews and stacks](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-cpu-isa-v1/results/targeted/phase-markers.jsonl)
- [Our generated C, ELF objects and disassemblies](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-cpu-isa-offline-v1/ours/report.json)
- [Our CPU-kernel comparison inputs](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-cpu-isa-offline-v1/ours/comparison-input.json)
- [CPU-only pthread affinity check](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-cpu-isa-offline-v1/ours/pthread-affinity.json)
- [OpenMP polling-loop analysis](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/staged/rkllm-cpu-isa-offline-v1/libgomp-hot/report.json)
- [Restored serving and fresh replies](/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/rkllm-cpu-isa-v1-fresh.json)
