# llama.cpp CPU kernel review for oNPU

Reviewed local upstream checkout `~/rkllama/ggml-org_llama.cpp`, commit `b92761a515ea31e852e7fbc1fad5f874b46f3718` (October 3). The RK3588 exposes NEON, FP16 and dot-product features; its reported feature set lacks SVE and I8MM. The relevant comparisons use the 128-bit NEON path.

| Area | llama.cpp implementation | oNPU decision |
|---|---|---|
| SiLU/SwiGLU and softmax | `ggml/src/ggml-cpu/vec.cpp`, NEON helpers in `vec.h` | A normal-input fast path reduces repeated exceptional-value handling. Keep oNPU's coefficients and operation order for the first candidate. |
| Quantization | `ggml/src/ggml-cpu/arch/arm/quants.c`: eight independent vector loads/maxima, horizontal max and nearest-even conversion | Evaluate the reduction structure while retaining per-token/per-K-group FP32 scales, min-scale behavior, nonfinite policy and packed byte layout. |
| RMS normalization | `ggml/src/ggml-cpu/ops.cpp`: optional fused weight multiplication, with double-precision accumulation | oNPU already fuses the weight multiplication. Changing the reduction precision/order needs separate qualification. |
| Worker scheduling | `ggml/src/ggml-cpu/ggml-cpu.c`: polling, barriers and condition-variable waits | oNPU's scoped persistent workers already provide active waiting during inference and park outside it. Extra parking during an NPU request can affect both wakeup latency and adaptive CPU frequencies. |

## Measured activation comparison

One CPU4, 131,072 gate/up elements, identical input distribution and compiler optimization, warm alternating blocks:

| Variant | Median component time, ms | Output scope |
|---|---:|---|
| Existing oNPU polynomial path | 0.617 | Baseline |
| oNPU normal-range shortcut, original polynomial | 0.432 | Bitwise equal in tested exponential and gate outputs |
| llama.cpp NEON helper from the local header | 0.274 | 55,987 of 131,072 output float bit patterns differ; max absolute error about 3.81e-6 |

These are component timings, not complete inference results. The new oNPU shortcut takes the unchanged checked path for nonfinite inputs or magnitude at least 128. It reuses the original range reduction, polynomial coefficients, multiplication order and exceptional handling. The CPU probe checked all 65,536 FP16 encodings converted to FP32 plus 1,048,576 random FP32 bit patterns. Repo tests additionally execute the implementation through the actual CPU ELF loader and compare activation quantization payloads/scales for finite and nonfinite inputs.

The implementation uses inline helpers. A separate cold text section exposed an existing internal cross-section call-relocation limitation in the CPU ELF loader; an isolated test process faulted. The inline implementation passes the actual-loader tests. That loader limitation remains a distinct follow-up, and no cold out-of-line helper is required by this candidate.

`ONPU_EXP2_FAST_PATH` defaults off and is selected before graph capture. A matched Qwen2.5 control/candidate/candidate/control cohort completed under unchanged adaptive governors, with exact full-vocabulary teacher regression gates. Candidate medians were 344.54 prefill-proxy / 16.80 decode tok/s; control 344.63 / 16.70. Prefill is tied; decode improves about 0.6%. The component gain does not translate into a comparable full-model gain. Independent RKLLM quality equivalence remains outside these local regression checks.

[Probe data and source hashes](llama-cpu-kernel-probe-v1.json). Sources: [NEON activation implementation](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/ggml/src/ggml-cpu/vec.h), [ARM quantization](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/ggml/src/ggml-cpu/arch/arm/quants.c), [CPU operators](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/ggml/src/ggml-cpu/ops.cpp).


## Qwen2 0.5B follow-up — October 30

The current `0893ec9` CPU-only binary was run on the same Qwen2 Q8_0 checkpoint and exact 128 backend prompt IDs. Three context512 CLI requests measured medians **291.95 prompt-eval / 43.48 decode tok/s**. The separate synthetic pp128/tg64 means were **289.37 / 39.22**. Prompt evaluation is an internal counter, not first-answer latency. The app's five measured requests now report 741.77 input/TTFT proxy and 43.94 decode tok/s, with the qualification scope in [the Qwen2 receipt](benchmark-qwen2-dense-software-v1.json).

The inspected ARM `quantize_row_q8_0` works in 32-element blocks, stores each scale in FP16, and uses NEON max and nearest-even conversion. Its block-scale arithmetic differs from oNPU's per-token/per-K-group FP32 scales; Q8_0 storage alone does not make these engines equivalent. `ggml_vec_dot_q8_0_q8_0` and ARM repacking are relevant CPU references, not register-compatible replacements for NPU projections.

Clang14 `-mcpu=native` resolves to Cortex-A76 on this board. A CPU-only O2/O3 comparison of the actual Qwen2-shaped attention and raw SwiGLU helpers produced exact outputs and essentially equal times (attention 0.8261/0.8266 ms; SwiGLU 1.1669/1.1688 ms). No compiler option was changed. The retained improvement uses prepared QKV/shared workers and the existing two-KV-head decode pool. The latter passes exact-output checks for Qwen2's 14-head, 2-KV-head, dim64 geometry and NaN-poisoned future cache through length512. No approximate activation polynomial or alternative Q8 quantizer was ported.
