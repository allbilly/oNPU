# Model Performance Benchmark

Updated: 2026-10-25. Platform: Orange Pi 5, RK3588S, 16 GB RAM.

Reference: [Rockchip rknn-llm benchmark.md](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md).
The reference table uses maximum CPU/NPU frequencies and W8A8 exports made with `optimization_level=0`.
The allocated context is not stated in the upstream table. Local pilots use context 512 and existing host clock policies.

Performance cells below show **RKLLM / oNPU**. **Bold identifies the better reported number for the same model**,
not a certified win under matched conditions. A dash means a matching local measurement is unavailable.
The main table records GPU-first OpenCL baselines. Dtype cells show reference / local precision;
FP16 GPU operands currently come from Q8_0 GGUF checkpoints. Experimental CPU+NPU results are listed separately.
No row is currently qualified as faster in both prefill and decode at comparable output quality.
The 21 RK3588 baseline rows remain FP16 GPU measurements. New CPU+NPU INT8 measurements are recorded separately in the W8A8 sweep section below.

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
RWKV7 also uses one-token generic recurrent execution: its measured TTFT is 160.08 seconds (0.80 prefill proxy tok/s), and decode is 0.80 tok/s. Its full-model output quality is not qualified.
Qwen3-VL-2B measures its language decoder with text input only; image encoder/connector performance is not included.
DeepSeekOCR also measures text-only language decoding; its image encoder and connector are excluded.

<!-- onpu-w8a8-sweep:start -->
## oNPU W8A8 sweep: CPU+NPU candidates

These are actual CPU+NPU INT8 projection runs, shown as **published RKLLM / oNPU**.
The measurement pass has stopped; unsuccessful rows still need fixes and reruns. A dash is pending or failed, never an inferred result.
Each timing row requires three full warmups and three measured 128-input/64-output requests at context 512,
stable measured IDs and positive NPU submissions in both phases. **Bold marks only the better reported number.**
Whole-model INT8 coverage and comparable output quality remain unqualified; no row is a certified RKLLM win.

A controlled Qwen3.5-4B follow-up captured an errno-14 failure for a 20 MiB
INT8 projection buffer (8,192×2,560) while more than 5 GiB remained available.
The running kernel explicitly reported an IOVA allocation failure. The tracked
DMA ranges span almost 4 GiB; intervals between tracked buffers are not proven
free. This is allocation evidence with deferred weight realization, not a new
successful timing row or a cause claim for every failed model. Both original
models were restored and answered fresh requests.
[Controlled allocation and kernel-log evidence](docs/benchmark-w8a8-allocation-context-v3.json).

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :---------- | :--------: | :---: | :----: | :--------: | :------: | :------: | :--------: |
| Qwen2 | 0.5B | w8a8 / INT8† | 128 | 64 | **145.90** / 439.06 | **41.58** / 33.74 | 669.56 / 2177.62* |
| MiniCPM4 | 0.5B | w8a8 / INT8† | 128 | 64 | **135.29** / 177.75 | **45.34** / 42.73 | 534.82 / 1977.30* |
| Qwen3 | 0.6B | w8a8 / INT8† | 128 | 64 | **199.09** / 227.34 | 32.91 / **34.24** | 791.11 / 2447.89* |
| TinyLLAMA | 1.1B | w8a8 / INT8† | 128 | 64 | **243.93** / 544.56 | **24.43** / 22.35 | 1093.66 / 4517.74* |
| Qwen2.5 | 1.5B | w8a8 / INT8† | 128 | 64 | **378.31** / 958.38 | **16.69** / 14.27 | 1689.21 / 6470.10* |
| RWKV7 | 1.5B | w8a8 / INT8† | 128 | 64 | **634.34** / 15275.51 | **13.82** / 9.11 | 1486.19 / 5113.72* |
| InternLM2 | 1.8B | w8a8 / INT8† | 128 | 64 | **380.14** / 803.98 | **15.45** / 14.09 | 1775.56 / 6222.81* |
| Gemma2 | 2B | w8a8 / — | 128 | 64 | 598.41 / — | 10.37 / — | 2779.22 / — |
| Gemma3n | 2B | w8a8 / — | 128 | 64 | 1124.51 / — | 9.91 / — | 2702.59 / — |
| TeleChat2 | 3B | w8a8 / — | 128 | 64 | 618.48 / — | 10.18 / — | 2788.39 / — |
| Phi3 | 3.8B | w8a8 / — | 128 | 64 | 1017.28 / — | 7.45 / — | 3758.34 / — |
| MiniCPM3 | 4B | w8a8 / — | 128 | 64 | 1408.49 / — | 5.94 / — | 4384.02 / — |
| ChatGLM3 | 6B | w8a8 / — | 128 | 64 | 1352.94 / — | 4.98 / — | 5985.99 / — |
| Qwen3-VL | 2B | w8a8 / INT8† | 128 | 64 | **383.62** / 732.18 | **14.98** / 14.39 | 1868.98 / 5752.71* |
| DeepSeekOCR | 3B(A570M) | w8a8 / INT8† | 128 | 64 | **701.21** / 37366.15 | **31.60** / 3.15 | 3066.66 / 6833.05* |
| Qwen3.5 | 0.8B | w8a8 / INT8† | 128 | 64 | **587.74** / 16676.34 | **27.05** / 8.00 | 1039.66 / 2878.52* |
| Qwen3.5 | 2B | w8a8 / INT8† | 128 | 64 | **776.95** / 1602.53 | **13.59** / 10.73 | 2121.91 / 6326.94* |
| Qwen3.5 | 4B | w8a8 / — | 128 | 64 | 2068.88 / — | 6.22 / — | 4575.78 / — |
| Gemma4 | E2B | w8a8 / — | 128 | 64 | 599.06 / — | 11.12 / — | 2498.70 / — |
| LFM2.5-VL | 450M | w8a8 / INT8† | 128 | 64 | **102.74** / 274.46 | **62.16** / 54.25 | 436.94 / 1681.74* |
| Spark-X2.5 | 1.7B | w8a8 / INT8† | 128 | 64 | **386.23** / 2407.40 | **15.61** / 9.10 | 1830.32 / 5817.18* |

Additional candidate with no RKLLM reference row:

| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |
| :---------- | :--------: | :---: | :----: | :--------: | :------: | :------: | :--------: |
| Spark-X2.6 | 4B | — / — | 128 | 64 | — / — | — / — | — / — |
Spark-X2.6 candidate status: invalid.

**Recorded main-table models: 13/21.** Original sweep snapshot: 2026-10-23T09:48:47.724248+00:00.
The Qwen3.5-2B preferred row now uses the October 25 CPU sigmoid/pool follow-up.
Its 128-row CPU scan, parked head workers and pooled prefill sigmoid preserve
the prior INT8 output sequence across all twelve control/candidate requests.
[Updated model evidence](docs/benchmark-cpu-sigmoid-backend-model-v1.json).
† Dynamic per-token activation/per-channel weight INT8 projections, INT32 accumulation and FP32 host arithmetic.
Qwen3 uses the preferred matched FP16 source where selected. Other checkpoint provenance is in the result record.
The learned-matrix coverage audit is separate; INT8 placement alone does not establish that every learned projection was quantized.
* Local memory is lifetime peak process RSS in MiB and is not the reference memory metric; no memory winner is declared.

Prefill proxies are `128000 / TTFT(ms)` and include first-token work.

| Model | Model Size | Prefill proxy(tok/s), RKLLM / oNPU INT8 |
| :--- | :---: | ---: |
| Qwen2 | 0.5B | **877.31** / 291.53 |
| MiniCPM4 | 0.5B | **946.12** / 720.13 |
| Qwen3 | 0.6B | **642.93** / 563.03 |
| TinyLLAMA | 1.1B | **524.74** / 235.05 |
| Qwen2.5 | 1.5B | **338.35** / 133.56 |
| RWKV7 | 1.5B | **201.78** / 8.38 |
| InternLM2 | 1.8B | **336.72** / 159.21 |
| Gemma2 | 2B | 213.90 / — |
| Gemma3n | 2B | 113.83 / — |
| TeleChat2 | 3B | 206.96 / — |
| Phi3 | 3.8B | 125.83 / — |
| MiniCPM3 | 4B | 90.88 / — |
| ChatGLM3 | 6B | 94.61 / — |
| Qwen3-VL | 2B | **333.66** / 174.82 |
| DeepSeekOCR | 3B(A570M) | **182.54** / 3.43 |
| Qwen3.5 | 0.8B | **217.78** / 7.68 |
| Qwen3.5 | 2B | **164.75** / 79.87 |
| Qwen3.5 | 4B | 61.87 / — |
| Gemma4 | E2B | 213.67 / — |
| LFM2.5-VL | 450M | **1245.86** / 466.37 |
| Spark-X2.5 | 1.7B | **331.41** / 53.17 |
| Spark-X2.6 | 4B | — / — |

Pending/failed rows: Gemma2-2B: invalid; Gemma3n-2B: invalid; TeleChat2-3B: invalid; Phi3-3.8B: invalid; MiniCPM3-4B: invalid; ChatGLM3-6B: resource-interrupted; Qwen3.5-4B: invalid; Gemma4-E2B: resource-interrupted.
[Per-model runs, variants, ranges, failures and hashes](docs/benchmark-w8a8-results.json).
Vision/connector performance is excluded for text-only VLM decoder cases.
<!-- onpu-w8a8-sweep:end -->

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
<!-- onpu-additional-benchmarks:end -->

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
