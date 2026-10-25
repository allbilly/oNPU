# oNPU and oMLX feature comparison

This snapshot compares oNPU with selected oMLX documentation at [revision `78fa08bc1aef7290c93c83bb70811547763ebf33`](https://github.com/jundot/omlx/tree/78fa08bc1aef7290c93c83bb70811547763ebf33). It describes documented features and local evidence; it does not claim benchmark parity.

| Area | oNPU status | Current gap or limit |
|---|---|---|
| DFlash2 | A trained Qwen3.5-2B/DFlash2 pair has a frozen V48 near-8K API qualification and a separate actual 16K-prefix core qualification for one greedy workload. All seven core cases passed independent audit at prompt 16,256/context 16,384/chunk 64. | The 16K result does not qualify HTTP/API16K, sampled16K, broad quality or benchmark speed. Speculation was slower in both single-request pairs: 1,111.006s vs 973.890s cold and 9.787s vs 7.202s cached. The API cap remains 8K; model maximum 262,144 remains unqualified. See [16K core evidence](speculative-actual16k-core-v1.json), [V48 API evidence](speculative-forced-batched-retention-api-v48.json), and [CPU prefill evidence](speculative-context-aware-prefill-cpu-v1.json). |
| EAGLE-4 | No EAGLE-4 execution path or tree verification is qualified. | No matched small Qwen3.5-2B pair appears in the inspected support matrix; this is not a claim that no such checkpoint exists elsewhere. See [EAGLE-4 support matrix](https://github.com/eagle-hq/EAGLE-4/blob/60bab6b469e795d47bda82b16243b82bb7b6206f/docs/support-matrix.md) and [freshness snapshot](omlx-eagle-freshness-20261025-v48.json). |
| Continuous batching | Independent GPU and optional NPU workers are available. | Each worker admits one active request; continuous batching within a model is missing. |
| KV cache | Per-worker RAM prefix reuse exists. | Shared paged RAM cache and SSD KV persistence are missing; oMLX documents shared paged hot RAM/SSD caching. |
| Context capacity | Small-request allocation probes completed at 8, 16, 32 and 64K configured capacities; a separate core test completed a real 16,256-token prompt. | The experimental DFlash2 API cap remains 8K; ordinary model limits differ. The capacity probes themselves used short prompts. Model maximum 262,144 is unqualified. See [capacity evidence](speculative-context-capacity-allocation-v1.json) and [actual 16K core evidence](speculative-actual16k-core-v1.json). |
| Serving API | The served OpenAPI surface lists `/v1/models` and `/v1/chat/completions`; chat supports streaming, tool/reasoning deltas and cancellation. | Anthropic, legacy completion, embeddings, reranking, Responses, public decision/tokenization, speech and schema-constrained routes are absent. Tool handling is template-dependent; multimodal support is partial. See [feature comparison JSON](omlx-feature-comparison.json). |
| Hardware | OpenCL GPU with optional Rockchip NPU assistance; independent GPU/NPU workers. | oMLX targets Apple MLX/Metal. oNPU's GPU-first plus optional NPU design is a different hardware focus; comparable performance is not established. |

## Evidence notes

- V48 remains the latest frozen near-8K HTTP/API cohort. The separate 16K proof qualifies one actual core workload, not API16K or sampled16K. The 64/128 alignment CPU proof remains CPU/application scoped.
- V48 did not execute a fresh probability audit. Historical probability-grid results belong to earlier cohorts and are not represented as current V48 parity.
- oMLX freshness is limited to the selected README and DFlash integration pages; it is not a full upstream code audit.
- See the machine-readable [comparison record](omlx-feature-comparison.json) for detailed feature rows, source pins, and historical cohorts.
