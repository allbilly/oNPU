"""Explicit experimental greedy DFlash2 serving; ordinary decoding is the default.

The API process only verifies local files and budgets memory. Device imports and
resident draft tensors belong to the single-owner inference worker.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .execution import checkpoint_checksum, qualified_mali_profile

MODE = "dflash2-experimental"
# Keep these pins aligned with tinygrad.llm.dflash2_config. The worker's loader
# independently verifies all three files and the actual target tensor shapes.
TARGET_SHA256 = "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1"
CONFIG_SHA256 = "f290e9af60a11850b747015ae437da91706405a180d85d95c736ee6ad3c6420d"
WEIGHTS_SHA256 = "a9f6d56744e3a23ea50eaf20fe2115832ae5364ad607a416a48f96b8800fbb8e"
WEIGHTS_BYTES = 708494328
MAX_CONTEXT = 8192
# Whole-pair experimental envelope, including both checkpoints and warm graphs.
# HTTP-v3's 36 context512 requests peaked at 9896.81640625 MiB of owned RSS,
# including its probe/server overhead. Reserve another 1536 MiB plus 512 MiB
# for the new single-row continuation and startup graphs. This conservative
# admission estimate is not a maximum-memory or speed/quality qualification.
# The V16 context8192 core trial used 8978.14 MiB at 2061 input tokens.
# Keep the larger reservation while qualifying the API and its full input range.
MEASURED_TARGET_SIZE = 2012012800
MEASURED_TARGET_PARAMETERS = 1881825088
DRAFT_FC_PACKED_BYTES = 2048 * 12288 * 2
# Selective draft feature-projection packing retains one extra FP16 weight copy.
MEASURED_PAIR_BYTES = (9897 + 1536 + 512) * 1024**2 + DRAFT_FC_PACKED_BYTES
TARGET_PACKED_BYTES = 1017118720 + 905969664


def prefix_cache_template(source: str, mode: str, *, checkpoint_sha256: str | None) -> str:
    """Honor preserve_thinking for the pinned pair's previous assistant turns.

    Its embedded template otherwise drops the generated think header whenever
    a newer user turn follows, invalidating strict recurrent-prefix reuse.
    """
    if mode != MODE:
        return source
    if checkpoint_sha256 != TARGET_SHA256:
        raise ValueError("Prefix-preserving template requires the pinned target")
    condition = "loop.index0 > ns.last_query_index"
    preserved = "preserve_thinking|default(false) or " + condition
    if source.count(condition) != 1:
        raise ValueError("Unsupported pinned speculative chat template")
    return source if preserved in source else source.replace(condition, preserved)


def _validate_draft_files(root: Path, *, refresh: bool = True) -> None:
    config, weights = root / "config.json", root / "model.safetensors"
    if not config.is_file() or config.stat().st_size > 1024 * 1024 or checkpoint_checksum(config, refresh=refresh) != CONFIG_SHA256:
        raise ValueError("DFlash2 configuration does not match the pinned checkpoint")
    if not weights.is_file() or weights.stat().st_size != WEIGHTS_BYTES or checkpoint_checksum(weights, refresh=refresh) != WEIGHTS_SHA256:
        raise ValueError("DFlash2 weights do not match the pinned checkpoint")


def validate_speculation(model: dict, options: dict) -> str | None:
    mode, directory = options.get("speculation", "off"), options.get("draft_path")
    if mode == "off":
        if directory:
            raise ValueError("draft_path requires speculation=dflash2-experimental")
        return
    if mode != MODE:
        raise ValueError("Unknown speculative decoding mode")
    if (options.get("profile") != "CL" or not options.get("realize", True)
            or any(options.get(key, "off") != "off" for key in ("npu_prefill", "npu_decode"))):
        raise ValueError("Experimental DFlash2 requires the GPU profile, prepared weights and NPU assistance off")
    context = options.get("context")
    if type(context) is not int or not 64 <= context <= MAX_CONTEXT:
        raise ValueError(f"Experimental DFlash2 currently requires context between 64 and {MAX_CONTEXT}")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Experimental DFlash2 requires a local draft_path directory")
    root = Path(directory).expanduser().resolve(strict=True)
    target = Path(model["path"]).expanduser().resolve(strict=True)
    if not root.is_dir() or not target.is_file() or target.suffix != ".gguf":
        raise ValueError("Experimental DFlash2 requires a local GGUF target and draft directory")
    _validate_draft_files(root)
    if checkpoint_checksum(target) != TARGET_SHA256:
        raise ValueError("Target checkpoint/tokenizer identity does not match the pinned DFlash2 pair")
    return str(root)


def measured_pair_memory(model: dict, context: int, realize: bool, options: dict | None, *, refresh: bool = True) -> int | None:
    """Use the whole-pair diagnostic only for its pinned files and worker layout.

    Admission reads fresh hashes; refresh=False is reserved for advisory display
    estimates, consistent with the ordinary checkpoint-qualified memory policy.
    """
    if (not options or options.get("speculation") != MODE or options.get("profile") != "CL" or not realize
            or any(options.get(key, "off") != "off" for key in ("npu_prefill", "npu_decode"))
            or type(context) is not int or not 64 <= context <= MAX_CONTEXT or model.get("architecture") != "qwen35"
            or (model.get("size"), model.get("parameters")) != (MEASURED_TARGET_SIZE, MEASURED_TARGET_PARAMETERS)
            or not model.get("path") or not isinstance(options.get("draft_path"), str) or not options["draft_path"].strip()):
        return None
    try:
        checksum, layout = qualified_mali_profile(Path(model["path"]), context, "CL", "off", "off", realize=True, refresh=refresh)
        if checksum != TARGET_SHA256:
            return None
        prepared_target_options(layout, MODE, checkpoint_sha256=checksum, context=context)
        _validate_draft_files(Path(options["draft_path"]).expanduser().resolve(strict=True), refresh=refresh)
    except (OSError, ValueError, TypeError, RuntimeError):
        return None
    return MEASURED_PAIR_BYTES


def draft_memory(context: int, options: dict | None) -> int:
    if not options or options.get("speculation", "off") == "off":
        return 0
    # Reserve BF16 source plus resident FP16 weights; all target features,
    # projected prefix, FP32 prefix K/V, block logits and extra working space.
    # This is a conservative reservation, not a measured peak or a speed claim.
    prefix = context * (6 * 2048 * 4 + 2048 * 2 + 2 * 4 * 4 * 128 * 4)
    logits = 2 * 8 * 248320 * 4
    return 2 * WEIGHTS_BYTES + prefix + logits + TARGET_PACKED_BYTES + DRAFT_FC_PACKED_BYTES + 1024 * 1024**2


def prepared_target_options(base: dict, mode: str, *, checkpoint_sha256: str | None = None, context: int | None = None) -> dict:
    """Configure the experimental pair before ordinary and speculative graph capture."""
    options = dict(base)
    if mode == "off":
        return options
    if mode != MODE:
        raise ValueError("Prepared DFlash2 requires the pinned recurrent GPU execution layout")
    if checkpoint_sha256 is not None and checkpoint_sha256 != TARGET_SHA256:
        raise ValueError("Prepared DFlash2 requires the pinned target checkpoint")
    if context is not None and (type(context) is not int or not 64 <= context <= MAX_CONTEXT):
        raise ValueError("Prepared DFlash2 context is outside the experimental range")
    expected = {"recurrent_scan": True, "pack_weights": True}
    extended = type(context) is int and 512 < context <= MAX_CONTEXT
    if extended and checkpoint_sha256 != TARGET_SHA256:
        raise ValueError("Extended DFlash2 context requires the pinned target checkpoint")
    # Ordinary models keep their separately qualified context ranges. Only this
    # verified pair may supply its base layout beyond the ordinary registry.
    if not options and extended and checkpoint_sha256 == TARGET_SHA256:
        options = expected.copy()
    if options != expected:
        raise ValueError("Prepared DFlash2 requires the pinned recurrent GPU execution layout")
    # Consistent decode uses the row-major head for this pair's <=8-row calls.
    # Keep its unused packed copy out of GPU residency; admission stays conservative.
    return {**options, "pack_output": False, "pack_gdn_gate_up": True, "consistent_decode": True,
            "native_state_snapshots": True, "gated_attention": True, "bounded_half_prefill": True}


def prepare_target_table(model, checkpoint: Path, mode: str, *, checkpoint_sha256: str | None) -> dict | None:
    """Install the measured shared table before worker warmup, for this pair only."""
    if mode != MODE:
        return None
    if checkpoint_sha256 != TARGET_SHA256:
        raise ValueError("Shared speculative table requires the pinned target checkpoint")
    from tinygrad import Device

    weight = model.token_embd.weight
    device_name, shape = weight.device, tuple(weight.shape)
    del weight  # Do not retain the old dense matrix across replacement/free_cache.
    if not isinstance(device_name, str) or device_name.split(":")[0] != "CL":
        raise ValueError("Shared speculative table requires a single OpenCL target")
    device = Device[device_name]
    if (getattr(device, "device_name", "") != "Mali-LODX r0p0"
            or getattr(device, "driver_version", "") != "2.1"):
        return None
    from tinygrad.llm.kernels.mali_q8_checkpoint import Q8Checkpoint
    from tinygrad.llm.kernels.mali_q8_shared import replace_tied_q8_table

    source = Q8Checkpoint(checkpoint)
    if "output.weight" in source.entries or shape != (248320, 2048):
        raise ValueError("Shared speculative table requires the pinned tied vocabulary matrix")
    raw = source.weight("token_embd.weight", shape, device_name)
    metadata = replace_tied_q8_table(model, raw)
    return {"enabled": True, "helper_metadata": metadata}


def target_table_metadata(model) -> dict | None:
    words, head = getattr(model, "token_embd", None), getattr(model, "output", None)
    table = getattr(words, "table", None)
    if table is None:
        return None
    from tinygrad.llm.kernels.mali_q8_shared import MaliQ8Embedding, MaliQ8Head

    if not isinstance(words, MaliQ8Embedding) or not isinstance(head, MaliQ8Head) or head.table is not table:
        raise RuntimeError("Speculative embedding and head must share the same Q8 table")
    return {"enabled": True, "shared_table": True, **table.metadata()}


def calibrate_ordinary_decode(model) -> dict:
    """Time already-warmed, synchronized single-token calls, excluding prefill."""
    if type(model.max_context) is not int or model.max_context < 3:
        raise ValueError("Decode calibration needs room for a prompt and two outputs")
    prompt = [0] * min(128, model.max_context - 2)
    samples = []
    try:
        for _ in range(3):
            model._cached_tokens = []
            tokens = model.generate(prompt.copy(), temperature=0.0)
            try:
                next(tokens)  # The ordinary generator realizes and reads each output ID.
                started = time.perf_counter()
                next(tokens)
                elapsed = time.perf_counter() - started
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("Ordinary decode calibration must be finite and positive")
                samples.append(elapsed)
            finally:
                tokens.close()
    finally:
        model._cached_tokens = []
    # A slow sample can include residual capture work. The fastest warmed sample
    # keeps the fallback conservative about speculation's expected benefit.
    return {"seconds_per_token": min(samples), "samples": samples, "prompt_tokens": len(prompt)}


def warm_prepared_decoder(decoder) -> dict:
    """Exercise prepared block and single-row continuation graphs before Ready."""
    count = 24
    if type(decoder.target.max_context) is not int or decoder.target.max_context <= count:
        raise ValueError("Prepared warmup needs room for a prompt and 24 outputs")
    chunk = getattr(decoder, "prefill_chunk_size", 128)
    if type(chunk) is not int or chunk not in (64, 128):
        raise ValueError("Prepared prefill chunk must be 64 or 128")
    prompt = [0] * min(chunk, decoder.target.max_context - count)
    observations, traces = [], []
    try:
        for _ in range(3):
            # A synthetic reference forces the continuation path for graph setup
            # only. Each real request receives its measured ordinary reference.
            tokens = decoder.generate(prompt, max_new_tokens=count, ordinary_decode_seconds=1e-12,
                                      prefill_chunk_size=chunk)
            try:
                if len(list(tokens)) != count:
                    raise ValueError("Prepared decoder warmup ended before its fixed output count")
            finally:
                tokens.close()
            observations.append(decoder.stats.prompt_observe_seconds)
            traces.append(decoder.stats.trace_prepare_seconds)
    finally:
        decoder.target._cached_tokens = []
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in observations + traces)
            or min(observations) <= 0):
        raise ValueError("Prepared observation calibration must be finite and positive")
    return {"requests": 3, "output_tokens_per_request": count, "forced_fallback_for_setup": True,
            "prompt_observation_calibration": {"prompt_tokens": len(prompt), "samples_seconds": observations,
                                               "seconds_per_token": min(observations) / len(prompt),
                                               "trace_samples_seconds": traces, "trace_seconds": min(traces)}}


def load_decoder(model, checkpoint: Path, directory: str, prefill_chunk_size: int = 128):
    if type(prefill_chunk_size) is not int or prefill_chunk_size not in (64, 128):
        raise ValueError("Speculative prefill chunk must be 64 or 128")
    from tinygrad import Device, dtypes
    from tinygrad.llm.dflash2 import DFlash2
    from tinygrad.llm.dflash2_config import load_dflash2
    from tinygrad.llm.dflash2_selector import CandidateSelector
    from tinygrad.llm.speculative import GreedySpeculativeDecoder

    calibration = calibrate_ordinary_decode(model)
    config, weights = load_dflash2(Path(directory).expanduser(), model.config, target_checkpoint=checkpoint,
                                  device="CL", dtype=dtypes.float16)
    draft = DFlash2(config, weights, model.token_embd, model.output, model.output_norm, CandidateSelector(config, weights))
    # Keep target packing ahead of graph capture in the worker. The draft borrows
    # that same head. Only the draft's input projection gets an extra packed copy,
    # whose exact storage is included in admission before the worker is launched.
    draft_mali = draft.prepare_mali(pack_weights=False, prepare_shared_head=True, pack_fc=True)
    if draft_mali["packed_fc"] is not True or draft_mali["packed_bytes"] != DRAFT_FC_PACKED_BYTES:
        raise ValueError("Prepared draft input projection does not match its reserved packed storage")
    draft.prepare_proposals(max_context=model.max_context)
    draft.prepare_observation(max_context=model.max_context, pad_decode_rows=True, native_cache_copy=True,
                              bound_prefix_memory=True, retain_target_features=False)
    decoder = GreedySpeculativeDecoder(model, draft, prepared_target=True, prepared_prefill_dynamic=True,
                                      prepared_dynamic_rows=True, prepared_exact_replay=True, prefix_trace=True,
                                      early_anchor=True, deferred_fallback=True, reuse_prefix=True)
    decoder.ordinary_decode_seconds = calibration["seconds_per_token"]
    decoder.prefill_chunk_size = prefill_chunk_size
    decoder.serving_setup = {"ordinary_decode_calibration": calibration, "draft_mali": draft_mali,
                            "prefill_chunk_size": prefill_chunk_size, "prepared_warmup": warm_prepared_decoder(decoder)}
    if table_metadata := target_table_metadata(model):
        decoder.serving_setup["target_table"] = table_metadata
    # Match the measured probe's setup cleanup before reporting worker readiness.
    Device["CL"].synchronize()
    Device["CL"].allocator.free_cache()
    return decoder


def use_speculation(decoder, temperature: float, mode: str = "auto") -> bool:
    if mode not in ("auto", "off", "force"):
        raise ValueError("Request speculation must be auto, off or force")
    eligible = (decoder is not None and math.isfinite(temperature) and temperature>=0
                and (temperature == 0.0 or callable(getattr(decoder,"sample",None))))
    if mode == "force" and not eligible:
        raise ValueError("Forced speculation requires a compatible loaded draft and sampling mode")
    return mode != "off" and eligible


def configure_request_retention(decoder, mode: str) -> None:
    """Retain fallback prefixes only for explicit forced experiments on the measured profile.

    Cached ordinary decoding had the lowest combined cost in the trained study.
    Automatic requests therefore retain their existing scalar, non-retaining path.
    Older backends and other target layouts keep their existing configuration.
    """
    if decoder is None:
        return
    setup = getattr(decoder, "serving_setup", None) or {}
    if not isinstance(setup, dict):
        return
    table = setup.get("target_table") or {}
    if (not isinstance(table, dict) or table.get("shared_table") is not True
            or type(getattr(decoder, "retain_fallback_prefix", None)) is not bool
            or type(getattr(decoder, "fallback_observation_batch_size", None)) is not int
            or not callable(getattr(getattr(decoder, "draft", None), "observe_features", None))):
        return
    forced = mode == "force"
    decoder.retain_fallback_prefix = forced
    decoder.fallback_observation_batch_size = 8 if forced else 1


def route_speculation(decoder, temperature: float, mode: str, *, prompt_tokens: int, max_tokens: int,
                      max_context: int, cached_tokens: int) -> dict:
    """Reject auto setup whose estimated cost cannot amortize within the reply.

    Startup rates are estimates, not bounds: context, acceptance and early EOS
    can change runtime. Marginal block fallback remains independent of this
    upfront decision. Force bypasses cost routing for explicit experiments.
    """
    eligible = use_speculation(decoder, temperature, mode)
    if (any(type(value) is not int for value in (prompt_tokens, max_tokens, max_context, cached_tokens))
            or not 0 <= cached_tokens <= prompt_tokens <= max_context or max_tokens < 0):
        raise ValueError("Invalid speculative routing token budget")
    budget = min(max_tokens, max_context - prompt_tokens)
    result = {"requested": mode, "selected": "ordinary", "reason": "request_off" if mode == "off" else "draft_unavailable",
              "prompt_tokens": prompt_tokens, "reply_budget": budget, "cached_tokens": cached_tokens,
              "estimated_setup_seconds": None, "estimated_ordinary_decode_seconds": None}
    if not eligible:
        if mode != "off" and decoder is not None:
            result["reason"] = "unsupported_temperature"
        return result
    if mode == "force":
        return {**result, "selected": MODE, "reason": "forced_experiment"}
    if budget <= 1:
        return {**result, "reason": "no_decode_budget"}
    if cached_tokens:
        return {**result, "reason": "cached_prefix"}
    setup = getattr(decoder, "serving_setup", None) or {}
    calibration = (setup.get("prepared_warmup") or {}).get("prompt_observation_calibration") or {}
    observation_rate = calibration.get("seconds_per_token")
    trace_seconds = calibration.get("trace_seconds")
    ordinary_rate = getattr(decoder, "ordinary_decode_seconds", None)
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (observation_rate, ordinary_rate))
            or type(trace_seconds) not in (int, float) or not math.isfinite(trace_seconds) or trace_seconds < 0):
        return {**result, "reason": "calibration_unavailable"}
    setup_seconds = prompt_tokens * observation_rate + trace_seconds
    decode_seconds = (budget - 1) * ordinary_rate
    if not math.isfinite(setup_seconds) or not math.isfinite(decode_seconds):
        return {**result, "reason": "calibration_unavailable"}
    return {**result, "selected": MODE if setup_seconds < decode_seconds else "ordinary",
            "reason": "setup_amortizable" if setup_seconds < decode_seconds else "draft_setup_cost",
            "estimated_setup_seconds": setup_seconds, "estimated_ordinary_decode_seconds": decode_seconds}


def speculative_chunks(handler, decoder, ids: list[int], model_name: str, *, max_tokens: int,
                       reasoning: bool, cancelled, cancel_error, temperature: float = 0.0, cached_tokens: int = 0):
    """Keep the established text/tool formatter and explicitly own its iterator.

    Closing the outer stream always closes the decoder, including a role-only
    disconnect or a formatter exception. Terminal worker events must follow this
    cleanup, since decoder.close() can replay a delivered prefix during rollback.
    """
    if type(cached_tokens) is not int or not 0 <= cached_tokens < len(ids):
        raise ValueError("Invalid speculative cached-token count")
    tok = handler.server.tok
    eos = {value for key in ("eos_id", "eot_id", "eom_id") if (value := getattr(tok, key, None)) is not None}
    chunk = getattr(decoder, "prefill_chunk_size", 128)
    if type(chunk) is not int or chunk not in (64, 128):
        raise ValueError("Speculative prefill chunk must be 64 or 128")
    options = {"max_new_tokens":max_tokens,"eos_token_ids":eos,"cancelled":cancelled,"prefill_chunk_size":chunk,
               "ordinary_decode_seconds":getattr(decoder,"ordinary_decode_seconds",None)}
    tokens = decoder.generate(ids,**options) if temperature==0.0 else decoder.sample(ids,temperature=temperature,**options)
    chunks = None
    try:
        source = iter(tokens)
        proxy = SimpleNamespace(get_start_pos=lambda _ids: cached_tokens, generate=lambda _ids, **_kwargs: source)
        formatted = copy.copy(handler)
        formatted.server = SimpleNamespace(model=proxy, tok=tok)
        chunks = formatted.run_model(ids, model_name, include_usage=True, max_tokens=max_tokens,
                                     temperature=temperature, reasoning=reasoning)
        for chunk in chunks:
            if cancelled():
                raise cancel_error()
            yield chunk
        if cancelled():
            raise cancel_error()
        if getattr(decoder, "reuse_prefix", False):
            # The formatter stops on the final yielded ID. Resume the bounded
            # decoder once so its normal finish and cache retention run. A
            # disconnect/error skips this path and still closes the iterator.
            end = object()
            if next(source, end) is not end:
                raise RuntimeError("Formatter ended before the bounded speculative iterator")
    finally:
        try:
            if chunks is not None:
                chunks.close()
        finally:
            tokens.close()


def decoder_metrics(decoder) -> dict:
    result = {"mode": MODE, "stats": asdict(decoder.stats),
              "prepared_draft": decoder.draft.proposal_preparation_metadata(),
              "prepared_observation": decoder.draft.observation_preparation_metadata(),
              "serving_setup": getattr(decoder, "serving_setup", None)}
    if table_metadata := target_table_metadata(getattr(decoder, "target", None)):
        result["target_table"] = table_metadata
    return result
