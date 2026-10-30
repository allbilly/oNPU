"""Single-owner tinygrad inference worker. JSON lines are the only stdout protocol.

All model operations run on one thread; the input thread only queues commands and
sets cancellation flags. No changes are made to the user's tinygrad checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from .execution import (
    QUALIFIED_MALI_CHECKPOINTS,
    QUALIFIED_MALI_RECURRENT_CHECKPOINTS,
    qualified_mali_options,
    qualified_mali_profile,
)
from .rockchip_options import (
    TEXT_INT8_CONTEXTS,
    checked_text_profile,
    qwen3_prefill_profile,
    require_text_backend,
    text_load_options,
    text_projection_options,
)
from .speculation import MODE as SPECULATIVE_MODE
from .speculation import (
    configure_request_retention,
    decoder_metrics,
    draft_memory,
    load_decoder,
    prefix_cache_template,
    prepare_target_table,
    prepared_target_options,
    route_speculation,
    speculative_chunks,
    validate_speculation,
)


class Cancelled(Exception):
    pass


# Pinned CPU/NPU execution profiles, with bounded functional qualification.
# These entries do not assert comparable-quality RKLLM performance or accuracy.
QUALIFIED_INT8_CHECKPOINTS: frozenset[str] = frozenset({
    "6782193cfc9dc18b3ab1db4479893ef2d554650df7b33b460aeefe349e4ce035",
    "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
    "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe",
} | TEXT_INT8_CONTEXTS.keys())
QUALIFIED_INT8_CONTEXTS: dict[str, int] = {
    "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e": 8192,
    "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe": 512,
    **TEXT_INT8_CONTEXTS,
}
# SmolVLM-256M: complete prepared encoder and W8A8 decoder qualified at context 512.
QUALIFIED_GPU_VISION_CHECKPOINTS: frozenset[str] = frozenset({
    "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
})

def mali_options(path: Path, context: int, profile: str, npu_prefill: str, npu_decode: str, *, realize: bool = True) -> dict:
    return qualified_mali_options(path, context, profile, npu_prefill, npu_decode, realize=realize,
                                  checkpoints=QUALIFIED_MALI_CHECKPOINTS,
                                  recurrent_checkpoints=QUALIFIED_MALI_RECURRENT_CHECKPOINTS)


def mali_profile(path: Path, context: int, profile: str, npu_prefill: str, npu_decode: str, *, realize: bool = True) -> tuple[str | None, dict]:
    return qualified_mali_profile(path, context, profile, npu_prefill, npu_decode, realize=realize,
                                  checkpoints=QUALIFIED_MALI_CHECKPOINTS,
                                  recurrent_checkpoints=QUALIFIED_MALI_RECURRENT_CHECKPOINTS)


def npu_profile(path: Path, profile: str, context: int) -> tuple[str, str | None]:
    if profile != "ROCKCHIP" or context > 8192 or not QUALIFIED_INT8_CHECKPOINTS:
        return "fp16", None
    with path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    precision = "int8" if checksum in QUALIFIED_INT8_CHECKPOINTS and context <= QUALIFIED_INT8_CONTEXTS.get(checksum, 512) else "fp16"
    return precision, checksum


def npu_precision(path: Path, profile: str, context: int) -> str:
    return npu_profile(path, profile, context)[0]


def vision_gpu_enabled(path: Path, precision: str, requested: bool) -> bool:
    if precision != "int8" or not requested or not QUALIFIED_GPU_VISION_CHECKPOINTS:
        return False
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() not in QUALIFIED_GPU_VISION_CHECKPOINTS:
            return False
    try:
        from tinygrad import Device
        extensions = Device["CL"].device_exts
    except (RuntimeError, OSError, AttributeError, ImportError, IndexError):
        return False
    return {"cl_arm_import_memory_dma_buf", "cl_khr_fp16"}.issubset(extensions)


def prefill_chunks(context: int) -> list[int]:
    return sorted({min(32, context), *(size for size in (64, 128, 256) if size <= context)})


def prefill_chunk(remaining: int, context: int) -> int:
    if remaining <= 1:
        return 1
    buckets = prefill_chunks(context)
    return next((size for size in buckets if remaining <= size), buckets[-1])


def prepare_prompt(body: dict, model, tok, template, vision_session=None) -> tuple[str, list[int]]:
    if vision_session:
        from .vlm import render_messages
        if body.get("tools"):
            raise ValueError("This vision model does not support tool calls")
        rendered, _ = render_messages(body["messages"], vision_session.vlm.image_tokens)
    else:
        from tinygrad.llm.serve import normalize_messages
        normalize_messages(body["messages"])
        rendered = template.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True,
                                   preserve_thinking=True, enable_thinking=body.get("enable_thinking", False))
    ids = tok.encode(rendered)
    limit = body["max_tokens"]
    if len(ids) + limit > model.max_context:
        raise ValueError(f"Prompt ({len(ids)} tokens) + reply limit ({limit}) exceeds context ({model.max_context}). "
                         "Reduce the reply limit or start a new chat.")
    if vision_session:
        vision_session.prepare(body["messages"], limit)
    return rendered, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--realize", type=int, default=1)
    parser.add_argument("--speculation", choices=("off", SPECULATIVE_MODE), default="off")
    parser.add_argument("--draft-path", default="")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    write_lock = threading.Lock()

    def emit(event: str, **data):
        with write_lock:
            protocol.write(json.dumps({"event": event, **data}, ensure_ascii=False) + "\n")
            protocol.flush()

    try:
        import tinygrad
        from tinygrad import Device, Tensor
        from tinygrad.helpers import GlobalCounters
        if args.probe:
            if os.environ.get("ONPU_PROFILE") == "ROCKCHIP":
                from tinygrad.llm.kernels.rockchip import pack_weight, packed_matmul
                weight = pack_weight(Tensor.ones(32, 32, device="CPU")).realize()
                result = packed_matmul(Tensor.ones(1, 32, device="CPU"), weight, 32).tolist()
                if result != [[32.0] * 32]:
                    raise ValueError("NPU matmul smoke test failed")
                emit("probe", available=True, device="ROCKCHIP", name="RK3588 NPU dense projections + CPU")
                return
            result = (Tensor([1., 2., 3.]) + 1).tolist()
            if result != [2., 3., 4.]:
                raise ValueError("Tensor smoke test failed")
            name = Device.DEFAULT
            if name.startswith("CL"):
                device = Device[name]
                name = device.device_name
            emit("probe", available=True, device=Device.DEFAULT, name=name,
                 driver=getattr(Device[Device.DEFAULT], "driver_version", None))
            return

        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tinygrad import nn

        start = time.perf_counter()
        emit("state", state="loading", message="Reading model weights")
        vision_session = None
        scalar_jit = None
        checkpoint_path = Path(args.model)
        speculative_options = {"speculation": args.speculation, "draft_path": args.draft_path,
                               "profile": os.environ.get("ONPU_PROFILE", "CL"), "context": args.context, "realize": bool(args.realize),
                               "npu_prefill": os.environ.get("ONPU_NPU_PREFILL", "off"),
                               "npu_decode": os.environ.get("ONPU_NPU_DECODE", "off")}
        validate_speculation({"path": args.model}, speculative_options)
        if args.speculation != "off" and Device.DEFAULT.split(":")[0] != "CL":
            raise ValueError("Experimental DFlash2 requires the OpenCL worker")
        text_checkpoint = not checkpoint_path.is_dir()
        cpu_npu = Device.DEFAULT == "CPU"
        npu_prefill, npu_decode = os.environ.get("ONPU_NPU_PREFILL", "off"), os.environ.get("ONPU_NPU_DECODE", "off")
        precision = "fp16"
        precision_checksum = None
        if npu_prefill != "off" or npu_decode != "off":
            precision_path = checkpoint_path if text_checkpoint else checkpoint_path / "model.safetensors"
            precision, precision_checksum = npu_profile(precision_path, os.environ.get("ONPU_PROFILE", "CPU"), args.context)
        cpu_layout = checked_text_profile(precision_checksum, args.context, os.environ.get("ONPU_PROFILE", "CPU"), npu_prefill,
            cpu_npu=cpu_npu, text_checkpoint=text_checkpoint, precision=precision,
            expected_checksum=os.environ.get("ONPU_CPU_INT8_CHECKPOINT"))
        if cpu_npu and text_checkpoint and precision == "int8":
            cpu_layout.update(qwen3_prefill_profile(precision_checksum, args.context, os.environ.get("ONPU_PROFILE", "CPU"),
                                                    npu_prefill, npu_decode))
        if cpu_layout and cpu_npu and text_checkpoint and precision == "int8":
            npu_prefill = cpu_layout["prefill"]
            required_environment = cpu_layout.get("environment", {})
            if os.environ.get("ONPU_CPU_INT8_CHECKPOINT") and any(os.environ.get(key) != value
                    for key, value in required_environment.items()):
                raise ValueError("CPU/NPU execution options changed after memory admission")
            os.environ.update(required_environment)
        # Tokenizer/model imports can initialize backend ContextVars too.
        from tinygrad.llm.cli import FallbackTemplate, SimpleTokenizer
        from tinygrad.llm.model import Transformer
        from tinygrad.llm.serve import Handler
        if cpu_layout and cpu_npu and text_checkpoint and precision == "int8":
            from tinygrad.llm.kernels.rockchip import configure as configure_text
            require_text_backend(Transformer.from_gguf, configure_text,
                                 recurrent_scan=cpu_layout.get("cpu_recurrent_scan", False),
                                 fast_prefill=cpu_layout.get("qwen3_fast_prefill", False),
                                 swiglu=cpu_layout.get("dense_swiglu", False))
            if cpu_layout.get("qwen3_fast_prefill") and not bool(getattr(sys.modules[Transformer.__module__], "ONPU_CPU_TOKEN_INPUTS", False)):
                raise ValueError("This model's accelerated prefill requires a newer token-input backend")
        if not text_checkpoint:
            from tinygrad.llm.vision import SmolVLM

            from .vlm import VisionSession, load_tokenizer
            vlm = SmolVLM.from_hf(checkpoint_path, args.context)
            vision_session = VisionSession(vlm, load_tokenizer(checkpoint_path))
            model, kv = vlm.text, {"general.name": "SmolVLM (Idefics3)"}
            vision_session.prepare_prefill()
            checkpoint_path = checkpoint_path / "model.safetensors"
            if os.environ.get("ONPU_PROFILE") == "ROCKCHIP":
                from tinygrad.llm.kernels.rockchip_decode import prepare_scalar_decode
                scalar_jit = prepare_scalar_decode(model)
        else:
            model, kv = Transformer.from_gguf(args.model, args.context,
                **text_load_options(Transformer.from_gguf, cpu_npu=cpu_npu, precision=precision, realize=bool(args.realize)))
        gpu_kernels = None
        if Device.DEFAULT.split(":")[0] == "CL":
            from tinygrad.llm.kernels.mali import configure as configure_mali
            checksum, target_options = mali_profile(checkpoint_path, args.context, os.environ.get("ONPU_PROFILE", "CL"),
                                                   npu_prefill, npu_decode, realize=bool(args.realize))
            gpu_kernels = configure_mali(model, **prepared_target_options(target_options, args.speculation,
                                                                        checkpoint_sha256=checksum, context=args.context))
            if shared_table := prepare_target_table(model, checkpoint_path, args.speculation, checkpoint_sha256=checksum):
                gpu_kernels["q8_shared_table"] = shared_table
        placement = None
        if npu_prefill != "off" or npu_decode != "off":
            from tinygrad.llm.kernels.rockchip import configure
            if cpu_npu and hasattr(os, "sched_getaffinity"):
                cores = os.sched_getaffinity(0).intersection({4, 5, 6, 7})
                if cores:
                    os.sched_setaffinity(0, cores)
            emit("state", state="preparing", message="Preparing NPU weights and measuring optional placement")
            long_cpu = cpu_npu and args.context > 512
            packed_cpu = cpu_npu and precision == "int8" and args.context >= 4096
            placement_rows = (gpu_kernels["prefill_chunk"] if gpu_kernels else
                              vision_session.prefill_chunk if vision_session else prefill_chunks(model.max_context)[-1])
            placement_rows = min(placement_rows, cpu_layout.get("prefill_chunk", placement_rows))
            placement = configure(model, prefill=npu_prefill, decode=npu_decode, tile_columns=cpu_layout.get("tile_columns", 24576),
                                  prefill_rows=placement_rows,
                                  fuse_projections=cpu_npu, cpu_attention=cpu_npu,
                                  native_replay=cpu_npu, fused_ffn=cpu_npu, fused_qkv=cpu_npu,
                                  cpu_norm=cpu_npu, split_down=cpu_npu, precision=precision,
                                  cpu_pool=precision == "int8", cpu_fast_softmax=precision == "int8" or long_cpu,
                                  cache_cleaning=precision == "int8" or long_cpu, wake_helper=precision == "int8" or long_cpu,
                                  cpu_packed_attention=packed_cpu, cpu_int8_qk=packed_cpu,
                                  **text_projection_options(configure, cpu_npu=cpu_npu, precision=precision,
                                      text_checkpoint=text_checkpoint, prefill=npu_prefill, decode=npu_decode,
                                      recurrent_scan=cpu_layout.get("cpu_recurrent_scan", False),
                                      fast_prefill=cpu_layout.get("qwen3_fast_prefill", False),
                                      swiglu=cpu_layout.get("dense_swiglu", False)))
            if cpu_layout.get("dense_swiglu"):
                if (placement.get("fused_swiglu_ffn_blocks") != len(model.blk)
                        or npu_decode == "on" and placement.get("fused_swiglu_decode_blocks") != len(model.blk)):
                    raise ValueError("This model's fused SwiGLU execution is unavailable in the selected backend")
                if placement.get("prepared_qkv_blocks") != len(model.blk) or not placement.get("shared_host_pool"):
                    raise ValueError("This model's shared prefill and prepared QKV execution is unavailable")
                from tinygrad.llm.kernels.rockchip_cpu import ONPU_CPU_DECODE_POOL, ONPU_CPU_PREFILL_LAYOUT, ONPU_EXP2_FAST_PATH
                from tinygrad.llm.kernels.rockchip_padding import ONPU_ZERO_PAD_VIEWS
                from tinygrad.llm.kernels.rockchip_shared_pool import ONPU_PREFILL_CPU_TUNING, ONPU_SHARED_HOST_POOL
                from tinygrad.helpers import getenv
                placement["prefill_features"] = {
                    "cpu_prefill_layout": bool(ONPU_CPU_PREFILL_LAYOUT), "zero_pad_views": bool(ONPU_ZERO_PAD_VIEWS),
                    "shared_host_pool": bool(ONPU_SHARED_HOST_POOL), "cpu_tuning": bool(ONPU_PREFILL_CPU_TUNING),
                    "cpu_token_inputs": bool(getattr(sys.modules[Transformer.__module__], "ONPU_CPU_TOKEN_INPUTS", False)),
                    **{name: bool(getenv(key, 0)) for name, key in (
                        ("borrow_outputs", "ONPU_REPLAY_BORROW_OUTPUTS"), ("borrow_batch", "ONPU_REPLAY_BORROW_BATCH"),
                        ("native_from", "ONPU_REPLAY_NATIVE_FROM"))},
                }
                if not all(placement["prefill_features"].values()):
                    raise ValueError("This model's CPU prefill features were not initialized")
                extra_features = {"large_native_from": ("ONPU_REPLAY_NATIVE_FROM_LARGE", bool(getenv("ONPU_REPLAY_NATIVE_FROM_LARGE", 0))),
                                  "decode_pool": ("ONPU_CPU_DECODE_POOL", bool(ONPU_CPU_DECODE_POOL)),
                                  "exp2_fast_path": ("ONPU_EXP2_FAST_PATH", bool(ONPU_EXP2_FAST_PATH))}
                for name, (key, actual) in extra_features.items():
                    if actual != (cpu_layout["environment"].get(key, "0") == "1"):
                        raise ValueError("This model's CPU execution features were not initialized")
                    placement["prefill_features"][name] = actual
                placement["prefill_profile"] = cpu_layout["prefill_profile"]
            if cpu_layout.get("qwen3_fast_prefill"):
                if placement.get("npu_prefill_attention_blocks") != len(model.blk) or not placement.get("shared_host_pool"):
                    raise ValueError("This model's accelerated prefill is unavailable in the selected backend")
                from tinygrad.helpers import getenv
                from tinygrad.llm.kernels.rockchip_padding import ONPU_ZERO_PAD_VIEWS
                from tinygrad.llm.kernels.rockchip_shared_pool import (
                    ONPU_PREFILL_CPU_TUNING,
                    ONPU_SHARED_HOST_POOL,
                )
                placement["prefill_features"] = {
                    "shared_host_pool": bool(ONPU_SHARED_HOST_POOL), "zero_pad_views": bool(ONPU_ZERO_PAD_VIEWS),
                    "cpu_tuning": bool(ONPU_PREFILL_CPU_TUNING),
                    "cpu_token_inputs": bool(getattr(sys.modules[Transformer.__module__], "ONPU_CPU_TOKEN_INPUTS", False)),
                    **{name: bool(getenv(key, 0)) for name, key in (
                        ("borrow_outputs", "ONPU_REPLAY_BORROW_OUTPUTS"), ("borrow_batch", "ONPU_REPLAY_BORROW_BATCH"),
                        ("native_from", "ONPU_REPLAY_NATIVE_FROM"), ("static_layout", "ONPU_SWIGLU_PACK_STATIC_LAYOUT"))},
                }
                if not all(placement["prefill_features"].values()):
                    raise ValueError("This model's accelerated prefill options were not initialized")
                placement["prefill_profile"] = "qwen3-0.6b-context512"
            if cpu_layout.get("cpu_recurrent_scan") and placement.get("cpu_recurrent_blocks") != len(model.blk):
                raise ValueError("The selected backend did not prepare every recurrent block")
            if not placement["decode_layers"] and not placement["prefill_layers"]:
                if os.environ.get("ONPU_PROFILE") == "ROCKCHIP":
                    raise ValueError("This model has no supported NPU projections. Choose the GPU or CPU profile.")
                if placement["layers"]:
                    Device["ROCKCHIP"].finalize()
        tok = vision_session.tokenizer if vision_session else SimpleTokenizer.from_gguf_kv(kv)
        if vision_session:
            if os.environ.get("ONPU_PROFILE") == "ROCKCHIP":
                from tinygrad.llm.kernels.rockchip_vision_pipeline import StagedVision
                vision_placement = configure(vision_session.vlm.vision, prefill="on", decode="off")
                gpu_vision = vision_gpu_enabled(checkpoint_path, precision, os.environ.get("ONPU_VISION_GPU", "1") == "1")
                vision_session.encoder = StagedVision(vision_session.vlm, "ROCKCHIP", npu_connector=True, fused_mlp=True,
                                                      gpu_softmax=gpu_vision, prepared_mlp=gpu_vision)
                vision_placement.update(gpu_vision=gpu_vision, prepared_mlp=len(vision_session.encoder.prepared_mlps))
                vision_session.reset_vision = precision == "int8"
                placement["vision"] = vision_placement
                if scalar_jit is not None:
                    from tinygrad.llm.kernels.rockchip_decode import ScalarDecoder
                    from tinygrad.llm.kernels.rockchip_idle import idle_helper
                    from tinygrad.llm.kernels.rockchip_pool import attention_pool
                    vision_session.scalar_decoder = ScalarDecoder(model, scalar_jit,
                        pool=attention_pool() if placement.get("cpu_pool") else None,
                        helper=idle_helper() if placement.get("wake_helper") else None)
            elif Device.DEFAULT == "CPU":
                from tinygrad.llm.kernels.rockchip_vision import vision_attention
                for block in vision_session.vlm.vision.layers:
                    block.self_attn.attention = vision_attention
        template = FallbackTemplate(tok)
        if source := kv.get("tokenizer.chat_template"):
            source = prefix_cache_template(source, args.speculation,
                                           checkpoint_sha256=checksum if gpu_kernels is not None else None)
            env = ImmutableSandboxedEnvironment()
            env.filters["tojson"] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
            env.globals.update(raise_exception=lambda msg: (_ for _ in ()).throw(ValueError(msg)),
                               strftime_now=lambda fmt: time.strftime(fmt),
                               bos_token=tok.decode([tok.bos_id]) if tok.bos_id is not None else "", eos_token=tok.decode([tok.eos_id]))
            template = env.from_string(source)

        commands: queue.Queue = queue.Queue()
        cancellations: dict[str, threading.Event] = {}
        active_cancel = threading.Event()

        # Check between prefill chunks and decode operations, rather than waiting
        # for a complete prompt to produce its first token before honoring Stop.
        def checkpoint(jit):
            def run(*a, **kw):
                if active_cancel.is_set():
                    raise Cancelled()
                return jit(*a, **kw)
            return run
        for attr in ("prefill_jit", "rollout_jit", "greedy_prefill_jit", "greedy_rollout_jit", "embeddings_prefill_jit",
                     "greedy_embeddings_prefill_jit", "sampled_embeddings_prefill_jit"):
            if hasattr(model, attr):
                setattr(model, attr, checkpoint(getattr(model, attr)))

        adaptive_prefill = bool(placement and placement.get("native_replay") and placement["prefill_layers"])
        prefill_limit = cpu_layout.get("prefill_chunk", model.max_context)
        if adaptive_prefill:
            original_generate = model.generate
            def generate(tokens, chunk_size=None, **kwargs):
                if chunk_size is None:
                    chunk_size = prefill_chunk(len(tokens) - model.get_start_pos(tokens), min(model.max_context, prefill_limit))
                return original_generate(tokens, chunk_size=chunk_size, **kwargs)
            model.generate = generate
        elif gpu_kernels and gpu_kernels.get("linear_layers"):
            original_generate = model.generate
            def generate(tokens, chunk_size=None, **kwargs):
                return original_generate(tokens, chunk_size=chunk_size or gpu_kernels["prefill_chunk"], **kwargs)
            model.generate = generate

        emit("state", state="preparing", message="Compiling and warming the model")
        model.warmup()
        # Warm the separate sampling/greedy prompt and token graphs before ready.
        # Optional CPU/NPU placement also prepares each adaptive prompt size.
        chunks_to_warm = (prefill_chunks(model.max_context) if adaptive_prefill else
                          [gpu_kernels["prefill_chunk"]] if gpu_kernels and gpu_kernels.get("linear_layers") else [min(32, model.max_context)])
        chunks_to_warm = [chunk for chunk in chunks_to_warm if chunk <= prefill_limit]
        native = bool(placement and placement.get("native_replay") and (placement["decode_layers"] or placement["prefill_layers"]))
        stages = len(chunks_to_warm) * 2
        stage = 0
        for temperature in (0.0, 0.6):
            for chunk in chunks_to_warm:
                stage += 1
                emit("state", state="preparing", message=f"Preparing inference ({stage}/{stages})")
                prompt_length = min(chunk, model.max_context - 2)
                for _ in range(4 if native else 2):
                    model._cached_tokens = []
                    list(itertools.islice(model.generate([0] * prompt_length, chunk_size=chunk, temperature=temperature), 2))
        model._cached_tokens = []
        if vision_session:
            def vision_checkpoint():
                if active_cancel.is_set():
                    raise Cancelled()
            vision_session.check = vision_checkpoint
            vision_session.cancelled = lambda: active_cancel.is_set()
            emit("state", state="preparing", message="Preparing image encoding")
            pixels = Tensor.zeros(1, 3, 512, 512, device=vision_session.vlm.vision.embeddings.patch_embedding.weight.device).realize()
            for _ in range(4 if os.environ.get("ONPU_PROFILE") == "ROCKCHIP" else 2):
                vision_session.encode(pixels)
            emit("state", state="preparing", message="Preparing image prompt and generation")
            if native:
                vision_session.prefill_buckets = tuple(size for size in (32, 64, 96, 128) if size <= vision_session.prefill_chunk)
            for capacity in vision_session.prefill_buckets or (vision_session.prefill_chunk,):
                length = min(capacity, model.max_context - 10)
                embeddings = Tensor.zeros(1, length, model.config.dim, device=model.token_embd.weight.device).contiguous().realize()
                for _ in range(4 if native else 2):
                    first_token = vision_session.prefill(embeddings, greedy=True)
                emit("state", state="preparing", message=f"Preparing image prompt ({capacity} positions, sampled)")
                for _ in range(4 if native else 2):
                    vision_session.prefill(embeddings, temperature=.6)
            if vision_session.scalar_decoder is not None:
                first = int(first_token.item())
                vision_session.scalar_decoder(first, length, min(8, model.max_context - length))
            vision_session.install()
        decoder = None
        if args.speculation == SPECULATIVE_MODE:
            emit("state", state="preparing", message="Loading experimental greedy DFlash2 draft")
            decoder = load_decoder(model, checkpoint_path, args.draft_path, gpu_kernels["prefill_chunk"])
        actual_device = nn.state.get_parameters(model)[0].device
        base_device = actual_device
        if placement and (placement["decode_layers"] or placement["prefill_layers"]):
            gpu_vision = bool(placement.get("vision", {}).get("gpu_vision"))
            actual_device = f"{base_device}+{'CL+' if gpu_vision else ''}ROCKCHIP"
        handler = object.__new__(Handler)
        handler.server = SimpleNamespace(model=model, tok=tok)
        emit("ready", device=actual_device, context=model.max_context, name=kv.get("general.name", args.model), supports_images=bool(vision_session),
             load_seconds=time.perf_counter() - start, weight_bytes=GlobalCounters.mem_used,
             base_device=base_device, placement=placement, gpu_kernels=gpu_kernels,
             speculation={"mode": args.speculation, "greedy_only": decoder is not None and not callable(getattr(decoder,"sample",None)),
                          "sampled": decoder is not None and callable(getattr(decoder,"sample",None)),
                          "reserved_bytes": draft_memory(args.context, speculative_options),
                          "setup": getattr(decoder, "serving_setup", None)},
             backend_path=str(Path(tinygrad.__file__).resolve().parents[1]))

        def input_loop():
            try:
                for line in sys.stdin:
                    command = json.loads(line)
                    if command["op"] == "cancel":
                        if flag := cancellations.get(command["id"]):
                            flag.set()
                    else:
                        if command["op"] == "generate":
                            cancellations[command["id"]] = threading.Event()
                        commands.put(command)
            finally:
                for flag in list(cancellations.values()):
                    flag.set()
                commands.put({"op": "shutdown"})

        threading.Thread(target=input_loop, daemon=True).start()
        while True:
            command = commands.get()
            if command["op"] == "shutdown":
                return
            if command["op"] != "generate":
                continue
            key, body = command["id"], command["body"]
            active_cancel = cancellations[key]
            chunks = None
            terminal = None
            started = first = last = None
            try:
                prep = time.perf_counter()
                rendered, ids = prepare_prompt(body, model, tok, template, vision_session)
                limit = body["max_tokens"]
                if not body.get("cache_prompt", True):
                    model._cached_tokens = []
                routing = route_speculation(decoder, body["temperature"], body.get("speculation", "auto"),
                                            prompt_tokens=len(ids), max_tokens=limit, max_context=model.max_context,
                                            cached_tokens=model.get_start_pos(ids))
                speculative = routing["selected"] == SPECULATIVE_MODE
                cached = routing["cached_tokens"]
                if speculative:
                    configure_request_retention(decoder, routing["requested"])
                    cache_query = getattr(decoder, "get_start_pos", None)
                    cached = cache_query(ids) if callable(cache_query) else 0
                    if type(cached) is not int or not 0 <= cached < len(ids) or cached > routing["cached_tokens"]:
                        raise RuntimeError("Invalid speculative prefix cache accounting")
                    routing = {**routing, "speculative_cached_tokens": cached}
                npu_before = Device["ROCKCHIP"].submit_count if placement else 0
                if active_cancel.is_set():
                    raise Cancelled()
                emit("started", id=key, prompt_tokens=len(ids), cached_tokens=cached, preparation_seconds=time.perf_counter() - prep,
                     generation_mode=routing["selected"], speculation_routing=routing)
                started = time.perf_counter()
                if speculative:
                    chunks = speculative_chunks(handler, decoder, ids.copy(), body["model"], max_tokens=limit,
                                                reasoning=rendered.rstrip().endswith("<think>"),
                                                cancelled=active_cancel.is_set, cancel_error=Cancelled,temperature=body["temperature"],
                                                cached_tokens=cached)
                else:
                    chunks = handler.run_model(ids.copy(), body["model"], include_usage=True, max_tokens=limit,
                                               temperature=body["temperature"], reasoning=rendered.rstrip().endswith("<think>"))
                usage = None
                for chunk in chunks:
                    if active_cancel.is_set():
                        raise Cancelled()
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        if any(choice.get("delta", {}).get(field) for field in ("content", "reasoning_content", "tool_calls")):
                            first = first or time.perf_counter()
                            last = time.perf_counter()
                    emit("chunk", id=key, data=chunk)
                if active_cancel.is_set():
                    raise Cancelled()
                if speculative and getattr(getattr(decoder, "stats", None), "cached_tokens", cached) != cached:
                    raise RuntimeError("Speculative cached-token count changed during generation")
                ended = time.perf_counter()
                tokens = (usage or {}).get("completion_tokens", 0)
                # TTFT measures the first emitted text; token timings are not fabricated
                # from the role event, which is emitted before inference begins.
                elapsed = ended - started
                metrics = {"prompt_tokens": len(ids), "cached_tokens": cached, "completion_tokens": tokens,
                           "ttft_ms": (first - started) * 1000 if first else None, "total_seconds": elapsed,
                           "decode_tps": (tokens - 1) / (last - first) if tokens > 1 and first and last and last > first else None,
                           "prefill_tps": (len(ids) - cached) / (first - started) if first and first > started else None,
                           "device": actual_device, "allocated_bytes": GlobalCounters.mem_used,
                           "generation_mode": routing["selected"], "speculation_routing": routing}
                if speculative:
                    metrics["speculation"] = decoder_metrics(decoder)
                if placement:
                    metrics["npu_submissions"] = Device["ROCKCHIP"].submit_count - npu_before
                if vision_session and vision_session.timings:
                    metrics.update(vision_session.timings)
                    metrics["prefill_tps"] = (len(ids) - cached) * 1000 / vision_session.timings["prefill_ms"]
                    metrics["image_to_answer_seconds"] = time.perf_counter() - prep
                terminal = {"event": "done", "metrics": metrics}
            except Cancelled:
                terminal = {"event": "cancelled"}
            except ValueError as exc:
                # Rejected input has not written KV state. Preserve the prefix
                # so a reply-budget error does not force another cold prefill.
                if started is not None:
                    model._cached_tokens = []
                terminal = {"event": "error", "message": str(exc), "status": 400}
            except Exception as exc:  # noqa: BLE001 - report any request failure through terminal IPC
                model._cached_tokens = []
                traceback.print_exc()
                terminal = {"event": "error", "message": str(exc), "status": 500}
            finally:
                try:
                    if chunks is not None:
                        chunks.close()
                except Exception as exc:  # noqa: BLE001 - any cleanup failure must reach the request owner
                    model._cached_tokens = []
                    traceback.print_exc()
                    terminal = {"event": "error", "message": str(exc), "status": 500}
                cancellations.pop(key, None)
            # Runtime may admit another request after this event. Finish stream
            # rollback and release all target ownership before acknowledging it.
            if terminal is not None:
                emit(terminal.pop("event"), id=key, **terminal)
    except Exception as exc:  # noqa: BLE001 - report any worker startup or protocol failure through fatal IPC
        traceback.print_exc()
        emit("fatal", message=str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
