"""Optional CPU/INT8 text-worker options for compatible tinygrad backends."""
from __future__ import annotations

import inspect
from collections.abc import Callable

RWKV7_CHECKPOINT = "dfad28509a04d6ea719f441834fa33b1f53ef5471e927a7bcae847894ff07d1c"
TINYLLAMA_CHECKPOINT = "a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59"
QWEN25_CHECKPOINT = "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8"
QWEN2_CHECKPOINT = "a07bd33b0e52babc1083e97d9bb6018197d7af25697552cc4d0b751e035158ff"
QWEN3_PREFILL_CHECKPOINT = "6782193cfc9dc18b3ab1db4479893ef2d554650df7b33b460aeefe349e4ce035"
TEXT_INT8_CONTEXTS: dict[str, int] = {
    "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe": 512,
    RWKV7_CHECKPOINT: 512,
    TINYLLAMA_CHECKPOINT: 512,
    QWEN25_CHECKPOINT: 512,
    QWEN2_CHECKPOINT: 512,
}


def qwen3_prefill_profile(checksum: str | None, context: int, profile: str, prefill: str, decode: str) -> dict:
    """The measured dense Qwen3 recipe, limited to its checked checkpoint/context."""
    if (checksum != QWEN3_PREFILL_CHECKPOINT or context != 512 or profile != "ROCKCHIP"
            or prefill not in ("auto", "on") or decode != "on"):
        return {}
    return {"prefill": "on", "mapped_cpu": True, "share_projection_weights": True, "prefill_chunk": 128,
            "tile_columns": 24384, "qwen3_fast_prefill": True, "environment": {
                "ONPU_SHARED_HOST_POOL": "1", "ONPU_ZERO_PAD_VIEWS": "1", "ONPU_PREFILL_CPU_TUNING": "1",
                "ONPU_CPU_TOKEN_INPUTS": "1",
                "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1",
                "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_SWIGLU_PACK_STATIC_LAYOUT": "1",
            }}


def text_execution_profile(checksum: str | None, context: int, profile: str, prefill: str) -> dict:
    dense_profiles = {QWEN2_CHECKPOINT: ("qwen2-0.5b-context512", 2, 24576),
                      TINYLLAMA_CHECKPOINT: ("tinyllama-1.1b-context512", 4, 24384),
                      QWEN25_CHECKPOINT: ("qwen2.5-1.5b-context512", 2, 24576)}
    if (checksum in dense_profiles and context == 512 and profile == "ROCKCHIP"
            and prefill in ("auto", "on")):
        name, rows, tile_columns = dense_profiles[checksum]
        environment = {
            "ONPU_SWIGLU_PREFILL_ROWS": str(rows), "ONPU_INT8_EXPERT_DEPTHS": "0",
            "ONPU_INT8_NATURAL2304": "1", "ONPU_INT8_DECODE_NARROW_CBUF": "1",
            "ONPU_NPU_DOMAINS": "1", "ONPU_NPU_DOMAIN_BYTES": str(3 * 1024**3),
            "ONPU_CPU_TOKEN_INPUTS": "1", "ONPU_ZERO_PAD_VIEWS": "1",
            "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1", "ONPU_REPLAY_NATIVE_FROM": "1",
            "ONPU_SHARED_HOST_POOL": "1", "ONPU_PREFILL_CPU_TUNING": "1", "ONPU_CPU_PREFILL_LAYOUT": "1",
        }
        if checksum in (QWEN2_CHECKPOINT, QWEN25_CHECKPOINT):
            environment.update(ONPU_REPLAY_NATIVE_FROM_LARGE="1", ONPU_CPU_DECODE_POOL="1", ONPU_EXP2_FAST_PATH="1")
        if checksum == QWEN2_CHECKPOINT:
            environment.update(ONPU_INT8_EXPERT_DEPTHS="1", ONPU_SWIGLU_PACK_STATIC_LAYOUT="1")
        return {"prefill": "on", "mapped_cpu": True, "share_projection_weights": False, "prefill_chunk": 128,
                "tile_columns": tile_columns, "dense_swiglu": True, "prefill_profile": name, "environment": environment}
    if (profile == "ROCKCHIP" and prefill == "auto" and checksum is not None
            and 1 <= context <= TEXT_INT8_CONTEXTS.get(checksum, 0)):
        # The pinned full-model comparison selected NPU projections for both
        # phases. Resolving auto before loading avoids materializing a second
        # CPU model just to repeat those placement measurements.
        options = {"prefill": "on", "mapped_cpu": True, "share_projection_weights": True}
        if checksum == RWKV7_CHECKPOINT:
            options.update(cpu_recurrent_scan=True, prefill_chunk=128, environment={
                "ONPU_INT8_SMALL_DEPTH": "1", "ONPU_RWKV_DECODE_SPIN": "1",
                "ONPU_NPU_DOMAINS": "3", "ONPU_NPU_DOMAIN_BYTES": str(3 * 1024**3),
            })
        return options
    return {}


def checked_text_profile(checksum: str | None, context: int, profile: str, prefill: str, *, cpu_npu: bool,
                         text_checkpoint: bool, precision: str, expected_checksum: str | None = None) -> dict:
    layout = (text_execution_profile(checksum, context, profile, prefill)
              if cpu_npu and text_checkpoint and precision == "int8" else {})
    if expected_checksum and (checksum != expected_checksum or not layout):
        raise ValueError("Checkpoint or execution profile changed after memory admission")
    return layout


def supported_options(function: Callable, options: dict) -> dict:
    """Pass optional keywords only when the selected backend advertises them."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(options)
    names = {name for name, parameter in parameters.items()
             if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    return {name: value for name, value in options.items() if name in names}


def text_load_options(loader: Callable, *, cpu_npu: bool, precision: str, realize: bool) -> dict:
    options = {"realize": realize}
    if cpu_npu and precision == "int8" and supported_options(loader, {"mapped_cpu": True}):
        # Prepared NPU weights and CPU kernels own their snapshots. Retaining
        # another fully decoded CPU model wastes the serving memory budget.
        options.update(realize=False, mapped_cpu=True)
    return options


def require_text_backend(loader: Callable, configure: Callable, *, recurrent_scan: bool = False, fast_prefill: bool = False,
                         swiglu: bool = False) -> None:
    if (not supported_options(loader, {"mapped_cpu": True})
            or not supported_options(configure, {"share_projection_weights": True})):
        raise ValueError("This CPU/NPU profile requires backend support for mapped CPU loading and shared projection weights")
    if recurrent_scan and not supported_options(configure, {"cpu_recurrent_scan": True}):
        raise ValueError("This CPU/NPU profile requires backend support for native CPU recurrent scans")
    if fast_prefill and not supported_options(configure, {"npu_prefill_attention": True}):
        raise ValueError("This model's accelerated prefill requires a newer NPU backend")

    if swiglu and len(supported_options(configure, {"fused_swiglu_ffn": True, "fused_swiglu_decode": True,
                                                  "fused_qkv_without_norm": True})) != 3:
        raise ValueError("This model requires fused SwiGLU prefill and decode support")


def text_projection_options(configure: Callable, *, cpu_npu: bool, precision: str,
                            text_checkpoint: bool, prefill: str, decode: str, recurrent_scan: bool = False,
                            fast_prefill: bool = False, swiglu: bool = False) -> dict:
    if not cpu_npu or precision != "int8" or not text_checkpoint:
        return {}
    options = {"cpu_static_gemm": True, "cpu_gelu": True, "cpu_softcap": True,
               "fused_gelu_ffn": True, "cpu_prefill_pool": prefill != "off"}
    if prefill == decode == "on":
        options["share_projection_weights"] = True
    if recurrent_scan:
        options["cpu_recurrent_scan"] = True
    if fast_prefill:
        options.update(cpu_prefill_pool=False, npu_prefill_attention=True)
    if swiglu:
        options.update(fused_swiglu_ffn=True, fused_swiglu_decode=decode == "on", fused_qkv_without_norm=True,
                       share_projection_weights=False, cpu_static_gemm=False, cpu_gelu=False, cpu_softcap=False,
                       fused_gelu_ffn=False)
    return supported_options(configure, options)
