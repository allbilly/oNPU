"""CPU/INT8 serving uses the qualified layouts without assuming a backend version."""
from onpu.rockchip_options import (
    checked_text_profile,
    require_text_backend,
    supported_options,
    text_execution_profile,
    text_load_options,
    text_projection_options,
)


def current_loader(path, context, *, realize=True, mapped_cpu=False):
    return path, context, realize, mapped_cpu


def old_loader(path, context, *, realize=True):
    return path, context, realize


def current_configure(model, *, share_projection_weights=False, cpu_prefill_pool=False,
                      cpu_static_gemm=False, cpu_gelu=False, cpu_softcap=False, fused_gelu_ffn=False,
                      cpu_recurrent_scan=False):
    return locals()


def test_int8_text_loader_borrows_file_weights_instead_of_decoding_a_second_model():
    options = text_load_options(current_loader, cpu_npu=True, precision="int8", realize=True)
    assert current_loader("weights", 512, **options) == ("weights", 512, False, True)
    assert text_load_options(current_loader, cpu_npu=False, precision="int8", realize=True) == {"realize": True}
    assert text_load_options(current_loader, cpu_npu=True, precision="fp16", realize=False) == {"realize": False}


def test_qwen2_profile_pins_the_measured_checkpoint_context_and_dense_execution():
    from onpu.rockchip_options import QWEN2_CHECKPOINT, TEXT_INT8_CONTEXTS

    assert TEXT_INT8_CONTEXTS[QWEN2_CHECKPOINT] == 512
    for prefill in ("auto", "on"):
        recipe = text_execution_profile(QWEN2_CHECKPOINT, 512, "ROCKCHIP", prefill)
        assert recipe["dense_swiglu"] is True and recipe["share_projection_weights"] is False
        assert recipe["prefill_profile"] == "qwen2-0.5b-context512"
        assert recipe["prefill_chunk"] == 128 and recipe["tile_columns"] == 24576
        for flag in ("ONPU_CPU_DECODE_POOL", "ONPU_EXP2_FAST_PATH", "ONPU_INT8_EXPERT_DEPTHS",
                     "ONPU_REPLAY_NATIVE_FROM_LARGE", "ONPU_SHARED_HOST_POOL", "ONPU_INT8_SDK_PREFILL",
                     "ONPU_FFN_PACKED_UPLOAD", "ONPU_CPU_GQA_PREFILL"):
            assert recipe["environment"][flag] == "1"
        assert recipe["environment"]["ONPU_SWIGLU_PREFILL_ROWS"] == "4"
    for args in ((QWEN2_CHECKPOINT, 513, "ROCKCHIP", "auto"), (QWEN2_CHECKPOINT, 512, "CL", "auto"),
                 (QWEN2_CHECKPOINT, 512, "ROCKCHIP", "off"), ("other", 512, "ROCKCHIP", "auto")):
        assert text_execution_profile(*args) == {}


def test_older_loader_keeps_its_original_loading_contract():
    options = text_load_options(old_loader, cpu_npu=True, precision="int8", realize=True)
    assert options == {"realize": True}
    assert old_loader("weights", 512, **options) == ("weights", 512, True)


def test_full_cpu_int8_text_profile_enables_shared_and_supported_host_kernels():
    options = text_projection_options(current_configure, cpu_npu=True, precision="int8", text_checkpoint=True,
                                      prefill="on", decode="on")
    assert options == {"share_projection_weights": True, "cpu_prefill_pool": True, "cpu_static_gemm": True,
                       "cpu_gelu": True, "cpu_softcap": True, "fused_gelu_ffn": True}
    assert current_configure(None, **options)["fused_gelu_ffn"] is True


def test_partial_placement_does_not_request_incompatible_weight_sharing():
    options = text_projection_options(current_configure, cpu_npu=True, precision="int8", text_checkpoint=True,
                                      prefill="auto", decode="off")
    assert "share_projection_weights" not in options
    assert options["cpu_prefill_pool"] is True


def test_gpu_fp16_and_vision_profiles_keep_existing_options():
    for changes in ({"cpu_npu": False}, {"precision": "fp16"}, {"text_checkpoint": False}):
        settings = {"cpu_npu": True, "precision": "int8", "text_checkpoint": True, "prefill": "on", "decode": "on", **changes}
        assert text_projection_options(current_configure, **settings) == {}


def test_optional_keywords_are_filtered_for_old_or_positional_only_signatures():
    def old_configure(model, *, cpu_softcap=False):
        return cpu_softcap
    options = text_projection_options(old_configure, cpu_npu=True, precision="int8", text_checkpoint=True,
                                      prefill="on", decode="on")
    assert options == {"cpu_softcap": True}
    assert old_configure(None, **options) is True
    def positional(cpu_gelu, /):
        return cpu_gelu
    assert supported_options(positional, {"cpu_gelu": True}) == {}
    assert supported_options(lambda **kwargs: kwargs, {"cpu_gelu": True}) == {"cpu_gelu": True}


def test_new_pinned_auto_profile_resolves_to_measured_npu_placement():
    from onpu.rockchip_options import TEXT_INT8_CONTEXTS
    checksum = next(iter(TEXT_INT8_CONTEXTS))
    assert text_execution_profile(checksum, 512, "ROCKCHIP", "auto") == {
        "prefill": "on", "mapped_cpu": True, "share_projection_weights": True}
    for context, profile, prefill, digest in ((513, "ROCKCHIP", "auto", checksum),
                                             (512, "CL", "auto", checksum), (512, "ROCKCHIP", "off", checksum),
                                             (512, "ROCKCHIP", "auto", "different")):
        assert text_execution_profile(digest, context, profile, prefill) == {}


def test_required_memory_layout_features_fail_before_loading_on_old_backend():
    import pytest
    require_text_backend(current_loader, current_configure)
    with pytest.raises(ValueError, match="mapped CPU loading"):
        require_text_backend(old_loader, current_configure)
    def old_configure(model):
        raise AssertionError("Do not execute a backend while checking its signature")
    with pytest.raises(ValueError, match="shared projection weights"):
        require_text_backend(current_loader, old_configure)


def test_worker_rejects_changed_identity_or_incompatible_layout_before_loading():
    import pytest

    from onpu.rockchip_options import TEXT_INT8_CONTEXTS

    checksum = next(iter(TEXT_INT8_CONTEXTS))
    settings = {"cpu_npu": True, "text_checkpoint": True, "precision": "int8", "expected_checksum": checksum}
    assert checked_text_profile(checksum, 512, "ROCKCHIP", "auto", **settings)["mapped_cpu"] is True
    for changed in ({"cpu_npu": False}, {"text_checkpoint": False}, {"precision": "fp16"}):
        with pytest.raises(ValueError, match="after memory admission"):
            checked_text_profile(checksum, 512, "ROCKCHIP", "auto", **{**settings, **changed})
    with pytest.raises(ValueError, match="after memory admission"):
        checked_text_profile("changed", 512, "ROCKCHIP", "auto", **settings)
    with pytest.raises(ValueError, match="after memory admission"):
        checked_text_profile(checksum, 513, "ROCKCHIP", "auto", **settings)


def test_rwkv_profile_is_pinned_to_checked_identity_context_and_npu_mode():
    from onpu.rockchip_options import RWKV7_CHECKPOINT, TEXT_INT8_CONTEXTS

    layout = text_execution_profile(RWKV7_CHECKPOINT, 512, "ROCKCHIP", "auto")
    assert layout["cpu_recurrent_scan"] is True
    assert layout["prefill_chunk"] == 128
    assert layout["environment"] == {"ONPU_INT8_SMALL_DEPTH": "1", "ONPU_RWKV_DECODE_SPIN": "1",
                                      "ONPU_NPU_DOMAINS": "3", "ONPU_NPU_DOMAIN_BYTES": str(3 * 1024**3)}
    for checksum, context, profile, prefill in ((RWKV7_CHECKPOINT, 513, "ROCKCHIP", "auto"),
                                               (RWKV7_CHECKPOINT, 512, "CL", "auto"),
                                               (RWKV7_CHECKPOINT, 512, "ROCKCHIP", "off"),
                                               ("different", 512, "ROCKCHIP", "auto")):
        assert text_execution_profile(checksum, context, profile, prefill) == {}
    other = next(checksum for checksum in TEXT_INT8_CONTEXTS if checksum != RWKV7_CHECKPOINT)
    assert "environment" not in text_execution_profile(other, 512, "ROCKCHIP", "auto")


def test_recurrent_profile_requires_supported_scan_before_loading():
    import pytest

    require_text_backend(current_loader, current_configure, recurrent_scan=True)
    def no_scan(model, *, share_projection_weights=False):
        raise AssertionError("Do not execute the backend while checking it")
    with pytest.raises(ValueError, match="native CPU recurrent scans"):
        require_text_backend(current_loader, no_scan, recurrent_scan=True)
    options = text_projection_options(current_configure, cpu_npu=True, precision="int8", text_checkpoint=True,
                                      prefill="on", decode="on", recurrent_scan=True)
    assert current_configure(None, **options)["cpu_recurrent_scan"] is True


def test_qwen3_prefill_recipe_requires_exact_checkpoint_context_and_explicit_npu_decode():
    from onpu.rockchip_options import QWEN3_PREFILL_CHECKPOINT, qwen3_prefill_profile

    for prefill in ("auto", "on"):
        recipe = qwen3_prefill_profile(QWEN3_PREFILL_CHECKPOINT, 512, "ROCKCHIP", prefill, "on")
        assert recipe["prefill"] == "on" and recipe["prefill_chunk"] == 128
        assert recipe["tile_columns"] == 24384 and recipe["qwen3_fast_prefill"]
        assert recipe["environment"]["ONPU_SHARED_HOST_POOL"] == "1"
        assert recipe["environment"]["ONPU_REPLAY_BORROW_BATCH"] == "1"
        assert recipe["environment"]["ONPU_CPU_TOKEN_INPUTS"] == "1"
    baseline = [QWEN3_PREFILL_CHECKPOINT, 512, "ROCKCHIP", "auto", "on"]
    for index, values in ((0, ("other-checkpoint", None)), (1, (128, 511, 513, 40960)),
                          (2, ("CPU", "CL")), (3, ("off",)), (4, ("off", "auto"))):
        for value in values:
            args = baseline.copy()
            args[index] = value
            assert qwen3_prefill_profile(*args) == {}


def test_qwen3_attention_profile_checks_backend_and_uses_shared_host_pool():
    import pytest

    def attention_backend(model, *, share_projection_weights=False, cpu_prefill_pool=False, npu_prefill_attention=False):
        return locals()
    require_text_backend(current_loader, attention_backend, fast_prefill=True)
    with pytest.raises(ValueError, match="accelerated prefill"):
        require_text_backend(current_loader, current_configure, fast_prefill=True)
    options = text_projection_options(attention_backend, cpu_npu=True, precision="int8", text_checkpoint=True,
                                      prefill="on", decode="on", fast_prefill=True)
    assert options == {"share_projection_weights": True, "cpu_prefill_pool": False, "npu_prefill_attention": True}


def test_tinyllama_profile_selects_measured_layout_and_rejects_other_devices():
    from onpu.rockchip_options import TINYLLAMA_CHECKPOINT
    for prefill in ("auto", "on"):
        options = text_execution_profile(TINYLLAMA_CHECKPOINT, 512, "ROCKCHIP", prefill)
        assert options["dense_swiglu"] is True
        assert options["prefill_chunk"] == 128
        assert options["mapped_cpu"] is True and options["share_projection_weights"] is False
        assert options["environment"]["ONPU_SWIGLU_PREFILL_ROWS"] == "4"
        for key in ("ONPU_CPU_PREFILL_LAYOUT", "ONPU_SHARED_HOST_POOL", "ONPU_CPU_TOKEN_INPUTS",
                    "ONPU_ZERO_PAD_VIEWS", "ONPU_REPLAY_NATIVE_FROM"):
            assert options["environment"][key] == "1"
    for context, profile, prefill, digest in ((513, "ROCKCHIP", "auto", TINYLLAMA_CHECKPOINT),
                                             (512, "CL", "auto", TINYLLAMA_CHECKPOINT),
                                             (512, "ROCKCHIP", "off", TINYLLAMA_CHECKPOINT),
                                             (512, "ROCKCHIP", "auto", "different")):
        assert text_execution_profile(digest, context, profile, prefill) == {}


def test_tinyllama_requires_swiglu_support_and_selects_both_phases():
    import pytest
    with pytest.raises(ValueError, match="fused SwiGLU"):
        require_text_backend(current_loader, current_configure, swiglu=True)
    def configure(model, *, share_projection_weights=False, fused_swiglu_ffn=False, fused_swiglu_decode=False,
                  fused_qkv_without_norm=False):
        return locals()
    require_text_backend(current_loader, configure, swiglu=True)
    for decode in ("on", "off"):
        options = text_projection_options(configure, cpu_npu=True, precision="int8", text_checkpoint=True,
                                          prefill="on", decode=decode, swiglu=True)
        assert options == {"share_projection_weights": False, "fused_swiglu_ffn": True,
                           "fused_swiglu_decode": decode == "on", "fused_qkv_without_norm": True}


def test_qwen25_profile_pins_all_measured_phase_options_and_context():
    from onpu.rockchip_options import QWEN25_CHECKPOINT, TINYLLAMA_CHECKPOINT
    for prefill in ("auto", "on"):
        layout = text_execution_profile(QWEN25_CHECKPOINT, 512, "ROCKCHIP", prefill)
        assert layout["dense_swiglu"] and layout["prefill_profile"] == "qwen2.5-1.5b-context512"
        assert layout["tile_columns"] == 24576
        assert layout["environment"]["ONPU_SWIGLU_PREFILL_ROWS"] == "2"
        for key in ("ONPU_REPLAY_NATIVE_FROM_LARGE", "ONPU_CPU_DECODE_POOL", "ONPU_EXP2_FAST_PATH"):
            assert layout["environment"][key] == "1"
            assert key not in text_execution_profile(TINYLLAMA_CHECKPOINT, 512, "ROCKCHIP", prefill)["environment"]
    for context, profile, phase in ((513, "ROCKCHIP", "auto"), (512, "CL", "auto"), (512, "ROCKCHIP", "off")):
        assert not text_execution_profile(QWEN25_CHECKPOINT, context, profile, phase)


def test_dense_profile_does_not_prepare_unused_cpu_gemm_weights():
    options = text_projection_options(lambda **kwargs: kwargs, cpu_npu=True, precision="int8", text_checkpoint=True,
                                      prefill="on", decode="on", swiglu=True)
    assert options["cpu_static_gemm"] is False
    assert options["fused_gelu_ffn"] is False
    assert options["fused_swiglu_ffn"] and options["fused_swiglu_decode"]
