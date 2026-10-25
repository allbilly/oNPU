import pytest

from onpu.worker import prefill_chunk, prefill_chunks


def test_prefill_chunks_fit_context_and_keep_single_token_decode_small():
    assert prefill_chunks(512) == [32, 64, 128, 256]
    assert prefill_chunks(96) == [32, 64]
    assert prefill_chunks(16) == [16]
    for remaining, expected in [(1, 1), (17, 32), (33, 64), (65, 128), (129, 256), (400, 256)]:
        assert prefill_chunk(remaining, 512) == expected
    assert prefill_chunk(80, 96) == 64


def test_int8_selection_requires_qualified_checkpoint_and_npu_profile(tmp_path, monkeypatch):
    import hashlib

    from onpu import worker

    path = tmp_path / "model.gguf"
    path.write_bytes(b"qualified fixture")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "QUALIFIED_INT8_CHECKPOINTS", frozenset({checksum}))
    assert worker.npu_precision(path, "ROCKCHIP", 512) == "int8"
    assert worker.npu_precision(path, "ROCKCHIP", 1024) == "fp16"
    assert worker.npu_precision(path, "CL", 512) == "fp16"
    assert worker.npu_precision(path, "CPU", 512) == "fp16"
    path.write_bytes(b"different weights")
    assert worker.npu_precision(path, "ROCKCHIP", 512) == "fp16"


def test_int8_context_extension_is_specific_to_the_qualified_checkpoint(tmp_path, monkeypatch):
    import hashlib

    from onpu import worker

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"long-context qualified fixture")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "QUALIFIED_INT8_CHECKPOINTS", frozenset({checksum}))
    monkeypatch.setattr(worker, "QUALIFIED_INT8_CONTEXTS", {checksum: 8192})
    assert worker.npu_precision(path, "ROCKCHIP", 8192) == "int8"
    assert worker.npu_precision(path, "ROCKCHIP", 8193) == "fp16"
    assert worker.npu_precision(path, "CL", 8192) == "fp16"
    monkeypatch.setattr(worker, "QUALIFIED_INT8_CONTEXTS", {})
    assert worker.npu_precision(path, "ROCKCHIP", 1024) == "fp16"


def test_vision_gpu_requires_qualified_precision_opt_in_and_driver_support(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace

    from onpu import worker
    from onpu.worker import vision_gpu_enabled

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"vision checkpoint fixture")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "QUALIFIED_GPU_VISION_CHECKPOINTS", frozenset({checksum}))
    gpu = SimpleNamespace(device_exts=["cl_arm_import_memory_dma_buf", "cl_khr_fp16"])
    monkeypatch.setitem(sys.modules, "tinygrad", SimpleNamespace(Device={"CL": gpu}))
    assert vision_gpu_enabled(path, "int8", True)
    assert not vision_gpu_enabled(path, "fp16", True)
    assert not vision_gpu_enabled(path, "int8", False)
    gpu.device_exts = ["cl_khr_fp16"]
    assert not vision_gpu_enabled(path, "int8", True)
    gpu.device_exts = ["cl_arm_import_memory_dma_buf", "cl_khr_fp16"]
    path.write_bytes(b"other checkpoint")
    assert not vision_gpu_enabled(path, "int8", True)


@pytest.mark.parametrize("error", [RuntimeError("No GPU"), ModuleNotFoundError("No CL backend"), IndexError("No CL devices")])
def test_vision_gpu_unavailable_backend_falls_back(tmp_path, monkeypatch, error):
    import hashlib
    import sys
    from types import SimpleNamespace

    from onpu import worker

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"qualified vision weights")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "QUALIFIED_GPU_VISION_CHECKPOINTS", frozenset({checksum}))

    class UnavailableGPU:
        def __getitem__(self, device):
            raise error

    monkeypatch.setitem(sys.modules, "tinygrad", SimpleNamespace(Device=UnavailableGPU()))
    assert not worker.vision_gpu_enabled(path, "int8", True)


@pytest.mark.parametrize("content", ["Text only", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]])
def test_vision_budget_rejection_precedes_cache_changes(content):
    from types import SimpleNamespace

    from onpu.worker import prepare_prompt

    model = SimpleNamespace(max_context=64, _cached_tokens=[1, 2, 3])

    def prepare(messages, limit):
        model._cached_tokens = []
        pytest.fail("Rejected prompt must not prepare an image or invalidate caches")

    session = SimpleNamespace(vlm=SimpleNamespace(image_tokens=64), prepare=prepare)
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 8}
    with pytest.raises(ValueError, match="exceeds context"):
        prepare_prompt(body, model, SimpleNamespace(encode=lambda rendered: [1] * 60), None, session)
    assert model._cached_tokens == [1, 2, 3]


def test_vision_prompt_at_context_boundary_prepares_request():
    from types import SimpleNamespace

    from onpu.worker import prepare_prompt

    prepared = []
    session = SimpleNamespace(vlm=SimpleNamespace(image_tokens=64), prepare=lambda *args: prepared.append(args))
    body = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8}
    rendered, ids = prepare_prompt(body, SimpleNamespace(max_context=64),
                                   SimpleNamespace(encode=lambda text: [1] * 56), None, session)
    assert rendered == "<|im_start|>User: Hello<end_of_utterance>\nAssistant:"
    assert len(ids) == 56
    assert prepared == [(body["messages"], 8)]


@pytest.mark.parametrize("thinking_options", [{}, {"enable_thinking": False}, {"enable_thinking": True}])
def test_fallback_prompt_accepts_worker_thinking_options_and_keeps_context_bounds(thinking_options):
    from types import SimpleNamespace

    from tinygrad.llm.cli import FallbackTemplate

    from onpu.worker import prepare_prompt

    tok = SimpleNamespace(preset="qwen2", bos_id=None, eos_id=0,
                          decode=lambda ids: "".join({0: "<|im_end|>"}[token] for token in ids),
                          encode=lambda text: [ord(char) for char in text])
    template = FallbackTemplate(tok)
    body = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8, **thinking_options}
    expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    expected_ids = [ord(char) for char in expected]
    model = SimpleNamespace(max_context=len(expected_ids) + 8, _cached_tokens=[1, 2, 3])
    assert template.render(body["messages"]) == expected
    assert prepare_prompt(body, model, tok, template) == (expected, expected_ids)
    assert model._cached_tokens == [1, 2, 3]
    model.max_context -= 1
    with pytest.raises(ValueError, match="exceeds context"):
        prepare_prompt(body, model, tok, template)
    assert model._cached_tokens == [1, 2, 3]
