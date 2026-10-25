"""Optional backend integration checks; run with the configured tinygrad on PYTHONPATH."""
import base64
import struct
from types import SimpleNamespace

import pytest

from onpu.vlm import VisionSession


def test_image_prefix_cache_matches_full_recompute_and_invalidates_scene_changes(monkeypatch):
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from tinygrad.llm.model import TransformerConfig
    from tinygrad.llm.vision import SmolVLM, VisionConfig

    import onpu.vlm as serving

    Tensor.manual_seed(17)
    model = SmolVLM(VisionConfig(8, 16, 2, 1, 8, 2, 1e-6, 2),
                   TransformerConfig(1, 8, 16, 2, 1, 1e-5, 96, 4, 10000., 4, 4, max_context=256), 95)
    state = VisionSession(model, SimpleNamespace(is_end=lambda token: False))
    state.prepare_prefill()
    state.install()
    calls = []
    def encode(pixels):
        calls.append(1)
        return (Tensor.arange(32).float().reshape(1, 4, 8) / 32 + len(calls)).clone(device="CPU").realize()
    state.encoder = encode
    monkeypatch.setattr(serving, "decode_image", lambda url: bytes([128, 128, 128]) * 512 * 512)
    png_header = b"\x89PNG\r\n\x1a\n" + bytes(8) + struct.pack(">II", 1, 1)
    def messages(scene):
        url = "data:image/png;base64," + base64.b64encode(png_header + scene).decode()
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    prefix = [1, 2, 95, 95, 95, 95, 3]
    state.prepare(messages(b"A"), 3)
    initial = list(state.generate(prefix))
    assert 95 not in initial
    extended = prefix + initial + [4, 5]
    state.prepare(messages(b"A"), 3)
    assert model.text.get_start_pos(extended) >= len(prefix) - 1
    incremental = list(state.generate(extended))
    model.text._cached_tokens = []
    state.prepare(messages(b"A"), 3)
    assert list(state.generate(extended)) == incremental
    assert len(calls) == 1
    state.prepare(messages(b"B"), 3)
    assert model.text._cached_tokens == []
    changed = list(state.generate(prefix))
    model.text._cached_tokens = []
    state.prepare(messages(b"B"), 3)
    assert list(state.generate(prefix)) == changed
    assert len(calls) == 2
    state.prepare([{"role": "user", "content": "Text only"}], 3)
    assert model.text._cached_tokens == [] and not state.has_image
    list(state.generate([1, 2, 3]))
    state.prepare(messages(b"B"), 3)
    assert model.text._cached_tokens == []


def test_cancelled_image_prefill_retains_completed_prefix_and_retry_matches_full():
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from tinygrad.llm.model import TransformerConfig
    from tinygrad.llm.vision import SmolVLM, VisionConfig

    Tensor.manual_seed(23)
    model = SmolVLM(VisionConfig(8, 16, 2, 1, 8, 2, 1e-6, 2),
                   TransformerConfig(1, 8, 16, 2, 1, 1e-5, 96, 4, 10000., 4, 4, max_context=64), 95)
    state = VisionSession(model, SimpleNamespace(is_end=lambda token: False))
    state.prepare_prefill()
    state.prefill_chunk, state.limit = 4, 2
    state.has_image, state.image_key = True, "identified-image"
    state.pixels = Tensor.zeros(1, 3, 8, 8).realize()
    state.encoder = lambda pixels: (Tensor.arange(32).float().reshape(1, 4, 8) / 32).clone(device="CPU").realize()
    original = model.text.greedy_embeddings_prefill_jit
    calls, cancelled = 0, False
    state.cancelled = lambda: cancelled

    def interrupted(*args):
        nonlocal calls, cancelled
        calls += 1
        if calls == 3:
            cancelled = True
            raise InterruptedError("Stop between complete chunks")
        return original(*args)

    model.text.greedy_embeddings_prefill_jit = interrupted
    tokens = [1, 2, 95, 95, 95, 95, 3] + [4] * 20
    with pytest.raises(InterruptedError):
        list(state.generate(tokens))
    assert model.text._cached_tokens == tokens[:8]
    cancelled = False
    model.text.greedy_embeddings_prefill_jit = original
    incremental = list(state.generate(tokens))
    model.text._cached_tokens = []
    assert list(state.generate(tokens)) == incremental


def test_chunked_image_prefill_matches_full_prompt_with_replaced_history():
    np = pytest.importorskip("numpy")
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from tinygrad.llm.model import TransformerConfig
    from tinygrad.llm.vision import SmolVLM, VisionConfig

    Tensor.manual_seed(41)
    model = SmolVLM(VisionConfig(8, 16, 2, 1, 8, 2, 1e-6, 2),
                   TransformerConfig(1, 8, 16, 2, 1, 1e-5, 32, 4, 10000., 4, 4, max_context=256), 31)
    state = VisionSession(model, None)
    state.prepare_prefill()
    # Both full and partial chunks, and a new image prompt replacing longer KV history.
    for length in (82, 159, 105):
        values = Tensor.randn(1, length, 8).realize()
        expected = model.text.logits_from_embeddings(values, 0).numpy()
        actual = state.prefill(values).numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
        greedy = int(state.prefill(values, greedy=True).item())
        assert greedy == int(expected.argmax())


def test_image_sampling_reuses_warmed_text_decoder_variables():
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor, UOp
    from tinygrad.llm.model import TransformerConfig
    from tinygrad.llm.vision import SmolVLM, VisionConfig

    model = SmolVLM(VisionConfig(8, 16, 2, 1, 8, 2, 1e-6, 2),
                   TransformerConfig(1, 8, 16, 2, 1, 1e-5, 32, 4, 10000., 4, 4, max_context=32), 31)
    position = UOp.variable("start_pos", 0, 31)
    for index in range(3, 6):
        model.text.rollout_jit(Tensor([[5]]).realize(), position.bind(index), Tensor([.6]).realize()).realize()
    state = VisionSession(model, SimpleNamespace(is_end=lambda token: False))
    state.pixels = Tensor.zeros(1, 3, 8, 8).realize()
    state.encoder = lambda pixels: Tensor.randn(1, 4, 8).realize()
    state.limit = 2
    tokens = list(state.generate([1, 2, 31, 31, 31, 31, 3], temperature=.6))
    assert len(tokens) == 2 and all(0 <= token < 32 for token in tokens)


def test_adaptive_prefill_avoids_padding_short_prompts_and_keeps_long_chunks():
    np = pytest.importorskip("numpy")
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from tinygrad.llm.model import TransformerConfig
    from tinygrad.llm.vision import SmolVLM, VisionConfig

    model = SmolVLM(VisionConfig(8, 16, 2, 1, 8, 2, 1e-6, 2),
                   TransformerConfig(1, 8, 16, 2, 1, 1e-5, 32, 4, 10000., 4, 4, max_context=512), 31)
    state = VisionSession(model, None)
    state.prepare_prefill()
    state.prefill_buckets = (32, 64, 96, 128)
    seen = []

    def prefill(values, start, rows):
        seen.append(values.shape[1])
        return model.text.logits_from_embeddings(values[:, :rows], start).realize()

    model.text.embeddings_prefill_jit = prefill
    for length, capacities in ((82, [96]), (128, [128]), (159, [128, 32]), (333, [128, 128, 96])):
        seen.clear()
        values = Tensor.randn(1, length, 8).realize()
        expected = model.text.logits_from_embeddings(values, 0).numpy()
        actual = state.prefill(values).numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
        assert seen == capacities


def test_sampled_image_prefill_matches_full_logits_and_draws_only_once():
    np = pytest.importorskip("numpy")
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from tinygrad.llm.model import TransformerConfig
    from tinygrad.llm.vision import SmolVLM, VisionConfig

    Tensor.manual_seed(91)
    model = SmolVLM(VisionConfig(8, 16, 2, 1, 8, 2, 1e-6, 2),
                   TransformerConfig(1, 8, 16, 2, 1, 1e-5, 32, 4, 10000., 4, 4, max_context=512), 31)
    state = VisionSession(model, None)
    state.prepare_prefill()
    # Include JIT capture, replay, replacement of longer KV history and runtime temperatures.
    for length, temperature in ((82, .6), (159, 2.), (333, .05), (105, .6), (159, 1.2), (82, .6)):
        values = Tensor.randn(1, length, 8).realize()
        logits = model.text.logits_from_embeddings(values, 0).realize()
        counter = Tensor._device_rng_counters[values.device]
        before = counter.numpy().copy()
        expected = int((logits / max(temperature, 1e-12) - Tensor.rand_like(logits).maximum(1e-12).log().neg().log()).argmax(-1).item())
        after = counter.numpy().copy()
        counter.assign(Tensor(before, device=values.device)).realize()
        assert int(state.prefill(values, temperature=temperature).item()) == expected
        np.testing.assert_array_equal(counter.numpy(), after)
        assert not np.array_equal(before, after)
