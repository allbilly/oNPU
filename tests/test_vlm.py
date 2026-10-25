import base64
import struct
import sys
import zlib
from types import SimpleNamespace

import pytest

from onpu.vlm import VisionSession, decode_image, image_bytes, render_messages


def png_pixel():
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(bytes([0, 240, 20, 10])))
    data += chunk(b"IEND", b"")
    return data


def test_image_decode_uses_real_rgb_pixels():
    data = png_pixel()
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    assert image_bytes(url) == data
    pixels = decode_image(url)
    assert len(pixels) == 512 * 512 * 3
    assert pixels == bytes([240, 20, 10]) * (512 * 512)


@pytest.mark.parametrize("url", ["https://example.com/image.png", "data:image/png;base64,not-base64!", "data:image/png;base64,AAAA"])
def test_rejects_unsupported_image_inputs(url):
    with pytest.raises(ValueError):
        image_bytes(url)


def test_prompt_matches_native_single_image_template():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                               {"type": "text", "text": "What is shown?"}]}]
    text, image = render_messages(messages, 64)
    assert text == ("<|im_start|>User:<fake_token_around_image><global-img>" + "<image>" * 64
                    + "<fake_token_around_image>What is shown?<end_of_utterance>\nAssistant:")
    assert image == "data:image/png;base64,AAAA"
    with pytest.raises(ValueError, match="one image"):
        render_messages(messages + messages, 64)


def test_warmup_and_requests_share_image_transition_protocol(monkeypatch):
    events = []
    device = SimpleNamespace(reset=lambda: events.append("reset"), synchronize=lambda: events.append("sync"))
    monkeypatch.setitem(sys.modules, "tinygrad", SimpleNamespace(Device={"ROCKCHIP": device, "CPU": device}))
    image = SimpleNamespace(realize=lambda: events.append("encode") or image)
    state = VisionSession(SimpleNamespace(text=object(), encode_jit=lambda pixels: image), None)
    state.reset_vision = True
    assert state.encode(SimpleNamespace(device="CPU")) is image
    assert events == ["reset", "encode", "sync"]
    events.clear()
    state.reset_vision = False
    state.encode(SimpleNamespace(device="CPU"))
    assert events == ["encode", "sync"]


def test_rejected_image_decode_preserves_previous_request_cache(monkeypatch):
    import onpu.vlm as serving

    monkeypatch.setitem(sys.modules, "tinygrad", SimpleNamespace(Tensor=object()))
    model = SimpleNamespace(_cached_tokens=[1, 2, 3])
    state = VisionSession(SimpleNamespace(text=model, encode_jit=None, image_tokens=64), None)
    state.has_image = True
    state.image_key = state.last_image_key = state.cached_image_key = "previous-image"
    previous_image = state.cached_image = object()
    state.limit, state.timings = 8, {"vision_cached": True}
    url = "data:image/png;base64," + base64.b64encode(png_pixel()).decode()

    def fail_decode(url):
        raise ValueError("Image decoding failed")

    monkeypatch.setattr(serving, "decode_image", fail_decode)
    with pytest.raises(ValueError, match="Image decoding failed"):
        state.prepare([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}], 4)
    assert model._cached_tokens == [1, 2, 3]
    assert state.image_key == state.last_image_key == state.cached_image_key == "previous-image"
    assert state.cached_image is previous_image
    assert state.limit == 8 and state.timings == {"vision_cached": True}
