"""Single-image SmolVLM requests for the existing worker protocol."""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import subprocess
import time
from pathlib import Path


def image_bytes(url: str) -> bytes:
    prefix, separator, encoded = url.partition(",")
    if not separator or prefix not in ("data:image/png;base64", "data:image/jpeg;base64"):
        raise ValueError("Use an uploaded PNG or JPEG image")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Invalid image encoding") from exc
    if not data or len(data) > 1536 * 1024:
        raise ValueError("Image data exceeds 1.5 MiB")
    width = height = 0
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
    elif data.startswith(b"\xff\xd8"):
        position = 2
        while position + 4 <= len(data):
            if data[position] != 255:
                break
            while position < len(data) and data[position] == 255:
                position += 1
            if position >= len(data):
                break
            marker = data[position]
            position += 1
            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            if position + 2 > len(data):
                break
            length = int.from_bytes(data[position:position + 2], "big")
            if length < 2 or position + length > len(data):
                break
            if marker in (0xC0, 0xC1, 0xC2) and length >= 7:
                height, width = struct.unpack(">HH", data[position + 3:position + 7])
                break
            position += length
    if not width or not height or width * height > 16 * 1024**2:
        raise ValueError("Use a PNG or JPEG image of at most 16 megapixels")
    return data


def decode_image(url: str) -> bytes:
    data = image_bytes(url)
    try:
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-protocol_whitelist", "pipe",
                                 "-i", "pipe:0", "-frames:v", "1", "-vf", "scale=512:512:flags=lanczos",
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                input=data, capture_output=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Image decoding failed") from exc
    if result.returncode or len(result.stdout) != 512 * 512 * 3:
        raise ValueError("Image decoding failed")
    return result.stdout


def render_messages(messages: list[dict], image_tokens: int) -> tuple[str, str | None]:
    rendered, images = ["<|im_start|>"], []
    marker = "<fake_token_around_image><global-img>" + "<image>" * image_tokens + "<fake_token_around_image>"
    for message in messages:
        content = message.get("content") or ""
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
            raise ValueError("Invalid vision message content")
        image_first = bool(blocks and blocks[0].get("type") == "image_url")
        rendered.append(str(message["role"]).capitalize() + (":" if image_first else ": "))
        for block in blocks:
            if block.get("type") == "text":
                if not isinstance(block.get("text"), str):
                    raise ValueError("Message text must be a string")
                rendered.append(block["text"])
            elif block.get("type") == "image_url":
                image = block.get("image_url")
                url = image.get("url") if isinstance(image, dict) else None
                if message["role"] != "user" or not isinstance(url, str):
                    raise ValueError("Images must be supplied in a user message")
                images.append(url)
                rendered.append(marker)
            else:
                raise ValueError("This model accepts text and one uploaded image")
        rendered.append("<end_of_utterance>\n")
    if len(images) > 1:
        raise ValueError("Use one image per conversation; start a new chat for another image")
    rendered.append("Assistant:")
    return "".join(rendered), images[0] if images else None


def load_tokenizer(path: Path):
    from tinygrad.llm.cli import SimpleTokenizer
    data = json.loads((path / "tokenizer.json").read_text())
    special = {item["content"]: item["id"] for item in data["added_tokens"]}
    normal = {token: index for token, index in data["model"]["vocab"].items() if token not in special}
    eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
    return SimpleTokenizer(normal, special, preset="smollm", eos_id=eos)


class VisionSession:
    def __init__(self, vlm, tokenizer):
        self.vlm, self.model, self.tokenizer = vlm, vlm.text, tokenizer
        self.encoder = vlm.encode_jit
        self.pixels = None
        self.scalar_decoder = None
        self.check = lambda: None
        self.cancelled = lambda: False
        self.limit = 128
        self.original_generate = None
        self.reset_vision = False
        self.timings = {}
        self.prefill_chunk = None
        self.prefill_buckets = None
        self.has_image = False
        self.image_key = None
        self.last_image_key = None
        self.cached_image_key = None
        self.cached_image = None

    def prepare_prefill(self):
        from tinygrad import TinyJit
        model = self.model
        self.prefill_chunk = min(128, model.max_context)
        model.embeddings_prefill_jit = TinyJit(lambda values, start, rows: model.logits_from_embeddings(values[:, :rows], start).realize())
        model.greedy_embeddings_prefill_jit = TinyJit(lambda values, start, rows: model.greedy_from_embeddings(values[:, :rows], start).realize())
        model.sampled_embeddings_prefill_jit = TinyJit(
            lambda values, start, rows, temperature: model.sample_from_embeddings(values[:, :rows], start, temperature).realize())

    def prefill(self, embeddings, greedy=False, temperature=None, start_pos=0, tokens=None):
        prefill = self.model.greedy_embeddings_prefill_jit if greedy else self.model.embeddings_prefill_jit
        arguments = ()
        if temperature is not None:
            from tinygrad import Tensor
            prefill = self.model.sampled_embeddings_prefill_jit
            arguments = (Tensor([temperature], device=embeddings.device).realize(),)
        if self.prefill_chunk is None:
            result = prefill(embeddings, *arguments)
            if tokens is not None:
                self.model._cached_tokens = tokens.copy()
            return result
        from tinygrad import UOp
        start = UOp.variable("start_pos", 0, self.model.max_context - 1)
        logits = None
        for first in range(start_pos, embeddings.shape[1], self.prefill_chunk):
            self.check()
            count = min(self.prefill_chunk, embeddings.shape[1] - first)
            capacity = next((size for size in (self.prefill_buckets or (self.prefill_chunk,)) if size >= count), self.prefill_chunk)
            rows = UOp.variable("image_rows", 1, capacity)
            if embeddings.device == "CPU":
                from tinygrad.llm.kernels.vision_embed import stage_embeddings
                values = stage_embeddings(embeddings, first, count, capacity)
            else:
                values = embeddings[:, first:first + count].pad_to((1, capacity, embeddings.shape[2])).clone().realize()
            if temperature is not None and first + count < embeddings.shape[1]:
                # Populate intermediate KV history without consuming random draws.
                logits = self.model.greedy_embeddings_prefill_jit(values, start.bind(first), rows.bind(count))
            else:
                logits = prefill(values, start.bind(first), rows.bind(count), *arguments)
            if tokens is not None:
                self.model._cached_tokens = tokens[:first + count]
        return logits

    def prepare(self, messages: list[dict], limit: int) -> str:
        from tinygrad import Tensor
        rendered, url = render_messages(messages, self.vlm.image_tokens)
        image_key = hashlib.sha256(image_bytes(url)).hexdigest() if url is not None else None
        pixels = None
        if url is not None and self.cached_image_key != image_key:
            data = decode_image(url)
            device = self.vlm.vision.embeddings.patch_embedding.weight.device
            pixels = (Tensor(data, device=device).reshape(512, 512, 3).permute(2, 0, 1).unsqueeze(0).float() / 255 - .5) / .5
            pixels = pixels.contiguous().realize()
        # Rejected input must leave the previous request's reusable cache intact.
        if image_key != self.last_image_key:
            self.model._cached_tokens = []
        self.limit, self.timings, self.pixels = limit, {}, pixels
        self.has_image = url is not None
        self.image_key = self.last_image_key = image_key
        if pixels is not None:
            # Invalidate before encoding: a failed encoder may have touched
            # its reusable output storage, so old features cannot survive it.
            self.cached_image_key, self.cached_image = None, None
        return rendered

    def install(self):
        self.original_generate = self.model.generate
        self.model.generate = self.generate

    def encode(self, pixels):
        from tinygrad import Device
        if self.reset_vision:
            Device["ROCKCHIP"].reset()
        image = self.encoder(pixels).realize()
        Device[pixels.device].synchronize()
        return image

    def generate(self, tokens: list[int], chunk_size=None, temperature=0.0):
        if not self.has_image and self.pixels is None:
            kwargs = {"temperature": temperature}
            if chunk_size is not None:
                kwargs["chunk_size"] = chunk_size
            yield from self.original_generate(tokens, **kwargs)
            return
        from tinygrad import Tensor, UOp
        cache_start = self.model.get_start_pos(tokens) if self.prefill_chunk is not None and self.image_key is not None else 0
        self.model._cached_tokens = tokens[:cache_start]
        try:
            self.check()
            start = time.perf_counter()
            cached = self.image_key is not None and self.cached_image_key == self.image_key
            image = self.cached_image if cached else self.encode(self.pixels)
            if not cached:
                self.cached_image_key, self.cached_image = self.image_key, image
            encoded = time.perf_counter()
            self.timings["vision_ms"] = (encoded - start) * 1000
            self.timings["vision_cached"] = cached
            self.check()
            embeddings = self.vlm.embed_prompt(tokens, image).realize()
            token = int(self.prefill(embeddings, greedy=True, start_pos=cache_start, tokens=tokens).item() if temperature == 0 else
                        self.prefill(embeddings, temperature=temperature, start_pos=cache_start, tokens=tokens).item())
            self.timings["prefill_ms"] = (time.perf_counter() - encoded) * 1000
            cached_tokens = list(tokens)
            self.model._cached_tokens = cached_tokens.copy()
            yield token
            position = len(tokens)
            remaining = min(self.limit - 1, self.model.max_context - position)
            while remaining:
                self.check()
                if temperature == 0 and self.scalar_decoder is not None:
                    ids = self.scalar_decoder(token, position, min(8, remaining))
                else:
                    pos = UOp.variable("start_pos", 0, self.model.max_context - 1).bind(position)
                    source = Tensor([[token]], device=self.model.token_embd.weight.device).contiguous()
                    result = (self.model.greedy_rollout_jit(source, pos) if temperature == 0 else
                              self.model.rollout_jit(source, pos, Tensor([temperature])))
                    ids = [int(result.item())]
                for following in ids:
                    cached_tokens.append(token)
                    self.model._cached_tokens = cached_tokens.copy()
                    token = following
                    self.check()
                    yield token
                    if self.tokenizer.is_end(token):
                        return
                position += len(ids)
                remaining -= len(ids)
        except Exception:
            if not self.cancelled():
                self.model._cached_tokens = []
            raise
