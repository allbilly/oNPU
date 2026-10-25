from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shutil
import struct
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import httpx

from .execution import qualified_mali_profile
from .speculation import draft_memory, measured_pair_memory
from .store import Store

CATALOG = [
    {"id": "qwen3-0.6b-q8", "name": "Qwen3 · 0.6B", "repo": "Qwen/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q8_0.gguf",
     "quantization": "Q8_0", "parameters": "0.6B", "description": "A small starting point for this board. Supports thinking and regular chat."},
    {"id": "qwen3-1.7b-q4", "name": "Qwen3 · 1.7B", "repo": "unsloth/Qwen3-1.7B-GGUF", "file": "Qwen3-1.7B-Q4_K_M.gguf",
     "quantization": "Q4_K_M", "parameters": "1.7B", "description": "More capable chat with a compact download. Runtime weights may expand to FP16."},
    {"id": "llama3.2-1b-q4", "name": "Llama 3.2 · 1B", "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF", "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "quantization": "Q4_K_M", "parameters": "1B", "description": "An alternative dense transformer for short conversations."},
]

QUANTS = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
          12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
SIMPLE = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# GGUF block sizes for the common formats. Unknown types retain a conservative
# lower bound and are finally accepted/rejected by tinygrad's own loader.
BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
          10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2)}


def inspect_gguf(path: Path) -> dict:
    """Read bounded metadata and tensor shapes without realizing any model weights."""
    size = path.stat().st_size
    with path.open("rb") as f:
        def unpack(fmt: str):
            count = struct.calcsize("<" + fmt)
            data = f.read(count)
            if len(data) != count:
                raise ValueError("Truncated GGUF metadata")
            return struct.unpack("<" + fmt, data)[0]

        def string(keep=True):
            count = unpack("Q")
            if count > 16 * 1024 * 1024 or f.tell() + count > size:
                raise ValueError("Invalid GGUF string length")
            if keep:
                return f.read(count).decode("utf-8", errors="replace")
            f.seek(count, 1)

        def value(kind: int, keep: bool, depth=0):
            if depth > 2:
                raise ValueError("Invalid nested GGUF metadata")
            if kind in SIMPLE:
                return unpack(SIMPLE[kind])
            if kind == 8:
                return string(keep)
            if kind == 9:
                subtype, count = unpack("I"), unpack("Q")
                if count > 2_000_000:
                    raise ValueError("GGUF metadata array is too large")
                if subtype in SIMPLE:
                    length = count * struct.calcsize("<" + SIMPLE[subtype])
                    if f.tell() + length > size:
                        raise ValueError("Truncated GGUF array")
                    f.seek(length, 1)
                else:
                    for _ in range(count):
                        value(subtype, False, depth + 1)
                return None
            raise ValueError(f"Unknown GGUF metadata type {kind}")

        if f.read(4) != b"GGUF" or unpack("I") not in (2, 3):
            raise ValueError("Select a GGUF version 2 or 3 model")
        tensors, keys = unpack("Q"), unpack("Q")
        if tensors > 100_000 or keys > 100_000 or tensors == 0:
            raise ValueError("Invalid GGUF tensor or metadata count")
        metadata = {}
        for _ in range(keys):
            key, kind = string(), unpack("I")
            suffixes = ("context_length", "block_count", "embedding_length", "head_count_kv", "head_count", "key_length")
            keep = key.startswith(("general.", "split.")) or key.endswith(suffixes)
            result = value(kind, keep)
            if keep and result is not None:
                metadata[key] = result
        if metadata.get("split.count", 1) > 1:
            raise ValueError("Split GGUF files are not supported; choose a single-file model")
        count, types, required_end = 0, Counter(), 0
        for _ in range(tensors):
            string(False)
            dims = unpack("I")
            if not 1 <= dims <= 4:
                raise ValueError("Invalid GGUF tensor dimensions")
            elements = math.prod(unpack("Q") for _ in range(dims))
            if elements < 1 or elements > 10**12:
                raise ValueError("Invalid GGUF tensor shape")
            kind, offset = unpack("I"), unpack("Q")
            if offset >= size:
                raise ValueError("GGUF tensor offset exceeds file size")
            count += elements
            types[kind] += elements
            block, block_bytes = BLOCKS.get(kind, (8, 1))
            required_end = max(required_end, offset + ((elements + block - 1) // block) * block_bytes)
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment > 65536 or alignment & (alignment - 1):
            raise ValueError("Invalid GGUF data alignment")
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        if data_start >= size:
            raise ValueError("GGUF has no weight data")
        if data_start + required_end > size:
            raise ValueError("Truncated GGUF weight data")
    arch = metadata.get("general.architecture", "unknown")
    return {"name": metadata.get("general.name") or path.stem, "architecture": arch, "size": size, "parameters": count,
            "quantization": QUANTS.get(types.most_common(1)[0][0], "Mixed"), "context_limit": metadata.get(f"{arch}.context_length", 4096),
            "layers": metadata.get(f"{arch}.block_count", 0), "kv_heads": metadata.get(f"{arch}.attention.head_count_kv", 0),
            "head_dim": metadata.get(f"{arch}.attention.key_length", metadata.get(f"{arch}.embedding_length", 0) //
                                     max(1, metadata.get(f"{arch}.attention.head_count", 1)))}


HF_FILES = ("config.json", "model.safetensors", "tokenizer.json", "generation_config.json")


def model_present(model: dict) -> bool:
    path = Path(model["path"])
    return all((path / name).is_file() for name in HF_FILES) if model.get("format") == "HF" else path.is_file()


def inspect_hf(path: Path) -> dict:
    """Read a single-file SmolVLM checkpoint's metadata without loading weights."""
    if not all((path / name).is_file() for name in HF_FILES):
        raise ValueError("The model directory needs config.json, model.safetensors, tokenizer.json and generation_config.json")
    if (path / "config.json").stat().st_size > 1024**2:
        raise ValueError("Model configuration is too large")
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("Invalid vision model configuration")  # noqa: TRY004 - invalid file data maps to HTTP 400
    if config.get("model_type") != "idefics3":
        raise ValueError("Only SmolVLM-v1 Hugging Face directories are supported")
    text, vision = config.get("text_config"), config.get("vision_config")
    for section, fields in ((text, ("max_position_embeddings", "num_hidden_layers", "num_key_value_heads", "head_dim")),
                            (vision, ("image_size", "patch_size")), (config, ("scale_factor",))):
        if not isinstance(section, dict) or any(type(section.get(field)) is not int or section[field] < 1 for field in fields):
            raise ValueError("Invalid vision model configuration: dimensions must be positive integers")
    image_size, patch, scale = vision["image_size"], vision["patch_size"], config["scale_factor"]
    if image_size % (patch * scale):
        raise ValueError("Invalid vision model configuration: image size must be divisible by patch size and scale factor")
    weights = path / "model.safetensors"
    size = weights.stat().st_size
    with weights.open("rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= 32 * 1024**2 or length + 8 > size:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(file.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")  # noqa: TRY004 - invalid file data maps to HTTP 400
    parameters, types = 0, set()
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError("Invalid safetensors tensor metadata")  # noqa: TRY004 - invalid file data maps to HTTP 400
        shape, dtype = tensor.get("shape"), tensor.get("dtype")
        if (dtype not in ("F16", "BF16", "F32") or not isinstance(shape, list) or not 1 <= len(shape) <= 4
                or any(type(n) is not int or n < 1 for n in shape)):
            raise ValueError("Unsupported vision checkpoint tensor")
        count = math.prod(shape)
        offsets = tensor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError("Invalid safetensors tensor offsets")
        first, end = offsets
        if first < 0 or end - first != count * (4 if dtype == "F32" else 2) or end > size - length - 8:
            raise ValueError("Truncated safetensors weight data")
        parameters += count
        types.add(dtype)
    if not parameters:
        raise ValueError("The checkpoint has no weights")
    return {"name": "SmolVLM (Idefics3)", "architecture": "idefics3", "format": "HF", "supports_images": True,
            "size": sum((path / name).stat().st_size for name in HF_FILES), "parameters": parameters,
            "quantization": "/".join(sorted(types)), "context_limit": text["max_position_embeddings"],
            "layers": text["num_hidden_layers"], "kv_heads": text["num_key_value_heads"], "head_dim": text["head_dim"],
            "vision_resolution": image_size, "image_tokens": (image_size // patch // scale)**2}


def register(store: Store, path: Path, managed=False) -> dict:
    path = path.expanduser().resolve(strict=True)
    if path.is_dir():
        if managed:
            raise ValueError("Vision model directories are imported as references")
        model = inspect_hf(path)
    elif path.is_file() and path.suffix.lower() == ".gguf":
        model = inspect_gguf(path)
    else:
        raise ValueError("Select a local .gguf file or a supported Hugging Face model directory")
    key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    previous = store.get("models", key) or {}
    return store.put("models", {**previous, **model, "id": key, "path": str(path), "managed": managed or previous.get("managed", False),
                                 "imported_at": previous.get("imported_at", time.time())})


# The complete 4B GPU run peaks at 9830.328125 MiB with all packing disabled.
# See docs/benchmark-recurrent-family.json. Reserve another 1536 MiB for startup,
# additional warm graph variants and workspace. This measured total includes both
# attention KV and FP32 GDN/conv state; it is not extrapolated to other contexts,
# checkpoints or the smaller profiles that still pack some projection weights.
_UNPACKED_MALI_MEMORY = {
    "10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1":
        (4482403488, 4205751296, 9831 * 1024**2 + 1536 * 1024**2),
}


def estimated_memory(model: dict, context: int, realize: bool, options: dict | None = None, *, refresh: bool = True) -> int:
    if (pair := measured_pair_memory(model, context, realize, options, refresh=refresh)) is not None:
        # This whole-worker peak already includes the draft; do not reserve it twice.
        return pair
    kv_bytes = 2 if model.get("architecture") in ("qwen3", "llama", "idefics3") else 4
    kv = 2 * model.get("layers", 0) * model.get("kv_heads", 0) * model.get("head_dim", 0) * context * kv_bytes
    # Source, execution and prepared weight layouts, FP16 standard-transformer KV,
    # conservative cache storage for other architectures, and bounded workspace.
    draft = draft_memory(context, options)
    conservative = model["size"] + model.get("parameters", 0) * (4 if realize else 2) + kv + 512 * 1024**2 + draft
    if (options and model.get("architecture") == "qwen35" and model.get("path")
            and any((model["size"], model.get("parameters")) == record[:2] for record in _UNPACKED_MALI_MEMORY.values())):
        path = Path(model["path"])
        try:
            checksum, layout = qualified_mali_profile(path, context, options.get("profile", ""), options.get("npu_prefill", "off"),
                                                     options.get("npu_decode", "off"), realize=realize, refresh=refresh)
            if checksum and layout.get("recurrent_scan") and layout.get("pack_weights") is False:
                measured = _UNPACKED_MALI_MEMORY.get(checksum)
                if measured and (model["size"], model.get("parameters")) == measured[:2]:
                    return measured[2] + draft
        except OSError:
            # Missing or concurrently modified files must not get a smaller
            # budget based on a stored or stale checkpoint checksum.
            pass
    return conservative


class Downloads:
    def __init__(self, store: Store):
        self.store = store
        self.tasks: dict[str, asyncio.Task] = {}
        for job in store.list("downloads"):
            if job["state"] in ("downloading", "verifying", "queued"):
                store.put("downloads", {**job, "state": "paused", "error": "Service restarted. Resume to continue."})

    def start(self, spec: dict, job_id: str | None = None) -> dict:
        if not spec["file"].lower().endswith(".gguf") or Path(spec["file"]).name != spec["file"] or "/" not in spec["repo"]:
            raise ValueError("Choose a single GGUF file in a Hugging Face repository")
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", spec["file"]):
            raise ValueError("Split GGUF files are not supported; choose a single-file model")
        if len([t for t in self.tasks.values() if not t.done()]) >= 2:
            raise ValueError("Two downloads are already active")
        job = {"id": job_id or uuid.uuid4().hex[:16], "name": spec.get("name", spec["file"]), "repo": spec["repo"], "file": spec["file"],
               "revision": spec.get("revision", "main"), "state": "queued", "received": 0, "total": None, "error": None}
        self.store.put("downloads", job)
        self.tasks[job["id"]] = asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: dict):
        root = self.store.root / "models"
        target = root / (job["id"] + "-" + job["file"])
        partial = target.with_suffix(".gguf.part")
        offset = partial.stat().st_size if partial.exists() else 0
        job.update(state="downloading", received=offset)
        self.store.put("downloads", job)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30, read=60), trust_env=False) as client:
                # Resolve a mutable branch once; resumes use the immutable snapshot.
                revision_url = f"https://huggingface.co/api/models/{quote(job['repo'], safe='/')}/revision/{quote(job['revision'], safe='')}"
                info = await client.get(revision_url)
                info.raise_for_status()
                manifest = info.json()
                job["revision"] = manifest["sha"]
                url = f"https://huggingface.co/{quote(job['repo'], safe='/')}/resolve/{job['revision']}/{quote(job['file'], safe='')}"
                async with client.stream("GET", url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"}) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        offset = 0
                    elif not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Download server returned an unexpected range")
                    job["total"] = offset + int(response.headers["content-length"]) if response.headers.get("content-length") else None
                    if job["total"] and shutil.disk_usage(root).free < job["total"] - offset + 128 * 1024**2:
                        raise ValueError("Not enough free storage for this model")
                    updated = 0.0
                    with partial.open("ab" if offset else "wb") as f:
                        job["received"] = offset
                        async for chunk in response.aiter_bytes(256 * 1024):
                            f.write(chunk)
                            job["received"] += len(chunk)
                            if time.monotonic() - updated > 0.5:
                                self.store.put("downloads", job)
                                updated = time.monotonic()
                    if job["total"] is not None and job["received"] != job["total"]:
                        raise ValueError("Download is incomplete")
            job["state"] = "verifying"
            self.store.put("downloads", job)
            await asyncio.to_thread(inspect_gguf, partial)
            digest = await asyncio.to_thread(hash_file, partial)
            # A local digest is retained; it is not advertised as a remote checksum match.
            partial.replace(target)
            model = await asyncio.to_thread(register, self.store, target, True)
            self.store.put("models", {**model, "repo": job["repo"], "revision": job["revision"], "sha256": digest})
            job.update(state="complete", model_id=model["id"], sha256=digest)
        except asyncio.CancelledError:
            job.update(state="paused", error=None)
        except Exception as exc:  # noqa: BLE001 - persist any background download failure for the client
            job.update(state="error", error=str(exc))
        finally:
            self.store.put("downloads", job)

    async def cancel(self, key: str):
        if (task := self.tasks.get(key)) and not task.done():
            task.cancel()
            await task

    async def close(self):
        for key in list(self.tasks):
            await self.cancel(key)


def hash_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
