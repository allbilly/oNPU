"""Checkpoint-qualified execution policy, without importing an inference device."""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path

# Exact projection arithmetic at 256 held-out positions and synthetic 32K/40K
# cache boundaries. This does not qualify semantic answer quality at 40K.
QUALIFIED_MALI_CHECKPOINTS: dict[str, int] = {
    "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031": 40960,
}

# Values are (maximum checked context, global projection packing policy).
# These are local greedy-output/throughput checks, not RKLLM qualification.
QUALIFIED_MALI_RECURRENT_CHECKPOINTS: dict[str, tuple[int, bool]] = {
    "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c": (512, True),
    "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1": (512, True),
    "10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1": (512, False),
}


def _signature(stat: os.stat_result) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _read_checksum(path: Path, signature: tuple[int, ...]) -> str:
    with path.open("rb") as source:
        if _signature(os.fstat(source.fileno())) != signature:
            raise OSError("Checkpoint changed before checksum verification")
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        if _signature(os.fstat(source.fileno())) != signature or _signature(path.stat()) != signature:
            raise OSError("Checkpoint changed during checksum verification")
    return digest


_checksum_cache: OrderedDict[tuple[Path, tuple[int, ...]], str] = OrderedDict()
_checksum_lock = threading.Lock()


def checkpoint_checksum(path: Path, *, refresh: bool = True) -> str:
    """Fresh content verification by default; cached hashes are advisory only.

    File timestamps can collide even after a same-size rewrite. Admission and
    worker selection must therefore retain refresh=True. Only display estimates
    may explicitly reuse the bounded stat-keyed cache.
    """
    path = path.resolve(strict=True)
    signature = _signature(path.stat())
    key = path, signature
    if not refresh:
        with _checksum_lock:
            if key in _checksum_cache:
                _checksum_cache.move_to_end(key)
                return _checksum_cache[key]
    digest = _read_checksum(path, signature)
    with _checksum_lock:
        _checksum_cache[key] = digest
        _checksum_cache.move_to_end(key)
        while len(_checksum_cache) > 32:
            _checksum_cache.popitem(last=False)
    return digest


def qualified_mali_profile(path: Path, context: int, profile: str, npu_prefill: str, npu_decode: str, *,
                           realize: bool = True, checkpoints: dict[str, int] | None = None,
                           recurrent_checkpoints: dict[str, tuple[int, bool]] | None = None,
                           refresh: bool = True) -> tuple[str | None, dict]:
    """Return the verified checksum and its selected options from one read."""
    checkpoints = QUALIFIED_MALI_CHECKPOINTS if checkpoints is None else checkpoints
    recurrent_checkpoints = QUALIFIED_MALI_RECURRENT_CHECKPOINTS if recurrent_checkpoints is None else recurrent_checkpoints
    if (not realize or profile != "CL" or npu_prefill != "off" or npu_decode != "off" or path.suffix != ".gguf"
            or not (checkpoints or recurrent_checkpoints)):
        return None, {}
    checksum = checkpoint_checksum(path, refresh=refresh)
    if recurrent := recurrent_checkpoints.get(checksum):
        maximum, pack_weights = recurrent
        if 1 <= context <= maximum:
            return checksum, {"recurrent_scan": True, "pack_weights": pack_weights}
    if 1 <= context <= checkpoints.get(checksum, 0):
        return checksum, {"fuse_projections": True, "q8_head_checkpoint": path}
    return checksum, {}


def qualified_mali_options(path: Path, context: int, profile: str, npu_prefill: str, npu_decode: str, *,
                           realize: bool = True, checkpoints: dict[str, int] | None = None,
                           recurrent_checkpoints: dict[str, tuple[int, bool]] | None = None) -> dict:
    return qualified_mali_profile(path, context, profile, npu_prefill, npu_decode, realize=realize,
                                  checkpoints=checkpoints, recurrent_checkpoints=recurrent_checkpoints)[1]
