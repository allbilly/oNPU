from __future__ import annotations

import struct
from pathlib import Path

import pytest


def write_gguf(path: Path):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    metadata = [("general.name", 8, string("Test model")), ("general.architecture", 8, string("qwen3")),
                ("qwen3.context_length", 4, struct.pack("<I", 1024))]
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, kind, value in metadata:
        header += string(key) + struct.pack("<I", kind) + value
    header += string("weight") + struct.pack("<IQIQ", 1, 4, 1, 0)
    path.write_bytes(header + bytes((-len(header)) % 32) + bytes(8))
    return path


@pytest.fixture
def gguf(tmp_path):
    return write_gguf(tmp_path / "test.gguf")
