import hashlib

from onpu import worker


def test_gpu_improvements_require_qualified_weights_capacity_and_assistance_off(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"qualified test weights")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "QUALIFIED_MALI_CHECKPOINTS", {checksum: 40960})
    assert worker.mali_options(path, 40960, "CL", "off", "off") == {"fuse_projections": True, "q8_head_checkpoint": path}
    assert not worker.mali_options(path, 512, "CL", "off", "off", realize=False)
    for context, profile, prefill, decode in ((40961, "CL", "off", "off"), (0, "CL", "off", "off"),
                                             (512, "CPU", "off", "off"), (512, "CL", "auto", "off"),
                                             (512, "CL", "off", "on")):
        assert not worker.mali_options(path, context, profile, prefill, decode)
    path.write_bytes(b"other weights")
    assert not worker.mali_options(path, 512, "CL", "off", "off")


def test_recurrent_options_require_exact_weights_checked_context_and_gpu_only(tmp_path, monkeypatch):
    path = tmp_path / "recurrent.gguf"
    path.write_bytes(b"recurrent test weights")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "QUALIFIED_MALI_CHECKPOINTS", {})
    for pack_weights in (False, True):
        monkeypatch.setattr(worker, "QUALIFIED_MALI_RECURRENT_CHECKPOINTS", {checksum: (512, pack_weights)})
        assert worker.mali_options(path, 512, "CL", "off", "off") == {
            "recurrent_scan": True, "pack_weights": pack_weights,
        }
        assert not worker.mali_options(path, 512, "CL", "off", "off", realize=False)
        for context, profile, prefill, decode in ((513, "CL", "off", "off"), (0, "CL", "off", "off"),
                                                (512, "CPU", "off", "off"), (512, "CL", "auto", "off"),
                                                (512, "CL", "off", "on")):
            assert not worker.mali_options(path, context, profile, prefill, decode)
    path.write_bytes(b"unqualified recurrent weights")
    assert not worker.mali_options(path, 512, "CL", "off", "off")
