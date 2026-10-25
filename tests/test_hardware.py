import pytest

from onpu import hardware
from onpu.hardware import npu_present


@pytest.mark.parametrize("file_kib,private_clean_kib,private_dirty_kib,expected_kib", [
    (2 * 1024**2, 2 * 1024**2, 1024**2, 1024**2),
    (3 * 1024**2, 3 * 1024**2, 0, 0),
    (0, 0, 3 * 1024**2, 3 * 1024**2),
    (0, 0, 1024**2, 1024**2),
])
def test_reload_credit_excludes_file_cache_and_shared_ownership(monkeypatch, file_kib, private_clean_kib, private_dirty_kib, expected_kib):
    # Private_Dirty covers anonymous memory and shmem mapped only by this owner.
    # The last case leaves 2 GiB shared with another process, which cannot be
    # reclaimed by stopping this worker.
    anon_kib = min(1024**2, 3 * 1024**2 - file_kib)
    shmem_kib = 3 * 1024**2 - file_kib - anon_kib
    status = f"VmRSS: {3 * 1024**2} kB\nRssFile: {file_kib} kB\nRssAnon: {anon_kib} kB\nRssShmem: {shmem_kib} kB\n"
    rollup = f"Rss: {3 * 1024**2} kB\nPrivate_Clean: {private_clean_kib} kB\nPrivate_Dirty: {private_dirty_kib} kB\n"
    monkeypatch.setattr(hardware, "read", lambda path: status if path == "/proc/123/status" else rollup)
    assert hardware.reclaimable_rss(123) == expected_kib * 1024
    assert hardware.rss(123) == 3 * 1024**3


@pytest.mark.parametrize("status,rollup", [
    ("", ""),
    ("VmRSS: 10 kB", "Rss: 10 kB\nPrivate_Clean: 0 kB\nPrivate_Dirty: 10 kB"),
    ("VmRSS: nope kB\nRssFile: 0 kB", ""),
    ("VmRSS: 10 MB\nRssFile: 0 kB", ""),
    ("VmRSS: 10 kB\nRssFile: -1 kB", ""),
    ("VmRSS: 10 kB\nRssFile: 11 kB", "Rss: 10 kB\nPrivate_Clean: 0 kB\nPrivate_Dirty: 10 kB"),
    ("VmRSS: 10 kB\nRssFile: 0 kB", ""),
    ("VmRSS: 10 kB\nRssFile: 0 kB", "Rss: 10 kB\nPrivate_Clean: 0 kB"),
    ("VmRSS: 10 kB\nRssFile: 0 kB", "Rss: 10 kB\nPrivate_Clean: 0 kB\nPrivate_Dirty: invalid kB"),
    ("VmRSS: 10 kB\nRssFile: 0 kB", "Rss: 10 kB\nPrivate_Clean: 0 kB\nPrivate_Dirty: 11 kB"),
    ("VmRSS: 10 kB\nRssFile: 0 kB\nRssFile: 0 kB", ""),
])
def test_reload_credit_fails_closed_for_missing_or_malformed_proc_data(monkeypatch, status, rollup):
    monkeypatch.setattr(hardware, "read", lambda path: status if path.endswith("/status") else rollup)
    assert hardware.reclaimable_rss(123) == 0
    assert hardware.reclaimable_rss(None) == 0


def test_reload_credit_returns_zero_when_worker_proc_disappears(monkeypatch):
    def vanished(path):
        raise FileNotFoundError

    monkeypatch.setattr(hardware.Path, "read_text", vanished)
    assert hardware.reclaimable_rss(123) == 0


def test_npu_detection_reads_parent_driver_without_opening_device(tmp_path):
    card = tmp_path / "card1"
    device = card / "device"
    device.mkdir(parents=True)
    (card / "uevent").write_text("MAJOR=226\nMINOR=1\nDEVNAME=dri/card1\nDEVTYPE=drm_minor\n")
    (device / "uevent").write_text("DRIVER=rockchip-drm\n")
    assert not npu_present(tmp_path)
    (device / "uevent").write_text("DRIVER=RKNPU\nOF_COMPATIBLE_0=rockchip,rk3588-rknpu\n")
    assert npu_present(tmp_path)
    (device / "uevent").unlink()
    assert not npu_present(tmp_path)
    (card / "uevent").write_text("DRIVER=rknpu\n")
    assert npu_present(tmp_path)
