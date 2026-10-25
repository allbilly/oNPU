from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import shutil
import time
from pathlib import Path


def read(path: str, default: str = "") -> str:
    try:
        return Path(path).read_text().strip().replace("\x00", " ")
    except (OSError, UnicodeError):
        return default


def memory() -> dict:
    items = {p[0].rstrip(":"): int(p[1]) * 1024 for line in read("/proc/meminfo").splitlines() if len(p := line.split()) >= 2}
    return {"total": items.get("MemTotal", 0), "available": items.get("MemAvailable", 0)}


def rss(pid: int | None) -> int:
    if pid:
        for line in read(f"/proc/{pid}/status").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def reclaimable_rss(pid: int | None) -> int:
    """Conservative reload credit, excluding file cache and memory shared by other processes."""
    if not pid:
        return 0

    def fields(path: str, names: tuple[str, ...]) -> dict[str, int]:
        values = {}
        for line in read(path).splitlines():
            parts = line.split()
            if parts and parts[0].endswith(":") and (key := parts[0][:-1]) in names:
                if key in values or len(parts) != 3 or parts[2] != "kB" or not parts[1].isdecimal():
                    raise ValueError
                values[key] = int(parts[1]) * 1024
        if len(values) != len(names):
            raise ValueError
        return values

    try:
        resident = fields(f"/proc/{pid}/status", ("VmRSS", "RssFile"))
        private = fields(f"/proc/{pid}/smaps_rollup", ("Rss", "Private_Clean", "Private_Dirty"))
        private_bytes = private["Private_Clean"] + private["Private_Dirty"]
        if resident["RssFile"] > resident["VmRSS"] or private_bytes > private["Rss"]:
            return 0
        # Private_* includes uniquely mapped shmem; shared mappings receive no
        # credit. Subtract all file RSS, even when some of it is shared, because
        # reclaimable file cache is already represented in MemAvailable.
        return max(0, min(resident["VmRSS"] - resident["RssFile"], private_bytes - resident["RssFile"]))
    except ValueError:
        return 0


def npu_present(drm: Path = Path("/sys/class/drm")) -> bool:
    return any("rknpu" in read(str(p / suffix)).lower()
               for p in drm.glob("*") for suffix in ("uevent", "device/uevent"))


def snapshot(data: Path) -> dict:
    temps = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            value = float(read(str(zone / "temp"))) / 1000
            temps.append({"name": read(str(zone / "type")), "celsius": value})
        except ValueError:
            pass
    cpus = []
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        freq = read(str(cpu / "cpufreq/cpuinfo_max_freq"))
        cpus.append({"id": int(cpu.name[3:]), "max_mhz": int(freq) / 1000 if freq else None})
    disk = shutil.disk_usage(data)
    return {"board": read("/proc/device-tree/model", platform.node()), "soc": read("/proc/device-tree/compatible"),
            "arch": platform.machine(), "kernel": platform.release(), "memory": memory(), "thermal_zones": temps,
            "temperature": max((t["celsius"] for t in temps), default=None), "cpus": cpus,
            "load": list(os.getloadavg()), "disk": {"total": disk.total, "free": disk.free},
            "npu_present": npu_present(),
            "timestamp": time.time()}


async def backend_info(path: Path) -> dict:
    process = await asyncio.create_subprocess_exec("git", "-C", str(path), "rev-parse", "HEAD", stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await process.communicate()
    except BaseException:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await asyncio.shield(process.communicate())
        raise
    return {"path": str(path), "revision": out.decode().strip() if process.returncode == 0 else None,
            "available": (path / "tinygrad/llm/model.py").is_file()}
