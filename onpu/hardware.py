from __future__ import annotations

import asyncio
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
            "npu_present": any("rknpu" in read(str(p / "uevent")) for p in Path("/sys/class/drm").glob("*")),
            "timestamp": time.time()}


async def backend_info(path: Path) -> dict:
    process = await asyncio.create_subprocess_exec("git", "-C", str(path), "rev-parse", "HEAD", stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.DEVNULL)
    out, _ = await process.communicate()
    return {"path": str(path), "revision": out.decode().strip() if process.returncode == 0 else None,
            "available": (path / "tinygrad/llm/model.py").is_file()}
