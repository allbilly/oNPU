"""Bounded RK3588 stock-governor/IRQ experiment; always restore on normal exit/signals."""
import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path


def settings():
    governors = {f"/sys/devices/system/cpu/cpufreq/policy{i}/scaling_governor": "performance" for i in (0, 4, 6)}
    governors.update({f"/sys/class/devfreq/{dev}/governor": "performance" for dev in ("fdab0000.npu", "dmc")})
    irqs = {}
    for line in Path("/proc/interrupts").read_text().splitlines():
        if "fdab0000" in line:
            irq = line.split(":", 1)[0].strip()
            if irq.isdigit():
                irqs[f"/proc/irq/{irq}/smp_affinity_list"] = "6"
    if len(irqs) != 3:
        raise RuntimeError(f"Expected three fdab0000 NPU/IOMMU IRQs; found {len(irqs)}")
    return governors | irqs | {"/sys/devices/system/cpu/cpu6/cpuidle/state1/disable": "1"}


def snapshot(targets):
    result = {}
    for name, target in targets.items():
        path = Path(name)
        if name.endswith("scaling_governor"):
            available = path.with_name("scaling_available_governors")
        elif name.endswith("/governor"):
            available = path.with_name("available_governors")
        else:
            available = None
        if available is not None and target not in available.read_text().split():
            raise RuntimeError(f"Unsupported governor at {name}")
        result[name] = {"before": path.read_text().strip(), "target": target}
    return result


def transact(rows, write, read, hold, report):
    """Injectable IO lets the rollback contract be checked without hardware writes."""
    attempted = []
    try:
        for name, row in rows.items():
            attempted.append(name)
            write(name, row["target"])
            row["applied_readback"] = read(name)
            if row["applied_readback"] != row["target"]:
                raise RuntimeError(f"Setting readback mismatch: {name}")
        report("applied", [])
        hold()
    finally:
        errors = []
        for name in reversed(attempted):
            try:
                write(name, rows[name]["before"])
                rows[name]["restored_readback"] = read(name)
                if rows[name]["restored_readback"] != rows[name]["before"]:
                    raise RuntimeError("restoration readback mismatch")
            except Exception as exc:  # noqa: BLE001 - Attempt every remaining restore even if one fails.
                errors.append(f"{name}: {exc}")
        report("restore_failed" if errors else "restored", errors)
        if errors:
            raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="requires root; default is read-only preview")
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error("seconds must be between 1 and 1800")
    rows = snapshot(settings())
    if not args.apply:
        print(json.dumps(rows, indent=2))
        return
    if os.geteuid() != 0:
        parser.error("--apply requires sudo; do not send a password to the assistant")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    # /run is root-owned; exclusive receipt creation and flock prevent overlapping tuning owners.
    with open("/run/onpu-stock-tuning.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = snapshot(settings())
        receipt = Path(f"/run/onpu-stock-tuning-{os.getpid()}.json")
        with receipt.open("x") as output:
            os.chmod(receipt, 0o644)

            def report(phase, errors):
                record = {"phase": phase, "pid": os.getpid(), "time": time.time(), "seconds": args.seconds, "settings": rows, "errors": errors}
                output.seek(0)
                output.write(json.dumps(record, indent=2) + "\n")
                output.truncate()
                output.flush()
                print(json.dumps({"phase": phase, "receipt": str(receipt), "errors": errors}), flush=True)

            def hold():
                deadline = time.monotonic() + args.seconds
                while not stopped and time.monotonic() < deadline:
                    time.sleep(min(0.25, max(0, deadline - time.monotonic())))

            transact(rows, lambda p, v: Path(p).write_text(v), lambda p: Path(p).read_text().strip(), hold, report)


if __name__ == "__main__":
    main()
