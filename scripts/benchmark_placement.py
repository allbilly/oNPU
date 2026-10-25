"""Compare GPU-only and measured GPU/NPU placement through real chat requests."""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    rows = []
    report = {"complete": False, "rows": rows}

    def save():
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    save()
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=300, trust_env=False) as client:
            async def status():
                response = await client.get("/api/status")
                response.raise_for_status()
                return next(s for s in response.json()["runtime"]["sessions"] if s["model_id"] == args.model)

            for mode in ("gpu", "auto", "gpu"):
                response = await client.post(f"/api/models/{args.model}/load", json={
                    "profile": "CL", "context": 512, "realize": True,
                    "npu_prefill": "auto" if mode == "auto" else "off",
                    "npu_decode": "auto" if mode == "auto" else "off"})
                assert response.status_code == 202, response.text
                end = time.monotonic() + 300
                while time.monotonic() < end:
                    state = await status()
                    assert state["state"] != "error", state
                    if state["state"] == "ready":
                        break
                    await asyncio.sleep(.5)
                assert state["state"] == "ready", state
                print("ready", mode, state["actual_device"], flush=True)
                trials = []
                prompts = ["Write every number from one to ten in words.", "Explain why the sky looks blue.",
                           "Explain how to grow a tomato plant."]
                for prompt in prompts:
                    response = await client.post("/v1/chat/completions", json={
                        "model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 16})
                    assert response.status_code == 200, response.text
                    body = response.json()
                    assert body["choices"][0]["message"]["content"], body
                    state = await status()
                    trials.append({"prompt": prompt, "response": body, "metrics": state["last_metrics"]})
                    print(mode, state["last_metrics"]["decode_tps"], flush=True)
                rows.append({"mode": mode, "device": state["actual_device"], "placement": state["placement"], "trials": trials})
                save()
        summary = [{"mode": row["mode"], "median_decode_tps": statistics.median(t["metrics"]["decode_tps"] for t in row["trials"]),
                    "median_ttft_ms": statistics.median(t["metrics"]["ttft_ms"] for t in row["trials"])} for row in rows]
        responses = [[t["response"]["choices"][0]["message"]["content"] for t in row["trials"]] for row in rows]
        matches = [texts == responses[0] for texts in responses]
        report.update(summary=summary, response_matches_gpu_baseline=matches)
        assert all(matches), "Placement changed a deterministic benchmark response"
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
