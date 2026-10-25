"""Verify real, independent GPU and NPU model requests through oNPU's HTTP API."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--npu", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    async with httpx.AsyncClient(base_url=args.url, timeout=300, trust_env=False) as client:
        async def sessions():
            response = await client.get("/api/status")
            response.raise_for_status()
            data = response.json()
            assert data["backend"]["path"] == "/home/orangepi/tinygrad", data["backend"]
            return {s["model_id"]: s for s in data["runtime"]["sessions"]}

        for key, profile in [(args.gpu, "CL"), (args.npu, "ROCKCHIP")]:
            states = await sessions()
            existing = states.get(key, {})
            if existing.get("selected_profile") != profile or existing.get("context") != 512:
                response = await client.post(f"/api/models/{key}/load", json={"profile": profile, "context": 512})
                assert response.status_code == 202, response.text
            end = time.monotonic() + 300
            while time.monotonic() < end:
                state = (await sessions())[key]
                assert state["state"] != "error", state
                if state["state"] == "ready":
                    break
                await asyncio.sleep(.5)
            assert state["state"] == "ready", state
            print("ready", key, state["actual_device"], flush=True)

        async def generate(key):
            start = time.monotonic()
            response = await client.post("/v1/chat/completions", json={
                "model": key, "messages": [{"role": "user", "content": "Explain in one sentence why the sky is blue."}],
                "temperature": 0, "max_tokens": 24})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["choices"][0]["message"]["content"], body
            return {"model": key, "start": start, "end": time.monotonic(), "response": body}

        gpu_task, npu_task = asyncio.create_task(generate(args.gpu)), asyncio.create_task(generate(args.npu))
        saw_overlap = False
        while not gpu_task.done() or not npu_task.done():
            states = await sessions()
            saw_overlap |= bool(states[args.gpu]["active_request"] and states[args.npu]["active_request"])
            await asyncio.sleep(.1)
        gpu, npu = await asyncio.gather(gpu_task, npu_task)
        states = await sessions()
        assert saw_overlap, "Both requests must be observed active at the same time"
        assert states[args.gpu]["actual_device"] == "CL", states[args.gpu]
        assert "ROCKCHIP" in states[args.npu]["actual_device"], states[args.npu]
        for key in (args.gpu,args.npu):
            assert states[key]["backend_path"] == "/home/orangepi/tinygrad", states[key]
        assert states[args.npu]["last_metrics"]["npu_submissions"] > 0
        result = {"passed": True, "backend": "/home/orangepi/tinygrad", "overlapping_active_requests": saw_overlap,
                  "gpu": gpu, "npu": npu, "sessions": list(states.values())}
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print("PASS concurrent GPU and NPU generation", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
