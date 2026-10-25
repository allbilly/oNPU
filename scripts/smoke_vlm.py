"""Check a ready VLM worker with real image requests, sampling, Stop and disconnect."""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx

STOP_TIMEOUT = 5


async def stop_generation(client, response, lines, model, ready):
    request_id = response.headers.get("X-Request-ID")
    assert request_id, "Cancellation stream has no request ID"
    cancelled = False
    try:
        async with asyncio.timeout(STOP_TIMEOUT):
            stopped = await client.post("/api/chat/stop", params={"model": model, "request_id": request_id})
            assert stopped.status_code == 200, stopped.text
            async for line in lines:
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    assert cancelled, "Stop completed without a cancellation acknowledgement"
                    await ready(STOP_TIMEOUT)
                    return
                event = json.loads(line[6:])
                if error := event.get("error"):
                    assert error.get("message") == "Generation stopped", event
                    cancelled = True
    except TimeoutError as exc:
        raise AssertionError("Stop did not cancel the connected request before its deadline") from exc
    raise AssertionError("Stop stream ended without a cancellation acknowledgement")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--gpu-url")
    parser.add_argument("--gpu-model")
    args = parser.parse_args()
    report = {"complete": False, "primary_temperature": args.temperature}
    started = time.monotonic()

    def save():
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    save()
    try:
        await verify(args, report)
    except BaseException as exc:
        report["complete"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()


async def verify(args, report):
    media = "png" if args.image.suffix.lower() == ".png" else "jpeg"
    image = f"data:image/{media};base64," + base64.b64encode(args.image.read_bytes()).decode()
    payload = {"model": args.model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "What is shown in this image?"}]}], "temperature": args.temperature, "max_tokens": 32}
    async with httpx.AsyncClient(base_url=args.url, timeout=300, trust_env=False) as client:
        async def state():
            response = await client.get("/api/status")
            response.raise_for_status()
            return next(s for s in response.json()["runtime"]["sessions"] if s["model_id"] == args.model)

        async def ready(seconds=45):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                current = await state()
                assert current["state"] != "error", current
                if current["state"] == "ready":
                    return current
                await asyncio.sleep(.1)
            raise AssertionError("VLM did not return to ready")

        current = await ready()
        assert current["supports_images"] and "ROCKCHIP" in current["actual_device"], current
        primary = "greedy" if args.temperature == 0 else "sampled_primary"
        for label, options in ((primary, {}), ("sampling", {"temperature": .6, "max_tokens": 2})):
            response = await client.post("/v1/chat/completions", json={**payload, **options})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["choices"][0]["message"]["content"], body
            current = await ready()
            assert current["last_metrics"]["npu_submissions"] > 0, current
            report[label] = {"response": body, "metrics": current["last_metrics"]}
            print(label, json.dumps(current["last_metrics"]), flush=True)

        streaming = {**payload, "max_tokens": 128, "stream": True}
        async with client.stream("POST", "/v1/chat/completions", json=streaming) as response:
            assert response.status_code == 200
            lines = response.aiter_lines()
            async for line in lines:
                if not line.startswith("data: "):
                    continue
                assert line != "data: [DONE]", "Stream completed before cancellation"
                assert "error" not in json.loads(line[6:]), line
                competing = await client.post("/v1/chat/completions", json=payload)
                assert competing.status_code == 409, competing.text
                started = time.monotonic()
                await stop_generation(client, response, lines, args.model, ready)
                break
            else:
                raise AssertionError("Stream ended before cancellation")
        await ready()
        report["stop_seconds"] = time.monotonic() - started
        assert report["stop_seconds"] < 5, report

        async with client.stream("POST", "/v1/chat/completions", json=streaming) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    assert line != "data: [DONE]", "Stream completed before cancellation"
                    assert "error" not in json.loads(line[6:]), line
                    break
            else:
                raise AssertionError("Stream ended before cancellation")
        await ready()
        response = await client.post("/v1/chat/completions", json={**payload, "max_tokens": 4})
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"], response.text
        await ready()
        report["disconnect_and_recovery"] = True

        if args.gpu_url and args.gpu_model:
            async with httpx.AsyncClient(base_url=args.gpu_url, timeout=300, trust_env=False) as gpu:
                gpu_payload = {"model": args.gpu_model, "messages": [{"role": "user", "content": "Say hello."}],
                               "temperature": 0, "max_tokens": 8}
                tasks = [asyncio.create_task(client.post("/v1/chat/completions", json=payload)),
                         asyncio.create_task(gpu.post("/v1/chat/completions", json=gpu_payload))]
                overlap = False
                while not all(task.done() for task in tasks):
                    gpu_status = await gpu.get("/api/status")
                    gpu_status.raise_for_status()
                    gpu_state = next(s for s in gpu_status.json()["runtime"]["sessions"] if s["model_id"] == args.gpu_model)
                    npu_state = await state()
                    overlap |= bool(gpu_state["active_request"] and npu_state["active_request"])
                    await asyncio.sleep(.1)
                results = await asyncio.gather(*tasks)
                for result in results:
                    assert result.status_code == 200, result.text
                    assert result.json()["choices"][0]["message"]["content"], result.text
                assert overlap and gpu_state["actual_device"] == "CL"
                report["concurrent_gpu"] = {"overlapping_active_requests": overlap,
                                           "gpu_model": args.gpu_model, "separate_servers": args.gpu_url != args.url}
        report["complete"] = True
        print("PASS image serving, sampling, Stop, disconnect and recovery", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
