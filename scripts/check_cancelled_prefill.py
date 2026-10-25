"""Verify live Stop retains completed KV chunks and a retry finishes correctly."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--retry-input", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stop-after", type=float, default=5)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    retry = json.loads((args.retry_input or args.input).read_text())
    model = payload["model"]
    report = {"complete": False, "input": str(args.input), "retry_input": str(args.retry_input or args.input)}

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    async with httpx.AsyncClient(base_url=args.url, trust_env=False, timeout=300) as client:
        async def ready():
            for _ in range(450):
                response = await client.get("/api/status")
                response.raise_for_status()
                state = next(s for s in response.json()["runtime"]["sessions"] if s["model_id"] == model)
                if state["state"] == "ready" and not state["active_request"]:
                    return state
                assert state["state"] != "error", state
                await asyncio.sleep(.1)
            raise AssertionError("Stop failed to recover")

        try:
            report["initial"] = await ready()
            save()
            async with client.stream("POST", "/v1/chat/completions", json={**payload, "stream": True}) as stream:
                stream.raise_for_status()
                lines = stream.aiter_lines()
                while True:
                    line = await anext(lines)
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        assert "error" not in event, event
                        break
                await asyncio.sleep(args.stop_after)
                current_response = await client.get("/api/status")
                current_response.raise_for_status()
                before_stop = next(s for s in current_response.json()["runtime"]["sessions"] if s["model_id"] == model)
                assert before_stop["state"] == "generating" and before_stop["active_request"], "Request finished before Stop"
                report["before_stop"] = before_stop
                started = time.monotonic()
                response = await client.post("/api/chat/stop", params={"model": model})
                response.raise_for_status()
                await ready()
                report["stop_recovery_seconds"] = time.monotonic() - started
                save()
            started = time.monotonic()
            response = await client.post("/v1/chat/completions", json={**retry, "stream": False})
            response.raise_for_status()
            body = response.json()
            state = await ready()
            report.update(retry_seconds=time.monotonic() - started, response=body, metrics=state["last_metrics"])
            assert report["metrics"]["cached_tokens"] >= 128, report["metrics"]
            assert args.expected.lower() in body["choices"][0]["message"]["content"].lower(), body
            if args.exact:
                assert body["choices"][0]["message"]["content"] == args.expected, body
            report["complete"] = True
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
