"""Real-device HTTP checks. Run against an oNPU server with an installed model.

This intentionally loads the chosen model, runs real generation, checks Stop and
disconnect, and leaves it loaded. It does not use synthetic model output.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", default="CPU", choices=["CPU", "CL"])
    args = parser.parse_args()
    async with httpx.AsyncClient(base_url=args.url, timeout=300, trust_env=False) as client:
        async def status():
            response = await client.get("/api/status")
            response.raise_for_status()
            return response.json()["runtime"]

        async def wait_ready(limit=300):
            end = time.monotonic() + limit
            while time.monotonic() < end:
                state = await status()
                assert state["model_id"] == args.model, "Model changed during verification"
                if state["state"] == "ready":
                    return state
                assert state["state"] != "error", state
                await asyncio.sleep(0.5)
            raise AssertionError("Worker did not become ready")

        state = await status()
        if state["model_id"] != args.model or state["selected_profile"] != args.profile or state["state"] != "ready":
            result = await client.post(f"/api/models/{args.model}/load", json={"profile": args.profile, "context": 512, "realize": True})
            assert result.status_code == 202, result.text
        state = await wait_ready()
        print("ready", json.dumps(state), flush=True)
        payload = {"model": args.model, "messages": [{"role": "user", "content": "Reply with exactly the word hello."}], "temperature": 0, "max_tokens": 8}
        result = await client.post("/v1/chat/completions", json=payload)
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["choices"][0]["message"]["content"], body
        state = await wait_ready()
        assert state["last_metrics"]["prompt_tokens"] == body["usage"]["prompt_tokens"]
        assert state["last_metrics"]["completion_tokens"] == body["usage"]["completion_tokens"]
        print("nonstream", json.dumps(body), flush=True)
        print("measured", json.dumps(state["last_metrics"]), flush=True)
        result = await client.post("/v1/chat/completions", json={**payload, "max_tokens": 1})
        assert result.status_code == 200, result.text
        assert result.json()["usage"]["completion_tokens"] <= 1
        print("one-token limit passed", flush=True)
        result = await client.post("/v1/chat/completions", json={**payload, "max_tokens": 512})
        assert result.status_code == 400, result.text
        await wait_ready()
        print("context validation passed", flush=True)

        collected, usage, saw_done = "", None, False
        async with client.stream("POST", "/v1/chat/completions", json={**payload, "stream": True, "stream_options": {"include_usage": True}}) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    saw_done = True
                    break
                chunk = json.loads(line[6:])
                assert "error" not in chunk, chunk
                usage = chunk.get("usage", usage)
                for choice in chunk.get("choices", []):
                    collected += choice.get("delta", {}).get("content", "")
        assert collected and usage and saw_done
        print("streaming", repr(collected), usage, flush=True)
        await wait_ready()

        long = {**payload, "messages": [{"role": "user", "content": "Count every number from 1 to 100 and explain each."}], "max_tokens": 128, "stream": True}
        async with client.stream("POST", "/v1/chat/completions", json=long) as response:
            assert response.status_code == 200
            # Header/role delta arrives while prefill is still in progress.
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    concurrent = await client.post("/v1/chat/completions", json=payload)
                    assert concurrent.status_code == 409
                    stop = await client.post("/api/chat/stop")
                    assert stop.status_code == 200
                    break
        await wait_ready(45)
        print("Stop and concurrent admission passed", flush=True)
        async with client.stream("POST", "/v1/chat/completions", json=long) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    break
        await wait_ready(45)
        print("client disconnect cancellation passed", flush=True)
        print("PASS", args.profile, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
