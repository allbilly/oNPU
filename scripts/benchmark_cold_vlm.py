"""Measure uncached image requests through the public API, with unchanged pixels.

JPEG comments give each request a fresh image identity. Verify identical decoded
RGB bytes, and require zero retained text positions and uncached vision metrics.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import statistics
import time
from pathlib import Path

import httpx

from onpu.vlm import decode_image


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--warmup-input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("Need at least one measured request")
    original = json.loads(args.input.read_text())
    warmup = json.loads(args.warmup_input.read_text())
    report = {"complete": False, "trials": [], "input": str(args.input), "warmup_input": str(args.warmup_input)}

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    def image_block(payload):
        return next(b for m in payload["messages"] for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url")

    pixels = decode_image(image_block(original)["image_url"]["url"])
    report["decoded_rgb_sha256"] = hashlib.sha256(pixels).hexdigest()
    async with httpx.AsyncClient(base_url=args.url, trust_env=False, timeout=1800) as client:
        try:
            for index in range(3 + args.runs):
                payload = json.loads(json.dumps(warmup if index < 3 else original))
                payload.update(stream=True, stream_options={"include_usage": True})
                block = image_block(payload)
                prefix, encoded = block["image_url"]["url"].split(",", 1)
                image = base64.b64decode(encoded)
                assert prefix == "data:image/jpeg;base64" and image[:2] == b"\xff\xd8"
                comment = f"oNPU uncached trial {time.time_ns()} {index}".encode()
                image = image[:2] + b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment + image[2:]
                block["image_url"]["url"] = prefix + "," + base64.b64encode(image).decode()
                assert decode_image(block["image_url"]["url"]) == pixels
                trial = {"index": index, "warmup": index < 3,
                         "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
                started = time.monotonic()
                answer, usage, done, finish_reason = [], None, False, None
                async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line == "data: [DONE]":
                            done = True
                            break
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        assert "error" not in event, event
                        usage = event.get("usage", usage)
                        for choice in event.get("choices", []):
                            if choice.get("finish_reason") is not None:
                                finish_reason = choice["finish_reason"]
                        content = "".join(c.get("delta", {}).get("content", "") for c in event.get("choices", []))
                        if content and "first_content_seconds" not in trial:
                            trial["first_content_seconds"] = time.monotonic() - started
                        answer.append(content)
                state_response = await client.get("/api/status")
                state_response.raise_for_status()
                state = next(s for s in state_response.json()["runtime"]["sessions"] if s["model_id"] == payload["model"])
                trial.update(wall_seconds=time.monotonic() - started, answer="".join(answer), usage=usage,
                             finish_reason=finish_reason, metrics=state["last_metrics"])
                report["trials"].append(trial)
                save()
                assert done and usage and state["state"] == "ready" and not state["active_request"], trial
                assert trial["metrics"]["cached_tokens"] == 0 and trial["metrics"]["vision_cached"] is False, trial
                assert usage["prompt_tokens"] + payload["max_tokens"] <= state["context"], trial
                print(json.dumps(trial), flush=True)
            measured = [r for r in report["trials"] if not r["warmup"]]
            report.update(complete=True, median_prefill_tps=statistics.median(r["metrics"]["prefill_tps"] for r in measured),
                          median_decode_tps=statistics.median(r["metrics"]["decode_tps"] for r in measured),
                          median_first_content_seconds=statistics.median(r["first_content_seconds"] for r in measured))
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save()


if __name__ == "__main__":
    asyncio.run(main())
