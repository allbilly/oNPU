"""Bounded live-API context, retrieval, cancellation and resource checks.

Run with the configured tinygrad checkout and its dependencies on PYTHONPATH.
The target model must already be ready at --context. No model settings change.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path

import httpx

STOP_TIMEOUT = 45


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
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--labels", help="Comma-separated request labels to run; Stop/disconnect checks still run")
    parser.add_argument("--fixtures", type=Path, help="Save selected request payloads for independent reference checks")
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    selected = set(args.labels.split(",")) if args.labels else None
    report = {"complete": False, "model": args.model, "context": args.context, "started": time.time(), "cases": [], "health": [],
              "selected_labels": sorted(selected) if selected else None, "request_deadline_seconds": args.request_timeout}

    def save():
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    save()
    try:
        await verify(args, report, save, selected)
    except BaseException as exc:
        report["complete"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        save()


async def verify(args, report, save, selected):
    from tinygrad.llm.cli import SimpleTokenizer
    if args.checkpoint.is_dir():
        from onpu.vlm import load_tokenizer, render_messages
        tokenizer = load_tokenizer(args.checkpoint)
        def render(messages):
            return render_messages(messages, 64)[0]
    else:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tinygrad.llm.gguf import gguf_parse
        kv, _ = gguf_parse(args.checkpoint)
        tokenizer = SimpleTokenizer.from_gguf_kv(kv)
        env = ImmutableSandboxedEnvironment()
        env.filters["tojson"] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
        env.globals.update(raise_exception=lambda msg: (_ for _ in ()).throw(ValueError(msg)),
                           strftime_now=lambda fmt: time.strftime(fmt), bos_token="", eos_token=tokenizer.decode([tokenizer.eos_id]))
        template = env.from_string(kv["tokenizer.chat_template"])
        def render(messages):
            return template.render(messages=messages, tools=None, add_generation_prompt=True,
                                   preserve_thinking=True, enable_thinking=False)
    image_block = None
    if args.image:
        image_block = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(args.image.read_bytes()).decode()}}
    def count(messages):
        return len(tokenizer.encode(render(messages)))

    # Tokenize a deterministic, varied filler once. It contains no retrieval answers.
    records = "".join(f"Record {index}: The quiet garden has a path beside the wooden fence.\n" for index in range(args.context))
    filler = tokenizer.encode(records)
    serial = 0

    def messages_at(target, position=.5, history=None, word="ORCHID", with_image=False):
        nonlocal serial
        serial += 1
        history = list(history or [])
        budget = max(0, target - count(history + [{"role": "user", "content": ""}]) - 100)
        padding = 1
        for _ in range(8):
            split = int(budget * position)
            text = (f"Trial {serial}. Read the records and copy only the requested secret word.\n" + tokenizer.decode(filler[:split]) +
                    f"\nThe secret word is {word}.\n" + tokenizer.decode(filler[split:budget]) +
                    "\n" + " x" * padding + "\nWhat is the secret word? Reply with only that word.")
            content = [{"type": "text", "text": text}]
            if with_image:
                content.insert(0, image_block)
            messages = history + [{"role": "user", "content": content if with_image else text}]
            delta = target - count(messages)
            if not delta:
                return messages
            if padding + delta >= 1:
                padding += delta
            else:
                budget = max(0, budget + delta - 8)
                padding += 8
        raise AssertionError(f"Could not construct target {target}: got {count(messages)}")

    async with httpx.AsyncClient(base_url=args.url, timeout=600, trust_env=False) as client:
        async def state():
            response = await client.get("/api/status", timeout=10)
            response.raise_for_status()
            body = response.json()
            current = next(s for s in body["runtime"]["sessions"] if s["model_id"] == args.model)
            return current, body

        async def ready(limit=45):
            deadline = time.monotonic() + limit
            while time.monotonic() < deadline:
                current, _ = await state()
                if current["state"] == "ready" and not current["active_request"]:
                    return current
                assert current["state"] != "error", current
                await asyncio.sleep(.1)
            raise AssertionError("Worker did not recover")

        initial = await ready()
        assert initial["context"] == args.context, initial
        report["initial"] = initial
        save()
        monitoring = True

        async def monitor():
            while monitoring:
                started = time.monotonic()
                try:
                    current, body = await state()
                    report["health"].append({"seconds": time.time() - report["started"], "state": current["state"],
                        "status_ms": (time.monotonic() - started) * 1000, "rss_bytes": current["rss_bytes"],
                        "available_bytes": body["hardware"]["memory"]["available"], "temperature": body["hardware"]["temperature"]})
                except (httpx.HTTPError, OSError, ValueError, TypeError, LookupError, RuntimeError, AssertionError) as exc:
                    report["health"].append({"seconds": time.time() - report["started"], "error": str(exc)})
                if len(report["health"]) % 5 == 0:
                    save()
                await asyncio.sleep(1)

        monitoring_task = asyncio.create_task(monitor())

        async def run(label, messages, limit=8, temperature=0., stream=False, expected="ORCHID", status=200, expected_answer=None):
            if selected is not None and label not in selected:
                return {"label": label, "skipped": True}
            payload = {"model": args.model, "messages": messages, "max_tokens": limit, "temperature": temperature, "stream": stream}
            if stream:
                payload["stream_options"] = {"include_usage": True}
            if args.fixtures:
                args.fixtures.mkdir(parents=True, exist_ok=True)
                (args.fixtures / (label + ".json")).write_text(json.dumps(payload) + "\n")
            if args.fixtures_only:
                return {"label": label, "fixture_only": True}
            row = {"label": label, "prompt_tokens": count(messages), "max_tokens": limit, "temperature": temperature,
                   "stream": stream, "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
            started = time.monotonic()
            try:
                if stream and status == 200:
                    chunks, usage, done, keepalive = [], None, False, 0
                    async with asyncio.timeout(args.request_timeout):
                        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                            row["status"] = response.status_code
                            async for line in response.aiter_lines():
                                if line.startswith(":"):
                                    keepalive += 1
                                    row["keepalives"] = keepalive
                                if not line.startswith("data: "):
                                    continue
                                if line == "data: [DONE]":
                                    done = True
                                    break
                                event = json.loads(line[6:])
                                assert "error" not in event, event
                                usage = event.get("usage", usage)
                                pieces = [choice.get("delta", {}).get("content", "") for choice in event.get("choices", [])]
                                chunks.extend(pieces)
                                if any(pieces) and "first_content_seconds" not in row:
                                    row["first_content_seconds"] = time.monotonic() - started
                                row["partial_answer"] = "".join(chunks)
                    answer = "".join(chunks)
                    row.update(answer=answer, usage=usage, done=done, keepalives=keepalive)
                    assert done and usage, row
                else:
                    response = await asyncio.wait_for(client.post("/v1/chat/completions", json=payload), args.request_timeout)
                    row["status"] = response.status_code
                    body = response.json()
                    answer = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    row.update(answer=answer, usage=body.get("usage"))
                    if response.status_code != 200:
                        row["error_body"] = body
                assert row["status"] == status, row
                current = await ready()
                if status == 200:
                    assert answer and row["usage"]["prompt_tokens"] == row["prompt_tokens"], row
                    assert 0 < row["usage"]["completion_tokens"] <= limit, row
                    row["metrics"] = current["last_metrics"]
                    row["correct"] = expected.lower() in answer.lower() if expected else None
                    if label in ("cached-near-limit", "cached-after-overflow"):
                        row["cache_pass"] = row["metrics"]["cached_tokens"] >= row["prompt_tokens"] - 1
                        if image_block:
                            row["cache_pass"] &= row["metrics"].get("vision_cached") is True
                        assert row["cache_pass"], row
                    if expected_answer is not None:
                        row["cache_output_pass"] = answer == expected_answer
                        assert row["cache_output_pass"], row
                row["api_pass"] = True
            except (httpx.HTTPError, OSError, ValueError, TypeError, LookupError, RuntimeError, AssertionError) as exc:
                row.update(api_pass=False, failure=f"{type(exc).__name__}: {exc}")
                if isinstance(exc, TimeoutError):
                    row["deadline_exceeded"] = True
                    stopped = await client.post("/api/chat/stop", params={"model": args.model})
                    row["stop_after_deadline_status"] = stopped.status_code
            row["wall_seconds"] = time.monotonic() - started
            report["cases"].append(row)
            save()
            print(json.dumps(row), flush=True)
            await ready()
            return row

        try:
            near = args.context - 8
            targets = ([128, 256, 384, near] if image_block else [min(1024, near), min(4096, near), near])
            for target in dict.fromkeys(targets):
                await run(f"length-{target}", messages_at(target, .1, with_image=bool(image_block)))
            for fraction, name in ((.02, "beginning"), (.5, "middle"), (.98, "end")):
                await run(f"retrieval-{name}", messages_at(near, fraction, with_image=bool(image_block)), stream=name == "middle")
            await run("sampled-near-limit", messages_at(near, with_image=bool(image_block)), temperature=.6)
            boundary = messages_at(near, with_image=bool(image_block))
            baseline = await run("exact-context-total", boundary, limit=args.context - count(boundary))
            await run("cached-near-limit", boundary, limit=args.context - count(boundary), stream=True,
                      expected_answer=baseline.get("answer"))
            # One remaining output position prevents early EOS from leaving an
            # apparently full-budget request below the actual capacity.
            actual_messages = messages_at(args.context - 1, with_image=bool(image_block))
            actual = await run("actual-context-total", actual_messages, limit=1, expected=None)
            for stream in (False, True):
                await run(f"overflow-stream-{stream}", boundary, limit=args.context - count(boundary) + 1, stream=stream, status=400, expected=None)
            await run("cached-after-overflow", actual_messages, limit=1, stream=True, expected=None,
                      expected_answer=actual.get("answer"))
            await run("recovery-after-overflow", messages_at(128, with_image=bool(image_block)))
            history = []
            growth = dict.fromkeys(min(target, near) for target in
                                   ([128, 240, 360, near] if image_block else [256, 1024, 4096, near]))
            for index, target in enumerate(growth):
                assert count(history + [{"role": "user", "content": ""}]) <= target <= near, "Conversation turn has no room in context"
                messages = messages_at(target, history=history, with_image=bool(image_block) and not history)
                assert count(messages) + 8 <= args.context, "Conversation turn exceeds context"
                row = await run(f"conversation-turn-{index + 1}", messages)
                history = messages + [{"role": "assistant", "content": row.get("answer") or ""}]
            for index in range(args.repeats):
                # Alternate unrelated answers to expose stale/cross-request KV history.
                word = "ORCHID" if index % 2 == 0 else "NEPTUNE"
                target = near if image_block or index % 2 == 0 else 128
                await run(f"repeat-{index}", messages_at(target, word=word, with_image=bool(image_block)), expected=word)
            for label in (() if args.fixtures_only else ("stop", "disconnect")):
                messages = messages_at(near, with_image=bool(image_block))
                payload = {"model": args.model, "messages": messages, "max_tokens": 8, "temperature": .6, "stream": True}
                started = time.monotonic()
                async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    assert response.status_code == 200
                    lines = response.aiter_lines()
                    async for line in lines:
                        if line.startswith("data: "):
                            assert line != "data: [DONE]", "Stream completed before cancellation"
                            assert "error" not in json.loads(line[6:]), line
                            if label == "stop":
                                competing = await client.post("/v1/chat/completions", json={**payload, "stream": False})
                                assert competing.status_code == 409, competing.text
                                await stop_generation(client, response, lines, args.model, ready)
                            break
                    else:
                        raise AssertionError("Stream ended before cancellation")
                await ready()
                row = {"label": label, "api_pass": True, "recovery_seconds": time.monotonic() - started}
                report["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
                await run(f"recovery-after-{label}", messages_at(128, with_image=bool(image_block)))
            report["final"] = await ready()
            report["complete"] = True
        finally:
            monitoring = False
            await monitoring_task
            report["elapsed_seconds"] = time.time() - report["started"]
            report["api_failures"] = sum(not row["api_pass"] for row in report["cases"])
            scored = [row for row in report["cases"] if row.get("correct") is not None]
            report["retrieval_correct"] = sum(row["correct"] for row in scored)
            report["retrieval_total"] = len(scored)
            save()
            summary_keys = ("complete", "elapsed_seconds", "api_failures", "retrieval_correct", "retrieval_total")
            print(json.dumps({k: report[k] for k in summary_keys}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
