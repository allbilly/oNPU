"""Measure cold and cached serving at a GGUF model's declared context limit.

The first request must report zero cached tokens. Unload and reload the target
worker before rerunning this fixture; a previous conversation can retain even a
short shared chat prefix. This script does not reset or reload serving workers.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=7200)
    parser.add_argument("--other-model", help="Independent ready model to query during cold prefill")
    parser.add_argument("--wait-report", type=Path, help="Wait for a preceding benchmark to complete")
    parser.add_argument("--multiple-needles", action="store_true", help="Check beginning/middle/end retrieval and the exact context boundary")
    parser.add_argument("--max-tokens", type=int, default=8, help="Output budget for scored retrieval replies")
    args = parser.parse_args()
    report = {"complete": False, "started": time.time(), "model": args.model, "cases": [], "health": [],
              "multiple_needles": args.multiple_needles, "reply_budget": args.max_tokens}

    def save():
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    save()
    try:
        await verify(args, report, save)
    except BaseException as exc:
        report["complete"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        save()


async def verify(args, report, save):
    if args.wait_report:
        while True:
            try:
                preceding = json.loads(args.wait_report.read_text()) if args.wait_report.exists() else {}
            except json.JSONDecodeError:
                # Older running harnesses can still write reports in place.
                await asyncio.sleep(1)
                continue
            if preceding.get("complete"):
                break
            if preceding.get("error"):
                raise RuntimeError(f"Preceding benchmark failed: {preceding['error']}")
            if "elapsed_seconds" in preceding:
                raise RuntimeError("Preceding benchmark did not complete")
            await asyncio.sleep(5)
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tinygrad.llm.cli import SimpleTokenizer
    from tinygrad.llm.gguf import gguf_parse
    kv, _ = gguf_parse(args.checkpoint)
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    env = ImmutableSandboxedEnvironment()
    env.filters["tojson"] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
    env.globals.update(raise_exception=lambda msg: (_ for _ in ()).throw(ValueError(msg)),
                       strftime_now=lambda fmt: time.strftime(fmt), bos_token="", eos_token=tokenizer.decode([tokenizer.eos_id]))
    template = env.from_string(kv["tokenizer.chat_template"])

    def count(messages):
        rendered = template.render(messages=messages, tools=None, add_generation_prompt=True,
                                   preserve_thinking=True, enable_thinking=False)
        return len(tokenizer.encode(rendered))

    async with httpx.AsyncClient(base_url=args.url, trust_env=False, timeout=30) as client:
        async def state():
            response = await client.get("/api/status")
            response.raise_for_status()
            return response.json()

        initial = await state()
        session = next(s for s in initial["runtime"]["sessions"] if s["model_id"] == args.model)
        arch = kv["general.architecture"]
        context = kv[f"{arch}.context_length"]
        assert session["context"] == context and session["state"] == "ready" and not session["active_request"], session
        report.update(context=context, initial=initial)
        filler = tokenizer.encode("".join(f"Record {i}: A quiet garden has a path beside a wooden fence.\n" for i in range(context)))

        def prompt(target, history, word, location=None):
            budget = max(0, target - count(history + [{"role": "user", "content": ""}]) - 80)
            padding = 1
            for _ in range(10):
                if args.multiple_needles and not history:
                    cuts = [0, int(budget * .02), int(budget * .50), int(budget * .98), budget]
                    chunks = [tokenizer.decode(filler[a:b]) for a, b in pairwise(cuts)]
                    text = (chunks[0] + "\nThe BEGINNING secret word is ORCHID.\n" + chunks[1] +
                            "\nThe MIDDLE secret word is NEPTUNE.\n" + chunks[2] +
                            "\nThe END secret word is QUARTZ.\n" + chunks[3])
                else:
                    text = tokenizer.decode(filler[:budget])
                    if not args.multiple_needles:
                        text += f"\nThe secret word is {word}.\n"
                question = f"What is the {location} secret word?" if location else "What is the secret word?"
                text += " x" * padding + f"\n{question} Reply with only that word."
                messages = history + [{"role": "user", "content": text}]
                delta = target - count(messages)
                if not delta:
                    return messages
                if padding + delta >= 1:
                    padding += delta
                else:
                    budget = max(0, budget + delta - 8)
                    padding += 8
            raise AssertionError("Could not construct exact prompt length")

        monitoring = True

        async def monitor():
            while monitoring:
                started = time.monotonic()
                body = await state()
                report["health"].append({"seconds": time.time() - report["started"], "status_ms": (time.monotonic() - started) * 1000,
                    "memory": body["hardware"]["memory"], "temperature": body["hardware"]["temperature"],
                    "sessions": [{key: s[key] for key in ("model_id", "state", "rss_bytes", "active_request")} for s in body["runtime"]["sessions"]]})
                save()
                await asyncio.sleep(2)

        async def run(label, messages, expected, max_tokens=None):
            max_tokens = args.max_tokens if max_tokens is None else max_tokens
            payload = {"model": args.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}
            row = {"label": label, "prompt_tokens": count(messages),
                   "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                   "deadline_seconds": args.deadline, "keepalives": 0}
            args.report.with_name(args.report.stem + "-" + label + "-input.json").write_text(json.dumps(payload) + "\n")
            report["cases"].append(row)
            save()
            started = time.monotonic()
            pieces, usage, done = [], None, False
            try:
                async with asyncio.timeout(args.deadline):
                    async with client.stream("POST", "/v1/chat/completions", json=payload, timeout=None) as response:
                        row["status"] = response.status_code
                        assert response.status_code == 200, await response.aread()
                        async for line in response.aiter_lines():
                            if line.startswith(":"):
                                row["keepalives"] += 1
                            if line == "data: [DONE]":
                                done = True
                                break
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            assert "error" not in event, event
                            usage = event.get("usage", usage)
                            text = "".join(c.get("delta", {}).get("content", "") for c in event.get("choices", []))
                            if text and not pieces:
                                row["first_content_seconds"] = time.monotonic() - started
                            if text:
                                pieces.append(text)
                            row["partial_answer"] = "".join(pieces)
                row.update(answer="".join(pieces), usage=usage, done=done)
                assert done and usage and usage["prompt_tokens"] == row["prompt_tokens"] and pieces, row
                current = next(s for s in (await state())["runtime"]["sessions"] if s["model_id"] == args.model)
                row.update(metrics=current["last_metrics"],
                           correct=expected.lower() in row["answer"].lower() if expected else None, api_pass=True)
                if label == "actual-context-total":
                    assert usage["total_tokens"] == context, row
                if label == "cold":
                    row["cold_pass"] = row["metrics"].get("cached_tokens") == 0
                    if not row["cold_pass"]:
                        raise RuntimeError(f"Cold trial reused {row['metrics'].get('cached_tokens')} cached tokens. "
                                           "Unload and reload the target model, then rerun before sending any chat requests. "
                                           "This timing is not an uncached maximum-context measurement.")
                else:
                    assert row["metrics"]["cached_tokens"] >= report["cases"][0]["prompt_tokens"] - 8, row
            except BaseException as exc:
                row.update(api_pass=False, failure=f"{type(exc).__name__}: {exc}")
                await client.post("/api/chat/stop", params={"model": args.model})
                raise
            finally:
                row["wall_seconds"] = time.monotonic() - started
                save()
                print(json.dumps(row), flush=True)
            return row

        async def independent():
            if args.other_model:
                await asyncio.sleep(10)
                started = time.monotonic()
                response = await client.post("/v1/chat/completions", json={"model": args.other_model,
                    "messages": [{"role": "user", "content": "Reply with only the word hello."}], "max_tokens": 8, "temperature": 0}, timeout=60)
                response.raise_for_status()
                report["independent_model"] = {"model": args.other_model, "seconds": time.monotonic() - started, "response": response.json()}
                save()

        if args.multiple_needles:
            initial_messages = prompt(context - 256, [], "ORCHID", "BEGINNING")
            # Check the longest permitted assistant replies before starting an
            # expensive prefill, so every following question still fits.
            dummy_reply = " x" * args.max_tokens
            middle_probe = prompt(context - 128, initial_messages + [{"role": "assistant", "content": dummy_reply}], "NEPTUNE", "MIDDLE")
            probe_history = middle_probe + [{"role": "assistant", "content": dummy_reply}]
            end_probe = prompt(context - args.max_tokens, probe_history, "QUARTZ", "END")
            actual_probe = prompt(context - 1, probe_history, "QUARTZ", "END")
            report["fixture_preflight"] = [count(messages) for messages in (initial_messages, middle_probe, end_probe, actual_probe)]
        else:
            initial_messages = prompt(context - 128, [], "ORCHID")

        observer, other = asyncio.create_task(monitor()), asyncio.create_task(independent())
        try:
            first = await run("cold", initial_messages, "ORCHID")
            repeated = await run("cached", initial_messages, "ORCHID")
            repeated["matches_cold"] = repeated["answer"] == first["answer"]
            assert repeated["matches_cold"], repeated
            history = initial_messages + [{"role": "assistant", "content": first["answer"]}]
            if args.multiple_needles:
                middle_messages = prompt(context - 128, history, "NEPTUNE", "MIDDLE")
                middle = await run("middle", middle_messages, "NEPTUNE")
                history = middle_messages + [{"role": "assistant", "content": middle["answer"]}]
                await run("growing-to-max", prompt(context - args.max_tokens, history, "QUARTZ", "END"), "QUARTZ")
                exact_messages = prompt(context - 1, history, "QUARTZ", "END")
                await run("actual-context-total", exact_messages, None, max_tokens=1)
                for streaming in (False, True):
                    response = await client.post("/v1/chat/completions", json={"model": args.model, "messages": exact_messages,
                        "max_tokens": 2, "temperature": 0, "stream": streaming})
                    assert response.status_code == 400, response.text
                    report["cases"].append({"label": f"overflow-stream-{streaming}", "status": response.status_code, "api_pass": True})
                    save()
                await run("cached-after-overflow", exact_messages, None, max_tokens=1)
            else:
                await run("growing-to-max", prompt(context - args.max_tokens, history, "NEPTUNE"), "NEPTUNE")
            await other
            report["final"] = await state()
            report["complete"] = True
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            monitoring = False
            await observer
            if not other.done():
                other.cancel()
            await asyncio.gather(other, return_exceptions=True)
            report["elapsed_seconds"] = time.time() - report["started"]
            save()


if __name__ == "__main__":
    asyncio.run(main())
