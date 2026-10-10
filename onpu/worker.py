"""Single-owner tinygrad inference worker. JSON lines are the only stdout protocol.

All model operations run on one thread; the input thread only queues commands and
sets cancellation flags. No changes are made to the user's tinygrad checkout.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from types import SimpleNamespace


class Cancelled(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--realize", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    write_lock = threading.Lock()

    def emit(event: str, **data):
        with write_lock:
            protocol.write(json.dumps({"event": event, **data}, ensure_ascii=False) + "\n")
            protocol.flush()

    try:
        from tinygrad import Device, Tensor
        from tinygrad.helpers import GlobalCounters
        if args.probe:
            result = (Tensor([1., 2., 3.]) + 1).tolist()
            if result != [2., 3., 4.]:
                raise ValueError("Tensor smoke test failed")
            name = Device.DEFAULT
            if name.startswith("CL"):
                device = Device[name]
                name = device.device_name
            emit("probe", available=True, device=Device.DEFAULT, name=name,
                 driver=getattr(Device[Device.DEFAULT], "driver_version", None))
            return

        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tinygrad import nn
        from tinygrad.llm.cli import SimpleTokenizer, FallbackTemplate
        from tinygrad.llm.model import Transformer
        from tinygrad.llm.serve import Handler, normalize_messages

        start = time.perf_counter()
        emit("state", state="loading", message="Reading model weights")
        model, kv = Transformer.from_gguf(args.model, args.context, realize=bool(args.realize))
        tok = SimpleTokenizer.from_gguf_kv(kv)
        template = FallbackTemplate(tok)
        if source := kv.get("tokenizer.chat_template"):
            env = ImmutableSandboxedEnvironment()
            env.filters["tojson"] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
            env.globals.update(raise_exception=lambda msg: (_ for _ in ()).throw(ValueError(msg)),
                               strftime_now=lambda fmt: time.strftime(fmt),
                               bos_token=tok.decode([tok.bos_id]) if tok.bos_id is not None else "", eos_token=tok.decode([tok.eos_id]))
            template = env.from_string(source)

        commands: queue.Queue = queue.Queue()
        cancellations: dict[str, threading.Event] = {}
        active_cancel = threading.Event()

        # Check between prefill chunks and decode operations, rather than waiting
        # for a complete prompt to produce its first token before honoring Stop.
        def checkpoint(jit):
            def run(*a, **kw):
                if active_cancel.is_set():
                    raise Cancelled()
                return jit(*a, **kw)
            return run
        model.prefill_jit = checkpoint(model.prefill_jit)
        model.rollout_jit = checkpoint(model.rollout_jit)

        emit("state", state="preparing", message="Compiling and warming the model")
        model.warmup()
        actual_device = nn.state.get_parameters(model)[0].device
        handler = object.__new__(Handler)
        handler.server = SimpleNamespace(model=model, tok=tok)
        emit("ready", device=actual_device, context=model.max_context, name=kv.get("general.name", args.model),
             load_seconds=time.perf_counter() - start, weight_bytes=GlobalCounters.mem_used)

        def input_loop():
            try:
                for line in sys.stdin:
                    command = json.loads(line)
                    if command["op"] == "cancel":
                        if flag := cancellations.get(command["id"]):
                            flag.set()
                    else:
                        if command["op"] == "generate":
                            cancellations[command["id"]] = threading.Event()
                        commands.put(command)
            finally:
                for flag in list(cancellations.values()):
                    flag.set()
                commands.put({"op": "shutdown"})

        threading.Thread(target=input_loop, daemon=True).start()
        while True:
            command = commands.get()
            if command["op"] == "shutdown":
                return
            if command["op"] != "generate":
                continue
            key, body = command["id"], command["body"]
            active_cancel = cancellations[key]
            chunks = None
            started = first = last = None
            try:
                prep = time.perf_counter()
                normalize_messages(body["messages"])
                rendered = template.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True,
                                           preserve_thinking=True, enable_thinking=body.get("enable_thinking", False))
                ids = tok.encode(rendered)
                limit = body["max_tokens"]
                if len(ids) + limit > model.max_context:
                    raise ValueError(f"Prompt ({len(ids)} tokens) + reply limit ({limit}) exceeds context ({model.max_context}). Reduce the reply limit or start a new chat.")
                cached = model.get_start_pos(ids)
                if active_cancel.is_set():
                    raise Cancelled()
                emit("started", id=key, prompt_tokens=len(ids), cached_tokens=cached, preparation_seconds=time.perf_counter() - prep)
                started = time.perf_counter()
                chunks = handler.run_model(ids.copy(), body["model"], include_usage=True, max_tokens=limit,
                                           temperature=body["temperature"], reasoning=rendered.rstrip().endswith("<think>"))
                usage = None
                for chunk in chunks:
                    if active_cancel.is_set():
                        raise Cancelled()
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        if any(choice.get("delta", {}).get(field) for field in ("content", "reasoning_content", "tool_calls")):
                            first = first or time.perf_counter()
                            last = time.perf_counter()
                    emit("chunk", id=key, data=chunk)
                ended = time.perf_counter()
                tokens = (usage or {}).get("completion_tokens", 0)
                # TTFT measures the first emitted text; token timings are not fabricated
                # from the role event, which is emitted before inference begins.
                elapsed = ended - started
                metrics = {"prompt_tokens": len(ids), "cached_tokens": cached, "completion_tokens": tokens,
                           "ttft_ms": (first - started) * 1000 if first else None, "total_seconds": elapsed,
                           "decode_tps": (tokens - 1) / (last - first) if tokens > 1 and first and last and last > first else None,
                           "prefill_tps": (len(ids) - cached) / (first - started) if first and first > started else None,
                           "device": actual_device, "allocated_bytes": GlobalCounters.mem_used}
                emit("done", id=key, metrics=metrics)
            except Cancelled:
                model._cached_tokens = []
                emit("cancelled", id=key)
            except ValueError as exc:
                model._cached_tokens = []
                emit("error", id=key, message=str(exc), status=400)
            except Exception as exc:
                model._cached_tokens = []
                traceback.print_exc()
                emit("error", id=key, message=str(exc), status=500)
            finally:
                if chunks is not None:
                    chunks.close()
                cancellations.pop(key, None)
    except Exception as exc:
        traceback.print_exc()
        emit("fatal", message=str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
