import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from tinygrad.llm.serve import Handler

from onpu import speculation
from onpu.library import estimated_memory
from onpu.runtime import EngineError, Runtime
from onpu.store import Store
from onpu.worker import Cancelled


@pytest.fixture
def pair(tmp_path, monkeypatch):
    target = tmp_path / "target.gguf"
    target.write_bytes(b"pinned target and tokenizer")
    directory = tmp_path / "draft"
    directory.mkdir()
    config, weights = directory / "config.json", directory / "model.safetensors"
    config.write_bytes(b"pinned draft config")
    weights.write_bytes(b"pinned draft tensors")
    for name, path in (("TARGET_SHA256", target), ("CONFIG_SHA256", config), ("WEIGHTS_SHA256", weights)):
        monkeypatch.setattr(speculation, name, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(speculation, "WEIGHTS_BYTES", weights.stat().st_size)
    model = {"id": "target", "path": str(target), "size": target.stat().st_size, "parameters": 0}
    options = {"speculation": speculation.MODE, "draft_path": str(directory), "profile": "CL", "context": 512, "realize": True,
               "memory_fraction": 0.8}
    return model, options


def test_default_is_ordinary_and_draft_path_requires_explicit_opt_in():
    speculation.validate_speculation({}, {})
    assert speculation.draft_memory(512, {}) == 0
    assert speculation.draft_memory(512, {"speculation": "off"}) == 0
    with pytest.raises(ValueError, match="requires speculation"):
        speculation.validate_speculation({}, {"draft_path": "/local/draft"})


@pytest.mark.parametrize("change", [{"profile": "CPU"}, {"profile": "ROCKCHIP"}, {"realize": False}, {"npu_prefill": "auto"},
                                    {"npu_decode": "auto"}, {"context": 8193}, {"context": 63}, {"draft_path": ""},
                                    {"speculation": "eagle4"}])
def test_unqualified_execution_is_rejected(pair, change):
    model, options = pair
    with pytest.raises(ValueError):
        speculation.validate_speculation(model, {**options, **change})


@pytest.mark.parametrize("filename, message", [("target.gguf", "Target checkpoint"), ("draft/config.json", "configuration"),
                                              ("draft/model.safetensors", "weights")])
def test_pin_validation_rejects_modified_local_files_even_after_prior_validation(pair, tmp_path, filename, message):
    model, options = pair
    speculation.validate_speculation(model, options)
    path = tmp_path / filename
    old = path.read_bytes()
    path.write_bytes(b"x" + old[1:])
    with pytest.raises(ValueError, match=message):
        speculation.validate_speculation(model, options)


def test_draft_reservation_is_added_to_model_memory(pair):
    model, options = pair
    ordinary = estimated_memory(model, 512, True)
    assert estimated_memory(model, 512, True, options) == ordinary + speculation.draft_memory(512, options)
    assert speculation.draft_memory(512, options) > 512 * 1024**2
    assert speculation.draft_memory(512, options) > speculation.draft_memory(128, options)


@pytest.fixture
def measured_pair(pair, monkeypatch):
    from onpu import execution

    model, options = pair
    model.update(architecture="qwen35", size=2012012800, parameters=1881825088, layers=24, kv_heads=2, head_dim=256)
    monkeypatch.setitem(execution.QUALIFIED_MALI_RECURRENT_CHECKPOINTS, speculation.TARGET_SHA256, (512, True))
    return model, options


def test_experimental_whole_pair_memory_includes_draft_once_and_leaves_ordinary_unchanged(measured_pair):
    model, options = measured_pair
    measured = (9897 + 1536 + 512 + 48) * 1024**2
    assert estimated_memory(model, 512, True, options) == measured
    assert measured > (9896.81640625 + 1536 + 512) * 1024**2
    assert estimated_memory(model, 64, True, options) == measured
    assert estimated_memory(model, 512, True) == 10126515712
    assert estimated_memory(model, 512, True, {**options, "speculation": "off"}) == 10126515712


def test_extended_pair_admission_does_not_expand_the_ordinary_profile(measured_pair):
    from onpu import execution, worker

    model, options = measured_pair
    ordinary = estimated_memory(model, 8192, True)
    checksum, layout = worker.mali_profile(Path(model["path"]), 8192, "CL", "off", "off")
    assert checksum == speculation.TARGET_SHA256 and layout == {}
    assert worker.mali_options(Path(model["path"]), 8192, "CL", "off", "off") == {}
    assert estimated_memory(model, 8192, True, options) == speculation.MEASURED_PAIR_BYTES
    assert execution.QUALIFIED_MALI_RECURRENT_CHECKPOINTS[speculation.TARGET_SHA256] == (512, True)
    assert estimated_memory(model, 8192, True) == ordinary


@pytest.mark.parametrize("context,realize,changes", [
    (8193, True, {}), (63, True, {}), (512, False, {}), (512.0, True, {}),
    (512, True, {"profile": "CPU"}), (512, True, {"profile": "ROCKCHIP"}),
    (512, True, {"npu_prefill": "auto"}), (512, True, {"npu_decode": "auto"}),
    (512, True, {"speculation": "unknown"}), (512, True, {"draft_path": None}),
])
def test_unqualified_pair_keeps_conservative_target_plus_draft_memory(measured_pair, context, realize, changes):
    model, options = measured_pair
    options = {**options, **changes}
    assert estimated_memory(model, context, realize, options) == (
        estimated_memory(model, context, realize) + speculation.draft_memory(context, options))


@pytest.mark.parametrize("metadata", [{"size": 2012012799}, {"parameters": 1881825087}, {"architecture": "llama"},
                                      {"path": ""}, {"path": 42}])
def test_changed_pair_metadata_cannot_borrow_measured_envelope(measured_pair, metadata):
    model, options = measured_pair
    candidate = {**model, **metadata}
    assert estimated_memory(candidate, 512, True, options) == estimated_memory(candidate, 512, True) + speculation.draft_memory(512, options)


@pytest.mark.parametrize("layout", [{"recurrent_scan": True, "pack_weights": False}, {},
                                    {"recurrent_scan": True, "pack_weights": True, "extra_layout": True}])
def test_changed_target_layout_keeps_conservative_pair_estimate(measured_pair, monkeypatch, layout):
    model, options = measured_pair
    monkeypatch.setattr(speculation, "qualified_mali_profile", lambda *args, **kwargs: (speculation.TARGET_SHA256, layout))
    assert estimated_memory(model, 512, True, options) == estimated_memory(model, 512, True) + speculation.draft_memory(512, options)


@pytest.mark.parametrize("filename", ["target.gguf", "draft/config.json", "draft/model.safetensors"])
def test_measured_pair_rechecks_all_pins_after_cached_display_even_when_signatures_collide(measured_pair, tmp_path, monkeypatch, filename):
    from onpu import execution

    model, options = measured_pair
    path = tmp_path / filename
    signature = execution._signature(path.stat())
    monkeypatch.setattr(execution, "_signature", lambda stat: signature)
    assert estimated_memory(model, 512, True, options, refresh=False) == speculation.MEASURED_PAIR_BYTES
    path.write_bytes(b"X" * path.stat().st_size)
    candidate = {**model, "sha256": speculation.TARGET_SHA256}
    conservative = estimated_memory(candidate, 512, True) + speculation.draft_memory(512, options)
    assert estimated_memory(candidate, 512, True, options) == conservative
    assert estimated_memory(candidate, 512, True, options, refresh=False) == conservative


def test_measured_pair_admission_reads_each_file_fresh_once(measured_pair, monkeypatch):
    model, options = measured_pair
    calls, original = [], hashlib.file_digest

    def digest(source, algorithm):
        calls.append(source.name)
        return original(source, algorithm)

    monkeypatch.setattr(hashlib, "file_digest", digest)
    for request in range(2):
        assert estimated_memory(model, 512, True, options) == speculation.MEASURED_PAIR_BYTES
        assert len(calls) == (request + 1) * 3


@pytest.mark.parametrize("available,admitted", [(13 * 1024**3, True), (12 * 1024**3, False)])
def test_runtime_uses_whole_pair_budget_and_retains_available_ram_guard(measured_pair, tmp_path, monkeypatch, available, admitted):
    from onpu import runtime as serving

    model, options = measured_pair
    monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 16739618816, "available": available})
    monkeypatch.setattr(serving.hardware, "rss", lambda pid: 0)
    store = Store(tmp_path / "data")
    runtime = Runtime(store, tmp_path)
    stopped = []

    async def stop():
        stopped.append(True)

    async def launch(*args, **kwargs):
        raise OSError("measured pair admitted")

    monkeypatch.setattr(runtime, "_stop", stop)
    monkeypatch.setattr(serving.asyncio, "create_subprocess_exec", launch)
    try:
        with pytest.raises(EngineError, match="measured pair admitted" if admitted else "available RAM") as error:
            asyncio.run(runtime.load(model, options))
        assert error.value.status == (500 if admitted else 400)
        assert bool(stopped) == admitted
    finally:
        store.close()


def test_decoder_loader_uses_the_repeated_real_pair_execution_settings(monkeypatch, tmp_path):
    import tinygrad
    from tinygrad import dtypes
    from tinygrad.llm import dflash2, dflash2_config, dflash2_selector
    from tinygrad.llm import speculative as backend

    model = SimpleNamespace(config=object(), token_embd=object(), output=object(), output_norm=object(), max_context=512)
    config, weights, selector = object(), {}, object()
    calls = []
    configured = SimpleNamespace()
    calibration = {"seconds_per_token": .4, "samples": [.42,.4,.41], "prompt_tokens": 128}

    def calibrate(target):
        assert target is model
        calls.append(("calibrate", {}))
        return calibration

    def warm(candidate):
        assert candidate is configured and candidate.ordinary_decode_seconds == .4
        calls.append(("warm", {}))
        return {"requests": 3}

    def load(directory, target_config, **kwargs):
        assert directory == tmp_path / "draft" and target_config is model.config
        assert kwargs == {"target_checkpoint": tmp_path / "target.gguf", "device": "CL", "dtype": dtypes.float16}
        return config, weights

    class Draft:
        def __init__(self, draft_config, draft_weights, embedding, head, norm, candidate, *, cache_prefix_kv=True):
            assert draft_config is config and draft_weights is weights
            assert embedding is model.token_embd and head is model.output and norm is model.output_norm
            assert candidate is selector and cache_prefix_kv

        def prepare_mali(self, **options):
            calls.append(("prepare", options))
            return {"linear_layers": 19, "packed_weights": False, "prepared_shared_head": True,
                    "packed_fc": True, "packed_bytes": 48 * 1024**2}

        def prepare_proposals(self, **options):
            calls.append(("proposals", options))

        def prepare_observation(self, **options):
            calls.append(("observation", options))

    def decoder(target, draft, **options):
        assert target is model and isinstance(draft, Draft)
        calls.append(("decoder", options))
        return configured

    class Devices:
        def __getitem__(self, name):
            assert name == "CL"
            return SimpleNamespace(synchronize=lambda: calls.append(("synchronize", {})),
                                   allocator=SimpleNamespace(free_cache=lambda: calls.append(("free_cache", {}))))

    monkeypatch.setattr(tinygrad, "Device", Devices())
    monkeypatch.setattr(dflash2_config, "load_dflash2", load)
    monkeypatch.setattr(dflash2_selector, "CandidateSelector", lambda *args: selector)
    monkeypatch.setattr(dflash2, "DFlash2", Draft)
    monkeypatch.setattr(backend, "GreedySpeculativeDecoder", decoder)
    monkeypatch.setattr(speculation, "calibrate_ordinary_decode", calibrate)
    monkeypatch.setattr(speculation, "warm_prepared_decoder", warm)
    assert speculation.load_decoder(model, tmp_path / "target.gguf", str(tmp_path / "draft")) is configured
    assert configured.serving_setup == {"ordinary_decode_calibration": calibration, "prefill_chunk_size": 128, "prepared_warmup": {"requests": 3},
                                       "draft_mali": {"linear_layers": 19, "packed_weights": False, "prepared_shared_head": True,
                                                      "packed_fc": True, "packed_bytes": 48 * 1024**2}}
    assert calls == [("calibrate", {}), ("prepare", {"pack_weights": False, "prepare_shared_head": True, "pack_fc": True}),
                     ("proposals", {"max_context": 512}),
                     ("observation", {"max_context": 512, "pad_decode_rows": True, "native_cache_copy": True,
                                      "bound_prefix_memory": True, "retain_target_features": False}),
                     ("decoder", {"prepared_target": True, "prepared_prefill_dynamic": True,
                                  "prepared_dynamic_rows": True, "prepared_exact_replay": True, "prefix_trace": True,
                                  "early_anchor": True, "deferred_fallback": True, "reuse_prefix": True}),
                     ("warm", {}),
                     ("synchronize", {}), ("free_cache", {})]


@pytest.mark.parametrize("invalid_pin", [False, True])
def test_admission_checks_draft_before_stopping_existing_worker(pair, tmp_path, monkeypatch, invalid_pin):
    from onpu import runtime as serving

    model, options = pair
    store = Store(tmp_path / "data")
    runtime = Runtime(store, tmp_path)
    previous = {"id": "existing"}
    runtime.state, runtime.model = "ready", previous
    if invalid_pin:
        monkeypatch.setattr(speculation, "TARGET_SHA256", "0" * 64)
    monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 8 * 1024**3, "available": 1300 * 1024**2})
    monkeypatch.setattr(serving.hardware, "rss", lambda pid: 0)

    async def forbidden_stop():
        pytest.fail("Rejected admission must preserve the existing worker")

    monkeypatch.setattr(runtime, "_stop", forbidden_stop)
    try:
        with pytest.raises(EngineError, match="Target checkpoint" if invalid_pin else "available RAM") as error:
            asyncio.run(runtime.load(model, options))
        assert error.value.status == 400
        assert runtime.model is previous and runtime.state == "ready"
    finally:
        store.close()


def test_only_greedy_requests_select_loaded_draft():
    assert speculation.use_speculation(object(), 0.0)
    assert not speculation.use_speculation(object(), 0.6)
    assert not speculation.use_speculation(None, 0.0)


def test_relative_draft_path_is_pinned_to_the_directory_validated_before_worker_launch(pair, tmp_path, monkeypatch):
    from onpu import runtime as serving

    model, options = pair
    options = {**options, "draft_path": "draft"}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(serving.hardware, "memory", lambda: {"total": 16 * 1024**3, "available": 14 * 1024**3})
    monkeypatch.setattr(serving.hardware, "rss", lambda pid: 0)
    store = Store(tmp_path / "data")
    runtime = Runtime(store, tmp_path / "different-worker-directory")
    launches = []

    async def launch(*args, **kwargs):
        launches.append((args, kwargs))
        raise OSError("launch observed without starting a process")

    monkeypatch.setattr(serving.asyncio, "create_subprocess_exec", launch)
    try:
        with pytest.raises(EngineError, match="launch observed"):
            asyncio.run(runtime.load(model, options))
        args, kwargs = launches[0]
        assert args[args.index("--draft-path") + 1] == str(tmp_path / "draft")
        assert kwargs["cwd"] == str(runtime.backend)
        assert runtime.options["draft_path"] == str(tmp_path / "draft")
        assert options["draft_path"] == "draft"
    finally:
        store.close()


class Tokenizer:
    eos_id, eot_id, eom_id = 90, 91, None

    def __init__(self, pieces):
        self.pieces = pieces

    def is_end(self, token):
        return token in (self.eos_id, self.eot_id, self.eom_id)

    def stream_decoder(self):
        return lambda token=None: "" if token is None else self.pieces[token]


class OwnedTokens:
    def __init__(self, decoder, tokens, eos, cancelled):
        self.decoder, self.tokens, self.eos, self.cancelled = decoder, iter(tokens), eos, cancelled
        self.ended = self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.ended or self.cancelled():
            raise StopIteration
        self.decoder.busy = True
        token = next(self.tokens)
        self.decoder.delivered.append(token)
        self.ended = token in self.eos
        return token

    def close(self):
        self.closed = True
        self.decoder.busy = False
        self.decoder.closed += 1


class Decoder:
    def __init__(self, tokens):
        self.tokens, self.calls, self.delivered, self.closed, self.busy = tokens, [], [], 0, False

    def generate(self, ids, **kwargs):
        assert not self.busy
        self.calls.append((list(ids), kwargs))
        self.iterator = OwnedTokens(self, self.tokens[:kwargs["max_new_tokens"]], kwargs["eos_token_ids"], kwargs["cancelled"])
        return self.iterator


def handler_and_stream(pieces, tokens, *, limit=8, cancelled=lambda: False, reasoning=False):
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(tok=Tokenizer(pieces), model=SimpleNamespace(get_start_pos=lambda ids: 3))
    decoder = Decoder(tokens)
    stream = speculation.speculative_chunks(handler, decoder, [7, 8, 9], "target", max_tokens=limit, reasoning=reasoning,
                                           cancelled=cancelled, cancel_error=Cancelled)
    return handler, decoder, stream


@pytest.mark.parametrize("tokens,limit,finish,count", [([1, 2, 91, 3], 8, "stop", 2), ([90, 1], 8, "stop", 0),
                                                      ([1, 2, 3], 1, "length", 1), ([1, 2, 3], 3, "length", 3)])
def test_real_handler_eos_usage_limit_and_cleanup(tokens, limit, finish, count):
    handler, decoder, stream = handler_and_stream({1: "Hello", 2: " world", 3: "!"}, tokens, limit=limit)
    original_model = handler.server.model
    chunks = list(stream)
    assert chunks[-2]["choices"][0]["finish_reason"] == finish
    assert chunks[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": count, "total_tokens": 3 + count}
    assert decoder.calls[0][1]["eos_token_ids"] == {90, 91}
    assert decoder.closed == 1 and not decoder.busy
    assert handler.server.model is original_model


def test_real_handler_preserves_reasoning_and_tool_calls():
    pieces = {1: "<think>reason", 2: "</think>answer", 3: '<tool_call>{"name":"weather","arguments":{"city":"Taipei"}}</tool_call>'}
    _, decoder, stream = handler_and_stream(pieces, [1, 2, 3, 91])
    chunks = list(stream)
    deltas = [choice["delta"] for chunk in chunks for choice in chunk["choices"]]
    assert "".join(delta.get("reasoning_content", "") for delta in deltas) == "reason"
    assert "".join(delta.get("content", "") for delta in deltas) == "answer"
    calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
    assert calls[0]["function"] == {"name": "weather", "arguments": '{"city": "Taipei"}'}
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert decoder.closed == 1


@pytest.mark.parametrize("after_content", [False, True])
def test_disconnect_explicitly_closes_owned_source_and_allows_next_request(after_content):
    handler, decoder, stream = handler_and_stream({1: "hello", 2: "more"}, [1, 2])
    next(stream)
    if after_content:
        next(stream)
    stream.close()
    assert decoder.closed == 1 and not decoder.busy
    replacement = speculation.speculative_chunks(handler, decoder, [9], "target", max_tokens=1, reasoning=False,
                                                cancelled=lambda: False, cancel_error=Cancelled)
    assert list(replacement)[-1]["usage"]["completion_tokens"] == 1
    assert decoder.closed == 2


@pytest.mark.parametrize("before_first_token", [False, True])
def test_cancellation_never_emits_finish_or_usage_and_closes_before_acknowledgement(before_first_token):
    stopped = False
    _, decoder, stream = handler_and_stream({1: "hello", 2: "more"}, [1, 2], cancelled=lambda: stopped)
    next(stream)
    if not before_first_token:
        next(stream)
    stopped = True
    with pytest.raises(Cancelled):
        next(stream)
    assert decoder.closed == 1 and not decoder.busy


def test_formatter_failure_still_releases_decoder():
    _, decoder, stream = handler_and_stream({}, [1])
    next(stream)
    with pytest.raises(KeyError):
        next(stream)
    assert decoder.closed == 1 and not decoder.busy


def test_worker_cancellation_waits_for_draft_close_and_next_sampled_request_is_ordinary(tmp_path, monkeypatch):
    import tinygrad
    from tinygrad import nn
    from tinygrad.llm import cli
    from tinygrad.llm.kernels import mali
    from tinygrad.llm.model import Transformer

    from onpu import worker

    trace, events, ordinary = [], [], []
    content, cancel_sent, terminal = threading.Event(), threading.Event(), threading.Event()
    target = SimpleNamespace(max_context=512, _cached_tokens=[], warmup=lambda: None, get_start_pos=lambda ids: 2)

    def generate(ids, **options):
        if ids == [7, 8, 9]:
            ordinary.append(options["temperature"])
        return iter([1, 2, 91])

    target.generate = generate
    decoder = Decoder([1, 2, 91])
    original_generate = decoder.generate

    def owned_generate(*args, **kwargs):
        iterator = original_generate(*args, **kwargs)
        close = iterator.close

        def traced_close():
            trace.append("draft_closed")
            close()

        iterator.close = traced_close
        return iterator

    decoder.generate = owned_generate

    class Protocol:
        def write(self, line):
            event = json.loads(line)
            events.append(event)
            if (event.get("id") == "greedy" and event["event"] == "chunk"
                    and any(choice.get("delta", {}).get("content") for choice in event["data"].get("choices", []))):
                content.set()
                assert cancel_sent.wait(3), "Reader did not process cancellation"
            if event["event"] in ("done", "cancelled", "error"):
                trace.append(event["event"])
                terminal.set()

        def flush(self):
            pass

    class Input:
        def __iter__(self):
            body = {"model": "target", "max_tokens": 3, "temperature": 0.0, "messages": [],
                    "speculation": "force"}
            yield json.dumps({"op": "generate", "id": "greedy", "body": body})
            assert content.wait(3), "Worker did not emit content"
            yield json.dumps({"op": "cancel", "id": "greedy"})
            cancel_sent.set()
            assert terminal.wait(3), "Worker did not acknowledge cancellation"
            terminal.clear()
            yield json.dumps({"op": "generate", "id": "sampled", "body": {**body, "temperature": 0.6,
                                                                               "speculation": "auto"}})
            assert terminal.wait(3), "Worker did not finish the sampled request"
            yield json.dumps({"op": "shutdown"})

    monkeypatch.setattr(sys, "stdout", Protocol())
    monkeypatch.setattr(sys, "stdin", Input())
    monkeypatch.setattr(sys, "argv", ["worker", "--model", str(tmp_path / "target.gguf"), "--context", "512",
                                      "--speculation", speculation.MODE, "--draft-path", str(tmp_path)])
    monkeypatch.setenv("ONPU_PROFILE", "CL")
    monkeypatch.setenv("ONPU_NPU_PREFILL", "off")
    monkeypatch.setenv("ONPU_NPU_DECODE", "off")
    monkeypatch.setattr(tinygrad, "Device", SimpleNamespace(DEFAULT="CL"))
    monkeypatch.setattr(Transformer, "from_gguf", lambda *args, **kwargs: (target, {}))
    monkeypatch.setattr(nn.state, "get_parameters", lambda model: [SimpleNamespace(device="CL")])
    monkeypatch.setattr(cli.SimpleTokenizer, "from_gguf_kv", lambda metadata: Tokenizer({1: "hello", 2: " more"}))
    monkeypatch.setattr(mali, "configure", lambda *args, **kwargs: {"linear_layers": 0, "prefill_chunk": 128})
    monkeypatch.setattr(worker, "mali_profile", lambda *args, **kwargs: (speculation.TARGET_SHA256, {"recurrent_scan": True, "pack_weights": True}))
    monkeypatch.setattr(worker, "validate_speculation", lambda *args: None)
    monkeypatch.setattr(worker, "prepare_target_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "load_decoder", lambda *args: decoder)
    monkeypatch.setattr(worker, "prepare_prompt", lambda *args: ("prompt", [7, 8, 9]))
    worker.main()
    assert trace == ["draft_closed", "cancelled", "done"]
    assert len(decoder.calls) == 1 and ordinary == [0.6]
    starts = {event["id"]: event for event in events if event["event"] == "started"}
    assert starts["greedy"]["cached_tokens"] == 0
    assert starts["greedy"]["generation_mode"] == speculation.MODE
    assert starts["sampled"]["generation_mode"] == "ordinary"
    assert not any(event["event"] == "done" and event["id"] == "greedy" for event in events)
