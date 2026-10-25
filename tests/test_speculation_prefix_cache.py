"""CPU fake tests for the worker-to-formatter speculative prefix cache bridge."""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from onpu.speculation import MODE, TARGET_SHA256, speculative_chunks

PROMPT_IDS = [7, 8, 9, 10]
OUTPUT_IDS = [11, 12, 91]


class FakeTokenizer:
    bos_id, eos_id, eot_id, eom_id = 0, 91, 90, None
    preset = 'qwen2'

    def decode(self, ids):
        pieces = {0: '', 11: 'hello', 12: ' world', 90: '<eot>', 91: '<eos>'}
        return ''.join(pieces[token] for token in ids)

    def stream_decoder(self):
        return lambda token=None: '' if token is None else self.decode([token])

    def is_end(self, token):
        return token in (self.eos_id, self.eot_id, self.eom_id)


class FakeSource:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.tokens)

    def close(self):
        self.closed = True
        close = getattr(self.tokens, 'close', None)
        if callable(close):
            close()


@dataclass
class FakeStats:
    proposed_tokens: int = 2
    accepted_tokens: int = 1
    output_tokens: int = 0
    fallback_tokens: int = 0
    fallback_calls: int = 0
    blocks: int = 1
    snapshot_calls: int = 1
    prompt_observe_seconds: float = 0.01


class FakeDraftMetrics:
    def proposal_preparation_metadata(self):
        return {'enabled': True, 'graphs': 1}

    def observation_preparation_metadata(self):
        return {'enabled': True, 'graphs': 1}


class FakeDecoder:
    ordinary_decode_seconds = 0.25

    def __init__(self, model, count_override='model', *, expose_get_start=True, fail_get_start=False,
                 ignore_max_new_tokens=False, reuse_prefix=True):
        self.model = model
        self.count_override = count_override
        self.fail_get_start = fail_get_start
        self.ignore_max_new_tokens = ignore_max_new_tokens
        self.reuse_prefix = reuse_prefix
        self.get_start_queries = []
        self.generate_calls = []
        self.sources = []
        self.normal_finishes = 0
        self.interrupted_generations = 0
        self.stats = FakeStats()
        self.draft = FakeDraftMetrics()
        self.serving_setup = {'ordinary_decode_calibration': {'seconds_per_token': 0.25}}
        if expose_get_start:
            self.get_start_pos = self._get_start_pos

    def _get_start_pos(self, ids):
        self.get_start_queries.append((list(ids), list(self.model._cached_tokens)))
        if self.fail_get_start:
            raise AssertionError('Automatic cached-prefix routing queried the decoder')
        if self.count_override == 'model':
            return self.model.get_start_pos(ids)
        return self.count_override

    def generate(self, ids, **kwargs):
        self.generate_calls.append((list(ids), dict(kwargs)))
        values = OUTPUT_IDS if self.ignore_max_new_tokens else OUTPUT_IDS[:kwargs.get('max_new_tokens', len(OUTPUT_IDS))]

        def stream():
            completed = False
            try:
                yield from values
                completed = True
            finally:
                if completed:
                    self.normal_finishes += 1
                else:
                    self.interrupted_generations += 1

        source = FakeSource(stream())
        self.sources.append(source)
        return source

    def sample(self, ids, *, temperature, **kwargs):
        return self.generate(ids, temperature=temperature, **kwargs)


class FakeModel:
    max_context = 512

    def __init__(self):
        self._cached_tokens = []
        self.generate_calls = []
        self.warmup_calls = 0

    def warmup(self):
        self.warmup_calls += 1

    def get_start_pos(self, ids):
        count = 0
        for left, right in zip(ids, self._cached_tokens):
            if left != right:
                break
            count += 1
        return count

    def generate(self, ids, **kwargs):
        self.generate_calls.append((list(ids), dict(kwargs)))
        yield from OUTPUT_IDS


class WorkerRun:
    def __init__(self, events, model, decoder):
        self.events, self.model, self.decoder = events, model, decoder


def run_worker_case(monkeypatch, tmp_path, *, request_id, speculation, cache_prompt=True,
                    prefix_tokens=2, decoder_count='model', expose_get_start=True,
                    fail_get_start=False, stats_count=None):
    import tinygrad
    from tinygrad import nn
    from tinygrad.llm import cli
    from tinygrad.llm.kernels import mali
    from tinygrad.llm.model import Transformer

    from onpu import worker

    model = FakeModel()
    decoder = FakeDecoder(model, decoder_count, expose_get_start=expose_get_start,
                          fail_get_start=fail_get_start)
    if stats_count is not None:
        decoder.stats.cached_tokens = stats_count
    events = []
    terminal = threading.Event()

    class Protocol:
        def write(self, line):
            event = json.loads(line)
            events.append(event)
            if event.get('event') in ('done', 'error', 'cancelled'):
                terminal.set()

        def flush(self):
            pass

    class Input:
        def __iter__(self):
            body = {'model': 'target', 'messages': [{'role': 'user', 'content': request_id}],
                    'max_tokens': 3, 'temperature': 0.0, 'speculation': speculation,
                    'cache_prompt': cache_prompt}
            yield json.dumps({'op': 'generate', 'id': request_id, 'body': body})
            assert terminal.wait(3), 'Worker did not complete the fake request'
            yield json.dumps({'op': 'shutdown'})

    def prepare_prompt(body, *_args):
        model._cached_tokens = PROMPT_IDS[:prefix_tokens]
        return 'prompt', PROMPT_IDS.copy()

    monkeypatch.setattr(sys, 'stdout', Protocol())
    monkeypatch.setattr(sys, 'stdin', Input())
    monkeypatch.setattr(sys, 'argv', ['worker', '--model', str(tmp_path / 'target.gguf'), '--context', '512',
                                      '--speculation', MODE, '--draft-path', str(tmp_path)])
    monkeypatch.setenv('ONPU_PROFILE', 'CL')
    monkeypatch.setenv('ONPU_NPU_PREFILL', 'off')
    monkeypatch.setenv('ONPU_NPU_DECODE', 'off')
    monkeypatch.setattr(tinygrad, 'Device', SimpleNamespace(DEFAULT='CL'))
    monkeypatch.setattr(Transformer, 'from_gguf', lambda *args, **kwargs: (model, {'general.name': 'fake'}))
    monkeypatch.setattr(nn.state, 'get_parameters', lambda _model: [SimpleNamespace(device='CL')])
    monkeypatch.setattr(cli.SimpleTokenizer, 'from_gguf_kv', lambda _metadata: FakeTokenizer())
    monkeypatch.setattr(mali, 'configure', lambda *args, **kwargs: {'linear_layers': 0, 'prefill_chunk': 128})
    monkeypatch.setattr(worker, 'mali_profile', lambda *args, **kwargs: (TARGET_SHA256,
        {'recurrent_scan': True, 'pack_weights': True}))
    monkeypatch.setattr(worker, 'validate_speculation', lambda *args: None)
    monkeypatch.setattr(worker, "prepare_target_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, 'load_decoder', lambda *args: decoder)
    monkeypatch.setattr(worker, 'prepare_prompt', prepare_prompt)
    worker.main()
    return WorkerRun(events, model, decoder)


def test_forced_prefix_reuse_bridge_preserves_full_prompt_usage_and_formatter(monkeypatch, tmp_path):
    result = run_worker_case(monkeypatch, tmp_path, request_id='force-prefix', speculation='force')
    started = next(event for event in result.events if event.get('event') == 'started')
    done = next(event for event in result.events if event.get('event') == 'done')
    routing = started['speculation_routing']
    assert routing['selected'] == MODE and routing['requested'] == 'force'
    assert routing['speculative_cached_tokens'] == 2
    assert routing == done['metrics']['speculation_routing']
    assert started['cached_tokens'] == done['metrics']['cached_tokens'] == 2
    assert started['generation_mode'] == done['metrics']['generation_mode'] == MODE
    assert result.decoder.get_start_queries == [(PROMPT_IDS, PROMPT_IDS[:2])]
    assert result.decoder.generate_calls[0][0] == PROMPT_IDS
    assert result.decoder.generate_calls[0][1]['max_new_tokens'] == 3
    usage = next(event['data']['usage'] for event in result.events
                 if event.get('event') == 'chunk' and event.get('data', {}).get('usage'))
    assert usage == {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}
    text = ''.join(choice.get('delta', {}).get('content', '')
                   for event in result.events if event.get('event') == 'chunk'
                   for choice in event['data'].get('choices', []))
    assert text == 'hello world'
    assert result.decoder.sources[0].closed
    assert result.decoder.normal_finishes == 1


def test_cache_prompt_false_clears_marker_before_decoder_prefix_query(monkeypatch, tmp_path):
    result = run_worker_case(monkeypatch, tmp_path, request_id='uncached-force', speculation='force',
                             cache_prompt=False)
    started = next(event for event in result.events if event.get('event') == 'started')
    done = next(event for event in result.events if event.get('event') == 'done')
    query_ids, marker_at_query = result.decoder.get_start_queries[0]
    assert query_ids == PROMPT_IDS
    assert marker_at_query == []
    assert started['speculation_routing']['speculative_cached_tokens'] == 0
    assert done['metrics']['speculation_routing']['speculative_cached_tokens'] == 0
    assert started['cached_tokens'] == done['metrics']['cached_tokens'] == 0
    usage = next(event['data']['usage'] for event in result.events
                 if event.get('event') == 'chunk' and event.get('data', {}).get('usage'))
    assert usage['prompt_tokens'] == len(PROMPT_IDS)
    assert result.decoder.sources[0].closed


def test_legacy_decoder_without_prefix_query_defaults_to_zero(monkeypatch, tmp_path):
    result = run_worker_case(monkeypatch, tmp_path, request_id='legacy-decoder', speculation='force',
                             expose_get_start=False)
    started = next(event for event in result.events if event.get('event') == 'started')
    done = next(event for event in result.events if event.get('event') == 'done')
    assert started['speculation_routing']['speculative_cached_tokens'] == 0
    assert done['metrics']['speculation_routing']['speculative_cached_tokens'] == 0
    assert result.decoder.generate_calls[0][0] == PROMPT_IDS
    assert result.decoder.sources[0].closed


@pytest.mark.parametrize('reported', [True, -1, 3, len(PROMPT_IDS)])
def test_forced_speculative_prefix_counts_outside_ordinary_cache_are_rejected_before_generation(
        monkeypatch, tmp_path, reported):
    result = run_worker_case(monkeypatch, tmp_path, request_id='bad-count-' + str(reported),
                             speculation='force', decoder_count=reported)
    errors = [event for event in result.events if event.get('event') == 'error']
    assert len(errors) == 1
    assert not result.decoder.generate_calls
    assert not result.decoder.sources
    assert not any(event.get('event') == 'started' for event in result.events)


def test_auto_cached_prefix_stays_ordinary_without_querying_decoder(monkeypatch, tmp_path):
    result = run_worker_case(monkeypatch, tmp_path, request_id='auto-cached', speculation='auto',
                             fail_get_start=True)
    started = next(event for event in result.events if event.get('event') == 'started')
    done = next(event for event in result.events if event.get('event') == 'done')
    assert started['speculation_routing']['selected'] == 'ordinary'
    assert started['speculation_routing']['reason'] == 'cached_prefix'
    assert 'speculative_cached_tokens' not in started['speculation_routing']
    assert done['metrics']['speculation_routing'] == started['speculation_routing']
    assert not result.decoder.get_start_queries and not result.decoder.generate_calls
    assert result.model.generate_calls and result.model.generate_calls[-1][0] == PROMPT_IDS
    usage = next(event['data']['usage'] for event in result.events
                 if event.get('event') == 'chunk' and event.get('data', {}).get('usage'))
    assert usage['prompt_tokens'] == len(PROMPT_IDS)


def test_worker_does_not_publish_done_when_actual_cached_count_disagrees(monkeypatch, tmp_path):
    result = run_worker_case(monkeypatch, tmp_path, request_id='count-mismatch', speculation='force', stats_count=1)
    errors = [event for event in result.events if event.get('event') == 'error']
    assert len(errors) == 1 and errors[0]['status'] == 500
    assert not any(event.get('event') == 'done' for event in result.events)
    assert result.model._cached_tokens == []
    assert result.decoder.sources[0].closed


def test_speculative_chunks_formatter_receives_actual_cache_count_and_full_prompt_usage():
    from tinygrad.llm.serve import Handler

    model = SimpleNamespace()
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(model=model, tok=FakeTokenizer())
    decoder = FakeDecoder(model, 2)
    chunks = list(speculative_chunks(handler, decoder, PROMPT_IDS, 'target', max_tokens=3,
                                     reasoning=False, cancelled=lambda: False, cancel_error=RuntimeError,
                                     cached_tokens=2))
    usage = next(chunk['usage'] for chunk in chunks if chunk.get('usage'))
    assert usage == {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}
    assert decoder.generate_calls[0][0] == PROMPT_IDS
    assert decoder.sources[0].closed
    assert decoder.normal_finishes == 1


def test_one_token_length_finish_resumes_normal_cache_finalizer():
    from tinygrad.llm.serve import Handler

    decoder = FakeDecoder(SimpleNamespace(), 0, reuse_prefix=True)
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(model=None, tok=FakeTokenizer())
    chunks = list(speculative_chunks(handler, decoder, PROMPT_IDS, 'target', max_tokens=1,
                                     reasoning=False, cancelled=lambda: False, cancel_error=RuntimeError))
    usage = next(chunk['usage'] for chunk in chunks if chunk.get('usage'))
    assert usage['completion_tokens'] == 1
    assert decoder.normal_finishes == 1 and decoder.interrupted_generations == 0
    assert decoder.sources[0].closed


@pytest.mark.parametrize('cached', [True, -1, 4, 5])
def test_speculative_chunks_rejects_invalid_cache_count_before_source_creation(cached):
    from tinygrad.llm.serve import Handler

    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(tok=FakeTokenizer())
    model = SimpleNamespace()
    decoder = FakeDecoder(model, 0)
    chunks = speculative_chunks(handler, decoder, PROMPT_IDS, 'target', max_tokens=3,
                                reasoning=False, cancelled=lambda: False, cancel_error=RuntimeError,
                                cached_tokens=cached)
    with pytest.raises(ValueError, match='Invalid speculative cached-token count'):
        next(chunks)
    assert not decoder.generate_calls and not decoder.sources


@pytest.mark.parametrize('visible_chunks', [1, 2])
def test_prefix_reuse_formatter_interruption_does_not_finalize_cache(visible_chunks):
    from tinygrad.llm.serve import Handler

    decoder = FakeDecoder(SimpleNamespace(), 0, reuse_prefix=True)
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(model=None, tok=FakeTokenizer())
    stream = speculative_chunks(handler, decoder, PROMPT_IDS, 'target', max_tokens=3,
                                reasoning=False, cancelled=lambda: False, cancel_error=RuntimeError,
                                cached_tokens=0)
    for _ in range(visible_chunks):
        next(stream)
    stream.close()
    assert decoder.sources and decoder.sources[0].closed
    assert decoder.normal_finishes == 0


def test_synchronous_formatter_construction_error_closes_owned_source():
    decoder = FakeDecoder(SimpleNamespace(), 0, reuse_prefix=True)

    class BrokenHandler:
        server = SimpleNamespace(tok=FakeTokenizer())

        def run_model(self, *args, **kwargs):
            raise ValueError('formatter setup failed')

    stream = speculative_chunks(BrokenHandler(), decoder, PROMPT_IDS, 'target', max_tokens=3,
                                reasoning=False, cancelled=lambda: False, cancel_error=RuntimeError,
                                cached_tokens=0)
    with pytest.raises(ValueError, match='formatter setup failed'):
        next(stream)
    assert decoder.sources and decoder.sources[0].closed
    assert decoder.normal_finishes == 0


def test_formatter_resume_rejects_unconsumed_extra_token_for_reuse_decoder():
    from tinygrad.llm.serve import Handler

    decoder = FakeDecoder(SimpleNamespace(), 0, ignore_max_new_tokens=True, reuse_prefix=True)
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(model=None, tok=FakeTokenizer())
    stream = speculative_chunks(handler, decoder, PROMPT_IDS, 'target', max_tokens=1,
                                reasoning=False, cancelled=lambda: False, cancel_error=RuntimeError,
                                cached_tokens=0)
    with pytest.raises(RuntimeError, match='Formatter ended before the bounded speculative iterator'):
        list(stream)
    assert decoder.sources and decoder.sources[0].closed
    assert decoder.normal_finishes == 0


def test_legacy_decoder_without_reuse_flag_keeps_prior_formatter_stop_semantics():
    from tinygrad.llm.serve import Handler

    decoder = FakeDecoder(SimpleNamespace(), 0, ignore_max_new_tokens=True)
    del decoder.reuse_prefix
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(model=None, tok=FakeTokenizer())
    stream = speculative_chunks(handler, decoder, PROMPT_IDS, 'target', max_tokens=1,
                                reasoning=False, cancelled=lambda: False, cancel_error=RuntimeError,
                                cached_tokens=0)
    chunks = list(stream)
    assert chunks and decoder.sources and decoder.sources[0].closed
    assert decoder.normal_finishes == 0
    text = ''.join(piece
                   for chunk in chunks for choice in chunk.get('choices', [])
                   for key in ('content', 'reasoning_content')
                   if (piece := choice.get('delta', {}).get(key)))
    assert text == 'hello'


def test_pinned_template_keeps_generated_empty_thinking_header_in_next_turn():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from onpu.speculation import prefix_cache_template

    source = (
        "{%- set ns = namespace(last_query_index=messages|length-1) %}"
        "{%- for message in messages %}"
        "{%- if message.role == 'assistant' %}"
        "{%- if loop.index0 > ns.last_query_index %}"
        "{{- '<assistant>\\n<think>\\n\\n</think>\\n\\n' + message.content }}"
        "{%- else %}{{- '<assistant>\\n' + message.content }}{%- endif %}"
        "{%- else %}{{- '<user>\\n' + message.content }}{%- endif %}"
        "{{- '<end>\\n' }}{%- endfor %}"
        "{{- '<assistant>\\n<think>\\n\\n</think>\\n\\n' }}"
    )
    original = ImmutableSandboxedEnvironment().from_string(source)
    adapted_source = prefix_cache_template(source, MODE, checkpoint_sha256=TARGET_SHA256)
    adapted = ImmutableSandboxedEnvironment().from_string(adapted_source)
    first = [{'role': 'user', 'content': 'Count'}]
    prefix = original.render(messages=first, preserve_thinking=True) + '1'
    following = first + [{'role': 'assistant', 'content': '1'}, {'role': 'user', 'content': 'Continue'}]
    assert not original.render(messages=following, preserve_thinking=True).startswith(prefix)
    assert adapted.render(messages=following, preserve_thinking=True).startswith(prefix)
    assert adapted.render(messages=first, preserve_thinking=True) == original.render(messages=first, preserve_thinking=True)
    assert adapted.render(messages=following, preserve_thinking=False) == original.render(messages=following, preserve_thinking=False)
    assert prefix_cache_template(adapted_source, MODE, checkpoint_sha256=TARGET_SHA256) == adapted_source


def test_template_adaptation_is_limited_to_the_pinned_experimental_pair():
    from onpu.speculation import prefix_cache_template

    source = 'arbitrary unrelated template'
    assert prefix_cache_template(source, 'off', checkpoint_sha256=None) == source
    with pytest.raises(ValueError, match='pinned target'):
        prefix_cache_template(source, MODE, checkpoint_sha256='wrong')
    with pytest.raises(ValueError, match='Unsupported'):
        prefix_cache_template(source, MODE, checkpoint_sha256=TARGET_SHA256)
