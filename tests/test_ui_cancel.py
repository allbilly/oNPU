"""Exercise the browser's actual generation and Stop handlers without a server."""

import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "onpu" / "static" / "app.js"
NODE = shutil.which("node")
HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
const element = (id) => {
  if (!elements.has(id)) elements.set(id, {
    value: '', checked: false, style: {}, dataset: {},
    classList: { add() {}, remove() {}, toggle() {} }, focus() {},
    querySelector: (selector) => element(`${id} ${selector}`),
    removeAttribute(name) { delete this[name]; },
  });
  return elements.get(id);
};
function deferred() {
  let resolve, reject;
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}
const requests = [], stops = [];
const workers = { gpu: { cancelled: false, active: null }, npu: { cancelled: false, active: 'other-tab-npu' } };
let selected = 'gpu', delayStop = false, delayNetwork = false, failStop = false, pendingStop;
let delayHeaders = false, omitRequestId = false;
const imageReady = deferred();
const context = vm.createContext({
  document: { getElementById: element, querySelectorAll: () => [],
    createElement: () => ({ getContext: () => ({ fillRect() {}, drawImage() {} }),
      toDataURL: () => 'data:image/jpeg;base64,new-image' }),
    addEventListener() {}, documentElement: { dataset: {} } },
  window: { addEventListener() {} }, location: { origin: 'http://board', hash: '#chat' },
  localStorage: { getItem: () => null, setItem() {} }, navigator: {},
  AbortController, TextDecoder, Uint8Array, setTimeout: () => 0,
  clearTimeout() {}, setInterval() {},
  createImageBitmap: () => imageReady.promise,
  fetch: async (path, options) => {
    // Leave automatic initialization pending; tests explicitly control model status.
    if (path === '/api/settings') return new Promise(() => {});
    if (path === '/v1/chat/completions') {
      const pendingRead = deferred(), pendingHeaders = deferred();
      const request = { ...JSON.parse(options.body), id: `request/${requests.length + 1}?tab=A&owner=GPU`,
        signal: options.signal, end: () => {
          if (workers[request.model].active === request.id) workers[request.model].active = null;
          pendingRead.resolve({ done: true });
        } };
      // The pre-header case has no reader yet, but the same abort also rejects fetch.
      pendingRead.promise.catch(() => {});
      options.signal.addEventListener('abort', () => {
        const error = new Error('aborted'); error.name = 'AbortError';
        pendingRead.reject(error);
        if (!request.headersDelivered) pendingHeaders.reject(error);
      }, { once: true });
      requests.push(request);
      workers[request.model].active = request.id;
      if (delayHeaders) await pendingHeaders.promise;
      request.headersDelivered = true;
      return { ok: true, headers: { get: (name) => name === 'X-Request-ID' && !omitRequestId ? request.id : null },
        body: { getReader: () => ({ read: () => pendingRead.promise }) } };
    }
    if (path.startsWith('/api/chat/stop')) {
      stops.push(path);
      if (delayNetwork) { pendingStop = deferred(); await pendingStop.promise; }
      const params = new URL(path, 'http://board').searchParams;
      const model = params.get('model') || selected, key = params.get('request_id');
      // Runtime.cancel(key) must only cancel the active request matching that key.
      if (!key || workers[model].active === key) workers[model].cancelled = true;
      if (failStop) throw Error('Stop endpoint unavailable');
      if (delayStop && !delayNetwork) { pendingStop = deferred(); await pendingStop.promise; }
      return { ok: true, json: async () => ({ ok: true }) };
    }
    throw Error(`Unexpected request: ${path}`);
  },
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
// Keep rendering and background polling outside this request-ownership unit test.
vm.runInContext(`
  renderChat = () => {};
  refreshStatus = async () => {};
  refreshAll = async () => {};
  persistHistory = () => {};
  histories = { gpu: [{role: 'user', content: 'GPU prompt'}],
                npu: [{role: 'user', content: 'NPU prompt'}] };
`, context);
function select(model) {
  selected = model;
  context.model = model;
  vm.runInContext('status = {runtime: {model_id: model, state: "ready"}}', context);
}
async function start(model) {
  select(model);
  const promise = vm.runInContext('generateReply()', context);
  // Drain cross-context promise continuations through headers and into the read.
  await new Promise((resolve) => setImmediate(resolve));
  return { promise, request: requests.at(-1) };
}
const stop = () => element('stop-generation').onclick({ preventDefault() {} });
async function main() {
  const scenario = process.argv[2];
  if (scenario.startsWith('image-')) {
    select('gpu');
    if (scenario === 'image-remove') vm.runInContext("setPendingImage('data:image/jpeg;base64,old-image')", context);
    element('chat-image').files = [{ type: 'image/jpeg', size: 100 }];
    const loading = element('chat-image').onchange({ preventDefault() {} });
    if (scenario === 'image-new-chat') element('new-chat').onclick();
    if (scenario === 'image-remove') element('remove-chat-image').onclick();
    imageReady.resolve({ width: 512, height: 512, close() {} });
    await loading;
    const attached = vm.runInContext('pendingImage', context);
    if (scenario === 'image-attach') assert.equal(attached, 'data:image/jpeg;base64,new-image');
    else assert.equal(attached, null, 'A discarded image must not reappear after decoding finishes');
    return;
  }
  if (scenario === 'idle') {
    select('npu');
    await stop();
    assert.equal(stops.length, 0, 'An idle tab must not cancel another tab\'s request');
    assert.equal(workers.npu.cancelled, false);
    return;
  }
  delayHeaders = scenario === 'before-headers';
  omitRequestId = scenario === 'missing-request-id';
  const first = await start('gpu');
  assert.equal(first.request.model, 'gpu');
  // Tab B changes server selection; Tab A's status poll observes that selection.
  select('npu');
  if (delayHeaders || omitRequestId) {
    await stop();
    assert.equal(stops.length, 0, 'Without a request ID, Stop must never issue model-only cancellation');
    assert.equal(first.request.signal.aborted, true, 'Stop must abort its originating fetch');
    await first.promise;
    assert.equal(workers.gpu.cancelled, false);
    assert.equal(workers.npu.cancelled, false);
    return;
  }
  if (scenario === 'same-model-delayed-network') delayNetwork = true;
  if (scenario === 'delayed-stop') delayStop = true;
  if (scenario === 'stop-error') failStop = true;
  const stopping = stop();
  if (delayNetwork) {
    first.request.end();
    await first.promise;
    // Another tab starts a new request on the SAME worker before Stop reaches it.
    workers.gpu.active = 'other-tab-new-gpu-request';
    pendingStop.resolve();
    await stopping;
    assert.equal(workers.gpu.active, 'other-tab-new-gpu-request');
    assert.equal(workers.gpu.cancelled, false, 'Stale Stop must preserve a newer request on the same model');
    assert.equal(first.request.signal.aborted, true);
    assert.equal(stops.length, 1);
    assert.equal(new URL(stops[0], 'http://board').searchParams.get('request_id'), first.request.id);
    return;
  }
  if (delayStop) {
    // A completes and a new request starts before the old Stop response returns.
    first.request.end();
    await first.promise;
    const second = await start('npu');
    pendingStop.resolve();
    await stopping;
    assert.equal(first.request.signal.aborted, true, 'Stop must retain its original controller');
    assert.equal(second.request.signal.aborted, false, 'Late Stop must not abort the next request');
    second.request.end();
    await second.promise;
  } else {
    await stopping;
    assert.equal(first.request.signal.aborted, true, 'Stop must disconnect its own stream even on error');
    await first.promise;
  }
  assert.equal(stops.length, 1);
  assert.equal(new URL(stops[0], 'http://board').searchParams.get('model'), 'gpu');
  assert.equal(new URL(stops[0], 'http://board').searchParams.get('request_id'), first.request.id);
  assert.equal(workers.gpu.cancelled, true, 'GPU origin must receive cancellation');
  assert.equal(workers.npu.cancelled, false, 'Independent NPU request must remain running');
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.skipif(NODE is None, reason="Node.js is required for browser JavaScript unit tests")
@pytest.mark.parametrize("scenario", ["selection-change", "delayed-stop", "stop-error", "idle", "same-model-delayed-network",
                                      "before-headers", "missing-request-id"])
def test_stop_owns_its_generation(scenario):
    result = subprocess.run([NODE, "-e", HARNESS, str(APP_JS), scenario], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(NODE is None, reason="Node.js is required for browser JavaScript unit tests")
@pytest.mark.parametrize("scenario", ["image-attach", "image-new-chat", "image-remove"])
def test_image_decode_respects_composer_changes(scenario):
    result = subprocess.run([NODE, "-e", HARNESS, str(APP_JS), scenario], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
