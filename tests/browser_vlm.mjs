// Requires an already running local Chromium CDP endpoint; no npm dependencies.
import fs from 'node:fs';
const base = process.argv[2] || 'http://127.0.0.1:8092';
const cdp = process.argv[3] || 'http://127.0.0.1:8093';
const imagePath = process.argv[4] || '/home/orangepi/rkllm/examples/multimodal_model_demo/data/demo.jpg';
const output = process.argv[5] || '/home/orangepi/vlm-bench/browser-vlm';
fs.writeFileSync(`${output}.json`, JSON.stringify({ complete: false }, null, 2));
const model = process.argv[6] || 'b4b92c783d8643a1';
if (!/^[a-f0-9]+$/.test(model)) throw Error('Invalid local model ID');
const loadSelector = `[data-action="load"][data-id="${model}"]`;
const targets = await (await fetch(`${cdp}/json/list`)).json();
const target = targets.find((t) => t.type === 'page');
if (!target) throw Error('No Chromium page target');
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let sequence = 0;
const pending = new Map(), exceptions = [];
socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
  if (!message.id) return;
  const entry = pending.get(message.id); if (!entry) return;
  pending.delete(message.id); clearTimeout(entry.timer);
  if (message.error) entry.reject(Error(JSON.stringify(message.error))); else entry.resolve(message.result);
};
function command(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve, reject, timer: setTimeout(() => reject(Error(`CDP timeout: ${method}`)), 30000) });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const value = await command('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (value.exceptionDetails) throw Error(value.exceptionDetails.exception?.description || value.exceptionDetails.text);
  return value.result.value;
}
async function until(check, label, seconds = 60) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (await check()) return;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw Error(`Timed out: ${label}`);
}
await command('Page.enable'); await command('Runtime.enable');
await command('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
await command('Page.navigate', { url: `${base}/#models` });
await until(() => evaluate(`Boolean(document.querySelector('${loadSelector}'))`), 'vision model library');
await evaluate(`document.querySelector('${loadSelector}').click(); document.querySelector('#load-profile').value='CL'; document.querySelector('#load-profile').dispatchEvent(new Event('change')); if(!document.querySelector('#load-vision-gpu').closest('label').hidden) throw Error('Vision GPU control should be hidden for GPU-only profile'); document.querySelector('#load-profile').value='ROCKCHIP'; document.querySelector('#load-profile').dispatchEvent(new Event('change')); if(document.querySelector('#load-vision-gpu').closest('label').hidden || !document.querySelector('#load-vision-gpu').checked) throw Error('Vision GPU choice missing'); document.querySelector('#load-context').value=512; document.querySelector('#load-form').requestSubmit();`);
let latest;
await until(async () => {
  latest = await (await fetch(`${base}/api/status`)).json();
  if (latest.runtime.state === 'error') throw Error(latest.runtime.message);
  return latest.runtime.state === 'ready' && latest.runtime.supports_images;
}, 'vision worker ready', 480);
console.log(JSON.stringify({ event: 'ready', device: latest.runtime.actual_device, placement: latest.runtime.placement }));
if (!latest.runtime.actual_device.includes('CL') || !latest.runtime.placement?.vision?.gpu_vision || latest.runtime.placement.vision.prepared_mlp !== 12) throw Error('Prepared GPU/NPU vision path was not selected');
await evaluate(`location.hash='chat'; document.querySelector('#chat-temperature').value=0; document.querySelector('#chat-max-tokens').value=32;`);
await until(() => evaluate(`!document.querySelector('#attach-image').hidden`), 'image composer');
await evaluate(`document.querySelector('#new-chat').click();`);
const document = await command('DOM.getDocument');
const input = await command('DOM.querySelector', { nodeId: document.root.nodeId, selector: '#chat-image' });
await command('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [imagePath] });
await until(() => evaluate(`!document.querySelector('#chat-image-preview').hidden`), 'image preview');
await evaluate(`document.querySelector('#chat-input').value='What is shown in this image?'; document.querySelector('#chat-form').requestSubmit();`);
await until(() => evaluate(`document.querySelectorAll('.message.assistant').length>0 && document.querySelector('#stop-generation').hidden`), 'image answer', 180);
const answer = await evaluate(`({text:[...document.querySelectorAll('.message.assistant .message-content')].at(-1)?.textContent,error:[...document.querySelectorAll('.message-error')].map((n)=>n.textContent),images:document.querySelectorAll('.message-image').length})`);
if (answer.error.length || !/astronaut/i.test(answer.text || '') || answer.images !== 1) throw Error(`Invalid image response: ${JSON.stringify(answer)}`);
latest = await (await fetch(`${base}/api/status`)).json();
const screenshot = await command('Page.captureScreenshot', { format: 'png' });
fs.writeFileSync(`${output}.png`, Buffer.from(screenshot.data, 'base64'));
if (exceptions.length) { socket.close(); throw Error(JSON.stringify(exceptions)); }
fs.writeFileSync(`${output}.json`, JSON.stringify({ complete: true, answer, runtime: latest.runtime, exceptions }, null, 2));
console.log(JSON.stringify({ event: 'verified', answer, metrics: latest.runtime.last_metrics, exceptions: exceptions.length }));
socket.close();
