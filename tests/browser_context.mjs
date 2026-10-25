// Read-only browser validation of model-specific context limits.
import fs from 'node:fs';
const base = process.argv[2] || 'http://127.0.0.1:8091';
const cdp = process.argv[3] || 'http://127.0.0.1:8093';
const output = process.argv[4] || '/home/orangepi/vlm-bench/fix-context-browser';
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
await until(() => evaluate(`document.querySelector('#connection-text')?.textContent==='Server online' && document.querySelectorAll('#model-library [data-id]').length>0`), 'model library');
const models = (await (await fetch(`${base}/api/models`)).json()).models;
const checks = [];
for (const model of models.filter(m => ['55a95dc3c1b93b72','b4b92c783d8643a1'].includes(m.id))) {
  // Loaded cards expose selection rather than loading; invoke the same load-dialog
  // click handler with a temporary button without unloading or submitting a load.
  const result = await evaluate(`(() => {
    const b=document.createElement('button');b.dataset.action='load';b.dataset.id=${JSON.stringify(model.id)};
    document.body.append(b);b.click();b.remove();
    const input=document.querySelector('#load-context');
    input.value=${model.context_limit};const acceptsMax=input.checkValidity();
    input.value=${model.context_limit+1};const rejectsOverflow=!input.checkValidity();
    input.value=${model.context_limit};
    const result={id:${JSON.stringify(model.id)},max:input.max,acceptsMax,rejectsOverflow,dialogOpen:document.querySelector('#model-dialog').open};
    document.querySelector('#model-dialog').close();return result;
  })()`);
  if (!result.acceptsMax || !result.rejectsOverflow || Number(result.max)!==model.context_limit || !result.dialogOpen) throw Error(JSON.stringify(result));
  checks.push(result);
}
if (checks.length!==2) throw Error('Expected both target models');
const settings = await evaluate(`(() => { const i=document.querySelector('#setting-context');i.value=40960;return {max:i.max,accepts40960:i.checkValidity()};})()`);
if (!settings.accepts40960) throw Error(JSON.stringify(settings));
await command('Page.navigate', {url:`${base}/#overview`});
await until(() => evaluate(`document.querySelector('#connection-text')?.textContent==='Server online'`), 'overview');
const screenshot=await command('Page.captureScreenshot', {format:'png'});
fs.writeFileSync(`${output}.png`,Buffer.from(screenshot.data,'base64'));
if (exceptions.length) throw Error(JSON.stringify(exceptions));
const report={complete:true,checks,settings,exceptions,mutatedServerSettings:false};
fs.writeFileSync(`${output}.json`,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));socket.close();
