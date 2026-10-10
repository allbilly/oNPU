'use strict';
const $ = (id) => document.getElementById(id);
const paths = {
  overview: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  models: '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  chat: '<path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z"/>',
  benchmark: '<path d="M4 20h16M7 16V9m5 7V4m5 12v-5"/>',
  settings: '<path d="m9.2 3-.6 2.2-1.7 1-2.2-.6-2.8 4.8 1.6 1.6v2l-1.6 1.6 2.8 4.8 2.2-.6 1.7 1 .6 2.2h5.6l.6-2.2 1.7-1 2.2.6 2.8-4.8-1.6-1.6v-2l1.6-1.6-2.8-4.8-2.2.6-1.7-1L14.8 3Z"/><circle cx="12" cy="12" r="3"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 3v3m6-3v3m-6 12v3m6-3v3M3 9h3m-3 6h3m12-6h3m-3 6h3"/>',
  memory: '<rect x="3" y="6" width="18" height="11" rx="2"/><path d="M7 10v3m5-3v3m5-3v3M7 17v3m5-3v3m5-3v3"/>',
  temperature: '<path d="M14 14.7V5a2 2 0 0 0-4 0v9.7a4 4 0 1 0 4 0Z"/><path d="M12 8v8"/>',
  timer: '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l3 2M9 2h6M12 2v3m6 2 2-2"/>',
  bolt: '<path d="m13 2-9 12h7l-1 8 10-12h-7l1-8Z"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  'arrow-up': '<path d="M12 19V5m-5 5 5-5 5 5"/>',
  terminal: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="m7 9 3 3-3 3m6 0h4"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>',
  activity: '<path d="M2 12h5l3-9 4 18 3-9h5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  refresh: '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 6.1A8 8 0 0 1 20 12M4 12a8 8 0 0 0 13.9 5.9"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/>',
  sliders: '<path d="M4 7h7m4 0h5M4 17h3m4 0h9"/><circle cx="13" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  play: '<path d="m7 4 14 8-14 8V4Z"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.chip}</svg>`;
document.querySelectorAll('[data-icon]').forEach((el) => { el.innerHTML = icon(el.dataset.icon); });
const escape = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const bytes = (n, precision = 1) => n == null ? '—' : n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(precision)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`;
const number = (n, suffix = '', precision = 1) => n == null ? '—' : `${n.toFixed(precision)}${suffix}`;
const badge = (state, label) => `<span class="badge ${escape(state)}">${escape(label || state)}</span>`;
let status = null, modelList = [], catalog = [], settings = {}, jobs = [], benchmarks = [], metrics = [], view = 'overview';
let selectedModel = null, sending = false, controller = null, toastTimer, temperatures = [], lastModel = null;
let histories;
try { histories = JSON.parse(localStorage.getItem('onpu-history') || '{}'); } catch { histories = {}; }
if (!histories || typeof histories !== 'object' || Array.isArray(histories)) histories = {};
const currentHistory = () => histories[status?.runtime.model_id || 'none'] ||= [];
function persistHistory() { try { localStorage.setItem('onpu-history', JSON.stringify(histories)); } catch { toast('Browser storage is full; this chat will only last until refresh.', true); } }
try { document.documentElement.dataset.theme = localStorage.getItem('onpu-theme') || 'light'; } catch {}
function toast(message, error = false) {
  $('toast').textContent = message; $('toast').hidden = false; $('toast').classList.toggle('error', error);
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 8000 : 3500);
}
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  const result = await response.json();
  if (response.status === 401) { if (!$('login-dialog').open) $('login-dialog').showModal(); throw Error('Sign in to connect to this board.'); }
  if (!response.ok) {
    const detail = result.error?.message || result.detail;
    throw Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((x) => `${x.loc.at(-1)}: ${x.msg}`).join('; ') : 'Request failed');
  }
  return result;
}
const post = (path, body = {}) => api(path, { method: 'POST', body: JSON.stringify(body) });
function action(fn, preventDefault = true) { return async (event) => { if (preventDefault) event?.preventDefault(); try { await fn(event); } catch (e) { toast(e.message, true); } }; }
function setView() {
  view = ['overview', 'models', 'chat', 'benchmarks', 'settings'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview';
  document.querySelectorAll('.view').forEach((el) => { el.hidden = el.id !== `view-${view}`; });
  document.querySelectorAll('[data-view]').forEach((el) => { el.classList.toggle('active', el.dataset.view === view); if (el.dataset.view === view) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current'); });
  $('breadcrumb').textContent = view[0].toUpperCase() + view.slice(1);
  if (view === 'chat') renderChat();
  if (view === 'settings') populateSettings();
  refreshAll().catch(() => {});
}
window.addEventListener('hashchange', setView);
const endpoint = `${location.origin}/v1`;
$('endpoint-url').textContent = endpoint; $('settings-endpoint').textContent = endpoint;
$('copy-endpoint').onclick = action(async () => { await navigator.clipboard.writeText(endpoint); toast('API endpoint copied'); });
$('theme-toggle').onclick = () => { const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = theme; try { localStorage.setItem('onpu-theme', theme); } catch {} };
async function refreshStatus() {
  try {
    status = await api('/api/status');
    $('connection-text').textContent = 'Server online'; $('top-dot').classList.remove('offline'); $('connection-dot').classList.remove('offline');
    renderOverview(); renderChatStatus();
    if (lastModel !== status.runtime.model_id) { lastModel = status.runtime.model_id; if (view === 'chat' && !sending) renderChat(); }
  } catch (e) {
    $('connection-text').textContent = 'Disconnected'; $('top-dot').classList.add('offline'); $('connection-dot').classList.add('offline');
    $('send-message').disabled = true;
    if (status) $('chat-state').textContent = 'Server disconnected. Reconnect before sending.';
    throw e;
  }
}
async function refreshAll() {
  const data = await Promise.all([api('/api/models'), api('/api/downloads'), api('/api/metrics'), api('/api/benchmarks')]);
  modelList = data[0].models; catalog = data[0].catalog; jobs = data[1].jobs; metrics = data[2].metrics; benchmarks = data[3].jobs;
  renderLibrary(); renderCatalog(); renderJobs(); renderBenchmarks(); renderActivity();
}
function renderOverview() {
  const h = status.hardware, r = status.runtime, m = r.last_metrics;
  $('sidebar-board').textContent = h.board; $('sidebar-memory').textContent = `${bytes(h.memory.total, 0)} RAM · ${h.arch}`;
  $('stat-memory').innerHTML = `${(h.memory.available / 1024 ** 3).toFixed(1)}<span>GB</span>`;
  $('stat-memory-note').textContent = `${bytes(h.memory.total)} total · ${bytes(r.rss_bytes)} worker RSS`;
  $('memory-meter').style.width = `${Math.min(100, 100 * h.memory.available / h.memory.total)}%`;
  $('stat-temp').innerHTML = h.temperature == null ? 'Unavailable' : `${h.temperature.toFixed(1)}<span>°C</span>`;
  temperatures.push(h.temperature); temperatures = temperatures.slice(-30);
  $('temp-chart').innerHTML = temperatures.filter((t) => t != null).map((t) => `<i style="height:${Math.max(2, Math.min(16, (t / 100) * 16))}px"></i>`).join('');
  $('stat-ttft').innerHTML = m?.ttft_ms == null ? '—' : m.ttft_ms >= 1000 ? `${(m.ttft_ms / 1000).toFixed(2)}<span>s</span>` : `${m.ttft_ms.toFixed(0)}<span>ms</span>`;
  $('stat-speed').innerHTML = m?.decode_tps == null ? '—' : `${m.decode_tps.toFixed(1)}<span>tok/s</span>`;
  $('stat-token-total').textContent = `${status.totals.tokens.toLocaleString()} tokens · ${status.totals.requests} recent requests`;
  $('active-state').className = `badge ${r.state}`; $('active-state').textContent = r.state[0].toUpperCase() + r.state.slice(1);
  if (!r.model_id) {
    $('active-model-body').innerHTML = `<div class="empty-model"><div class="model-glyph">${icon('models')}</div><div><h3>No model loaded</h3><p class="${r.state === 'error' ? 'state-error' : ''}">${r.state === 'error' ? escape(r.message) : 'Your board is ready. Give it something to think about.'}</p><a class="button secondary small" href="#models">Browse your library ${icon('arrow')}</a></div></div>`;
  } else {
    $('active-model-body').innerHTML = `<div class="model-name">${escape(r.model_name)}</div><div class="model-meta">${badge('', r.actual_device ? `Executing on ${r.actual_device}` : `Selected: ${r.selected_profile}`)}${badge('', `${r.context || r.options.context} context`)}${r.load_seconds != null ? badge('', `Prepared in ${r.load_seconds.toFixed(1)}s`) : ''}</div><div class="model-live-row"><p class="${r.state === 'error' ? 'state-error' : ''}">${['loading', 'preparing', 'generating'].includes(r.state) ? '<span class="spinner"></span> ' : ''}${escape(r.message)}</p><div class="model-controls"><a class="button primary small" href="#chat">Chat ${icon('arrow')}</a><button class="button secondary small" data-action="unload" ${r.active_request ? 'disabled' : ''}>Unload</button></div></div>`;
  }
  $('device-list').innerHTML = ['CPU', 'CL', 'ROCKCHIP'].map((key) => {
    const p = status.profiles[key]; const label = key === 'CL' ? 'Mali GPU' : key === 'CPU' ? 'ARM CPU' : 'Rockchip NPU';
    return `<div class="device-row"><div class="device-symbol">${icon('chip')}</div><div><strong>${label}${key === 'CL' ? ' · experimental' : ''}</strong><small title="${escape(p.reason || p.name)}">${escape(p.available ? p.name : p.reason || 'Checking device')}</small></div>${badge(p.available ? 'ready' : '', key === 'ROCKCHIP' ? 'Experimental' : p.available == null ? 'Checking' : p.available ? key === 'CL' ? 'Tensor OK' : 'Available' : 'Unavailable')}</div>`;
  }).join('');
}
function renderActivity() {
  $('activity-table').innerHTML = metrics.length ? metrics.slice(0, 5).map((m) => `<tr><td>${escape(modelList.find((x) => x.id === m.model)?.name || m.model)}</td><td>${badge('', m.device)}</td><td>${m.completion_tokens}</td><td>${number(m.ttft_ms, ' ms', 0)}</td><td>${number(m.decode_tps, ' tok/s')}</td><td class="muted">${new Date(m.created * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td></tr>`).join('') : '<tr><td colspan="6" class="table-empty">Your first conversation will appear here.</td></tr>';
}
function renderLibrary() {
  $('model-count').textContent = modelList.length; $('library-count').textContent = modelList.length;
  const query = $('model-search').value.toLowerCase();
  const visible = modelList.filter((m) => `${m.name} ${m.architecture}`.toLowerCase().includes(query));
  $('model-library').innerHTML = visible.length ? visible.map((m) => `<article class="card model-card"><div class="model-card-top"><div class="model-glyph">${escape(m.architecture.slice(0, 2).toUpperCase())}</div>${badge(m.state, m.present ? m.state : 'File missing')}</div><h3>${escape(m.name)}</h3><div class="model-meta">${badge('', m.architecture)}${badge('', m.quantization)}${badge('', 'GGUF')}</div><dl class="model-properties"><div><dt>FILE SIZE</dt><dd>${bytes(m.size)}</dd></div><div><dt>WEIGHT PARAMETERS</dt><dd>${(m.parameters / 1e9).toFixed(2)}B</dd></div><div><dt>ESTIMATED MEMORY</dt><dd>${bytes(m.estimated_bytes)}</dd></div><div><dt>MODEL CONTEXT LIMIT</dt><dd>${m.context_limit.toLocaleString()}</dd></div></dl><div class="model-card-actions">${m.state === 'installed' || m.state === 'error' ? `<button class="button primary small" data-action="load" data-id="${m.id}" ${!m.present ? 'disabled' : ''}>${icon('play')} Load model</button>` : `<button class="button secondary small" data-action="unload" ${status?.runtime.active_request ? 'disabled' : ''}>Unload</button>`}<button class="text-button" data-action="remove" data-id="${m.id}" ${m.state !== 'installed' ? 'disabled' : ''} aria-label="Remove ${escape(m.name)} from library">Remove reference</button></div></article>`).join('') : `<div class="empty-state"><div class="model-glyph">${icon('models')}</div><h3>${query ? 'No matching models' : 'Your model library starts here'}</h3><p>${query ? 'Try another search.' : 'Import a GGUF already on your board, or download a small model.'}</p>${query ? '' : '<button class="button primary" data-action="import">Import model</button> <button class="button secondary" data-action="discover">Discover models</button>'}</div>`;
}
function renderCatalog() {
  $('model-catalog').innerHTML = catalog.map((m) => `<article class="card model-card"><div class="model-card-top"><div class="model-glyph">${escape(m.name.slice(0, 2).toUpperCase())}</div>${badge('', m.parameters)}</div><h3>${escape(m.name)}</h3><div class="model-meta">${badge('', m.quantization)}${badge('', 'GGUF')}</div><p>${escape(m.description)}</p><div class="model-card-actions"><small>Hugging Face</small><button class="button primary small" data-action="download" data-id="${escape(m.id)}">${icon('download')} Download</button></div></article>`).join('');
}
function renderJobs() {
  $('download-jobs').innerHTML = jobs.filter((j) => j.state !== 'complete').map((j) => `<article class="card download-job"><div class="download-job-header"><h3>${escape(j.file)}</h3>${badge(j.state)}</div><p>${j.error ? escape(j.error) : `${bytes(j.received)}${j.total ? ' of ' + bytes(j.total) : ''} · ${escape(j.repo)}`}</p><div class="meter"><i style="width:${j.total ? Math.min(100, j.received / j.total * 100) : 0}%"></i></div><div class="dialog-actions">${['paused', 'error'].includes(j.state) ? `<button class="button secondary small" data-action="resume" data-id="${j.id}">Resume</button>` : `<button class="button secondary small" data-action="pause" data-id="${j.id}">Pause</button>`}</div></article>`).join('');
}
function renderBenchmarks() {
  $('benchmark-run').disabled = status?.runtime.state !== 'ready' || benchmarks.some((j) => j.state === 'running');
  $('benchmark-results').innerHTML = benchmarks.length ? benchmarks.map((j) => `<article class="card benchmark-result"><div class="card-heading"><div><h2>${escape(j.model)}</h2><small>${escape(j.profile)} · ${j.context} context · ${new Date(j.created * 1000).toLocaleString()}</small></div>${badge(j.state)}${j.state === 'running' ? `<button class="button secondary small" data-action="cancel-benchmark" data-id="${j.id}">Cancel</button>` : ''}</div><div class="table-scroll"><table><thead><tr><th>Trial</th><th>Prompt / cached</th><th>Output</th><th>First response</th><th>Prefill estimate</th><th>Decode</th><th>Total</th></tr></thead><tbody>${j.results.length ? j.results.map((r, i) => `<tr><td>${i + 1}</td><td>${r.prompt_tokens} / ${r.cached_tokens}</td><td>${r.completion_tokens}</td><td>${number(r.ttft_ms, ' ms', 0)}</td><td>${number(r.prefill_tps, ' tok/s')}</td><td>${number(r.decode_tps, ' tok/s')}</td><td>${number(r.total_seconds, ' s', 2)}</td></tr>`).join('') : '<tr><td colspan="7" class="table-empty">Processing the first trial…</td></tr>'}</tbody></table></div><div class="card-bottom">${escape(j.error || j.qualification)} · ${j.completed}/${j.repeats} trials · Backend ${escape(j.revision?.slice(0, 10) || 'unknown')}</div></article>`).join('') : '<div class="empty-state"><div class="model-glyph">' + icon('benchmark') + '</div><h3>Know what is fast on your board.</h3><p>Load a model, choose a device profile, then run your first benchmark.</p></div>';
}
function populateSettings() {
  for (const [id, key] of [['setting-context','context'], ['setting-max-tokens','max_tokens'], ['setting-temperature','temperature'], ['setting-profile','profile'], ['setting-idle','idle_timeout']]) $(id).value = settings[key];
  $('setting-realize').checked = settings.realize; $('setting-memory').value = Math.round(settings.memory_fraction * 100);
  $('settings-model-id').textContent = status?.runtime.model_id || 'Load a model first';
  $('api-example').textContent = `curl ${endpoint}/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify({ model: status?.runtime.model_id || '<model-id>', messages: [{ role: 'user', content: 'Hello!' }], stream: true, max_tokens: 64 })}'`;
  if (status) $('backend-info').innerHTML = [['Backend checkout',status.backend.path], ['Revision',status.backend.revision || 'Unknown'], ['Board',status.hardware.board], ['Kernel',status.hardware.kernel], ['Architecture',status.hardware.arch], ['Data directory',settings.data_dir || ''], ['Execution',status.runtime.actual_device || 'No model loaded']].map(([k,v]) => `<dt>${escape(k)}</dt><dd>${escape(v)}</dd>`).join('');
}
function modelTab(download) {
  $('library-panel').hidden = download; $('download-panel').hidden = !download;
  $('library-tab').classList.toggle('active', !download); $('download-tab').classList.toggle('active', download);
  $('library-tab').setAttribute('aria-selected', String(!download)); $('download-tab').setAttribute('aria-selected', String(download));
}
$('library-tab').onclick = () => modelTab(false); $('download-tab').onclick = () => modelTab(true);
$('import-open').onclick = () => $('import-dialog').showModal();
$('model-search').oninput = renderLibrary; $('refresh-models').onclick = action(refreshAll);
$('check-devices').onclick = action(async () => { $('check-devices').disabled = true; try { await post('/api/profiles/check'); await refreshStatus(); toast('Device checks complete'); } finally { $('check-devices').disabled = false; } });
document.addEventListener('click', action(async (event) => {
  const close = event.target.closest('[data-close]'); if (close) $(close.dataset.close).close();
  const button = event.target.closest('[data-action]'); if (!button || button.disabled) return;
  const id = button.dataset.id;
  switch (button.dataset.action) {
    case 'import': $('import-dialog').showModal(); break;
    case 'discover': modelTab(true); location.hash = 'models'; break;
    case 'load': {
      selectedModel = modelList.find((m) => m.id === id);
      $('load-model-title').textContent = selectedModel.name;
      $('load-profile').value = settings.profile || 'CPU';
      $('load-profile').querySelector('[value=CL]').disabled = status?.profiles.CL.available === false;
      $('load-context').value = Math.min(settings.context || 1024, selectedModel.context_limit);
      $('load-context').max = Math.min(32768, selectedModel.context_limit);
      $('load-realize').checked = settings.realize ?? true;
      $('load-estimate').textContent = `Conservative footprint: ${bytes(selectedModel.estimated_bytes)}. Available RAM: ${bytes(status?.hardware.memory.available)}.`;
      $('model-dialog').showModal(); break;
    }
    case 'unload': await post('/api/models/unload'); await refreshStatus(); await refreshAll(); toast('Model unloaded'); break;
    case 'remove': await api(`/api/models/${id}`, { method: 'DELETE' }); await refreshAll(); toast('Reference removed. The model file is preserved.'); break;
    case 'download': await post('/api/downloads', { catalog_id: id }); await refreshAll(); toast('Download started'); break;
    case 'pause': case 'resume': await post(`/api/downloads/${id}/${button.dataset.action}`); await refreshAll(); break;
    case 'cancel-benchmark': await post(`/api/benchmarks/${id}/cancel`); await refreshAll(); break;
    case 'suggest': $('chat-input').value = button.dataset.prompt; $('chat-input').focus(); break;
    case 'copy-message': await navigator.clipboard.writeText(currentHistory()[Number(id)].content); toast('Reply copied'); break;
    case 'regenerate': if (!sending) { const history = currentHistory(); history.pop(); persistHistory(); await generateReply(); } break;
    case 'edit-message': if (!sending) { const history = currentHistory(); const index = Number(id); $('chat-input').value = history[index].content; history.splice(index); persistHistory(); renderChat(); $('chat-input').focus(); } break;
  }
}, false));
$('import-form').onsubmit = action(async () => { await post('/api/models/import', { path: $('import-path').value.trim() }); $('import-dialog').close(); await refreshAll(); toast('Model added to your library'); });
$('load-form').onsubmit = action(async () => {
  await post(`/api/models/${selectedModel.id}/load`, { profile: $('load-profile').value, context: Number($('load-context').value), realize: $('load-realize').checked });
  $('model-dialog').close(); await refreshStatus(); await refreshAll(); toast('Preparing model. Progress appears in Overview.');
});
$('settings-form').onsubmit = action(async () => {
  const body = { context: Number($('setting-context').value), max_tokens: Number($('setting-max-tokens').value), temperature: Number($('setting-temperature').value), profile: $('setting-profile').value, realize: $('setting-realize').checked, idle_timeout: Number($('setting-idle').value), memory_fraction: Number($('setting-memory').value) / 100 };
  settings = { ...settings, ...await api('/api/settings', { method: 'PUT', body: JSON.stringify(body) }) };
  $('settings-saved').textContent = 'Saved'; toast('Settings saved');
});
$('login-form').onsubmit = action(async () => {
  try { await post('/api/login', { key: $('login-key').value }); $('login-dialog').close(); $('login-key').value = ''; await initialize(); }
  catch (e) { $('login-error').textContent = e.message; }
});
$('show-logs').onclick = action(async () => { const data = await api('/api/logs'); $('worker-logs').textContent = data.lines.join('\n') || 'No worker output yet.'; $('worker-logs').hidden = !$('worker-logs').hidden; });
$('hf-search-form').onsubmit = action(async () => {
  $('hf-results').textContent = 'Searching…';
  const data = await api(`/api/huggingface/search?q=${encodeURIComponent($('hf-query').value)}`);
  $('hf-results').innerHTML = data.repos.map((r) => `<div class="hf-repo"><span>${escape(r.id)}</span><button class="button secondary small hf-browse" data-repo="${escape(r.id)}">Choose a GGUF</button></div>`).join('') || '<p>No matching GGUF repositories.</p>';
  document.querySelectorAll('.hf-browse').forEach((button) => { button.onclick = action(async () => {
    const repo = button.dataset.repo; button.disabled = true;
    try {
      const files = await api(`/api/huggingface/files?repo=${encodeURIComponent(repo)}`);
      let panel = button.parentElement.querySelector('.hf-files'); if (panel) panel.remove();
      panel = document.createElement('div'); panel.className = 'hf-files';
      panel.innerHTML = `<select aria-label="GGUF file">${files.files.map((f) => `<option value="${escape(f)}">${escape(f)}</option>`).join('')}</select><button class="button primary small">Download</button>`;
      panel.querySelector('button').disabled = !files.files.length;
      panel.querySelector('button').onclick = action(async () => { await post('/api/downloads', { repo, file: panel.querySelector('select').value, revision: files.revision }); toast('Download started'); await refreshAll(); });
      button.parentElement.append(panel);
    } finally { button.disabled = false; }
  }); });
});
$('benchmark-form').onsubmit = action(async () => { await post('/api/benchmarks', { prompt: $('benchmark-prompt').value, max_tokens: Number($('benchmark-tokens').value), repeats: Number($('benchmark-repeats').value) }); await refreshAll(); });
function formatText(text) {
  return text.split('```').map((part, index) => index % 2 ? `<code>${escape(part.replace(/^\w*\n/, ''))}</code>` : escape(part).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')).join('');
}
function renderChat() {
  const history = currentHistory();
  if (!history.length) {
    $('chat-messages').innerHTML = `<div class="chat-welcome"><div class="model-glyph">o</div><h2>A little board. A lot to explore.</h2><p>${status?.runtime.model_id ? 'Start a conversation with your local model.' : 'Choose and load a model from your library to get started.'}</p><div class="suggestions">${['Explain how a Rockchip NPU works', 'Write a Python script to monitor temperature', 'Help me plan a local AI project', 'What makes a good inference benchmark?'].map((s) => `<button class="suggestion" data-action="suggest" data-prompt="${escape(s)}">${escape(s)} ${icon('arrow')}</button>`).join('')}</div></div>`;
  } else {
    $('chat-messages').innerHTML = history.map((m, index) => `<article class="message ${escape(m.role)}"><div class="message-avatar">${m.role === 'user' ? 'You' : 'o'}</div><div class="message-body"><div class="message-label">${m.role === 'user' ? 'You' : escape(status?.runtime.model_name || 'Assistant')}</div>${m.reasoning ? `<details class="reasoning"><summary>Thinking</summary><div class="message-content">${formatText(m.reasoning)}</div></details>` : ''}<div class="message-content">${formatText(m.content || '')}${sending && index === history.length - 1 && !m.content ? '<span class="spinner"></span>' : ''}</div>${m.error ? `<div class="message-error">${escape(m.error)}</div>` : ''}<div class="message-actions">${m.role === 'assistant' ? `<button class="text-button" data-action="copy-message" data-id="${index}" aria-label="Copy reply">${icon('copy')}</button>${index === history.length - 1 && !sending ? '<button class="text-button" data-action="regenerate">Regenerate</button>' : ''}` : !sending ? `<button class="text-button" data-action="edit-message" data-id="${index}">Edit & retry</button>` : ''}${m.tokens != null ? `<span>${m.tokens} tokens</span>` : ''}</div></div></article>`).join('');
  }
  renderChatStatus();
  $('chat-messages').scrollTop = $('chat-messages').scrollHeight;
}
function renderChatStatus() {
  const r = status?.runtime;
  $('chat-model-label').textContent = r?.model_name ? `${r.model_name} · ${r.actual_device || r.selected_profile}` : 'Load a model to start a conversation.';
  $('chat-context').textContent = r?.context ? `${r.context.toLocaleString()} token capacity` : 'Context unavailable';
  $('send-message').disabled = sending || r?.state !== 'ready'; $('send-message').hidden = sending; $('stop-generation').hidden = !sending;
  $('chat-input').disabled = sending; $('new-chat').disabled = sending;
  $('chat-state').textContent = sending ? r?.message || 'Generating…' : r?.state !== 'ready' ? r?.message || 'Load a model to begin.' : 'Ready · runs entirely on this board';
}
async function generateReply() {
  if (sending || !status?.runtime.model_id) return;
  const history = currentHistory();
  const reply = { role: 'assistant', content: '', reasoning: '' }; history.push(reply);
  sending = true; controller = new AbortController(); renderChat();
  try {
    const response = await fetch('/v1/chat/completions', { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: status.runtime.model_id, messages: history.slice(0, -1).map((m) => ({ role: m.role, content: m.content })), temperature: Number($('chat-temperature').value), max_tokens: Number($('chat-max-tokens').value), enable_thinking: $('chat-thinking').checked, stream: true, stream_options: { include_usage: true } }) });
    if (!response.ok) { const error = await response.json(); throw Error(error.error?.message || JSON.stringify(error.detail) || 'Generation failed'); }
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, '\n');
      const frames = buffer.split('\n\n'); buffer = frames.pop();
      for (const frame of frames) {
        const text = frame.split('\n').filter((s) => s.startsWith('data: ')).map((s) => s.slice(6)).join('\n');
        if (!text || text === '[DONE]') continue;
        const event = JSON.parse(text);
        if (event.error) throw Error(event.error.message);
        for (const choice of event.choices || []) { reply.content += choice.delta?.content || ''; reply.reasoning += choice.delta?.reasoning_content || ''; }
        if (event.usage) reply.tokens = event.usage.completion_tokens;
      }
      renderChat();
      if (done) break;
    }
  } catch (e) { reply.error = e.name === 'AbortError' ? 'Stopped. Partial reply preserved.' : e.message; }
  finally { sending = false; controller = null; persistHistory(); renderChat(); refreshStatus().catch(() => {}); refreshAll().catch(() => {}); $('chat-input').focus(); }
}
$('chat-form').onsubmit = action(async () => { const text = $('chat-input').value.trim(); if (!text || sending || status?.runtime.state !== 'ready') return; currentHistory().push({ role: 'user', content: text }); $('chat-input').value = ''; $('chat-input').style.height = ''; persistHistory(); await generateReply(); });
$('chat-input').onkeydown = (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('chat-form').requestSubmit(); } };
$('chat-input').oninput = () => { $('chat-input').style.height = 'auto'; $('chat-input').style.height = `${Math.min(160, $('chat-input').scrollHeight)}px`; };
$('stop-generation').onclick = action(async () => { try { await post('/api/chat/stop'); } finally { controller?.abort(); } });
$('new-chat').onclick = () => { if (!sending) { histories[status?.runtime.model_id || 'none'] = []; persistHistory(); renderChat(); } };
$('chat-options-toggle').onclick = () => { $('chat-options').hidden = !$('chat-options').hidden; };
async function initialize() {
  const data = await api('/api/settings'); settings = { ...data.settings, data_dir: data.data_dir };
  $('auth-status').textContent = data.auth_enabled ? 'Authentication enabled' : 'No API key required';
  $('chat-temperature').value = settings.temperature; $('chat-max-tokens').value = settings.max_tokens;
  await refreshStatus(); await refreshAll(); populateSettings(); renderChat(); setView();
}
initialize().catch((e) => { if (!$('login-dialog').open) toast(e.message, true); });
setInterval(() => { refreshStatus().catch(() => {}); }, 2000);
setInterval(() => { refreshAll().catch(() => {}); }, 6000);
