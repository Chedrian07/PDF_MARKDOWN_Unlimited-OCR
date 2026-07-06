// Unlimited-OCR — PDF → Markdown : frontend logic
// Vanilla ES module. No external dependencies. Same-origin /api calls.
//
// Active-job live view = synchronized 3 panes fed by one SSE token stream:
//   left   : source page image + layout boxes parsed from grounding tokens
//   middle : raw token stream (with <PAGE> dividers)
//   right  : cleaned markdown rendered server-side via POST /render-preview

'use strict';

/* ============================ Constants ============================ */

const PHASE_LABELS = { render: '렌더링', ocr: 'OCR', merge: '병합' };
const STATUS_LABELS = {
  queued: '대기중',
  running: '변환중',
  done: '완료',
  error: '오류',
  canceled: '취소됨',
};
const PAGE_MARKER = '<PAGE>';
const THEME_KEY = 'uocr-theme';
const IMG_PLACEHOLDER = '\n\n> 🖼 그림 감지됨\n\n';
// literals whose partial prefix at a chunk boundary must be held back
const MARKER_LITERALS = ['<PAGE>', '<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>'];

const BOX_COLORS = {
  title: '#e5484d',
  text: '#4662d9',
  image: '#2f9e6e',
  table: '#8e4ec6',
  formula: '#d97706',
  equation: '#d97706',
  page_number: '#8b8d98',
  footnote: '#8b8d98',
  header: '#8b8d98',
  footer: '#8b8d98',
};
const BOX_FALLBACK_COLOR = '#6b7280';

const ICON = {
  moon: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
  sun: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/></svg>',
  x: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  chip: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
};

/* ============================ State ============================ */

const state = {
  jobs: [],
  currentJobId: null,
  displayedStatus: null,
  selectedFile: null,
  // raw stream pane
  streamPending: '',
  streamPageNo: 1,
  streamAutoScroll: true,
  streamConnected: false,
  rafId: 0,
  // accumulated raw model output (for cleaning + preview)
  rawText: '',
  // grounding parser
  groundBuf: '',
  boxPage: 1,          // page the stream is currently parsing (boxes attach here)
  viewPage: 1,         // page shown in the left pane
  followLive: true,
  ocrSeen: false,      // true once an OCR-phase progress event arrived for this job
  totalPages: 0,
  pageBoxes: new Map(), // pageNo -> [{label,x1,y1,x2,y2}]
  imgFailed: false,
  imgLastTry: 0,
  // live rendered preview (right pane)
  liveGen: 0,
  previewDirty: false,
  previewTimer: 0,
  previewInFlight: false,
  previewFails: 0,
  previewAutoScroll: true,
  // cancel
  cancelRequestedFor: null,
  // sse / fallback
  es: null,
  sseErrorCount: 0,
  fallbackActive: false,
  fallbackTimer: 0,
  // result tab caches
  previewLoaded: false,
  markdownLoaded: false,
  // timers
  jobsTimer: 0,
  healthTimer: 0,
  toastTimer: 0,
};

const armTimers = new Map();

/* ============================ DOM refs ============================ */

const el = {};
const EL_IDS = {
  healthBadges: 'health-badges',
  themeToggle: 'theme-toggle',
  dropzone: 'dropzone',
  fileInput: 'file-input',
  fileInfo: 'file-info',
  fileName: 'file-name',
  fileSize: 'file-size',
  fileClear: 'file-clear',
  uploadError: 'upload-error',
  uploadBtn: 'upload-btn',
  dpiInput: 'dpi-input',
  jobList: 'job-list',
  jobListEmpty: 'job-list-empty',
  emptyState: 'empty-state',
  jobView: 'job-view',
  jobChip: 'job-status-chip',
  jobFilename: 'job-filename',
  jobTime: 'job-time',
  jobStop: 'job-stop',
  jobStopLabel: 'job-stop-label',
  jobDelete: 'job-delete',
  progressSection: 'progress-section',
  progressPhase: 'progress-phase',
  progressSpinner: 'progress-spinner',
  progressCount: 'progress-count',
  progressTrack: 'progress-track',
  progressFill: 'progress-fill',
  progressChunk: 'progress-chunk',
  liveDetails: 'live-details',
  streamPane: 'stream-pane',
  pageImg: 'page-img',
  boxOverlay: 'box-overlay',
  pageNote: 'page-note',
  pagerPrev: 'pager-prev',
  pagerNext: 'pager-next',
  pagerLabel: 'pager-label',
  followChip: 'follow-chip',
  livePreview: 'live-preview',
  errorSection: 'error-section',
  errorTitle: 'error-title',
  errorMessage: 'error-message',
  errorHint: 'error-hint',
  resultSection: 'result-section',
  dlMd: 'dl-md',
  dlZip: 'dl-zip',
  previewBody: 'preview-body',
  mdCode: 'md-code',
  copyMd: 'copy-md',
  layoutsGrid: 'layouts-grid',
  pagesGrid: 'pages-grid',
  toast: 'toast',
};

function grabEls() {
  for (const key of Object.keys(EL_IDS)) el[key] = document.getElementById(EL_IDS[key]);
  el.tabs = Array.from(document.querySelectorAll('.tab'));
  el.panels = Array.from(document.querySelectorAll('.tab-panel'));
  el.modeRadios = Array.from(document.querySelectorAll('input[name="mode"]'));
}

/* ============================ Utilities ============================ */

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const key of Object.keys(attrs)) {
      const val = attrs[key];
      if (val == null || val === false) continue;
      if (key === 'class') node.className = val;
      else if (key === 'text') node.textContent = val;
      else if (key === 'html') node.innerHTML = val; // only used with trusted literal SVG strings
      else node.setAttribute(key, val === true ? '' : val);
    }
  }
  for (const child of children) {
    if (child == null || child === false) continue;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

function safeParse(text) {
  try { return JSON.parse(text); } catch (_) { return null; }
}

function localGet(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}
function localSet(key, val) {
  try { localStorage.setItem(key, val); } catch (_) { /* ignore */ }
}

function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  try {
    return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false });
  } catch (_) {
    const p = (n) => String(n).padStart(2, '0');
    return `${p(d.getHours())}:${p(d.getMinutes())}`;
  }
}

function fmtBytes(n) {
  if (!Number.isFinite(n) || n < 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = n, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  const decimals = i > 0 && v < 10 ? 1 : 0;
  return `${v.toFixed(decimals)} ${units[i]}`;
}

function isTerminal(status) {
  return status === 'done' || status === 'error' || status === 'canceled';
}

async function apiGet(path) {
  const res = await fetch(path, { headers: { Accept: 'application/json' } });
  const text = await res.text().catch(() => '');
  const data = text ? safeParse(text) : null;
  if (!res.ok) {
    const msg = (data && typeof data.detail === 'string') ? data.detail : `요청 실패 (${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

async function apiDelete(path) {
  const res = await fetch(path, { method: 'DELETE' });
  if (!res.ok) {
    const err = new Error(`삭제 실패 (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return true;
}

function showToast(message, kind) {
  el.toast.textContent = message;
  el.toast.className = 'toast' + (kind ? ' ' + kind : '');
  el.toast.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { el.toast.hidden = true; }, 3600);
}

/* ============================ Theme ============================ */

function resolvedTheme() {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const label = theme === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환';
  el.themeToggle.innerHTML = theme === 'dark' ? ICON.sun : ICON.moon;
  el.themeToggle.setAttribute('aria-label', label);
  el.themeToggle.title = label;
}

function setupTheme() {
  applyTheme(resolvedTheme()); // sync icon with the value set by the inline bootstrap
  el.themeToggle.addEventListener('click', () => {
    const next = resolvedTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localSet(THEME_KEY, next);
  });
  try {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', (e) => {
      const stored = localGet(THEME_KEY);
      if (stored !== 'light' && stored !== 'dark') applyTheme(e.matches ? 'dark' : 'light');
    });
  } catch (_) { /* ignore */ }
}

/* ============================ Health ============================ */

function shortenGpu(name) {
  return String(name || '').replace(/^NVIDIA\s+GeForce\s+/i, '').replace(/^NVIDIA\s+/i, '').trim();
}

async function loadHealth() {
  clearTimeout(state.healthTimer);
  let data;
  try {
    data = await apiGet('/api/health');
  } catch (_) {
    renderHealthError();
    state.healthTimer = setTimeout(loadHealth, 10000);
    return;
  }
  renderHealth(data || {});
  if (data && data.model_loaded === false) {
    state.healthTimer = setTimeout(loadHealth, 10000);
  }
}

function renderHealth(d) {
  const c = el.healthBadges;
  c.textContent = '';

  const modelId = d.model_id || 'baidu/Unlimited-OCR';
  c.appendChild(h('span', { class: 'badge badge-model', title: modelId },
    h('span', { class: 'badge-ico', html: ICON.chip }),
    h('span', { text: modelId }),
  ));

  const isCuda = d.device === 'cuda';
  const devName = isCuda ? 'CUDA' : (d.device === 'cpu' ? 'CPU' : String(d.device || '?').toUpperCase());
  let devText = devName;
  if (isCuda && d.gpu_name) {
    const short = shortenGpu(d.gpu_name);
    if (short) devText = `${devName} · ${short}`;
  }
  const devTitle = `디바이스: ${devName}` +
    (d.gpu_name ? ` (${d.gpu_name})` : '') +
    ` · dtype: ${d.dtype || '-'} · 네이티브 연산: ${d.native_ops ? 'on' : 'off'}`;
  c.appendChild(h('span', { class: `badge badge-device ${isCuda ? 'is-cuda' : 'is-cpu'}`, title: devTitle },
    h('span', { class: 'badge-dot' }),
    h('span', { text: devText }),
  ));

  if (d.engine === 'fake') {
    c.appendChild(h('span', {
      class: 'badge badge-warn',
      title: '실제 모델 대신 데모용 가짜 엔진이 실행 중입니다.',
    }, 'FAKE 엔진'));
  }

  if (d.model_loaded === false) {
    c.appendChild(h('span', {
      class: 'badge badge-loading',
      title: '모델을 메모리에 로딩하는 중입니다. 첫 작업에서 시간이 걸릴 수 있습니다.',
    }, h('span', { class: 'spinner spinner-xs' }), h('span', { text: '모델 로딩 중…' })));
  }
}

function renderHealthError() {
  el.healthBadges.textContent = '';
  el.healthBadges.appendChild(h('span', {
    class: 'badge badge-error',
    title: '서버 상태를 확인할 수 없습니다. 자동으로 재시도합니다.',
  }, '서버 연결 실패'));
}

/* ============================ Job history ============================ */

async function refreshJobs() {
  let data;
  try {
    data = await apiGet('/api/jobs');
  } catch (_) {
    return; // keep last known list on transient failure
  }
  const jobs = (data && Array.isArray(data.jobs)) ? data.jobs : [];
  state.jobs = jobs.slice(0, 50);
  renderJobList();

  if (state.currentJobId) {
    const open = state.jobs.find((j) => j.job_id === state.currentJobId);
    if (open) {
      updateHeaderChip(open.status);
      if (isTerminal(open.status) && !isTerminal(state.displayedStatus)) {
        syncOpenJob();
      }
    }
  }
}

function renderJobList() {
  const list = el.jobList;
  list.textContent = '';
  if (!state.jobs.length) {
    el.jobListEmpty.hidden = false;
    return;
  }
  el.jobListEmpty.hidden = true;
  for (const job of state.jobs) list.appendChild(jobListItem(job));
}

function jobListItem(job) {
  const status = job.status || 'queued';
  const active = job.job_id === state.currentJobId;

  const name = h('span', { class: 'ji-name', text: job.filename || '(이름 없음)', title: job.filename || '' });
  const chip = h('span', { class: `chip chip-${status}`, text: STATUS_LABELS[status] || status });
  const time = h('span', { class: 'ji-time muted', text: fmtTime(job.created_at) });
  const sub = h('div', { class: 'ji-sub' }, chip, time);
  const main = h('div', { class: 'ji-main' }, name, sub);

  const del = h('button', { class: 'ji-del icon-btn-sm', type: 'button', 'aria-label': '삭제', title: '삭제', html: ICON.x });
  del.addEventListener('click', (ev) => {
    ev.stopPropagation();
    armDelete(del, () => deleteJob(job.job_id));
  });

  const li = h('li', { class: `job-item${active ? ' active' : ''}`, role: 'button', tabindex: '0' }, main, del);
  li.addEventListener('click', () => openJob(job.job_id));
  li.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openJob(job.job_id); }
  });
  return li;
}

function armDelete(btn, onConfirm) {
  if (btn.classList.contains('armed')) {
    clearTimeout(armTimers.get(btn));
    armTimers.delete(btn);
    btn.classList.remove('armed');
    btn.title = btn.dataset.baseTitle || '삭제';
    onConfirm();
    return;
  }
  btn.dataset.baseTitle = btn.title || '삭제';
  btn.classList.add('armed');
  btn.title = '한 번 더 클릭하면 삭제됩니다';
  const t = setTimeout(() => {
    btn.classList.remove('armed');
    btn.title = btn.dataset.baseTitle || '삭제';
    armTimers.delete(btn);
  }, 2600);
  armTimers.set(btn, t);
}

function removeJobFromList(id) {
  state.jobs = state.jobs.filter((j) => j.job_id !== id);
  renderJobList();
}

function upsertJob(job) {
  state.jobs = state.jobs.filter((j) => j.job_id !== job.job_id);
  state.jobs.unshift(job);
  state.jobs = state.jobs.slice(0, 50);
  renderJobList();
}

async function deleteJob(id) {
  try {
    await apiDelete(`/api/jobs/${id}`);
  } catch (e) {
    if (e.status !== 404) {
      showToast('삭제에 실패했습니다.', 'error');
      return;
    }
    // 404 → already gone; fall through to local cleanup
  }
  removeJobFromList(id);
  if (state.currentJobId === id) {
    teardownConnections();
    state.currentJobId = null;
    state.displayedStatus = null;
    showEmptyState();
  }
  refreshJobs();
}

/* ============================ View switching ============================ */

function showEmptyState() {
  el.jobView.hidden = true;
  el.emptyState.hidden = false;
}

function showJobView() {
  el.emptyState.hidden = true;
  el.jobView.hidden = false;
}

function updateHeaderChip(status) {
  el.jobChip.className = `chip chip-${status}`;
  el.jobChip.textContent = STATUS_LABELS[status] || status;
}

/* ============================ Open / render a job ============================ */

async function openJob(id) {
  if (!id || id === state.currentJobId) return;

  teardownConnections();
  state.currentJobId = id;
  state.displayedStatus = null;
  state.previewLoaded = false;
  state.markdownLoaded = false;
  resetLiveState();
  showJobView();
  renderJobList(); // refresh active highlight

  let job;
  try {
    job = await apiGet(`/api/jobs/${id}`);
  } catch (e) {
    if (state.currentJobId !== id) return;
    if (e.status === 404) {
      showToast('해당 작업을 찾을 수 없습니다.', 'warn');
      removeJobFromList(id);
    } else {
      showToast('작업 정보를 불러오지 못했습니다.', 'error');
    }
    state.currentJobId = null;
    showEmptyState();
    return;
  }
  if (state.currentJobId !== id) return; // user switched away during await

  renderJob(job);
  if (job.status === 'queued' || job.status === 'running') startStream(id);
}

// Re-render the currently open job without tearing down the live panes.
async function syncOpenJob() {
  const id = state.currentJobId;
  if (!id) return;
  let job;
  try {
    job = await apiGet(`/api/jobs/${id}`);
  } catch (_) {
    return;
  }
  if (state.currentJobId !== id) return;
  if (isTerminal(job.status)) {
    flushStream(true);
    processGroundBuf(true);
    teardownConnections();
  }
  renderJob(job);
}

function renderJob(job) {
  state.displayedStatus = job.status;

  el.jobFilename.textContent = job.filename || '(이름 없음)';
  el.jobFilename.title = job.filename || '';
  el.jobTime.textContent = job.created_at ? fmtTime(job.created_at) : '';
  updateHeaderChip(job.status);

  const running = job.status === 'queued' || job.status === 'running';
  const done = job.status === 'done';
  const canceled = job.status === 'canceled';
  const failed = job.status === 'error';

  el.progressSection.hidden = !running;
  el.resultSection.hidden = !(done || canceled);
  el.errorSection.hidden = !(failed || canceled);
  el.liveDetails.hidden = !running && !hasLiveContent();
  el.liveDetails.open = running;
  setStopButton(job.status);

  if (running) {
    updateProgress(job.progress || {}, job.status);
    const p = job.progress || {};
    const total = Number(p.total_pages) || 0;
    if (total > state.totalPages) state.totalPages = total;
    // Same phase gate as applyProgress: only an OCR-phase snapshot may seed
    // the box page (job opened mid-OCR); render/merge snapshots must not pin.
    if (p.phase === 'ocr') {
      state.ocrSeen = true;
      const cur = Number(p.current_page) || 0;
      if (cur > state.boxPage) {
        state.boxPage = state.totalPages ? Math.min(cur, state.totalPages) : cur;
        if (state.followLive) state.viewPage = state.boxPage;
      }
    }
    updateLeftPane();
  }
  if (done) renderResult(job);
  if (canceled) {
    renderError(job.error, true);
    if (job.result) renderResult(job);
    else renderPartialResult(job);
  }
  if (failed) renderError(job.error, false);
}

function hasLiveContent() {
  return state.rawText.length > 0 || state.pageBoxes.size > 0 || el.streamPane.childNodes.length > 0;
}

/* ============================ Progress ============================ */

function updateProgress(p, status) {
  const queued = status === 'queued';
  const total = Number(p.total_pages) || 0;
  const cur = Number(p.current_page) || 0;
  const totalChunks = Number(p.total_chunks) || 0;
  const chunk = Number(p.chunk) || 0;

  el.progressPhase.textContent = queued ? '대기 중' : (PHASE_LABELS[p.phase] || '처리 중');
  el.progressSpinner.hidden = !queued;

  const determinate = !queued && total > 0;
  el.progressTrack.classList.toggle('indeterminate', !determinate);
  if (determinate) {
    const pct = Math.min(100, Math.max(0, (cur / total) * 100));
    el.progressFill.style.width = `${pct}%`;
    el.progressCount.textContent = `${cur} / ${total} 페이지`;
  } else {
    el.progressFill.style.width = '';
    el.progressCount.textContent = '';
  }

  el.progressChunk.textContent = (!queued && totalChunks > 0) ? `청크 ${chunk} / ${totalChunks}` : '';
}

// SSE / poll progress payloads are flat objects that include "status".
function applyProgress(d) {
  const status = d.status || state.displayedStatus;
  const wasRunning = state.displayedStatus === 'queued' || state.displayedStatus === 'running';
  state.displayedStatus = status;
  updateHeaderChip(status);

  const running = status === 'queued' || status === 'running';
  el.progressSection.hidden = !running;
  if (!running) return;

  el.resultSection.hidden = true;
  el.errorSection.hidden = true;
  el.liveDetails.hidden = false;
  if (!wasRunning) el.liveDetails.open = true; // open on transition only (respect manual collapse)
  setStopButton(status);
  updateProgress(d, status);

  // sync the left pane with progress info
  const total = Number(d.total_pages) || 0;
  if (total > state.totalPages) {
    state.totalPages = total;
    updateLeftPane();
  }
  // drain parsed-but-buffered grounding events before adopting a page bump,
  // so buffered boxes stay attributed to their own page
  processGroundBuf(false);
  // Only OCR-phase progress may advance the box page. The render phase emits
  // current_page=1..N in quick succession while rasterizing (before any token
  // exists), and merge walks the pages again after OCR — adopting either would
  // pin boxPage at N and pile every box onto the last page.
  if (d.phase === 'ocr') {
    if (!state.ocrSeen) {
      state.ocrSeen = true;
      // stale pre-OCR advancement (e.g. rerun leftovers): restart box tracking
      if (state.boxPage !== 1 || state.pageBoxes.size > 0) {
        state.boxPage = 1;
        state.pageBoxes = new Map();
        if (state.followLive) state.viewPage = 1;
        updateLeftPane();
      }
    }
    const cur = Number(d.current_page) || 0;
    if (cur > state.boxPage) bumpBoxPageTo(cur); // per_page mode / chunk boundaries (no <PAGE> marker)
  }
  retryPageImageIfNeeded();
}

/* ============================ Live state ============================ */

function resetLiveState() {
  state.liveGen += 1;
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  state.previewDirty = false;
  state.previewFails = 0;
  state.previewAutoScroll = true;
  state.streamPending = '';
  state.streamPageNo = 1;
  state.streamAutoScroll = true;
  state.streamConnected = false;
  state.rawText = '';
  state.groundBuf = '';
  state.boxPage = 1;
  state.viewPage = 1;
  state.followLive = true;
  state.ocrSeen = false;
  state.totalPages = 0;
  state.pageBoxes = new Map();
  state.imgFailed = false;
  state.imgLastTry = 0;
  state.cancelRequestedFor = null;

  el.streamPane.textContent = '';
  el.livePreview.innerHTML = '';
  el.boxOverlay.textContent = '';
  el.pageImg.hidden = true;
  el.pageImg.removeAttribute('src');
  delete el.pageImg.dataset.url;
  el.pageNote.hidden = false;
  el.pageNote.textContent = '페이지 이미지 대기 중…';
  el.pagerLabel.textContent = '– / –';
  el.pagerPrev.disabled = true;
  el.pagerNext.disabled = true;
  el.followChip.hidden = true;
}

/* ============================ Raw stream (middle pane) ============================ */

function enqueueToken(text) {
  if (!text) return;
  state.streamPending += text;
  state.rawText += text;
  state.groundBuf += text;
  scheduleFlush();
}

function scheduleFlush() {
  if (state.rafId) return;
  state.rafId = requestAnimationFrame(() => {
    state.rafId = 0;
    flushStream(false);
    processGroundBuf(false);
    schedulePreviewRender();
  });
}

// Append pending stream text, converting <PAGE> markers into page-break dividers.
// A partial marker at the tail is held back (unless final) so it is never split.
function flushStream(final) {
  const buf = state.streamPending;
  state.streamPending = '';
  if (!buf) return;

  const frag = document.createDocumentFragment();
  let i = 0;
  while (true) {
    const idx = buf.indexOf(PAGE_MARKER, i);
    if (idx === -1) break;
    if (idx > i) frag.appendChild(document.createTextNode(buf.slice(i, idx)));
    state.streamPageNo += 1;
    frag.appendChild(makePageDivider(state.streamPageNo));
    i = idx + PAGE_MARKER.length;
  }

  let rest = buf.slice(i);
  if (!final && rest) {
    // hold back the longest suffix that could be the start of "<PAGE>"
    const maxCheck = Math.min(rest.length, PAGE_MARKER.length - 1);
    for (let k = maxCheck; k > 0; k -= 1) {
      if (PAGE_MARKER.startsWith(rest.slice(rest.length - k))) {
        state.streamPending = rest.slice(rest.length - k) + state.streamPending;
        rest = rest.slice(0, rest.length - k);
        break;
      }
    }
  }
  if (rest) frag.appendChild(document.createTextNode(rest));

  if (frag.childNodes.length) {
    el.streamPane.appendChild(frag);
    if (state.streamAutoScroll) el.streamPane.scrollTop = el.streamPane.scrollHeight;
  }
}

function makePageDivider(n) {
  return h('div', { class: 'stream-page-break' }, h('span', { class: 'spb-label', text: `페이지 ${n}` }));
}

function appendSystemLine(text, kind) {
  const line = h('div', { class: 'stream-sys' + (kind ? ' ' + kind : ''), text });
  el.streamPane.appendChild(line);
  if (state.streamAutoScroll) el.streamPane.scrollTop = el.streamPane.scrollHeight;
}

function onStreamScroll() {
  const pane = el.streamPane;
  state.streamAutoScroll = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;
}

/* ============================ Grounding parser (left pane) ============================ */
//
// The token stream is also fed into state.groundBuf. On every rAF flush we
// extract COMPLETE grounding matches (inline dets, ref blocks) and <PAGE>
// markers in positional order, then consume the buffer up to the last complete
// match. Whatever remains is kept only from the first potentially-incomplete
// structure onward (an unclosed <|ref|>/<|det|> block, an unterminated <|…,
// or a partial marker prefix at the tail), so matches split across SSE chunk
// boundaries are parsed exactly once, after they fully assemble.

function scanQuads(payload) {
  const nums = String(payload).match(/\d+/g);
  if (!nums) return [];
  const quads = [];
  for (let i = 0; i + 3 < nums.length; i += 4) {
    quads.push([Number(nums[i]), Number(nums[i + 1]), Number(nums[i + 2]), Number(nums[i + 3])]);
  }
  return quads;
}

// Index from which `s` may contain an incomplete grounding structure / marker.
// Returns s.length when the whole string is safe to consume. `cap` guards
// against holding back forever on a malformed block that never closes.
function incompleteTailIndex(s, cap) {
  const n = s.length;
  let cut = n;

  // an opened ref block that has not seen its closing <|/det|> yet
  const lastRef = s.lastIndexOf('<|ref|>');
  if (lastRef !== -1 && s.indexOf('<|/det|>', lastRef) === -1) cut = Math.min(cut, lastRef);

  // an opened det that has not closed yet
  const lastDet = s.lastIndexOf('<|det|>');
  if (lastDet !== -1 && s.indexOf('<|/det|>', lastDet + 7) === -1) cut = Math.min(cut, lastDet);

  // an unterminated special token: "<|" with no "|>" after it
  const lastPipe = s.lastIndexOf('<|');
  if (lastPipe !== -1 && s.indexOf('|>', lastPipe + 2) === -1) cut = Math.min(cut, lastPipe);

  // a partial literal prefix at the very tail (e.g. "<PA", "<|de", "<|/re")
  for (let k = Math.min(7, n); k > 0; k -= 1) {
    const tail = s.slice(n - k);
    let isPrefix = false;
    for (const mk of MARKER_LITERALS) {
      if (mk.length > k && mk.startsWith(tail)) { isPrefix = true; break; }
    }
    if (isPrefix) { cut = Math.min(cut, n - k); break; }
  }

  if (cap && n - cut > cap) return n; // stale/malformed opener: stop holding back
  return cut;
}

function processGroundBuf(final) {
  const buf = state.groundBuf;
  if (!buf) return;

  const events = [];
  let m;
  // inline dets: <|det|>label [x1,y1,x2,y2]<|/det|>
  const reDet = /<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*<\|\/det\|>/g;
  while ((m = reDet.exec(buf)) !== null) {
    events.push({
      start: m.index,
      end: reDet.lastIndex,
      label: m[1],
      quads: [[Number(m[2]), Number(m[3]), Number(m[4]), Number(m[5])]],
    });
  }
  // ref blocks: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2],...]<|/det|>
  const reRef = /<\|ref\|>([^<]{1,40})<\|\/ref\|><\|det\|>(\[\[?[\d,\s\[\]]*\]\]?)<\|\/det\|>/g;
  while ((m = reRef.exec(buf)) !== null) {
    events.push({ start: m.index, end: reRef.lastIndex, label: m[1].trim(), quads: scanQuads(m[2]) });
  }
  // page markers (advance the box page in stream order)
  let idx = -1;
  while ((idx = buf.indexOf(PAGE_MARKER, idx + 1)) !== -1) {
    events.push({ start: idx, end: idx + PAGE_MARKER.length, page: true });
  }

  events.sort((a, b) => a.start - b.start);
  let pos = 0;
  for (const ev of events) {
    if (ev.start < pos) continue;
    if (ev.page) bumpBoxPageTo(state.boxPage + 1);
    else if (ev.quads.length) addBoxes(state.boxPage, ev.label, ev.quads);
    pos = ev.end;
  }

  if (final) { state.groundBuf = ''; return; }
  const rest = buf.slice(pos);
  state.groundBuf = rest.slice(incompleteTailIndex(rest, 1200));
}

function labelColor(label) {
  const key = String(label || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  return BOX_COLORS[key] || BOX_FALLBACK_COLOR;
}

const clamp999 = (v) => Math.max(0, Math.min(999, Number(v) || 0));

function addBoxes(page, label, quads) {
  const boxes = [];
  for (const q of quads) {
    const x1 = clamp999(q[0]), y1 = clamp999(q[1]), x2 = clamp999(q[2]), y2 = clamp999(q[3]);
    if (x2 <= x1 || y2 <= y1) continue; // degenerate box
    boxes.push({ label, x1, y1, x2, y2 });
  }
  if (!boxes.length) return;
  let arr = state.pageBoxes.get(page);
  if (!arr) { arr = []; state.pageBoxes.set(page, arr); }
  for (const b of boxes) arr.push(b);
  if (page === state.viewPage) {
    const frag = document.createDocumentFragment();
    for (const b of boxes) frag.appendChild(makeBoxEl(b, true));
    el.boxOverlay.appendChild(frag);
  }
}

function makeBoxEl(b, animate) {
  const div = h('div', { class: 'gbox' + (animate ? ' gbox-in' : '') });
  div.style.left = `${(b.x1 / 999) * 100}%`;
  div.style.top = `${(b.y1 / 999) * 100}%`;
  div.style.width = `${((b.x2 - b.x1) / 999) * 100}%`;
  div.style.height = `${((b.y2 - b.y1) / 999) * 100}%`;
  div.style.setProperty('--gbox-c', labelColor(b.label));
  div.appendChild(h('span', { class: 'gbox-label', text: b.label }));
  return div;
}

// Advance the live parse page — driven by <PAGE> markers (multi mode) or
// progress.current_page (per_page mode / chunk boundaries), whichever moves
// first. Never goes backwards; capped at total_pages when known.
function bumpBoxPageTo(n) {
  const target = state.totalPages ? Math.min(n, state.totalPages) : n;
  if (target <= state.boxPage) return;
  state.boxPage = target;
  if (state.followLive) state.viewPage = target;
  updateLeftPane();
}

function pageImageUrl(id, n) {
  return `/api/jobs/${id}/files/pages/page_${String(n).padStart(4, '0')}.png`;
}

function updateLeftPane() {
  const id = state.currentJobId;
  if (!id) return;
  const total = Math.max(state.totalPages || 0, state.boxPage, 1);
  if (state.viewPage > total) state.viewPage = total;

  el.pagerLabel.textContent = `${state.viewPage} / ${total}`;
  el.pagerPrev.disabled = state.viewPage <= 1;
  el.pagerNext.disabled = state.viewPage >= total;
  el.followChip.hidden = state.followLive;

  const url = pageImageUrl(id, state.viewPage);
  if (el.pageImg.dataset.url !== url) {
    el.pageImg.dataset.url = url;
    state.imgFailed = false;
    el.pageImg.src = url; // visibility settled by the load/error handlers
  }
  renderOverlay();
}

function renderOverlay() {
  el.boxOverlay.textContent = '';
  const boxes = state.pageBoxes.get(state.viewPage);
  if (!boxes || !boxes.length) return;
  const frag = document.createDocumentFragment();
  for (const b of boxes) frag.appendChild(makeBoxEl(b, false));
  el.boxOverlay.appendChild(frag);
}

function pageNav(dir) {
  const total = Math.max(state.totalPages || 0, state.boxPage, 1);
  const next = Math.min(total, Math.max(1, state.viewPage + dir));
  if (next === state.viewPage) return;
  state.viewPage = next;
  state.followLive = next === state.boxPage; // paging away disables follow; reaching the live page re-enables
  updateLeftPane();
}

function onPageImgLoad() {
  state.imgFailed = false;
  el.pageImg.hidden = false;
  el.pageNote.hidden = true;
}

function onPageImgError() {
  if (!el.pageImg.dataset.url) return; // src was cleared on reset
  state.imgFailed = true;
  el.pageImg.hidden = true;
  el.pageNote.hidden = false;
  el.pageNote.textContent = '페이지 이미지 준비 중…';
}

// Page PNGs appear once the render phase finishes; retry quietly on progress.
function retryPageImageIfNeeded() {
  if (!state.imgFailed) return;
  const url = el.pageImg.dataset.url;
  if (!url) return;
  const now = Date.now();
  if (now - state.imgLastTry < 1500) return;
  state.imgLastTry = now;
  el.pageImg.src = `${url}?r=${now}`; // cache-bust the failed attempt
}

/* ============================ Live rendered preview (right pane) ============================ */

function isImageInlineLabel(inner) {
  const m = /^\s*([A-Za-z_][\w-]*)/.exec(String(inner));
  return !!m && m[1].toLowerCase() === 'image';
}

// Clean accumulated raw text for server-side rendering (order matters):
// tail holdback → ref blocks → inline dets → <PAGE> → stray specials.
function cleanRawText(raw, final) {
  let s = raw;
  if (!final) s = s.slice(0, incompleteTailIndex(s, 2000));
  s = s.replace(
    /<\|ref\|>([\s\S]{0,80}?)<\|\/ref\|><\|det\|>[\s\S]{0,400}?<\|\/det\|>/g,
    (mm, label) => (String(label).trim().toLowerCase() === 'image' ? IMG_PLACEHOLDER : ''),
  );
  s = s.replace(
    /<\|det\|>([^<]{0,160}?)<\|\/det\|>/g,
    (mm, inner) => (isImageInlineLabel(inner) ? IMG_PLACEHOLDER : ''),
  );
  s = s.replace(/<PAGE>/g, '\n\n---\n\n');
  s = s.replace(/<\|[^|>]{0,64}\|>/g, '');
  return s;
}

function schedulePreviewRender() {
  state.previewDirty = true;
  if (state.previewTimer || state.previewInFlight) return;
  state.previewTimer = setTimeout(runPreviewRender, 600);
}

function maybeReschedulePreview() {
  if (state.previewDirty && state.currentJobId && !state.previewTimer && !state.previewInFlight) {
    state.previewTimer = setTimeout(runPreviewRender, state.previewFails >= 4 ? 3000 : 600);
  }
}

// Throttled, latest-wins (queue of 1): at most one POST in flight; tokens
// arriving mid-flight mark it dirty and exactly one follow-up is scheduled.
async function runPreviewRender() {
  state.previewTimer = 0;
  if (state.previewInFlight || !state.previewDirty) return;
  const id = state.currentJobId;
  if (!id) { state.previewDirty = false; return; }
  const gen = state.liveGen;
  state.previewDirty = false;

  const cleaned = cleanRawText(state.rawText, false);
  if (!cleaned.trim()) { maybeReschedulePreview(); return; }

  state.previewInFlight = true;
  let html = null;
  try {
    const res = await fetch(`/api/jobs/${id}/render-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: cleaned,
    });
    if (res.ok) html = await res.text();
  } catch (_) { /* network error → retried on the next schedule */ }
  state.previewInFlight = false;
  state.previewFails = html == null ? state.previewFails + 1 : 0;

  if (html != null && state.currentJobId === id && state.liveGen === gen) {
    // Trusted server-rendered fragment (same renderer as /html).
    el.livePreview.innerHTML = html;
    if (state.previewAutoScroll) el.livePreview.scrollTop = el.livePreview.scrollHeight;
  }
  maybeReschedulePreview();
}

function onPreviewScroll() {
  const pane = el.livePreview;
  state.previewAutoScroll = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;
}

/* ============================ Cancel (STOP) ============================ */

function setStopButton(status) {
  const running = status === 'queued' || status === 'running';
  el.jobStop.hidden = !running;
  if (!running) return;
  const canceling = state.cancelRequestedFor === state.currentJobId;
  el.jobStop.disabled = canceling;
  el.jobStopLabel.textContent = canceling ? '취소 중…' : '정지';
}

async function requestCancel() {
  const id = state.currentJobId;
  if (!id) return;
  const status = state.displayedStatus;
  if (status !== 'queued' && status !== 'running') return;

  state.cancelRequestedFor = id;
  setStopButton(status);

  let ok = false;
  let gone = false;
  try {
    const res = await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
    ok = res.ok;
    gone = res.status === 404;
  } catch (_) { /* network error */ }

  if (state.currentJobId !== id) return;
  if (gone) {
    removeJobFromList(id);
    teardownConnections();
    state.currentJobId = null;
    state.displayedStatus = null;
    showEmptyState();
    showToast('해당 작업을 찾을 수 없습니다.', 'warn');
    return;
  }
  if (!ok) {
    state.cancelRequestedFor = null;
    setStopButton(state.displayedStatus);
    showToast('취소 요청에 실패했습니다.', 'error');
  }
  // success: the SSE error event (canceled:true) or status polling finalizes the UI
}

/* ============================ SSE + fallback ============================ */

function parseEventData(e) {
  if (!e || e.data == null) return null; // connection errors have no data
  return safeParse(e.data);
}

function teardownConnections() {
  if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
  if (state.fallbackTimer) { clearInterval(state.fallbackTimer); state.fallbackTimer = 0; }
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  state.previewDirty = false;
  state.fallbackActive = false;
  state.sseErrorCount = 0;
}

function startStream(id) {
  state.sseErrorCount = 0;
  state.fallbackActive = false;

  if (typeof EventSource === 'undefined') {
    appendSystemLine('이 브라우저는 실시간 스트림을 지원하지 않아 상태 폴링을 사용합니다.', 'warn');
    startFallbackPolling(id);
    return;
  }

  let es;
  try {
    es = new EventSource(`/api/jobs/${id}/events`);
  } catch (_) {
    startFallbackPolling(id);
    return;
  }
  state.es = es;

  es.addEventListener('open', () => {
    if (state.currentJobId !== id) return;
    state.sseErrorCount = 0;
    if (!state.streamConnected) {
      state.streamConnected = true;
      appendSystemLine('실시간 스트림에 연결되었습니다.');
    }
  });

  es.addEventListener('progress', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d) applyProgress(d);
  });

  es.addEventListener('token', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d && typeof d.text === 'string') enqueueToken(d.text);
  });

  es.addEventListener('done', (e) => {
    if (state.currentJobId !== id) return;
    onJobDone(id, parseEventData(e) || {});
  });

  es.addEventListener('error', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d) onJobError(id, d);      // server-sent job error (has JSON data)
    else handleSseConnError(id);   // transport-level error (no data)
  });
}

function handleSseConnError(id) {
  if (state.currentJobId !== id || state.fallbackActive) return;
  state.sseErrorCount += 1;
  if (state.sseErrorCount >= 2) {
    if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
    appendSystemLine('라이브 스트림을 사용할 수 없어 상태 폴링으로 전환했습니다.', 'warn');
    startFallbackPolling(id);
  }
}

function startFallbackPolling(id) {
  state.fallbackActive = true;
  if (state.fallbackTimer) clearInterval(state.fallbackTimer);
  state.fallbackTimer = setInterval(async () => {
    if (state.currentJobId !== id) { clearInterval(state.fallbackTimer); state.fallbackTimer = 0; return; }
    let job;
    try {
      job = await apiGet(`/api/jobs/${id}`);
    } catch (e) {
      if (e.status === 404) {
        clearInterval(state.fallbackTimer);
        state.fallbackTimer = 0;
        removeJobFromList(id);
        if (state.currentJobId === id) { state.currentJobId = null; showEmptyState(); }
      }
      return;
    }
    if (state.currentJobId !== id) return;
    if (isTerminal(job.status)) {
      clearInterval(state.fallbackTimer);
      state.fallbackTimer = 0;
      state.fallbackActive = false;
      flushStream(true);
      processGroundBuf(true);
      renderJob(job);
      refreshJobs();
    } else {
      applyProgress(Object.assign({}, job.progress || {}, { status: job.status }));
    }
  }, 1000);
}

async function onJobDone(id, data) {
  if (state.currentJobId !== id) return;
  flushStream(true);
  processGroundBuf(true);
  teardownConnections();
  state.displayedStatus = 'done';
  updateHeaderChip('done');
  setStopButton('done');

  let job = null;
  try {
    job = await apiGet(`/api/jobs/${id}`);
  } catch (_) { /* fall back to event data below */ }
  if (state.currentJobId !== id) return;

  if (job && job.result) {
    renderJob(job);
  } else {
    // Minimal render from the done event payload (URLs only).
    el.progressSection.hidden = true;
    el.errorSection.hidden = true;
    el.resultSection.hidden = false;
    el.liveDetails.open = false;
    const base = 'document';
    setDownload(el.dlMd, data.markdown_url, `${base}.md`);
    setDownload(el.dlZip, data.archive_url, `${base}.md.zip`);
    renderThumbGrid(el.layoutsGrid, [], '레이아웃 이미지를 불러오지 못했습니다.');
    renderThumbGrid(el.pagesGrid, [], '페이지 이미지를 불러오지 못했습니다.');
    state.previewLoaded = false;
    state.markdownLoaded = false;
    el.previewBody.innerHTML = '';
    el.mdCode.textContent = '';
    activateTab('preview');
  }
  refreshJobs();
}

function onJobError(id, d) {
  if (state.currentJobId !== id) return;
  flushStream(true);
  processGroundBuf(true);
  teardownConnections();
  const canceled = !!(d && d.canceled);
  state.displayedStatus = canceled ? 'canceled' : 'error';
  state.cancelRequestedFor = null;
  updateHeaderChip(state.displayedStatus);
  el.progressSection.hidden = true;
  el.errorSection.hidden = false;
  el.liveDetails.open = false;
  el.liveDetails.hidden = !hasLiveContent();
  setStopButton(state.displayedStatus);
  renderError(d && d.message, canceled);
  if (canceled) {
    // partial markdown stays available → offer the Markdown/미리보기 tabs
    el.resultSection.hidden = false;
    renderPartialResult({ job_id: id, filename: el.jobFilename.textContent || '' });
  } else {
    el.resultSection.hidden = true;
  }
  refreshJobs();
}

/* ============================ Result rendering ============================ */

function baseName(filename) {
  return String(filename || 'document').replace(/\.pdf$/i, '') || 'document';
}

function setDownload(anchor, url, downloadName) {
  if (url) {
    anchor.href = url;
    anchor.setAttribute('download', downloadName);
    anchor.classList.remove('disabled');
    anchor.removeAttribute('aria-disabled');
  } else {
    anchor.removeAttribute('href');
    anchor.classList.add('disabled');
    anchor.setAttribute('aria-disabled', 'true');
  }
}

function renderResult(job) {
  const r = job.result || {};
  const base = baseName(job.filename);
  setDownload(el.dlMd, r.markdown_url, `${base}.md`);
  setDownload(el.dlZip, r.archive_url, `${base}.md.zip`);

  renderThumbGrid(el.layoutsGrid, r.layouts, '레이아웃 이미지가 없습니다.');
  renderThumbGrid(el.pagesGrid, r.pages, '원본 페이지 이미지가 없습니다.');

  // reset lazy caches for the newly opened result
  state.previewLoaded = false;
  state.markdownLoaded = false;
  el.previewBody.innerHTML = '';
  el.mdCode.textContent = '';

  activateTab('preview');
}

// Canceled job: no result object, but partial markdown endpoints still work.
function renderPartialResult(job) {
  const id = job.job_id;
  const base = baseName(job.filename);
  setDownload(el.dlMd, `/api/jobs/${id}/markdown`, `${base}.partial.md`);
  setDownload(el.dlZip, null); // archive returns 409 for unfinished jobs

  renderThumbGrid(el.layoutsGrid, [], '취소된 작업에는 레이아웃 이미지가 제공되지 않습니다.');
  renderThumbGrid(el.pagesGrid, [], '취소된 작업에는 원본 페이지 목록이 제공되지 않습니다.');

  state.previewLoaded = false;
  state.markdownLoaded = false;
  el.previewBody.innerHTML = '';
  el.mdCode.textContent = '';

  activateTab('markdown');
}

function renderThumbGrid(grid, arr, emptyMsg) {
  grid.textContent = '';
  if (!Array.isArray(arr) || !arr.length) {
    grid.appendChild(h('p', { class: 'grid-empty muted', text: emptyMsg }));
    return;
  }
  arr.forEach((url, i) => {
    const img = h('img', { class: 'thumb-img', loading: 'lazy', decoding: 'async', alt: `${i + 1}번 이미지`, src: url });
    img.addEventListener('error', () => { img.replaceWith(h('span', { class: 'grid-empty muted', text: '로드 실패' })); });
    const a = h('a', { class: 'thumb', href: url, target: '_blank', rel: 'noopener', title: `${i + 1} — 새 탭에서 원본 열기` },
      img, h('span', { class: 'thumb-no', text: String(i + 1) }));
    grid.appendChild(a);
  });
}

/* ============================ Tabs ============================ */

function activateTab(name) {
  el.tabs.forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', on ? 'true' : 'false');
    t.tabIndex = on ? 0 : -1;
  });
  el.panels.forEach((p) => { p.hidden = p.dataset.panel !== name; });
  if (name === 'preview') loadPreview();
  else if (name === 'markdown') loadMarkdown();
}

async function loadPreview() {
  if (state.previewLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  el.previewBody.textContent = '';
  el.previewBody.appendChild(h('p', { class: 'muted', text: '미리보기를 불러오는 중…' }));
  let html;
  try {
    const res = await fetch(`/api/jobs/${id}/html`, { headers: { Accept: 'text/html' } });
    if (!res.ok) throw new Error(String(res.status));
    html = await res.text();
  } catch (_) {
    if (state.currentJobId === id) {
      el.previewBody.textContent = '';
      el.previewBody.appendChild(h('p', { class: 'muted', text: '미리보기를 불러오지 못했습니다.' }));
    }
    return;
  }
  if (state.currentJobId !== id) return;
  state.previewLoaded = true;
  // Trusted server-rendered fragment (/html, same renderer as /render-preview).
  el.previewBody.innerHTML = html;
}

async function loadMarkdown() {
  if (state.markdownLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  el.mdCode.textContent = '불러오는 중…';
  let text;
  try {
    const res = await fetch(`/api/jobs/${id}/markdown`, { headers: { Accept: 'text/markdown' } });
    if (!res.ok) throw new Error(String(res.status));
    text = await res.text();
  } catch (_) {
    if (state.currentJobId === id) el.mdCode.textContent = 'Markdown을 불러오지 못했습니다.';
    return;
  }
  if (state.currentJobId !== id) return;
  state.markdownLoaded = true;
  el.mdCode.textContent = text;
}

/* ============================ Error rendering ============================ */

function renderError(message, canceled) {
  el.errorSection.classList.toggle('canceled', !!canceled);
  el.errorTitle.textContent = canceled ? '취소됨' : '오류';
  el.errorMessage.textContent = message || (canceled ? '작업이 취소되었습니다.' : '변환 중 오류가 발생했습니다.');
  el.errorHint.textContent = canceled
    ? '중단 시점까지의 부분 결과를 아래 Markdown 탭에서 확인할 수 있습니다.'
    : '다시 시도하려면 왼쪽에서 PDF를 다시 업로드해 주세요.';
}

/* ============================ Upload ============================ */

function validateFile(file) {
  const name = file && file.name ? file.name : '';
  if (!/\.pdf$/i.test(name)) return 'PDF 파일만 업로드할 수 있습니다. (.pdf)';
  const type = file.type || '';
  if (type && !/pdf/i.test(type)) return '올바른 PDF 파일이 아닌 것 같습니다. 파일을 확인해 주세요.';
  return null;
}

function setSelectedFile(file) {
  const errMsg = validateFile(file);
  if (errMsg) {
    state.selectedFile = null;
    el.fileInfo.hidden = true;
    el.uploadBtn.disabled = true;
    showUploadError(errMsg);
    return;
  }
  hideUploadError();
  state.selectedFile = file;
  el.fileName.textContent = file.name;
  el.fileName.title = file.name;
  el.fileSize.textContent = fmtBytes(file.size);
  el.fileInfo.hidden = false;
  el.uploadBtn.disabled = false;
}

function clearSelectedFile() {
  state.selectedFile = null;
  el.fileInput.value = '';
  el.fileInfo.hidden = true;
  el.uploadBtn.disabled = true;
  hideUploadError();
}

function showUploadError(msg) {
  el.uploadError.textContent = msg;
  el.uploadError.hidden = false;
}
function hideUploadError() {
  el.uploadError.hidden = true;
  el.uploadError.textContent = '';
}

function setUploading(on) {
  el.uploadBtn.disabled = on || !state.selectedFile;
  el.uploadBtn.textContent = on ? '업로드 중…' : '변환 시작';
  el.dropzone.classList.toggle('disabled', on);
}

function readMode() {
  const checked = el.modeRadios.find((r) => r.checked);
  return checked ? checked.value : 'multi';
}

function readDpi() {
  let dpi = parseInt(el.dpiInput.value, 10);
  if (!Number.isFinite(dpi)) dpi = 200;
  dpi = Math.min(400, Math.max(72, dpi));
  el.dpiInput.value = String(dpi);
  return dpi;
}

async function handleUpload() {
  if (!state.selectedFile) return;
  hideUploadError();

  const file = state.selectedFile;
  const fileName = file.name;
  const mode = readMode();
  const dpi = readDpi();

  const form = new FormData();
  form.append('file', file);
  form.append('mode', mode);
  form.append('dpi', String(dpi));

  setUploading(true);
  let res;
  try {
    res = await fetch('/api/jobs', { method: 'POST', body: form });
  } catch (_) {
    setUploading(false);
    showUploadError('업로드 중 네트워크 오류가 발생했습니다. 다시 시도해 주세요.');
    return;
  }

  const text = await res.text().catch(() => '');
  const data = text ? safeParse(text) : null;

  if (!res.ok) {
    setUploading(false);
    if (res.status === 413) {
      showUploadError('파일이 너무 큽니다. 더 작은 PDF를 업로드해 주세요. (최대 100MB)');
    } else if (res.status === 400) {
      showUploadError((data && data.detail) || '유효하지 않은 PDF 파일입니다.');
    } else {
      showUploadError((data && data.detail) || `업로드에 실패했습니다. (${res.status})`);
    }
    return;
  }

  setUploading(false);
  const jobId = data && data.job_id;
  if (!jobId) {
    showUploadError('서버 응답이 올바르지 않습니다.');
    return;
  }

  // Optimistically add to history, then open + stream.
  upsertJob({
    job_id: jobId,
    filename: fileName,
    status: data.status || 'queued',
    mode,
    created_at: new Date().toISOString(),
    progress: {},
    result: null,
    error: null,
  });
  clearSelectedFile();
  await openJob(jobId);
  refreshJobs();
}

/* ============================ Dropzone wiring ============================ */

function setupDropzone() {
  const dz = el.dropzone;
  dz.addEventListener('click', () => el.fileInput.click());
  dz.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); el.fileInput.click(); }
  });
  dz.addEventListener('dragover', (ev) => { ev.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', (ev) => {
    if (ev.target === dz) dz.classList.remove('dragover');
  });
  dz.addEventListener('drop', (ev) => {
    ev.preventDefault();
    dz.classList.remove('dragover');
    const files = ev.dataTransfer && ev.dataTransfer.files;
    if (files && files.length) setSelectedFile(files[0]);
  });
  el.fileInput.addEventListener('change', () => {
    if (el.fileInput.files && el.fileInput.files.length) setSelectedFile(el.fileInput.files[0]);
  });
  el.fileClear.innerHTML = ICON.x;
  el.fileClear.addEventListener('click', clearSelectedFile);
}

/* ============================ Tabs / result wiring ============================ */

function setupTabs() {
  el.tabs.forEach((t) => {
    t.addEventListener('click', () => activateTab(t.dataset.tab));
  });
  // basic roving-tabindex keyboard nav
  const tablist = el.tabs.length ? el.tabs[0].parentElement : null;
  if (tablist) {
    tablist.addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft') return;
      const idx = el.tabs.findIndex((t) => t.classList.contains('active'));
      if (idx === -1) return;
      const dir = ev.key === 'ArrowRight' ? 1 : -1;
      const next = el.tabs[(idx + dir + el.tabs.length) % el.tabs.length];
      ev.preventDefault();
      activateTab(next.dataset.tab);
      next.focus();
    });
  }

  el.copyMd.addEventListener('click', async () => {
    const text = el.mdCode.textContent || '';
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        throw new Error('no clipboard');
      }
      el.copyMd.textContent = '복사됨';
      el.copyMd.classList.add('copied');
      setTimeout(() => { el.copyMd.textContent = '복사'; el.copyMd.classList.remove('copied'); }, 1600);
    } catch (_) {
      showToast('클립보드 복사에 실패했습니다.', 'error');
    }
  });
}

/* ============================ Init ============================ */

function init() {
  grabEls();
  setupTheme();
  setupDropzone();
  setupTabs();

  el.uploadBtn.addEventListener('click', handleUpload);
  el.streamPane.addEventListener('scroll', onStreamScroll, { passive: true });
  el.livePreview.addEventListener('scroll', onPreviewScroll, { passive: true });
  el.jobStop.addEventListener('click', requestCancel);
  el.jobDelete.addEventListener('click', () => {
    if (!state.currentJobId) return;
    armDelete(el.jobDelete, () => deleteJob(state.currentJobId));
  });
  el.pagerPrev.addEventListener('click', () => pageNav(-1));
  el.pagerNext.addEventListener('click', () => pageNav(1));
  el.followChip.addEventListener('click', () => {
    state.followLive = true;
    state.viewPage = state.boxPage;
    updateLeftPane();
  });
  el.pageImg.addEventListener('load', onPageImgLoad);
  el.pageImg.addEventListener('error', onPageImgError);

  showEmptyState();
  loadHealth();
  refreshJobs();
  state.jobsTimer = setInterval(refreshJobs, 5000);

  window.addEventListener('beforeunload', teardownConnections);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
