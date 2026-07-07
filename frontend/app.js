// Unlimited-OCR — PDF → Markdown : frontend logic
// Vanilla ES module. No external dependencies. Same-origin /api calls.
//
// Active-job live view = synchronized 3 panes fed by one SSE token stream:
//   left   : source page image + layout boxes parsed from grounding tokens
//   middle : raw token stream (with <PAGE> dividers)
//   right  : det-block structured markdown rendered via POST /render-preview
//
// 스트림 문법 (실캡처 확정, docs/ARCHITECTURE.md §5 / frontend/tests/fixtures):
//   각 페이지는 progress(phase=ocr, current_page=p)로 먼저 "선언"된 뒤
//   토큰 스트림의 <PAGE> 마커로 시작한다. 선언 직후의 첫 마커는 재확인(no-op)이고,
//   선언 없이 만나는 마커만 +1 이다. 블록 문법: <|det|>label [x1,y1,x2,y2]<|/det|>텍스트…
//
// 테스트: node --test frontend/tests/   (또는 frontend/ 에서: npm test)
//   픽스처 리플레이 테스트가 아래 "Pure live-stream core" 익스포트를 직접 임포트한다.

'use strict';

/* ============================================================================
 * Pure live-stream core — exported for frontend/tests/, no DOM access.
 * ========================================================================== */

export const PAGE_MARKER = '<PAGE>';
// literals whose partial prefix at a chunk boundary must be held back
const MARKER_LITERALS = ['<PAGE>', '<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>'];
const IMAGE_BLOCK = '> 🖼 그림 감지됨';
// noise labels dropped from the reading-view preview
const DROP_LABELS = new Set(['page_number', 'header', 'footer', 'footnote']);

export function normalizeLabel(label) {
  return String(label || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
}

export function scanQuads(payload) {
  const nums = String(payload).match(/\d+/g);
  if (!nums) return [];
  const quads = [];
  for (let i = 0; i + 3 < nums.length; i += 4) {
    quads.push([Number(nums[i]), Number(nums[i + 1]), Number(nums[i + 2]), Number(nums[i + 3])]);
  }
  return quads;
}

const clampCoord = (v) => Math.max(0, Math.min(999, Number(v) || 0));

// Index from which `s` may contain an incomplete grounding structure / marker.
// Returns s.length when the whole string is safe to consume. `cap` guards
// against holding back forever on a malformed block that never closes.
export function incompleteTailIndex(s, cap) {
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

// Marker/page state machine + grounding buffer. `page` is the page currently
// being parsed — every box attaches to it.
export function createGroundState() {
  return {
    buf: '',
    page: 1,
    // Job start: page 1 counts as pre-announced, so the very first <PAGE>
    // marker of the stream is consumed as its confirmation (no advance).
    expectAnnounce: true,
    ocrSeen: false,
    markerCount: 0,
    totalPages: 0,
  };
}

// Apply one progress event to the state machine.
// Only phase==="ocr" may drive page tracking: the render phase emits
// current_page=1..N in quick succession while rasterizing (before any token
// exists) and merge walks the pages again — adopting either would pin the
// page at N and pile every box onto the last page.
export function groundAnnounce(g, phase, currentPage, totalPages) {
  const out = { firstOcr: false, pageChanged: false, totalChanged: false };
  const total = Number(totalPages) || 0;
  if (total > g.totalPages) { g.totalPages = total; out.totalChanged = true; }
  if (phase !== 'ocr') return out;

  if (!g.ocrSeen) {
    g.ocrSeen = true;
    out.firstOcr = true;
    if (g.page !== 1) { g.page = 1; out.pageChanged = true; } // stale pre-OCR advancement guard
  }
  // The next <PAGE> marker is the start-of-page confirmation of this
  // announcement — it must not advance the page again.
  g.expectAnnounce = true;
  const cur = Number(currentPage) || 0;
  const target = g.totalPages ? Math.min(cur, g.totalPages) : cur;
  if (target > g.page) { g.page = target; out.pageChanged = true; } // never backwards
  return out;
}

export function groundPush(g, text) {
  if (text) g.buf += text;
}

// Drain the grounding buffer: emit COMPLETE det/ref matches and apply <PAGE>
// markers in positional order, then consume up to the last complete match.
// The remainder is kept only from the first potentially-incomplete structure
// onward (see incompleteTailIndex), so matches split across SSE chunk
// boundaries are parsed exactly once, after they fully assemble.
// Returns events: {type:'page', page} | {type:'boxes', page, label, boxes:[{x1,y1,x2,y2}]}
export function groundDrain(g, final) {
  const out = [];
  const buf = g.buf;
  if (!buf) return out;

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
  // page markers
  let idx = -1;
  while ((idx = buf.indexOf(PAGE_MARKER, idx + 1)) !== -1) {
    events.push({ start: idx, end: idx + PAGE_MARKER.length, page: true });
  }

  events.sort((a, b) => a.start - b.start);
  let pos = 0;
  for (const ev of events) {
    if (ev.start < pos) continue;
    if (ev.page) {
      g.markerCount += 1;
      if (g.expectAnnounce) {
        g.expectAnnounce = false; // start-of-page confirmation of the announced page
      } else {
        const next = g.totalPages ? Math.min(g.page + 1, g.totalPages) : g.page + 1;
        if (next > g.page) {
          g.page = next;
          out.push({ type: 'page', page: g.page });
        }
      }
    } else {
      const boxes = [];
      for (const q of ev.quads) {
        const x1 = clampCoord(q[0]), y1 = clampCoord(q[1]), x2 = clampCoord(q[2]), y2 = clampCoord(q[3]);
        if (x2 <= x1 || y2 <= y1) continue; // degenerate box
        boxes.push({ x1, y1, x2, y2 });
      }
      if (boxes.length) out.push({ type: 'boxes', page: g.page, label: ev.label, boxes });
    }
    pos = ev.end;
  }

  if (final) { g.buf = ''; return out; }
  const rest = buf.slice(pos);
  g.buf = rest.slice(incompleteTailIndex(rest, 1200));
  return out;
}

// Build STRUCTURED markdown from the raw stream for the live preview pane.
// The model carries structure only in det labels — a flat cleanup collapses
// everything into run-on paragraphs. Instead each det block becomes its own
// markdown block: title → "## ", image → placeholder blockquote, page
// furniture → dropped, everything else (text / raw <table> html / LaTeX) →
// its own paragraph. <PAGE> → "---" separator. Blank lines between blocks.
export function structurePreview(raw, final) {
  let s = raw;
  if (!final) s = s.slice(0, incompleteTailIndex(s, 2000));
  if (!s) return '';

  // structural tokens, position-ordered
  const toks = [];
  let m;
  const reDet = /<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*<\|\/det\|>/g;
  while ((m = reDet.exec(s)) !== null) toks.push({ start: m.index, end: reDet.lastIndex, label: m[1] });
  const reRef = /<\|ref\|>([^<]{1,40})<\|\/ref\|><\|det\|>\[\[?[\d,\s\[\]]*\]\]?<\|\/det\|>/g;
  while ((m = reRef.exec(s)) !== null) toks.push({ start: m.index, end: reRef.lastIndex, label: m[1].trim(), ref: true });
  let idx = -1;
  while ((idx = s.indexOf(PAGE_MARKER, idx + 1)) !== -1) {
    toks.push({ start: idx, end: idx + PAGE_MARKER.length, page: true });
  }
  toks.sort((a, b) => a.start - b.start);

  const parts = [];
  const pushSep = () => {
    if (parts.length && parts[parts.length - 1] !== '---') parts.push('---'); // no leading/duplicate hr
  };
  const pushBlock = (label, text) => {
    const key = normalizeLabel(label);
    if (DROP_LABELS.has(key)) return;
    if (key === 'image') { parts.push(IMAGE_BLOCK); return; }
    const body = String(text).replace(/<\|[^|>]{0,64}\|>/g, '').trim(); // strip stray specials
    if (!body) return;
    if (key === 'title') parts.push('## ' + body.replace(/\s*\n+\s*/g, ' '));
    else parts.push(body); // text / table(raw html) / equation(LaTeX literal) / unknown
  };

  let pos = 0;
  let currentLabel = null; // det label owning the text that follows it
  for (const t of toks) {
    if (t.start < pos) continue; // overlap safety
    pushBlock(currentLabel, s.slice(pos, t.start));
    if (t.page) {
      pushSep();
      currentLabel = null;
    } else if (normalizeLabel(t.label) === 'image') {
      pushBlock('image', '');
      currentLabel = null;
    } else if (t.ref) {
      currentLabel = null; // non-image ref: grounding only, no reading content
    } else {
      currentLabel = t.label;
    }
    pos = t.end;
  }
  pushBlock(currentLabel, s.slice(pos));

  if (final && parts.length && parts[parts.length - 1] === '---') parts.pop();
  return parts.join('\n\n');
}

/* ============================ UI constants ============================ */

const PHASE_LABELS = { render: '렌더링', ocr: 'OCR', merge: '병합' };
const STATUS_LABELS = {
  queued: '대기중',
  running: '변환중',
  done: '완료',
  error: '오류',
  canceled: '취소됨',
};
const THEME_KEY = 'uocr-theme';

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
  streamPageNo: 0, // markers seen by the raw pane — divider k reads "페이지 k"
  streamAutoScroll: true,
  streamConnected: false,
  rafId: 0,
  // accumulated raw model output (for the preview structurer)
  rawText: '',
  // grounding state machine (pure core) + left-pane view state
  ground: createGroundState(),
  viewPage: 1,          // page shown in the left pane
  followLive: true,
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
  docLayoutLoaded: false,
  markdownLoaded: false,
  // translation (완료된 잡 결과 화면 전용 — 잡 전환 시 초기화)
  currentLang: 'orig',      // 'orig' | 'ko' — 현재 결과 뷰 언어
  translateState: 'none',   // none|running|done|error|canceled
  translateEs: null,        // 번역 진행 EventSource
  translatePollTimer: 0,    // SSE 불가/실패 시 state 폴링 폴백
  translateSseErrors: 0,
  resultUrls: null,         // { markdown, archive, layoutHtml } — 언어별 다운로드 빌드용
  currentBaseName: 'document',
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
  dlLayout: 'dl-layout',
  translateBtn: 'translate-btn',
  translateProgress: 'translate-progress',
  translateProgressLabel: 'translate-progress-label',
  translateProgressTrack: 'translate-progress-track',
  translateProgressFill: 'translate-progress-fill',
  translateCancel: 'translate-cancel',
  langToggle: 'lang-toggle',
  langOrig: 'lang-orig',
  langKo: 'lang-ko',
  previewBody: 'preview-body',
  doclayoutBody: 'doclayout-body',
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
  return String(name || '')
    .replace(/^NVIDIA\s+GeForce\s+/i, '').replace(/^NVIDIA\s+/i, '')
    .replace(/^Apple\s+/i, '').trim();
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
  const isMetal = d.device === 'metal';
  const devName = isCuda ? 'CUDA' : (isMetal ? 'Metal' : (d.device === 'cpu' ? 'CPU' : String(d.device || '?').toUpperCase()));
  let devText = devName;
  if ((isCuda || isMetal) && d.gpu_name) {
    const short = shortenGpu(d.gpu_name);
    if (short) devText = `${devName} · ${short}`;
  }
  const devTitle = `디바이스: ${devName}` +
    (d.gpu_name ? ` (${d.gpu_name})` : '') +
    ` · dtype: ${d.dtype || '-'} · 네이티브 연산: ${d.native_ops ? 'on' : 'off'}`;
  const devClass = isCuda ? 'is-cuda' : (isMetal ? 'is-metal' : 'is-cpu');
  c.appendChild(h('span', { class: `badge badge-device ${devClass}`, title: devTitle },
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
  state.docLayoutLoaded = false;
  state.currentLang = 'orig'; // 잡이 바뀌면 언어 선택 초기화
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
    drainGroundToUI(true);
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
    const p = job.progress || {};
    updateProgress(p, job.status);
    // A status snapshot goes through the same announce machine as SSE
    // progress (phase-gated: an OCR snapshot seeds the page when opening a
    // job mid-OCR; render/merge snapshots must not pin it).
    const r = groundAnnounce(state.ground, p.phase, p.current_page, p.total_pages);
    if (r.firstOcr && state.pageBoxes.size > 0) state.pageBoxes = new Map();
    if (state.followLive) state.viewPage = state.ground.page;
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

// The progress BAR consumes every phase (render progress is real progress);
// page tracking for the left pane is delegated to groundAnnounce, which
// filters to phase==="ocr".
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

  // Drain buffered markers/boxes FIRST so content preceding this announcement
  // stays attributed to its own page, then apply the announcement.
  drainGroundToUI(false);
  const r = groundAnnounce(state.ground, d.phase, d.current_page, d.total_pages);
  if (r.firstOcr && state.pageBoxes.size > 0) {
    state.pageBoxes = new Map(); // stale pre-OCR boxes (rerun leftovers)
    renderOverlay();
  }
  if (r.pageChanged || r.firstOcr || r.totalChanged) {
    if (state.followLive) state.viewPage = state.ground.page;
    updateLeftPane();
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
  state.streamPageNo = 0;
  state.streamAutoScroll = true;
  state.streamConnected = false;
  state.rawText = '';
  state.ground = createGroundState();
  state.viewPage = 1;
  state.followLive = true;
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
  groundPush(state.ground, text);
  scheduleFlush();
}

function scheduleFlush() {
  if (state.rafId) return;
  state.rafId = requestAnimationFrame(() => {
    state.rafId = 0;
    flushStream(false);
    drainGroundToUI(false);
    schedulePreviewRender();
  });
}

// Append pending stream text, converting <PAGE> markers into page-break
// dividers. Divider k reads "페이지 k" — marker k announces page k (each
// page's stream segment BEGINS with its marker). A partial marker at the
// tail is held back (unless final) so it is never split.
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

/* ============================ Grounding → left pane ============================ */

function drainGroundToUI(final) {
  const events = groundDrain(state.ground, final);
  if (!events.length) return;
  let pageMoved = false;
  for (const ev of events) {
    if (ev.type === 'page') pageMoved = true;
    else addBoxes(ev.page, ev.label, ev.boxes);
  }
  if (pageMoved) {
    if (state.followLive) state.viewPage = state.ground.page;
    updateLeftPane();
  }
}

function labelColor(label) {
  return BOX_COLORS[normalizeLabel(label)] || BOX_FALLBACK_COLOR;
}

function addBoxes(page, label, boxes) {
  let arr = state.pageBoxes.get(page);
  if (!arr) { arr = []; state.pageBoxes.set(page, arr); }
  const labeled = boxes.map((b) => ({ label, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 }));
  for (const b of labeled) arr.push(b);
  if (page === state.viewPage) {
    const frag = document.createDocumentFragment();
    for (const b of labeled) frag.appendChild(makeBoxEl(b, true));
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

function pageImageUrl(id, n) {
  return `/api/jobs/${id}/files/pages/page_${String(n).padStart(4, '0')}.png`;
}

function updateLeftPane() {
  const id = state.currentJobId;
  if (!id) return;
  const g = state.ground;
  const total = Math.max(g.totalPages || 0, g.page, 1);
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
  const g = state.ground;
  const total = Math.max(g.totalPages || 0, g.page, 1);
  const next = Math.min(total, Math.max(1, state.viewPage + dir));
  if (next === state.viewPage) return;
  state.viewPage = next;
  state.followLive = next === g.page; // paging away disables follow; reaching the live page re-enables
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

  const structured = structurePreview(state.rawText, false);
  if (!structured.trim()) { maybeReschedulePreview(); return; }

  state.previewInFlight = true;
  let html = null;
  try {
    const res = await fetch(`/api/jobs/${id}/render-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: structured,
    });
    if (res.ok) html = await res.text();
  } catch (_) { /* network error → retried on the next schedule */ }
  state.previewInFlight = false;
  state.previewFails = html == null ? state.previewFails + 1 : 0;

  if (html != null && state.currentJobId === id && state.liveGen === gen) {
    // Trusted server-rendered fragment (same renderer as /html).
    el.livePreview.innerHTML = html;
    typesetMath(el.livePreview);
    if (state.previewAutoScroll) el.livePreview.scrollTop = el.livePreview.scrollHeight;
  }
  maybeReschedulePreview();
}

function onPreviewScroll() {
  const pane = el.livePreview;
  state.previewAutoScroll = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;
}

/* ── KaTeX 타이포셋 (로컬 벤더 — vendor/katex) ────────────────────────── */
// 서버 렌더러(render.py)가 tex를 이스케이프해 .math-inline/.math-display로
// 내보낸다. KaTeX 미로드(자산 누락 등) 시에는 raw LaTeX 텍스트가 그대로
// 보이는 그레이스풀 폴백.
function typesetMath(root) {
  if (!window.katex || !root) return;
  root.querySelectorAll('.math-inline, .math-display').forEach((elm) => {
    if (elm.dataset.mathDone) return;
    const tex = elm.textContent;
    try {
      window.katex.render(tex, elm, {
        displayMode: elm.classList.contains('math-display'),
        throwOnError: false,
      });
      elm.dataset.mathDone = '1';
    } catch (_) { /* 렌더 불가 tex는 원문 유지 */ }
  });
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
  teardownTranslate(); // 잡 전환·삭제·페이지 이탈 시 번역 구독도 함께 정리
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
      drainGroundToUI(true);
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
  drainGroundToUI(true);
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
    resetTranslateUI();
    const base = 'document';
    state.currentBaseName = base;
    state.resultUrls = {
      markdown: data.markdown_url,
      archive: data.archive_url,
      layoutHtml: `/api/jobs/${id}/layout.html`,
    };
    applyDownloadLangs();
    renderThumbGrid(el.layoutsGrid, [], '레이아웃 이미지를 불러오지 못했습니다.');
    renderThumbGrid(el.pagesGrid, [], '페이지 이미지를 불러오지 못했습니다.');
    state.previewLoaded = false;
    state.markdownLoaded = false;
    state.docLayoutLoaded = false;
    el.previewBody.innerHTML = '';
    el.doclayoutBody.innerHTML = '';
    el.mdCode.textContent = '';
    activateTab('preview');
    initTranslateForJob();
  }
  refreshJobs();
}

function onJobError(id, d) {
  if (state.currentJobId !== id) return;
  flushStream(true);
  drainGroundToUI(true);
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

  // 결과를 새로 렌더할 때마다 번역 UI를 원문 상태로 리셋한다
  // (이전 잡의 ko 선택/EventSource 구독 정리 + 언어 속성 제거).
  resetTranslateUI();

  state.currentBaseName = base;
  state.resultUrls = {
    markdown: r.markdown_url,
    archive: r.archive_url,
    layoutHtml: `/api/jobs/${job.job_id}/layout.html`,
  };
  applyDownloadLangs(); // currentLang='orig' → 원문 URL로 세팅

  renderThumbGrid(el.layoutsGrid, r.layouts, '레이아웃 이미지가 없습니다.');
  renderThumbGrid(el.pagesGrid, r.pages, '원본 페이지 이미지가 없습니다.');

  // reset lazy caches for the newly opened result
  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
  el.mdCode.textContent = '';

  activateTab('preview');

  // 번역 컨트롤은 완료(done) 잡에만 붙인다 (취소본은 renderPartialResult가 숨김 유지).
  if (job.status === 'done') initTranslateForJob();
}

// Canceled job: no result object, but partial markdown endpoints still work.
function renderPartialResult(job) {
  const id = job.job_id;
  const base = baseName(job.filename);
  resetTranslateUI(); // 취소본은 번역 대상이 아니다 (컨트롤 숨김)
  setDownload(el.dlMd, `/api/jobs/${id}/markdown`, `${base}.partial.md`);
  setDownload(el.dlZip, null); // archive returns 409 for unfinished jobs
  setDownload(el.dlLayout, `/api/jobs/${id}/layout.html`, `${base}.layout.html`); // 부분 레이아웃도 유효

  renderThumbGrid(el.layoutsGrid, [], '취소된 작업에는 레이아웃 이미지가 제공되지 않습니다.');
  renderThumbGrid(el.pagesGrid, [], '취소된 작업에는 원본 페이지 목록이 제공되지 않습니다.');

  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
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

/* ============================ Translation (한국어 번역) ============================
 * 완료된 잡 결과 화면 전용. 상태 머신:
 *   none/error/canceled → [한국어 번역] 버튼 (error/canceled는 재시도 의미)
 *   running             → 진행바 + 취소(✕), EventSource 재접속
 *   done                → [원문 | 한국어] 세그먼트 토글
 * 라이브 변환 뷰(result-section이 hidden)에는 렌더되지 않는다 — done일 때만 init.
 * ================================================================================ */

// URL 언어 파라미터 빌더 (순수 — 테스트 대상). ko가 아니면 원본 URL 그대로.
export function withLangUrl(url, lang) {
  if (lang !== 'ko' || !url) return url;
  return url + (url.indexOf('?') === -1 ? '?' : '&') + 'lang=ko';
}

// translate/state·POST 응답의 status → 노출할 UI (순수 — 테스트 대상).
export function translateUiStateFor(status) {
  if (status === 'running') return 'progress';
  if (status === 'done') return 'toggle';
  return 'button'; // none | error | canceled | 미지의 값 → 버튼(재시도)
}

function teardownTranslate() {
  if (state.translateEs) { try { state.translateEs.close(); } catch (_) { /* ignore */ } state.translateEs = null; }
  if (state.translatePollTimer) { clearInterval(state.translatePollTimer); state.translatePollTimer = 0; }
  state.translateSseErrors = 0;
}

// 번역 UI를 원문 기준으로 완전 초기화 (구독 정리 + 세 컨트롤 숨김 + 언어 속성 제거).
function resetTranslateUI() {
  teardownTranslate();
  state.currentLang = 'orig';
  state.translateState = 'none';
  el.translateBtn.hidden = true;
  el.translateProgress.hidden = true;
  el.langToggle.hidden = true;
  setLangSegActive('orig');
  setResultLangAttr();
}

/* ── 컨트롤 3종 교체 노출 ─────────────────────────────────────────────── */
function showTranslateButton() {
  el.translateBtn.hidden = false;
  el.translateProgress.hidden = true;
  el.langToggle.hidden = true;
  el.translateBtn.disabled = false;
}
function showTranslateProgress(current, total) {
  el.translateBtn.hidden = true;
  el.translateProgress.hidden = false;
  el.langToggle.hidden = true;
  el.translateCancel.disabled = false;
  updateTranslateProgress(current, total);
}
function showLangToggle() {
  el.translateBtn.hidden = true;
  el.translateProgress.hidden = true;
  el.langToggle.hidden = false;
}

function updateTranslateProgress(current, total) {
  const cur = Number(current) || 0;
  const tot = Number(total) || 0;
  el.translateProgressLabel.textContent = tot > 0 ? `번역 중 ${cur}/${tot}` : '번역 중…';
  const determinate = tot > 0;
  el.translateProgressTrack.classList.toggle('indeterminate', !determinate);
  el.translateProgressFill.style.width = determinate
    ? `${Math.min(100, Math.max(0, (cur / tot) * 100))}%`
    : '';
}

function setLangSegActive(lang) {
  const ko = lang === 'ko';
  el.langOrig.classList.toggle('active', !ko);
  el.langKo.classList.toggle('active', ko);
  el.langOrig.setAttribute('aria-pressed', ko ? 'false' : 'true');
  el.langKo.setAttribute('aria-pressed', ko ? 'true' : 'false');
}

// 문서/레이아웃 뷰 컨테이너의 lang="ko" 속성 토글 (CJK 조판 CSS 적용용).
function setResultLangAttr() {
  const ko = state.currentLang === 'ko';
  for (const node of [el.previewBody, el.doclayoutBody]) {
    if (!node) continue;
    if (ko) node.setAttribute('lang', 'ko');
    else node.removeAttribute('lang');
  }
}

// 현재 언어에 맞춰 다운로드 링크(markdown·layout.html)를 다시 세팅. 아카이브는
// ko 파일이 자동 포함되므로 원본 URL 그대로 둔다.
function applyDownloadLangs() {
  const u = state.resultUrls || {};
  const base = state.currentBaseName || 'document';
  const suffix = state.currentLang === 'ko' ? '.ko' : '';
  setDownload(el.dlMd, u.markdown ? withLangUrl(u.markdown, state.currentLang) : null, `${base}${suffix}.md`);
  setDownload(el.dlZip, u.archive || null, `${base}.md.zip`);
  setDownload(el.dlLayout, u.layoutHtml ? withLangUrl(u.layoutHtml, state.currentLang) : null, `${base}${suffix}.layout.html`);
}

// 결과 뷰 진입 시 번역 상태를 조회해 알맞은 컨트롤을 노출한다 (done 잡에서만 호출).
async function initTranslateForJob() {
  const id = state.currentJobId;
  if (!id) return;
  let st = null;
  try {
    st = await apiGet(`/api/jobs/${id}/translate/state?lang=ko`);
  } catch (_) { /* state 엔드포인트 불가 → 버튼 노출로 폴백 */ }
  if (state.currentJobId !== id) return;
  const status = (st && st.status) || 'none';
  state.translateState = status;
  const ui = translateUiStateFor(status);
  if (ui === 'progress') {
    showTranslateProgress(st && st.current, st && st.total);
    connectTranslateEvents(id); // 진행 중이던 번역에 재접속
  } else if (ui === 'toggle') {
    showLangToggle(); // 이미 번역 완료 → 토글만 노출(원문 기본, 사용자가 선택)
  } else {
    showTranslateButton();
  }
}

// [한국어 번역] 클릭 → 번역 시작.
async function startTranslate() {
  const id = state.currentJobId;
  if (!id) return;
  el.translateBtn.disabled = true;
  let res = null;
  let data = null;
  try {
    res = await fetch(`/api/jobs/${id}/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ lang: 'ko', force: false }),
    });
    const text = await res.text().catch(() => '');
    data = text ? safeParse(text) : null;
  } catch (_) {
    if (state.currentJobId !== id) return;
    el.translateBtn.disabled = false;
    showToast('번역 요청 중 네트워크 오류가 발생했습니다.', 'error');
    return;
  }
  if (state.currentJobId !== id) return;

  if (!res.ok) {
    el.translateBtn.disabled = false;
    const detail = (data && typeof data.detail === 'string') ? data.detail : null;
    if (res.status === 503) showToast(detail || '번역 프로바이더가 설정되지 않았습니다.', 'error');
    else if (res.status === 409) showToast(detail || '아직 완료되지 않은 작업은 번역할 수 없습니다.', 'warn');
    else if (res.status === 400) showToast(detail || '지원하지 않는 번역 언어입니다.', 'error');
    else showToast(detail || `번역 요청에 실패했습니다. (${res.status})`, 'error');
    return;
  }

  // 202/200 — 이미 번역돼 있으면(done) 바로 토글, 아니면 진행 UI + 구독.
  state.translateState = 'running';
  if (data && data.status === 'done') { onTranslateDone(id); return; }
  showTranslateProgress(0, 0);
  connectTranslateEvents(id);
}

function connectTranslateEvents(id) {
  teardownTranslate(); // 중복 구독 방지
  if (typeof EventSource === 'undefined') { startTranslatePolling(id); return; }
  let es;
  try {
    es = new EventSource(`/api/jobs/${id}/translate/events?lang=ko`);
  } catch (_) { startTranslatePolling(id); return; }
  state.translateEs = es;

  es.addEventListener('progress', (e) => {
    if (state.currentJobId !== id) return;
    state.translateSseErrors = 0;
    const d = parseEventData(e);
    if (d) { state.translateState = 'running'; showTranslateProgress(d.current, d.total); }
  });
  es.addEventListener('done', (e) => {
    if (state.currentJobId !== id) return;
    onTranslateDone(id);
  });
  es.addEventListener('error', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d) onTranslateError(id, d);        // 서버가 보낸 번역 오류(JSON)
    else handleTranslateConnError(id);     // 전송 계층 오류(데이터 없음)
  });
}

function handleTranslateConnError(id) {
  if (state.currentJobId !== id || !state.translateEs) return;
  state.translateSseErrors += 1;
  if (state.translateSseErrors >= 2) { teardownTranslate(); startTranslatePolling(id); }
}

// SSE 불가/불안정 시 state를 폴링해 진행/완료/오류를 반영하는 폴백.
function startTranslatePolling(id) {
  teardownTranslate();
  state.translatePollTimer = setInterval(async () => {
    if (state.currentJobId !== id) { clearInterval(state.translatePollTimer); state.translatePollTimer = 0; return; }
    let st;
    try { st = await apiGet(`/api/jobs/${id}/translate/state?lang=ko`); }
    catch (_) { return; }
    if (state.currentJobId !== id) return;
    const status = st && st.status;
    if (status === 'running') { showTranslateProgress(st.current, st.total); return; }
    clearInterval(state.translatePollTimer); state.translatePollTimer = 0;
    if (status === 'done') onTranslateDone(id);
    else if (status === 'error') onTranslateError(id, { message: st.error });
    else if (status === 'canceled') onTranslateError(id, { canceled: true });
    else showTranslateButton();
  }, 1500);
}

function onTranslateDone(id) {
  if (state.currentJobId !== id) return;
  teardownTranslate();
  state.translateState = 'done';
  showLangToggle();
  setLang('ko'); // 완료 직후 자동으로 한국어 뷰로 전환
}

function onTranslateError(id, d) {
  if (state.currentJobId !== id) return;
  teardownTranslate();
  const canceled = !!(d && d.canceled);
  state.translateState = canceled ? 'canceled' : 'error';
  showTranslateButton(); // 버튼 복원(재시도 가능)
  if (canceled) showToast('번역이 취소되었습니다.', 'warn');
  else showToast((d && d.message) || '번역 중 오류가 발생했습니다.', 'error');
}

// 취소(✕) — 요청만 보내고, UI 확정은 error(canceled) 이벤트/폴링에 맡긴다.
async function cancelTranslate() {
  const id = state.currentJobId;
  if (!id) return;
  el.translateCancel.disabled = true;
  let ok = false;
  try {
    const res = await fetch(`/api/jobs/${id}/translate/cancel?lang=ko`, { method: 'POST' });
    ok = res.ok || res.status === 404;
  } catch (_) { /* 네트워크 오류 */ }
  if (state.currentJobId !== id) return;
  if (!ok) {
    el.translateCancel.disabled = false;
    showToast('번역 취소 요청에 실패했습니다.', 'error');
  }
  // 성공: error(canceled) 이벤트 또는 state 폴링이 버튼을 복원한다.
}

// 언어 토글. 캐시를 무효화하고 현재 탭을 새 언어로 다시 로드한다.
function setLang(lang) {
  const next = lang === 'ko' ? 'ko' : 'orig';
  setLangSegActive(next);
  if (state.currentLang === next) return;
  state.currentLang = next;
  setResultLangAttr();
  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
  el.mdCode.textContent = '';
  applyDownloadLangs();
  reloadActiveResultTab();
}

// 번역본 fetch가 404/실패일 때 조용히 원문으로 되돌린다 (호출부가 재로드).
function revertToOriginal(reason) {
  if (state.currentLang !== 'ko') return false;
  state.currentLang = 'orig';
  setLangSegActive('orig');
  setResultLangAttr();
  applyDownloadLangs();
  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
  el.mdCode.textContent = '';
  showToast(reason || '번역본을 불러오지 못해 원문을 표시합니다.', 'warn');
  return true;
}

// 현재 활성 결과 탭만 다시 로드 (썸네일 탭은 언어 무관 → 스킵).
function reloadActiveResultTab() {
  const active = el.tabs.find((t) => t.classList.contains('active'));
  const name = active ? active.dataset.tab : 'preview';
  if (name === 'preview') loadPreview();
  else if (name === 'markdown') loadMarkdown();
  else if (name === 'doclayout') loadDocLayout();
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
  else if (name === 'doclayout') loadDocLayout();
}

async function loadDocLayout() {
  if (state.docLayoutLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  const lang = state.currentLang; // 응답 도착 시점에 언어가 바뀌었는지 판별용
  el.doclayoutBody.textContent = '';
  el.doclayoutBody.appendChild(h('p', { class: 'muted', text: '레이아웃을 불러오는 중…' }));
  let html = null;
  let missing = false;
  try {
    const res = await fetch(withLangUrl(`/api/jobs/${id}/layout`, lang), { headers: { Accept: 'text/html' } });
    if (res.status === 404) missing = true;
    else if (res.ok) html = await res.text();
  } catch (_) { /* 아래 공통 실패 처리 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return; // 잡/언어 전환 → 최신 로더에 위임
  // 한국어 뷰에서 번역본을 못 받으면(404·실패) 조용히 원문으로 폴백 + 토스트.
  if ((missing || html == null) && lang === 'ko' &&
      revertToOriginal('한국어 레이아웃을 불러오지 못해 원문을 표시합니다.')) {
    loadDocLayout();
    return;
  }
  el.doclayoutBody.textContent = '';
  if (missing) {
    state.docLayoutLoaded = true; // 404는 재시도해도 같음
    el.doclayoutBody.appendChild(h('p', {
      class: 'muted',
      text: '이 작업에는 레이아웃 데이터가 없습니다 (이 기능 추가 이전에 변환된 결과).',
    }));
    return;
  }
  if (html == null) {
    el.doclayoutBody.appendChild(h('p', { class: 'muted', text: '레이아웃 뷰를 불러오지 못했습니다.' }));
    return;
  }
  state.docLayoutLoaded = true;
  // Trusted server-rendered fragment (pipeline/layout.py — 텍스트 전부 이스케이프됨).
  // 번역본은 루트에 lang="ko"가 붙어 오지만, 컨테이너에도 setResultLangAttr로 반영해 둔다.
  el.doclayoutBody.innerHTML = html;
  typesetMath(el.doclayoutBody);
  if (window.uocrFitLayout) window.uocrFitLayout(el.doclayoutBody);
}

async function loadPreview() {
  if (state.previewLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  const lang = state.currentLang;
  el.previewBody.textContent = '';
  el.previewBody.appendChild(h('p', { class: 'muted', text: '미리보기를 불러오는 중…' }));
  let html = null;
  try {
    const res = await fetch(withLangUrl(`/api/jobs/${id}/html`, lang), { headers: { Accept: 'text/html' } });
    if (res.ok) html = await res.text();
  } catch (_) { /* 아래 공통 실패 처리 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return;
  if (html == null) {
    // 한국어 뷰에서 번역본을 못 받으면 조용히 원문으로 폴백.
    if (lang === 'ko' && revertToOriginal('한국어 미리보기를 불러오지 못해 원문을 표시합니다.')) { loadPreview(); return; }
    el.previewBody.textContent = '';
    el.previewBody.appendChild(h('p', { class: 'muted', text: '미리보기를 불러오지 못했습니다.' }));
    return;
  }
  state.previewLoaded = true;
  // Trusted server-rendered fragment (/html, same renderer as /render-preview).
  el.previewBody.innerHTML = html;
  typesetMath(el.previewBody);
}

async function loadMarkdown() {
  if (state.markdownLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  const lang = state.currentLang;
  el.mdCode.textContent = '불러오는 중…';
  let text = null;
  try {
    const res = await fetch(withLangUrl(`/api/jobs/${id}/markdown`, lang), { headers: { Accept: 'text/markdown' } });
    if (res.ok) text = await res.text();
  } catch (_) { /* 아래 공통 실패 처리 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return;
  if (text == null) {
    if (lang === 'ko' && revertToOriginal('한국어 Markdown을 불러오지 못해 원문을 표시합니다.')) { loadMarkdown(); return; }
    el.mdCode.textContent = 'Markdown을 불러오지 못했습니다.';
    return;
  }
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

  // 번역 컨트롤
  el.translateCancel.innerHTML = ICON.x;
  el.translateBtn.addEventListener('click', startTranslate);
  el.translateCancel.addEventListener('click', cancelTranslate);
  el.langOrig.addEventListener('click', () => setLang('orig'));
  el.langKo.addEventListener('click', () => setLang('ko'));
  el.pagerPrev.addEventListener('click', () => pageNav(-1));
  el.pagerNext.addEventListener('click', () => pageNav(1));
  el.followChip.addEventListener('click', () => {
    state.followLive = true;
    state.viewPage = state.ground.page;
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

// Browser bootstrap only — the module is also imported by frontend/tests/
// under Node, where no DOM exists (only the exported pure core is used).
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}
