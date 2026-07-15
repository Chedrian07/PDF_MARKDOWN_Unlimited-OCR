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

// ── 라이브 프리뷰 증분 분할 (순수 — frontend/tests/에서 직접 검증) ──────────
// raw를 <PAGE> 마커 기준으로 "확정 페이지(뒤에 새 페이지가 시작된 세그먼트)"와
// "미확정 꼬리"로 나눈다. 마커가 조각나 도착하면(<PA + GE>) 완성되기 전까지
// 꼬리에 남는다. raw는 잡 안에서 append-only라 확정 세그먼트는 불변 → 캐시 가능.
export function splitPreviewPages(raw) {
  const pages = [];
  let pos = 0;
  let idx;
  while ((idx = raw.indexOf(PAGE_MARKER, pos)) !== -1) {
    pages.push(raw.slice(pos, idx));
    pos = idx + PAGE_MARKER.length;
  }
  return { pages, tail: raw.slice(pos) };
}

// 이번 사이클에 렌더해야 할 조각 계산: 캐시에 없는 확정 페이지들의 markdown과
// 꼬리 markdown. sep은 "앞에 렌더된 내용이 있으면 페이지 경계 hr을 붙여라" —
// 전체 텍스트 structurePreview의 pushSep(선두/중복 hr 억제)과 동치다.
// cachedHtmls: 확정 페이지별 렌더 HTML 캐시 (빈 문자열 = 내용 없는 페이지).
export function planPreviewRender(raw, cachedHtmls, lastTailMd, lastTailSep) {
  const { pages, tail } = splitPreviewPages(raw);
  let hasBefore = cachedHtmls.some((html) => !!html);
  const newPages = [];
  for (let i = cachedHtmls.length; i < pages.length; i += 1) {
    const md = structurePreview(pages[i], true); // 확정 세그먼트는 완결 — 홀드백 불필요
    newPages.push({ idx: i, md, sep: !!md && hasBefore });
    if (md) hasBefore = true;
  }
  const tailMd = structurePreview(tail, false);
  const tailSep = !!tailMd && hasBefore;
  // 꼬리 md가 같아도 sep이 바뀌면(앞에 내용 있는 페이지가 확정) 재렌더 대상
  const tailChanged = tailMd !== lastTailMd || tailSep !== !!lastTailSep;
  return { newPages, tailMd, tailSep, tailChanged };
}

// ── SSE 폴링 강등 → 재승격 백오프 (순수 — frontend/tests/에서 직접 검증) ──────
// 강등 후 attempt번째(0부터) 재시도까지 기다릴 지연: 10초 → 20초 → 30초 상한.
export function ssePromoteDelay(attempt) {
  return Math.min(30000, 10000 * ((Number(attempt) || 0) + 1));
}

// raw pane 디바이더 번호(streamPageNo)를 ground 상태머신의 페이지로 재동기화.
// 디바이더 k는 "페이지 k"이고 마커 k가 페이지 k를 시작하므로, streamPageNo는
// "다음 마커가 시작할 페이지 - 1"이어야 한다: 선언 대기 중(expectAnnounce)이면
// 다음 마커는 g.page의 시작 확인이라 g.page-1, 아니면 g.page+1을 시작하라 g.page.
// 재연결 갭으로 마커가 유실돼도 ground.page는 progress 선언(폴링 포함)으로
// 따라가므로 이 보정으로 이후 디바이더 번호가 복구된다. 절대 뒤로 가지 않는다.
export function syncedStreamPageNo(streamPageNo, g) {
  const target = g.expectAnnounce ? g.page - 1 : g.page;
  return Math.max(Number(streamPageNo) || 0, target);
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

// 잡 상태 라벨 (순수 — frontend/tests/에서 직접 검증). queued 잡에 선택 필드
// queue_position(1-base, 큐 앞의 queued 잡 수 + 1)이 있으면 '대기중 · N번째'.
// 필드 부재(구버전 서버·SSE 스냅샷)·비정상 값은 기존 라벨 그대로 — 안전 폴백.
export function statusLabel(job) {
  const status = (job && job.status) || 'queued';
  const base = STATUS_LABELS[status] || status;
  if (status === 'queued') {
    const pos = job && job.queue_position;
    if (Number.isInteger(pos) && pos >= 1) return `${base} · ${pos}번째`;
  }
  return base;
}
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
  queuePos: null, // 열린 잡의 마지막 대기열 위치 — queued가 아니게 되면 해제
  selectedFiles: [], // 다중 선택 지원 — 검증을 통과한 파일들만 담긴다
  uploading: false, // 업로드 루프 재진입 가드 — 진행 중 새 선택이 버튼을 되살리지 않게
  // /api/health 스냅샷 — 필드 부재·미수신 시 undefined (검증·비활성은 fail-open)
  maxUploadMb: undefined,
  translateAvailable: undefined,
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
  previewStopped: false,   // 413/연속 실패로 라이브 프리뷰 중단됨
  previewPageCache: [],    // 확정 페이지 렌더 HTML 캐시 (인덱스 = 확정 페이지 순번)
  previewTailNodes: [],    // 현재 꼬리 렌더가 소유한 DOM 노드들 (선행 hr 포함)
  previewTailMd: '',       // 마지막으로 렌더된 꼬리 markdown
  previewTailSep: false,   // 마지막 꼬리 렌더의 선행 hr 유무
  previewAutoScroll: true,
  // cancel
  cancelRequestedFor: null,
  // sse / fallback
  es: null,
  sseErrorCount: 0,
  fallbackActive: false,
  fallbackTimer: 0,
  ssePromoteTimer: 0,     // 폴링 강등 후 SSE 재승격 재시도 타이머
  ssePromoteAttempts: 0,  // 재승격 백오프 단계 — 성공(open)·teardown 시 0으로
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

// 2단계 삭제 확인의 무장(armed) 상태: key → { t: 만료 타이머, btn: 현재 버튼 }.
// key는 목록 항목이면 잡 id, 헤더 버튼이면 'header:<잡 id>' — DOM 버튼이 아니라
// 키로 관리해 5초 주기 재렌더(renderJobList)에도 무장이 유지된다.
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
  uploadProgress: 'upload-progress',
  uploadProgressFill: 'upload-progress-fill',
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
  // 업로드 사전 검증·번역 버튼 가용성이 소비하는 계약 필드 보관.
  // 구버전 서버 응답(필드 부재)은 undefined 유지 — 두 소비처 모두 fail-open.
  state.maxUploadMb = typeof d.max_upload_mb === 'number' ? d.max_upload_mb : undefined;
  state.translateAvailable = typeof d.translate_available === 'boolean' ? d.translate_available : undefined;
  applyTranslateAvailability(); // 잡 뷰가 열려 있는 동안의 health 갱신도 버튼에 반영

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
      noteQueuePosition(open.status, open.queue_position);
      updateHeaderChip(open.status);
      // queued 동안은 SSE progress가 없어 이 5초 목록 폴링이 유일한 대기열 위치
      // 갱신원이다 — 진행 영역의 '대기중 · N번째' 문구도 여기서 함께 갱신한다.
      if (open.status === 'queued' && state.displayedStatus === 'queued') {
        updateProgress(open.progress || {}, 'queued');
      }
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
  const fname = job.filename || '(이름 없음)';

  const name = h('span', { class: 'ji-name', text: fname, title: job.filename || '' });
  const chip = h('span', { class: `chip chip-${status}`, text: statusLabel(job) });
  const time = h('span', { class: 'ji-time muted', text: fmtTime(job.created_at) });
  const sub = h('span', { class: 'ji-sub' }, chip, time);

  // 잡 열기·삭제를 형제 버튼으로 분리 — role="button" li 안에 버튼을 중첩하면
  // 스크린리더가 내부 삭제 버튼에 진입할 수 없다(중첩 인터랙티브 컨트롤 금지).
  const open = h('button', { class: 'ji-open', type: 'button' },
    h('span', { class: 'ji-main' }, name, sub));
  open.addEventListener('click', () => openJob(job.job_id));

  const del = h('button', {
    class: 'ji-del icon-btn-sm', type: 'button',
    'aria-label': `"${fname}" 삭제`, title: '삭제', html: ICON.x,
  });
  del.addEventListener('click', () => armDelete(del, job.job_id, () => deleteJob(job.job_id)));
  // 재렌더가 무장(armed) 상태를 파괴하지 않도록 살아있는 무장을 새 버튼에 복원.
  // 만료 타이머가 최신 버튼을 해제하도록 참조도 교체한다.
  const arm = armTimers.get(job.job_id);
  if (arm) {
    arm.btn = del;
    del.dataset.baseTitle = del.title;
    del.classList.add('armed');
    del.title = '한 번 더 클릭하면 삭제됩니다';
  }

  return h('li', { class: `job-item${active ? ' active' : ''}` }, open, del);
}

// 2단계 삭제 확인의 클릭 전이 (순수 — 테스트 대상). entries는 [key, owner] 쌍의
// 이터러블(owner = 물리 버튼), key는 이번 클릭의 키, owner는 클릭된 버튼.
// 클릭된 key가 이미 무장돼 있으면 confirm(실삭제), 아니면 같은 owner에 남은
// 옛 무장(잡 전환 뒤의 헤더 버튼 등)을 회수(clearKeys)하고 새로 무장한다.
export function armTransition(entries, key, owner) {
  const clearKeys = [];
  let confirm = false;
  for (const [k, o] of entries) {
    if (k === key) confirm = true;
    else if (o === owner) clearKeys.push(k);
  }
  if (confirm) clearKeys.push(key);
  return { confirm, clearKeys };
}

function disarmDeleteBtn(btn) {
  btn.classList.remove('armed');
  btn.title = btn.dataset.baseTitle || '삭제';
}

function armDelete(btn, key, onConfirm) {
  const { confirm, clearKeys } = armTransition(
    Array.from(armTimers, ([k, e]) => [k, e.btn]), key, btn);
  for (const k of clearKeys) {
    const e = armTimers.get(k);
    if (e) clearTimeout(e.t);
    armTimers.delete(k);
  }
  if (confirm) {
    disarmDeleteBtn(btn);
    onConfirm();
    return;
  }
  if (!btn.dataset.baseTitle) btn.dataset.baseTitle = btn.title || '삭제';
  btn.classList.add('armed');
  btn.title = '한 번 더 클릭하면 삭제됩니다';
  const t = setTimeout(() => {
    const e = armTimers.get(key);
    armTimers.delete(key);
    if (e) disarmDeleteBtn(e.btn); // 재렌더로 교체됐어도 최신 버튼을 해제
  }, 2600);
  armTimers.set(key, { t, btn });
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
    syncJobHash(null); // 삭제된 잡을 가리키는 해시 정리
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
  el.jobChip.textContent = statusLabel({ status, queue_position: state.queuePos });
}

// 잡 JSON/진행 페이로드의 대기열 위치를 상태에 흡수. queued가 아니면 해제하고,
// queued인데 필드가 없으면(SSE 스냅샷·구버전 서버) 마지막 값을 유지한다 —
// 계약상 필드 부재는 "기존 표시 그대로"가 안전 폴백이다.
function noteQueuePosition(status, pos) {
  if (status !== 'queued') state.queuePos = null;
  else if (Number.isInteger(pos) && pos >= 1) state.queuePos = pos;
}

/* ============================ location.hash 잡 복원 ============================ */

// location.hash → 잡 id (순수 — frontend/tests/에서 직접 검증).
// '#abc' → 'abc'. 빈 해시·잡 id에 쓰이지 않는 문자가 섞인 이상값은 null.
export function jobIdFromHash(hash) {
  const id = String(hash || '').replace(/^#/, '');
  return /^[\w-]+$/.test(id) ? id : null;
}

// 현재 잡을 주소창 해시에 반영 — 새로고침 복원·영속 링크용. replaceState라
// 히스토리 스택을 오염시키지 않고 hashchange도 발생하지 않는다(자기 변경 루프
// 없음). id=null이면 해시 제거 — 잡 삭제·404로 현재 잡이 사라진 경우.
function syncJobHash(id) {
  try {
    if (id) history.replaceState(null, '', '#' + id);
    else if (location.hash) history.replaceState(null, '', location.pathname + location.search);
  } catch (_) { /* ignore */ }
}

/* ============================ Open / render a job ============================ */

async function openJob(id) {
  if (!id || id === state.currentJobId) return;

  teardownConnections();
  state.currentJobId = id;
  // 잡 전환 시 이전 잡의 헤더 삭제 무장 잔상 제거 — 기능상 armTransition이 키
  // 불일치로 confirm을 거부하지만, armed 시각 표시가 남으면 거짓 안내가 된다.
  for (const [k, e] of armTimers) {
    if (k.startsWith('header:') && k !== `header:${id}`) {
      clearTimeout(e.t);
      armTimers.delete(k);
      disarmDeleteBtn(e.btn);
    }
  }
  state.displayedStatus = null;
  state.queuePos = null; // 이전 잡의 대기열 위치가 새 잡 칩에 새지 않도록
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
      syncJobHash(null); // 사라진 잡을 가리키는 해시(공유 링크 등) 정리
    } else {
      showToast('작업 정보를 불러오지 못했습니다.', 'error');
    }
    state.currentJobId = null;
    showEmptyState();
    return;
  }
  if (state.currentJobId !== id) return; // user switched away during await

  syncJobHash(id); // 성공 경로에서만 해시 동기화 — 새로고침 복원·링크 공유
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
  noteQueuePosition(job.status, job.queue_position);

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

  // queued 문구는 헤더/목록 칩과 동일 조합('대기중 · N번째')으로 통일
  el.progressPhase.textContent = queued
    ? statusLabel({ status: 'queued', queue_position: state.queuePos })
    : (PHASE_LABELS[p.phase] || '처리 중');
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
  noteQueuePosition(status, d.queue_position);
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
  state.previewStopped = false;
  state.previewPageCache = [];
  state.previewTailNodes = [];
  state.previewTailMd = '';
  state.previewTailSep = false;
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
    // pane이 방금 pending 마커를 전부 소화한 시점 — 재연결 갭 등으로 마커가
    // 유실됐다면 ground 페이지 기준으로 다음 디바이더 번호를 재동기화한다.
    state.streamPageNo = syncedStreamPageNo(state.streamPageNo, state.ground);
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
  if (state.previewStopped) return;
  state.previewDirty = true;
  if (state.previewTimer || state.previewInFlight) return;
  state.previewTimer = setTimeout(runPreviewRender, 600);
}

function maybeReschedulePreview() {
  if (state.previewDirty && state.currentJobId && !state.previewStopped &&
      !state.previewTimer && !state.previewInFlight) {
    state.previewTimer = setTimeout(runPreviewRender, state.previewFails >= 4 ? 3000 : 600);
  }
}

// POST 한 번 — 성공 시 {html}, HTTP 실패 시 {status}, 네트워크 오류 시 {status: 0}.
async function postPreviewRender(id, body) {
  try {
    const res = await fetch(`/api/jobs/${id}/render-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body,
    });
    if (res.ok) return { html: await res.text() };
    return { status: res.status };
  } catch (_) {
    return { status: 0 }; // network error → retried on the next schedule
  }
}

// Trusted server-rendered fragment(/html과 동일 렌더러)를 pane 끝에 붙이고
// 붙인 노드 목록을 돌려준다. withSep이면 페이지 경계 hr을 그룹 선두에 포함.
// 타이포셋은 새로 붙인 노드로만 제한 — 기존 확정 노드는 재타이포셋하지 않는다.
function appendPreviewFragment(html, withSep) {
  const nodes = [];
  if (withSep) nodes.push(h('hr'));
  const tpl = document.createElement('template');
  tpl.innerHTML = html;
  nodes.push(...tpl.content.childNodes);
  for (const n of nodes) {
    el.livePreview.appendChild(n);
    if (n.nodeType === 1) typesetMath(n);
  }
  return nodes;
}

// 413(서버 2MB 상한) 또는 연속 실패 시 라이브 프리뷰를 중단하고 원인에 맞는
// 한 줄 안내를 남긴다(일시 장애 중단에 '문서가 커서'라고 표시하지 않도록).
// 잡 완료 후 결과 탭(/html) 렌더는 이와 무관하게 기존 경로로 동작한다.
function stopLivePreview(message) {
  state.previewStopped = true;
  state.previewDirty = false;
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  el.livePreview.appendChild(h('div', {
    class: 'lp-note',
    text: message || '라이브 미리보기를 중단했습니다 — 완료 후 결과 탭에서 확인하세요',
  }));
}

// Throttled, latest-wins (queue of 1): at most one cycle in flight; tokens
// arriving mid-flight mark it dirty and exactly one follow-up is scheduled.
// 증분 렌더: 확정 페이지는 최초 1회만 POST해 HTML을 캐시하고, 이후에는
// 미확정 꼬리만 재전송한다 — 누적 전체 재전송(O(n²)·2MB 413 루프)을 피한다.
async function runPreviewRender() {
  state.previewTimer = 0;
  if (state.previewInFlight || !state.previewDirty || state.previewStopped) return;
  const id = state.currentJobId;
  if (!id) { state.previewDirty = false; return; }
  const gen = state.liveGen;
  state.previewDirty = false;

  const plan = planPreviewRender(
    state.rawText, state.previewPageCache, state.previewTailMd, state.previewTailSep);
  if (!plan.newPages.length && !plan.tailChanged) { maybeReschedulePreview(); return; }

  state.previewInFlight = true;
  let failStatus = -1; // -1 = 실패 없음
  const pageHtmls = [];
  for (const p of plan.newPages) {
    if (!p.md) { pageHtmls.push(''); continue; }
    const r = await postPreviewRender(id, p.md);
    if (state.currentJobId !== id || state.liveGen !== gen) { // 잡 전환 가드
      state.previewInFlight = false;
      maybeReschedulePreview();
      return;
    }
    if (r.html == null) { failStatus = r.status; break; }
    pageHtmls.push(r.html);
  }
  let tailHtml = '';
  if (failStatus < 0 && plan.tailChanged && plan.tailMd) {
    const r = await postPreviewRender(id, plan.tailMd);
    if (state.currentJobId !== id || state.liveGen !== gen) { // 잡 전환 가드
      state.previewInFlight = false;
      maybeReschedulePreview();
      return;
    }
    if (r.html == null) failStatus = r.status;
    else tailHtml = r.html;
  }
  state.previewInFlight = false;

  if (failStatus >= 0) {
    state.previewFails += 1;
    state.previewDirty = true; // 전송하지 못한 조각은 다음 사이클에 재시도
    if (failStatus === 413) {
      stopLivePreview('문서가 커서 라이브 미리보기를 중단했습니다 — 완료 후 결과 탭에서 확인하세요');
    } else if (state.previewFails >= 5) {
      stopLivePreview('라이브 미리보기 렌더가 계속 실패해 중단했습니다 — 완료 후 결과 탭에서 확인하세요');
    } else {
      maybeReschedulePreview();
    }
    return;
  }
  state.previewFails = 0;

  // DOM 증분 적용: 확정 페이지 노드는 유지하고 꼬리 노드만 이동/교체한다.
  const oldTail = state.previewTailNodes;
  for (const n of oldTail) n.remove();
  state.previewTailNodes = [];
  plan.newPages.forEach((p, i) => {
    state.previewPageCache.push(pageHtmls[i]); // p.idx === 캐시 길이 (순서 보장)
    if (pageHtmls[i]) appendPreviewFragment(pageHtmls[i], p.sep);
  });
  if (plan.tailChanged) {
    state.previewTailMd = plan.tailMd;
    state.previewTailSep = plan.tailSep;
    if (tailHtml) state.previewTailNodes = appendPreviewFragment(tailHtml, plan.tailSep);
  } else {
    // 꼬리 내용은 그대로인데 앞에 확정 페이지가 생긴 경우 — 같은 노드를 재부착
    for (const n of oldTail) el.livePreview.appendChild(n);
    state.previewTailNodes = oldTail;
  }
  if (state.previewAutoScroll) el.livePreview.scrollTop = el.livePreview.scrollHeight;
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
  // root 자신이 수식 블록일 수 있다 (증분 프리뷰는 최상위 노드 단위로 붙인다)
  const targets = root.matches && root.matches('.math-inline, .math-display')
    ? [root, ...root.querySelectorAll('.math-inline, .math-display')]
    : root.querySelectorAll('.math-inline, .math-display');
  targets.forEach((elm) => {
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
    syncJobHash(null); // 404로 사라진 잡 — 해시 정리
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
  clearSsePromote(); // 잡 전환·삭제·터미널 상태에서 재승격 재시도도 함께 정리
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  state.previewDirty = false;
  state.fallbackActive = false;
  state.sseErrorCount = 0;
  teardownTranslate(); // 잡 전환·삭제·페이지 이탈 시 번역 구독도 함께 정리
}

// 최초 구독(selectJob 경로)과 폴링 강등 후 재승격 시도가 공유하는 진입점.
// 재승격 중에는 fallbackActive를 건드리지 않는다 — open 성공까지 폴링 유지.
function startStream(id) {
  state.sseErrorCount = 0;

  if (typeof EventSource === 'undefined') {
    appendSystemLine('이 브라우저는 실시간 스트림을 지원하지 않아 상태 폴링을 사용합니다.', 'warn');
    startFallbackPolling(id);
    return;
  }

  let es;
  try {
    es = new EventSource(`/api/jobs/${id}/events`);
  } catch (_) {
    if (!state.fallbackActive) startFallbackPolling(id);
    scheduleSsePromote(id); // 강등은 임시 — 백오프 후 다시 시도
    return;
  }
  state.es = es;

  es.addEventListener('open', () => {
    if (state.currentJobId !== id) return;
    state.sseErrorCount = 0;
    clearSsePromote(); // 재승격 성공 — 백오프 단계 리셋
    if (state.fallbackActive) stopFallbackPolling(); // 폴링 해제, 정상 복귀
    if (!state.streamConnected) {
      state.streamConnected = true;
      appendSystemLine('실시간 스트림에 연결되었습니다.');
    } else {
      // 재연결 — 서버는 백로그를 리플레이하지 않으므로 끊긴 동안 토큰이 유실된다.
      // 갭에 <PAGE> 마커가 있었을 수 있으니 디바이더 번호를 ground 페이지로 보정.
      flushStream(false);
      drainGroundToUI(false);
      state.streamPageNo = syncedStreamPageNo(state.streamPageNo, state.ground);
      appendSystemLine('스트림 재연결됨 — 끊긴 동안의 출력 일부가 누락될 수 있습니다.', 'warn');
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
  if (state.currentJobId !== id) return;
  if (state.fallbackActive) {
    // 폴링 중의 재승격 시도가 실패 — es를 닫고(브라우저 자동 재시도 차단)
    // 다음 백오프 단계로 재시도만 예약한다. 폴링은 그대로 유지된다.
    if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
    scheduleSsePromote(id);
    return;
  }
  state.sseErrorCount += 1;
  if (state.sseErrorCount >= 2) {
    if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
    appendSystemLine('라이브 스트림을 사용할 수 없어 상태 폴링으로 전환했습니다 — 주기적으로 재연결을 시도합니다.', 'warn');
    startFallbackPolling(id);
    scheduleSsePromote(id); // 강등은 임시 — 백오프 후 SSE 재승격 시도
  }
}

// 폴링 강등 후 SSE 재승격 시도를 백오프(10s→20s→30s 상한)로 예약한다.
// 타이머는 teardownConnections / 터미널 폴링 / open 성공에서 정리된다.
function scheduleSsePromote(id) {
  if (state.ssePromoteTimer) clearTimeout(state.ssePromoteTimer);
  const delay = ssePromoteDelay(state.ssePromoteAttempts);
  state.ssePromoteAttempts += 1;
  state.ssePromoteTimer = setTimeout(() => {
    state.ssePromoteTimer = 0;
    // 잡 전환·터미널(폴링 해제)·이미 복귀한 경우에는 시도하지 않는다
    if (state.currentJobId !== id || !state.fallbackActive) return;
    startStream(id);
  }, delay);
}

function clearSsePromote() {
  if (state.ssePromoteTimer) { clearTimeout(state.ssePromoteTimer); state.ssePromoteTimer = 0; }
  state.ssePromoteAttempts = 0;
}

function stopFallbackPolling() {
  if (state.fallbackTimer) { clearInterval(state.fallbackTimer); state.fallbackTimer = 0; }
  state.fallbackActive = false;
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
        clearSsePromote();
        if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; } // 재승격 시도 중이던 es
        removeJobFromList(id);
        if (state.currentJobId === id) { state.currentJobId = null; showEmptyState(); syncJobHash(null); }
      }
      return;
    }
    if (state.currentJobId !== id) return;
    if (isTerminal(job.status)) {
      clearInterval(state.fallbackTimer);
      state.fallbackTimer = 0;
      state.fallbackActive = false;
      clearSsePromote(); // 터미널 — 재승격 재시도도 정리
      if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; } // 재승격 시도 중이던 es
      flushStream(true);
      drainGroundToUI(true);
      renderJob(job);
      refreshJobs();
    } else {
      applyProgress(Object.assign({}, job.progress || {}, {
        status: job.status,
        queue_position: job.queue_position, // 상세 폴링도 대기열 위치를 반영
      }));
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
    // 레이아웃 기능 이전에 변환된 옛 잡(layout.json 없음)은 /layout.html이 404 —
    // 눌러도 조용히 실패하는 버튼 대신 비활성화한다 (has_layout 미제공 구버전 응답은 허용)
    layoutHtml: r.has_layout === false ? null : `/api/jobs/${job.job_id}/layout.html`,
  };
  applyDownloadLangs(); // currentLang='orig' → 원문 URL로 세팅
  if (r.has_layout === false) {
    el.dlLayout.title = '이 작업은 구버전 변환이라 레이아웃 데이터가 없습니다 — PDF를 다시 변환하면 생깁니다';
  } else {
    el.dlLayout.removeAttribute('title');
  }

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

// health의 translate_available을 버튼에 반영 — false일 때만 비활성.
// undefined(미수신·구버전 서버)는 활성 유지(fail-open, 서버 503이 최후 방어).
// 가용성 사유의 비활성만 dataset으로 표시해, 요청 중 일시 비활성(startTranslate)을
// health 갱신이 잘못 풀어버리지 않게 한다.
function applyTranslateAvailability() {
  const btn = el.translateBtn;
  if (state.translateAvailable === false) {
    btn.disabled = true;
    btn.title = '번역 프로바이더가 설정되지 않았습니다 (.env 설정 후 재시작)';
    btn.dataset.unavailable = '1';
  } else {
    btn.removeAttribute('title');
    if (btn.dataset.unavailable) {
      delete btn.dataset.unavailable;
      btn.disabled = false;
    }
  }
}

/* ── 컨트롤 3종 교체 노출 ─────────────────────────────────────────────── */
function showTranslateButton() {
  el.translateBtn.hidden = false;
  el.translateProgress.hidden = true;
  el.langToggle.hidden = true;
  el.translateBtn.disabled = false;
  applyTranslateAvailability(); // 프로바이더 미설정이면 비활성 + 안내 title
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
    const noLayout = state.resultUrls && state.resultUrls.layoutHtml === null;
    el.doclayoutBody.appendChild(h('p', {
      class: 'muted',
      text: noLayout
        ? '이 작업은 레이아웃 기능 이전에 변환되어 레이아웃 데이터가 없습니다 — PDF를 다시 변환하면 생깁니다.'
        : '레이아웃 뷰를 불러오지 못했습니다.',
    }));
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

// 파일 크기 사전 검증 (순수 — frontend/tests/에서 직접 검증). 상한 초과면 안내
// 문구, 통과면 null. limitMb 미수신(undefined 등 비정상)이면 검증을 생략한다
// — 서버 413이 최후 방어.
export function fileSizeError(sizeBytes, limitMb) {
  const limit = Number(limitMb);
  const size = Number(sizeBytes);
  if (!Number.isFinite(limit) || limit <= 0 || !Number.isFinite(size)) return null;
  if (size <= limit * 1024 * 1024) return null;
  // 표시 반올림이 상한과 같아지는 경계(상한+1바이트 → '100 MB — 상한 100MB')의
  // 자기모순 문구를 피한다 — 반올림 표시가 상한을 명확히 넘을 때만 크기를 병기.
  const sizeMb = size / (1024 * 1024);
  return sizeMb - limit >= 0.05
    ? `파일이 너무 큽니다 (${fmtBytes(size)} — 서버 상한 ${limit}MB)`
    : `파일이 너무 큽니다 (서버 상한 ${limit}MB 초과)`;
}

function validateFile(file) {
  const name = file && file.name ? file.name : '';
  if (!/\.pdf$/i.test(name)) return 'PDF 파일만 업로드할 수 있습니다. (.pdf)';
  const type = file.type || '';
  if (type && !/pdf/i.test(type)) return '올바른 PDF 파일이 아닌 것 같습니다. 파일을 확인해 주세요.';
  return null;
}

// 다중 선택 검증 분류 (순수 — frontend/tests/에서 직접 검증). validate(file)는
// 오류 문구 또는 null을 반환. 유효 파일과 '건너뜀' 대상(파일명+사유)으로 나눈다.
export function classifyFiles(files, validate) {
  const valid = [];
  const skipped = [];
  for (const f of Array.from(files || [])) {
    const reason = validate(f);
    if (reason) skipped.push({ file: f, name: (f && f.name) || '(이름 없음)', reason });
    else valid.push(f);
  }
  return { valid, skipped };
}

// file-info 표시 문구 (순수 — 테스트 대상). 1개면 기존 단일 표시(이름·크기),
// 여러 개면 'N개 파일 · 총 X' + title에 파일명 나열. 빈 선택은 null.
export function selectionSummary(files) {
  const list = Array.from(files || []);
  if (!list.length) return null;
  if (list.length === 1) {
    return { name: list[0].name, size: fmtBytes(Number(list[0].size) || 0), title: list[0].name };
  }
  const total = list.reduce((sum, f) => sum + (Number(f.size) || 0), 0);
  return {
    name: `${list.length}개 파일`,
    size: `총 ${fmtBytes(total)}`,
    title: list.map((f) => f.name).join(', '),
  };
}

// '첫 건 + 나머지 개수' 요약 (순수 — 테스트 대상). entries: [{name, reason}].
// prefix 예: '건너뜀' | '업로드 실패'. 빈 배열이면 null.
export function summarizeIssues(prefix, entries) {
  if (!entries || !entries.length) return null;
  const first = `${prefix}: ${entries[0].name} — ${entries[0].reason}`;
  return entries.length === 1 ? first : `${first} 외 ${entries.length - 1}건`;
}

// 형식 검증에 이어 서버 상한 크기 사전 검증 — 선택·업로드 직전 공통.
function fileValidationError(file) {
  return validateFile(file) || fileSizeError(file && file.size, state.maxUploadMb);
}

// 현재 선택(state.selectedFiles)을 file-info와 업로드 버튼에 반영.
function renderFileInfo() {
  const s = selectionSummary(state.selectedFiles);
  if (!s) {
    el.fileInfo.hidden = true;
    el.uploadBtn.disabled = true;
    return;
  }
  el.fileName.textContent = s.name;
  el.fileName.title = s.title;
  el.fileSize.textContent = s.size;
  el.fileInfo.hidden = false;
  // 업로드 진행 중의 새 선택은 버튼을 되살리지 않는다 — 루프 이중 진입 방지.
  // 진행 중 선택은 setUploading(false)가 끝나며 재활성화된다.
  el.uploadBtn.disabled = state.uploading;
}

// 픽커·드래그드롭 공통 진입점 — 파일별 검증으로 유효분만 선택에 담고, 무효분은
// '건너뜀' 요약을 기존 업로드 에러 영역에 안내한다(전부 무효면 선택 없음).
function setSelectedFiles(files) {
  const { valid, skipped } = classifyFiles(files, fileValidationError);
  const skipMsg = summarizeIssues('건너뜀', skipped);
  if (skipMsg) showUploadError(skipMsg);
  else hideUploadError();
  state.selectedFiles = valid;
  renderFileInfo();
}

function clearSelectedFiles() {
  state.selectedFiles = [];
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
  state.uploading = on;
  el.uploadBtn.disabled = on || !state.selectedFiles.length;
  el.uploadBtn.textContent = on ? '업로드 중…' : '변환 시작';
  el.dropzone.classList.toggle('disabled', on);
}

// 업로드 진행바 (progress-track/fill 재사용). frac=null이면 총량을 알 수 없는
// 전송(lengthComputable=false) — indeterminate 애니메이션으로 표시.
function showUploadProgress(frac) {
  el.uploadProgress.hidden = false;
  const indet = frac == null;
  el.uploadProgress.classList.toggle('indeterminate', indet);
  el.uploadProgressFill.style.width = indet ? '' : `${Math.min(100, Math.max(0, frac * 100))}%`;
}

function hideUploadProgress() {
  el.uploadProgress.hidden = true;
  el.uploadProgress.classList.remove('indeterminate');
  el.uploadProgressFill.style.width = '';
}

// XHR 업로드 — fetch에는 업로드 진행 이벤트가 없어 진행률 표시용으로만 XHR을
// 쓴다. 응답은 fetch 경로와 같은 의미의 {status, text}로 통일하고, 전송 실패
// (네트워크 오류)만 reject한다. HTTP 오류 상태는 resolve — 호출부가 분기한다.
function uploadWithProgress(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    if (xhr.upload) {
      xhr.upload.addEventListener('progress', (e) => {
        onProgress(e.lengthComputable && e.total > 0 ? e.loaded / e.total : null);
      });
    }
    xhr.addEventListener('load', () => resolve({ status: xhr.status, text: xhr.responseText || '' }));
    xhr.addEventListener('error', () => reject(new Error('network error')));
    xhr.addEventListener('abort', () => reject(new Error('aborted')));
    xhr.send(form);
  });
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

// HTTP 실패 상태 → 실패 사유 문구. 서버 detail(실시간 상한 포함)을 우선하고,
// 413은 health로 받은 상한, 그마저 없으면 중립 문구.
function uploadFailureMessage(status, data) {
  const detail = data && typeof data.detail === 'string' ? data.detail : null;
  if (detail) return detail;
  if (status === 413) {
    return state.maxUploadMb
      ? `파일이 너무 큽니다. 더 작은 PDF를 업로드해 주세요. (최대 ${state.maxUploadMb}MB)`
      : '파일이 너무 커서 서버 업로드 상한을 초과했습니다. 더 작은 PDF를 업로드해 주세요.';
  }
  if (status === 400) return '유효하지 않은 PDF 파일입니다.';
  return `업로드에 실패했습니다. (${status})`;
}

// 순차 다중 업로드 — 파일별로 기존 XHR 경로(uploadWithProgress)를 재사용하고
// 진행바는 파일 단위로 리셋한다. 첫 성공 잡은 즉시 openJob(대기하지 않음 —
// 업로드 도중 잡 전환이 일어나도 루프는 계속), 이후 성공은 목록 upsert만.
// 개별 실패는 수집해 끝에 요약하고, 실패분만 선택에 남겨 재시도할 수 있게 한다.
async function handleUpload() {
  if (state.uploading || !state.selectedFiles.length) return; // 재진입 가드
  hideUploadError();

  // 선택 시점에는 health 미수신이었어도 이후 수신됐으면 여기서 한 번 더 차단
  const { valid, skipped } = classifyFiles(state.selectedFiles, fileValidationError);
  if (!valid.length) {
    showUploadError(summarizeIssues('건너뜀', skipped) || '업로드할 수 있는 파일이 없습니다.');
    return;
  }

  const selectionAtStart = state.selectedFiles;
  const mode = readMode();
  const dpi = readDpi();
  const failures = skipped.slice(); // {file, name, reason} — 뒤늦게 걸러진 파일도 요약에 포함
  let successCount = 0;
  let firstJobId = null;

  setUploading(true);
  for (let i = 0; i < valid.length; i += 1) {
    const file = valid[i];
    if (valid.length > 1) el.uploadBtn.textContent = `업로드 중… (${i + 1}/${valid.length})`;
    showUploadProgress(0); // 파일 단위 리셋

    const form = new FormData();
    form.append('file', file);
    form.append('mode', mode);
    form.append('dpi', String(dpi));

    let res;
    try {
      res = await uploadWithProgress('/api/jobs', form, showUploadProgress);
    } catch (_) {
      failures.push({ file, name: file.name, reason: '네트워크 오류가 발생했습니다. 다시 시도해 주세요.' });
      continue;
    }
    const data = res.text ? safeParse(res.text) : null;
    if (!(res.status >= 200 && res.status < 300)) {
      failures.push({ file, name: file.name, reason: uploadFailureMessage(res.status, data) });
      continue;
    }
    const jobId = data && data.job_id;
    if (!jobId) {
      failures.push({ file, name: file.name, reason: '서버 응답이 올바르지 않습니다.' });
      continue;
    }

    successCount += 1;
    // Optimistically add to history; the first success opens + streams.
    upsertJob({
      job_id: jobId,
      filename: file.name,
      status: data.status || 'queued',
      mode,
      created_at: new Date().toISOString(),
      progress: {},
      result: null,
      error: null,
    });
    if (firstJobId === null) {
      firstJobId = jobId;
      openJob(jobId); // 나머지 업로드와 병행 — 루프는 currentJobId에 의존하지 않는다
    }
  }
  hideUploadProgress();

  // 업로드 도중 사용자가 선택을 바꿨다면(드롭 등) 그 새 선택은 건드리지 않는다.
  if (state.selectedFiles === selectionAtStart) {
    state.selectedFiles = failures.map((f) => f.file); // 실패분만 남겨 재시도 가능
    if (!failures.length) el.fileInput.value = '';
    renderFileInfo();
  }
  setUploading(false);

  if (failures.length) {
    showUploadError(summarizeIssues('업로드 실패', failures));
    showToast(`업로드 요약 — 성공 ${successCount} · 실패 ${failures.length}`,
      successCount ? 'warn' : 'error');
  } else if (valid.length > 1) {
    showToast(`${successCount}개 파일이 업로드되었습니다.`);
  }
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
    if (files && files.length) setSelectedFiles(files);
  });
  el.fileInput.addEventListener('change', () => {
    if (el.fileInput.files && el.fileInput.files.length) setSelectedFiles(el.fileInput.files);
  });
  el.fileClear.innerHTML = ICON.x;
  el.fileClear.addEventListener('click', clearSelectedFiles);
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
    // 키에 잡 id 포함 — 잡 전환 뒤 남은 무장이 다른 잡을 삭제하지 못하게 한다.
    armDelete(el.jobDelete, `header:${state.currentJobId}`, () => deleteJob(state.currentJobId));
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

  // 사용자가 주소창을 직접 고치거나 뒤로가기로 해시가 바뀐 경우 해당 잡을 연다.
  // 우리가 만드는 변경은 replaceState라 hashchange가 발생하지 않는다(루프 없음).
  window.addEventListener('hashchange', () => {
    const id = jobIdFromHash(location.hash);
    if (id && id !== state.currentJobId) openJob(id);
  });

  showEmptyState();
  loadHealth();
  refreshJobs().then(() => {
    // 첫 잡 목록 수신 직후 해시의 잡 복원 — 새로고침·공유 링크 진입.
    // 404면 openJob의 기존 처리(토스트)가 동작하고 해시를 비운다.
    const id = jobIdFromHash(location.hash);
    if (id) openJob(id);
  });
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
