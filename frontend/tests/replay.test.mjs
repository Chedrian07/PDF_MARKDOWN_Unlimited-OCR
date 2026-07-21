// Unlimited-OCR frontend — live-view replay test suite.
//
// 테스트 실행 (의존성 0, Node 내장 러너만 사용):
//   node --test frontend/tests/          (저장소 루트에서)
//   cd frontend && npm test              (또는: node --test tests/)
//   node --test frontend/tests/replay.test.mjs
//
// fixtures/*.sse.txt 는 실제 서버 SSE 캡처다. 이 스위트는 캡처를 이벤트로 파싱해
// app.js가 실제로 export하는 순수 코어(마커/페이지 상태머신 groundAnnounce+groundDrain,
// 그라운딩 파서, 프리뷰 구조화기 structurePreview)에 그대로 재생한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  PAGE_MARKER,
  createGroundState,
  groundAnnounce,
  groundPush,
  groundDrain,
  structurePreview,
  splitPreviewPages,
  planPreviewRender,
  incompleteTailIndex,
  scanQuads,
  ssePromoteDelay,
  syncedStreamPageNo,
  withLangUrl,
  translateUiStateFor,
  armTransition,
  fileSizeError,
  jobIdFromHash,
  statusLabel,
  classifyFiles,
  selectionSummary,
  summarizeIssues,
  healthCapabilities,
  providerIssue,
  jobModelChip,
  docLayoutNoteFor,
} from '../app.js';

const FIXTURES = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

function loadFixture(name) {
  return fs.readFileSync(path.join(FIXTURES, name), 'utf8');
}

/* ---------------- SSE fixture parsing ---------------- */

// Parse a raw SSE capture into ordered {event, data} records.
// Records are separated by blank lines; comment (:) and retry lines ignored.
function parseSse(text) {
  const events = [];
  for (const record of text.split(/\r?\n\r?\n/)) {
    let event = null;
    let data = '';
    for (const line of record.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).trim();
    }
    if (event && data) events.push({ event, data: JSON.parse(data) });
  }
  return events;
}

/* ---------------- Replay driver (mirrors app.js wiring) ---------------- */

// Feeds parsed SSE events through the real state machine exactly like app.js:
// tokens → groundPush; progress → drain buffered tokens FIRST, then announce.
// opts.drainEveryToken simulates the tightest rAF batching (drain per token);
// default drains only at progress events + once at the end (coarsest batching).
// opts.rechunk re-splits the concatenated token text into fixed-size chunks
// between progress events, to prove chunk-boundary independence.
function replay(events, opts = {}) {
  const g = createGroundState();
  const boxes = [];
  const pageSeq = [g.page];
  let raw = '';

  const record = () => {
    if (pageSeq[pageSeq.length - 1] !== g.page) pageSeq.push(g.page);
  };
  const drain = (final) => {
    for (const ev of groundDrain(g, final)) {
      if (ev.type === 'boxes') boxes.push(ev);
    }
    record();
  };
  const pushText = (text) => {
    raw += text;
    if (opts.rechunk) {
      for (let i = 0; i < text.length; i += opts.rechunk) {
        groundPush(g, text.slice(i, i + opts.rechunk));
        drain(false);
      }
    } else {
      groundPush(g, text);
      if (opts.drainEveryToken) drain(false);
    }
  };

  for (const e of events) {
    if (e.event === 'progress') {
      drain(false); // app.js: drain buffered markers/boxes before announcing
      groundAnnounce(g, e.data.phase, e.data.current_page, e.data.total_pages);
      record();
    } else if (e.event === 'token') {
      pushText(String(e.data.text || ''));
    }
  }
  drain(true);
  return { g, boxes, pageSeq, raw };
}

function boxesOnPage(boxes, page) {
  return boxes.filter((b) => b.page === page);
}

function findBox(boxes, page, label, quad) {
  return boxesOnPage(boxes, page).some((ev) =>
    ev.label === label &&
    ev.boxes.some((b) => b.x1 === quad[0] && b.y1 === quad[1] && b.x2 === quad[2] && b.y2 === quad[3]));
}

/* ================= real fixture: 2 pages / 1 chunk ================= */

const realEvents = parseSse(loadFixture('real-2page-1chunk.sse.txt'));

test('real fixture: parses into ordered events', () => {
  assert.ok(realEvents.length > 30, `got ${realEvents.length} events`);
  assert.equal(realEvents[0].event, 'progress');
  assert.equal(realEvents[0].data.phase, 'render');
  const tokens = realEvents.filter((e) => e.event === 'token');
  assert.ok(tokens[0].data.text.startsWith(PAGE_MARKER), 'first token starts with <PAGE>');
});

for (const [name, opts] of [
  ['drain at progress events (coarse batching)', {}],
  ['drain after every token (tight batching)', { drainEveryToken: true }],
  ['re-chunked into 3-char chunks', { rechunk: 3 }],
]) {
  test(`real fixture — ${name}: boxPage sequence is 1 then 2`, () => {
    const { g, boxes, pageSeq } = replay(realEvents, opts);
    assert.deepEqual(pageSeq, [1, 2], `pageSeq=${JSON.stringify(pageSeq)}`);
    assert.equal(pageSeq[0], 1, 'never starts at 2');
    assert.ok(!pageSeq.includes(3), 'never reaches 3');
    assert.equal(g.page, 2);
    assert.equal(g.markerCount, 2, 'a 2-page doc has exactly 2 markers');
    // page-1 boxes include the title det (leading marker must NOT shift it to page 2)
    assert.ok(findBox(boxes, 1, 'title', [118, 76, 705, 106]),
      `page-1 title missing: ${JSON.stringify(boxesOnPage(boxes, 1))}`);
    assert.ok(findBox(boxes, 1, 'table', [109, 520, 860, 596]), 'page-1 table det on page 1');
    assert.ok(findBox(boxes, 2, 'title', [117, 80, 497, 106]), 'page-2 title on page 2');
    assert.equal(boxesOnPage(boxes, 1).length, 6, 'page 1 has 6 det blocks');
    assert.equal(boxesOnPage(boxes, 2).length, 7, 'page 2 has 7 det blocks');
    assert.ok(boxes.every((b) => b.page === 1 || b.page === 2), 'no boxes beyond page 2');
  });
}

test('real fixture: render-phase progress storm cannot pin the box page', () => {
  const storm = [];
  for (let i = 1; i <= 25; i += 1) {
    storm.push({ event: 'progress', data: { phase: 'render', current_page: i, total_pages: 25, status: 'running' } });
  }
  // sanity mid-check: after the storm alone, page must still be 1
  const stormOnly = replay(storm);
  assert.equal(stormOnly.g.page, 1, 'boxPage stays 1 through the render storm');
  assert.equal(stormOnly.g.ocrSeen, false, 'render never marks OCR as seen');

  const { g, boxes, pageSeq } = replay([...storm, ...realEvents]);
  assert.deepEqual(pageSeq, [1, 2]);
  assert.equal(g.page, 2);
  assert.ok(findBox(boxes, 1, 'title', [118, 76, 705, 106]), 'attribution unchanged by the storm');
});

test('real fixture: merge-phase progress does not move the box page', () => {
  // the capture ends with progress(merge, current_page=2); replay a stronger case
  const g = createGroundState();
  groundAnnounce(g, 'ocr', 1, 25);
  groundPush(g, '<PAGE>content');
  groundDrain(g, false);
  groundAnnounce(g, 'merge', 25, 25);
  assert.equal(g.page, 1, 'merge announce must not advance');
});

/* ================= fake fixture: 3 pages / 2 chunks ================= */

const fakeEvents = parseSse(loadFixture('fake-3page-2chunk.sse.txt'));

for (const [name, opts] of [
  ['drain at progress events', {}],
  ['drain after every token', { drainEveryToken: true }],
  ['re-chunked into 5-char chunks', { rechunk: 5 }],
]) {
  test(`fake fixture — ${name}: chunk-2 leading marker does not double-bump`, () => {
    const { g, pageSeq } = replay(fakeEvents, opts);
    assert.deepEqual(pageSeq, [1, 2, 3], `pageSeq=${JSON.stringify(pageSeq)}`);
    assert.equal(g.page, 3, 'ends on page 3, never 4');
    assert.equal(g.markerCount, 3, '3 markers for 3 pages across 2 chunks');
    assert.equal(g.totalPages, 3);
  });
}

/* ================= state machine unit cases ================= */

test('markers without fresh announcements advance by +1 (announced page confirmed once)', () => {
  const g = createGroundState();
  groundAnnounce(g, 'ocr', 1, 3);
  const events = [];
  // chunk streams pages 1..3 with only the chunk-start announcement
  groundPush(g, '<PAGE>p1 <|det|>text [1,1,9,9]<|/det|>a <PAGE>p2 <|det|>text [2,2,9,9]<|/det|>b <PAGE>p3 <|det|>text [3,3,9,9]<|/det|>c');
  events.push(...groundDrain(g, true));
  const boxPages = events.filter((e) => e.type === 'boxes').map((e) => e.page);
  assert.deepEqual(boxPages, [1, 2, 3]);
  assert.equal(g.page, 3);
  assert.equal(g.markerCount, 3);
});

test('per_page mode: announcements alone drive the page (no markers at all)', () => {
  const g = createGroundState();
  const pages = [];
  for (let i = 1; i <= 5; i += 1) {
    groundAnnounce(g, 'ocr', i, 12);
    groundPush(g, `<|det|>text [1,1,9,9]<|/det|> page ${i} `);
    for (const ev of groundDrain(g, false)) if (ev.type === 'boxes') pages.push(ev.page);
  }
  assert.deepEqual(pages, [1, 2, 3, 4, 5]);
  assert.equal(g.page, 5);
});

test('joining mid-OCR: snapshot announcement seeds the page; later pages self-correct', () => {
  const g = createGroundState();
  groundAnnounce(g, 'ocr', 7, 25); // job opened at OCR page 7
  assert.equal(g.page, 7);
  assert.equal(g.ocrSeen, true);
  groundPush(g, 'tail of page 7 <|det|>text [1,1,9,9]<|/det|>x');
  let boxes = groundDrain(g, false).filter((e) => e.type === 'boxes');
  assert.equal(boxes[0].page, 7);
  groundAnnounce(g, 'ocr', 8, 25);          // page 8 announced first (grammar)
  groundPush(g, '<PAGE><|det|>title [2,2,9,9]<|/det|>y'); // then its marker
  boxes = groundDrain(g, false).filter((e) => e.type === 'boxes');
  assert.equal(g.page, 8, 'marker after announcement is consumed, page stays 8');
  assert.equal(boxes[0].page, 8);
});

test('first-ocr announcement resets stale pre-OCR advancement', () => {
  const g = createGroundState();
  g.page = 25; // stale leftovers (e.g. rerun)
  g.expectAnnounce = false;
  const r = groundAnnounce(g, 'ocr', 1, 25);
  assert.equal(r.firstOcr, true);
  assert.equal(g.page, 1);
});

test('marker count is capped by total_pages (spurious trailing markers)', () => {
  const g = createGroundState();
  groundAnnounce(g, 'ocr', 1, 2);
  groundPush(g, '<PAGE>a<PAGE>b<PAGE>c<PAGE>d');
  groundDrain(g, true);
  assert.equal(g.page, 2, 'never exceeds total_pages');
  assert.equal(g.markerCount, 4, 'markers still counted');
});

/* ================= grounding parser: chunk-boundary splits ================= */

function feedChunks(text, chunkSize, { announced = true } = {}) {
  const g = createGroundState();
  if (announced) groundAnnounce(g, 'ocr', 1, 0); // arm like a real chunk start
  const boxes = [];
  let pageEvents = 0;
  for (let i = 0; i < text.length; i += chunkSize) {
    groundPush(g, text.slice(i, i + chunkSize));
    for (const ev of groundDrain(g, false)) {
      if (ev.type === 'boxes') boxes.push(ev);
      else pageEvents += 1;
    }
  }
  for (const ev of groundDrain(g, true)) {
    if (ev.type === 'boxes') boxes.push(ev);
    else pageEvents += 1;
  }
  return { g, boxes, pageEvents };
}

test('split ref block (3-char chunks) parses exactly once with 2 quads', () => {
  const { boxes } = feedChunks('intro <|ref|>image<|/ref|><|det|>[[100,200,300,400],[500,600,700,800]]<|/det|> outro', 3);
  assert.equal(boxes.length, 1);
  assert.equal(boxes[0].label, 'image');
  assert.equal(boxes[0].boxes.length, 2);
  assert.deepEqual(boxes[0].boxes[0], { x1: 100, y1: 200, x2: 300, y2: 400 });
});

test('split inline det (2-char chunks) parses exactly once', () => {
  const { boxes } = feedChunks('x<|det|>title [10, 20, 30, 40]<|/det|>y', 2);
  assert.equal(boxes.length, 1);
  assert.equal(boxes[0].label, 'title');
  assert.deepEqual(boxes[0].boxes[0], { x1: 10, y1: 20, x2: 30, y2: 40 });
});

test('markers split across 4-char chunks attribute boxes to the right pages', () => {
  const stream = '<PAGE><|det|>text [1,1,5,5]<|/det|><PAGE><|det|>table [2,2,6,6]<|/det|>';
  const { g, boxes } = feedChunks(stream, 4);
  // leading marker consumed (announced), second marker advances to page 2
  assert.deepEqual(boxes.map((b) => b.page), [1, 2]);
  assert.equal(g.page, 2);
});

test('degenerate boxes are dropped; single-bracket ref payload accepted', () => {
  const { boxes } = feedChunks('a<|ref|>table<|/ref|><|det|>[5,6,900,900]<|/det|>b<|det|>text [7,7,7,7]<|/det|>c', 1000);
  assert.equal(boxes.length, 1, 'the zero-area det is dropped');
  assert.deepEqual(boxes[0].boxes[0], { x1: 5, y1: 6, x2: 900, y2: 900 });
});

test('idempotent across repeated drains of the same buffer', () => {
  const g = createGroundState();
  groundAnnounce(g, 'ocr', 1, 0);
  groundPush(g, 'x<|det|>text [1,2,3,4]<|/det|>y');
  const all = [];
  for (let i = 0; i < 3; i += 1) all.push(...groundDrain(g, false));
  assert.equal(all.filter((e) => e.type === 'boxes').length, 1);
});

/* ================= incompleteTailIndex boundaries ================= */

test('incompleteTailIndex holdback boundaries', () => {
  assert.equal(incompleteTailIndex('plain text', 1200), 10, 'plain text fully consumable');
  assert.equal(incompleteTailIndex('text <PA', 1200), 5, 'partial <PAGE prefix held');
  assert.equal(incompleteTailIndex('text <|de', 1200), 5, 'partial <|det prefix held');
  const openRef = 'a<|ref|>image<|/ref|><|det|>[[1,2';
  assert.equal(incompleteTailIndex(openRef, 1200), 1, 'open ref block held from <|ref|>');
  const closedRef = 'a<|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|> tail';
  assert.equal(incompleteTailIndex(closedRef, 1200), closedRef.length, 'complete block not held');
  assert.equal(incompleteTailIndex('x<|foo|>y', 1200), 9, 'complete stray special not held');
  assert.equal(incompleteTailIndex('x<|untermina', 1200), 1, 'unterminated <| held');
  const stale = '<|ref|>' + 'x'.repeat(1500);
  assert.equal(incompleteTailIndex(stale, 1200), stale.length, 'cap gives up on stale opener');
  assert.equal(incompleteTailIndex('abc<|det|>text [1,2', 1200), 3, 'open det held');
});

test('scanQuads: number-quad scanner, not JSON', () => {
  assert.deepEqual(scanQuads('[[1, 2,3 ,4],[5,6,7,8]]'), [[1, 2, 3, 4], [5, 6, 7, 8]]);
  assert.deepEqual(scanQuads('[9,9,9]'), [], 'incomplete quad ignored');
  assert.deepEqual(scanQuads(''), []);
});

/* ================= preview structurer ================= */

test('structurer on the real fixture: heading, intact table html, no det remnants', () => {
  const { raw } = replay(realEvents);
  const md = structurePreview(raw, true);

  assert.ok(md.includes('## Unlimited-OCR End-to-End Sample'), 'title det → ## heading');
  assert.ok(md.includes('## Section 2: Figures and Lists'), 'page-2 title → ## heading');
  assert.ok(md.includes('<table><tr><td>Mode</td>'), 'table html start intact');
  assert.ok(md.includes('</table>'), 'table html end intact');
  assert.ok(!md.includes('<|det|>') && !md.includes('<|/det|>'), 'no det remnants');
  assert.ok(!md.includes(PAGE_MARKER), 'no raw <PAGE> markers');
  assert.ok(md.includes('\\( E = mc^{2} \\)'), 'LaTeX stays literal inside its paragraph');

  const parts = md.split('\n\n');
  assert.ok(parts.length >= 8, `blocks separated by blank lines (got ${parts.length})`);
  assert.equal(parts.filter((p) => p === '---').length, 1, 'one page separator (leading marker suppressed)');
  assert.equal(parts[0].startsWith('##'), true, 'no leading ---');
  assert.equal(parts.filter((p) => p.includes('그림 감지됨')).length, 2, 'two image placeholders');
});

test('structurer: streaming holdback keeps incomplete tails out', () => {
  const md = structurePreview('<PAGE><|det|>title [1,1,9,9]<|/det|>Hello\n<|det|>text [1,1', false);
  assert.equal(md, '## Hello', 'incomplete det held back, no leak');
});

test('structurer: block semantics', () => {
  const raw = '<PAGE>' +
    '<|det|>title [1,1,9,9]<|/det|>Multi\nline title\n' +
    '<|det|>page_number [1,1,9,9]<|/det|>3\n' +
    '<|det|>header [1,1,9,9]<|/det|>Running header\n' +
    '<|det|>text [1,1,9,9]<|/det|>Body paragraph.\n' +
    '<|det|>image [1,1,9,9]<|/det|>\n' +
    '<|ref|>image<|/ref|><|det|>[[1,1,9,9]]<|/det|>\n' +
    '<|ref|>table<|/ref|><|det|>[[1,1,9,9]]<|/det|>\n' +
    '<|det|>equation [1,1,9,9]<|/det|>\\( a^2 + b^2 = c^2 \\)\n' +
    '<PAGE>' +
    '<|det|>text [1,1,9,9]<|/det|>Second page.\n';
  const md = structurePreview(raw, true);
  const parts = md.split('\n\n');

  assert.equal(parts[0], '## Multi line title', 'title newlines collapsed');
  assert.ok(!md.includes('Running header') && !md.includes('\n3\n'), 'page furniture dropped');
  assert.ok(md.includes('Body paragraph.'));
  assert.equal(parts.filter((p) => p.includes('그림 감지됨')).length, 2, 'image det + image ref → placeholders');
  assert.ok(md.includes('\\( a^2 + b^2 = c^2 \\)'), 'equation literal');
  assert.equal(parts.filter((p) => p === '---').length, 1, 'one separator between the two pages');
  assert.ok(md.endsWith('Second page.'), 'no trailing separator on final render');
  assert.ok(!md.includes('<|'), 'all specials stripped');
});

test('structurer: fake-engine plain markdown passes through with page separators', () => {
  const { raw } = replay(fakeEvents);
  const md = structurePreview(raw, true);
  assert.ok(md.includes('## 페이지 1 — page_0001'), 'plain markdown preserved');
  assert.ok(md.includes('| 항목 | 값 |'), 'markdown table preserved');
  assert.equal(md.split('\n\n').filter((p) => p === '---').length, 2, '2 separators for 3 pages');
  assert.ok(!md.includes(PAGE_MARKER));
});

/* ================= live preview incremental split/plan ================= */

test('splitPreviewPages: 뒤에 새 페이지가 시작된 세그먼트만 확정, 나머지는 꼬리', () => {
  assert.deepEqual(splitPreviewPages(''), { pages: [], tail: '' });
  assert.deepEqual(splitPreviewPages('no marker yet'), { pages: [], tail: 'no marker yet' });
  assert.deepEqual(splitPreviewPages('<PAGE>p1<PAGE>p2<PAGE>p3'),
    { pages: ['', 'p1', 'p2'], tail: 'p3' });
});

test('splitPreviewPages: 조각난 마커(<PA + GE>)는 완성 전까지 경계가 아니다', () => {
  const a = '<PAGE>one<PA';
  assert.deepEqual(splitPreviewPages(a), { pages: [''], tail: 'one<PA' });
  const b = a + 'GE>two'; // 다음 청크로 마커 완성 → one이 확정된다
  assert.deepEqual(splitPreviewPages(b), { pages: ['', 'one'], tail: 'two' });
});

test('splitPreviewPages: 확정 프리픽스는 append에 안정 — 캐시 재사용 가능', () => {
  const raw1 = '<PAGE>alpha<PAGE>beta';
  const raw2 = raw1 + ' more tail';
  assert.deepEqual(splitPreviewPages(raw2).pages, splitPreviewPages(raw1).pages);
  assert.equal(splitPreviewPages(raw2).tail, 'beta more tail');
});

const det = (label, text) => `<|det|>${label} [1,1,9,9]<|/det|>${text}\n`;

test('planPreviewRender: 첫 사이클 — 확정 페이지 + 꼬리, hr(sep)은 앞 내용이 있을 때만', () => {
  const raw = '<PAGE>' + det('text', 'p1 body') + '<PAGE>' + det('text', 'p2 tail');
  const plan = planPreviewRender(raw, [], '', false);
  // 확정 세그먼트: 첫 마커 앞 ''(빈) + p1
  assert.equal(plan.newPages.length, 2);
  assert.deepEqual(plan.newPages.map((p) => p.idx), [0, 1]);
  assert.equal(plan.newPages[0].md, '');
  assert.equal(plan.newPages[1].md, 'p1 body');
  assert.equal(plan.newPages[1].sep, false, '첫 내용 페이지 앞에는 hr 없음');
  assert.equal(plan.tailMd, 'p2 tail');
  assert.equal(plan.tailSep, true, '확정 내용 뒤 꼬리 → hr');
  assert.equal(plan.tailChanged, true);
});

test('planPreviewRender: 꼬리만 변하는 경우 — 확정 페이지 캐시 재사용, 재렌더 없음', () => {
  const raw1 = '<PAGE>' + det('text', 'p1 body') + '<PAGE>' + det('text', 'p2');
  const cache = ['', '<p>p1 body</p>']; // 사이클 1이 채운 확정 페이지 HTML 캐시
  const plan1 = planPreviewRender(raw1, cache, '', false);
  assert.equal(plan1.newPages.length, 0, '캐시된 확정 페이지는 다시 렌더하지 않는다');
  assert.equal(plan1.tailMd, 'p2');
  // 꼬리에 토큰이 더 붙으면 꼬리만 재렌더 대상
  const raw2 = raw1 + det('text', 'p2 more');
  const plan2 = planPreviewRender(raw2, cache, plan1.tailMd, plan1.tailSep);
  assert.equal(plan2.newPages.length, 0);
  assert.equal(plan2.tailChanged, true);
  assert.ok(plan2.tailMd.includes('p2 more'));
  // 아무것도 변하지 않으면 POST 자체를 생략한다
  const plan3 = planPreviewRender(raw2, cache, plan2.tailMd, plan2.tailSep);
  assert.equal(plan3.newPages.length, 0);
  assert.equal(plan3.tailChanged, false);
});

test('planPreviewRender: 같은 꼬리 md라도 sep이 바뀌면 재렌더 대상', () => {
  const plan1 = planPreviewRender('<PAGE>x', [], '', false);
  assert.equal(plan1.tailMd, 'x');
  assert.equal(plan1.tailSep, false);
  // 내용 있는 페이지가 확정되면 같은 'x' 꼬리라도 앞에 hr이 필요하다
  const plan2 = planPreviewRender('<PAGE>x<PAGE>x', ['', '<p>x</p>'], plan1.tailMd, plan1.tailSep);
  assert.equal(plan2.tailMd, plan1.tailMd);
  assert.equal(plan2.tailSep, true);
  assert.equal(plan2.tailChanged, true);
});

test('planPreviewRender: 증분 조각을 이어 붙이면 전체 structurePreview와 동치', () => {
  const { raw } = replay(realEvents);
  const plan = planPreviewRender(raw, [], '', false);
  const parts = [];
  for (const p of plan.newPages) {
    if (!p.md) continue;
    if (p.sep) parts.push('---');
    parts.push(p.md);
  }
  if (plan.tailMd) {
    if (plan.tailSep) parts.push('---');
    parts.push(plan.tailMd);
  }
  assert.equal(parts.join('\n\n'), structurePreview(raw, false));
});

/* ================= SSE 폴링 강등 → 재승격 백오프 ================= */

test('ssePromoteDelay: 10초 → 20초 → 30초 상한 백오프', () => {
  assert.equal(ssePromoteDelay(0), 10000);
  assert.equal(ssePromoteDelay(1), 20000);
  assert.equal(ssePromoteDelay(2), 30000);
  assert.equal(ssePromoteDelay(9), 30000, '상한 30초를 넘지 않는다');
  assert.equal(ssePromoteDelay(undefined), 10000, '방어: 미지 입력은 첫 단계');
});

test('syncedStreamPageNo: 정상 흐름에서는 항상 no-op (디바이더 = 마커 카운트)', () => {
  const g = createGroundState();
  let pane = 0; // flushStream이 마커마다 올리는 streamPageNo 시뮬레이션
  const feed = (text) => {
    pane += (text.match(/<PAGE>/g) || []).length;
    groundPush(g, text);
    groundDrain(g, false);
  };
  assert.equal(syncedStreamPageNo(pane, g), 0, '잡 시작: 다음 마커가 페이지 1 시작 → 0 유지');
  groundAnnounce(g, 'ocr', 1, 3);
  feed('<PAGE>p1 ');
  assert.equal(syncedStreamPageNo(pane, g), pane);
  groundAnnounce(g, 'ocr', 2, 3); // 선언 → 마커 (문법 순서)
  assert.equal(syncedStreamPageNo(pane, g), pane, '선언 직후(마커 전)에도 no-op');
  feed('<PAGE>p2 ');
  assert.equal(syncedStreamPageNo(pane, g), pane);
  feed('<PAGE>p3 '); // 선언 없는 마커(+1 경로)
  assert.equal(syncedStreamPageNo(pane, g), pane);
});

test('syncedStreamPageNo: 재연결 갭 뒤 디바이더 번호를 ground 페이지로 재동기화', () => {
  const g = createGroundState();
  // 페이지 3 마커까지 정상 수신(pane=3) 후 스트림 단절 — 갭 동안 폴링
  // announce가 페이지 7까지 진행시키고 마커 4~7은 유실됐다.
  groundAnnounce(g, 'ocr', 3, 25);
  groundPush(g, '<PAGE>');
  groundDrain(g, false);
  groundAnnounce(g, 'ocr', 7, 25); // 갭 동안의 마지막 선언 (expectAnnounce=true)
  assert.equal(syncedStreamPageNo(3, g), 6, '다음 마커는 7 시작 → 6으로 끌어올림');
  // 페이지 7의 마커가 소비된 뒤라면 다음 마커는 8 시작 → 7
  groundPush(g, '<PAGE>');
  groundDrain(g, false);
  assert.equal(syncedStreamPageNo(3, g), 7);
  assert.equal(syncedStreamPageNo(7, g), 7, '이미 맞으면 no-op');
  assert.equal(syncedStreamPageNo(9, g), 9, '절대 뒤로 가지 않는다 (여분 마커 허용)');
});

test('syncedStreamPageNo: 실행 중 잡을 중간에 연 경우(스냅샷 선언) 첫 디바이더 보정', () => {
  const g = createGroundState();
  groundAnnounce(g, 'ocr', 42, 100); // openJob 스냅샷이 페이지를 시드
  // streamPageNo=0이라면 다음 마커의 디바이더는 "페이지 42"여야 한다 → 41
  assert.equal(syncedStreamPageNo(0, g), 41);
});

/* ================= translation pure core ================= */

test('withLangUrl: appends ?lang=ko only for ko, preserving existing query', () => {
  // 원문(orig)은 URL을 건드리지 않는다
  assert.equal(withLangUrl('/api/jobs/x/markdown', 'orig'), '/api/jobs/x/markdown');
  assert.equal(withLangUrl('/api/jobs/x/markdown', undefined), '/api/jobs/x/markdown');
  // ko는 쿼리 파라미터 추가
  assert.equal(withLangUrl('/api/jobs/x/markdown', 'ko'), '/api/jobs/x/markdown?lang=ko');
  assert.equal(withLangUrl('/api/jobs/x/layout', 'ko'), '/api/jobs/x/layout?lang=ko');
  // 기존 쿼리가 있으면 &로 이어붙인다
  assert.equal(withLangUrl('/api/jobs/x/files?r=1', 'ko'), '/api/jobs/x/files?r=1&lang=ko');
  // falsy URL은 그대로 반환
  assert.equal(withLangUrl(null, 'ko'), null);
  assert.equal(withLangUrl('', 'ko'), '');
});

test('translateUiStateFor: state → control mapping', () => {
  assert.equal(translateUiStateFor('running'), 'progress');
  assert.equal(translateUiStateFor('done'), 'toggle');
  // none/error/canceled/미지 값은 모두 버튼(재시도)
  assert.equal(translateUiStateFor('none'), 'button');
  assert.equal(translateUiStateFor('error'), 'button');
  assert.equal(translateUiStateFor('canceled'), 'button');
  assert.equal(translateUiStateFor(undefined), 'button');
});

/* ================= upload size preflight (fileSizeError) ================= */

test('fileSizeError: 경계값 — 정확히 상한이면 통과, 1바이트 초과부터 차단', () => {
  const limitBytes = 100 * 1024 * 1024;
  assert.equal(fileSizeError(limitBytes, 100), null, '상한과 같으면 허용');
  const err = fileSizeError(limitBytes + 1, 100);
  assert.ok(err && err.includes('파일이 너무 큽니다'), '초과분은 안내 문구 반환');
  assert.ok(err.includes('서버 상한 100MB'), '서버 상한이 문구에 표기된다');
});

test('fileSizeError: 문구에 파일 크기와 서버 상한이 함께 표기된다', () => {
  assert.equal(fileSizeError(150 * 1024 * 1024, 100),
    '파일이 너무 큽니다 (150 MB — 서버 상한 100MB)');
});

test('fileSizeError: 반올림 경계에서 자기모순 문구를 피한다 (100 MB — 상한 100MB 금지)', () => {
  // 상한+1바이트는 fmtBytes 반올림상 '100 MB'로 표시돼 상한과 같아 보이므로
  // 크기 병기를 생략하고 '상한 초과'로만 안내한다.
  assert.equal(fileSizeError(100 * 1024 * 1024 + 1, 100),
    '파일이 너무 큽니다 (서버 상한 100MB 초과)');
});

test('fileSizeError: health 미수신/비정상 상한이면 검증 생략 (서버 413이 최후 방어)', () => {
  const big = 500 * 1024 * 1024;
  assert.equal(fileSizeError(big, undefined), null, '필드 부재(undefined)');
  assert.equal(fileSizeError(big, null), null);
  assert.equal(fileSizeError(big, 0), null, '0 이하 상한은 무시');
  assert.equal(fileSizeError(big, -1), null);
  assert.equal(fileSizeError(big, 'abc'), null, '숫자가 아닌 상한은 무시');
});

/* ================= two-step delete confirm (armTransition) ================= */

test('armTransition: first click arms, second click on the same key confirms', () => {
  const btn = { id: 'ji-del' };
  // 첫 클릭 — 무장만 하고 아무것도 회수하지 않는다
  assert.deepEqual(armTransition([], 'job-1', btn), { confirm: false, clearKeys: [] });
  // 같은 키 재클릭 — 확인(실삭제) + 해당 키 회수
  assert.deepEqual(armTransition([['job-1', btn]], 'job-1', btn), { confirm: true, clearKeys: ['job-1'] });
});

test('armTransition: key-based arming survives a list re-render (button replaced)', () => {
  const oldBtn = { id: 'old' };
  const newBtn = { id: 'new' };
  // 5초 주기 renderJobList가 버튼을 새로 만들어도 키가 같으면 confirm이다
  // (armed 항목의 entry.btn은 재렌더 시 최신 버튼으로 교체된다)
  assert.equal(armTransition([['job-1', newBtn]], 'job-1', newBtn).confirm, true);
  assert.equal(armTransition([['job-1', oldBtn]], 'job-1', newBtn).confirm, true);
});

test('armTransition: header click after job switch clears the stale arm instead of deleting', () => {
  const hdr = { id: 'job-delete' };
  // 잡 A에서 무장된 헤더 버튼으로 잡 B를 클릭 → 오삭제 없이 옛 무장 회수 + 재무장
  assert.deepEqual(armTransition([['header:A', hdr]], 'header:B', hdr),
    { confirm: false, clearKeys: ['header:A'] });
});

test('armTransition: arms on different buttons stay independent', () => {
  const hdr = { id: 'job-delete' };
  const item = { id: 'ji-del' };
  // 목록 항목이 무장돼 있어도 헤더 버튼 클릭은 confirm도 회수도 아니다
  assert.deepEqual(armTransition([['job-1', item]], 'header:job-1', hdr),
    { confirm: false, clearKeys: [] });
  // 반대 방향도 동일
  assert.deepEqual(armTransition([['header:job-1', hdr]], 'job-1', item),
    { confirm: false, clearKeys: [] });
});

/* ================= 상태 라벨 + 대기열 위치 (statusLabel) ================= */

test('statusLabel: queued + queue_position → 대기중 · N번째', () => {
  assert.equal(statusLabel({ status: 'queued', queue_position: 1 }), '대기중 · 1번째');
  assert.equal(statusLabel({ status: 'queued', queue_position: 3 }), '대기중 · 3번째');
});

test('statusLabel: 필드 부재·비정상 값·다른 상태는 기존 라벨 그대로 (안전 폴백)', () => {
  assert.equal(statusLabel({ status: 'queued' }), '대기중', '필드 부재(구버전 서버·SSE 스냅샷)');
  assert.equal(statusLabel({ status: 'queued', queue_position: 0 }), '대기중', '1-base 미만 무시');
  assert.equal(statusLabel({ status: 'queued', queue_position: '2' }), '대기중', '문자열 무시');
  assert.equal(statusLabel({ status: 'queued', queue_position: 1.5 }), '대기중', '비정수 무시');
  assert.equal(statusLabel({ status: 'running', queue_position: 2 }), '변환중', 'queued 외 상태에는 붙지 않는다');
  assert.equal(statusLabel({ status: 'done' }), '완료');
  assert.equal(statusLabel({ status: 'weird' }), 'weird', '미지 상태는 원문 표기');
});

/* ================= location.hash 잡 복원 (jobIdFromHash) ================= */

test('jobIdFromHash: 해시에서 잡 id 추출 — 선행 # 제거', () => {
  assert.equal(jobIdFromHash('#abc'), 'abc');
  assert.equal(jobIdFromHash('#j_0a1b2c3d4e5f'), 'j_0a1b2c3d4e5f', '실제 잡 id 형식(j_ + hex)');
  assert.equal(jobIdFromHash('abc'), 'abc', '# 없는 입력도 방어적으로 허용');
});

test('jobIdFromHash: 빈/이상값은 null', () => {
  assert.equal(jobIdFromHash(''), null);
  assert.equal(jobIdFromHash('#'), null);
  assert.equal(jobIdFromHash(null), null);
  assert.equal(jobIdFromHash(undefined), null);
  assert.equal(jobIdFromHash('#foo/bar'), null, '잡 id에 없는 문자(/) 거부');
  assert.equal(jobIdFromHash('#<script>'), null, '이상값 거부');
});

/* ================= 다중 업로드: 검증 분류·선택 요약 ================= */

test('classifyFiles: 유효/무효(파일명+사유) 분리 — 순서 유지, 전부 무효면 선택 없음', () => {
  const files = [
    { name: 'a.pdf', size: 10 },
    { name: 'b.txt', size: 10 },
    { name: 'c.pdf', size: 999 },
  ];
  const validate = (f) => (/\.pdf$/i.test(f.name) ? (f.size > 100 ? '너무 큼' : null) : 'PDF 아님');
  const { valid, skipped } = classifyFiles(files, validate);
  assert.deepEqual(valid.map((f) => f.name), ['a.pdf']);
  assert.deepEqual(skipped.map((s) => [s.name, s.reason]),
    [['b.txt', 'PDF 아님'], ['c.pdf', '너무 큼']]);
  assert.equal(skipped[0].file, files[1], '재시도용 파일 참조 보존');
  assert.equal(classifyFiles(files, () => 'x').valid.length, 0, '전부 무효 → 유효 없음');
  assert.deepEqual(classifyFiles([], validate), { valid: [], skipped: [] });
});

test('selectionSummary: 1개면 이름·크기 그대로, 여러 개면 N개 파일 · 총 X', () => {
  assert.equal(selectionSummary([]), null, '빈 선택은 null (file-info 숨김)');
  assert.deepEqual(selectionSummary([{ name: 'doc.pdf', size: 512 }]),
    { name: 'doc.pdf', size: '512 B', title: 'doc.pdf' }, '단일 파일은 기존 표시와 동일');
  const s = selectionSummary([
    { name: 'a.pdf', size: 1024 * 1024 },
    { name: 'b.pdf', size: 2 * 1024 * 1024 },
  ]);
  assert.equal(s.name, '2개 파일');
  assert.equal(s.size, '총 3.0 MB', '총합 크기 표기');
  assert.equal(s.title, 'a.pdf, b.pdf', 'title에 파일명 나열');
});

test('summarizeIssues: 첫 건 + 나머지 개수 축약 (건너뜀/업로드 실패 공용)', () => {
  assert.equal(summarizeIssues('건너뜀', []), null);
  assert.equal(summarizeIssues('건너뜀', [{ name: 'b.txt', reason: 'PDF 아님' }]),
    '건너뜀: b.txt — PDF 아님');
  assert.equal(summarizeIssues('업로드 실패', [
    { name: 'a.pdf', reason: '네트워크 오류' },
    { name: 'b.pdf', reason: '서버 오류' },
    { name: 'c.pdf', reason: '서버 오류' },
  ]), '업로드 실패: a.pdf — 네트워크 오류 외 2건');
});

/* ================= 멀티 엔진 health/잡 메타 (신규 계약) ================= */

test('healthCapabilities: 신규 health 응답에서 capability 추출', () => {
  const d = {
    engine: 'ovisocr2',
    model_id: 'ATH-MaaS/OvisOCR2',
    provider: 'local-sidecar',
    capabilities: { multi_page_context: false, stream_granularity: 'page', layout: 'figure_only', figures: true },
  };
  assert.deepEqual(healthCapabilities(d), {
    engine: 'ovisocr2', streamGranularity: 'page', layoutCapability: 'figure_only',
  });
});

test('healthCapabilities: legacy health(capabilities 부재)는 전부 undefined — 기존 UI 그대로', () => {
  const legacy = { engine: 'unlimited', device: 'cuda', model_id: 'baidu/Unlimited-OCR', model_loaded: true };
  const hc = healthCapabilities(legacy);
  assert.equal(hc.engine, 'unlimited');
  assert.equal(hc.streamGranularity, undefined);
  assert.equal(hc.layoutCapability, undefined);
  assert.deepEqual(healthCapabilities(null), { engine: undefined, streamGranularity: undefined, layoutCapability: undefined });
  assert.deepEqual(healthCapabilities({ capabilities: 'evil' }).streamGranularity, undefined, '비객체 capabilities 방어');
});

test('healthCapabilities: unlimited 토큰 스트리밍은 page 칩을 켜지 않는다', () => {
  const d = { engine: 'unlimited', capabilities: { stream_granularity: 'token', layout: 'full' } };
  const hc = healthCapabilities(d);
  assert.equal(hc.streamGranularity, 'token');
  assert.notEqual(hc.streamGranularity, 'page', 'token 엔진에는 페이지 단위 칩 없음');
});

test('providerIssue: sidecar 다운이면 요약 반환, 정상/in-process면 null', () => {
  assert.equal(providerIssue({ provider: 'local-sidecar', provider_health: { status: 'ok' } }), null);
  assert.equal(providerIssue({ provider: 'in-process', provider_health: null }), null);
  assert.equal(providerIssue({ provider: 'in-process' }), null, 'legacy 응답(provider 부재 포함)');
  assert.equal(providerIssue(undefined), null);
  assert.equal(
    providerIssue({ provider: 'local-sidecar', provider_health: { status: 'unreachable', error: '연결 거부' } }),
    '연결 거부');
  assert.equal(
    providerIssue({ provider: 'local-sidecar', provider_health: { status: 'error' } }),
    'error', 'error 필드 없으면 status로 폴백');
});

test('jobModelChip: 잡 모델 메타 → 칩 텍스트/툴팁, 구버전 잡은 null', () => {
  assert.equal(jobModelChip({ job_id: 'j', filename: 'a.pdf' }), null, '메타 없는 구버전 잡');
  assert.equal(jobModelChip(null), null);
  const chip = jobModelChip({
    engine: 'paddleocr_vl', model_id: 'PaddlePaddle/PaddleOCR-VL-1.6',
    model_revision: '66317acc4c9fc17bd154591ce650735cd2855f3e', provider: 'local-sidecar',
  });
  assert.equal(chip.text, 'PaddlePaddle/PaddleOCR-VL-1.6');
  assert.ok(chip.title.includes('@ 66317acc'), 'revision 축약 표기');
  assert.ok(chip.title.includes('engine: paddleocr_vl'));
  // model_id 없이 engine만 있어도 칩은 뜬다
  assert.equal(jobModelChip({ engine: 'ovisocr2' }).text, 'ovisocr2');
});

test('docLayoutNoteFor: figure_only 엔진 잡에만 안내, 엔진 불일치·full은 null', () => {
  assert.ok(docLayoutNoteFor('figure_only', 'ovisocr2', 'ovisocr2'), '현재 엔진 잡 → 안내');
  assert.equal(docLayoutNoteFor('figure_only', 'unlimited', 'ovisocr2'), null, '다른 엔진의 잡 → 무표시');
  assert.equal(docLayoutNoteFor('full', 'paddleocr_vl', 'paddleocr_vl'), null, 'full layout은 안내 불필요');
  assert.equal(docLayoutNoteFor(undefined, undefined, undefined), null, 'legacy health');
});

test('docLayoutNoteFor: 구버전 잡(엔진 메타 없음)에는 안내하지 않는다', () => {
  // 엔진 메타가 없는 잡 = 이 기능 이전 변환 = Unlimited(full layout)로 만든 결과.
  // 현재 엔진이 figure_only여도 그 잡의 레이아웃은 완전하므로 오안내 금지.
  assert.equal(docLayoutNoteFor('figure_only', undefined, 'ovisocr2'), null);
  assert.equal(docLayoutNoteFor('figure_only', null, 'ovisocr2'), null);
});
