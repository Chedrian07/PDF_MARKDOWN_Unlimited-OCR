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
  incompleteTailIndex,
  scanQuads,
  withLangUrl,
  translateUiStateFor,
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
