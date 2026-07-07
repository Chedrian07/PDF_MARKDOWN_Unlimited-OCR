/* Unlimited-OCR — 레이아웃 뷰 클라이언트 맞춤(shrink-to-fit) 패스.
 *
 * 앱 뷰(app.js가 loadDocLayout에서 호출)와 다운로드용 standalone
 * (layout.py가 이 파일을 그대로 <script>로 인라인)이 공유하는 단일 소스.
 * → 반드시 순수 클래식 스크립트(모듈 문법 금지)로 유지할 것.
 *
 * 서버(layout.py)는 면적 기반으로 폰트 크기를 cqw 단위로 인라인하되 일부러
 * 작게(underfill) 잡는다. 이 패스는 넘치는 블록만 축소해 맞추고(확대는 안 함),
 * 수식/표가 가로로 넘치면 transform:scale로 눌러 담는다.
 */
(function () {
  'use strict';

  // ── 세로 shrink-to-fit ──────────────────────────────────────────────
  // CRITICAL(cqw 단위 보존): 서버가 font-size를 cqw로 심었으면 축소값도 cqw로
  // 되써야 창 리사이즈 시 폰트가 캔버스 폭에 비례해 따라간다(container-type:
  // inline-size). px로 되쓰면 리사이즈 비례성이 깨진다.
  // 멱등성: 최초 접촉 시 원본 크기를 el.dataset.uocrBaseFs에 저장하고 항상
  // 그 base에서 다시 축소한다 — 재실행/리사이즈에도 누적되지 않는다.
  function shrinkVertical(el) {
    var baseVal, unit;
    var baseRaw = el.dataset.uocrBaseFs;
    if (baseRaw) {
      var bm = /^([\d.]+)(cqw|px)$/.exec(baseRaw);
      if (bm) { baseVal = parseFloat(bm[1]); unit = bm[2]; }
    }
    if (baseVal == null) {
      var im = /^([\d.]+)cqw$/.exec(el.style.fontSize || '');
      if (im) {
        baseVal = parseFloat(im[1]);
        unit = 'cqw';
      } else {
        // cqw 인라인이 없으면 계산된 px로 폴백(앱 뷰의 기본 CSS 등).
        baseVal = parseFloat(getComputedStyle(el).fontSize);
        unit = 'px';
      }
      if (!(baseVal > 0)) return;
      el.dataset.uocrBaseFs = baseVal + unit;
    }
    // 항상 base에서 재시작.
    el.style.fontSize = baseVal + unit;
    var floorVal = baseVal * 0.55; // 시작 크기의 55%가 바닥
    var cur = baseVal;
    for (var pass = 0; pass < 5; pass++) {
      var sh = el.scrollHeight;
      var ch = el.clientHeight;
      if (!(sh > ch + 1) || sh <= 0 || ch <= 0) break;
      var factor = Math.max(ch / sh, 0.9); // 패스당 최대 10% 축소
      cur = Math.max(cur * factor, floorVal);
      el.style.fontSize = cur + unit;
      if (cur <= floorVal) break;
    }
  }

  // ── 가로 맞춤: 수식/표가 블록 폭을 넘치면 transform:scale로 눌러 담기 ──
  function scaleHorizontal(block, elms) {
    var avail = block.clientWidth;
    if (!(avail > 0)) return;
    for (var i = 0; i < elms.length; i++) {
      var e = elms[i];
      e.style.transform = '';          // 재실행 시 자연 폭 재측정
      var natural = e.scrollWidth;
      if (natural > avail) {
        e.style.transformOrigin = 'left top';
        e.style.transform = 'scale(' + (avail / natural) + ')';
      }
    }
  }

  function fitBlock(block) {
    try {
      if (block.classList.contains('layout-image')) return;
      // 세로쓰기 블록(writing-mode)은 scrollHeight 축이 달라 피팅 제외 —
      // 실측 폰트 크기가 정확하고 단일 줄 스탬프라 축소가 필요 없다.
      if (/layout-vertical-/.test(block.className)) return;
      shrinkVertical(block);
      var maths = block.querySelectorAll('.math-display');
      if (maths.length) scaleHorizontal(block, maths);
      var tables = [];
      for (var i = 0; i < block.children.length; i++) {
        if (block.children[i].tagName === 'TABLE') tables.push(block.children[i]);
      }
      if (tables.length) scaleHorizontal(block, tables);
    } catch (_) { /* 블록 하나 실패해도 전체 패스는 계속 (never throw) */ }
  }

  window.uocrFitLayout = function (root) {
    if (!root) return;
    var run = function () {
      var blocks = root.querySelectorAll('.layout-block');
      for (var i = 0; i < blocks.length; i++) fitBlock(blocks[i]);
    };
    // 레이아웃이 확정된 뒤(폰트 로드·KaTeX 렌더 반영) 측정하도록 rAF로 감싼다.
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else run();
  };
})();
