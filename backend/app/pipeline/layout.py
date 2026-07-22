"""좌표 기반 레이아웃 뷰 (Phase B) — 마크다운 뷰와 독립적인 부가 산출물.

벤더 P14가 남긴 raw_pages.json(치환 전 그라운딩 태그 원문)을 파싱해
페이지별 블록(type, bbox 0–999 정규화, content)을 만들고(merge.py가 layout.json으로
통합), 절대 배치된 HTML로 재구성한다.

**트레이드오프(의도된 한계)**: 웹 폰트·자간이 원본과 달라 텍스트가 박스를
넘치거나 남을 수 있다 — `overflow:hidden`과 대략적 폰트 스케일만 적용하는
best-effort 재현이며 완벽 재현은 목표가 아니다. 텍스트 리플로우·검색·편집성은
마크다운 뷰가 담당한다.

XSS: 블록 텍스트는 escapeHtml로 이스케이프하고, 표만 render.py와 동일한
화이트리스트(_restore_table_tags)로 복원한다. 서버가 만든 좌표/이름만 주입.
"""

from __future__ import annotations

import re

import base64
import functools
from pathlib import Path

from markdown_it.common.utils import escapeHtml

from .render import _restore_table_tags, text_with_math_html

# 벤더 re_match와 동일한 두 그라운딩 문법
_REF_BLOCK = re.compile(r"<\|ref\|>(.{1,40}?)<\|/ref\|><\|det\|>(.{0,400}?)<\|/det\|>", re.DOTALL)
_DET_INLINE = re.compile(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[([^\]]+)\]\s*<\|/det\|>")
_SPECIAL = re.compile(r"<\|[^|>]{0,64}\|>")
_SAFE_TYPE = re.compile(r"^[a-z][a-z0-9_-]{0,24}$")


def _quads(payload: str) -> list[tuple[int, int, int, int]]:
    nums = [int(n) for n in re.findall(r"\d+", payload)]
    return [tuple(nums[i : i + 4]) for i in range(0, len(nums) - 3, 4)]


def parse_page_blocks(raw: str) -> list[dict]:
    """한 페이지 raw 출력 → 문서 순서의 블록 리스트.

    image 블록의 `crop_index`는 벤더 저장 순서와 동일해야 크롭 파일과 매핑된다:
    re_match는 ref류 매치 전체를 먼저, inline det류를 나중에 모으고
    draw_bounding_boxes가 그 순서로 image 크롭 인덱스를 증가시킨다 —
    (ref류 이미지의 박스 각각이 크롭 1개씩) 그 순서를 그대로 재현한다.
    """
    ref_events = []
    for m in _REF_BLOCK.finditer(raw):
        ref_events.append({
            "start": m.start(), "end": m.end(),
            "label": m.group(1).strip(), "boxes": _quads(m.group(2)), "kind": "ref",
        })
    ref_spans = [(e["start"], e["end"]) for e in ref_events]

    det_events = []
    for m in _DET_INLINE.finditer(raw):
        # ref 블록 내부의 det 태그는 중복 매치 — 제외
        if any(s <= m.start() < e for s, e in ref_spans):
            continue
        det_events.append({
            "start": m.start(), "end": m.end(),
            "label": m.group(1).strip(), "boxes": _quads(f"[{m.group(2)}]"), "kind": "det",
        })

    # 크롭 인덱스: 벤더 순서 (ref류 전체 → det류), image 라벨의 박스당 1개
    crop_index = 0
    for e in ref_events + det_events:
        if e["label"] == "image":
            e["crop_indices"] = list(range(crop_index, crop_index + len(e["boxes"])))
            crop_index += len(e["boxes"])

    events = sorted(ref_events + det_events, key=lambda e: e["start"])
    blocks: list[dict] = []
    for i, e in enumerate(events):
        content_end = events[i + 1]["start"] if i + 1 < len(events) else len(raw)
        content = _SPECIAL.sub("", raw[e["end"]:content_end]).strip()
        for bi, box in enumerate(e["boxes"]):
            block: dict = {"type": e["label"].lower(), "bbox": list(box), "content": content}
            if "crop_indices" in e:
                block["crop_index"] = e["crop_indices"][bi]
                block["content"] = ""
            blocks.append(block)
    return blocks


def _pct(v: int) -> str:
    return f"{v / 999 * 100:.2f}"


# ── 면적 기반 폰트 크기 추정 (cqw 단위, 해상도 독립) ──────────────────────
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# CJK/전각: 한글 자모·음절, 한자(라디컬 포함), 가나, 전각 폼 — 폭 1.0em 취급
_CJK_RE = re.compile(
    r"[ᄀ-ᇿ⺀-鿿ꥠ-꥿가-퟿豈-﫿＀-￯]"
)

# 면적 모델의 채움 상수. 글자 1개가 대략 (fs·weight) 폭 × (fs·line-height) 높이를
# 차지한다 → 글자당 면적 ≈ fs²·weight·line-height. 박스 면적을 가중 글자수로 나눠
# fs를 역산하며, 이 분모 상수가 곧 "글자당 면적 / fs²"의 기하 항 line-height(≈1.2)다.
# 진실 앵커(2504.19874v1.pdf 실측): 612×792pt 페이지·본문 10.9pt = 1.78cqw. 이를
# 재현하는 케이스(bbox(60,100,940,280)·A4 비율·ASCII 1180자)에서 fs≈1.78cqw가
# 나오도록 1.2로 맞췄다. (구 4.1은 "600자 수용" 잘못된 앵커에서 온 값 — 실제 그
# 박스는 원본 타이포로 ~1180자를 담아 폰트를 55% 크기로 과소추정했다.)
# 이 함수는 어디까지나 폴백이다: 원본 PDF 텍스트 레이어가 있으면 pdf_fonts가
# block["fs"]에 실측값을 심고 렌더러가 그걸 우선한다.
_AREA_FILL = 1.2


def estimate_font_size_cqw(bbox, content: str, page_aspect: float) -> float | None:
    """블록 면적·가중 글자수로 폰트 크기(cqw)를 추정. 불가하면 None(CSS 기본값 유지).

    - bbox = (x1,y1,x2,y2) 0–999 정규화, page_aspect = 페이지높이px / 페이지폭px.
    - cqw = 캔버스 폭의 1% (container-type: inline-size). 창 크기가 바뀌어도
      비율이 유지되는 해상도 독립 단위.
    - 태그 제거(표 블록은 <table> 마크업 포함) → 공백 런 축약 → 가중 글자수:
      ASCII 0.5, CJK/전각 1.0, 그 외 0.65. 가중합<1 또는 빈 텍스트면 None.
    """
    if not content:
        return None
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    text = _WS_RE.sub(" ", _TAG_RE.sub(" ", content)).strip()
    if not text:
        return None
    weighted = 0.0
    for ch in text:
        if ord(ch) < 128:
            weighted += 0.5
        elif _CJK_RE.match(ch):
            weighted += 1.0
        else:
            weighted += 0.65
    if weighted < 1:
        return None
    w = (x2 - x1) / 999 * 100
    h = (y2 - y1) / 999 * 100 * page_aspect
    if w <= 0 or h <= 0:
        return None
    # 면적 모델: fs² · _AREA_FILL · weighted = w·h  →  fs = sqrt(w·h / (_AREA_FILL·weighted))
    fs = ((w * h) / (_AREA_FILL * weighted)) ** 0.5
    fs = min(fs, h / 1.25)          # 단일 줄 상한 — 얕은 박스에서 과대추정 방지
    return max(0.8, min(3.6, fs))   # cqw 클램프 [0.8, 3.6]


def render_layout_html(
    pages: list[dict], files_base_url: str, image_src=None, lang: str | None = None
) -> str:
    """layout.json(merge가 통합한 페이지 블록들) → 절대 배치 HTML 프래그먼트.
    image_src(name)->str 을 주면 이미지 src를 그 값으로 (standalone의 data URI용).
    lang을 주면 최상위 컨테이너(.doclayout-body)에 lang 속성을 부여해
    `[lang="ko"] .layout-block` 규칙(한글 서리프·word-break)이 적용되게 한다.
    (standalone은 <main>에 직접 lang을 넣으므로 여기로 전달하지 않는다.)"""
    sections: list[str] = []
    for p in pages:
        width = p.get("width") or 1000
        height = p.get("height") or 1414
        aspect_pct = height / width * 100 if width else 141.4
        page_aspect = height / width if width else 1.414
        blocks_html: list[str] = []
        for b in p.get("blocks", ()):
            bbox = b.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (max(0, min(999, int(v))) for v in bbox)
            style = (
                f"left:{_pct(x1)}%;top:{_pct(y1)}%;"
                f"width:{_pct(max(x2 - x1, 4))}%;height:{_pct(max(y2 - y1, 4))}%;"
            )
            btype = b.get("type") or "text"
            if not _SAFE_TYPE.match(btype):
                btype = "text"
            image_name = b.get("image")
            if image_name and re.fullmatch(r"[\w.-]+", str(image_name)):
                src = image_src(image_name) if image_src else f"{files_base_url}/images/{image_name}"
                blocks_html.append(
                    f'<img class="layout-block layout-image" '
                    f'src="{src}" style="{style}" alt="">'
                )
                continue
            # 이스케이프 + \(..\)/\[..\] 구간은 KaTeX 스팬으로 (클라이언트가 타이포셋)
            content = text_with_math_html(str(b.get("content") or ""))
            if btype == "table":
                content = _restore_table_tags(content)

            # 세로쓰기 블록 (arXiv 왼쪽 여백 스탬프 등 90° 회전 텍스트):
            # 텍스트 레이어의 줄 방향(pdf_fonts block["vertical"])에서 감지되고,
            # 텍스트 레이어가 없으면(스캔 PDF) 극단적으로 좁고 긴 박스를 세로로 간주.
            vertical = b.get("vertical") if b.get("vertical") in ("up", "down") else None
            if vertical is None and "fs" not in b:
                w_cqw = (x2 - x1) / 999 * 100
                h_cqw = (y2 - y1) / 999 * 100 * page_aspect
                if w_cqw > 0 and h_cqw / w_cqw >= 6 and len(str(b.get("content") or "")) >= 12:
                    vertical = "up"
            vcls = f" layout-vertical-{vertical}" if vertical else ""
            # 폰트 크기(cqw): pdf_fonts가 심은 실측 block["fs"]를 우선([0.6,6.0] 클램프),
            # 없으면 면적 휴리스틱으로 폴백. None이면 CSS 기본값 유지.
            # KaTeX도 이 font-size를 상속하므로 수식이 박스와 함께 스케일된다.
            raw_fs = b.get("fs")
            if isinstance(raw_fs, (int, float)):
                fs = max(0.6, min(6.0, float(raw_fs)))
            else:
                fs = estimate_font_size_cqw((x1, y1, x2, y2), content, page_aspect)
            if fs is not None:
                # 논문 조판은 줄간격 ≈1.15–1.2 — 1.22로 좁힌다(잔여 오버플로우는
                # 클라이언트 fitter가 축소로 흡수). 볼드 실측 블록은 굵게(제목 제외 —
                # 제목은 이미 CSS로 굵다).
                fs_css = f"font-size:{fs:.2f}cqw;line-height:1.22;"
                if b.get("bold") and btype != "title":
                    fs_css += "font-weight:600;"
            else:
                fs_css = ""
            blocks_html.append(
                f'<div class="layout-block layout-{btype}{vcls}" style="{style}{fs_css}" '
                f'title="{btype}">{content}</div>'
            )
        sections.append(
            f'<section class="layout-page" data-page="{int(p.get("page", 0))}">'
            f'<div class="layout-canvas" style="padding-top:{aspect_pct:.2f}%">'
            + "".join(blocks_html)
            + "</div></section>"
        )
    inner = "\n".join(sections)
    if lang:
        # 페이지 섹션을 감싸는 최상위 컨테이너. .doclayout-body 클래스를 유지해
        # 인앱 주입 시(#doclayout-body 안) 페이지 간 그리드 간격이 보존된다.
        return f'<div class="doclayout-body" lang="{lang}">\n{inner}\n</div>'
    return inner


# ── standalone HTML (다운로드용 단일 파일 — PDF 대응 뷰) ─────────────────
# ⚠ SYNC: frontend/styles.css의 .layout-* 규칙과 시각적으로 동기 유지할 것
# (.layout-block의 pre-wrap/justify/serif, .layout-canvas의 container-type 포함).
_STANDALONE_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; margin: 0; }
body { background: #eceef2; font-family: system-ui, -apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif; padding: 24px 12px; }
.doclayout-body { display: grid; gap: 26px; max-width: 900px; margin-inline: auto; }
.layout-page::before { content: "페이지 " attr(data-page); display: block; font-size: 11px; color: #666; margin-bottom: 4px; }
.layout-canvas { position: relative; width: 100%; height: 0; background: #fff; border: 1px solid #d5d7dd; border-radius: 6px; box-shadow: 0 1px 5px rgba(0,0,0,.1); overflow: hidden; container-type: inline-size; }
.layout-block { position: absolute; overflow: hidden; font-size: 11px; line-height: 1.35; color: #1a1c22; padding: 1px 3px; white-space: pre-wrap; text-align: justify; hyphens: auto; font-family: Georgia, 'Times New Roman', 'Noto Serif KR', serif; }
.layout-title { font-weight: 700; font-size: 14px; color: #b0262c; }
.layout-image { object-fit: contain; padding: 0; }
.layout-table { font-size: 9px; }
.layout-table table { border-collapse: collapse; width: 100%; }
.layout-table td, .layout-table th { border: 1px solid #c9c9d4; padding: 1px 3px; }
.layout-formula, .layout-equation { font-size: 11px; display: flex; align-items: center; justify-content: center; }
.layout-page_number, .layout-header, .layout-footer, .layout-footnote { opacity: .45; font-size: 9px; }
.layout-block .math-display { display: block; text-align: center; }
/* ⚠ SYNC: frontend/styles.css에도 동일 규칙 (번역 뷰 한글 타이포) — 앱의 <html lang="ko">에
   오적용되지 않도록 lang을 받는 래퍼(.doclayout-body) 자체에 스코프 */
.doclayout-body[lang="ko"] .layout-block { font-family: "Noto Serif KR", "Source Han Serif K", "Apple SD Gothic Neo", "Malgun Gothic", serif; word-break: keep-all; }
.layout-vertical-up { writing-mode: sideways-lr; white-space: nowrap; text-align: center; }
.layout-vertical-down { writing-mode: vertical-rl; white-space: nowrap; text-align: center; }
@media print { body { background: #fff; padding: 0; } .layout-page { break-inside: avoid; } }
"""

_TYPESET_JS = (
    "document.querySelectorAll('.math-inline,.math-display').forEach(function(e){"
    "try{katex.render(e.textContent,e,{displayMode:e.classList.contains('math-display'),"
    "throwOnError:false});}catch(_){}});"
)


@functools.lru_cache(maxsize=1)
def _katex_inline_bundle(frontend_dir_str: str) -> str:
    """벤더 KaTeX css(woff2 폰트 data-URI 인라인)+js — 단일 파일 배포용.
    자산이 없으면 빈 문자열 (수식은 원문 LaTeX 표기로 폴백)."""
    fd = Path(frontend_dir_str) / "vendor" / "katex"
    css_p, js_p = fd / "katex.min.css", fd / "katex.min.js"
    if not css_p.is_file() or not js_p.is_file():
        return ""

    def _font(m: re.Match) -> str:
        p = fd / m.group(1)
        if p.is_file() and p.suffix == ".woff2":
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"url(data:font/woff2;base64,{b64})"
        return "url(data:,)"  # woff/ttf 미벤더 — 브라우저는 woff2를 우선 선택

    css = re.sub(r"url\((fonts/[^)]+)\)", _font, css_p.read_text(encoding="utf-8"))
    js = js_p.read_text(encoding="utf-8")
    return (
        f"<style>{css}</style><script>{js}</script>"
        f"<script>window.addEventListener('DOMContentLoaded',function(){{{_TYPESET_JS}}});</script>"
    )


def _layout_fit_script(frontend_dir: Path | None) -> str:
    """frontend/layout-fit.js를 인라인 <script>로. 앱 뷰와 동일한 단일 소스.
    캐시하지 않고 매번 읽는다(작은 파일 — dev 중 수정이 즉시 반영되도록;
    lru_cache된 KaTeX 번들과 달리 캐시 키 혼동을 피함). 파일이 없으면
    빈 문자열 — fitter 없이도 정상 동작(그레이스풀 디그레이드)."""
    if not frontend_dir:
        return ""
    try:
        js = (Path(frontend_dir) / "layout-fit.js").read_text(encoding="utf-8")
    except OSError:
        return ""
    # KaTeX 타이포셋(위 _katex_inline_bundle의 DOMContentLoaded) 뒤에 등록되도록
    # head에서 katex 다음에 배치한다 → 순서: 타이포셋 → uocrFitLayout(document).
    return (
        f"<script>{js}</script>"
        "<script>window.addEventListener('DOMContentLoaded',function(){"
        "if(window.uocrFitLayout){try{window.uocrFitLayout(document);}catch(_){}}"
        "});</script>"
    )


def _image_data_uri(job_dir: Path, name: str) -> str:
    """images/{name}을 data URI로 — standalone 파일(layout/document) 공용."""
    p = job_dir / "images" / name
    try:
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except OSError:
        return "data:,"  # 결측 크롭 — 빈 이미지로 폴백


def render_layout_standalone(
    pages: list[dict], job_dir: Path, title: str, frontend_dir: Path | None,
    lang: str | None = None,
) -> str:
    """이미지 base64·KaTeX 인라인의 완전 자립형 HTML 문서 — 오프라인에서 그대로 열림.
    lang을 주면 <html>·<main>에 lang 속성을 부여해 번역본에 `[lang="ko"] .layout-block`
    (한글 서리프·word-break) 규칙이 적용된다. 원본(lang=None)에는 lang을 붙이지 않아
    비한국어 문서에 한글 타이포가 잘못 적용되는 것을 막는다."""
    # body에는 lang을 전달하지 않는다 — 컨테이너(<main>)에서 한 번만 부여.
    body = render_layout_html(
        pages, files_base_url="", image_src=lambda name: _image_data_uri(job_dir, name),
    )
    katex = _katex_inline_bundle(str(frontend_dir)) if frontend_dir else ""
    fitter = _layout_fit_script(frontend_dir)  # katex 뒤에 배치 → 타이포셋 후 실행
    lang_attr = f' lang="{lang}"' if lang else ""
    return (
        f'<!doctype html>\n<html{lang_attr}>\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escapeHtml(title)}</title>\n"
        f"<style>{_STANDALONE_CSS}</style>\n{katex}\n{fitter}\n</head>\n"
        f'<body><main class="doclayout-body"{lang_attr}>\n{body}\n</main></body>\n</html>\n'
    )


# ── standalone 문서 HTML (다운로드용 단일 파일 — 미리보기 뷰) ─────────────
# 레이아웃 좌표가 없는 figure_only 엔진(OvisOCR2·PaddleOCR-VL)의 layout.html은
# 빈 캔버스라 내보내기 구실을 못 한다 — 모든 엔진에서 동작하는 문서 뷰(/html과
# 동일 렌더) 내보내기가 이것. ⚠ SYNC: frontend/styles.css의 .markdown-body 규칙과
# 시각적으로 동기 유지할 것 (표 테두리·수식 중앙정렬·doc-page 구분선).
_DOCUMENT_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 40px 24px 60px; max-width: 860px; background: #fff;
  color: #1a1c22; font: 15px/1.7 system-ui, -apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
  overflow-wrap: break-word; }
h1, h2, h3, h4, h5, h6 { font-weight: 650; line-height: 1.3; margin: 1.4em 0 0.6em; }
h1 { font-size: 1.7em; padding-bottom: 0.3em; border-bottom: 1px solid #e3e5ea; }
h2 { font-size: 1.4em; padding-bottom: 0.25em; border-bottom: 1px solid #e3e5ea; }
h3 { font-size: 1.2em; }
p, ul, ol, blockquote, table, pre { margin: 0.8em 0; }
ul, ol { padding-left: 1.6em; }
a { color: #4650e5; }
img { max-width: 100%; height: auto; border: 1px solid #e3e5ea; border-radius: 4px; }
blockquote { padding: 0.4em 1em; border-left: 3px solid #d5d7dd; color: #626875; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em;
  background: #f0f1f4; padding: 0.15em 0.4em; border-radius: 4px; }
pre { background: #f0f1f4; padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; display: block; max-width: 100%; overflow-x: auto; }
th, td { border: 1px solid #d5d7dd; padding: 7px 11px; text-align: left; }
th { background: #f0f1f4; font-weight: 600; }
tr:nth-child(even) td { background: #f7f8fa; }
.math-display { display: block; text-align: center; margin: 1em 0; overflow-x: auto; }
.doc-page + .doc-page { border-top: 2px dashed #d5d7dd; margin-top: 28px; padding-top: 24px; }
main[lang="ko"] { word-break: keep-all; }
@media print { body { padding: 0; } }
"""

# 렌더가 files_base_url로 재작성한 이미지 참조. 파일명 캡처는 [^"/]+ — 슬래시 배제로
# `..%2F`류 트래버설 문자열은 매치 자체가 안 되고, 이름은 images/ 하위에서만 조회된다.
_DOC_IMG_SRC = re.compile(r'src="[^"]*/images/([^"/]+)"')


def render_document_standalone(
    inner_html: str, job_dir: Path, title: str, frontend_dir: Path | None,
    lang: str | None = None,
) -> str:
    """문서 뷰(/html과 동일 렌더 결과)를 완전 자립형 HTML로 — 이미지 base64·KaTeX
    인라인. inner_html은 신뢰 경로(render_document_html 출력)만 받는다."""
    body = _DOC_IMG_SRC.sub(
        lambda m: f'src="{_image_data_uri(job_dir, m.group(1))}"', inner_html,
    )
    katex = _katex_inline_bundle(str(frontend_dir)) if frontend_dir else ""
    lang_attr = f' lang="{lang}"' if lang else ""
    return (
        f'<!doctype html>\n<html{lang_attr}>\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escapeHtml(title)}</title>\n"
        f"<style>{_DOCUMENT_CSS}</style>\n{katex}\n</head>\n"
        f'<body><main{lang_attr}>\n{body}\n</main></body>\n</html>\n'
    )
