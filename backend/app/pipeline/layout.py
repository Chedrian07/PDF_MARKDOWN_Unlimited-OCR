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


def render_layout_html(pages: list[dict], files_base_url: str, image_src=None) -> str:
    """layout.json(merge가 통합한 페이지 블록들) → 절대 배치 HTML 프래그먼트.
    image_src(name)->str 을 주면 이미지 src를 그 값으로 (standalone의 data URI용)."""
    sections: list[str] = []
    for p in pages:
        width = p.get("width") or 1000
        height = p.get("height") or 1414
        aspect_pct = height / width * 100 if width else 141.4
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
            blocks_html.append(
                f'<div class="layout-block layout-{btype}" style="{style}" '
                f'title="{btype}">{content}</div>'
            )
        sections.append(
            f'<section class="layout-page" data-page="{int(p.get("page", 0))}">'
            f'<div class="layout-canvas" style="padding-top:{aspect_pct:.2f}%">'
            + "".join(blocks_html)
            + "</div></section>"
        )
    return "\n".join(sections)


# ── standalone HTML (다운로드용 단일 파일 — PDF 대응 뷰) ─────────────────
# frontend/styles.css의 .layout-* 규칙과 시각적으로 동기 유지할 것.
_STANDALONE_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; margin: 0; }
body { background: #eceef2; font-family: system-ui, -apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif; padding: 24px 12px; }
.doclayout-body { display: grid; gap: 26px; max-width: 900px; margin-inline: auto; }
.layout-page::before { content: "페이지 " attr(data-page); display: block; font-size: 11px; color: #666; margin-bottom: 4px; }
.layout-canvas { position: relative; width: 100%; height: 0; background: #fff; border: 1px solid #d5d7dd; border-radius: 6px; box-shadow: 0 1px 5px rgba(0,0,0,.1); overflow: hidden; }
.layout-block { position: absolute; overflow: hidden; font-size: 11px; line-height: 1.35; color: #1a1c22; padding: 1px 3px; }
.layout-title { font-weight: 700; font-size: 14px; color: #b0262c; }
.layout-image { object-fit: contain; padding: 0; }
.layout-table { font-size: 9px; }
.layout-table table { border-collapse: collapse; width: 100%; }
.layout-table td, .layout-table th { border: 1px solid #c9c9d4; padding: 1px 3px; }
.layout-formula, .layout-equation { font-size: 11px; display: flex; align-items: center; justify-content: center; }
.layout-page_number, .layout-header, .layout-footer, .layout-footnote { opacity: .45; font-size: 9px; }
.layout-block .math-display { display: block; text-align: center; }
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


def render_layout_standalone(
    pages: list[dict], job_dir: Path, title: str, frontend_dir: Path | None
) -> str:
    """이미지 base64·KaTeX 인라인의 완전 자립형 HTML 문서 — 오프라인에서 그대로 열림."""

    def _inline_image(name: str) -> str:
        p = job_dir / "images" / name
        try:
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except OSError:
            return "data:,"  # 결측 크롭 — 빈 이미지로 폴백

    body = render_layout_html(pages, files_base_url="", image_src=_inline_image)
    katex = _katex_inline_bundle(str(frontend_dir)) if frontend_dir else ""
    return (
        '<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escapeHtml(title)}</title>\n"
        f"<style>{_STANDALONE_CSS}</style>\n{katex}\n</head>\n"
        f'<body><main class="doclayout-body">\n{body}\n</main></body>\n</html>\n'
    )
