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

from markdown_it.common.utils import escapeHtml

from .render import _restore_table_tags

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


def render_layout_html(pages: list[dict], files_base_url: str) -> str:
    """layout.json(merge가 통합한 페이지 블록들) → 절대 배치 HTML 프래그먼트."""
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
                blocks_html.append(
                    f'<img class="layout-block layout-image" '
                    f'src="{files_base_url}/images/{image_name}" style="{style}" alt="">'
                )
                continue
            content = escapeHtml(str(b.get("content") or ""))
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
