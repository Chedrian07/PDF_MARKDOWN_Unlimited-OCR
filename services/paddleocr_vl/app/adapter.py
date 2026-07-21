"""PaddleOCR-VL 공식 결과 → 내부 프로토콜 어댑터 (모델 없이 임포트 가능, 표준 라이브러리만).

공식 결과 스키마 (PaddleOCR 3.6.x `PaddleOCRVL.predict()` 결과의 `.json`,
fixture: tests/fixtures/official_page.json):

    {"res": {
        "input_path": …, "page_index": …,
        "layout_det_res": {"boxes": [{"cls_id", "label", "score", "coordinate"}]},
        "parsing_res_list": [
            {"block_bbox": [x1,y1,x2,y2],   # 입력 이미지 픽셀 좌표
             "block_label": "text|doc_title|paragraph_title|table|image|chart|formula|
                             seal|vision_footnote|footnote|number|aside_text|header|
                             header_image|footer|footer_image",
             "block_content": "…",           # 텍스트/markdown/표 HTML/수식 LaTeX
             "block_id": int, "block_order": int|None},  # block_order = 읽기 순서
        ],
        "markdown": {"text": …, "images": {…}},   # 사용하지 않음 — base64 이미지 배제
    }}

마크다운은 parsing_res_list에서 **결정적으로 재조립**한다 (base64 이미지·경로가
포함된 공식 markdown dict는 쓰지 않는다 — 프로토콜은 이미지 바이너리 금지).
공식 기본 동작과 동일하게 number/footnote/header(+image)/footer(+image)/aside_text는
markdown에서 제외하되 블록으로는 보존한다 (레이아웃 뷰 표시용).
"""

from __future__ import annotations

import json
import re

BBOX_MAX = 999
MAX_BLOCKS = 512
MAX_FIGURES = 64
MAX_CONTENT_CHARS = 100_000
_PROVIDER_RAW_CAP = 100_000

# 공식 라벨 → 프로토콜 정규화 타입
LABEL_MAP = {
    "doc_title": "title",
    "paragraph_title": "title",
    "text": "text",
    "abstract": "text",
    "content": "text",
    "aside_text": "text",
    # 캡션류(*_title)는 문서 제목이 아니라 그림/표/차트에 딸린 설명문이다 —
    # title로 매핑하면 마크다운에서 '## 캡션'으로 승격돼 문서 구조가 왜곡된다.
    # (figure_title은 2026-07-20 RTX 5070 Ti 실측 출력에서 확인)
    "figure_title": "text",
    "table_title": "text",
    "chart_title": "text",
    "reference": "text",
    # 실측 관측(2504.19874 25p 논문): 참고문헌 항목·수식 번호·알고리즘(의사코드) 라벨
    "reference_content": "text",
    "formula_number": "text",  # 수식 우측 '(3)' 번호 — 본문으로 보존
    "algorithm": "text",       # 알고리즘 박스(의사코드) — 본문으로 보존
    "table": "table",
    "formula": "formula",
    # 수식 라벨 변형 (display_formula는 2026-07-20 실측 출력에서 확인)
    "display_formula": "formula",
    "inline_formula": "formula",
    "equation": "formula",
    "image": "image",
    "chart": "image",
    "seal": "image",
    "figure": "image",
    "header": "header",
    "header_image": "header",
    "footer": "footer",
    "footer_image": "footer",
    "footnote": "footnote",
    "vision_footnote": "footnote",
    "number": "page_number",
}

# 공식 파이프라인의 markdown 제외 기본값 (blocks에는 보존)
MARKDOWN_EXCLUDED_LABELS = frozenset({
    "number", "footnote", "header", "header_image",
    "footer", "footer_image", "aside_text",
})

# 참고: PaddleOCR-VL의 라벨 어휘는 공식 문서 목록보다 넓다 — 실측으로 확인한
# 라벨만 위에 매핑하고, 그 외는 "unknown"으로 두되 **content는 markdown에 그대로
# 보존**된다(_markdown_fragment). 즉 미지 라벨은 내용 손실이 아니라 경고 로그일 뿐이다.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MATH_DELIM_RE = re.compile(r"(\$|\\\[|\\\()")


def _norm_coord(v: float, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(BBOX_MAX, round(v / size * BBOX_MAX)))


def _norm_bbox(
    bbox: object, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """픽셀 bbox → [0,999] 정규화. 형식/범위 위반은 None."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(-1e7 < v < 1e7 for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    # 경미한 초과(2% + 2px)만 clamp 허용 — 그 밖은 심각한 이상으로 폐기
    mx, my = width * 0.02 + 2, height * 0.02 + 2
    if x1 < -mx or y1 < -my or x2 > width + mx or y2 > height + my:
        return None
    nx1, ny1 = _norm_coord(x1, width), _norm_coord(y1, height)
    nx2, ny2 = _norm_coord(x2, width), _norm_coord(y2, height)
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return None
    return (nx1, ny1, nx2, ny2)


def _clean_text(s: object) -> str:
    if not isinstance(s, str):
        return ""
    return _CONTROL_RE.sub("", s)[:MAX_CONTENT_CHARS]


def _markdown_fragment(
    label: str, btype: str, content: str, figure_ref: str | None
) -> str | None:
    """블록 1개의 마크다운 조각 (None = 마크다운에서 제외).

    제외 여부는 **원시 라벨**(공식 파이프라인 기본값과 동일한 어휘)로, 서식 결정은
    **정규화 타입**으로 한다 — display_formula처럼 라벨 변형이 와도 수식 처리가
    동일하게 적용된다(원시 라벨로 분기하면 변형마다 조용히 누락된다).
    """
    if label in MARKDOWN_EXCLUDED_LABELS:
        return None
    if figure_ref is not None:
        return figure_ref
    if not content.strip():
        return None
    if label == "doc_title":
        return f"# {content.strip()}"
    if label == "paragraph_title":
        return f"## {content.strip()}"
    if btype == "formula":
        body = content.strip()
        # 이미 수식 구분자가 있으면 그대로, 없으면 display math로 감싼다
        if _MATH_DELIM_RE.search(body):
            return body
        return f"\\[ {body} \\]"
    # text / table(HTML) / 기타는 내용 그대로
    return content.strip()


def adapt_page(page_json: dict, width: int, height: int) -> dict:
    """공식 결과 JSON 1페이지 → 프로토콜 page dict (markdown/blocks/warnings).

    한국어 텍스트(한글 음절·자모·한자·영문 혼용)는 어떤 변환도 없이 그대로
    보존된다 — 이 함수는 재배치만 하고 문자열을 변형하지 않는다 (제어 문자
    제거 제외).
    """
    warnings: list[str] = []
    res = page_json.get("res", page_json)
    if not isinstance(res, dict):
        raise ValueError("공식 결과 JSON 형식이 아닙니다 (res 객체 없음)")
    blocks_in = res.get("parsing_res_list")
    if not isinstance(blocks_in, list):
        raise ValueError("공식 결과 JSON 형식이 아닙니다 (parsing_res_list 없음)")

    if len(blocks_in) > MAX_BLOCKS:
        warnings.append(f"블록 수({len(blocks_in)})가 상한({MAX_BLOCKS})을 초과해 절단됨")
        blocks_in = blocks_in[:MAX_BLOCKS]

    # 읽기 순서: block_order(정수) 우선, 없으면 목록 순서 유지 (안정 정렬)
    def _order_key(item: tuple[int, dict]) -> tuple[int, int]:
        i, b = item
        order = b.get("block_order")
        if isinstance(order, int) and not isinstance(order, bool):
            return (order, i)
        return (i, i)

    ordered = sorted(enumerate(blocks_in), key=_order_key)

    out_blocks: list[dict] = []
    md_parts: list[str] = []
    figure_count = 0
    for order, (_, raw_block) in enumerate(ordered):
        if not isinstance(raw_block, dict):
            warnings.append("블록 항목이 객체가 아님 — 건너뜀")
            continue
        label = str(raw_block.get("block_label", "")).strip().lower()
        btype = LABEL_MAP.get(label, "unknown")
        if label and btype == "unknown":
            warnings.append(f"알 수 없는 블록 라벨 '{label[:40]}' — unknown으로 처리")
        content = _clean_text(raw_block.get("block_content"))
        bbox = _norm_bbox(raw_block.get("block_bbox"), width, height)

        figure_ref: str | None = None
        figure_index: int | None = None
        if btype == "image":
            if bbox is None:
                warnings.append(f"image 블록(order={order})의 bbox 이상 — 폐기")
                continue
            if figure_count >= MAX_FIGURES:
                warnings.append(f"figure 수 상한({MAX_FIGURES}) 초과 — 이후 폐기")
                continue
            figure_index = figure_count
            figure_count += 1
            if label not in MARKDOWN_EXCLUDED_LABELS:
                figure_ref = f"[[FIGURE:{figure_index}]]"
        elif bbox is None and raw_block.get("block_bbox") is not None:
            warnings.append(f"블록(order={order}, type={btype})의 bbox 이상 — 좌표 제외")

        out_blocks.append({
            "type": btype,
            "bbox": list(bbox) if bbox is not None else None,
            "content": "" if btype == "image" else content,
            "order": order,
            "figure_index": figure_index,
            "confidence": None,
        })
        frag = _markdown_fragment(label, btype, content, figure_ref)
        if frag is not None:
            md_parts.append(frag)

    provider_raw = json.dumps(
        res.get("parsing_res_list", []), ensure_ascii=False, default=str
    )[:_PROVIDER_RAW_CAP]

    return {
        "markdown": "\n\n".join(md_parts).strip(),
        "blocks": out_blocks,
        "warnings": warnings,
        "provider_raw": provider_raw,
    }
