"""sidecar 내부 프로토콜 v1 — 스키마 검증과 비신뢰 입력 정화.

계약 문서: docs/OCR_ENGINE_PROTOCOL.md. sidecar(services/*)에도 같은 모델의
사본이 있다(독립 배포 단위라 임포트 공유 불가) — 스키마를 바꾸면 양쪽과 문서를
함께 갱신하고 protocol_version을 올린다.

모델 출력은 공격자가 문서를 통해 조작할 수 있는 **비신뢰 입력**이다:
- bbox는 [0,999] 정규화 정수만. 경미한 초과(_CLAMP_MARGIN)만 clamp하고
  심각한 이상(음수·거대값·좌표 역전·비정수)은 블록을 폐기하고 warning을 남긴다.
- 문자열 안의 `<|…|>` 특수 토큰 패턴은 제거한다 (그라운딩 문법 합성 오염 방지).
- 블록 수·문자열 길이·figure 수에 상한을 둔다 (response bomb 방어).
"""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL_VERSION = 1

BBOX_MAX = 999          # 정규화 좌표 상한 (0–999, 벤더/레이아웃 파서와 동일)
_CLAMP_MARGIN = 25      # 이 이내의 범위 초과만 "합리적 오차"로 보고 clamp (2.5%)
_ABS_LIMIT = 10_000_000  # 이 이상의 정수는 무조건 거부 (오버플로/폭주 방어)

MAX_BLOCKS_PER_PAGE = 512
MAX_FIGURES_PER_PAGE = 64
MAX_CONTENT_CHARS = 100_000
MAX_MARKDOWN_CHARS = 400_000
MAX_WARNINGS = 32
MIN_BBOX_SIDE = 2       # 정규화 좌표 기준 최소 변 (이보다 작으면 퇴화 bbox로 폐기)

# 앱이 생성하는 제한된 figure placeholder 형식 — 이 외의 경로/태그는 파일이 되지 않는다
FIGURE_PLACEHOLDER_RE = re.compile(r"\[\[FIGURE:(\d{1,3})\]\]")

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]{0,64}\|>")
# 파이프라인의 페이지 구분자 — 모델이 본문에 리터럴로 뱉으면 청크의 페이지 수 계약이
# 깨진다(merge가 초과 마커를 마지막 페이지에 병합 → 내용 이동). sidecar 경계에서 제거.
_PAGE_MARKER_RE = re.compile(r"<PAGE>")

# 정규화 블록 타입 (레이아웃 파서·프론트가 아는 어휘)
CANONICAL_TYPES = frozenset({
    "title", "text", "table", "formula", "image",
    "header", "footer", "footnote", "page_number", "unknown",
})
# 프로바이더별 라벨 → 정규화 타입 (image/figure는 하나의 안정 타입 "image"로 통일)
TYPE_ALIASES = {
    "figure": "image", "picture": "image", "chart": "image", "seal": "image",
    "equation": "formula",
    "doc_title": "title", "paragraph_title": "title",
    "vision_footnote": "footnote",
    "number": "page_number",
    "aside_text": "text", "content": "text", "paragraph": "text",
    "header_image": "header", "footer_image": "footer",
}


_STRIP_ROUNDS = 8  # 정상 문서는 1~2회에 고정점 도달 (2회차에 변화 없음 → 조기 종료)


def strip_special_tokens(text: str) -> str:
    """`<|…|>` 특수 토큰과 `<PAGE>` 페이지 구분자 제거.

    둘 다 파이프라인이 소유한 제어 문법이라 모델 본문에 실려 오면 안 된다
    (특수 토큰은 layout 파서를, `<PAGE>`는 청크의 페이지 수 계약을 오염시킨다).

    **단일 패스로는 부족하다**: `re.sub`는 치환 결과를 재스캔하지 않으므로
    `a<<PAGE>PAGE>b` → `a<PAGE>b`, `x<|d<|q|>et|>y` → `x<|det|>y`처럼 중첩
    입력이 제어 문법을 **되살린다**(실측 확인). 그래서 변화가 없을 때까지 반복하고,
    상한 안에서 수렴하지 않는 병적 입력은 `<`를 제거해 두 패턴의 생성 자체를
    불가능하게 만든다 — 종료가 보장되고(문자열이 매 회 짧아짐) 우회로가 없다.
    """
    out = text
    for _ in range(_STRIP_ROUNDS):
        stripped = _PAGE_MARKER_RE.sub("", _SPECIAL_TOKEN_RE.sub("", out))
        if stripped == out:
            return out
        out = stripped
    if _PAGE_MARKER_RE.search(out) or _SPECIAL_TOKEN_RE.search(out):
        # 비정상적으로 깊은 중첩 — 정상 모델 출력에서는 도달하지 않는 경로
        out = out.replace("<", " ")
    return out


def normalize_block_type(raw: str) -> str:
    t = raw.strip().lower()
    if t in CANONICAL_TYPES:
        return t
    return TYPE_ALIASES.get(t, "unknown")


def _valid_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and abs(v) < _ABS_LIMIT


class NormalizedBlock(BaseModel):
    """모델 독립 페이지 블록. bbox는 [0,999] 정규화 (없으면 None — figure_only 엔진의 텍스트)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    type: str
    bbox: tuple[int, int, int, int] | None = None
    content: str = ""
    order: int = 0
    figure_index: int | None = None
    confidence: float | None = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _check_bbox(cls, v: object) -> object:
        if v is None:
            return None
        # 정확히 4개의 진짜 정수만 — 문자열/불리언/NaN/무한대/거대값 거부
        if not isinstance(v, (list, tuple)) or len(v) != 4:
            raise ValueError("bbox는 정수 4개여야 합니다")
        if not all(_valid_int(x) for x in v):
            raise ValueError("bbox 좌표는 유한한 정수여야 합니다")
        return tuple(v)

    @field_validator("content", mode="before")
    @classmethod
    def _check_content(cls, v: object) -> object:
        if not isinstance(v, str):
            raise ValueError("content는 문자열이어야 합니다")
        return v

    @field_validator("order", "figure_index", mode="before")
    @classmethod
    def _check_ints(cls, v: object) -> object:
        if v is None:
            return None
        if not _valid_int(v):
            raise ValueError("정수여야 합니다")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _check_conf(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("confidence는 유한한 수여야 합니다")
        return float(v)


class PageResult(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    page_index: int
    markdown: str
    blocks: list[NormalizedBlock] = Field(default_factory=list)
    provider_raw: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ParseResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    protocol_version: int
    engine: str
    model_id: str
    model_revision: str = ""
    page: PageResult
    timings: dict[str, float] = Field(default_factory=dict)


class SidecarHealth(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    status: str
    protocol_version: int
    engine: str
    model_id: str
    model_revision: str = ""
    runtime: str = ""
    runtime_version: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    gpu_name: str | None = None
    gpu_total_mb: int | None = None
    gpu_free_mb: int | None = None
    model_loaded: bool = False
    load_error: str | None = None   # sidecar 자체 로드 실패 사유 (CUDA 가드 트립 등) — 대기 무의미


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """경미한 범위 초과만 clamp. 심각한 이상은 None(폐기)."""
    x1, y1, x2, y2 = bbox
    lo, hi = -_CLAMP_MARGIN, BBOX_MAX + _CLAMP_MARGIN
    if not all(lo <= v <= hi for v in bbox):
        return None
    x1 = min(max(x1, 0), BBOX_MAX)
    y1 = min(max(y1, 0), BBOX_MAX)
    x2 = min(max(x2, 0), BBOX_MAX)
    y2 = min(max(y2, 0), BBOX_MAX)
    if x2 - x1 < MIN_BBOX_SIDE or y2 - y1 < MIN_BBOX_SIDE:
        return None
    return (x1, y1, x2, y2)


def sanitize_page(page: PageResult) -> tuple[PageResult, list[str]]:
    """스키마를 통과한 페이지를 파이프라인에 넣기 전 최종 정화.

    반환: (정화된 PageResult, 경고 목록). 원본은 수정하지 않는다.
    - markdown/content 길이 상한 + 특수 토큰 제거
    - 블록 수 상한, 타입 정규화, bbox clamp/폐기, order 안정 정렬
    - image 블록의 figure_index 중복/상한 검증 (중복·초과는 폐기)
    """
    warnings: list[str] = []

    markdown = strip_special_tokens(page.markdown)
    if _PAGE_MARKER_RE.search(page.markdown) or _SPECIAL_TOKEN_RE.search(page.markdown):
        warnings.append("모델 출력의 제어 문법(<PAGE>·특수 토큰)을 제거함 (페이지·레이아웃 계약 보호)")
    if len(markdown) > MAX_MARKDOWN_CHARS:
        warnings.append(f"페이지 markdown이 상한({MAX_MARKDOWN_CHARS}자)을 초과해 절단됨")
        markdown = markdown[:MAX_MARKDOWN_CHARS]

    blocks = list(page.blocks)
    if len(blocks) > MAX_BLOCKS_PER_PAGE:
        warnings.append(
            f"블록 수({len(blocks)})가 상한({MAX_BLOCKS_PER_PAGE})을 초과해 절단됨"
        )
        blocks = blocks[:MAX_BLOCKS_PER_PAGE]

    cleaned: list[NormalizedBlock] = []
    seen_figures: set[int] = set()
    figure_count = 0
    for b in sorted(blocks, key=lambda b: b.order):  # 읽기 순서 보존 (안정 정렬)
        btype = normalize_block_type(b.type)
        bbox = b.bbox
        if bbox is not None:
            bbox = _clamp_bbox(bbox)
            if bbox is None:
                warnings.append(f"블록(type={btype}, order={b.order}): 비정상 bbox 폐기")
                if btype == "image":
                    continue  # bbox 없는 figure는 crop 불가 — 블록째 폐기
        content = strip_special_tokens(b.content)
        if len(content) > MAX_CONTENT_CHARS:
            warnings.append(f"블록(order={b.order}) content 상한 초과로 절단")
            content = content[:MAX_CONTENT_CHARS]
        figure_index = b.figure_index
        if btype == "image":
            if bbox is None:
                warnings.append(f"image 블록(order={b.order})에 bbox 없음 — 폐기")
                continue
            figure_count += 1
            if figure_count > MAX_FIGURES_PER_PAGE:
                warnings.append(f"figure 수 상한({MAX_FIGURES_PER_PAGE}) 초과 — 이후 폐기")
                continue
            if figure_index is not None:
                if figure_index < 0 or figure_index in seen_figures:
                    warnings.append(f"figure_index {figure_index} 중복/이상 — 연결 해제")
                    figure_index = None
                else:
                    seen_figures.add(figure_index)
        else:
            figure_index = None
        cleaned.append(NormalizedBlock(
            type=btype, bbox=bbox, content=content,
            order=b.order, figure_index=figure_index, confidence=b.confidence,
        ))

    page_warnings = [str(w)[:500] for w in page.warnings[:MAX_WARNINGS]]
    if len(page.warnings) > MAX_WARNINGS:
        page_warnings.append("…경고 상한 초과로 생략")

    result = PageResult(
        page_index=page.page_index,
        markdown=markdown,
        blocks=cleaned,
        provider_raw=page.provider_raw,
        warnings=page_warnings + warnings,
    )
    return result, warnings
