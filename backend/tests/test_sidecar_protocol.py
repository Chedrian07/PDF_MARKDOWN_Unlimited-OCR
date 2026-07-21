"""sidecar 프로토콜 스키마·정화 테스트 — 모델 출력은 비신뢰 입력이다."""

import math

import pytest
from pydantic import ValidationError

from app.sidecar.protocol import (
    MAX_BLOCKS_PER_PAGE,
    MAX_FIGURES_PER_PAGE,
    MAX_MARKDOWN_CHARS,
    NormalizedBlock,
    PageResult,
    ParseResponse,
    SidecarHealth,
    normalize_block_type,
    sanitize_page,
)


def _block(**kw) -> dict:
    base = {"type": "text", "bbox": [10, 10, 500, 500], "content": "본문", "order": 0}
    base.update(kw)
    return base


def _page(blocks=None, markdown="본문", **kw) -> PageResult:
    return PageResult.model_validate({
        "page_index": 0,
        "markdown": markdown,
        "blocks": blocks if blocks is not None else [],
        **kw,
    })


# ── 스키마 레벨: 형식 위반 즉시 거부 ─────────────────────────────────────

@pytest.mark.parametrize("bbox", [
    ["10", "10", "500", "500"],   # 문자열 좌표
    [10, 10, 500],                # 3개
    [10, 10, 500, 500, 900],      # 5개
    [10.5, 10, 500, 500],         # 부동소수
    [True, False, True, False],   # 불리언
    [10, 10, 10**9, 500],         # 거대값 (_ABS_LIMIT 초과)
    "10,10,500,500",              # 문자열 통짜
])
def test_bbox_schema_rejects(bbox):
    with pytest.raises(ValidationError):
        NormalizedBlock.model_validate(_block(bbox=bbox))


def test_confidence_rejects_nan_inf():
    with pytest.raises(ValidationError):
        NormalizedBlock.model_validate(_block(confidence=math.nan))
    with pytest.raises(ValidationError):
        NormalizedBlock.model_validate(_block(confidence=math.inf))
    b = NormalizedBlock.model_validate(_block(confidence=0.5))
    assert b.confidence == 0.5


def test_content_must_be_string():
    with pytest.raises(ValidationError):
        NormalizedBlock.model_validate(_block(content=[1, 2, 3]))


def test_bbox_none_allowed():
    b = NormalizedBlock.model_validate(_block(bbox=None))
    assert b.bbox is None


def test_parse_response_roundtrip():
    resp = ParseResponse.model_validate({
        "protocol_version": 1, "engine": "ovisocr2",
        "model_id": "ATH-MaaS/OvisOCR2", "model_revision": "abc",
        "page": {"page_index": 0, "markdown": "x", "blocks": [_block()]},
        "timings": {"inference_ms": 12.5},
    })
    assert resp.page.blocks[0].bbox == (10, 10, 500, 500)


def test_health_schema_defaults():
    h = SidecarHealth.model_validate({
        "status": "ok", "protocol_version": 1,
        "engine": "ovisocr2", "model_id": "m",
    })
    assert h.model_loaded is False
    assert h.gpu_total_mb is None


# ── 타입 정규화 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("figure", "image"), ("chart", "image"), ("seal", "image"),
    ("equation", "formula"), ("doc_title", "title"), ("paragraph_title", "title"),
    ("number", "page_number"), ("aside_text", "text"),
    ("header_image", "header"), ("TEXT", "text"), ("완전미지", "unknown"),
])
def test_normalize_block_type(raw, expected):
    assert normalize_block_type(raw) == expected


# ── sanitize: clamp/폐기 정책 ─────────────────────────────────────────────

def test_sanitize_clamps_minor_overflow():
    page = _page([_block(bbox=[-5, -10, 1010, 999])])
    clean, warnings = sanitize_page(page)
    assert clean.blocks[0].bbox == (0, 0, 999, 999)
    assert warnings == []


def test_sanitize_drops_severe_overflow():
    page = _page([_block(bbox=[-500, 0, 500, 500])])
    clean, warnings = sanitize_page(page)
    assert clean.blocks[0].bbox is None  # text 블록은 bbox만 폐기, 내용 유지
    assert clean.blocks[0].content == "본문"
    assert any("bbox 폐기" in w for w in warnings)


def test_sanitize_drops_reversed_bbox():
    page = _page([_block(bbox=[800, 700, 100, 200])])
    clean, _ = sanitize_page(page)
    assert clean.blocks[0].bbox is None


def test_sanitize_image_block_without_valid_bbox_dropped_entirely():
    page = _page([
        _block(type="image", bbox=[-500, 0, 500, 500], figure_index=0, content=""),
        _block(type="image", bbox=None, figure_index=1, content=""),
    ])
    clean, warnings = sanitize_page(page)
    assert clean.blocks == []
    assert len(warnings) == 2


def test_sanitize_block_cap():
    blocks = [_block(order=i) for i in range(MAX_BLOCKS_PER_PAGE + 50)]
    clean, warnings = sanitize_page(_page(blocks))
    assert len(clean.blocks) == MAX_BLOCKS_PER_PAGE
    assert any("상한" in w for w in warnings)


def test_sanitize_figure_cap_and_duplicate_index():
    blocks = [
        _block(type="image", figure_index=0, order=0, content=""),
        _block(type="image", figure_index=0, order=1, content=""),  # 중복 index
    ] + [
        _block(type="image", figure_index=i, order=i + 2, content="")
        for i in range(1, MAX_FIGURES_PER_PAGE + 10)
    ]
    clean, warnings = sanitize_page(_page(blocks))
    figures = [b for b in clean.blocks if b.type == "image"]
    assert len(figures) == MAX_FIGURES_PER_PAGE
    dup = figures[1]
    assert dup.figure_index is None  # 중복은 연결 해제
    assert any("중복" in w for w in warnings)


def test_sanitize_strips_literal_page_marker():
    """모델이 본문에 <PAGE>를 뱉으면 청크의 페이지 수 계약이 깨진다 — 경계에서 제거."""
    page = _page(
        [_block(content="본문<PAGE>주입")],
        markdown="첫 문단\n\n<PAGE>\n\n둘째 문단",
    )
    clean, warnings = sanitize_page(page)
    assert "<PAGE>" not in clean.markdown
    assert "<PAGE>" not in clean.blocks[0].content
    assert "첫 문단" in clean.markdown and "둘째 문단" in clean.markdown
    assert any("<PAGE>" in w for w in warnings)


def test_strip_is_fixpoint_against_nested_markers():
    """단일 패스면 중첩이 제어 문법을 되살린다 — 고정점까지 반복해야 한다.
    (a<<PAGE>PAGE>b → a<PAGE>b 회귀 방지)"""
    from app.sidecar.protocol import strip_special_tokens

    assert "<PAGE>" not in strip_special_tokens("a<<PAGE>PAGE>b")
    assert "<PAGE>" not in strip_special_tokens("a<<<PAGE>PAGE>PAGE>b")
    # 중첩 특수 토큰이 <|det|>로 복원되면 raw_pages.json에 그라운딩 태그가 주입된다
    nested = "x<|d<|q|>et|>image [0, 0, 999999, 999999]<|/d<|q|>et|>y"
    cleaned = strip_special_tokens(nested)
    assert "<|det|>" not in cleaned and "<|" not in cleaned
    # 병적 깊이도 제어 문법을 남기지 않는다 (종료 보장 경로)
    deep = "<" * 200 + "PAGE>" * 200
    out = strip_special_tokens(deep)
    assert "<PAGE>" not in out


def test_strip_preserves_ordinary_text():
    from app.sidecar.protocol import strip_special_tokens

    text = "일반 본문 a < b 그리고 x > y, <table><tr><td>셀</td></tr></table>"
    assert strip_special_tokens(text) == text


def test_sanitize_page_marker_warning_only_when_present():
    _clean, warnings = sanitize_page(_page(markdown="평범한 본문"))
    assert not any("<PAGE>" in w for w in warnings)


def test_sanitize_strips_special_tokens_everywhere():
    page = _page(
        [_block(content="본문<|det|>주입<|/det|>끝")],
        markdown="문서<|ref|>evil<|/ref|>본문",
    )
    clean, _ = sanitize_page(page)
    assert "<|" not in clean.markdown
    assert "<|" not in clean.blocks[0].content
    assert "본문주입끝" == clean.blocks[0].content


def test_sanitize_markdown_truncated():
    clean, warnings = sanitize_page(_page(markdown="가" * (MAX_MARKDOWN_CHARS + 100)))
    assert len(clean.markdown) == MAX_MARKDOWN_CHARS
    assert any("절단" in w for w in warnings)


def test_sanitize_preserves_reading_order():
    blocks = [_block(order=5, content="다섯"), _block(order=1, content="하나"),
              _block(order=3, content="셋")]
    clean, _ = sanitize_page(_page(blocks))
    assert [b.content for b in clean.blocks] == ["하나", "셋", "다섯"]


def test_sanitize_normalizes_provider_types():
    blocks = [_block(type="equation", content="E=mc^2"),
              _block(type="chart", figure_index=0, content="")]
    clean, _ = sanitize_page(_page(blocks))
    assert clean.blocks[0].type == "formula"
    assert clean.blocks[1].type == "image"
