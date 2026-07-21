"""PaddleOCR-VL 어댑터 테스트 — 모델/paddle 없이 실행 (표준 라이브러리만).

fixture는 공식 문서(PaddleOCR 3.6.x PaddleOCRVL 결과 스키마)에 기록된
필드명(parsing_res_list[].block_bbox/block_label/block_content/block_order)을
그대로 재현한 것이다 — 실 GPU 환경에서 스키마 드리프트를 발견하면 fixture를
실측 결과로 교체하고 어댑터를 함께 갱신한다.
"""

import json
from pathlib import Path

import pytest

from app.adapter import BBOX_MAX, MAX_FIGURES, adapt_page

FIXTURES = Path(__file__).parent / "fixtures"
PAGE_W, PAGE_H = 1240, 1754


@pytest.fixture
def official() -> dict:
    return json.loads((FIXTURES / "official_page.json").read_text(encoding="utf-8"))


def test_markdown_assembly_order_and_content(official):
    page = adapt_page(official, PAGE_W, PAGE_H)
    md = page["markdown"]
    # 제목 → 본문 → 표 → 수식 → figure 순서 (block_order 기준, header=order 0은 제외)
    assert md.startswith("# 2026년 상반기 연구 보고서 (硏究報告書)")
    assert md.index("# 2026년") < md.index("본 문서는 한국어와") < md.index("<table>")
    assert md.index("<table>") < md.index("\\sigma") < md.index("[[FIGURE:0]]")
    assert md.index("[[FIGURE:0]]") < md.index("[[FIGURE:1]]")


def test_korean_text_preserved_exactly(official):
    """한글 음절·자모·한자·영문·숫자와 단위가 무변형 보존."""
    page = adapt_page(official, PAGE_W, PAGE_H)
    assert "한국어와 English가 혼용된 문단입니다" in page["markdown"]
    assert "ㄱㄴㄷ" in page["markdown"]
    assert "1,234.56 kg" in page["markdown"]
    assert "漢字" in page["markdown"]
    assert "硏究報告書" in page["markdown"]


def test_table_html_and_cell_newline_preserved(official):
    page = adapt_page(official, PAGE_W, PAGE_H)
    assert "<table><tr><th>항목</th>" in page["markdown"]
    assert "매출\n(분기)" in page["markdown"]  # 셀 내부 줄바꿈 보존


def test_formula_wrapped_as_display_math(official):
    page = adapt_page(official, PAGE_W, PAGE_H)
    assert "\\[ \\sigma = \\sqrt" in page["markdown"]


def test_formula_label_variants_wrapped_identically():
    """display_formula 같은 라벨 변형도 수식으로 처리된다 (실측 관측 라벨 — 회귀 방지).
    서식 분기가 원시 라벨에 묶여 있으면 변형마다 조용히 누락된다."""
    for label in ("formula", "display_formula", "inline_formula", "equation"):
        data = {"res": {"parsing_res_list": [{
            "block_bbox": [0, 0, 500, 100], "block_label": label,
            "block_content": "a^2 + b^2 = c^2", "block_id": 0, "block_order": 0,
        }]}}
        page = adapt_page(data, 1000, 1000)
        assert page["blocks"][0]["type"] == "formula", label
        assert page["markdown"] == "\\[ a^2 + b^2 = c^2 \\]", label
        assert page["warnings"] == [], label


def test_formula_with_existing_delimiters_not_double_wrapped():
    data = {"res": {"parsing_res_list": [{
        "block_bbox": [0, 0, 500, 100], "block_label": "formula",
        "block_content": "\\( x^2 \\)", "block_id": 0, "block_order": 0,
    }]}}
    page = adapt_page(data, 1000, 1000)
    assert page["markdown"] == "\\( x^2 \\)"


def test_excluded_labels_not_in_markdown_but_in_blocks(official):
    page = adapt_page(official, PAGE_W, PAGE_H)
    assert "각주 내용은" not in page["markdown"]
    assert "기밀" not in page["markdown"]
    types = [b["type"] for b in page["blocks"]]
    assert "footnote" in types and "header" in types and "page_number" in types


def test_bbox_normalized_to_0_999(official):
    page = adapt_page(official, PAGE_W, PAGE_H)
    for b in page["blocks"]:
        assert b["bbox"] is not None
        assert all(0 <= v <= BBOX_MAX for v in b["bbox"])
        assert b["bbox"][0] < b["bbox"][2] and b["bbox"][1] < b["bbox"][3]
    title = next(b for b in page["blocks"] if b["type"] == "title")
    assert title["bbox"] == [round(90.5 / PAGE_W * 999), round(60.2 / PAGE_H * 999),
                             round(1150.0 / PAGE_W * 999), round(130.8 / PAGE_H * 999)]


def test_reading_order_header_first(official):
    """block_order가 읽기 순서 — header(order 0)가 blocks의 맨 앞."""
    page = adapt_page(official, PAGE_W, PAGE_H)
    assert page["blocks"][0]["type"] == "header"
    orders = [b["order"] for b in page["blocks"]]
    assert orders == sorted(orders)


def test_figures_get_sequential_indices(official):
    page = adapt_page(official, PAGE_W, PAGE_H)
    figures = [b for b in page["blocks"] if b["type"] == "image"]
    assert [f["figure_index"] for f in figures] == [0, 1]
    assert all(f["content"] == "" for f in figures)


def test_provider_raw_has_no_base64_images(official):
    page = adapt_page(official, PAGE_W, PAGE_H)
    assert "base64" not in page["provider_raw"]
    assert "공식 markdown dict" not in page["provider_raw"]  # markdown dict 미사용


# ── 방어적 처리 ────────────────────────────────────────────────────────────

def test_missing_parsing_res_list_raises():
    with pytest.raises(ValueError, match="parsing_res_list"):
        adapt_page({"res": {"markdown": {}}}, 100, 100)


def test_non_dict_res_raises():
    with pytest.raises(ValueError):
        adapt_page({"res": [1, 2, 3]}, 100, 100)


def test_image_block_with_bad_bbox_dropped():
    data = {"res": {"parsing_res_list": [
        {"block_bbox": [500, 500, 100, 100], "block_label": "image",
         "block_content": "", "block_id": 0, "block_order": 0},
        {"block_bbox": "evil", "block_label": "chart",
         "block_content": "", "block_id": 1, "block_order": 1},
        {"block_bbox": [0, 0, 1e12, 1e12], "block_label": "image",
         "block_content": "", "block_id": 2, "block_order": 2},
    ]}}
    page = adapt_page(data, 1000, 1000)
    assert page["blocks"] == []
    assert "[[FIGURE" not in page["markdown"]
    assert len(page["warnings"]) == 3


def test_text_block_with_bad_bbox_kept_without_coords():
    data = {"res": {"parsing_res_list": [{
        "block_bbox": [-5000, 0, 100, 100], "block_label": "text",
        "block_content": "내용은 살아남는다", "block_id": 0, "block_order": 0,
    }]}}
    page = adapt_page(data, 1000, 1000)
    assert page["blocks"][0]["bbox"] is None
    assert "내용은 살아남는다" in page["markdown"]


def test_caption_labels_are_text_not_headings():
    """*_title 캡션은 문서 제목이 아니다 — '## '로 승격되면 구조가 왜곡된다.
    (figure_title은 실 GPU 실행에서 관측된 라벨 — 회귀 방지)"""
    data = {"res": {"parsing_res_list": [
        {"block_bbox": [0, 0, 500, 60], "block_label": "figure_title",
         "block_content": "그림 1. 분기별 매출", "block_id": 0, "block_order": 0},
        {"block_bbox": [0, 100, 500, 160], "block_label": "table_title",
         "block_content": "표 1. 모델 구성", "block_id": 1, "block_order": 1},
    ]}}
    page = adapt_page(data, 1000, 1000)
    assert [b["type"] for b in page["blocks"]] == ["text", "text"]
    assert not page["markdown"].startswith("#")
    assert "그림 1. 분기별 매출" in page["markdown"]
    assert "표 1. 모델 구성" in page["markdown"]
    assert page["warnings"] == [], "알려진 라벨이므로 경고 없음"


def test_real_paper_labels_mapped_to_text():
    """실 논문에서 관측된 라벨(reference_content·formula_number)은 text로 보존."""
    data = {"res": {"parsing_res_list": [
        {"block_bbox": [0, 0, 500, 60], "block_label": "reference_content",
         "block_content": "[1] Shannon, C. (1948). A mathematical theory.",
         "block_id": 0, "block_order": 0},
        {"block_bbox": [400, 100, 500, 130], "block_label": "formula_number",
         "block_content": "(3)", "block_id": 1, "block_order": 1},
    ]}}
    page = adapt_page(data, 1000, 1000)
    assert [b["type"] for b in page["blocks"]] == ["text", "text"]
    assert page["warnings"] == [], "알려진 라벨이므로 경고 없음"
    assert "Shannon" in page["markdown"] and "(3)" in page["markdown"]


def test_unknown_label_becomes_unknown_type():
    data = {"res": {"parsing_res_list": [{
        "block_bbox": [0, 0, 500, 100], "block_label": "hologram",
        "block_content": "미지의 블록", "block_id": 0, "block_order": 0,
    }]}}
    page = adapt_page(data, 1000, 1000)
    assert page["blocks"][0]["type"] == "unknown"
    assert any("hologram" in w for w in page["warnings"])


def test_figure_cap():
    items = [
        {"block_bbox": [0, i * 10, 500, i * 10 + 9], "block_label": "image",
         "block_content": "", "block_id": i, "block_order": i}
        for i in range(MAX_FIGURES + 5)
    ]
    page = adapt_page({"res": {"parsing_res_list": items}}, 1000, 1000)
    figures = [b for b in page["blocks"] if b["type"] == "image"]
    assert len(figures) == MAX_FIGURES
    assert any("상한" in w for w in page["warnings"])


def test_control_chars_stripped_null_byte():
    data = {"res": {"parsing_res_list": [{
        "block_bbox": [0, 0, 500, 100], "block_label": "text",
        "block_content": "안전\x00하지 않은\x1b 제어문자", "block_id": 0, "block_order": 0,
    }]}}
    page = adapt_page(data, 1000, 1000)
    assert "\x00" not in page["markdown"]
    assert "\x1b" not in page["markdown"]
    assert "안전하지 않은 제어문자" in page["markdown"]


def test_block_order_none_falls_back_to_list_order():
    data = {"res": {"parsing_res_list": [
        {"block_bbox": [0, 0, 500, 100], "block_label": "text",
         "block_content": "첫째", "block_id": 0, "block_order": None},
        {"block_bbox": [0, 200, 500, 300], "block_label": "text",
         "block_content": "둘째", "block_id": 1, "block_order": None},
    ]}}
    page = adapt_page(data, 1000, 1000)
    assert page["markdown"].index("첫째") < page["markdown"].index("둘째")
