"""OvisOCR2 파서 테스트 — 모델/vLLM 없이 실행 (표준 라이브러리만)."""

from pathlib import Path

import pytest

from app.parser import (
    BBOX_MAX,
    MAX_FIGURES,
    clean_truncated_repeats,
    parse_page,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── 유효 figure 태그 ───────────────────────────────────────────────────────

def test_valid_bbox_tag():
    page = parse_page('앞 텍스트\n\n<img src="images/bbox_100_200_800_700.jpg" />\n\n뒤 텍스트')
    assert page["markdown"] == "앞 텍스트\n\n[[FIGURE:0]]\n\n뒤 텍스트"
    assert page["blocks"] == [{
        "type": "image", "bbox": [100, 200, 800, 700], "content": "",
        "order": 0, "figure_index": 0, "confidence": None,
    }]
    assert page["warnings"] == []


def test_multiple_figures_ordered():
    raw = (
        '<img src="images/bbox_0_0_100_100.jpg" />\n'
        '<img src="images/bbox_200_200_300_300.jpg" />\n'
        '<img src="images/bbox_400_400_500_500.jpg" />'
    )
    page = parse_page(raw)
    assert page["markdown"] == "[[FIGURE:0]]\n[[FIGURE:1]]\n[[FIGURE:2]]"
    assert [b["figure_index"] for b in page["blocks"]] == [0, 1, 2]


def test_whitespace_and_no_slash_variants_accepted():
    page = parse_page(
        '<img  src="images/bbox_10_10_500_500.jpg"/>'
        '<img src="images/bbox_20_20_600_600.jpg" >'
    )
    assert page["markdown"] == "[[FIGURE:0]][[FIGURE:1]]"
    assert len(page["blocks"]) == 2


def test_coordinate_1000_clamped_to_999():
    """[0,1000) 규약의 경계값 1000은 999로 clamp (0.1% 오차)."""
    page = parse_page('<img src="images/bbox_0_0_1000_1000.jpg" />')
    assert page["blocks"][0]["bbox"] == [0, 0, BBOX_MAX, BBOX_MAX]


def test_official_fixture_preserves_structure():
    raw = (FIXTURES / "sample_raw.md").read_text(encoding="utf-8")
    page = parse_page(raw)
    md = page["markdown"]
    # 제목·본문·표 HTML·LaTeX·코드 블록·목록 순서 보존
    assert md.startswith("# Quarterly Report 2026")
    assert "한국어·영문 혼용" in md
    assert "<table><tr><th>항목</th>" in md
    assert "\\[ \\int_0^1 x^2 \\, dx = \\frac{1}{3} \\]" in md
    assert 'print("code block preserved")' in md
    assert md.index("[[FIGURE:0]]") < md.index("[[FIGURE:1]]")
    assert len(page["blocks"]) == 2
    assert md.rstrip().endswith("마지막 문단입니다.")


# ── 악성/비정상 태그 ───────────────────────────────────────────────────────

@pytest.mark.parametrize("evil", [
    '<img src="../../etc/passwd" />',
    '<img src="images/bbox_1_2_3.jpg" />',                    # 좌표 3개
    '<img src="images/bbox_a_b_c_d.jpg" />',                  # 숫자 아님
    '<img src="images/bbox_1_2_3_4_5.jpg" />',                # 좌표 5개
    '<img src="https://attacker.example/x.jpg" />',           # 외부 URL
    '<img src="file:///etc/passwd" />',
    '<img src="C:\\Windows\\system32\\x.jpg" />',
    '<img src="images/bbox_10_10_500_500.png" />',            # 확장자 다름
    '<img src="images/bbox_10_10_500_500.jpg" onerror="x" />',  # 비정상 속성
])
def test_malicious_tags_removed(evil):
    page = parse_page(f"본문 앞 {evil} 본문 뒤")
    assert "img" not in page["markdown"]
    assert "FIGURE" not in page["markdown"]
    assert page["blocks"] == []
    assert page["warnings"]  # 제거 경고
    assert "본문 앞" in page["markdown"] and "본문 뒤" in page["markdown"]


def test_out_of_range_bbox_rejected():
    page = parse_page('<img src="images/bbox_0_0_1001_500.jpg" />')
    assert page["blocks"] == []
    assert any("좌표 이상" in w for w in page["warnings"])


def test_reversed_bbox_rejected():
    page = parse_page('<img src="images/bbox_800_700_100_200.jpg" />')
    assert page["blocks"] == []


def test_degenerate_bbox_rejected():
    page = parse_page('<img src="images/bbox_100_100_101_500.jpg" />')
    assert page["blocks"] == []


def test_duplicate_bbox_dropped():
    raw = '<img src="images/bbox_10_10_500_500.jpg" />' * 3
    page = parse_page(raw)
    assert len(page["blocks"]) == 1
    assert any("중복" in w for w in page["warnings"])


def test_figure_count_cap():
    tags = "\n".join(
        f'<img src="images/bbox_0_{i * 10}_500_{i * 10 + 9}.jpg" />'
        for i in range(MAX_FIGURES + 10)
    )
    page = parse_page(tags)
    assert len(page["blocks"]) == MAX_FIGURES
    assert any("상한" in w for w in page["warnings"])


def test_unclosed_tag_remnant_removed():
    page = parse_page('본문 <img src="images/bbox_1_2_3_4.jpg"')
    assert "<img" not in page["markdown"]
    assert "본문" in page["markdown"]


def test_huge_single_tag_not_matched():
    page = parse_page('<img src="images/bbox_' + "1" * 100_000 + '_2_3_4.jpg" />')
    assert page["blocks"] == []
    assert "<img" not in page["markdown"]


def test_nested_figure_tags():
    page = parse_page(
        '<img src="images/bbox_1_1_500_500.jpg" '
        '<img src="images/bbox_2_2_600_600.jpg" />'
    )
    # 안쪽 태그가 유효하면 하나만 인정되고 바깥 잔여물은 제거된다
    assert "<img" not in page["markdown"]


def test_raw_size_cap():
    page = parse_page("가" * 500_000)
    assert len(page["markdown"]) <= 400_000
    assert any("상한" in w for w in page["warnings"])


# ── 반복 suffix 정리 (모델 카드 공식 알고리즘) ──────────────────────────────

def test_repeats_short_text_untouched():
    text = "abc" * 100  # 8000자 미만
    assert clean_truncated_repeats(text) == text


def test_repeats_cleaned_keeps_one_period():
    base = "정상 본문입니다. " * 900   # 9000자 ≥ min_text_len(8000)
    repeat = "표 셀 반복 " * 40        # 주기 7자, 5회 이상, 100자 이상
    cleaned = clean_truncated_repeats(base + repeat)
    assert cleaned.startswith(base[:100])
    assert len(cleaned) < len(base + repeat)
    # 반복 유닛이 정확히 1회만 남는다
    assert cleaned.endswith("표 셀 반복 ")
    assert not cleaned.endswith("표 셀 반복 표 셀 반복 ")


def test_repeats_with_partial_tail():
    base = "x" * 8000
    cleaned = clean_truncated_repeats(base + "ABCD" * 30 + "AB")
    assert cleaned.endswith("ABCDAB")
    assert cleaned.count("ABCD") <= base.count("ABCD") + 1


def test_repeats_below_threshold_untouched():
    text = "y" * 8000 + "ABCD" * 4  # 5회 미만 (마지막 문자 y와 달라 주기 후보에서 걸러짐)
    # ABCD 4회 = 16자 < 100자 최소치 → 정리되지 않아야 한다
    assert clean_truncated_repeats(text).endswith("ABCD" * 4)


def test_parse_page_applies_repeat_cleanup():
    base = "본문 " * 3000
    page = parse_page(base + "루프 " * 60)
    assert any("반복 suffix" in w for w in page["warnings"])
    assert not page["markdown"].endswith("루프 루프 루프")
