"""세그먼트 — identity 골든·references 스킵·hr 새니타이즈·layout 불변."""

from app.translate.segment import (
    apply_layout,
    assemble_markdown,
    layout_units,
    split_markdown,
)

SEP = "\n\n---\n\n"

# 표·펜스·heading·인용·references 포함 3페이지 합성 (merge.py 스타일: strip + 후행 \n)
MD = (
    "# Introduction\n\n"
    "This is the opening paragraph with a citation [1] and $x^2$ math.\n\n"
    "| Name | Score |\n|------|-------|\n| A | 0.9 |\n| B | 0.8 |"
    + SEP +
    "## Method\n\n"
    "We describe the approach below.\n\n"
    "```python\ndef f(x):\n    return x + 1\n```\n\n"
    "> An indented quote block.\n\n"
    "- first bullet\n- second bullet"
    + SEP +
    "## References\n\n"
    "[1] Author A. A paper title. Venue, 2020.\n\n"
    "[2] Author B. Another title. Venue, 2021.\n"
)


def test_identity_골든_바이트_동일():
    """전 유닛을 src 그대로 매핑하면 조립 결과가 원본과 바이트 동일해야 한다."""
    units = split_markdown(MD, SEP)
    translations = {u.id: u.src for u in units}
    assert assemble_markdown(MD, SEP, translations) == MD


def test_유닛_종류와_id():
    units = split_markdown(MD, SEP)
    kinds = {u.id: u.kind for u in units}
    assert kinds["md:0:0"] == "heading"
    assert kinds["md:0:2"] == "table"
    assert kinds["md:1:2"] == "fence"
    assert kinds["md:1:3"] == "blockquote"
    assert kinds["md:1:4"] == "list"


def test_references_스킵():
    units = split_markdown(MD, SEP)
    by_id = {u.id: u for u in units}
    # References heading과 그 뒤 항목 전부 skip
    assert by_id["md:2:0"].skip_reason == "references"
    assert by_id["md:2:1"].skip_reason == "references"
    assert by_id["md:2:2"].skip_reason == "references"
    # 그 외 페이지는 스킵되지 않음
    assert by_id["md:0:1"].skip_reason == ""
    assert by_id["md:1:1"].skip_reason == ""


def test_references_구간_상위heading에서_종료():
    """references(h2) 하위 소제목(h3)은 계속 스킵, 다음 동급/상위 heading(h1)에서 해제."""
    md = (
        "## References\n\n[1] a paper.\n\n### Sub note\n\nstill in refs.\n\n"
        "# Appendix Data\n\nback to translatable content."
    )
    units = split_markdown(md, SEP)
    by_id = {u.id: u for u in units}
    assert by_id["md:0:0"].skip_reason == "references"   # ## References
    assert by_id["md:0:1"].skip_reason == "references"   # [1] a paper
    assert by_id["md:0:2"].skip_reason == "references"   # ### Sub note (하위)
    assert by_id["md:0:3"].skip_reason == "references"   # still in refs
    # "# Appendix Data"는 acknowledg/references 패턴이 아니고 h1(상위) → 해제
    assert by_id["md:0:4"].skip_reason == ""
    assert by_id["md:0:5"].skip_reason == ""


def test_hr_새니타이즈_페이지수_불변():
    """번역문에 '---' 줄이 생겨도 ⸻로 치환되어 페이지 수가 유지된다."""
    units = split_markdown(MD, SEP)
    trans = {}
    for u in units:
        if u.kind == "paragraph" and not u.skip_reason:
            trans[u.id] = "번역 첫 줄\n---\n번역 둘째 줄"
        else:
            trans[u.id] = u.src
    out = assemble_markdown(MD, SEP, trans)
    assert len(out.split(SEP)) == 3       # 페이지 수 보존
    assert "⸻" in out                     # hr → ⸻ 치환
    assert "\n---\n다시" not in out


def test_페이지구분자_유발_유닛_원문유지():
    # 기본 '---' 구분자는 새니타이즈가 중화하므로, 새니타이즈가 못 잡는 커스텀
    # 구분자로 유닛 단위 선방어(page_separator in new_text → 원문 유지)를 검증한다.
    sep = "\n\n@@@\n\n"
    md = "# A\n\nfirst page paragraph." + sep + "# B\n\nsecond page paragraph."
    units = split_markdown(md, sep)
    trans = {u.id: u.src for u in units}
    para = next(u for u in units if u.kind == "paragraph")
    trans[para.id] = "오염 시도" + sep + "뒷부분"   # 구분자 유발
    out = assemble_markdown(md, sep, trans)
    assert len(out.split(sep)) == 2        # 페이지 수 보존
    assert para.src in out                 # 해당 유닛은 원문 유지


def test_layout_units_필터():
    pages = [{
        "page": 3, "width": 1000, "height": 1400, "blocks": [
            {"type": "text", "bbox": [0, 0, 999, 50], "content": "translate me"},
            {"type": "image", "bbox": [0, 60, 999, 400], "content": "", "image": "p0003_0.jpg"},
            {"type": "text", "bbox": [0, 410, 999, 450], "content": "   "},  # 빈 content
        ],
    }]
    units = layout_units(pages)
    assert [u.id for u in units] == ["lay:3:0"]  # 이미지·빈 content 제외
    assert units[0].page == 3


def test_apply_layout_content외_필드_불변():
    pages = [{
        "page": 1, "width": 612, "height": 792, "fonts_v": "2", "blocks": [
            {"type": "title", "bbox": [0, 0, 999, 80], "content": "Title", "fs": 2.5, "bold": True},
            {"type": "text", "bbox": [10, 300, 40, 900], "content": "vertical", "fs": 1.47,
             "vertical": "up"},
            {"type": "image", "bbox": [0, 100, 500, 300], "content": "", "image": "p0001_0.jpg"},
        ],
    }]
    units = layout_units(pages)
    out = apply_layout(pages, {u.id: "번역:" + u.src for u in units})
    b = out[0]["blocks"]
    assert b[0]["content"] == "번역:Title" and b[0]["fs"] == 2.5 and b[0]["bold"] is True
    assert b[1]["content"] == "번역:vertical" and b[1]["vertical"] == "up"
    assert out[0]["fonts_v"] == "2"
    # 이미지 블록·원본은 손대지 않음
    assert b[2]["content"] == "" and b[2]["image"] == "p0001_0.jpg"
    assert pages[0]["blocks"][0]["content"] == "Title"  # 원본 불변(deep copy)
