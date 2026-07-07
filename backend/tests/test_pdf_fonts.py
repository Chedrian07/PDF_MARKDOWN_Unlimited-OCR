"""pdf_fonts.enrich_layout_fonts — 원본 PDF 텍스트 레이어에서 실측 폰트 크기 주입.

합성 PDF(612×792pt)를 테스트 안에서 만든다:
- 본문: (100,200)에 11pt로 삽입 → span bbox 세로 ≈188–203pt, 중심 ≈(356,196)
- 제목: (100,100)에 16pt Helvetica-Bold(hebo)로 삽입 → 중심 ≈(136,94)
insert_text의 y는 **베이스라인**이므로 det 사각형은 넉넉히 잡는다.
det bbox는 0–999 정규화 = (x/612×999, y/792×999).
"""

from pathlib import Path

from app.pipeline.pdf_fonts import enrich_layout_fonts


def _make_pdf(tmp_path: Path) -> Path:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((100, 200), "body text ..." * 10, fontsize=11)
    page.insert_text((100, 100), "Bold Title", fontsize=16, fontname="hebo")  # Helvetica-Bold
    p = tmp_path / "source.pdf"
    doc.save(str(p))
    doc.close()
    return p


def _norm(x_pt: float, y_pt: float) -> tuple[int, int]:
    return round(x_pt / 612 * 999), round(y_pt / 792 * 999)


def test_enrich_injects_measured_font_sizes(tmp_path):
    pdf = _make_pdf(tmp_path)
    # 제목 사각형 (80,80)–(400,120)pt, 본문 (80,180)–(600,215)pt
    tx1, ty1 = _norm(80, 80)
    tx2, ty2 = _norm(400, 120)
    bx1, by1 = _norm(80, 180)
    bx2, by2 = _norm(600, 215)
    # 빈 영역 (80,400)–(500,500)pt, 이미지 블록은 임의 bbox + image 키
    ex1, ey1 = _norm(80, 400)
    ex2, ey2 = _norm(500, 500)

    pages = [{
        "page": 1, "width": 1000, "height": 1294,  # 픽셀 크기(무관 — enrich는 pt 사용)
        "blocks": [
            {"type": "title", "bbox": [tx1, ty1, tx2, ty2], "content": "Bold Title"},
            {"type": "text", "bbox": [bx1, by1, bx2, by2], "content": "body text"},
            {"type": "image", "bbox": [tx1, ty1, tx2, ty2], "content": "", "image": "p0001_0.jpg"},
            {"type": "text", "bbox": [ex1, ey1, ex2, ey2], "content": "빈 영역"},
        ],
    }]

    changed = enrich_layout_fonts(pdf, pages)
    assert changed is True

    title, body, image, empty = pages[0]["blocks"]

    # 본문 11pt → 11/612×100 = 1.80cqw ±0.15
    assert "fs" in body
    assert abs(body["fs"] - 11 / 612 * 100) < 0.15, body["fs"]

    # 제목 16pt → 16/612×100 = 2.61cqw ±0.2, 볼드
    assert "fs" in title
    assert abs(title["fs"] - 16 / 612 * 100) < 0.2, title["fs"]
    assert title.get("bold") is True

    # 이미지 블록은 손대지 않는다
    assert "fs" not in image and "bold" not in image

    # 빈 영역 블록엔 span이 없어 fs 미주입
    assert "fs" not in empty and "bold" not in empty


def test_enrich_empty_page_returns_false(tmp_path):
    import fitz

    doc = fitz.open()
    doc.new_page(width=612, height=792)  # 텍스트 없음
    p = tmp_path / "empty.pdf"
    doc.save(str(p))
    doc.close()

    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [100, 100, 900, 300], "content": "무엇이든"},
    ]}]
    assert enrich_layout_fonts(p, pages) is False
    assert "fs" not in pages[0]["blocks"][0]


def test_enrich_corrupt_pdf_returns_false(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [0, 0, 999, 999], "content": "x"},
    ]}]
    assert enrich_layout_fonts(bad, pages) is False


def test_enrich_page_index_out_of_range(tmp_path):
    pdf = _make_pdf(tmp_path)
    pages = [{"page": 99, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [131, 227, 979, 271], "content": "x"},
    ]}]
    # 범위를 벗어난 페이지는 조용히 스킵 — 아무것도 주입 안 함
    assert enrich_layout_fonts(pdf, pages) is False
    assert "fs" not in pages[0]["blocks"][0]
