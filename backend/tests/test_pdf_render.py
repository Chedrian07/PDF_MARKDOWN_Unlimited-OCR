import logging

import pytest

from app.pipeline.pdf import drain_mupdf_warnings, probe_pdf, render_pdf_pages
from app.pipeline.render import render_document_html, render_markdown_html

from conftest import make_pdf_bytes

# 알 수 없는 CIDFont 서브타입(/CIDFontType5)을 가진 최소 PDF — MuPDF가 텍스트를
# 그릴 때 복구성 "syntax error: unknown cid font type"을 내지만 렌더는 계속된다.
# 실사용자 문서(손상 CJK 폰트)가 페이지당 이 에러를 수십 줄씩 stderr에 쏟던 리프로.
BROKEN_CID_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 40 >> stream
BT /F1 12 Tf 50 100 Td <0001> Tj ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type0 /BaseFont /Broken /Encoding /Identity-H /DescendantFonts [6 0 R] >> endobj
6 0 obj << /Type /Font /Subtype /CIDFontType5 /BaseFont /Broken /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor 7 0 R >> endobj
7 0 obj << /Type /FontDescriptor /FontName /Broken /Flags 4 /FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 >> endobj
trailer << /Root 1 0 R /Size 8 >>
%%EOF
"""


def _write_pdf(tmp_path, **kw):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_pdf_bytes(**kw))
    return p


def test_render_pages(tmp_path):
    pdf = _write_pdf(tmp_path, pages=2)
    seen = []
    out = render_pdf_pages(pdf, tmp_path / "pages", dpi=100, max_pages=10,
                           progress_cb=lambda a, b: seen.append((a, b)))
    assert [p.name for p in out] == ["page_0001.png", "page_0002.png"]
    assert all(p.stat().st_size > 0 for p in out)
    assert seen == [(1, 2), (2, 2)]


def test_render_progress_cb_예외가_즉시_중단(tmp_path):
    """progress_cb에서 던진 예외는 렌더 루프를 관통한다 — 러너가 콜백에서
    JobCanceled를 던져 렌더 단계 취소를 구현하는 계약의 기반."""
    pdf = _write_pdf(tmp_path, pages=3)
    calls = []

    def cb(done, total):
        calls.append(done)
        raise RuntimeError("취소 시뮬레이션")

    with pytest.raises(RuntimeError, match="취소 시뮬레이션"):
        render_pdf_pages(pdf, tmp_path / "pages", dpi=100, max_pages=10, progress_cb=cb)
    assert calls == [1]  # 첫 페이지 직후 중단 — 나머지는 렌더되지 않음


def test_render_page_failure_replaced_with_blank(tmp_path, monkeypatch, caplog):
    """한 페이지의 get_pixmap 실패는 흰색 페이지로 대체되고 렌더는 계속된다
    — 페이지 수·파일명 정합은 유지 (청크/글로벌 페이지 번호 계약)."""
    import fitz
    from PIL import Image

    pdf = _write_pdf(tmp_path, pages=3)
    orig = fitz.Page.get_pixmap

    def flaky(self, *a, **kw):
        if self.number == 1:  # 2번째 페이지만 실패
            raise RuntimeError("모의 pixmap 실패")
        return orig(self, *a, **kw)

    monkeypatch.setattr(fitz.Page, "get_pixmap", flaky)
    with caplog.at_level(logging.WARNING, logger="app.pipeline.pdf"):
        out = render_pdf_pages(pdf, tmp_path / "pages", dpi=100, max_pages=10)
    assert [p.name for p in out] == ["page_0001.png", "page_0002.png", "page_0003.png"]
    with Image.open(out[1]) as blank, Image.open(out[0]) as good:
        # 전면 흰색 + 정상 페이지와 같은 크기(반올림 오차 허용)
        assert blank.convert("RGB").getextrema() == ((255, 255), (255, 255), (255, 255))
        assert abs(blank.size[0] - good.size[0]) <= 2
        assert abs(blank.size[1] - good.size[1]) <= 2
    assert any("흰색 페이지로 대체" in r.message for r in caplog.records)


def test_render_all_pages_failed_raises(tmp_path, monkeypatch):
    """전 페이지 렌더 실패 시에만 잡 오류(ValueError)로 승격된다."""
    import fitz

    pdf = _write_pdf(tmp_path, pages=2)

    def boom(self, *a, **kw):
        raise RuntimeError("모의 pixmap 실패")

    monkeypatch.setattr(fitz.Page, "get_pixmap", boom)
    with pytest.raises(ValueError, match="모든 페이지"):
        render_pdf_pages(pdf, tmp_path / "pages", dpi=100, max_pages=10)


def test_probe_pdf_limits(tmp_path):
    pdf = _write_pdf(tmp_path, pages=3)
    assert probe_pdf(pdf, max_pages=10) == 3
    with pytest.raises(ValueError, match="상한"):
        probe_pdf(pdf, max_pages=2)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-not really a pdf")
    with pytest.raises(ValueError):
        probe_pdf(bad, max_pages=10)


def _write_pdf_sized(tmp_path, w_pt, h_pt, name="sized.pdf"):
    """지정한 MediaBox(pt) 크기의 빈 1페이지 PDF."""
    import fitz

    doc = fitz.open()
    doc.new_page(width=w_pt, height=h_pt)
    p = tmp_path / name
    doc.save(str(p))
    doc.close()
    return p


def test_probe_pdf_rejects_oversized_page(tmp_path):
    """PDF 스펙 최대 MediaBox(14400pt) 1페이지는 수 KB 파일로도 기본 200dpi에서
    40000×40000px ≈ RGB 4.8GB를 할당시킨다 — 업로드 검증에서 거부(→ API 400).
    A0(2384×3370pt) 같은 정상 대형 문서는 통과."""
    big = _write_pdf_sized(tmp_path, 14400, 14400, "big.pdf")
    with pytest.raises(ValueError, match="한 변 상한"):
        probe_pdf(big, max_pages=10)
    a0 = _write_pdf_sized(tmp_path, 2384, 3370, "a0.pdf")
    assert probe_pdf(a0, max_pages=10) == 1


def test_render_pixel_cap_downscales_keeping_aspect(tmp_path, monkeypatch, caplog):
    """페이지당 픽셀 상한 초과는 거부가 아니라 비율 유지 축소 + warning 1줄
    — probe를 통과한 정상 문서도 dpi=400에선 걸릴 수 있기 때문."""
    from PIL import Image

    pdf = _write_pdf(tmp_path, pages=1)  # A4 595×842pt
    monkeypatch.setattr("app.pipeline.pdf.MAX_RENDER_PIXELS", 100_000)
    with caplog.at_level(logging.WARNING, logger="app.pipeline.pdf"):
        out = render_pdf_pages(pdf, tmp_path / "pages", dpi=200, max_pages=10)
    with Image.open(out[0]) as im:
        w, h = im.size
    assert w * h <= 100_000 * 1.02          # 상한 이하 (픽스맵 올림 여유 2%)
    assert abs(w / h - 595 / 842) < 0.02    # 가로세로 비 유지
    assert len([r for r in caplog.records if "축소" in r.message]) == 1


def test_normal_a4_unaffected_by_size_caps(tmp_path, caplog):
    """정상 A4 문서는 probe 치수 검사·렌더 픽셀 상한 모두의 영향을 받지 않는다."""
    from PIL import Image

    pdf = _write_pdf(tmp_path, pages=1)
    assert probe_pdf(pdf, max_pages=10) == 1
    with caplog.at_level(logging.WARNING, logger="app.pipeline.pdf"):
        out = render_pdf_pages(pdf, tmp_path / "pages", dpi=200, max_pages=10)
    with Image.open(out[0]) as im:
        w, h = im.size
    # 595×842pt @200dpi 원래 배율 그대로 (595·200/72 ≈ 1653, 842·200/72 ≈ 2339)
    assert abs(w - 595 * 200 / 72) <= 2 and abs(h - 842 * 200 / 72) <= 2
    assert not any("축소" in r.message for r in caplog.records)


def _write_broken_cid_pdf(tmp_path):
    """손상 CID 폰트 PDF 픽스처 — pymupdf로 1회 재저장해 xref를 정규화한다
    (리페어 잡음 제거, 손상 폰트 객체는 보존 → 남는 경고는 cid 에러뿐)."""
    from app.pipeline.pdf import quiet_fitz

    raw = tmp_path / "raw.pdf"
    raw.write_bytes(BROKEN_CID_PDF)
    fixed = tmp_path / "broken.pdf"
    fitz = quiet_fitz()  # 정규화 중 리페어 메시지도 stderr 대신 버퍼로
    doc = fitz.open(str(raw))
    doc.save(str(fixed))
    doc.close()
    drain_mupdf_warnings("픽스처 정규화")  # 리페어 메시지를 버퍼에서 제거
    return fixed


def test_broken_font_pdf_renders_quietly(tmp_path, capfd, caplog):
    """손상 CID 폰트 PDF: 렌더는 성공하고, MuPDF 에러는 stderr(fd 레벨) 대신
    로거 요약 한 줄로 나간다 — 서버 콘솔 스팸 방지."""
    pdf = _write_broken_cid_pdf(tmp_path)
    capfd.readouterr()  # 픽스처 단계 출력 비우기 — 아래는 본 검증분만 캡처
    with caplog.at_level(logging.INFO, logger="app.pipeline.pdf"):
        assert probe_pdf(pdf, max_pages=10) == 1
        out = render_pdf_pages(pdf, tmp_path / "pages", dpi=100, max_pages=10)
    assert [p.name for p in out] == ["page_0001.png"]
    assert out[0].stat().st_size > 0                      # 렌더 자체는 정상
    captured = capfd.readouterr()
    assert "MuPDF" not in captured.err                    # C 레벨 stderr 무출력
    assert "unknown cid font type" not in captured.err
    summaries = [r.message for r in caplog.records if "MuPDF 복구성 경고" in r.message]
    assert summaries                                      # 요약은 로거로 보존
    assert any("unknown cid font type" in m for m in summaries)


def test_drain_mupdf_warnings_summary(monkeypatch, caplog):
    """버퍼 요약 형식 — 종류별 건수, 상위 3종 + '외 N종', 총 건수."""
    import fitz

    monkeypatch.setattr(
        fitz.TOOLS, "mupdf_warnings",
        lambda reset=True: "cid err\ncid err\ncid err\nbogus ascent\nrepairing\nother",
    )
    with caplog.at_level(logging.INFO, logger="app.pipeline.pdf"):
        drain_mupdf_warnings("테스트")
    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "6건" in msg and "테스트" in msg
    assert "cid err (x3)" in msg
    assert "외 1종" in msg                                # 4종 중 상위 3종만 나열


def test_drain_mupdf_warnings_empty_is_silent(caplog):
    """버퍼가 비었으면 로그도 없다 (정상 PDF에서 소음 없음)."""
    drain_mupdf_warnings("사전 비우기")  # 이전 테스트 잔여 버퍼 제거
    with caplog.at_level(logging.INFO, logger="app.pipeline.pdf"):
        drain_mupdf_warnings("빈 버퍼")
    assert not caplog.records


def test_markdown_html_rewrite_and_escape():
    md = '# 제목\n\n![](images/p0001_0.jpg)\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n<script>alert(1)</script>'
    html = render_markdown_html(md, "/api/jobs/j_x/files")
    assert 'src="/api/jobs/j_x/files/images/p0001_0.jpg"' in html
    assert "<table>" in html
    assert "<script>" not in html  # raw HTML은 이스케이프
    assert "&lt;script&gt;" in html


def test_math_inline_normalized():
    md = "질량은 \\( E = mc^{2} \\) 이고 인용은 \\( [10, 30, 33] \\) 형태다."
    html = render_markdown_html(md, "/b")
    assert '<span class="math-inline">E = mc^{2}</span>' in html
    assert '<span class="math-inline">[10, 30, 33]</span>' in html
    assert "\\(" not in html


def test_math_display_multiline():
    md = "앞 문장.\n\n\\[\nx = \\frac{a}{b}, \\quad y_i^2\n\\]\n\n뒤 문장."
    html = render_markdown_html(md, "/b")
    assert '<div class="math-display">' in html
    assert "x = \\frac{a}{b}, \\quad y_i^2" in html
    # dollarmath가 emphasis/서브스크립트 오염을 막는다 — _, ^ 가 태그로 변하지 않음
    assert "<em>" not in html


def test_math_untouched_inside_code():
    md = "```\n\\( raw \\) 코드\n```\n\n그리고 `\\( inline \\)` 코드 스팬."
    html = render_markdown_html(md, "/b")
    assert "math-inline" not in html
    assert "\\( raw \\)" in html


def test_math_xss_escaped():
    html = render_markdown_html("\\( <script>alert(1)</script> \\)", "/b")
    assert "math-inline" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_currency_dollars_not_math():
    html = render_markdown_html("가격은 $5 그리고 $10 입니다.", "/b")
    assert "math-inline" not in html
    assert "$5" in html and "$10" in html


def test_figure_width_injected_from_boxes():
    boxes = {
        "p0001_0.jpg": {"x1": 100, "y1": 0, "x2": 500, "y2": 300, "image_width": 1000, "image_height": 1400},
        "p0002_0.jpg": {"x1": 20, "y1": 0, "x2": 980, "y2": 300, "image_width": 1000, "image_height": 1400},
    }
    md = "![](images/p0001_0.jpg)\n\n![](images/p0002_0.jpg)\n\n![](images/p0009_9.jpg)"
    html = render_markdown_html(md, "/b", figure_boxes=boxes)
    # 40% → 센터링 포함
    assert 'style="width:40.0%;height:auto;display:block;margin-left:auto;margin-right:auto;"' in html
    # 96% → 센터링 없음
    assert 'style="width:96.0%;height:auto;"' in html
    # 메타 없는 이미지는 원래 태그 유지 (폴백)
    assert '<img src="/b/images/p0009_9.jpg" alt="" />' in html


def test_figure_width_fallback_without_boxes():
    md = "![](images/p0001_0.jpg)"
    html = render_markdown_html(md, "/b")
    assert 'style=' not in html
    html2 = render_markdown_html(md, "/b", figure_boxes={"p0001_0.jpg": {"x1": 0}})  # 불완전 메타
    assert 'style=' not in html2


def test_document_html_wraps_pages_in_sections():
    md = "# P1 제목\n\n본문 1\n\n---\n\n본문 2 \\( x^2 \\)\n\n---\n\n본문 3"
    html = render_document_html(md, "/b", page_separator="\n\n---\n\n")
    assert html.count('<section class="doc-page"') == 3
    assert 'data-page="1"' in html and 'data-page="3"' in html
    assert "<hr" not in html  # 구분자가 hr로 렌더되지 않고 섹션 경계로 승격됨
    assert "본문 3" in html
    assert '<span class="math-inline">x^2</span>' in html  # 섹션 내부도 동일 렌더러


def test_document_html_single_page_stays_flat():
    html = render_document_html("한 페이지 문서", "/b", page_separator="\n\n---\n\n")
    assert "<section" not in html
    assert "한 페이지 문서" in html
    assert render_document_html("", "/b") == ""


def test_model_html_tables_restored_safely():
    # Unlimited-OCR 실출력 형태: HTML 표 + 잠재적 악성 태그 혼재
    md = ('<table><tr><td>Mode</td><td colspan="2">base</td></tr></table>\n\n'
          '<img src=x onerror=alert(1)> <table onclick="x"><tr><td>bad</td></tr></table>')
    html = render_markdown_html(md, "/base")
    assert "<table><tr><td>Mode</td>" in html          # 구조 태그 복원
    assert '<td colspan="2">' in html                   # 숫자 속성만 허용
    assert "<img" not in html                           # img 등 다른 태그는 그대로 이스케이프
    assert "onerror" in html and "&lt;img" in html
    assert '<table onclick' not in html                 # 속성 붙은 table은 복원 안 함
