import pytest

from app.pipeline.pdf import probe_pdf, render_pdf_pages
from app.pipeline.render import render_markdown_html

from conftest import make_pdf_bytes


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


def test_probe_pdf_limits(tmp_path):
    pdf = _write_pdf(tmp_path, pages=3)
    assert probe_pdf(pdf, max_pages=10) == 3
    with pytest.raises(ValueError, match="상한"):
        probe_pdf(pdf, max_pages=2)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-not really a pdf")
    with pytest.raises(ValueError):
        probe_pdf(bad, max_pages=10)


def test_markdown_html_rewrite_and_escape():
    md = '# 제목\n\n![](images/p0001_0.jpg)\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n<script>alert(1)</script>'
    html = render_markdown_html(md, "/api/jobs/j_x/files")
    assert 'src="/api/jobs/j_x/files/images/p0001_0.jpg"' in html
    assert "<table>" in html
    assert "<script>" not in html  # raw HTML은 이스케이프
    assert "&lt;script&gt;" in html


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
