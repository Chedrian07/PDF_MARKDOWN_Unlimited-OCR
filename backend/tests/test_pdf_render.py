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
