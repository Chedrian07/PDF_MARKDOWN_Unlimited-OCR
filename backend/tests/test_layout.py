import json
import re
from pathlib import Path

from app.pipeline.layout import (
    estimate_font_size_cqw,
    parse_page_blocks,
    render_layout_html,
    render_layout_standalone,
)
from app.pipeline.merge import ChunkResult, IncrementalMerger
from app.pipeline.render import text_with_math_html

# 실제 frontend 디렉터리 (repo/frontend) — layout-fit.js / KaTeX 자산 존재 확인용.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

RAW = (
    "<|det|>title [100, 50, 800, 100]<|/det|>문서 제목\n"
    "<|det|>text [100, 120, 900, 300]<|/det|>첫 단락 텍스트<|/ref|> 잔여토큰 포함\n"
    "<|det|>image [150, 320, 700, 600]<|/det|>\n"
    "<|det|>table [100, 620, 900, 800]<|/det|><table><tr><td>a</td></tr></table>"
)


def test_parse_page_blocks_document_order_and_types():
    blocks = parse_page_blocks(RAW)
    assert [b["type"] for b in blocks] == ["title", "text", "image", "table"]
    assert blocks[0]["bbox"] == [100, 50, 800, 100]
    assert blocks[0]["content"] == "문서 제목"
    assert "잔여토큰" in blocks[1]["content"] and "<|/ref|>" not in blocks[1]["content"]
    assert blocks[2]["crop_index"] == 0 and blocks[2]["content"] == ""
    assert "<table>" in blocks[3]["content"]


def test_parse_crop_index_matches_vendor_order():
    # 벤더 re_match: ref류 매치 전체가 det류보다 먼저 인덱싱된다 —
    # 문서상 det 이미지가 먼저 나와도 ref 이미지가 crop 0이어야 한다.
    raw = (
        "<|det|>image [10, 10, 100, 100]<|/det|>\n"
        "<|ref|>image<|/ref|><|det|>[[200, 200, 400, 400]]<|/det|>\n"
    )
    blocks = parse_page_blocks(raw)
    by_bbox = {tuple(b["bbox"]): b for b in blocks if b["type"] == "image"}
    assert by_bbox[(200, 200, 400, 400)]["crop_index"] == 0  # ref류 먼저
    assert by_bbox[(10, 10, 100, 100)]["crop_index"] == 1


def test_parse_ref_multibox_and_inner_det_dedup():
    raw = "<|ref|>image<|/ref|><|det|>[[10, 10, 50, 50], [60, 60, 90, 90]]<|/det|>"
    blocks = parse_page_blocks(raw)
    assert len(blocks) == 2  # 내부 det 태그가 중복 파싱되지 않음
    assert [b["crop_index"] for b in blocks] == [0, 1]


def test_render_layout_html_positions_and_escaping():
    pages = [{
        "page": 2, "width": 1000, "height": 1500,
        "blocks": [
            {"type": "title", "bbox": [0, 0, 999, 99], "content": "<script>x</script>"},
            {"type": "table", "bbox": [0, 100, 999, 300], "content": "<table><tr><td>a</td></tr></table>"},
            {"type": "image", "bbox": [100, 400, 600, 800], "content": "", "image": "p0002_0.jpg"},
            {"type": "bad type!", "bbox": [0, 0, 10, 10], "content": "x"},
            {"type": "text", "bbox": [1, 2, 3], "content": "무시됨"},  # 비정상 bbox
        ],
    }]
    html = render_layout_html(pages, "/api/jobs/j_x/files")
    assert 'data-page="2"' in html
    assert "padding-top:150.00%" in html  # 1500/1000
    assert "left:0.00%;top:0.00%" in html
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "<table><tr><td>a</td></tr></table>" in html  # 표는 화이트리스트 복원
    assert 'src="/api/jobs/j_x/files/images/p0002_0.jpg"' in html
    assert "layout-text" in html and "bad type!" not in html  # 타입 새니타이즈
    assert "무시됨" not in html


def test_text_with_math_html():
    out = text_with_math_html("질량은 \\( E = mc^2 \\) 이고 <b>태그</b>는 이스케이프.")
    assert '<span class="math-inline">E = mc^2</span>' in out
    assert "&lt;b&gt;" in out and "<b>" not in out
    out2 = text_with_math_html("\\[\nx^2 + y^2\n\\] 끝")
    assert '<span class="math-display">x^2 + y^2</span>' in out2
    assert text_with_math_html("수식 없음") == "수식 없음"


def test_layout_blocks_render_math_spans():
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "text", "bbox": [0, 0, 500, 100], "content": "본문 \\( a^2 \\) 수식"},
        {"type": "equation", "bbox": [0, 200, 900, 300], "content": "\\[ D = \\mathbb{E}[x] \\]"},
    ]}]
    html = render_layout_html(pages, "/b")
    assert '<span class="math-inline">a^2</span>' in html
    assert '<span class="math-display">D = \\mathbb{E}[x]</span>' in html
    assert "\\(" not in html and "\\[" not in html


# ── 면적 기반 폰트 크기 추정 (cqw) ─────────────────────────────────────
def test_estimate_font_size_calibration():
    # 진실 앵커(2504.19874v1.pdf 실측): 612×792pt 페이지 본문 10.9pt = 1.78cqw.
    # 그 박스는 원본 타이포로 ~1180 ASCII자를 담는다 → 재현 케이스
    # (bbox(60,100,940,280)·A4 비율·1180자)에서 fs가 1.6–2.0cqw여야 한다.
    fs = estimate_font_size_cqw((60, 100, 940, 280), "x" * 1180, 1.414)
    assert fs is not None
    assert 1.6 <= fs <= 2.0, fs


def test_estimate_cjk_smaller_than_ascii():
    # 같은 글자수라도 CJK는 가중치(1.0)가 ASCII(0.5)보다 커서 더 작은 fs가 나온다.
    box = (60, 100, 940, 280)
    ascii_fs = estimate_font_size_cqw(box, "x" * 300, 1.414)
    cjk_fs = estimate_font_size_cqw(box, "가" * 300, 1.414)
    assert ascii_fs is not None and cjk_fs is not None
    assert cjk_fs < ascii_fs


def test_estimate_single_line_title_cap():
    # 얕은 박스의 짧은 제목 — 면적 모델은 크게 잡지만 단일 줄 상한(h/1.25)이 눌러야 함.
    bbox = (100, 50, 900, 75)
    fs = estimate_font_size_cqw(bbox, "Title", 1.414)
    h = (75 - 50) / 999 * 100 * 1.414
    cap = h / 1.25
    assert fs is not None
    assert abs(fs - cap) < 0.02, (fs, cap)
    assert cap < 3.6  # 클램프가 아니라 '상한'이 작동함을 보장


def test_estimate_clamps_hold_at_extremes():
    # 상한: 큰 박스 + 극소 글자수 → 3.6 클램프
    hi = estimate_font_size_cqw((100, 100, 200, 900), "xx", 1.414)
    assert hi == 3.6
    # 하한: 큰 박스 + 초대량 글자수 → 0.8 클램프
    lo = estimate_font_size_cqw((0, 0, 999, 999), "가" * 50000, 1.414)
    assert lo == 0.8


def test_estimate_empty_and_none_safe():
    box = (60, 100, 940, 280)
    assert estimate_font_size_cqw(box, "", 1.414) is None
    assert estimate_font_size_cqw(box, "   ", 1.414) is None
    assert estimate_font_size_cqw(box, "<table></table>", 1.414) is None  # 태그만 → 빈 텍스트
    assert estimate_font_size_cqw(None, "본문", 1.414) is None
    assert estimate_font_size_cqw((1, 2, 3), "본문", 1.414) is None  # 비정상 bbox


def test_render_layout_html_font_size_cqw_text_not_image():
    pages = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "본문 텍스트 예시 " * 30},
        {"type": "image", "bbox": [100, 400, 600, 800], "content": "", "image": "p0001_0.jpg"},
    ]}]
    html = render_layout_html(pages, "/b")
    text_div = re.search(r'<div class="layout-block layout-text"[^>]*>', html).group(0)
    assert "font-size:" in text_div and "cqw" in text_div
    assert "line-height:1.22" in text_div
    # 이미지 블록엔 폰트 크기 인라인이 없어야 함
    img_tag = re.search(r'<img class="layout-block layout-image"[^>]*>', html).group(0)
    assert "cqw" not in img_tag and "font-size:" not in img_tag


def test_render_layout_html_fs_precedence():
    # 실측 block["fs"]가 있으면 휴리스틱을 무시하고 그 값을 그대로 쓴다([0.6,6.0] 클램프).
    pages = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "본문 " * 40, "fs": 1.78},
        {"type": "text", "bbox": [60, 300, 940, 480], "content": "굵게 " * 40, "fs": 2.5, "bold": True},
        {"type": "title", "bbox": [60, 500, 940, 560], "content": "제목", "fs": 3.0, "bold": True},
        {"type": "text", "bbox": [60, 600, 940, 700], "content": "과대", "fs": 99.0},
        {"type": "text", "bbox": [60, 720, 940, 900], "content": "폴백 텍스트 예시 " * 30},  # fs 없음
    ]}]
    html = render_layout_html(pages, "/b")
    divs = re.findall(r'<div class="layout-block layout-\w+"[^>]*style="([^"]*)"', html)
    assert "font-size:1.78cqw;line-height:1.22;" in divs[0]
    assert "font-weight:600;" not in divs[0]  # bold 아님
    # 볼드 실측 블록: 굵게 (제목 아님)
    assert "font-size:2.50cqw;line-height:1.22;" in divs[1] and "font-weight:600;" in divs[1]
    # 제목은 실측 bold라도 font-weight 인라인 안 함 (CSS가 이미 굵게)
    assert "font-size:3.00cqw;line-height:1.22;" in divs[2] and "font-weight:600;" not in divs[2]
    # 클램프 상한 6.0
    assert "font-size:6.00cqw;line-height:1.22;" in divs[3]
    # fs 없는 블록은 휴리스틱 폴백 — cqw는 있되 1.78/2.50 등 실측값과 다름
    assert "cqw" in divs[4] and "font-size:1.78cqw" not in divs[4]


def test_layout_standalone_includes_fitter_after_typeset(tmp_path):
    (tmp_path / "images").mkdir()
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "본문 텍스트 " * 40},
    ]}]
    html = render_layout_standalone(pages, tmp_path, "문서", FRONTEND_DIR)
    assert "window.uocrFitLayout" in html          # fitter 정의 인라인됨
    assert "uocrFitLayout(document)" in html        # 문서 전체에 호출
    assert html.index("window.uocrFitLayout") < html.index("uocrFitLayout(document)")
    assert "cqw" in html                            # 서버가 심은 면적 기반 폰트 크기
    assert "white-space: pre-wrap" in html          # 문서 타이포(줄바꿈 보존)
    assert "container-type: inline-size" in html    # cqw 기준 컨테이너
    if (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        # 순서: KaTeX 타이포셋 → uocrFitLayout(document)
        assert html.index("katex.render") < html.index("uocrFitLayout(document)")


def test_layout_standalone_self_contained(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p0001_0.jpg").write_bytes(b"\xff\xd8fakejpg")
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "title", "bbox": [0, 0, 900, 80], "content": "제목 \\( x \\)"},
        {"type": "image", "bbox": [100, 100, 800, 600], "content": "", "image": "p0001_0.jpg"},
        {"type": "image", "bbox": [100, 700, 300, 900], "content": "", "image": "missing.jpg"},
    ]}]
    html = render_layout_standalone(pages, tmp_path, "테스트 문서", FRONTEND_DIR)
    assert html.startswith("<!doctype html>")
    assert "<title>테스트 문서</title>" in html
    assert "data:image/jpeg;base64," in html          # 크롭 인라인
    assert 'src="data:,"' in html                      # 결측 크롭 폴백
    assert '<span class="math-inline">x</span>' in html
    if (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        assert "katex" in html and "data:font/woff2;base64," in html  # KaTeX 자립 인라인
    # 외부 참조 없음 (자립성)
    assert 'src="http' not in html and 'href="http' not in html


def test_document_standalone_self_contained(tmp_path):
    """문서 뷰 standalone(document.html) — figure_only 엔진에서도 동작하는 내보내기.
    이미지 base64 인라인·결측 폴백·서버 참조 제거·제목 이스케이프·lang 부여."""
    from app.pipeline.layout import render_document_standalone

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p0001_0.jpg").write_bytes(b"\xff\xd8fakejpg")
    inner = (
        '<section class="doc-page" data-page="1">\n'
        "<h1>제목</h1>\n<p>본문 문단</p>\n"
        '<p><img src="/api/jobs/j_x/files/images/p0001_0.jpg" alt="" '
        'style="width:31.7%;height:auto;"></p>\n'
        '<p><img src="/api/jobs/j_x/files/images/missing.jpg" alt=""></p>\n'
        '<p><span class="math-inline">\\tau^{2}</span></p>\n</section>'
    )
    html = render_document_standalone(inner, tmp_path, "테스트 <문서>", FRONTEND_DIR)
    assert html.startswith("<!doctype html>")
    assert "<title>테스트 &lt;문서&gt;</title>" in html   # 제목 이스케이프
    assert "data:image/jpeg;base64," in html               # 이미지 인라인
    assert 'src="data:,"' in html                          # 결측 이미지 폴백
    assert "/api/jobs/" not in html                        # 서버 참조 없는 완전 자립
    assert 'style="width:31.7%;height:auto;"' in html      # figure 상대폭 스타일 보존
    assert '<span class="math-inline">\\tau^{2}</span>' in html
    if (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        assert "katex" in html and "data:font/woff2;base64," in html
    assert 'src="http' not in html and 'href="http' not in html
    # lang: 원본엔 없음 / ko엔 html·main 모두 부여 (keep-all 타이포 스코프)
    assert "<html>" in html and "<main>" in html
    ko = render_document_standalone(inner, tmp_path, "문서", FRONTEND_DIR, lang="ko")
    assert '<html lang="ko">' in ko and '<main lang="ko">' in ko


def test_document_standalone_no_traversal(tmp_path):
    """이미지 파일명 캡처는 슬래시 배제 — `../` 류는 매치 자체가 안 돼 원문 유지
    (존재하지 않는 서버 경로로 남을 뿐, 파일 시스템 접근 없음)."""
    from app.pipeline.layout import render_document_standalone

    (tmp_path / "images").mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    inner = '<img src="/api/jobs/j_x/files/images/../secret.txt" alt="">'
    html = render_document_standalone(inner, tmp_path, "t", None)
    assert "secret" not in html.replace("secret.txt", "")  # 파일 내용 미포함
    assert "data:image/jpeg;base64," not in html


def test_vertical_blocks_render_writing_mode_class():
    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [10, 300, 40, 900], "content": "arXiv:1908.07836v1 [cs.CL]",
         "fs": 1.47, "vertical": "up"},
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "일반 본문", "fs": 1.78},
    ]}]
    html = render_layout_html(pages, "/b")
    assert "layout-vertical-up" in html
    # 일반 블록에는 세로 클래스가 붙지 않는다
    assert html.count("layout-vertical-") == 1


def test_vertical_geometric_fallback_without_text_layer():
    # 텍스트 레이어 없는(fs 미주입) 스캔 PDF: 극단적으로 좁고 긴 텍스트 박스는 세로 간주
    tall = {"type": "text", "bbox": [10, 200, 40, 900], "content": "arXiv:1908.07836v1 [cs.CL] 16 Aug 2019"}
    normal = {"type": "text", "bbox": [60, 100, 940, 280], "content": "일반 본문 " * 20}
    html = render_layout_html([{"page": 1, "width": 612, "height": 792, "blocks": [tall, normal]}], "/b")
    assert "layout-vertical-up" in html
    assert html.count("layout-vertical-") == 1
    # fs가 실측된 좁은 박스는 (세로 플래그 없이는) 폴백을 타지 않는다
    html2 = render_layout_html([{"page": 1, "width": 612, "height": 792, "blocks": [
        {**tall, "fs": 1.47},
    ]}], "/b")
    assert "layout-vertical-" not in html2


def test_merge_ingests_layout_json(tmp_path):
    (tmp_path / "pages").mkdir()
    m = IncrementalMerger(tmp_path, "\n\n---\n\n")
    c = tmp_path / "work" / "chunk_00"
    (c / "images").mkdir(parents=True)
    (c / "images" / "page_0_0.jpg").write_bytes(b"jpg")
    (c / "raw_pages.json").write_text(json.dumps({"pages": [RAW]}), encoding="utf-8")
    m.add_chunk(ChunkResult(c, 5, 1, "<PAGE>\n본문 ![](images/page_0_0.jpg)"))

    saved = json.loads((tmp_path / "layout.json").read_text(encoding="utf-8"))
    assert saved[0]["page"] == 5
    img_blocks = [b for b in saved[0]["blocks"] if b["type"] == "image"]
    assert img_blocks[0]["image"] == "p0005_0.jpg"  # 글로벌 이미지명 매핑
    assert "crop_index" not in img_blocks[0]


def test_merge_without_raw_pages_is_fine(tmp_path):
    m = IncrementalMerger(tmp_path, "\n\n---\n\n")
    c = tmp_path / "work" / "chunk_00"
    c.mkdir(parents=True)
    m.add_chunk(ChunkResult(c, 1, 1, "<PAGE>\n텍스트만"))
    assert not (tmp_path / "layout.json").exists()
