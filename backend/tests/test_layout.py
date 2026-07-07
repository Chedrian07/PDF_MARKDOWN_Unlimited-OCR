import json
from pathlib import Path

from app.pipeline.layout import parse_page_blocks, render_layout_html, render_layout_standalone
from app.pipeline.merge import ChunkResult, IncrementalMerger
from app.pipeline.render import text_with_math_html

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


def test_layout_standalone_self_contained(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p0001_0.jpg").write_bytes(b"\xff\xd8fakejpg")
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "title", "bbox": [0, 0, 900, 80], "content": "제목 \\( x \\)"},
        {"type": "image", "bbox": [100, 100, 800, 600], "content": "", "image": "p0001_0.jpg"},
        {"type": "image", "bbox": [100, 700, 300, 900], "content": "", "image": "missing.jpg"},
    ]}]
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    html = render_layout_standalone(pages, tmp_path, "테스트 문서", frontend_dir)
    assert html.startswith("<!doctype html>")
    assert "<title>테스트 문서</title>" in html
    assert "data:image/jpeg;base64," in html          # 크롭 인라인
    assert 'src="data:,"' in html                      # 결측 크롭 폴백
    assert '<span class="math-inline">x</span>' in html
    if (frontend_dir / "vendor" / "katex" / "katex.min.js").is_file():
        assert "katex" in html and "data:font/woff2;base64," in html  # KaTeX 자립 인라인
    # 외부 참조 없음 (자립성)
    assert 'src="http' not in html and 'href="http' not in html


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
