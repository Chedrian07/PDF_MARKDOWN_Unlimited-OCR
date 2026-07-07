import json

from app.pipeline.layout import parse_page_blocks, render_layout_html
from app.pipeline.merge import ChunkResult, IncrementalMerger

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
