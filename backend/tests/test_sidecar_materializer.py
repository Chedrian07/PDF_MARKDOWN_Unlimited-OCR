"""공통 materializer 테스트 — 기존 청크 산출물 규약(merge/layout 계약) 재현 검증."""

import json

import pytest
from PIL import Image

from app.pipeline.layout import parse_page_blocks
from app.sidecar.materializer import ChunkMaterializer
from app.sidecar.protocol import PageResult, sanitize_page


@pytest.fixture
def page_image(tmp_path):
    p = tmp_path / "page_0001.png"
    Image.new("RGB", (500, 700), "white").save(p)
    return p


def _page(markdown: str, blocks: list[dict]) -> PageResult:
    page = PageResult.model_validate(
        {"page_index": 0, "markdown": markdown, "blocks": blocks}
    )
    clean, _ = sanitize_page(page)
    return clean


def _figure(idx: int, bbox: list[int], order: int) -> dict:
    return {"type": "image", "bbox": bbox, "content": "", "order": order,
            "figure_index": idx}


def test_multi_naming_contract(tmp_path, page_image):
    out = tmp_path / "chunk_00"
    mat = ChunkMaterializer(out, single=False)
    md = mat.add_page(_page(
        "제목\n\n[[FIGURE:0]]\n\n본문\n\n[[FIGURE:1]]",
        [
            {"type": "title", "bbox": [50, 20, 900, 80], "content": "제목", "order": 0},
            _figure(0, [100, 200, 800, 700], 1),
            {"type": "text", "bbox": [50, 720, 900, 900], "content": "본문", "order": 2},
            _figure(1, [100, 750, 500, 990], 3),
        ],
    ), page_image, local_page=0)
    mat.finalize()

    assert (out / "images" / "page_0_0.jpg").is_file()
    assert (out / "images" / "page_0_1.jpg").is_file()
    assert (out / "result_with_boxes_0.jpg").is_file()
    assert "![](images/page_0_0.jpg)" in md
    assert "![](images/page_0_1.jpg)" in md
    assert md.index("page_0_0") < md.index("page_0_1")

    boxes = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    meta = boxes["page_0_0.jpg"]
    # 픽셀 crop 좌표 + 페이지 크기 (벤더 P13 계약)
    assert meta["image_width"] == 500 and meta["image_height"] == 700
    assert meta["x1"] == int(100 / 999 * 500)
    assert meta["y2"] == int(700 / 999 * 700)


def test_single_naming_contract(tmp_path, page_image):
    out = tmp_path / "chunk_00"
    mat = ChunkMaterializer(out, single=True)
    md = mat.add_page(
        _page("[[FIGURE:0]]", [_figure(0, [100, 200, 800, 700], 0)]),
        page_image, local_page=0,
    )
    mat.finalize()
    assert (out / "images" / "0.jpg").is_file()
    assert (out / "result_with_boxes.jpg").is_file()
    assert md == "![](images/0.jpg)"


def test_raw_pages_layout_roundtrip(tmp_path, page_image):
    """합성 raw가 기존 parse_page_blocks 문법으로 정확히 복원되는지 (crop_index 포함)."""
    out = tmp_path / "chunk_00"
    mat = ChunkMaterializer(out, single=False)
    mat.add_page(_page(
        "# 제목\n\n[[FIGURE:0]]\n\n본문 문단",
        [
            {"type": "title", "bbox": [50, 20, 900, 80], "content": "제목", "order": 0},
            _figure(0, [100, 200, 800, 700], 1),
            {"type": "text", "bbox": [50, 720, 900, 900], "content": "본문 문단", "order": 2},
            {"type": "table", "bbox": [50, 910, 900, 980],
             "content": "<table><tr><td>셀</td></tr></table>", "order": 3},
        ],
    ), page_image, local_page=0)
    mat.finalize()

    raw_pages = json.loads((out / "raw_pages.json").read_text(encoding="utf-8"))["pages"]
    assert len(raw_pages) == 1
    blocks = parse_page_blocks(raw_pages[0])
    types = [b["type"] for b in blocks]
    assert types == ["title", "image", "text", "table"]
    assert blocks[0]["content"] == "제목"
    assert blocks[0]["bbox"] == [50, 20, 900, 80]
    # image 블록의 crop_index가 저장된 crop 파일 순서와 일치 (merge가 이 규약으로 매핑)
    assert blocks[1]["crop_index"] == 0
    assert blocks[1]["content"] == ""
    assert "<table>" in blocks[3]["content"]


def test_placeholder_without_figure_removed(tmp_path, page_image):
    out = tmp_path / "c"
    mat = ChunkMaterializer(out, single=True)
    md = mat.add_page(_page("본문 [[FIGURE:7]] 끝", []), page_image, 0)
    assert "FIGURE" not in md
    assert "본문" in md and "끝" in md
    assert any("대응하는 figure 없음" in w for w in mat.warnings)


def test_orphan_figure_appended(tmp_path, page_image):
    """placeholder가 없는 crop은 내용 손실 방지를 위해 페이지 끝에 붙는다."""
    out = tmp_path / "c"
    mat = ChunkMaterializer(out, single=True)
    md = mat.add_page(
        _page("figure 참조가 없는 본문", [_figure(0, [100, 200, 800, 700], 0)]),
        page_image, 0,
    )
    assert md.endswith("![](images/0.jpg)")
    assert (out / "images" / "0.jpg").is_file()


def test_degenerate_crop_skipped(tmp_path, page_image):
    """픽셀 crop이 최소 크기 미만이면 저장하지 않고 placeholder도 제거된다."""
    out = tmp_path / "c"
    mat = ChunkMaterializer(out, single=True)
    md = mat.add_page(
        # 정규화 2×2는 500px 페이지에서 1px — 최소 crop(4px) 미만
        _page("[[FIGURE:0]]", [_figure(0, [10, 10, 12, 12], 0)]),
        page_image, 0,
    )
    mat.finalize()
    assert not (out / "images" / "0.jpg").exists()
    assert "FIGURE" not in md
    # 저장 실패한 figure는 합성 raw에도 들어가지 않는다 (crop_index 정합)
    raw = json.loads((out / "raw_pages.json").read_text(encoding="utf-8"))["pages"][0]
    assert "image" not in raw


def test_crop_out_of_bounds_clamped_to_page(tmp_path, page_image):
    out = tmp_path / "c"
    mat = ChunkMaterializer(out, single=True)
    mat.add_page(_page("[[FIGURE:0]]", [_figure(0, [900, 900, 999, 999], 0)]),
                 page_image, 0)
    mat.finalize()
    boxes = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    meta = boxes["0.jpg"]
    assert meta["x2"] <= 500 and meta["y2"] <= 700
    with Image.open(out / "images" / "0.jpg") as im:
        assert im.width <= 500 and im.height <= 700


def test_malicious_markdown_refs_are_inert(tmp_path, page_image):
    """모델 markdown의 악성 참조는 파일 생성으로 이어지지 않는다 (텍스트로만 남음)."""
    evil = "![](../../etc/passwd) ![](file:///etc/passwd) [[FIGURE:0]]"
    out = tmp_path / "c"
    mat = ChunkMaterializer(out, single=True)
    md = mat.add_page(_page(evil, [_figure(0, [100, 200, 800, 700], 0)]), page_image, 0)
    mat.finalize()
    # 앱이 만든 파일만 존재
    created = sorted(p.name for p in (out / "images").iterdir())
    assert created == ["0.jpg"]
    assert not (tmp_path / "etc").exists()
    # 악성 참조 문자열은 그대로 텍스트 (렌더 계층이 경로를 재작성/차단)
    assert "../../etc/passwd" in md
    assert "![](images/0.jpg)" in md


def test_unreadable_page_image_degrades_gracefully(tmp_path):
    out = tmp_path / "c"
    bogus = tmp_path / "nope.png"
    bogus.write_bytes(b"not an image")
    mat = ChunkMaterializer(out, single=True)
    md = mat.add_page(_page("본문 [[FIGURE:0]]", [_figure(0, [100, 200, 800, 700], 0)]),
                      bogus, 0)
    assert "본문" in md
    assert "FIGURE" not in md  # crop이 없으니 placeholder 제거
    assert any("페이지 이미지 열기 실패" in w for w in mat.warnings)
