from pathlib import Path

from app.pipeline.merge import ChunkResult, IncrementalMerger, split_pages

SEP = "\n\n---\n\n"


def _touch(p: Path, data: bytes = b"jpg") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _mk_multi_chunk(root: Path, name: str, num_pages: int, images_per_page: int = 1) -> Path:
    d = root / "work" / name
    for i in range(num_pages):
        for k in range(images_per_page):
            _touch(d / "images" / f"page_{i}_{k}.jpg")
        _touch(d / f"result_with_boxes_{i}.jpg")
    return d


def test_split_pages():
    assert split_pages("<PAGE>\nA\n<PAGE>\nB") == ["A", "B"]
    assert split_pages("A only") == ["A only"]
    assert split_pages("") == []
    assert split_pages("<PAGE>") == [""]


def test_multi_chunk_renumbering(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    c0 = _mk_multi_chunk(tmp_path, "chunk_00", 2)
    m.add_chunk(ChunkResult(c0, 1, 2, "<PAGE>\nA ![](images/page_0_0.jpg)\n<PAGE>\nB ![](images/page_1_0.jpg)"))
    c1 = _mk_multi_chunk(tmp_path, "chunk_01", 1)
    m.add_chunk(ChunkResult(c1, 3, 1, "<PAGE>\nC ![](images/page_0_0.jpg)"))
    out = m.finalize()

    assert "A ![](images/p0001_0.jpg)" in out
    assert "B ![](images/p0002_0.jpg)" in out
    assert "C ![](images/p0003_0.jpg)" in out
    assert out.count("---") == 2
    for name in ("p0001_0.jpg", "p0002_0.jpg", "p0003_0.jpg"):
        assert (tmp_path / "images" / name).is_file()
    for name in ("page_0001.jpg", "page_0002.jpg", "page_0003.jpg"):
        assert (tmp_path / "layout" / name).is_file()
    assert (tmp_path / "result.md").read_text(encoding="utf-8") == out
    assert m.warnings == []


def test_marker_count_mismatch_pads_and_warns(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 3)
    m.add_chunk(ChunkResult(c, 1, 3, "<PAGE>\nonly one page"))
    assert len(m.pages_md) == 3
    assert m.pages_md[1] == "" and m.pages_md[2] == ""
    assert len(m.warnings) == 1


def test_marker_count_excess_merges_tail(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 1)
    m.add_chunk(ChunkResult(c, 1, 1, "<PAGE>\nA\n<PAGE>\nB\n<PAGE>\nC"))
    assert len(m.pages_md) == 1
    assert "A" in m.pages_md[0] and "C" in m.pages_md[0]
    assert len(m.warnings) == 1


def test_single_mode_merge(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    d = tmp_path / "work" / "chunk_04"
    _touch(d / "images" / "0.jpg")
    _touch(d / "images" / "1.jpg")
    _touch(d / "result_with_boxes.jpg")
    md = "P5 ![](images/0.jpg) and ![](images/1.jpg)"
    m.add_chunk(ChunkResult(d, 5, 1, md, single=True))
    out = m.finalize()
    assert "![](images/p0005_0.jpg)" in out and "![](images/p0005_1.jpg)" in out
    assert (tmp_path / "images" / "p0005_0.jpg").is_file()
    assert (tmp_path / "images" / "p0005_1.jpg").is_file()
    assert (tmp_path / "layout" / "page_0005.jpg").is_file()


def test_special_tokens_stripped(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    d = tmp_path / "work" / "chunk_00"
    d.mkdir(parents=True)
    m.add_chunk(ChunkResult(d, 1, 1, "<PAGE>\nkeep <|ref|>text<|/ref|><|det|>[[1,2,3,4]]<|/det|> this"))
    out = m.finalize()
    assert "<|ref|>" not in out and "<|det|>" not in out
    assert "keep" in out and "this" in out
