"""청크 단위 모델 출력을 잡 단위 최종 결과로 병합.

입력 규약(엔진 출력, base.py 참조):
- multi 청크: `<PAGE>` 구분 마크다운 + chunk_dir/images/page_{i}_{k}.jpg
  + chunk_dir/result_with_boxes_{i}.jpg  (i는 청크 내 0-based)
- single 청크: 페이지 1장 마크다운 + chunk_dir/images/{k}.jpg
  + chunk_dir/result_with_boxes.jpg

출력(잡 디렉터리, ARCHITECTURE.md §4):
- images/p{글로벌페이지:04d}_{k}.jpg  (1-based)
- layout/page_{글로벌페이지:04d}.jpg
- result.md — 페이지들을 page_separator로 join (청크 완료 시마다 부분 갱신)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_IMG_MULTI = re.compile(r"!\[\]\(images/page_(\d+)_(\d+)\.jpg\)")
_IMG_SINGLE = re.compile(r"!\[\]\(images/(\d+)\.jpg\)")
_FILE_MULTI = re.compile(r"^page_(\d+)_(\d+)\.jpg$")
_FILE_SINGLE = re.compile(r"^(\d+)\.jpg$")
_BOXES_FILE = re.compile(r"^result_with_boxes_(\d+)\.jpg$")
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]{0,64}\|>")


def split_pages(markdown: str) -> list[str]:
    """`<PAGE>` 마커로 분리. 첫 마커 이전의 공백은 버린다."""
    parts = markdown.split("<PAGE>")
    if parts and not parts[0].strip():
        parts = parts[1:]
    return [p.strip() for p in parts]


def _global_image_name(global_page: int, k: int | str) -> str:
    return f"p{global_page:04d}_{k}.jpg"


def _clean(page_md: str) -> str:
    page_md = _SPECIAL_TOKEN.sub("", page_md)
    return page_md.strip()


@dataclass
class ChunkResult:
    chunk_dir: Path
    start_page: int  # 글로벌 1-based
    num_pages: int
    markdown: str
    single: bool = False


class IncrementalMerger:
    def __init__(self, job_dir: Path, page_separator: str) -> None:
        self.job_dir = job_dir
        self.page_separator = page_separator
        self.images_dir = job_dir / "images"
        self.layout_dir = job_dir / "layout"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.layout_dir.mkdir(parents=True, exist_ok=True)
        self.pages_md: list[str] = []
        self.warnings: list[str] = []
        # 글로벌 이미지명 → figure bbox 메타 (벤더 P13의 boxes.json — 렌더 폭 계산용)
        self.figure_boxes: dict[str, dict] = {}
        # 레이아웃 뷰용 페이지 블록 (벤더 P14의 raw_pages.json → layout.json)
        self.layout_pages: list[dict] = []

    # ── 파일 이동 ──────────────────────────────────────────────

    def _load_chunk_boxes(self, chunk: ChunkResult) -> dict:
        p = chunk.chunk_dir / "boxes.json"
        if not p.is_file():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _move_chunk_files(self, chunk: ChunkResult) -> None:
        chunk_boxes = self._load_chunk_boxes(chunk)
        img_src = chunk.chunk_dir / "images"
        if img_src.is_dir():
            for f in sorted(img_src.iterdir()):
                if chunk.single:
                    m = _FILE_SINGLE.match(f.name)
                    if not m:
                        continue
                    dest = self.images_dir / _global_image_name(chunk.start_page, m.group(1))
                else:
                    m = _FILE_MULTI.match(f.name)
                    if not m:
                        continue
                    local_page, k = int(m.group(1)), m.group(2)
                    dest = self.images_dir / _global_image_name(chunk.start_page + local_page, k)
                meta = chunk_boxes.get(f.name)
                if isinstance(meta, dict):
                    self.figure_boxes[dest.name] = meta
                shutil.move(str(f), str(dest))
        self._write_boxes()

        if chunk.single:
            boxes = chunk.chunk_dir / "result_with_boxes.jpg"
            if boxes.is_file():
                shutil.move(str(boxes), str(self.layout_dir / f"page_{chunk.start_page:04d}.jpg"))
        else:
            for f in sorted(chunk.chunk_dir.iterdir()):
                m = _BOXES_FILE.match(f.name)
                if m:
                    g = chunk.start_page + int(m.group(1))
                    shutil.move(str(f), str(self.layout_dir / f"page_{g:04d}.jpg"))

    def _write_boxes(self) -> None:
        if self.figure_boxes:
            (self.images_dir / "boxes.json").write_text(
                json.dumps(self.figure_boxes, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    # ── 레이아웃 뷰 (Phase B — 부가 산출물, 결측 시 조용히 스킵) ─────────

    def _page_size(self, global_page: int) -> tuple[int, int]:
        p = self.job_dir / "pages" / f"page_{global_page:04d}.png"
        try:
            from PIL import Image

            with Image.open(p) as im:
                return im.size
        except Exception:
            return (1000, 1414)  # A4 비율 폴백

    def _ingest_layout(self, chunk: ChunkResult) -> None:
        raw_path = chunk.chunk_dir / "raw_pages.json"
        if not raw_path.is_file():
            return
        from .layout import parse_page_blocks

        try:
            raw_pages = json.loads(raw_path.read_text(encoding="utf-8"))["pages"]
        except Exception:
            return
        new_pages: list[dict] = []
        for local, raw in enumerate(raw_pages[: chunk.num_pages]):
            g = chunk.start_page + (0 if chunk.single else local)
            blocks = parse_page_blocks(str(raw))
            for b in blocks:
                if "crop_index" in b:
                    # 벤더 크롭 순서 == boxes/이미지 저장 순서 → 글로벌 이미지명 매핑
                    b["image"] = _global_image_name(g, b.pop("crop_index"))
            w, h = self._page_size(g)
            page = {"page": g, "width": w, "height": h, "blocks": blocks}
            new_pages.append(page)
            self.layout_pages.append(page)
        # 원본 PDF 텍스트 레이어의 실측 폰트 크기를 이번 청크 페이지들에 주입.
        # (청크마다 pdf 재오픈 — ms 수준이라 무방. enrichment 실패는 잡을 깨지 않음.)
        try:
            from .pdf_fonts import enrich_layout_fonts

            enrich_layout_fonts(self.job_dir / "source.pdf", new_pages)
        except Exception:
            pass
        if self.layout_pages:
            (self.job_dir / "layout.json").write_text(
                json.dumps(self.layout_pages, ensure_ascii=False), encoding="utf-8"
            )

    # ── 마크다운 재작성 ────────────────────────────────────────

    def _rewrite_refs(self, page_md: str, chunk: ChunkResult) -> str:
        if chunk.single:
            return _IMG_SINGLE.sub(
                lambda m: f"![](images/{_global_image_name(chunk.start_page, m.group(1))})",
                page_md,
            )
        return _IMG_MULTI.sub(
            lambda m: f"![](images/{_global_image_name(chunk.start_page + int(m.group(1)), m.group(2))})",
            page_md,
        )

    # ── 공개 API ───────────────────────────────────────────────

    def add_chunk(self, chunk: ChunkResult) -> None:
        pages = [chunk.markdown] if chunk.single else split_pages(chunk.markdown)

        if len(pages) > chunk.num_pages:
            # 마커가 초과 생성됨 — 초과분을 마지막 페이지에 합침
            self.warnings.append(
                f"{chunk.start_page}페이지 청크: 페이지 마커 {len(pages)}개 (기대 {chunk.num_pages}) — 초과분 병합"
            )
            head = pages[: chunk.num_pages - 1]
            tail = "\n\n".join(pages[chunk.num_pages - 1 :])
            pages = head + [tail]
        elif len(pages) < chunk.num_pages:
            self.warnings.append(
                f"{chunk.start_page}페이지 청크: 페이지 마커 {len(pages)}개 (기대 {chunk.num_pages}) — 빈 페이지로 보정"
            )
            pages = pages + [""] * (chunk.num_pages - len(pages))

        pages = [self._rewrite_refs(p, chunk) for p in pages]
        self._move_chunk_files(chunk)
        self._ingest_layout(chunk)
        self.pages_md.extend(_clean(p) for p in pages)
        self._write_partial()

    @property
    def markdown(self) -> str:
        return self.page_separator.join(self.pages_md).strip() + "\n"

    def _write_partial(self) -> None:
        tmp = self.job_dir / ".result.md.tmp"
        tmp.write_text(self.markdown, encoding="utf-8")
        os.replace(tmp, self.job_dir / "result.md")

    def finalize(self) -> str:
        self._write_partial()
        return self.markdown
