"""PDF → 페이지 PNG 렌더링 (pymupdf)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def probe_pdf(pdf_path: Path, max_pages: int) -> int:
    """업로드 검증: 열 수 있는 PDF인지 확인하고 페이지 수를 돌려준다.
    문제가 있으면 사용자 메시지를 담은 ValueError."""
    import fitz

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"PDF를 열 수 없습니다: {e}") from e
    try:
        if doc.needs_pass:
            raise ValueError("암호화된 PDF는 지원하지 않습니다")
        n = doc.page_count
        if n == 0:
            raise ValueError("페이지가 없는 PDF입니다")
        if n > max_pages:
            raise ValueError(f"페이지 수({n})가 상한({max_pages})을 초과합니다")
        return n
    finally:
        doc.close()


def render_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    max_pages: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """모든 페이지를 pages_dir/page_%04d.png (1-based)로 렌더."""
    import fitz

    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass:
            raise ValueError("암호화된 PDF는 지원하지 않습니다")
        n = doc.page_count
        if n == 0:
            raise ValueError("페이지가 없는 PDF입니다")
        if n > max_pages:
            raise ValueError(f"페이지 수({n})가 상한({max_pages})을 초과합니다")
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        out: list[Path] = []
        for i in range(n):
            pix = doc[i].get_pixmap(matrix=mat)
            p = pages_dir / f"page_{i + 1:04d}.png"
            pix.save(str(p))
            out.append(p)
            if progress_cb:
                progress_cb(i + 1, n)
        return out
    finally:
        doc.close()
