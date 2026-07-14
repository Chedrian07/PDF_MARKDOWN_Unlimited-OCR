"""PDF → 페이지 PNG 렌더링 (pymupdf)."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def quiet_fitz():
    """fitz(pymupdf) 지연 임포트 + MuPDF 에러의 stderr 직접 출력 차단(프로세스 1회).

    MuPDF C 라이브러리는 복구 가능한 파싱 문제(예: 손상 CID 폰트의
    "syntax error: unknown cid font type")를 텍스트 객체마다 stderr에 직접 찍는다
    — 한 페이지에서 수십 줄씩 서버 콘솔을 뒤덮지만 렌더 자체는 폰트 폴백으로
    정상 진행된다(실측: 27p 문서에서 p5 하나가 49줄). 표시만 끄면 동작·예외는
    불변이고 메시지는 내부 버퍼에 계속 쌓이므로, 호출부가 작업 단위로
    drain_mupdf_warnings()로 요약해 로거에 남긴다."""
    import fitz

    if fitz.TOOLS.mupdf_display_errors():
        fitz.TOOLS.mupdf_display_errors(False)
    return fitz


def drain_mupdf_warnings(context: str) -> None:
    """MuPDF 내부 경고 버퍼를 비우고 종류별 건수로 요약해 한 줄 로깅.

    버퍼는 프로세스 전역이라 동시 사용 시 다른 작업의 메시지가 섞일 수 있으나
    (잡 러너는 단일 워커) 진단용 요약이므로 best-effort로 충분하다."""
    try:
        import fitz

        text = fitz.TOOLS.mupdf_warnings()
    except Exception:  # pragma: no cover - 방어적
        return
    if not text:
        return
    counts = Counter(text.splitlines())
    top = [f"{m} (x{c})" if c > 1 else m for m, c in counts.most_common(3)]
    extra = f" 외 {len(counts) - 3}종" if len(counts) > 3 else ""
    logger.info("MuPDF 복구성 경고 %d건 (%s — 처리는 계속됨): %s%s",
                sum(counts.values()), context, " · ".join(top), extra)


def probe_pdf(pdf_path: Path, max_pages: int) -> int:
    """업로드 검증: 열 수 있는 PDF인지 확인하고 페이지 수를 돌려준다.
    문제가 있으면 사용자 메시지를 담은 ValueError."""
    fitz = quiet_fitz()

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        drain_mupdf_warnings("업로드 검증")
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
        drain_mupdf_warnings("업로드 검증")


def _write_blank_page(doc, index: int, path: Path, dpi: int) -> None:
    """렌더 실패 페이지의 대체 흰색 PNG — 페이지 크기를 못 읽으면 A4(pt) 기준."""
    from PIL import Image

    try:
        rect = doc[index].rect
        w_pt, h_pt = float(rect.width), float(rect.height)
    except Exception:  # 페이지 객체 자체가 깨진 경우
        w_pt, h_pt = 595.0, 842.0
    scale = dpi / 72
    size = (max(1, round(w_pt * scale)), max(1, round(h_pt * scale)))
    Image.new("RGB", size, "white").save(path)


def render_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    max_pages: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """모든 페이지를 pages_dir/page_%04d.png (1-based)로 렌더.

    한 페이지가 깨져도(get_pixmap 예외) 잡 전체를 죽이지 않는다 — 흰색 페이지로
    대체하고 계속한다. 전 페이지 실패 시에만 ValueError."""
    fitz = quiet_fitz()

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
        failed = 0
        last_err: Exception | None = None
        for i in range(n):
            p = pages_dir / f"page_{i + 1:04d}.png"
            try:
                pix = doc[i].get_pixmap(matrix=mat)
                pix.save(str(p))
            except Exception as e:  # noqa: BLE001 — 페이지 단위 격리
                failed += 1
                last_err = e
                logger.warning("페이지 %d/%d 렌더 실패 (%s: %s) — 흰색 페이지로 대체",
                               i + 1, n, e.__class__.__name__, str(e)[:200])
                _write_blank_page(doc, i, p, dpi)
            out.append(p)
            if progress_cb:
                progress_cb(i + 1, n)
        if failed == n:
            raise ValueError(f"모든 페이지({n}) 렌더에 실패했습니다: {last_err}") from last_err
        return out
    finally:
        doc.close()
        drain_mupdf_warnings("페이지 렌더")
