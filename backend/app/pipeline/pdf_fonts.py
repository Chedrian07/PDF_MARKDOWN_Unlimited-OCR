"""layout.json 블록에 원본 PDF 텍스트 레이어의 **실측** 폰트 크기를 주입.

면적 휴리스틱(layout.py `estimate_font_size_cqw`)은 어디까지나 폴백이다. 원본
PDF에 텍스트 레이어가 있으면 그 안의 span 크기를 그대로 읽어 훨씬 정확한
폰트 크기를 심을 수 있다.

동작:
- det bbox(0–999 정규화)를 PDF 페이지의 pt 사각형으로 사상(x_pt = x/999×W,
  y_pt = y/999×H; W·H는 fitz 페이지의 pt 크기). ±3pt 여유로 확장.
- 그 사각형 안에 **중심점**이 드는 span들을 모아 글자수 가중 중앙값 크기를 구하고
  block["fs"] = size_pt / page_width_pt × 100 (cqw = 페이지 폭의 1%)로 심는다.
- 볼드 글자가 과반이면 block["bold"] = True.
- span이 하나도 없거나 텍스트 레이어가 없는 블록은 건드리지 않는다(폴백에 위임).

실패(텍스트 레이어 없음·손상 PDF·페이지 범위 초과)는 조용히 무시한다 —
enrichment은 절대 잡·렌더를 깨뜨리면 안 된다.
"""

from __future__ import annotations

from pathlib import Path

_BOLD_FLAG = 16  # fitz span flags: bit 4 == bold


def _weighted_median(pairs: list[tuple[float, int]]) -> float:
    """(size, chars) 목록의 글자수 가중 중앙값 크기.

    크기 오름차순으로 정렬하고 누적 글자수가 전체의 절반에 처음 도달하는
    지점의 크기를 반환한다 — 큰 폰트의 짧은 조각(각주 번호 등)에 휘둘리지 않게."""
    total = sum(c for _, c in pairs)
    half = total / 2
    acc = 0
    for size, chars in sorted(pairs):
        acc += chars
        if acc >= half:
            return size
    return pairs[-1][0]  # 방어 — 정상 입력에선 도달하지 않음


def _span_is_bold(span: dict) -> bool:
    if int(span.get("flags", 0)) & _BOLD_FLAG:
        return True
    return "bold" in str(span.get("font", "")).lower()


def enrich_layout_fonts(pdf_path: Path, pages: list[dict]) -> bool:
    """layout.json 페이지 블록에 원본 PDF 실측 폰트 크기(cqw)를 주입.

    pages 엔트리: {"page": N(1-based), "width", "height", "blocks": [...]}.
    블록을 제자리(in-place)로 수정하고, 하나라도 주입했으면 True를 돌려준다."""
    try:
        import fitz
    except Exception:
        return False
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return False

    changed = False
    try:
        for page in pages:
            try:
                pno = int(page.get("page", 0))
            except (TypeError, ValueError):
                continue
            if pno < 1 or pno > doc.page_count:  # 페이지 범위 방어
                continue
            fpage = doc[pno - 1]
            pw = float(fpage.rect.width)
            ph = float(fpage.rect.height)
            if pw <= 0 or ph <= 0:
                continue
            try:
                text_dict = fpage.get_text("dict")
            except Exception:
                continue
            # 페이지의 모든 span을 평면화 (bbox·size·text·flags·font)
            spans: list[dict] = []
            for tb in text_dict.get("blocks", ()):
                for line in tb.get("lines", ()):
                    spans.extend(line.get("spans", ()))
            if not spans:
                continue

            for block in page.get("blocks", ()):
                if block.get("image"):
                    continue  # 이미지 블록 — 폰트 크기 없음
                bbox = block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                # 0–999 정규화 → pt, ±3pt 확장
                rx1 = x1 / 999 * pw - 3
                ry1 = y1 / 999 * ph - 3
                rx2 = x2 / 999 * pw + 3
                ry2 = y2 / 999 * ph + 3

                pairs: list[tuple[float, int]] = []
                bold_chars = 0
                total_chars = 0
                for sp in spans:
                    sb = sp.get("bbox")
                    if not sb or len(sb) != 4:
                        continue
                    cx = (sb[0] + sb[2]) / 2
                    cy = (sb[1] + sb[3]) / 2
                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue
                    n = len((sp.get("text") or "").strip())
                    if n <= 0:
                        continue
                    size = float(sp.get("size", 0) or 0)
                    if size <= 0:
                        continue
                    pairs.append((size, n))
                    total_chars += n
                    if _span_is_bold(sp):
                        bold_chars += n

                if not pairs or total_chars <= 0:
                    continue  # 매칭 span 없음 — 블록 미변경(폴백에 위임)
                block["fs"] = _weighted_median(pairs) / pw * 100
                if bold_chars > total_chars * 0.5:
                    block["bold"] = True
                changed = True
    finally:
        doc.close()
    return changed
