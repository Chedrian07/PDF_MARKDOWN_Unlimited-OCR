#!/usr/bin/env python3
"""이미지·표·텍스트가 섞인 샘플 PDF 생성 (E2E 검증용).

사용법: python scripts/make_sample_pdf.py [출력경로=sample/sample.pdf]
의존성: pymupdf, pillow  (backend 환경: cd backend && uv run python ../scripts/make_sample_pdf.py)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path


def make_chart_png() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 360), (250, 250, 252))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 639, 359), outline=(180, 180, 190), width=2)
    d.text((24, 16), "Quarterly Revenue (Sample Chart)", fill=(30, 30, 40))
    bars = [(90, 130), (170, 200), (250, 90), (330, 260), (410, 180), (490, 300)]
    for x, h in bars:
        d.rectangle((x, 330 - h, x + 56, 330), fill=(72, 98, 214), outline=(40, 60, 160))
    d.line((60, 330, 600, 330), fill=(60, 60, 70), width=2)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def make_photo_png() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (480, 320), (18, 32, 58))
    d = ImageDraw.Draw(img)
    d.ellipse((300, 30, 380, 110), fill=(250, 240, 180))
    d.polygon([(0, 320), (140, 150), (260, 320)], fill=(52, 84, 60))
    d.polygon([(180, 320), (330, 120), (480, 320)], fill=(70, 110, 78))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def main() -> None:
    import fitz

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample/sample.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()

    # ── 1페이지: 제목 + 본문 + 차트 이미지 + 표 ──
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 84), "Unlimited-OCR End-to-End Sample", fontsize=22)
    page.insert_text((72, 112), "This document validates PDF to Markdown conversion,", fontsize=11)
    page.insert_text((72, 128), "including embedded figure extraction and table structure.", fontsize=11)
    page.insert_image(fitz.Rect(72, 150, 520, 402), stream=make_chart_png())
    y = 430
    page.insert_text((72, y), "Table 1. Model configurations", fontsize=12)
    rows = [
        ("Mode", "base_size", "image_size", "crop"),
        ("gundam", "1024", "640", "true"),
        ("base", "1024", "1024", "false"),
    ]
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            page.insert_text((72 + ci * 110, y + 24 + ri * 20), cell, fontsize=10)
    for ri in range(len(rows) + 1):
        page.draw_line(fitz.Point(70, y + 8 + ri * 20 + 2), fitz.Point(510, y + 8 + ri * 20 + 2))
    page.insert_text(
        (72, 580),
        "The pricing follows E = mc^2 style notation and includes 3.3B parameters.",
        fontsize=11,
    )

    # ── 2페이지: 두 번째 이미지 + 리스트 ──
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 84), "Section 2: Figures and Lists", fontsize=18)
    page2.insert_image(fitz.Rect(72, 110, 430, 350), stream=make_photo_png())
    bullets = [
        "First bullet item about long-horizon parsing.",
        "Second bullet item about page markers.",
        "Third bullet item about figure grounding boxes.",
    ]
    for i, b in enumerate(bullets):
        page2.insert_text((84, 390 + i * 22), f"•  {b}", fontsize=11)
    page2.insert_text((72, 480), "End of sample document. Page 2 of 2.", fontsize=10)

    doc.save(str(out))
    doc.close()
    print(f"created: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
