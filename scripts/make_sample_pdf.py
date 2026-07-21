#!/usr/bin/env python3
"""이미지·표·텍스트가 섞인 샘플 PDF 생성 (E2E 검증용).

사용법:
    python scripts/make_sample_pdf.py [출력경로=sample/sample.pdf]
    python scripts/make_sample_pdf.py sample/sample-ko.pdf --korean   # 한국어 검증용

의존성: pymupdf, pillow  (backend 환경: cd backend && uv run python ../scripts/make_sample_pdf.py)

`--korean`은 PaddleOCR-VL의 한국어 보존을 검증하기 위한 문서를 만든다 — 한글 음절·
자모·한자 혼용·숫자와 단위·표(셀 줄바꿈 포함)·수식 주변 한글·figure·각주·페이지
번호를 한 페이지에 담는다 (docs/PADDLEOCR_VL_BLACKWELL_5070TI.md §검증 상태).
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


# pymupdf 내장 한국어 폰트 (별도 폰트 파일 불필요)
_KO_FONT = "korea-s"


def _draw_korean_page(page) -> None:
    """한국어 검증 페이지 렌더 (make_korean_pdf·make_scan_pdf 공유 — 단일 진실)."""
    import fitz

    ko = {"fontname": _KO_FONT}
    page.insert_text((72, 90), "2026년 상반기 연구 보고서", fontsize=20, **ko)
    page.insert_text((72, 130), "본 문서는 한국어와 English가 혼용된 문단입니다.", fontsize=12, **ko)
    page.insert_text((72, 152), "한글 자모 ㄱㄴㄷ, 한자 硏究報告書, 숫자 1,234.56 kg 단위를 포함합니다.",
                     fontsize=12, **ko)
    page.insert_text((72, 174), "각주와 표, 수식 주변의 한글 보존을 확인합니다.", fontsize=12, **ko)

    y = 220
    page.insert_text((72, y), "표 1. 분기별 매출 현황", fontsize=13, **ko)
    rows = [
        ("항목", "1분기", "2분기"),
        ("매출액", "1,234", "2,345"),
        ("영업비용", "567", "678"),
        ("순이익", "667", "1,667"),
    ]
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            page.insert_text((78 + ci * 140, y + 28 + ri * 22), cell, fontsize=11, **ko)
    for ri in range(len(rows) + 1):
        page.draw_line(fitz.Point(72, y + 12 + ri * 22), fitz.Point(500, y + 12 + ri * 22))

    page.insert_text((72, 360), "표준편차는 다음 식으로 계산합니다:", fontsize=12, **ko)
    page.insert_text((100, 386), "sigma = sqrt( (1/N) * sum (x_i - mu)^2 )", fontsize=12)
    page.insert_text((72, 412), "여기서 N은 표본 수, mu는 평균입니다.", fontsize=12, **ko)

    page.draw_rect(fitz.Rect(72, 450, 400, 640), color=(0.3, 0.4, 0.8), width=2)
    page.insert_text((90, 480), "그림 영역 (figure)", fontsize=12, **ko)
    for i, h in enumerate((60, 120, 90, 150)):
        page.draw_rect(fitz.Rect(100 + i * 60, 620 - h, 140 + i * 60, 620),
                       color=(0.2, 0.3, 0.7), fill=(0.35, 0.45, 0.85))
    page.insert_text((72, 660), "그림 1. 분기별 매출 추이", fontsize=11, **ko)
    page.insert_text((72, 700), "1) 각주: 본 자료는 검증용으로 생성되었습니다.", fontsize=9, **ko)
    page.insert_text((280, 780), "- 1 -", fontsize=10, **ko)


def make_korean_pdf(out: Path) -> None:
    """한국어 보존 검증용 1페이지 문서 — 외부 폰트·저작권 자료 없이 생성한다."""
    import fitz

    doc = fitz.open()
    _draw_korean_page(doc.new_page(width=595, height=842))
    doc.save(str(out))
    doc.close()


def make_scan_pdf(out: Path) -> None:
    """스캔 문서 시뮬레이션 — **텍스트 레이어 없는 이미지 전용 PDF**.

    한국어 문서를 300dpi로 렌더 → 살짝 기울이고(1.4°) 가우시안 노이즈·명암 저하·
    JPEG 압축(q=55)을 입혀 스캔 결함을 재현한 뒤, 그 이미지만 담은 PDF로 다시
    감싼다. 텍스트 레이어가 없으므로 파이프라인의 내장 텍스트 fallback이 발동하지
    않고 **모델이 실제로 OCR을 수행**하는지 검증한다 (OCR 견고성의 진짜 테스트).
    """
    import io

    import fitz
    from PIL import Image, ImageEnhance

    src = fitz.open()
    tmp_page = src.new_page(width=595, height=842)
    _draw_korean_page(tmp_page)  # 텍스트를 그린 뒤 이미지로 굽는다 (텍스트 레이어 소멸)
    pix = tmp_page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    img = img.rotate(1.4, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    img = ImageEnhance.Contrast(img).enhance(0.82)
    img = ImageEnhance.Brightness(img).enhance(1.06)
    # 가우시안 노이즈 (numpy 없이 표준 라이브러리 난수로)
    import random

    random.seed(20260721)
    px = img.load()
    w, h = img.size
    for _ in range((w * h) // 12):  # 픽셀 일부에 점 노이즈
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        v = random.randint(-45, 45)
        r, g, b = px[x, y]
        px[x, y] = (max(0, min(255, r + v)), max(0, min(255, g + v)), max(0, min(255, b + v)))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=55)
    src.close()

    scan = fitz.open()
    page = scan.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    scan.save(str(out))
    scan.close()


def main() -> None:
    import fitz

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    korean = "--korean" in sys.argv[1:]
    scan = "--scan" in sys.argv[1:]
    default = "sample/sample-scan.pdf" if scan else (
        "sample/sample-ko.pdf" if korean else "sample/sample.pdf")
    out = Path(args[0]) if args else Path(default)
    out.parent.mkdir(parents=True, exist_ok=True)

    if scan:
        make_scan_pdf(out)
        print(f"created: {out} ({out.stat().st_size} bytes, 스캔 시뮬 — 텍스트 레이어 없음)")
        return
    if korean:
        make_korean_pdf(out)
        print(f"created: {out} ({out.stat().st_size} bytes, 한국어 검증용)")
        return

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
