"""모델 없이 파이프라인 전체를 구동하는 가짜 엔진 (테스트/데모용).

실엔진과 동일한 출력 규약을 지킨다 — base.py 모듈 docstring 참조.
figure 크롭은 페이지 좌상단 사분면, 오버레이는 빨간 사각형으로 흉내낸다.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .base import JobCanceled, OCREngine, StreamSink

_PAGE_MD = """## 페이지 {page} — {stem}

이 문서는 FakeEngine이 생성한 데모 출력입니다. 실제 모델을 사용하려면
`OCR_ENGINE=unlimited`로 실행하세요.

| 항목 | 값 |
| --- | --- |
| 페이지 | {page} |
| 원본 | {stem} |

![](images/{img_ref})

본문 텍스트 예시입니다. Unlimited-OCR은 문서의 레이아웃을 인식해 표, 수식,
그림을 마크다운으로 변환합니다."""


class FakeEngine(OCREngine):
    name = "fake"

    def __init__(self, device: str = "cpu", delay: float = 0.02) -> None:
        self.device = device
        self.dtype_name = "none"
        self._delay = delay
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    # ── 내부 유틸 ──────────────────────────────────────────────

    def _emit(self, sink: StreamSink, text: str) -> None:
        # 실제 스트리밍처럼 몇 조각으로 나눠 전달
        step = max(1, len(text) // 4)
        for i in range(0, len(text), step):
            sink.on_text(text[i : i + step])

    def _record_box(self, out_dir: Path, img_name: str, crop_box: tuple, size: tuple) -> None:
        """실엔진(벤더 P13)과 동일한 boxes.json 계약을 흉내낸다."""
        import json

        p = out_dir / "boxes.json"
        data = {}
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        x1, y1, x2, y2 = crop_box
        data[img_name] = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "image_width": size[0], "image_height": size[1],
        }
        p.write_text(json.dumps(data), encoding="utf-8")

    def _write_raw_pages(self, out_dir: Path, raw_pages: list[str]) -> None:
        """벤더 P14의 raw_pages.json 계약 재현 (레이아웃 뷰 E2E용)."""
        import json

        (out_dir / "raw_pages.json").write_text(
            json.dumps({"pages": raw_pages}, ensure_ascii=False), encoding="utf-8"
        )

    def _fake_page(
        self, image_path: Path, out_dir: Path, page_idx: int, img_name: str, overlay_name: str
    ) -> tuple[str, str]:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            crop_box = (w // 8, h // 8, w // 2, h // 2)
            im.crop(crop_box).save(out_dir / "images" / img_name, quality=90)
            self._record_box(out_dir, img_name, crop_box, (w, h))
            overlay = im.copy()
            ImageDraw.Draw(overlay).rectangle(crop_box, outline=(220, 30, 30), width=4)
            overlay.save(out_dir / overlay_name, quality=85)
        md = _PAGE_MD.format(page=page_idx + 1, stem=image_path.stem, img_ref=img_name)
        nx1, ny1, nx2, ny2 = (
            int(crop_box[0] / w * 999), int(crop_box[1] / h * 999),
            int(crop_box[2] / w * 999), int(crop_box[3] / h * 999),
        )
        raw = (
            f"<|det|>title [60, 30, 700, 80]<|/det|>페이지 {page_idx + 1} — {image_path.stem}\n"
            f"<|det|>image [{nx1}, {ny1}, {nx2}, {ny2}]<|/det|>\n"
            f"<|det|>text [60, 620, 930, 820]<|/det|>본문 텍스트 예시입니다. FakeEngine 레이아웃 블록."
        )
        return md, raw

    # ── OCREngine 구현 ─────────────────────────────────────────

    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        parts: list[str] = []
        raws: list[str] = []
        for i, p in enumerate(image_paths):
            if cancel.is_set():
                raise JobCanceled()
            md, raw = self._fake_page(p, out_dir, i, f"page_{i}_0.jpg", f"result_with_boxes_{i}.jpg")
            sink.on_text("<PAGE>\n")
            self._emit(sink, md)
            sink.on_text("\n")
            parts.append(md)
            raws.append(raw)
            time.sleep(self._delay)
        self._write_raw_pages(out_dir, raws)
        return "<PAGE>\n" + "\n<PAGE>\n".join(parts)

    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        if cancel.is_set():
            raise JobCanceled()
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        md, raw = self._fake_page(image_path, out_dir, 0, "0.jpg", "result_with_boxes.jpg")
        self._write_raw_pages(out_dir, [raw])
        self._emit(sink, md)
        time.sleep(self._delay)
        return md
