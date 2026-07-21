"""공통 Artifact Materializer — normalized 결과를 기존 청크 산출물 규약으로 변환.

sidecar 응답에는 파일 경로·이미지 바이너리가 없다. 이 모듈이 유일하게 파일을
만들며, 파일명은 전부 앱이 생성한다 (모델 출력 경로는 절대 신뢰하지 않는다):

- figure crop:   images/{k}.jpg (single) | images/page_{local}_{k}.jpg (multi)
- 레이아웃 오버레이: result_with_boxes.jpg | result_with_boxes_{local}.jpg
- boxes.json:    {crop파일명: {x1,y1,x2,y2(픽셀), image_width, image_height}}
- raw_pages.json: 기존 pipeline/layout.py::parse_page_blocks가 읽는 그라운딩
  문법을 normalized block에서 **합성**한다. inline det 문법만 사용해
  (`<|det|>label [x1,y1,x2,y2]<|/det|>내용`) 문서 순서 == crop_index 순서를
  보장한다 — 저장에 성공한 image 블록만 문법에 넣는다(크롭 파일과 1:1).

markdown의 `[[FIGURE:n]]` placeholder는 여기서만 `![](images/…)`로 치환된다.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from PIL import Image, ImageDraw

from .protocol import BBOX_MAX, FIGURE_PLACEHOLDER_RE, PageResult

logger = logging.getLogger(__name__)

_MIN_CROP_SIDE_PX = 4       # 이보다 작은 crop은 저장하지 않는다 (퇴화/노이즈)
_JPEG_QUALITY = 90
_OVERLAY_QUALITY = 85

# 레이아웃 오버레이 색 (타입별) — 프론트 라이브 오버레이와 유사한 팔레트
_BOX_COLORS = {
    "title": (219, 68, 55),
    "text": (66, 133, 244),
    "table": (15, 157, 88),
    "formula": (171, 71, 188),
    "image": (244, 180, 0),
    "header": (96, 125, 139),
    "footer": (96, 125, 139),
    "footnote": (121, 85, 72),
    "page_number": (158, 158, 158),
}
_BOX_FALLBACK = (107, 114, 128)


def _norm_to_px(v: int, size: int) -> int:
    """0–999 정규화 좌표 → 픽셀 (벤더 crop_regions와 동일한 절삭 의미론)."""
    return int(v / BBOX_MAX * size)


class ChunkMaterializer:
    """한 청크(작업 디렉터리)의 산출물 생성기. 단일 워커 스레드 전용."""

    def __init__(self, out_dir: Path, single: bool) -> None:
        self.out_dir = out_dir
        self.single = single
        self.images_dir = out_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.boxes: dict[str, dict] = {}
        self.raw_pages: list[str] = []
        self.warnings: list[str] = []

    # ── 내부 ───────────────────────────────────────────────────

    def _crop_name(self, local_page: int, k: int) -> str:
        return f"{k}.jpg" if self.single else f"page_{local_page}_{k}.jpg"

    def _overlay_name(self, local_page: int) -> str:
        return "result_with_boxes.jpg" if self.single else f"result_with_boxes_{local_page}.jpg"

    def _crop_figure(
        self, im: Image.Image, bbox: tuple[int, int, int, int], name: str
    ) -> bool:
        """bbox crop을 name으로 저장. 퇴화/미달 crop은 저장하지 않고 False."""
        w, h = im.size
        x1 = max(0, _norm_to_px(bbox[0], w))
        y1 = max(0, _norm_to_px(bbox[1], h))
        x2 = min(w, _norm_to_px(bbox[2], w))
        y2 = min(h, _norm_to_px(bbox[3], h))
        if x2 - x1 < _MIN_CROP_SIDE_PX or y2 - y1 < _MIN_CROP_SIDE_PX:
            self.warnings.append(f"figure crop 퇴화({x1},{y1},{x2},{y2}) — 건너뜀")
            return False
        im.crop((x1, y1, x2, y2)).save(self.images_dir / name, quality=_JPEG_QUALITY)
        self.boxes[name] = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "image_width": w, "image_height": h,
        }
        return True

    @staticmethod
    def _raw_fragment(btype: str, bbox: tuple[int, int, int, int], content: str) -> str:
        x1, y1, x2, y2 = bbox
        return f"<|det|>{btype} [{x1}, {y1}, {x2}, {y2}]<|/det|>{content}\n"

    # ── 공개 API ───────────────────────────────────────────────

    def add_page(self, page: PageResult, page_image: Path, local_page: int) -> str:
        """페이지 산출물 생성 후, figure 참조가 치환된 markdown을 반환한다.

        page는 protocol.sanitize_page를 통과한 상태를 전제한다 (bbox 검증 완료).
        """
        raw_parts: list[str] = []
        figure_files: dict[int, str] = {}   # figure_index → crop 파일명
        orphan_files: list[str] = []        # placeholder 연결이 없는 crop
        k = 0

        try:
            with Image.open(page_image) as im:
                im = im.convert("RGB")
                overlay = im.copy()
                draw = ImageDraw.Draw(overlay)
                any_box = False
                for b in page.blocks:
                    if b.bbox is None:
                        continue
                    w, h = im.size
                    px = (
                        max(0, _norm_to_px(b.bbox[0], w)),
                        max(0, _norm_to_px(b.bbox[1], h)),
                        min(w - 1, _norm_to_px(b.bbox[2], w)),
                        min(h - 1, _norm_to_px(b.bbox[3], h)),
                    )
                    if px[2] > px[0] and px[3] > px[1]:
                        draw.rectangle(px, outline=_BOX_COLORS.get(b.type, _BOX_FALLBACK), width=4)
                        any_box = True
                    if b.type == "image":
                        name = self._crop_name(local_page, k)
                        if self._crop_figure(im, b.bbox, name):
                            k += 1
                            raw_parts.append(self._raw_fragment("image", b.bbox, ""))
                            if b.figure_index is not None:
                                figure_files[b.figure_index] = name
                            else:
                                orphan_files.append(name)
                    else:
                        raw_parts.append(self._raw_fragment(b.type, b.bbox, b.content))
                if any_box:
                    overlay.save(self.out_dir / self._overlay_name(local_page),
                                 quality=_OVERLAY_QUALITY)
        except OSError as e:
            # 페이지 이미지를 못 읽으면 crop/오버레이 없이 markdown만 진행
            self.warnings.append(f"페이지 이미지 열기 실패({e.__class__.__name__}) — figure 생략")

        self.raw_pages.append("".join(raw_parts))

        used: set[int] = set()

        def _sub(m) -> str:
            idx = int(m.group(1))
            name = figure_files.get(idx)
            if name is None:
                self.warnings.append(f"[[FIGURE:{idx}]] placeholder에 대응하는 figure 없음 — 제거")
                return ""
            used.add(idx)
            return f"![](images/{name})"

        markdown = FIGURE_PLACEHOLDER_RE.sub(_sub, page.markdown)

        # placeholder가 없는 crop은 내용 손실 방지를 위해 페이지 끝에 덧붙인다
        leftovers = [name for idx, name in sorted(figure_files.items()) if idx not in used]
        for name in leftovers + orphan_files:
            markdown = markdown.rstrip() + f"\n\n![](images/{name})"
        if leftovers or orphan_files:
            self.warnings.append(
                f"본문 참조 없는 figure {len(leftovers) + len(orphan_files)}개를 페이지 끝에 추가"
            )
        return markdown.strip()

    def finalize(self) -> None:
        """boxes.json·raw_pages.json 기록 (원자적 교체 — merge와 동일 패턴)."""
        if self.boxes:
            self._atomic_json(self.out_dir / "boxes.json", self.boxes)
        self._atomic_json(self.out_dir / "raw_pages.json", {"pages": self.raw_pages})

    @staticmethod
    def _atomic_json(path: Path, obj: object) -> None:
        tmp = path.parent / f".{path.name}.tmp"
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
