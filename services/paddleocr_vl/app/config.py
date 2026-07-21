"""PaddleOCR-VL-1.6 sidecar 설정 — RTX 5070 Ti 16GB 기본값.

- 모델: PaddlePaddle/PaddleOCR-VL-1.6 (0.9B, BF16) — VRAM 여유가 커서
  layout detector(PP-DocLayout 계열)와 함께 GPU에 올려도 16GB에 충분하다.
  공식 파이프라인은 컴포넌트별 디바이스 분리를 지원하지 않으므로
  (지원되는 분리는 별도 genai-server 배포뿐 — docs/PADDLEOCR_VL_BLACKWELL_5070TI.md)
  PADDLEOCR_DEVICE 하나로 파이프라인 전체 디바이스를 정한다.
- 페이지 동시성 1: sidecar가 요청을 락으로 직렬화한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
# HF api sha (2026-06-05 lastModified 시점) — 업그레이드는 의식적으로만
DEFAULT_MODEL_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
PIPELINE_VERSION = "v1.6"


def _env_int_or_none(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


def _env_str(name: str, default: str) -> str:
    """빈 문자열도 기본값으로 — compose가 `VAR: ${VAR:-}`로 빈 값을 넘겨도
    revision 고정 같은 기본값이 무력화되지 않는다."""
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class PaddleConfig:
    model_id: str
    model_revision: str
    device: str            # "gpu:0" | "cpu"
    min_pixels: int | None
    max_pixels: int | None
    max_upload_mb: int

    @classmethod
    def from_env(cls) -> "PaddleConfig":
        device = os.environ.get("PADDLEOCR_DEVICE", "gpu:0").strip() or "gpu:0"
        if not (device == "cpu" or device.startswith("gpu")):
            raise ValueError(f"PADDLEOCR_DEVICE={device!r} — 'cpu' 또는 'gpu[:N]'만 지원")
        min_px = _env_int_or_none("PADDLEOCR_MIN_PIXELS")
        max_px = _env_int_or_none("PADDLEOCR_MAX_PIXELS")
        if min_px is not None and max_px is not None and max_px < min_px:
            raise ValueError("PADDLEOCR_MAX_PIXELS는 PADDLEOCR_MIN_PIXELS 이상이어야 합니다")
        return cls(
            model_id=_env_str("PADDLEOCR_MODEL_ID", DEFAULT_MODEL_ID),
            model_revision=_env_str("PADDLEOCR_MODEL_REVISION", DEFAULT_MODEL_REVISION),
            device=device,
            min_pixels=min_px,
            max_pixels=max_px,
            # backend의 페이지 렌더 상한(50M px 무손실 PNG)을 수용하는 값 —
            # 30MB였을 때 대형 스캔/도면 페이지가 413으로 청크 전체를 죽였다.
            max_upload_mb=int(os.environ.get("PADDLEOCR_MAX_UPLOAD_MB", "128") or "128"),
        )
