"""디바이스/엔진 선택의 단일 진입점. 계약: docs/ARCHITECTURE.md §6"""

from __future__ import annotations

from ..config import Settings
from .base import OCREngine

VALID_DEVICES = ("cpu", "cuda", "metal")
VALID_ENGINES = ("unlimited", "fake", "ovisocr2", "paddleocr_vl")


def build_engine(settings: Settings) -> OCREngine:
    device = settings.device
    if device not in VALID_DEVICES:
        raise ValueError(f"알 수 없는 OCR_DEVICE: {device!r} (사용 가능: {', '.join(VALID_DEVICES)})")

    if settings.engine == "fake":
        from .fake import FakeEngine

        return FakeEngine(device=device, delay=settings.fake_delay)

    if settings.engine in ("ovisocr2", "paddleocr_vl"):
        # GPU는 sidecar 컨테이너 몫 — 이 프로세스는 OCR_DEVICE와 무관하게 GPU를 안 쓴다
        if not settings.sidecar_url:
            raise ValueError(
                f"OCR_ENGINE={settings.engine}에는 OCR_SIDECAR_URL이 필요합니다 "
                "(예: http://ovisocr2:8080 — docker compose --profile ovis|paddle 참조)"
            )
        from .sidecar import SidecarEngine

        return SidecarEngine(settings, name=settings.engine)

    if settings.engine != "unlimited":
        raise ValueError(
            f"알 수 없는 OCR_ENGINE: {settings.engine!r} (사용 가능: {', '.join(VALID_ENGINES)})"
        )

    # torch는 무거우므로 여기서야 임포트
    from .unlimited import UnlimitedEngine

    return UnlimitedEngine(settings)
