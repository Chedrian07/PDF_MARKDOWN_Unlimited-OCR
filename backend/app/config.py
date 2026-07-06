"""환경변수 기반 설정. 계약: docs/ARCHITECTURE.md §7"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_REVISION = "ee63731b6461c8afcdcc7b15352e7d2ffecc2ead"


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


@dataclass
class Settings:
    device: str = "cpu"                 # cpu | cuda | metal (mps는 metal의 별칭)
    dtype: str = "auto"                 # auto | bfloat16 | float16 | float32
    engine: str = "unlimited"           # unlimited | fake
    model_id: str = "baidu/Unlimited-OCR"
    model_revision: str = _DEFAULT_REVISION
    preload_model: bool = True
    data_dir: Path = field(default_factory=lambda: Path("data"))
    frontend_dir: Path | None = None    # None이면 리포 상대 경로에서 탐색
    render_dpi: int = 200
    pages_per_chunk: int = 8
    max_pages: int = 200
    max_upload_mb: int = 100
    max_length: int = 32768
    page_separator: str = "\n\n---\n\n"
    cpu_threads: int = 0                # 0=torch 기본값 (CPU 백엔드 전용)
    fake_delay: float = 0.02            # FakeEngine 페이지당 지연(초)

    @classmethod
    def from_env(cls) -> "Settings":
        sep = os.environ.get("PAGE_SEPARATOR")
        frontend = os.environ.get("FRONTEND_DIR")
        device = os.environ.get("OCR_DEVICE", "cpu").strip().lower()
        return cls(
            device="metal" if device == "mps" else device,
            dtype=os.environ.get("OCR_DTYPE", "auto").strip().lower(),
            engine=os.environ.get("OCR_ENGINE", "unlimited").strip().lower(),
            model_id=os.environ.get("MODEL_ID", "baidu/Unlimited-OCR"),
            model_revision=os.environ.get("MODEL_REVISION", _DEFAULT_REVISION),
            preload_model=_env_bool("PRELOAD_MODEL", True),
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
            frontend_dir=Path(frontend) if frontend else None,
            render_dpi=_env_int("RENDER_DPI", 200),
            pages_per_chunk=_env_int("PAGES_PER_CHUNK", 8),
            max_pages=_env_int("MAX_PAGES", 200),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 100),
            max_length=_env_int("MAX_LENGTH", 32768),
            page_separator=sep.encode().decode("unicode_escape") if sep else "\n\n---\n\n",
            cpu_threads=_env_int("OCR_CPU_THREADS", 0),
            fake_delay=float(os.environ.get("FAKE_DELAY", "0.02")),
        )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolve_frontend_dir(self) -> Path | None:
        if self.frontend_dir is not None:
            return self.frontend_dir if self.frontend_dir.is_dir() else None
        candidate = Path(__file__).resolve().parents[2] / "frontend"
        return candidate if candidate.is_dir() else None
