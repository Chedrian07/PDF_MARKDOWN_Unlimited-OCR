"""환경변수 기반 설정. 계약: docs/ARCHITECTURE.md §7"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_REVISION = "ee63731b6461c8afcdcc7b15352e7d2ffecc2ead"
_DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1"


def load_dotenv_file(path: Path | None = None) -> None:
    """리포 루트 .env를 os.environ에 주입 — **이미 설정된 키는 건드리지 않는다**.

    docker-compose는 .env를 읽어 environment로 넘기지만(그 값이 우선 유지됨),
    로컬 실행(macOS Metal 등)은 아무도 .env를 읽지 않아 번역 프로바이더가
    503("프로바이더 미설정")으로 떨어졌다 — CPU/CUDA/Metal 범용성 결함 수정.
    파서는 KEY=VALUE 한 줄 형식만 지원하고 주석(#)·빈 줄을 건너뛰며,
    compose와 동일하게 값 양끝 따옴표를 벗긴다.
    """
    if path is None:
        for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
            cand = base / ".env"
            if cand.is_file():
                path = cand
                break
    if path is None or not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k:
            os.environ.setdefault(k, v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _split_hosts(v: str) -> list[str]:
    return [h.strip() for h in v.split(",") if h.strip()]


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
    fast_decode: bool = True            # 커스텀 그리디 디코드 루프 (0이면 HF generate 폴백)
    decode_block: int = 8               # fast_decode의 호스트 동기화 배칭 크기(토큰)
    fake_delay: float = 0.02            # FakeEngine 페이지당 지연(초)
    job_ttl_days: int = 0               # 터미널 잡(done/error/canceled) 자동 GC 보존 일수 — 0=비활성(opt-in)
    # Host 헤더 화이트리스트 (DNS rebinding 방어) — 포트는 비교 시 무시됨 (localhost:8000 → localhost)
    allowed_hosts: list[str] = field(default_factory=lambda: _split_hosts(_DEFAULT_ALLOWED_HOSTS))

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_file()  # 로컬 실행(Metal 등)에서도 .env의 번역/OCR 설정이 잡히게
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
            fast_decode=_env_bool("OCR_FAST_DECODE", True),
            decode_block=_env_int("OCR_DECODE_BLOCK", 8),
            fake_delay=float(os.environ.get("FAKE_DELAY", "0.02")),
            job_ttl_days=_env_int("JOB_TTL_DAYS", 0),
            allowed_hosts=_split_hosts(os.environ.get("ALLOWED_HOSTS") or _DEFAULT_ALLOWED_HOSTS),
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
