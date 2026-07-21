"""OvisOCR2 sidecar 설정 — RTX 5070 Ti 16GB 보수적 기본값.

기본값 근거 (docs/OVISOCR2_CUDA_5070TI.md):
- gpu_memory_utilization 0.80: 데스크톱 디스플레이/WSL 오버헤드 몫을 남긴다 (0.95+ 금지)
- max_model_len 24576: 기본 max_pixels(2880²)에서 비전 토큰이 ~8K까지 나오므로
  출력 상한(8192)과 프롬프트를 더해도 넘치지 않는 최소 여유. 0.9B 하이브리드
  어텐션 모델이라 KV 비용은 미미하다.
- max_num_seqs 1 / 페이지 동시성 1: 단일 GPU에서 예측 가능한 VRAM 사용.
- gdn_prefill_backend "triton": 컨슈머 Blackwell(sm_120)에서는 FlashInfer GDN
  경로가 데이터센터 아키텍처로 게이트되어 있어 triton(FLA 커널)이 동작 경로다
  (모델 카드 공식 예제와 동일).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# 모델 카드 공식 revision 고정 (2026-07-16 확인) — 업그레이드는 의식적으로만
DEFAULT_MODEL_ID = "ATH-MaaS/OvisOCR2"
DEFAULT_MODEL_REVISION = "65c619d374b55d4152e85150fc1b003700bc1f0c"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _env_str(name: str, default: str) -> str:
    """빈 문자열도 기본값으로 — compose가 `VAR: ${VAR:-}`로 빈 값을 넘겨도
    revision 고정 같은 기본값이 무력화되지 않는다."""
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class OvisConfig:
    model_id: str
    model_revision: str
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    max_output_tokens: int
    max_num_seqs: int
    min_pixels: int
    max_pixels: int
    gdn_prefill_backend: str
    max_upload_mb: int

    @classmethod
    def from_env(cls) -> "OvisConfig":
        cfg = cls(
            model_id=_env_str("OVIS_MODEL_ID", DEFAULT_MODEL_ID),
            model_revision=_env_str("OVIS_MODEL_REVISION", DEFAULT_MODEL_REVISION),
            dtype=_env_str("OVIS_DTYPE", "bfloat16"),
            gpu_memory_utilization=_env_float("OVIS_GPU_MEMORY_UTILIZATION", 0.80),
            max_model_len=_env_int("OVIS_MAX_MODEL_LEN", 24576),
            max_output_tokens=_env_int("OVIS_MAX_OUTPUT_TOKENS", 8192),
            max_num_seqs=_env_int("OVIS_MAX_NUM_SEQS", 1),
            # 모델 카드 공식 값: min 448², max 2880²
            min_pixels=_env_int("OVIS_MIN_PIXELS", 448 * 448),
            max_pixels=_env_int("OVIS_MAX_PIXELS", 2880 * 2880),
            gdn_prefill_backend=_env_str("OVIS_GDN_PREFILL_BACKEND", "triton"),
            # backend는 페이지당 MAX_RENDER_PIXELS(50M px)까지 렌더하고 무손실 PNG로
            # 보낸다 — 도면/대형 스캔 페이지는 30MB를 쉽게 넘어 413(청크 전멸)이 됐다.
            max_upload_mb=_env_int("OVIS_MAX_UPLOAD_MB", 128),
        )
        if not 0.3 <= cfg.gpu_memory_utilization <= 0.92:
            raise ValueError(
                f"OVIS_GPU_MEMORY_UTILIZATION={cfg.gpu_memory_utilization} — "
                "0.3~0.92 범위여야 합니다 (16GB 단일 GPU에서 0.95+는 디스플레이/런타임 몫 침범)"
            )
        if cfg.min_pixels <= 0 or cfg.max_pixels < cfg.min_pixels:
            raise ValueError("OVIS_MIN_PIXELS/OVIS_MAX_PIXELS 값이 올바르지 않습니다")
        if cfg.max_output_tokens <= 0 or cfg.max_model_len <= cfg.max_output_tokens:
            raise ValueError("OVIS_MAX_MODEL_LEN은 OVIS_MAX_OUTPUT_TOKENS보다 커야 합니다")
        return cfg
