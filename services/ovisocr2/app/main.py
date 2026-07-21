"""OvisOCR2 sidecar — 내부 프로토콜 v1 서버 (docs/OCR_ENGINE_PROTOCOL.md).

GET  /health   : 프로토콜 health (모델 미로드 시에도 200 — model_loaded로 구분)
POST /v1/parse : 페이지 이미지 1장 → normalized page (figure는 [[FIGURE:n]] placeholder)

응답에는 이미지 바이너리·로컬 경로·토큰이 없다. crop은 메인 backend가 수행한다.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .config import OvisConfig
from .model import OvisModel
from .parser import parse_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
ENGINE_NAME = "ovisocr2"
_MAX_IMAGE_PIXELS = 60_000_000  # 업로드 이미지 픽셀 폭탄 방어
_PROVIDER_RAW_CAP = 100_000

# PIL의 폭탄 임계값도 같은 값으로 낮춘다 — 1차 방어는 아래 헤더 검사이고, 이건
# 헤더를 속인 파일에 대한 2차 그물이다 (PIL은 이 값 초과에서 경고, 2배 초과에서
# DecompressionBombError). 기본값 ~89M보다 엄격해진다.
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS

cfg = OvisConfig.from_env()
model = OvisModel(cfg)


class ParseOptions(BaseModel):
    """제한된 요청 옵션 — 알 수 없는 키는 거부한다."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    max_pixels: int | None = Field(default=None, ge=64 * 64, le=6000 * 6000)
    max_output_tokens: int | None = Field(default=None, ge=64, le=32768)


def _load_in_background() -> None:
    try:
        model.load()
    except Exception:  # noqa: BLE001 — load_error로 health에 노출됨
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load_in_background, name="model-load", daemon=True).start()
    yield


app = FastAPI(title="OvisOCR2 sidecar", lifespan=lifespan)


def _gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_name": None, "gpu_total_mb": None, "gpu_free_mb": None}
        free_b, total_b = torch.cuda.mem_get_info(0)
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_mb": total_b // (1024 * 1024),
            "gpu_free_mb": free_b // (1024 * 1024),
        }
    except Exception:  # noqa: BLE001 — health는 항상 응답해야 한다
        return {"gpu_name": None, "gpu_total_mb": None, "gpu_free_mb": None}


def _runtime_version() -> str:
    try:
        import vllm

        return getattr(vllm, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unavailable"


@app.get("/health")
def health() -> dict:
    return {
        "status": "error" if model.load_error else "ok",
        "protocol_version": PROTOCOL_VERSION,
        "engine": ENGINE_NAME,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "runtime": "vllm",
        "runtime_version": _runtime_version(),
        "device": "cuda",
        "dtype": cfg.dtype,
        "model_loaded": model.loaded,
        "load_error": model.load_error,
        **_gpu_info(),
    }


def _decode_image(data: bytes) -> Image.Image:
    """형식·픽셀 수를 **디코드 전에** 헤더로 검증한 뒤에만 실제 디코드한다.

    Image.open은 헤더만 읽으므로(lazy) 이 순서라야 픽셀 폭탄이 메모리를 잡기 전에
    거부된다 — load() 후에 검사하면 이미 수 GB를 할당한 뒤다."""
    if len(data) > cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"이미지가 상한({cfg.max_upload_mb}MB)을 초과합니다")
    try:
        im = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, "이미지를 디코드할 수 없습니다") from e
    if im.format not in ("PNG", "JPEG"):
        raise HTTPException(400, f"지원하지 않는 이미지 형식: {im.format}")
    if im.width * im.height > _MAX_IMAGE_PIXELS:
        raise HTTPException(400, "이미지 픽셀 수가 상한을 초과합니다")
    try:
        im.load()
    except Exception as e:
        raise HTTPException(400, "이미지를 디코드할 수 없습니다") from e
    return im.convert("RGB")


def _is_oom(e: Exception) -> bool:
    name = e.__class__.__name__
    return "OutOfMemory" in name or "out of memory" in str(e).lower()


@app.post("/v1/parse")
def parse(
    file: UploadFile = File(...),
    page_index: int = Form(0),
    request_id: str = Form(""),
    options: str = Form("{}"),
) -> dict:
    if not model.loaded:
        detail = model.load_error or "모델이 아직 로드되지 않았습니다"
        raise HTTPException(503, detail)
    try:
        opts_raw = json.loads(options or "{}")
        opts = ParseOptions.model_validate(opts_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"options 스키마 위반: {e}") from e

    t0 = time.monotonic()
    data = file.file.read(cfg.max_upload_mb * 1024 * 1024 + 1)
    image = _decode_image(data)
    t1 = time.monotonic()

    warnings: list[str] = []
    max_pixels = opts.max_pixels or cfg.max_pixels
    # 옵션은 상한을 **낮추는 방향으로만** 유효하다 — 설정 상한을 넘기면
    # max_model_len 예산을 깨고 vLLM이 요청 자체를 거부한다.
    max_output_tokens = min(opts.max_output_tokens or cfg.max_output_tokens,
                            cfg.max_output_tokens)
    try:
        raw = model.infer(image, max_pixels=max_pixels, max_output_tokens=max_output_tokens)
    except Exception as e:  # noqa: BLE001 — OOM만 1회 강등 재시도, 그 외 502
        if not _is_oom(e):
            logger.exception("추론 실패 (req=%s)", request_id[:64])
            raise HTTPException(502, f"추론 실패: {e.__class__.__name__}") from e
        model.release_cache()
        # 강등은 반드시 **더 작은** 해상도여야 한다 — 요청이 이미 min_pixels 근처면
        # 낮출 여지가 없으므로 재시도하지 않고 즉시 명확한 오류를 낸다.
        reduced = max(cfg.min_pixels, max_pixels // 2)
        if reduced >= max_pixels:
            logger.warning("OOM (req=%s) — 더 낮출 해상도가 없어 재시도 생략", request_id[:64])
            raise HTTPException(
                502,
                "GPU 메모리 부족 — 이미 최소 해상도입니다. OVIS_MAX_MODEL_LEN/"
                "OVIS_MAX_OUTPUT_TOKENS를 낮추거나 다른 프로세스의 VRAM 점유를 확인하세요",
            ) from e
        logger.warning(
            "OOM (req=%s) — max_pixels %d→%d로 강등 후 1회 재시도",
            request_id[:64], max_pixels, reduced,
        )
        try:
            raw = model.infer(image, max_pixels=reduced, max_output_tokens=max_output_tokens)
            warnings.append(f"GPU 메모리 부족으로 해상도 강등(max_pixels {max_pixels}→{reduced})")
        except Exception as e2:  # noqa: BLE001 — 재시도 실패는 명확한 오류로
            model.release_cache()
            logger.exception("OOM 재시도 실패 (req=%s)", request_id[:64])
            raise HTTPException(
                502, "GPU 메모리 부족 — OVIS_MAX_PIXELS를 낮추거나 문서 해상도를 줄이세요"
            ) from e2
    t2 = time.monotonic()

    page = parse_page(raw)
    t3 = time.monotonic()

    logger.info(
        "parse 완료 (req=%s, page=%d, %.0fms, 출력 %d자, figure %d개)",
        request_id[:64], page_index, (t2 - t1) * 1000, len(page["markdown"]),
        sum(1 for b in page["blocks"] if b["type"] == "image"),
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "engine": ENGINE_NAME,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "page": {
            "page_index": page_index,
            "markdown": page["markdown"],
            "blocks": page["blocks"],
            "provider_raw": raw[:_PROVIDER_RAW_CAP],
            "warnings": page["warnings"] + warnings,
        },
        "timings": {
            "preprocess_ms": round((t1 - t0) * 1000, 1),
            "inference_ms": round((t2 - t1) * 1000, 1),
            "postprocess_ms": round((t3 - t2) * 1000, 1),
        },
    }
