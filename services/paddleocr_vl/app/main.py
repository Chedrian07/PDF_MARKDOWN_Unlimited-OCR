"""PaddleOCR-VL-1.6 sidecar — 내부 프로토콜 v1 서버 (docs/OCR_ENGINE_PROTOCOL.md).

GET  /health   : 프로토콜 health (모델 미로드 시에도 200 — model_loaded로 구분)
POST /v1/parse : 페이지 이미지 1장 → normalized page (figure는 [[FIGURE:n]])

파이프라인 입력은 파일 경로가 필요하므로 업로드 이미지를 sidecar 컨테이너의
임시 디렉터리에만 잠시 저장 후 삭제한다. 응답에는 어떤 파일 경로도 넣지 않는다.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .adapter import adapt_page
from .config import PaddleConfig
from .model import PaddleModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
ENGINE_NAME = "paddleocr_vl"
_MAX_IMAGE_PIXELS = 60_000_000
_OOM_FALLBACK_MAX_PIXELS = 2048 * 2048  # OOM 1회 강등 시 사용 (기본 max_pixels 미설정 대비)
_OOM_MIN_PIXELS = 512 * 512             # 강등 하한 — 이 아래로는 인식률이 무의미해진다

# PIL 폭탄 임계값도 같은 값으로 — 1차 방어는 아래 헤더 검사, 이건 헤더를 속인
# 파일에 대한 2차 그물이다 (초과 시 경고·2배 초과 시 DecompressionBombError)
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS

cfg = PaddleConfig.from_env()
model = PaddleModel(cfg)


class ParseOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    max_pixels: int | None = Field(default=None, ge=64 * 64, le=6000 * 6000)


def _load_in_background() -> None:
    try:
        model.load()
    except Exception:  # noqa: BLE001 — load_error로 health에 노출됨
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load_in_background, name="model-load", daemon=True).start()
    yield


app = FastAPI(title="PaddleOCR-VL sidecar", lifespan=lifespan)


_NO_GPU_INFO = {"gpu_name": None, "gpu_total_mb": None, "gpu_free_mb": None}


def _free_mb_via_smi(gpu_name: str, total_mb: int) -> int | None:
    """가용 VRAM은 nvidia-smi로 얻는다 — paddle에는 torch의 mem_get_info 대응 API가
    없고(실측 확인), memory_allocated/reserved는 파이프라인이 쓰는 할당자를 반영하지
    못해 0으로 나온다.

    nvidia-smi는 CUDA_VISIBLE_DEVICES를 무시하고 모든 GPU를 나열하므로 인덱스를
    그대로 믿으면 다른 카드의 값을 보고할 수 있다 — 이름과 총량이 **정확히 하나의
    행과 일치할 때만** 채택하고, 모호하면 None으로 둔다(틀린 수치보다 없는 게 낫다).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4.0,
        )
        if out.returncode != 0:
            return None
        matches = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            name, total, used = parts[0], int(float(parts[1])), int(float(parts[2]))
            if name == gpu_name and abs(total - total_mb) <= 4:  # MiB 반올림 오차 허용
                matches.append(total - used)
        return matches[0] if len(matches) == 1 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _gpu_info() -> dict:
    """GPU 정보 — 이름/총량은 로드 시 수집한 캐시값, 가용량만 nvidia-smi로 조회한다.

    health는 폴링 대상이므로 **요청 경로에서 paddle CUDA API를 호출하지 않는다**:
    파이프라인의 vlm 워커 생성 전에 다른 스레드가 CUDA를 건드리면 워커가
    static graph 모드로 올라와 모든 추론이 실패하는 것을 실측했다
    (model.py::_warmup 주석 참조)."""
    if model.gpu_name is None or model.gpu_total_mb is None:
        return dict(_NO_GPU_INFO)
    return {
        "gpu_name": model.gpu_name,
        "gpu_total_mb": model.gpu_total_mb,
        "gpu_free_mb": _free_mb_via_smi(model.gpu_name, model.gpu_total_mb),
    }


def _runtime_version() -> str:
    try:
        import paddleocr

        return getattr(paddleocr, "__version__", "unknown")
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
        "runtime": "paddleocr",
        "runtime_version": _runtime_version(),
        "device": "cuda" if cfg.device.startswith("gpu") else "cpu",
        "dtype": "bfloat16",
        "model_loaded": model.loaded,
        "load_error": model.load_error,
        **_gpu_info(),
    }


def _validate_image(data: bytes) -> tuple[int, int]:
    """형식·픽셀 수를 **디코드 전에** 헤더로 검증한 뒤에만 실제 디코드한다
    (load() 후 검사하면 픽셀 폭탄이 이미 메모리를 잡은 뒤다)."""
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
    return im.width, im.height


def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "outofmemory" in e.__class__.__name__.lower()


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
        opts = ParseOptions.model_validate(json.loads(options or "{}"))
    except Exception as e:
        raise HTTPException(422, f"options 스키마 위반: {e}") from e

    t0 = time.monotonic()
    data = file.file.read(cfg.max_upload_mb * 1024 * 1024 + 1)
    width, height = _validate_image(data)

    # 파이프라인은 파일 경로 입력 — sidecar 로컬 임시 파일로만 존재 (응답에 미포함)
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="page_")
    warnings: list[str] = []
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        t1 = time.monotonic()
        try:
            raw = model.predict_page(tmp_path, max_pixels=opts.max_pixels)
        except Exception as e:  # noqa: BLE001 — OOM만 1회 강등 재시도
            if not _is_oom(e):
                logger.exception("추론 실패 (req=%s)", request_id[:64])
                raise HTTPException(502, f"추론 실패: {e.__class__.__name__}") from e
            model.release_cache()
            base_px = opts.max_pixels or cfg.max_pixels or _OOM_FALLBACK_MAX_PIXELS
            # 강등은 반드시 더 작은 값으로 — 하한(_OOM_MIN_PIXELS)에 이미 닿았으면
            # 재시도해도 같은 조건이라 즉시 명확한 오류를 낸다.
            reduced = max(_OOM_MIN_PIXELS, base_px // 2)
            if reduced >= base_px:
                logger.warning("OOM (req=%s) — 더 낮출 해상도가 없어 재시도 생략", request_id[:64])
                raise HTTPException(
                    502,
                    "GPU 메모리 부족 — 이미 최소 해상도입니다. RENDER_DPI를 낮추거나 "
                    "PADDLEOCR_DEVICE=cpu로 전환하세요",
                ) from e
            logger.warning(
                "OOM (req=%s) — max_pixels %s→%d로 강등 후 1회 재시도",
                request_id[:64], base_px, reduced,
            )
            try:
                raw = model.predict_page(tmp_path, max_pixels=reduced)
                warnings.append(f"GPU 메모리 부족으로 해상도 강등(max_pixels→{reduced})")
            except Exception as e2:  # noqa: BLE001
                model.release_cache()
                logger.exception("OOM 재시도 실패 (req=%s)", request_id[:64])
                raise HTTPException(
                    502, "GPU 메모리 부족 — PADDLEOCR_MAX_PIXELS를 낮추거나 문서 해상도를 줄이세요"
                ) from e2
        t2 = time.monotonic()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    try:
        page = adapt_page(raw, width, height)
    except ValueError as e:
        logger.error("공식 결과 스키마 위반 (req=%s): %s", request_id[:64], e)
        raise HTTPException(502, f"파이프라인 결과 스키마 위반: {e}") from e
    t3 = time.monotonic()

    logger.info(
        "parse 완료 (req=%s, page=%d, %.0fms, 블록 %d개)",
        request_id[:64], page_index, (t2 - t1) * 1000, len(page["blocks"]),
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
            "provider_raw": page["provider_raw"],
            "warnings": page["warnings"] + warnings,
        },
        "timings": {
            "preprocess_ms": round((t1 - t0) * 1000, 1),
            "inference_ms": round((t2 - t1) * 1000, 1),
            "postprocess_ms": round((t3 - t2) * 1000, 1),
        },
    }
