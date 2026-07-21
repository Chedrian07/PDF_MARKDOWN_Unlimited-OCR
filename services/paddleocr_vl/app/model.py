"""PaddleOCR-VL-1.6 파이프라인 로더/추론 — 공식 Blackwell 경로 (wheel 방식).

공식 가이드: paddlepaddle-gpu(cu129 인덱스) + paddleocr[doc-parser] 3.6.x.
revision 고정: PaddleOCRVL 생성자는 HF revision 인자를 받지 않으므로, 기동 시
huggingface_hub.snapshot_download(revision=고정 SHA)로 캐시를 선점해
PADDLE_PDX_MODEL_SOURCE=huggingface 경로가 고정 스냅샷을 쓰게 한다
(잔여 위험은 docs/PADDLEOCR_VL_BLACKWELL_5070TI.md §revision 참조).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from .config import PIPELINE_VERSION, PaddleConfig

logger = logging.getLogger(__name__)


class PaddleModel:
    """단일 파이프라인. **모든 paddle 접촉을 전용 스레드 하나에 고정**한다.

    실측(2026-07-20, RTX 5070 Ti): 파이프라인을 로드 스레드에서 만들고 같은
    스레드에서 워밍업 추론을 하면 정상인데, 동일 객체를 FastAPI 요청 스레드에서
    호출하면 'vlm' 워커가 static graph 모드로 올라와
    "int(Tensor) is not supported in static graph mode"로 **모든** 추론이 실패했다.
    파이프라인 생성·워밍업·추론을 max_workers=1 실행기 한 곳에서 수행하면
    재현되지 않는다. 단일 GPU 직렬 처리라는 설계 의도와도 일치한다
    (이 실행기가 곧 직렬화 장치라 별도 락이 필요 없다).
    """

    def __init__(self, cfg: PaddleConfig) -> None:
        self.cfg = cfg
        self._pipeline = None
        # 파이프라인을 소유하는 유일한 스레드 — 생성·워밍업·추론이 모두 여기서 돈다
        self._owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paddle-infer")
        self.load_error: str | None = None
        # 로드 시점에 1회 수집한 GPU 정보 — health가 요청마다 paddle CUDA API를
        # 호출하지 않게 한다 (아래 warmup 주석 참조).
        self.gpu_name: str | None = None
        self.gpu_total_mb: int | None = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _pin_revision(self) -> None:
        """HF 캐시에 고정 revision 스냅샷을 선다운로드 (best-effort 아님 — 실패는 로드 실패)."""
        if not self.cfg.model_revision:
            return
        from huggingface_hub import snapshot_download

        logger.info("모델 스냅샷 고정: %s@%s", self.cfg.model_id, self.cfg.model_revision[:8])
        snapshot_download(repo_id=self.cfg.model_id, revision=self.cfg.model_revision)

    def load(self) -> None:
        """소유 스레드에서 파이프라인을 만들고 워밍업까지 끝낸다 (호출자는 대기)."""
        try:
            self._pin_revision()  # 네트워크 IO — paddle 미접촉이라 어느 스레드든 무방
            self._owner.submit(self._load_in_owner).result()
            self.load_error = None
        except Exception as e:
            self.load_error = f"{e.__class__.__name__}: {e}"[:500]
            logger.exception("PaddleOCR-VL 로딩 실패")
            raise

    def _load_in_owner(self) -> None:
        from paddleocr import PaddleOCRVL

        logger.info(
            "PaddleOCR-VL 파이프라인 로딩 (version=%s, device=%s)",
            PIPELINE_VERSION, self.cfg.device,
        )
        pipeline = PaddleOCRVL(
            pipeline_version=PIPELINE_VERSION,
            device=self.cfg.device,
        )
        self._collect_gpu_info()
        # 워밍업: vlm 워커는 첫 추론 때 지연 생성된다 — 소유 스레드에서 미리 돌려
        # 생성 시점을 통제하고 첫 사용자 요청의 지연도 없앤다.
        self._warmup(pipeline)
        self._pipeline = pipeline
        logger.info("PaddleOCR-VL 로딩 완료 (워밍업 포함)")

    def _collect_gpu_info(self) -> None:
        """GPU 이름/총량을 로드 스레드에서 1회만 읽는다 (요청 경로에서 CUDA 미접촉)."""
        try:
            import paddle

            if not paddle.device.is_compiled_with_cuda() or self.cfg.device == "cpu":
                return
            props = paddle.device.cuda.get_device_properties(0)
            self.gpu_name = props.name
            self.gpu_total_mb = int(props.total_memory) // (1024 * 1024)
        except Exception:  # noqa: BLE001 — 진단 정보일 뿐 로드를 막지 않는다
            logger.warning("GPU 정보 조회 실패 — health의 gpu 필드는 null로 보고됩니다")

    @staticmethod
    def _warmup(pipeline) -> None:
        """텍스트가 있는 작은 이미지로 1회 추론 — vlm 워커를 로드 스레드에서 생성한다.
        실패해도 로드는 계속한다(실사용 추론에서 다시 오류가 드러난다)."""
        import tempfile
        from pathlib import Path

        from PIL import Image, ImageDraw

        fd, path = tempfile.mkstemp(suffix=".png", prefix="warmup_")
        os.close(fd)
        try:
            im = Image.new("RGB", (800, 400), "white")
            draw = ImageDraw.Draw(im)
            draw.text((40, 40), "Warmup page for pipeline initialization.", fill="black")
            draw.text((40, 90), "Second line of warmup text.", fill="black")
            im.save(path)
            list(pipeline.predict(path))
            logger.info("파이프라인 워밍업 완료 (vlm 워커 생성됨)")
        except Exception as e:  # noqa: BLE001
            logger.warning("파이프라인 워밍업 실패 (%s) — 첫 요청에서 재시도됩니다",
                           e.__class__.__name__)
        finally:
            Path(path).unlink(missing_ok=True)

    def predict_page(self, image_path: str, max_pixels: int | None = None) -> dict:
        """이미지 1장 → 공식 결과 JSON(dict). OOM 강등은 호출자 몫.

        실제 추론은 소유 스레드에서 수행하고 호출 스레드는 결과를 기다린다
        (max_workers=1이라 이 대기가 곧 직렬화다)."""
        if self._pipeline is None:
            raise RuntimeError("파이프라인이 로드되지 않았습니다")
        kwargs: dict = {}
        if self.cfg.min_pixels is not None:
            kwargs["min_pixels"] = self.cfg.min_pixels
        effective_max = max_pixels or self.cfg.max_pixels
        if effective_max is not None:
            kwargs["max_pixels"] = effective_max
        return self._owner.submit(self._predict_in_owner, image_path, kwargs).result()

    def _predict_in_owner(self, image_path: str, kwargs: dict) -> dict:
        results = list(self._pipeline.predict(image_path, **kwargs))
        if not results:
            raise RuntimeError("파이프라인이 빈 결과를 반환했습니다")
        data = results[0].json  # PaddleX Result: numpy → 파이썬 기본형 dict
        if not isinstance(data, dict):
            raise RuntimeError("파이프라인 결과 형식이 예상과 다릅니다 (.json이 dict가 아님)")
        return data

    def release_cache(self) -> None:
        """OOM 후 CUDA 캐시 반환 — paddle 접촉이므로 소유 스레드에서 수행."""
        def _release() -> None:
            try:
                import paddle

                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
            except Exception:  # pragma: no cover - 방어적
                pass

        try:
            self._owner.submit(_release).result(timeout=30)
        except Exception:  # pragma: no cover - 방어적 (실행기 종료 등)
            pass
