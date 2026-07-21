"""sidecar 기반 OCR 엔진 (ovisocr2 · paddleocr_vl) — GPU는 sidecar 컨테이너만 사용.

메인 backend는 페이지 이미지를 HTTP로 보내고 normalized 결과를 받아
공통 materializer로 기존 청크 산출물 규약을 재현한다. 스트리밍은 페이지
단위(stream_granularity="page")다 — 가짜 토큰 스트리밍을 만들지 않고,
페이지가 완료될 때 `<PAGE>` 마커와 함께 전체 텍스트를 한 번에 발행한다
(기존 프론트의 `<PAGE>` 진행 계약과 그대로 호환).

단일 GPU 원칙: 이 엔진은 다른 GPU 모델로의 자동 fallback을 하지 않는다.
provider 실패는 명확한 오류로 표면화되고 전환은 사용자가 profile로 결정한다.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..sidecar.client import SidecarClient, SidecarError
from ..sidecar.materializer import ChunkMaterializer
from ..sidecar.protocol import PageResult, sanitize_page
from .base import EngineCapabilities, EngineError, JobCanceled, OCREngine, StreamSink

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings

# health 프로브 캐시 TTL — 성공·실패 **둘 다** 캐시한다. 실패를 캐시하지 않으면
# /api/health 폴링(프런트 10초 주기)마다 죽은 sidecar로 연결을 시도해 요청 스레드가
# connect timeout만큼 묶이고, 잡 실행 중에는 페이지마다 같은 대기가 반복된다.
_HEALTH_CACHE_TTL_S = 5.0
_MAX_JOB_WARNINGS = 40


class _AnyCancel:
    """여러 취소 신호의 OR — 잡 취소 + 청크 내부 실패(형제 요청 중단)를 합친다."""

    def __init__(self, *signals) -> None:
        self._signals = signals

    def is_set(self) -> bool:
        return any(s.is_set() for s in self._signals)


@dataclass(frozen=True)
class SidecarSpec:
    default_model_id: str
    layout_capability: str  # "full" | "figure_only"


# 지원 sidecar 엔진 선언 — 새 엔진은 여기와 services/에 추가한다
SIDECAR_SPECS: dict[str, SidecarSpec] = {
    "ovisocr2": SidecarSpec(
        default_model_id="ATH-MaaS/OvisOCR2",
        layout_capability="figure_only",  # 텍스트 bbox 미제공 — 거짓 full 금지
    ),
    "paddleocr_vl": SidecarSpec(
        default_model_id="PaddlePaddle/PaddleOCR-VL-1.6",
        layout_capability="full",
    ),
}


class SidecarEngine(OCREngine):
    def __init__(self, settings: "Settings", name: str) -> None:
        if name not in SIDECAR_SPECS:  # registry가 걸러주지만 방어적으로
            raise ValueError(f"알 수 없는 sidecar 엔진: {name!r}")
        self.name = name
        self.device = "cuda"          # provider가 CUDA에서 실행 (backend 자체는 GPU 미사용)
        self.dtype_name = "bfloat16"
        self._spec = SIDECAR_SPECS[name]
        self._settings = settings
        self._client = SidecarClient(
            settings.sidecar_url,
            engine_name=name,
            connect_timeout_s=settings.sidecar_connect_timeout_s,
            read_timeout_s=settings.sidecar_read_timeout_s,
            health_timeout_s=settings.sidecar_health_timeout_s,
            max_response_mb=settings.sidecar_max_response_mb,
            retries=settings.sidecar_retries,
        )
        self._health_lock = threading.Lock()
        self._last_health = None       # SidecarHealth | None (성공 프로브 결과)
        self._last_health_error = None  # str | None (실패 사유 — 실패도 캐시한다)
        self._last_probe_ts = 0.0
        self._warn_lock = threading.Lock()
        self._warnings: list[str] = []

    # ── 상태/메타 ──────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        h = self._last_health
        return h is not None and h.model_loaded

    def _probe_health(self):
        """health 프로브 (성공·실패 모두 TTL 캐시). 반환: (health|None, error|None)."""
        now = time.monotonic()
        with self._health_lock:
            if now - self._last_probe_ts < _HEALTH_CACHE_TTL_S and (
                self._last_health is not None or self._last_health_error is not None
            ):
                return self._last_health, self._last_health_error
        health = None
        error: str | None = None
        try:
            health = self._client.health()
        except SidecarError as e:
            error = str(e)[:300]
        with self._health_lock:
            self._last_health = health
            self._last_health_error = error
            self._last_probe_ts = time.monotonic()
        if health is not None:
            self.dtype_name = health.dtype or self.dtype_name
            self.device = health.device or self.device
        return health, error

    def load(self) -> None:
        """provider health 확인 — 모델 로드는 sidecar가 자체 수행한다.

        provider가 죽었거나 아직 로딩 중이면 명확한 EngineError를 던진다
        (무한 대기/재시도 없음 — 잡 단위 실패로 표면화되고 다음 잡에서 재확인).
        실패 결과도 캐시되므로 provider가 죽은 동안 페이지마다 연결을 재시도하며
        타임아웃을 반복하지 않는다 (TTL 후 자동 재확인).
        """
        if self.loaded:
            return
        health, error = self._probe_health()
        if error is not None:
            raise EngineError(error)
        if health.status != "ok":
            raise EngineError(f"sidecar 상태 이상: {health.status}")
        if not health.model_loaded:
            raise EngineError(
                "sidecar가 아직 모델을 로드하지 못했습니다 — 최초 기동은 모델 다운로드로 "
                "수 분 걸릴 수 있습니다. 잠시 후 다시 시도하세요 "
                "(진행: docker compose logs -f)"
            )

    def capabilities(self) -> EngineCapabilities:
        h = self._last_health
        return EngineCapabilities(
            model_id=(h.model_id if h else "") or self._spec.default_model_id,
            model_revision=(h.model_revision if h else ""),
            provider="local-sidecar",
            supports_multi_page=False,       # 페이지 단위 모델 — 문맥 공유 없음
            preferred_chunk_size=max(1, self._settings.remote_page_concurrency),
            stream_granularity="page",
            layout_capability=self._spec.layout_capability,
            figure_capability=True,
        )

    def provider_health(self) -> dict | None:
        """/api/health 폴링용 — 성공·실패 모두 TTL 캐시 (죽은 sidecar 폴링이
        요청 스레드를 connect timeout만큼 묶지 않게)."""
        health, error = self._probe_health()
        if health is None:
            return {"status": "unreachable", "error": error}
        return self._health_dict(health)

    # ── 경고 채널 ───────────────────────────────────────────────

    def _note(self, message: str) -> None:
        """사용자 노출용 경고 적재 (페이지 스레드에서도 호출되므로 락 보호)."""
        with self._warn_lock:
            if len(self._warnings) < _MAX_JOB_WARNINGS:
                self._warnings.append(message)

    def drain_warnings(self) -> list[str]:
        with self._warn_lock:
            drained, self._warnings = self._warnings, []
        # 페이지마다 반복되는 동일 경고는 1건으로 접는다 (순서 보존)
        seen: set[str] = set()
        unique: list[str] = []
        for w in drained:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique

    @staticmethod
    def _health_dict(h) -> dict:
        return {
            "status": h.status,
            "runtime": h.runtime,
            "version": h.runtime_version,
            "model_loaded": h.model_loaded,
            "gpu_total_mb": h.gpu_total_mb,
            "gpu_free_mb": h.gpu_free_mb,
        }

    def gpu_name(self) -> str | None:
        h = self._last_health
        return h.gpu_name if h else None

    # ── 실행 ───────────────────────────────────────────────────

    def _parse_one(
        self, image_path: Path, local_page: int, cancel
    ) -> PageResult:
        request_id = f"{uuid.uuid4().hex[:12]}-p{local_page}"
        resp = self._client.parse_page(
            image_path,
            page_index=local_page,
            request_id=request_id,
            options={},
            cancel=cancel,
        )
        page, warnings = sanitize_page(resp.page)
        # 정화로 버려진 블록·절단은 사용자에게 알린다 (조용한 내용 손실 방지).
        # sidecar가 스스로 보고한 경고(해상도 강등 등)도 함께 승격한다.
        # 페이지 번호는 붙이지 않는다 — local_page는 청크 내 인덱스라 기본 설정
        # (청크=1페이지)에서는 항상 0이다. 전역 페이지 범위는 runner가 붙인다.
        for w in warnings:
            self._note(w)
        for w in page.warnings:
            if w not in warnings:
                self._note(w)
        return page

    def _run_pages(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
        single: bool,
    ) -> str:
        self.load()
        out_dir.mkdir(parents=True, exist_ok=True)
        mat = ChunkMaterializer(out_dir, single=single)
        parts: list[str] = []

        concurrency = min(len(image_paths), max(1, self._settings.remote_page_concurrency))
        if concurrency <= 1:
            pages = self._iter_serial(image_paths, cancel)
        else:
            pages = self._iter_concurrent(image_paths, cancel, concurrency)

        for local_page, page in pages:
            # 취소 이후 도착한 결과는 병합하지 않는다 (여기 도달 전에 JobCanceled 전파)
            md = mat.add_page(page, image_paths[local_page], local_page)
            if single:
                sink.on_text(md)
            else:
                sink.on_text("<PAGE>\n")
                sink.on_text(md + "\n")
            parts.append(md)
        for w in mat.warnings:
            self._note(w)
        mat.finalize()
        if single:
            return parts[0] if parts else ""
        return "<PAGE>\n" + "\n<PAGE>\n".join(parts)

    def _iter_serial(self, image_paths: list[Path], cancel: threading.Event):
        for local_page, path in enumerate(image_paths):
            if cancel.is_set():
                raise JobCanceled()
            yield local_page, self._parse_one(path, local_page, cancel)

    def _iter_concurrent(
        self, image_paths: list[Path], cancel: threading.Event, concurrency: int
    ):
        """페이지를 동시 요청하되 **순서대로** 소비한다 (SSE·병합 순서 보존).

        OCR_REMOTE_PAGE_CONCURRENCY>1일 때만. sidecar가 자체 큐로 직렬화하더라도
        요청 파이프라이닝으로 왕복 지연을 숨긴다.

        중단(취소·페이지 실패) 시 내부 stop 신호를 잡 취소와 OR로 묶어 형제 요청의
        대기를 즉시 푼다. 다만 **이미 전송된 요청은 sidecar에서 완주**한다 —
        추론 중에는 응답 헤더가 없어 연결을 실제로 끊을 수단이 없기 때문이다
        (docs/OCR_ENGINE_PROTOCOL.md §취소 의미론과 한계). 재시도는 그 뒤에 줄을 선다.
        """
        stop = threading.Event()
        signal = _AnyCancel(cancel, stop)
        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="sidecar-page")
        try:
            futures = [
                executor.submit(self._parse_one, path, i, signal)
                for i, path in enumerate(image_paths)
            ]
            for i, fut in enumerate(futures):
                if cancel.is_set():
                    raise JobCanceled()
                yield i, fut.result()
        finally:
            stop.set()  # 미완료 형제 요청의 연결을 끊는다 (다음 시도의 큐를 비움)
            executor.shutdown(wait=False, cancel_futures=True)

    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        return self._run_pages(image_paths, out_dir, sink, cancel, single=False)

    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        return self._run_pages([image_path], out_dir, sink, cancel, single=True)
