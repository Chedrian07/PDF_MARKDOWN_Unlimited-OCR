"""FastAPI 앱 팩토리. 실행: uvicorn app.main:app"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Settings
from .engine import build_engine
from .jobs import EventBroker, JobStore, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_GC_INTERVAL_S = 6 * 60 * 60  # 잡 TTL GC 주기 — 시작 시 1회 + 6시간마다


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    store = JobStore(settings.jobs_dir)
    store.load_existing()
    broker = EventBroker()
    engine = build_engine(settings)  # 잘못된 OCR_DEVICE/OCR_ENGINE은 여기서 즉시 실패
    cancel_events: dict[str, threading.Event] = {}
    worker = Worker(store, broker, engine, settings, cancel_events)
    load_state: dict = {"error": None}

    def _preload() -> None:
        try:
            engine.load()
        except Exception as e:  # noqa: BLE001 — 헬스에 노출하고 잡 제출 시 재시도
            # 일시적 조건(sidecar가 아직 준비 중)은 정상적인 기동 과정이다 —
            # 무서운 traceback 대신 info로 남기고, 잡 제출 시 워커가 대기한다.
            if getattr(e, "transient", False):
                logger.info("모델 프리로드 대기: %s", str(e)[:200])
            else:
                logger.exception("모델 프리로드 실패")
            load_state["error"] = str(e)[:500]

    async def _gc_loop(app_: FastAPI) -> None:
        """잡 TTL GC — 시작 직후 1회 + _GC_INTERVAL_S 주기. 번역 스레드가 살아 있는
        잡은 삭제 직전 잡별 레지스트리 확인으로 보호(스냅샷 방식이면 GC 패스 도중
        시작된 번역이 빠진다), 파일 IO(rmtree)는 스레드로 오프로드."""

        def _is_protected(job_id: str) -> bool:
            with app_.state.translate_lock:
                return any(jid == job_id for jid, _lang in app_.state.translate_tasks)

        while True:
            try:
                await asyncio.to_thread(store.gc_expired, settings.job_ttl_days, _is_protected)
            except Exception:  # noqa: BLE001 — GC 실패가 다음 주기를 막지 않게
                logger.exception("잡 GC 실패")
            await asyncio.sleep(_GC_INTERVAL_S)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        if settings.preload_model and not engine.loaded:
            threading.Thread(target=_preload, name="model-preload", daemon=True).start()
        # JOB_TTL_DAYS>0일 때만 기동 — 기본 0 = 사용자 데이터 자동 삭제 비활성(opt-in)
        gc_task = asyncio.create_task(_gc_loop(_app)) if settings.job_ttl_days > 0 else None
        yield
        if gc_task is not None:
            gc_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await gc_task
        worker.stop()

    app = FastAPI(title="Unlimited-OCR — PDF → Markdown", lifespan=lifespan)
    # Host 헤더 화이트리스트 — DNS rebinding 방어 (무인증 서비스, README §보안).
    # Starlette가 포트를 떼고 비교하므로 localhost:8000도 localhost로 통과한다.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.state.settings = settings
    app.state.store = store
    app.state.broker = broker
    app.state.engine = engine
    app.state.worker = worker
    app.state.cancel_events = cancel_events
    app.state.load_state = load_state
    # 번역 태스크 레지스트리: 키 (job_id, lang) → {"thread","cancel"}.
    # OCR 워커(단일 스레드 직렬)와 달리 번역은 잡별 데몬 스레드로 병렬 실행된다.
    app.state.translate_tasks: dict[tuple[str, str], dict] = {}
    app.state.translate_lock = threading.Lock()

    app.include_router(router)

    frontend = settings.resolve_frontend_dir()
    if frontend is not None:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
        logger.info("프론트엔드 서빙: %s", frontend)
    else:
        logger.warning("프론트엔드 디렉터리를 찾지 못했습니다 (FRONTEND_DIR 설정 가능)")

    return app


app = create_app()
