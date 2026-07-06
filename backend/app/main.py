"""FastAPI 앱 팩토리. 실행: uvicorn app.main:app"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Settings
from .engine import build_engine
from .jobs import EventBroker, JobStore, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


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
            logger.exception("모델 프리로드 실패")
            load_state["error"] = str(e)[:500]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        if settings.preload_model and not engine.loaded:
            threading.Thread(target=_preload, name="model-preload", daemon=True).start()
        yield
        worker.stop()

    app = FastAPI(title="Unlimited-OCR — PDF → Markdown", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.broker = broker
    app.state.engine = engine
    app.state.worker = worker
    app.state.cancel_events = cancel_events
    app.state.load_state = load_state

    app.include_router(router)

    frontend = settings.resolve_frontend_dir()
    if frontend is not None:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
        logger.info("프론트엔드 서빙: %s", frontend)
    else:
        logger.warning("프론트엔드 디렉터리를 찾지 못했습니다 (FRONTEND_DIR 설정 가능)")

    return app


app = create_app()
