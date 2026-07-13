"""잡 실행 오케스트레이션: 렌더 → 청크 OCR → 병합. 워커 스레드에서 호출된다."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..engine.base import JobCanceled, OCREngine
from .merge import ChunkResult, IncrementalMerger
from .pdf import render_pdf_pages

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings
    from ..jobs import EventBroker, Job, JobStore

logger = logging.getLogger(__name__)

_TOKEN_FLUSH_CHARS = 256
_TOKEN_FLUSH_SECS = 0.1
_PAGE_MARKER = "<PAGE>"


class BrokerSink:
    """엔진 토큰 스트림 → SSE token 이벤트(코얼레싱) + <PAGE> 마커 기반 페이지 진행률."""

    def __init__(self, job: "Job", store: "JobStore", broker: "EventBroker") -> None:
        self._job = job
        self._store = store
        self._broker = broker
        self._buf: list[str] = []
        self._buf_len = 0
        self._last_flush = time.monotonic()
        self._marker_tail = ""
        self._chunk_start = 1
        self._pages_seen = 0

    def set_chunk(self, start_page: int) -> None:
        self.flush()
        self._chunk_start = start_page
        self._pages_seen = 0
        self._marker_tail = ""

    def on_text(self, text: str) -> None:
        # 페이지 마커 카운트 (조각 경계에 걸친 마커 대비 tail 유지)
        probe = self._marker_tail + text
        markers = probe.count(_PAGE_MARKER)
        self._marker_tail = probe[-(len(_PAGE_MARKER) - 1):] if len(probe) >= len(_PAGE_MARKER) else probe
        if markers:
            self._pages_seen += markers
            current = min(
                self._chunk_start + self._pages_seen - 1,
                max(self._job.progress.get("total_pages", 1), 1),
            )
            if current > self._job.progress.get("current_page", 0):
                self._job.progress["current_page"] = current
                self._broker.publish_progress(self._job)

        self._buf.append(text)
        self._buf_len += len(text)
        now = time.monotonic()
        if self._buf_len >= _TOKEN_FLUSH_CHARS or (now - self._last_flush) >= _TOKEN_FLUSH_SECS:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self._broker.publish(self._job.id, "token", {"text": "".join(self._buf)})
            self._buf = []
            self._buf_len = 0
        self._last_flush = time.monotonic()


def _chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def execute_job(
    job: "Job",
    store: "JobStore",
    broker: "EventBroker",
    engine: OCREngine,
    settings: "Settings",
    cancel: threading.Event,
) -> None:
    sink = BrokerSink(job, store, broker)
    try:
        job.status = "running"
        job.progress.update(phase="render", current_page=0, chunk=0, total_chunks=0)
        store.save(job)
        broker.publish_progress(job)

        def _render_cb(done: int, total: int) -> None:
            # 렌더 단계에서도 페이지 단위로 취소/삭제에 반응한다 — 대형 문서(수백 p)
            # 렌더가 끝날 때까지 취소가 무시되지 않게. 예외는 render_pdf_pages를
            # 관통해 아래 JobCanceled 핸들러로 떨어진다.
            if cancel.is_set():
                raise JobCanceled()
            job.progress.update(current_page=done, total_pages=total)
            broker.publish_progress(job)

        pages = render_pdf_pages(
            job.dir / "source.pdf", job.dir / "pages", job.dpi, settings.max_pages, _render_cb
        )
        if cancel.is_set():
            raise JobCanceled()

        total = len(pages)
        chunk_size = 1 if job.mode == "per_page" else settings.pages_per_chunk
        chunks = _chunked(pages, chunk_size)
        job.progress.update(total_pages=total, total_chunks=len(chunks), current_page=0)
        store.save(job)

        merger = IncrementalMerger(job.dir, settings.page_separator)
        done_pages = 0
        for ci, chunk in enumerate(chunks):
            if cancel.is_set():
                raise JobCanceled()
            start_page = done_pages + 1
            job.progress.update(phase="ocr", chunk=ci + 1, current_page=start_page)
            store.save(job)
            broker.publish_progress(job)

            work_dir = job.dir / "work" / f"chunk_{ci:02d}"
            sink.set_chunk(start_page)
            if job.mode == "per_page":
                md = engine.run_single(chunk[0], work_dir, sink, cancel)
                merger.add_chunk(ChunkResult(work_dir, start_page, 1, md, single=True))
            else:
                md = engine.run_multi(chunk, work_dir, sink, cancel)
                merger.add_chunk(ChunkResult(work_dir, start_page, len(chunk), md))
            sink.flush()
            # 취소돼도 이 청크의 부분 출력까지는 병합 후에 중단한다
            if cancel.is_set():
                raise JobCanceled()

            done_pages += len(chunk)
            job.progress["current_page"] = done_pages
            store.save(job)
            broker.publish_progress(job)

        job.progress["phase"] = "merge"
        broker.publish_progress(job)
        merger.finalize()
        job.warnings = merger.warnings
        job.status = "done"
        job.error = None
        store.save(job)
        broker.publish(
            job.id,
            "done",
            {
                "markdown_url": f"/api/jobs/{job.id}/markdown",
                "archive_url": f"/api/jobs/{job.id}/archive",
            },
        )
        logger.info("잡 완료: %s (%d페이지)", job.id, total)

    except JobCanceled:
        sink.flush()
        job.status = "canceled"
        job.error = "사용자에 의해 취소되었습니다"
        store.save(job)
        broker.publish(job.id, "error", {"message": job.error, "canceled": True})
        logger.info("잡 취소: %s", job.id)
    except Exception as e:  # noqa: BLE001 — 잡 단위 격리
        sink.flush()
        logger.exception("잡 실패: %s", job.id)
        job.status = "error"
        job.error = str(e)[:2000] or e.__class__.__name__
        store.save(job)
        broker.publish(job.id, "error", {"message": job.error})
    finally:
        if job.delete_requested:
            shutil.rmtree(job.dir, ignore_errors=True)
            store.remove(job.id)
