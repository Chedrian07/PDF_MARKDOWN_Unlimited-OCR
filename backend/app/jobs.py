"""잡 상태 저장(JobStore) · SSE 이벤트 브로커 · 단일 워커 스레드."""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from .config import Settings
    from .engine.base import OCREngine

logger = logging.getLogger(__name__)

_META_NAME = "meta.json"
_EVENT_QUEUE_MAX = 2000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_progress() -> dict:
    return {"phase": "render", "current_page": 0, "total_pages": 0, "chunk": 0, "total_chunks": 0}


@dataclass
class Job:
    id: str
    filename: str
    mode: str
    dpi: int
    dir: Path
    status: str = "queued"  # queued|running|done|error|canceled
    created_at: str = field(default_factory=_now_iso)
    progress: dict = field(default_factory=_default_progress)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    delete_requested: bool = False

    def _result_block(self) -> dict | None:
        if self.status != "done":
            return None
        base = f"/api/jobs/{self.id}"

        def _urls(subdir: str) -> list[str]:
            d = self.dir / subdir
            if not d.is_dir():
                return []
            # 이미지 파일만 — images/boxes.json 같은 메타 파일은 목록에서 제외
            return [
                f"{base}/files/{subdir}/{f.name}"
                for f in sorted(d.iterdir())
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]

        return {
            "markdown_url": f"{base}/markdown",
            "html_url": f"{base}/html",
            "archive_url": f"{base}/archive",
            "images": _urls("images"),
            "layouts": _urls("layout"),
            "pages": _urls("pages"),
            # 레이아웃 뷰/다운로드 가능 여부 — 레이아웃 기능(P14) 이전에 변환된
            # 잡에는 layout.json이 없어 /layout*이 404가 난다. 프런트가 이 플래그로
            # 버튼을 비활성화한다 (없으면 재변환 필요).
            "has_layout": (self.dir / "layout.json").is_file(),
        }

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "filename": self.filename,
            "status": self.status,
            "mode": self.mode,
            "created_at": self.created_at,
            "progress": dict(self.progress),
            "error": self.error,
            "warnings": list(self.warnings),
            "result": self._result_block(),
        }

    def meta(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "mode": self.mode,
            "dpi": self.dpi,
            "status": self.status,
            "created_at": self.created_at,
            "progress": self.progress,
            "error": self.error,
            "warnings": self.warnings,
        }


class JobStore:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self, filename: str, mode: str, dpi: int) -> Job:
        job_id = f"j_{uuid.uuid4().hex[:12]}"
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True)
        job = Job(id=job_id, filename=filename, mode=mode, dpi=dpi, dir=job_dir)
        with self._lock:
            self._jobs[job_id] = job
        self.save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def save(self, job: Job) -> None:
        tmp = job.dir / f".{_META_NAME}.tmp"
        try:
            tmp.write_text(json.dumps(job.meta(), ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, job.dir / _META_NAME)
        except FileNotFoundError:  # 삭제 경합 — 무시
            pass

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def delete_dir(self, job: Job) -> None:
        shutil.rmtree(job.dir, ignore_errors=True)
        self.remove(job.id)

    def gc_expired(
        self, ttl_days: int, is_protected: "Callable[[str], bool] | None" = None
    ) -> int:
        """TTL 지난 터미널 잡 자동 정리 (JOB_TTL_DAYS — 0 이하면 무동작).

        마지막 활동 시각은 meta.json mtime(save()가 상태 변화마다 재기록)과
        translations/*/state.json mtime의 최댓값 — OCR이 오래전에 끝났어도 최근
        번역된 잡은 보존한다. queued/running 잡은 절대 삭제하지 않고, is_protected
        (번역 스레드 활성 등)는 삭제 직전에 잡별로 호출한다 — 스냅샷 방식이면 GC
        패스 도중 시작된 번역이 보호되지 않는다. 삭제는 DELETE 엔드포인트와 같은
        delete_dir 경로. 삭제 수 반환."""
        if ttl_days <= 0:
            return 0
        now = time.time()
        cutoff = now - ttl_days * 86400
        with self._lock:
            jobs = list(self._jobs.values())
        removed = 0
        for job in jobs:
            if job.status in ("queued", "running"):
                continue
            try:
                mtime = (job.dir / _META_NAME).stat().st_mtime
            except OSError:  # meta 유실 — 나이를 알 수 없으니 보수적으로 보존
                continue
            tdir = job.dir / "translations"
            if tdir.is_dir():
                for st in tdir.glob("*/state.json"):
                    try:
                        mtime = max(mtime, st.stat().st_mtime)
                    except OSError:
                        pass
            if mtime >= cutoff:
                continue
            if is_protected is not None and is_protected(job.id):
                continue
            logger.info("잡 GC: %s 삭제 (status=%s, %.1f일 경과 > TTL %d일)",
                        job.id, job.status, (now - mtime) / 86400, ttl_days)
            self.delete_dir(job)
            removed += 1
        return removed

    def load_existing(self) -> None:
        """서버 재시작 시 디스크의 잡 복원. 실행 중이던 잡은 오류로 마킹."""
        if not self.jobs_dir.is_dir():
            return
        for d in sorted(self.jobs_dir.iterdir()):
            meta_path = d / _META_NAME
            if not meta_path.is_file():
                continue
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                job = Job(
                    id=m["id"], filename=m["filename"], mode=m.get("mode", "multi"),
                    dpi=int(m.get("dpi", 200)), dir=d, status=m.get("status", "error"),
                    created_at=m.get("created_at", _now_iso()),
                    progress=m.get("progress") or _default_progress(),
                    error=m.get("error"), warnings=m.get("warnings") or [],
                )
                changed = job.status in ("queued", "running")
                if changed:
                    job.status = "error"
                    job.error = "서버 재시작으로 중단되었습니다"
                with self._lock:
                    self._jobs[job.id] = job
                # 상태가 바뀐 잡만 재기록 — 터미널 잡의 meta.json mtime은 TTL GC의
                # "마지막 갱신" 시계라, 무조건 재저장하면 재시작마다 TTL이 리셋된다.
                if changed:
                    self.save(job)
            except Exception:
                logger.exception("잡 메타 복원 실패: %s", d)


class EventBroker:
    """잡별 SSE 구독 큐. token 이벤트는 가득 차면 버린다(진행/완료 이벤트는 보존)."""

    def __init__(self) -> None:
        self._subs: dict[str, list[queue.Queue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=_EVENT_QUEUE_MAX)
        with self._lock:
            self._subs.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subs.get(job_id)
            if subs and q in subs:
                subs.remove(q)
            if subs is not None and not subs:
                del self._subs[job_id]

    def publish(self, job_id: str, event: str, data: dict) -> None:
        with self._lock:
            subs = list(self._subs.get(job_id, ()))
        for q in subs:
            try:
                q.put_nowait((event, data))
            except queue.Full:
                if event == "token":
                    continue  # 토큰은 손실 허용
                try:  # 오래된 것 하나 버리고 재시도
                    q.get_nowait()
                    q.put_nowait((event, data))
                except (queue.Empty, queue.Full):  # pragma: no cover
                    pass

    def publish_progress(self, job: Job) -> None:
        self.publish(job.id, "progress", {**job.progress, "status": job.status})


class Worker(threading.Thread):
    """단일 워커: 모델이 프로세스당 1개이므로 잡을 직렬 처리한다."""

    def __init__(
        self,
        store: JobStore,
        broker: EventBroker,
        engine: "OCREngine",
        settings: "Settings",
        cancel_events: dict[str, threading.Event],
    ) -> None:
        super().__init__(name="ocr-worker", daemon=True)
        self.store = store
        self.broker = broker
        self.engine = engine
        self.settings = settings
        self.cancel_events = cancel_events
        self._queue: queue.Queue = queue.Queue()

    def submit(self, job: Job) -> None:
        self.cancel_events.setdefault(job.id, threading.Event())
        self._queue.put(job.id)

    def stop(self) -> None:
        self._queue.put(None)

    def run(self) -> None:
        from .pipeline.runner import execute_job

        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self.store.get(job_id)
            if job is None:
                continue
            cancel = self.cancel_events.setdefault(job_id, threading.Event())
            if job.delete_requested or cancel.is_set():
                job.status = "canceled"
                job.error = "사용자에 의해 취소되었습니다"
                self.store.save(job)
                if job.delete_requested:
                    self.store.delete_dir(job)
                self.cancel_events.pop(job_id, None)
                continue
            try:
                if not self.engine.loaded:
                    self.broker.publish(job_id, "progress", {"phase": "render", "status": "queued",
                                                             "current_page": 0, "total_pages": 0,
                                                             "chunk": 0, "total_chunks": 0})
                    self.engine.load()
            except Exception as e:  # noqa: BLE001 — 로드 실패를 잡 오류로 변환
                logger.exception("엔진 로드 실패")
                job.status = "error"
                job.error = f"모델 로드 실패: {e}"[:2000]
                self.store.save(job)
                self.broker.publish(job_id, "error", {"message": job.error})
                self.cancel_events.pop(job_id, None)
                continue
            execute_job(job, self.store, self.broker, self.engine, self.settings, cancel)
            self.cancel_events.pop(job_id, None)
