"""리소스 상한 라운드 검증 — 잡 TTL GC(JobStore.gc_expired)·work/ 터미널 정리.

디스크 시계 조작: meta.json mtime을 os.utime으로 과거로 밀어 TTL 경과를 흉내낸다.
"""

import os
import threading
import time

from app.config import Settings
from app.engine.fake import FakeEngine
from app.jobs import EventBroker, JobStore
from app.main import create_app
from app.pipeline.runner import execute_job

from conftest import make_pdf_bytes


def _make_job(store: JobStore, status: str = "done", age_days: float = 0.0):
    job = store.create("doc.pdf", "multi", dpi=72)
    job.status = status
    store.save(job)
    if age_days:
        past = time.time() - age_days * 86400
        os.utime(job.dir / "meta.json", (past, past))
    return job


# ── JobStore.gc_expired ─────────────────────────────────────────


def test_gc_removes_expired_terminal_jobs(tmp_path):
    store = JobStore(tmp_path / "jobs")
    old_done = _make_job(store, "done", age_days=10)
    old_error = _make_job(store, "error", age_days=10)
    assert store.gc_expired(7) == 2
    for job in (old_done, old_error):
        assert store.get(job.id) is None
        assert not job.dir.exists()


def test_gc_keeps_fresh_jobs(tmp_path):
    store = JobStore(tmp_path / "jobs")
    fresh = _make_job(store, "done", age_days=1)
    assert store.gc_expired(7) == 0
    assert store.get(fresh.id) is not None
    assert fresh.dir.exists()


def test_gc_never_deletes_active_jobs(tmp_path):
    """queued/running·보호(번역 스레드 활성) 잡은 아무리 오래돼도 삭제 금지.

    보호 검사는 삭제 직전 잡별 콜백 — GC 패스 도중 시작된 번역도 잡히도록."""
    store = JobStore(tmp_path / "jobs")
    running = _make_job(store, "running", age_days=100)
    queued = _make_job(store, "queued", age_days=100)
    translating = _make_job(store, "done", age_days=100)
    checked: list[str] = []

    def _is_protected(job_id: str) -> bool:
        checked.append(job_id)
        return job_id == translating.id

    assert store.gc_expired(7, is_protected=_is_protected) == 0
    assert translating.id in checked  # 콜백이 실제로 잡별 호출됨
    for job in (running, queued, translating):
        assert store.get(job.id) is not None
        assert job.dir.exists()


def test_gc_translation_activity_counts_as_activity(tmp_path):
    """OCR meta가 TTL을 넘겨도 최근 번역(state.json)이 있으면 보존한다."""
    store = JobStore(tmp_path / "jobs")
    job = _make_job(store, "done", age_days=100)
    tdir = job.dir / "translations" / "ko"
    tdir.mkdir(parents=True)
    (tdir / "state.json").write_text("{}", encoding="utf-8")  # 지금 = 신선한 번역 활동
    assert store.gc_expired(7) == 0
    assert job.dir.exists()

    # 번역 활동까지 오래되면 삭제된다
    past = time.time() - 100 * 86400
    os.utime(tdir / "state.json", (past, past))
    assert store.gc_expired(7) == 1
    assert not job.dir.exists()


def test_gc_disabled_when_ttl_zero(tmp_path):
    store = JobStore(tmp_path / "jobs")
    old = _make_job(store, "done", age_days=1000)
    assert store.gc_expired(0) == 0
    assert store.gc_expired(-1) == 0
    assert store.get(old.id) is not None
    assert old.dir.exists()


# ── work/ 터미널 정리 (runner.execute_job finally) ───────────────


def _run_fake_job(tmp_path, engine=None):
    store = JobStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=2, with_image=False))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=1,
    )
    engine = engine or FakeEngine(delay=0.0)
    engine.load()
    execute_job(job, store, EventBroker(), engine, settings, threading.Event())
    return job


def test_work_dir_removed_on_done(tmp_path):
    job = _run_fake_job(tmp_path)
    assert job.status == "done"
    assert not (job.dir / "work").exists()
    # 필요 산출물은 병합 시 이미 잡 루트로 이동돼 보존된다
    assert (job.dir / "result.md").is_file()
    assert list((job.dir / "images").glob("*.jpg"))
    assert list((job.dir / "layout").glob("*.jpg"))


def test_work_dir_removed_on_error(tmp_path):
    """전 청크 실패(status=error)여도 실패 청크의 work/ 잔여물이 남지 않는다."""

    class FailingEngine(FakeEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            out_dir.mkdir(parents=True, exist_ok=True)  # 실패 전 부분 산출물 흉내
            raise RuntimeError("모의 실패")

    job = _run_fake_job(tmp_path, engine=FailingEngine(delay=0.0))
    assert job.status == "error"
    assert not (job.dir / "work").exists()


# ── lifespan 배선 (main.create_app) ─────────────────────────────


def test_startup_gc_task_wired(settings):
    """JOB_TTL_DAYS>0면 시작 시 1회 GC가 돌아 만료 잡이 사라진다."""
    from fastapi.testclient import TestClient

    settings.job_ttl_days = 7
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    old = _make_job(JobStore(settings.jobs_dir), "done", age_days=10)

    app = create_app(settings)
    with TestClient(app) as client:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and old.dir.exists():
            time.sleep(0.02)
        assert not old.dir.exists()
        assert client.get(f"/api/jobs/{old.id}").status_code == 404


def test_startup_gc_disabled_by_default(settings):
    """기본값(JOB_TTL_DAYS=0)이면 GC 태스크가 아예 뜨지 않아 오래된 잡도 보존."""
    from fastapi.testclient import TestClient

    assert settings.job_ttl_days == 0
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    old = _make_job(JobStore(settings.jobs_dir), "done", age_days=1000)

    app = create_app(settings)
    with TestClient(app) as client:
        time.sleep(0.3)
        assert old.dir.exists()
        assert client.get(f"/api/jobs/{old.id}").status_code == 200
