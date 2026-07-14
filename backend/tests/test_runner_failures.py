"""청크 실패 격리(runner.py) 검증 — 한 청크가 죽어도 잡 전체가 죽지 않는다.

FlakyEngine: FakeEngine을 상속해 지정한 호출 번호(1-based)에서만 예외를 던진다.
호출 카운트로 재시도 횟수까지 검증한다.
"""

import json
import threading

from app.config import Settings
from app.engine.base import JobCanceled
from app.engine.fake import FakeEngine
from app.jobs import EventBroker, JobStore
from app.pipeline.runner import execute_job

from conftest import make_pdf_bytes

FAILED_MARK = "이 페이지는 변환에 실패했습니다"


class FlakyEngine(FakeEngine):
    """지정한 호출(1-based)에서 예외를 던지는 가짜 엔진 — 청크 격리/재시도 검증용."""

    def __init__(self, fail_calls=(), fail_all=False, exc=RuntimeError):
        super().__init__(delay=0.0)
        self.calls = 0
        self.fail_calls = set(fail_calls)
        self.fail_all = fail_all
        self.exc = exc

    def _maybe_fail(self):
        self.calls += 1
        if self.fail_all or self.calls in self.fail_calls:
            raise self.exc(f"모의 실패 (call {self.calls})")

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self._maybe_fail()
        return super().run_multi(image_paths, out_dir, sink, cancel)

    def run_single(self, image_path, out_dir, sink, cancel):
        self._maybe_fail()
        return super().run_single(image_path, out_dir, sink, cancel)


def _run_job(tmp_path, engine, pages=4, mode="multi", pages_per_chunk=2):
    """execute_job을 워커 없이 직접 구동 (4페이지 × 청크 2 → 청크 2개 구성)."""
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", mode, dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=pages, with_image=False))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=pages_per_chunk,
    )
    engine.load()
    execute_job(job, store, broker, engine, settings, threading.Event())
    return job


def test_failed_chunk_becomes_placeholder_and_job_done(tmp_path):
    """청크1이 재시도까지 실패해도 잡은 done — 해당 페이지들은 플레이스홀더,
    warnings가 meta.json에 남고 나머지 청크는 정상 병합된다."""
    engine = FlakyEngine(fail_calls={1, 2})  # 청크1: 최초 + 재시도 모두 실패
    job = _run_job(tmp_path, engine)

    assert job.status == "done"
    assert engine.calls == 3  # 청크1 ×2 + 청크2 ×1
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert md.count(FAILED_MARK) == 2  # 1–2페이지 플레이스홀더
    assert "![](images/p0003_0.jpg)" in md and "![](images/p0004_0.jpg)" in md
    assert len(md.split("\n\n---\n\n")) == 4  # 글로벌 페이지 수 정합 유지
    assert len(job.warnings) == 1
    assert "1–2페이지" in job.warnings[0] and "플레이스홀더" in job.warnings[0]
    meta = json.loads((job.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "done"
    assert meta["warnings"] == job.warnings


def test_retry_recovers_without_placeholder(tmp_path):
    """최초 실패 후 재시도가 성공하면 플레이스홀더/warnings 없이 정상 완료.
    호출 카운트로 재시도 1회가 실제 일어났음을 확인한다."""
    engine = FlakyEngine(fail_calls={1})
    job = _run_job(tmp_path, engine)

    assert job.status == "done"
    assert engine.calls == 3  # 청크1 실패 1 + 재시도 1 + 청크2 1
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    for name in ("p0001_0.jpg", "p0002_0.jpg", "p0003_0.jpg", "p0004_0.jpg"):
        assert f"![](images/{name})" in md
    assert job.warnings == []


def test_all_chunks_failed_is_error(tmp_path):
    """전 청크 실패면 부분 성공이 없으므로 기존대로 status=error."""
    engine = FlakyEngine(fail_all=True)
    job = _run_job(tmp_path, engine)

    assert job.status == "error"
    assert "모든 청크" in job.error
    assert engine.calls == 4  # 청크 2개 × (최초 + 재시도)


def test_job_canceled_is_not_swallowed(tmp_path):
    """JobCanceled는 청크 격리에 삼켜지지 않고 그대로 전파 — 재시도도 없다."""
    engine = FlakyEngine(fail_calls={1}, exc=JobCanceled)
    job = _run_job(tmp_path, engine)

    assert job.status == "canceled"
    assert engine.calls == 1  # 재시도 없이 즉시 취소 처리


def test_per_page_mode_failed_page_placeholder(tmp_path):
    """per_page 모드(single 청크)도 동일하게 격리된다 — 1페이지만 플레이스홀더."""
    engine = FlakyEngine(fail_calls={1, 2})
    job = _run_job(tmp_path, engine, pages=2, mode="per_page")

    assert job.status == "done"
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert md.count(FAILED_MARK) == 1
    assert "![](images/p0002_0.jpg)" in md
    assert len(job.warnings) == 1 and "1페이지" in job.warnings[0]
