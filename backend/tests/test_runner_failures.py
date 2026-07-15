"""청크 실패 격리(runner.py) 검증 — 한 청크가 죽어도 잡 전체가 죽지 않는다.

FlakyEngine: FakeEngine을 상속해 지정한 호출 번호(1-based)에서만 예외를 던진다.
호출 카운트로 재시도 횟수까지 검증한다.
"""

import json
import threading
from pathlib import Path

from app.config import Settings
from app.engine.base import JobCanceled, RepetitiveOutputError
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


class LoopFallbackEngine(FakeEngine):
    """첫 multi 호출을 반복 오류로 만들고 single 폴백 호출을 기록한다."""

    def __init__(
        self,
        *,
        fail_single_pages=(),
        fail_single_once_pages=(),
        loop_single_pages=(),
        cancel_on_multi_loop=False,
    ):
        super().__init__(delay=0.0)
        self.multi_calls = 0
        self.single_calls: dict[int, int] = {}
        self.fail_single_pages = set(fail_single_pages)
        self.fail_single_once_pages = set(fail_single_once_pages)
        self.loop_single_pages = set(loop_single_pages)
        self.cancel_on_multi_loop = cancel_on_multi_loop

    @staticmethod
    def _page_number(image_path) -> int:
        return int(Path(image_path).stem.rsplit("_", 1)[-1])

    @staticmethod
    def _write_single_poison(out_dir) -> None:
        images = out_dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        (images / "99.jpg").write_bytes(b"partial single output")
        (out_dir / "result_with_boxes.jpg").write_bytes(b"partial layout")
        (out_dir / "raw_pages.json").write_text(
            json.dumps({"pages": ["partial repeated layout"]}), encoding="utf-8"
        )

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.multi_calls += 1
        if self.multi_calls == 1:
            poison = out_dir / "images" / "page_0_99.jpg"
            poison.parent.mkdir(parents=True, exist_ok=True)
            poison.write_bytes(b"partial multi output")
            if self.cancel_on_multi_loop:
                cancel.set()
            raise RepetitiveOutputError("모의 의미 반복")
        return super().run_multi(image_paths, out_dir, sink, cancel)

    def run_single(self, image_path, out_dir, sink, cancel):
        page = self._page_number(image_path)
        self.single_calls[page] = self.single_calls.get(page, 0) + 1
        if page in self.loop_single_pages:
            self._write_single_poison(out_dir)
            raise RepetitiveOutputError(f"{page}페이지 모의 의미 반복")
        if page in self.fail_single_pages or (
            page in self.fail_single_once_pages and self.single_calls[page] == 1
        ):
            self._write_single_poison(out_dir)
            raise RuntimeError(f"{page}페이지 모의 실패")
        return super().run_single(image_path, out_dir, sink, cancel)


def _run_job(
    tmp_path,
    engine,
    pages=4,
    mode="multi",
    pages_per_chunk=2,
    *,
    embedded_text=True,
):
    """execute_job을 워커 없이 직접 구동 (4페이지 × 청크 2 → 청크 2개 구성)."""
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", mode, dpi=72)
    if embedded_text:
        pdf_bytes = make_pdf_bytes(pages=pages, with_image=False)
    else:
        import fitz

        doc = fitz.open()
        for _ in range(pages):
            doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()
    (job.dir / "source.pdf").write_bytes(pdf_bytes)
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


def test_per_page_mode_failed_page_recovers_from_embedded_text(tmp_path):
    """per_page single 최종 실패도 원본 PDF 텍스트 레이어로 복구한다."""
    engine = FlakyEngine(fail_calls={1, 2})
    job = _run_job(tmp_path, engine, pages=2, mode="per_page")

    assert job.status == "done"
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "PDF 내장 텍스트 레이어" in md and "Sample page 1" in md
    assert "![](images/p0002_0.jpg)" in md
    assert len(job.warnings) == 1 and "텍스트 레이어로 복구" in job.warnings[0]


def test_per_page_mode_without_text_layer_still_uses_placeholder(tmp_path):
    engine = FlakyEngine(fail_calls={1, 2})
    job = _run_job(
        tmp_path,
        engine,
        pages=2,
        mode="per_page",
        embedded_text=False,
    )

    assert job.status == "done"
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert md.count(FAILED_MARK) == 1
    assert len(job.warnings) == 1 and "플레이스홀더" in job.warnings[0]


def test_repetitive_multi_chunk_falls_back_to_single_pages(tmp_path):
    engine = LoopFallbackEngine()
    job = _run_job(tmp_path, engine, pages=4, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.multi_calls == 2  # 반복 난 청크는 같은 multi로 재시도하지 않음
    assert engine.single_calls == {1: 1, 2: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert len(md.split("\n\n---\n\n")) == 4
    for page in range(1, 5):
        assert f"![](images/p{page:04d}_0.jpg)" in md
    assert not (job.dir / "images" / "p0001_99.jpg").exists()
    assert any("반복/출력 상한 감지로 페이지별 재처리" in warning for warning in job.warnings)


def test_single_fallback_failure_recovers_only_that_page_from_pdf_text(tmp_path):
    engine = LoopFallbackEngine(fail_single_pages={2})
    job = _run_job(tmp_path, engine, pages=4, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.single_calls == {1: 1, 2: 2}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "![](images/p0001_0.jpg)" in md
    assert "![](images/p0003_0.jpg)" in md
    assert "![](images/p0004_0.jpg)" in md
    assert "![](images/p0002_0.jpg)" not in md
    assert "Sample page 2" in md and "PDF 내장 텍스트 레이어" in md
    assert not (job.dir / "images" / "p0002_99.jpg").exists()
    assert not (job.dir / "layout" / "page_0002.jpg").exists()
    layout = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    assert 2 not in {page["page"] for page in layout}


def test_single_fallback_repetition_goes_directly_to_pdf_text_without_retry(tmp_path):
    engine = LoopFallbackEngine(loop_single_pages={2})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.multi_calls == 1
    assert engine.single_calls == {1: 1, 2: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "Sample page 2" in md and "PDF 내장 텍스트 레이어" in md
    assert not (job.dir / "images" / "p0002_99.jpg").exists()
    assert any("RepetitiveOutputError" in warning for warning in job.warnings)


def test_single_fallback_retry_discards_first_attempt_artifacts(tmp_path):
    engine = LoopFallbackEngine(fail_single_once_pages={2})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.single_calls == {1: 1, 2: 2}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "![](images/p0002_0.jpg)" in md
    assert not (job.dir / "images" / "p0002_99.jpg").exists()


def test_all_single_fallback_pages_can_all_recover_from_pdf_text(tmp_path):
    engine = LoopFallbackEngine(fail_single_pages={1, 2})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.multi_calls == 1
    assert engine.single_calls == {1: 2, 2: 2}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert md.count("PDF 내장 텍스트 레이어") == 2
    assert "Sample page 1" in md and "Sample page 2" in md


def test_all_single_fallback_pages_without_text_keep_all_failed_contract(tmp_path):
    engine = LoopFallbackEngine(fail_single_pages={1, 2})
    job = _run_job(
        tmp_path,
        engine,
        pages=2,
        pages_per_chunk=2,
        embedded_text=False,
    )

    assert job.status == "error"
    assert "모든 청크" in job.error
    assert engine.multi_calls == 1
    assert engine.single_calls == {1: 2, 2: 2}


def test_per_page_repetition_goes_directly_to_embedded_text_without_retry(tmp_path):
    engine = LoopFallbackEngine(loop_single_pages={1})
    job = _run_job(tmp_path, engine, pages=2, mode="per_page")

    assert job.status == "done"
    assert engine.multi_calls == 0
    assert engine.single_calls == {1: 1, 2: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "Sample page 1" in md and "PDF 내장 텍스트 레이어" in md
    assert not (job.dir / "images" / "p0001_99.jpg").exists()
    assert not (job.dir / "layout" / "page_0001.jpg").exists()


def test_cancel_wins_over_multi_repetition_fallback(tmp_path):
    engine = LoopFallbackEngine(cancel_on_multi_loop=True)
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "canceled"
    assert engine.multi_calls == 1
    assert engine.single_calls == {}
