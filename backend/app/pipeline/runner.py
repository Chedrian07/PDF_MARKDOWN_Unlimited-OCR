"""잡 실행 오케스트레이션: 렌더 → 청크 OCR → 병합. 워커 스레드에서 호출된다."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..engine.base import EngineError, JobCanceled, OCREngine, RepetitiveOutputError
from .merge import ChunkResult, IncrementalMerger
from .pdf import extract_embedded_page_markdown, render_pdf_pages

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


_FAILED_PAGE_MD = "> ⚠️ 이 페이지는 변환에 실패했습니다"


def _empty_device_cache() -> None:
    """실패한 청크 재시도 전 디바이스 캐시 반환 — OOM류 실패 후 가용 메모리 복구.
    (unlimited.py의 _release_device_cache와 동일한 best-effort empty_cache 패턴)"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover — 방어적 (torch 미설치 등)
        pass


def _add_failed_chunk(
    merger: IncrementalMerger,
    work_dir: Path,
    start_page: int,
    num_pages: int,
    single: bool,
    err: Exception,
) -> None:
    """재시도까지 실패한 청크를 기대 페이지 수만큼의 플레이스홀더로 보정.

    merge의 <PAGE> 계약(청크당 num_pages개 페이지)을 그대로 지켜 글로벌 페이지
    번호 정합을 유지하고, warnings에 남긴 뒤 다음 청크로 계속하게 한다."""
    work_dir.mkdir(parents=True, exist_ok=True)  # 엔진이 만들기 전에 실패했을 수 있음
    if single:
        md = _FAILED_PAGE_MD
    else:
        md = "<PAGE>\n" + "\n<PAGE>\n".join([_FAILED_PAGE_MD] * num_pages)
    merger.add_chunk(ChunkResult(work_dir, start_page, num_pages, md, single=single))
    end = start_page + num_pages - 1
    span = f"{start_page}페이지" if num_pages == 1 else f"{start_page}–{end}페이지"
    merger.warnings.append(
        f"{span}: 변환 실패로 플레이스홀더 삽입 ({err.__class__.__name__}: {str(err)[:200]})"
    )


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

        def _try_embedded_text_fallback(
            page_number: int,
            recovery_dir: Path,
            error: Exception,
        ) -> bool:
            """single OCR 최종 실패 페이지를 원본 PDF 텍스트 레이어로 복구."""
            if cancel.is_set():
                raise JobCanceled()
            # 실패한 single 호출이 만든 crop/layout/raw 산출물은 절대 병합하지 않는다.
            shutil.rmtree(recovery_dir, ignore_errors=True)
            page_md = extract_embedded_page_markdown(
                job.dir / "source.pdf", page_number
            )
            if cancel.is_set():
                raise JobCanceled()
            if page_md is None:
                return False
            recovery_dir.mkdir(parents=True, exist_ok=True)
            if cancel.is_set():
                raise JobCanceled()
            merger.add_chunk(
                ChunkResult(recovery_dir, page_number, 1, page_md, single=True)
            )
            message = (
                f"{page_number}페이지: single OCR 실패 후 PDF 내장 텍스트 레이어로 복구 "
                f"(이미지·정밀 레이아웃 제외; {error.__class__.__name__}: "
                f"{str(error)[:160]})"
            )
            merger.warnings.append(message)
            logger.warning("%s", message)
            return True

        done_pages = 0
        failed_chunks = 0
        last_chunk_error: Exception | None = None
        for ci, chunk in enumerate(chunks):
            if cancel.is_set():
                raise JobCanceled()
            start_page = done_pages + 1
            job.progress.update(phase="ocr", chunk=ci + 1, current_page=start_page)
            store.save(job)
            broker.publish_progress(job)

            work_dir = job.dir / "work" / f"chunk_{ci:02d}"

            def _run_engine() -> str:
                sink.set_chunk(start_page)
                if job.mode == "per_page":
                    return engine.run_single(chunk[0], work_dir, sink, cancel)
                return engine.run_multi(chunk, work_dir, sink, cancel)

            def _run_with_retry(
                run: Callable[[], str],
                context: str,
                *,
                reset_output: Callable[[], None] | None = None,
            ) -> str:
                """엔진 호출을 1회 재시도하되 의미 반복은 즉시 호출자에게 넘긴다.

                반복 감지는 재시도해도 같은 내용에서 재발하므로 재시도 대상이
                아니다 — 복구(per_page 강등·텍스트 레이어 폴백)는 호출자 몫."""
                try:
                    return run()
                except JobCanceled:
                    raise
                except RepetitiveOutputError:
                    if cancel.is_set():
                        raise JobCanceled() from None
                    raise
                except Exception as error:  # noqa: BLE001 — 청크 단위 격리
                    first_error = error

                logger.warning(
                    "%s 실패 (%s: %s) — 캐시 해제 후 1회 재시도",
                    context,
                    first_error.__class__.__name__,
                    str(first_error)[:200],
                )
                _empty_device_cache()
                if cancel.is_set():
                    raise JobCanceled() from None
                if reset_output is not None:
                    reset_output()
                try:
                    return run()
                except JobCanceled:
                    raise
                except Exception as retry_error:  # noqa: BLE001 — 청크 단위 격리
                    if cancel.is_set():
                        raise JobCanceled() from None
                    logger.warning(
                        "%s 재시도 실패 (%s: %s)",
                        context,
                        retry_error.__class__.__name__,
                        str(retry_error)[:200],
                    )
                    raise

            def _recover_unsafe_generation_chunk(
                generation_error: RepetitiveOutputError,
            ) -> tuple[bool, Exception]:
                """반복/상한 초과 multi 산출물을 버리고 페이지별 single 재처리."""
                if cancel.is_set():
                    raise JobCanceled()
                end_page = start_page + len(chunk) - 1
                span = (
                    f"{start_page}페이지"
                    if len(chunk) == 1
                    else f"{start_page}–{end_page}페이지"
                )
                logger.warning(
                    "%s multi OCR 비정상 생성 감지 — 페이지별 재처리: %s",
                    span,
                    str(generation_error)[:200],
                )
                merger.warnings.append(
                    f"{span}: 반복/출력 상한 감지로 페이지별 재처리 "
                    f"({str(generation_error)[:200]})"
                )

                # multi와 single은 이미지/레이아웃 파일명 규약이 다르다. 부분 multi
                # 결과를 병합하지 않도록 제거하고 페이지마다 독립 디렉터리를 쓴다.
                shutil.rmtree(work_dir, ignore_errors=True)
                failed_pages = 0
                last_error: Exception = generation_error
                for local_page, image_path in enumerate(chunk):
                    if cancel.is_set():
                        raise JobCanceled()
                    global_page = start_page + local_page
                    page_dir = work_dir / "fallback" / f"page_{local_page:02d}"

                    def _run_page() -> str:
                        sink.set_chunk(global_page)
                        return engine.run_single(image_path, page_dir, sink, cancel)

                    try:
                        page_md = _run_with_retry(
                            _run_page,
                            f"{global_page}페이지 fallback OCR",
                            reset_output=lambda directory=page_dir: shutil.rmtree(
                                directory, ignore_errors=True
                            ),
                        )
                    except JobCanceled:
                        raise
                    except Exception as page_error:  # noqa: BLE001 — 페이지 단위 격리
                        last_error = page_error
                        if not _try_embedded_text_fallback(
                            global_page, page_dir, page_error
                        ):
                            failed_pages += 1
                            _add_failed_chunk(
                                merger, page_dir, global_page, 1, True, page_error
                            )
                    else:
                        merger.add_chunk(
                            ChunkResult(page_dir, global_page, 1, page_md, single=True)
                        )
                return failed_pages == len(chunk), last_error

            # 청크 단위 격리: 한 청크가 죽어도(OOM·벤더 예외 등) 잡 전체를 죽이지
            # 않는다 — 캐시 해제 후 1회 재시도, 그래도 실패하면 플레이스홀더로
            # 보정하고 계속. 취소(JobCanceled)는 절대 삼키지 않고 그대로 전파.
            # 재시도는 엔진 실행만 감싼다 — add_chunk는 비멱등(pages_md 확장)이라
            # 재시도에 포함하면 병합 도중 실패 시 페이지가 중복 병합될 수 있다.
            md: str | None = None
            try:
                md = _run_with_retry(
                    _run_engine,
                    f"청크 {ci + 1}/{len(chunks)}",
                    reset_output=lambda: shutil.rmtree(work_dir, ignore_errors=True),
                )
            except JobCanceled:
                raise
            except RepetitiveOutputError as repetition_error:
                if job.mode != "per_page":
                    all_failed, last_error = _recover_unsafe_generation_chunk(
                        repetition_error
                    )
                    if all_failed:
                        failed_chunks += 1
                        last_chunk_error = last_error
                else:
                    if not _try_embedded_text_fallback(
                        start_page, work_dir, repetition_error
                    ):
                        failed_chunks += 1
                        last_chunk_error = repetition_error
                        _add_failed_chunk(
                            merger, work_dir, start_page, len(chunk), True, repetition_error
                        )
            except Exception as chunk_error:  # noqa: BLE001 — 청크 단위 격리
                recovered = job.mode == "per_page" and _try_embedded_text_fallback(
                    start_page, work_dir, chunk_error
                )
                if not recovered:
                    logger.warning(
                        "청크 %d/%d 최종 실패 (%s: %s) — 플레이스홀더로 보정 후 계속",
                        ci + 1,
                        len(chunks),
                        chunk_error.__class__.__name__,
                        str(chunk_error)[:200],
                    )
                    failed_chunks += 1
                    last_chunk_error = chunk_error
                    shutil.rmtree(work_dir, ignore_errors=True)
                    _add_failed_chunk(
                        merger,
                        work_dir,
                        start_page,
                        len(chunk),
                        job.mode == "per_page",
                        chunk_error,
                    )
            if md is not None:
                # 병합은 엔진 성공 시 1회만 — 병합 실패는 잡 레벨 IO 오류로 취급한다.
                merger.add_chunk(
                    ChunkResult(work_dir, start_page,
                                1 if job.mode == "per_page" else len(chunk),
                                md, single=job.mode == "per_page")
                )
            sink.flush()
            # 취소돼도 이 청크의 부분 출력까지는 병합 후에 중단한다
            if cancel.is_set():
                raise JobCanceled()

            done_pages += len(chunk)
            # 단조 가드 — 마커 과잉 생성으로 sink가 선행시킨 진행률을 되돌리지 않는다
            job.progress["current_page"] = max(
                job.progress.get("current_page", 0), done_pages
            )
            store.save(job)
            broker.publish_progress(job)

        # 전 청크 실패면 부분 성공이 없으므로 기존대로 잡 오류로 마감
        if chunks and failed_chunks == len(chunks):
            raise EngineError(
                f"모든 청크({len(chunks)}개) 변환에 실패했습니다: {last_chunk_error}"
            ) from last_chunk_error

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
        else:
            # 터미널 상태(done/error/canceled) 마감 시 work/ 정리 — 필요 산출물
            # (images/·layout/·result.md·layout.json)은 add_chunk 시점에 이미 잡
            # 루트로 이동/기록됐고, work/에는 boxes.json·raw_pages.json·실패 청크
            # 잔여물만 남아 잡마다 무한 축적된다.
            shutil.rmtree(job.dir / "work", ignore_errors=True)
