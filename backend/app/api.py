"""REST + SSE 라우트. 계약: docs/ARCHITECTURE.md §5"""

from __future__ import annotations

import functools
import json
import os
import queue
import threading
import zipfile
from pathlib import Path

import anyio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from . import native_ops
from .pipeline.pdf import probe_pdf
from .pipeline.layout import render_layout_html, render_layout_standalone
from .pipeline.render import render_document_html, render_markdown_html
# 번역 코어는 아직 스켈레톤(run_translation은 NotImplementedError)이지만 import는 가능.
# 테스트는 app.api.run_translation을 몽키패치로 대체한다.
from .translate import SUPPORTED_LANGS, TranslateConfig, TranslateError, run_translation

router = APIRouter(prefix="/api")

_ALLOWED_FILE_DIRS = ("pages", "images", "layout")
_UPLOAD_CHUNK = 1024 * 1024


def _state(request: Request):
    return request.app.state


def _get_job(request: Request, job_id: str):
    job = _state(request).store.get(job_id)
    if job is None:
        raise HTTPException(404, "잡을 찾을 수 없습니다")
    return job


# ── 번역(한국어) 공용 헬퍼 ────────────────────────────────────────────────
def _check_lang(lang: str) -> None:
    """구체 lang 값 검증 (쿼리에서는 None(원본)을 먼저 걸러낸 뒤 호출)."""
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(400, "지원하지 않는 언어")


def _translate_dir(job, lang: str) -> Path:
    return job.dir / "translations" / lang


def _translate_channel(job_id: str, lang: str) -> str:
    """번역 SSE 브로커 채널 키 — OCR 잡 이벤트(job_id)와 네임스페이스를 분리한다."""
    return f"{job_id}:translate:{lang}"


def _translated_markdown_or_404(job, lang: str) -> str:
    p = job.dir / f"result.{lang}.md"
    if not p.is_file():
        raise HTTPException(404, "한국어 번역본이 없습니다 — 먼저 번역을 실행하세요")
    return p.read_text(encoding="utf-8")


def _read_translate_state(job, lang: str) -> dict | None:
    """translations/{lang}/state.json 로드 (없거나 손상되면 None)."""
    p = _translate_dir(job, lang) / "state.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_translate_state(job, lang: str, state: dict) -> None:
    d = _translate_dir(job, lang)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / ".state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / "state.json")


def _stale_adjusted_state(request: Request, job, lang: str) -> dict | None:
    """state.json을 읽되 stale-running을 조정한다: status=="running"인데 레지스트리에
    태스크가 없으면(서버 재시작 등) error로 원자적 재기록 후 반환. 파일이 없으면 None."""
    st = _state(request)
    state = _read_translate_state(job, lang)
    if state is None:
        return None
    if state.get("status") == "running":
        with st.translate_lock:
            alive = (job.id, lang) in st.translate_tasks
        if not alive:
            state["status"] = "error"
            state["error"] = "서버가 재시작되어 번역이 중단되었습니다 — 다시 실행하세요"
            _write_translate_state(job, lang, state)
    return state


def _run_translate_thread(
    st, job, lang: str, cfg: TranslateConfig,
    cancel: threading.Event, force: bool, page_separator: str,
) -> None:
    """번역 워커 스레드 본문. 진행/완료/오류를 브로커 채널로 중계하고 레지스트리를 정리한다.
    state.json은 run_translation(엔진)이 직접 기록하므로 여기서는 이벤트만 발행한다."""
    broker = st.broker
    channel = _translate_channel(job.id, lang)

    def _progress(current: int, total: int) -> None:
        broker.publish(channel, "progress", {
            "phase": "translate", "lang": lang,
            "current": current, "total": total, "status": "running",
        })

    try:
        result = run_translation(
            job.dir, lang, cfg,
            page_separator=page_separator,
            progress=_progress,
            cancel=cancel,
            force=force,
        )
        if getattr(result, "status", None) == "canceled":
            broker.publish(channel, "error", {
                "message": "번역이 취소되었습니다", "canceled": True,
            })
        else:
            broker.publish(channel, "done", {
                "phase": "translate", "lang": lang,
                "markdown_url": f"/api/jobs/{job.id}/markdown?lang={lang}",
                "html_url": f"/api/jobs/{job.id}/html?lang={lang}",
                "layout_url": f"/api/jobs/{job.id}/layout?lang={lang}",
                "counts": {
                    "total": getattr(result, "total", 0),
                    "translated": getattr(result, "translated", 0),
                    "cached": getattr(result, "cached", 0),
                    "skipped": getattr(result, "skipped", 0),
                    "kept_original": len(getattr(result, "kept_original", []) or []),
                },
            })
            # ko 번역본을 포함해 다시 만들도록 archive.zip 캐시를 무효화한다.
            (job.dir / "archive.zip").unlink(missing_ok=True)
    except TranslateError as e:
        broker.publish(channel, "error", {"message": str(e), "canceled": False})
    except Exception as e:  # noqa: BLE001 — 스레드가 조용히 죽지 않도록 SSE로 중계
        broker.publish(channel, "error", {"message": str(e), "canceled": False})
    finally:
        with st.translate_lock:
            st.translate_tasks.pop((job.id, lang), None)


@router.get("/health")
def health(request: Request) -> dict:
    st = _state(request)
    engine = st.engine
    return {
        "status": "ok",
        "engine": engine.name,
        "device": engine.device,
        "dtype": engine.dtype_name or st.settings.dtype,
        "model_id": st.settings.model_id,
        "model_loaded": engine.loaded,
        # 프리로드 실패 후 워커 재시도가 성공하면 과거 오류는 더 이상 유효하지 않다
        "model_load_error": None if engine.loaded else st.load_state.get("error"),
        "gpu_name": engine.gpu_name(),
        "native_ops": native_ops.HAVE_NATIVE,
    }


@router.post("/jobs", status_code=202)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("multi"),
    dpi: int | None = Form(None),
) -> dict:
    st = _state(request)
    settings = st.settings

    if mode not in ("multi", "per_page"):
        raise HTTPException(400, "mode는 multi 또는 per_page 여야 합니다")
    dpi = dpi if dpi is not None else settings.render_dpi
    if not (72 <= dpi <= 400):
        raise HTTPException(400, "dpi는 72–400 범위여야 합니다")
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF 파일만 업로드할 수 있습니다")

    job = st.store.create(filename=filename, mode=mode, dpi=dpi)
    dest = job.dir / "source.pdf"
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, f"업로드 상한({settings.max_upload_mb}MB)을 초과했습니다")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "빈 파일입니다")
        with dest.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise HTTPException(400, "PDF 형식이 아닙니다")
        try:
            probe_pdf(dest, settings.max_pages)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    except HTTPException:
        st.store.delete_dir(job)
        raise

    st.worker.submit(job)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs")
def list_jobs(request: Request) -> dict:
    return {"jobs": [j.to_dict() for j in _state(request).store.list()]}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    return _get_job(request, job_id).to_dict()


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_poll(q: queue.Queue):
    try:
        return q.get(timeout=1.0)
    except queue.Empty:
        return None


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    job = _get_job(request, job_id)
    broker = _state(request).broker

    async def gen():
        q = broker.subscribe(job_id)
        try:
            yield "retry: 3000\n\n"
            # 접속 시 스냅샷
            if job.status == "done":
                yield _sse_format("done", {
                    "markdown_url": f"/api/jobs/{job_id}/markdown",
                    "archive_url": f"/api/jobs/{job_id}/archive",
                })
                return
            if job.status in ("error", "canceled"):
                yield _sse_format("error", {
                    "message": job.error or "오류",
                    "canceled": job.status == "canceled",
                })
                return
            yield _sse_format("progress", {**job.progress, "status": job.status})

            idle = 0
            while True:
                if await request.is_disconnected():
                    return
                item = await anyio.to_thread.run_sync(functools.partial(_sse_poll, q))
                if item is None:
                    idle += 1
                    if idle >= 15:
                        idle = 0
                        yield ": ping\n\n"
                    continue
                idle = 0
                event, data = item
                yield _sse_format(event, data)
                if event in ("done", "error"):
                    return
        finally:
            broker.unsubscribe(job_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _read_markdown(job) -> tuple[str, bool]:
    md_path = job.dir / "result.md"
    text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    return text, job.status != "done"


def _load_figure_boxes(job) -> dict | None:
    """벤더 P13 → merge가 통합한 images/boxes.json (없으면 풀폭 폴백)."""
    p = job.dir / "images" / "boxes.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@router.get("/jobs/{job_id}/markdown")
def job_markdown(request: Request, job_id: str, lang: str | None = None) -> PlainTextResponse:
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
        text = _translated_markdown_or_404(job, lang)
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")
    text, partial = _read_markdown(job)
    headers = {"X-Partial": "true"} if partial else {}
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8", headers=headers)


@router.get("/jobs/{job_id}/html")
def job_html(request: Request, job_id: str, lang: str | None = None) -> HTMLResponse:
    job = _get_job(request, job_id)
    base = f"/api/jobs/{job_id}/files"
    sep = _state(request).settings.page_separator
    if lang is not None:
        _check_lang(lang)
        text = _translated_markdown_or_404(job, lang)
        html = render_document_html(
            text, base, figure_boxes=_load_figure_boxes(job), page_separator=sep,
        )
        return HTMLResponse(html)
    text, partial = _read_markdown(job)
    html = render_document_html(
        text, base, figure_boxes=_load_figure_boxes(job), page_separator=sep,
    )
    headers = {"X-Partial": "true"} if partial else {}
    return HTMLResponse(html, headers=headers)


def _backfill_layout_fonts(job, pages: list) -> None:
    """기존 잡 지연 백필: layout.json에 실측 폰트 크기(fs)가 빠진 비이미지 블록이
    있고 source.pdf가 있으면, 텍스트 레이어에서 뽑아 in-place 주입 후 원자적 저장.
    재변환 없이 이미 변환된 잡도 개선된다. enrichment 실패는 절대 500을 내지 않음."""
    src = job.dir / "source.pdf"
    if not src.exists():
        return
    try:
        from .pipeline.pdf_fonts import ENRICH_VERSION, enrich_layout_fonts
    except Exception:
        return
    # 버전 스탬프 기반: enrichment 스키마가 갱신되면(예: 세로쓰기 감지 추가)
    # 기존 잡도 1회 재백필된다. 스탬프가 최신이면 매 요청 재스캔하지 않는다.
    needs = any(
        isinstance(pg, dict) and int(pg.get("fonts_v") or 0) < ENRICH_VERSION
        for pg in pages
    )
    if not needs:
        return
    try:
        if enrich_layout_fonts(src, pages):
            import os

            p = job.dir / "layout.json"
            tmp = job.dir / ".layout.json.tmp"
            tmp.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
    except Exception:
        pass  # 백필 실패는 렌더를 막지 않는다 (폴백 휴리스틱으로 표시)


def _load_layout_pages(job, lang: str | None = None) -> list:
    """lang=None이면 원본 layout.json, lang이면 번역본 layout.{lang}.json을 로드."""
    if lang is not None:
        p = job.dir / f"layout.{lang}.json"
        missing = "한국어 번역본이 없습니다 — 먼저 번역을 실행하세요"
    else:
        p = job.dir / "layout.json"
        missing = "레이아웃 데이터가 없습니다"
    if not p.is_file():
        raise HTTPException(404, missing)
    try:
        pages = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(pages, list)
    except Exception as e:
        raise HTTPException(500, "레이아웃 데이터를 읽을 수 없습니다") from e
    # 폰트 백필은 원본 layout.json만 대상: 번역본 페이지는 엔진이 fonts_v 스탬프를
    # 복사해 두므로 no-op이고, _backfill_layout_fonts는 결과를 원본 경로에 쓰기 때문에
    # 번역본에 실행하면 안 된다.
    if lang is None:
        _backfill_layout_fonts(job, pages)
    return pages


@router.get("/jobs/{job_id}/layout.html")
def job_layout_download(request: Request, job_id: str, lang: str | None = None) -> HTMLResponse:
    """PDF 대응 standalone HTML 다운로드 — 이미지 base64·KaTeX 인라인 단일 파일.
    lang=ko면 번역본 layout.ko.json으로 렌더하고 파일명에 .ko.를 붙인다."""
    from urllib.parse import quote

    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
    pages = _load_layout_pages(job, lang)
    st = _state(request)
    stem = Path(job.filename).stem or "document"
    html = render_layout_standalone(
        pages, job.dir, stem, st.settings.resolve_frontend_dir(), lang=lang,
    )
    suffix = f".{lang}" if lang else ""
    fname = f"{stem}{suffix}.layout.html"
    return HTMLResponse(html, headers={
        "Content-Disposition":
            f"attachment; filename=\"document{suffix}.layout.html\"; filename*=UTF-8''{quote(fname)}",
    })


@router.get("/jobs/{job_id}/layout")
def job_layout(request: Request, job_id: str, lang: str | None = None) -> HTMLResponse:
    """좌표 기반 레이아웃 뷰(Phase B) — layout.json이 있는 잡만 (없으면 404).
    다단·절대 위치를 best-effort로 재구성한 부가 뷰. 마크다운 뷰와 독립.
    lang=ko면 번역본 layout.ko.json을 로드하고 컨테이너에 lang="ko"를 부여한다."""
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
    pages = _load_layout_pages(job, lang)
    return HTMLResponse(render_layout_html(pages, f"/api/jobs/{job_id}/files", lang=lang))


@router.get("/jobs/{job_id}/files/{file_path:path}")
def job_file(request: Request, job_id: str, file_path: str) -> FileResponse:
    job = _get_job(request, job_id)
    parts = Path(file_path).parts
    if not parts or parts[0] not in _ALLOWED_FILE_DIRS:
        raise HTTPException(404, "파일을 찾을 수 없습니다")
    full = (job.dir / file_path).resolve()
    if not full.is_relative_to(job.dir.resolve()) or not full.is_file():
        raise HTTPException(404, "파일을 찾을 수 없습니다")
    return FileResponse(full)


@router.get("/jobs/{job_id}/archive")
def job_archive(request: Request, job_id: str) -> FileResponse:
    job = _get_job(request, job_id)
    if job.status != "done":
        raise HTTPException(409, "아직 변환이 완료되지 않았습니다")
    zip_path = job.dir / "archive.zip"
    if not zip_path.is_file():
        tmp = job.dir / ".archive.zip.tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            md = job.dir / "result.md"
            if md.is_file():
                zf.write(md, "result.md")
            # 번역본(result.ko.md 등)도 포함 — 번역 완료 시 이 zip 캐시가 삭제돼
            # 다음 요청에서 번역본까지 담아 재생성된다. (glob은 result.md 자신은 제외)
            for extra in sorted(job.dir.glob("result.*.md")):
                zf.write(extra, extra.name)
            images = job.dir / "images"
            if images.is_dir():
                for f in sorted(images.iterdir()):
                    if f.is_file():
                        zf.write(f, f"images/{f.name}")
        tmp.replace(zip_path)
    stem = Path(job.filename).stem or "result"
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{stem}.markdown.zip",
    )


@router.post("/jobs/{job_id}/cancel", status_code=202)
def cancel_job(request: Request, job_id: str) -> dict:
    """삭제 없이 중단 — 부분 결과(result.md, 완료된 청크의 이미지)는 보존된다."""
    st = _state(request)
    job = _get_job(request, job_id)
    if job.status in ("done", "error", "canceled"):
        return {"job_id": job_id, "status": job.status}
    st.cancel_events.setdefault(job_id, threading.Event()).set()
    return {"job_id": job_id, "status": "canceling"}


# ── 번역(한국어) 라우트 ───────────────────────────────────────────────────
@router.post("/jobs/{job_id}/translate")
async def translate_start(request: Request, job_id: str) -> JSONResponse:
    """번역 시작. body JSON {"lang":"ko","force":false} (기본 lang="ko").

    반환: 이미 실행 중이면 200 {"status":"running"}, state가 done이고 force가
    아니면 200 {"status":"done"}, 그 외에는 데몬 스레드를 띄우고 202 {"status":"running"}.
    """
    st = _state(request)
    job = _get_job(request, job_id)
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        body = {}
    lang = body.get("lang") or "ko"
    force = bool(body.get("force", False))

    _check_lang(lang)
    if job.status != "done":
        raise HTTPException(409, "변환이 완료된 잡만 번역할 수 있습니다")
    try:
        cfg = TranslateConfig.from_env()
    except TranslateError as e:
        raise HTTPException(503, str(e)) from e

    with st.translate_lock:
        if (job_id, lang) in st.translate_tasks:
            return JSONResponse(
                {"job_id": job_id, "lang": lang, "status": "running"}, status_code=200
            )
        state = _read_translate_state(job, lang)
        if not force and state is not None and state.get("status") == "done":
            return JSONResponse(
                {"job_id": job_id, "lang": lang, "status": "done"}, status_code=200
            )
        cancel = threading.Event()
        thread = threading.Thread(
            target=_run_translate_thread,
            args=(st, job, lang, cfg, cancel, force, st.settings.page_separator),
            name=f"translate-{job_id}-{lang}", daemon=True,
        )
        st.translate_tasks[(job_id, lang)] = {"thread": thread, "cancel": cancel}
        thread.start()
    return JSONResponse({"job_id": job_id, "lang": lang, "status": "running"}, status_code=202)


@router.get("/jobs/{job_id}/translate/state")
def translate_state(request: Request, job_id: str, lang: str = "ko") -> dict:
    """번역 상태. 없으면 {"status":"none","lang"}. stale-running은 error로 조정해 반환."""
    job = _get_job(request, job_id)
    _check_lang(lang)
    state = _stale_adjusted_state(request, job, lang)
    if state is None:
        return {"status": "none", "lang": lang}
    return state


@router.post("/jobs/{job_id}/translate/cancel")
def translate_cancel(request: Request, job_id: str, lang: str = "ko") -> JSONResponse:
    """실행 중이면 cancel 이벤트를 set하고 202 canceling, 아니면 현재 상태를 반환."""
    st = _state(request)
    job = _get_job(request, job_id)
    _check_lang(lang)
    with st.translate_lock:
        task = st.translate_tasks.get((job_id, lang))
        if task is not None:
            task["cancel"].set()
            return JSONResponse(
                {"job_id": job_id, "lang": lang, "status": "canceling"}, status_code=202
            )
    state = _stale_adjusted_state(request, job, lang)
    if state is None:
        return JSONResponse(
            {"job_id": job_id, "lang": lang, "status": "none"}, status_code=200
        )
    return JSONResponse(state, status_code=200)


@router.get("/jobs/{job_id}/translate/events")
async def translate_events(request: Request, job_id: str, lang: str = "ko") -> StreamingResponse:
    """번역 진행 SSE (job_events와 동일 패턴). 스냅샷: done→done 후 종료, error/canceled→
    error 후 종료, running→progress 스냅샷 후 구독 루프. state가 없으면 404."""
    st = _state(request)
    job = _get_job(request, job_id)
    _check_lang(lang)
    if _stale_adjusted_state(request, job, lang) is None:
        raise HTTPException(404, "번역 상태가 없습니다")
    broker = st.broker
    channel = _translate_channel(job_id, lang)

    def _done_data() -> dict:
        return {
            "phase": "translate", "lang": lang,
            "markdown_url": f"/api/jobs/{job_id}/markdown?lang={lang}",
            "html_url": f"/api/jobs/{job_id}/html?lang={lang}",
            "layout_url": f"/api/jobs/{job_id}/layout?lang={lang}",
        }

    async def gen():
        # 구독 먼저, 그 다음 스냅샷 재조회 — job_events와 같은 순서라 구독~완료 사이 이벤트를
        # 놓치지 않는다 (엔진이 state를 먼저 쓰고 스레드가 이후 done/error를 발행하므로,
        # 스냅샷이 running이면 종료 이벤트는 아직 큐로 들어온다).
        q = broker.subscribe(channel)
        try:
            yield "retry: 3000\n\n"
            state = _stale_adjusted_state(request, job, lang) or {"status": "none"}
            status = state.get("status")
            if status == "done":
                yield _sse_format("done", _done_data())
                return
            if status in ("error", "canceled"):
                yield _sse_format("error", {
                    "message": state.get("error") or "오류",
                    "canceled": status == "canceled",
                })
                return
            yield _sse_format("progress", {
                "phase": "translate", "lang": lang,
                "current": state.get("current") or 0,
                "total": state.get("total") or 0,
                "status": "running",
            })

            idle = 0
            while True:
                if await request.is_disconnected():
                    return
                item = await anyio.to_thread.run_sync(functools.partial(_sse_poll, q))
                if item is None:
                    idle += 1
                    if idle >= 15:
                        idle = 0
                        yield ": ping\n\n"
                    continue
                idle = 0
                event, data = item
                yield _sse_format(event, data)
                if event in ("done", "error"):
                    return
        finally:
            broker.unsubscribe(channel, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/jobs/{job_id}/render-preview")
async def render_preview(request: Request, job_id: str) -> HTMLResponse:
    """클라이언트가 보낸 (정리된) 마크다운을 안전 렌더 — 라이브 미리보기용.
    /html과 동일한 렌더러라 XSS 이스케이프·표 복원·이미지 URL 재작성이 적용된다."""
    job = _get_job(request, job_id)
    body = await request.body()
    if len(body) > 2_000_000:
        raise HTTPException(413, "미리보기 본문이 너무 큽니다 (2MB 초과)")
    text = body.decode("utf-8", "replace")
    return HTMLResponse(
        render_markdown_html(
            text, f"/api/jobs/{job_id}/files", figure_boxes=_load_figure_boxes(job)
        )
    )


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(request: Request, job_id: str) -> Response:
    st = _state(request)
    job = _get_job(request, job_id)
    job.delete_requested = True
    ev = st.cancel_events.get(job_id)
    if ev is not None:
        ev.set()
    if job.status != "running":
        # queued 잡은 워커가 dequeue 시 delete_requested를 보고 정리하지만,
        # 디렉터리와 목록은 지금 바로 제거해 UI에서 사라지게 한다.
        st.store.delete_dir(job)
    return Response(status_code=204)
