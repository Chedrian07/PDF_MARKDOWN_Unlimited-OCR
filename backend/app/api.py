"""REST + SSE 라우트. 계약: docs/ARCHITECTURE.md §5"""

from __future__ import annotations

import functools
import json
import queue
import threading
import zipfile
from pathlib import Path

import anyio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response, StreamingResponse

from . import native_ops
from .pipeline.pdf import probe_pdf
from .pipeline.layout import render_layout_html, render_layout_standalone
from .pipeline.render import render_document_html, render_markdown_html

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
def job_markdown(request: Request, job_id: str) -> PlainTextResponse:
    job = _get_job(request, job_id)
    text, partial = _read_markdown(job)
    headers = {"X-Partial": "true"} if partial else {}
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8", headers=headers)


@router.get("/jobs/{job_id}/html")
def job_html(request: Request, job_id: str) -> HTMLResponse:
    job = _get_job(request, job_id)
    text, partial = _read_markdown(job)
    html = render_document_html(
        text,
        f"/api/jobs/{job_id}/files",
        figure_boxes=_load_figure_boxes(job),
        page_separator=_state(request).settings.page_separator,
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
    needs = any(
        isinstance(b, dict) and not b.get("image") and "fs" not in b
        for pg in pages if isinstance(pg, dict)
        for b in (pg.get("blocks") or ())
    )
    if not needs:
        return
    try:
        from .pipeline.pdf_fonts import enrich_layout_fonts

        if enrich_layout_fonts(src, pages):
            import os

            p = job.dir / "layout.json"
            tmp = job.dir / ".layout.json.tmp"
            tmp.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
    except Exception:
        pass  # 백필 실패는 렌더를 막지 않는다 (폴백 휴리스틱으로 표시)


def _load_layout_pages(job) -> list:
    p = job.dir / "layout.json"
    if not p.is_file():
        raise HTTPException(404, "레이아웃 데이터가 없습니다")
    try:
        pages = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(pages, list)
    except Exception as e:
        raise HTTPException(500, "레이아웃 데이터를 읽을 수 없습니다") from e
    _backfill_layout_fonts(job, pages)
    return pages


@router.get("/jobs/{job_id}/layout.html")
def job_layout_download(request: Request, job_id: str) -> HTMLResponse:
    """PDF 대응 standalone HTML 다운로드 — 이미지 base64·KaTeX 인라인 단일 파일."""
    from urllib.parse import quote

    job = _get_job(request, job_id)
    pages = _load_layout_pages(job)
    st = _state(request)
    html = render_layout_standalone(
        pages, job.dir, Path(job.filename).stem or "document",
        st.settings.resolve_frontend_dir(),
    )
    fname = f"{Path(job.filename).stem or 'document'}.layout.html"
    return HTMLResponse(html, headers={
        "Content-Disposition": f"attachment; filename=\"document.layout.html\"; filename*=UTF-8''{quote(fname)}",
    })


@router.get("/jobs/{job_id}/layout")
def job_layout(request: Request, job_id: str) -> HTMLResponse:
    """좌표 기반 레이아웃 뷰(Phase B) — layout.json이 있는 잡만 (없으면 404).
    다단·절대 위치를 best-effort로 재구성한 부가 뷰. 마크다운 뷰와 독립."""
    job = _get_job(request, job_id)
    pages = _load_layout_pages(job)
    return HTMLResponse(render_layout_html(pages, f"/api/jobs/{job_id}/files"))


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
