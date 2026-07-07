"""번역 API 레이어 테스트.

translate 코어(run_translation)는 아직 스켈레톤이므로 app.api.run_translation을
몽키패치로 대체한다. 페이크는 계약대로 동작한다: progress 콜백 호출,
translations/{lang}/state.json·result.ko.md·layout.ko.json 기록, TranslateResult 반환.
"""

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest
from conftest import wait_done

from app.translate import TranslateResult


# ── 페이크 run_translation ────────────────────────────────────────────────
def _make_fake(*, gate: threading.Event | None = None, wait_cancel: bool = False,
               total: int = 2):
    """계약을 지키는 페이크 run_translation을 만든다.

    gate: 주어지면 running 상태를 쓴 뒤 이 이벤트가 set될 때까지 블록(진행 관찰용).
    wait_cancel: True면 cancel 이벤트를 기다렸다가 취소로 종료.
    """
    md = "# 번역본\n\n안녕하세요. 번역된 문서입니다.\n"
    layout = [{
        "page": 1, "width": 1000, "height": 1414,
        "blocks": [{"type": "text", "bbox": [20, 20, 980, 120],
                    "content": "안녕하세요", "fs": 1.8, "fonts_v": 9999}],
    }]

    def fake(job_dir, lang, cfg, *, page_separator="\n\n---\n\n",
             progress=None, cancel=None, force=False, client=None):
        job_dir = Path(job_dir)
        tdir = job_dir / "translations" / lang
        tdir.mkdir(parents=True, exist_ok=True)

        def write_state(**over):
            base = {
                "lang": lang, "status": "running", "current": 0, "total": total,
                "error": None, "model": getattr(cfg, "model", ""),
                "api_mode": getattr(cfg, "api_mode", ""), "prompt_v": "1",
                "started_at": "2026-07-07T00:00:00+00:00", "finished_at": None,
            }
            base.update(over)
            (tdir / "state.json").write_text(
                json.dumps(base, ensure_ascii=False), encoding="utf-8")

        write_state(status="running", current=0, total=total)

        if wait_cancel:
            if cancel is not None:
                cancel.wait(timeout=10)
                if cancel.is_set():
                    write_state(status="canceled", current=0, error="번역이 취소되었습니다",
                                finished_at="2026-07-07T00:00:01+00:00")
                    return TranslateResult(status="canceled", total=total)

        if gate is not None:
            gate.wait(timeout=10)

        if progress is not None:
            progress(1, total)
            write_state(status="running", current=1, total=total)
            progress(total, total)

        (job_dir / f"result.{lang}.md").write_text(md, encoding="utf-8")
        (job_dir / f"layout.{lang}.json").write_text(
            json.dumps(layout, ensure_ascii=False), encoding="utf-8")
        write_state(status="done", current=total, total=total,
                    finished_at="2026-07-07T00:00:02+00:00")
        return TranslateResult(status="done", total=total, translated=total, cached=0)

    return fake


# ── 헬퍼 ──────────────────────────────────────────────────────────────────
@pytest.fixture
def provider_env(monkeypatch):
    """번역 프로바이더 env 설정 (TranslateConfig.from_env 통과용)."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    return monkeypatch


def _done_job(client, sample_pdf) -> str:
    r = client.post("/api/jobs", files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
                    data={"mode": "multi"})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    return jid


def _tstate(client, jid, lang="ko") -> dict:
    return client.get(f"/api/jobs/{jid}/translate/state?lang={lang}").json()


def _wait_until_status(client, jid, want, lang="ko", timeout=5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = _tstate(client, jid, lang)
        if body.get("status") == want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"상태가 {want}가 되지 않음: {_tstate(client, jid, lang)}")


def _wait_no_task(client, jid, lang="ko", timeout=10.0) -> None:
    """번역 데몬 스레드가 완전히 끝날 때까지 대기(레지스트리에서 제거 = finally 실행 완료)."""
    tasks = client.app.state.translate_tasks
    lock = client.app.state.translate_lock
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with lock:
            present = (jid, lang) in tasks
        if not present:
            return
        time.sleep(0.02)
    raise AssertionError("번역 스레드가 시간 내에 종료되지 않음")


def _collect_sse(client, url, max_lines=500):
    """SSE 스트림을 (event, data_dict) 목록으로 수집. done/error에서 종료."""
    out = []
    with client.stream("GET", url) as s:
        cur = None
        seen = 0
        for line in s.iter_lines():
            seen += 1
            if seen > max_lines:
                break
            if line.startswith("event: "):
                cur = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                out.append((cur, json.loads(line.removeprefix("data: "))))
                if cur in ("done", "error"):
                    break
    return out


# ── 1. POST → 202 → events SSE progress…done ──────────────────────────────
def test_translate_post_then_events_stream(client, sample_pdf, provider_env, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr("app.api.run_translation", _make_fake(gate=gate))
    jid = _done_job(client, sample_pdf)

    r = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"})
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "running"

    # 페이크가 running을 기록하고 게이트에서 대기할 때까지 (state가 있어야 events가 404 안 남)
    _wait_until_status(client, jid, "running")

    events = []
    with client.stream("GET", f"/api/jobs/{jid}/translate/events?lang=ko") as s:
        for line in s.iter_lines():
            if line.startswith("event: "):
                ev = line.removeprefix("event: ").strip()
                events.append(ev)
                # 스냅샷 progress를 받은 뒤 페이크를 해제 → live progress/done이 이어진다
                if ev == "progress" and not gate.is_set():
                    gate.set()
            if "event: done" in line or "event: error" in line:
                break

    assert "progress" in events, events
    assert "done" in events, events
    _wait_no_task(client, jid)
    # done 이벤트 시점에 산출물이 존재한다
    assert client.get(f"/api/jobs/{jid}/markdown?lang=ko").status_code == 200


# ── 2. done 후 재-POST 200; force=true 재실행 ─────────────────────────────
def test_translate_idempotent_and_force(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)
    assert _tstate(client, jid)["status"] == "done"

    # 재-POST (force 없음) → 재실행 없이 200 done
    r2 = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "done"

    # force=true → 재실행 (게이트로 running 확정 관찰)
    gate = threading.Event()
    monkeypatch.setattr("app.api.run_translation", _make_fake(gate=gate))
    r3 = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko", "force": True})
    assert r3.status_code == 202
    assert r3.json()["status"] == "running"
    _wait_until_status(client, jid, "running")
    gate.set()
    _wait_no_task(client, jid)
    assert _tstate(client, jid)["status"] == "done"


# ── 3. /markdown·/html·/layout·/layout.html?lang=ko — 전 404 / 후 200 ─────
def test_translated_output_routes(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)

    for path in (f"/api/jobs/{jid}/markdown?lang=ko",
                 f"/api/jobs/{jid}/html?lang=ko",
                 f"/api/jobs/{jid}/layout?lang=ko",
                 f"/api/jobs/{jid}/layout.html?lang=ko"):
        assert client.get(path).status_code == 404, path

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)
    assert _tstate(client, jid)["status"] == "done"

    md = client.get(f"/api/jobs/{jid}/markdown?lang=ko")
    assert md.status_code == 200
    assert "번역본" in md.text
    assert "X-Partial" not in md.headers

    html = client.get(f"/api/jobs/{jid}/html?lang=ko")
    assert html.status_code == 200
    assert "X-Partial" not in html.headers

    layout = client.get(f"/api/jobs/{jid}/layout?lang=ko")
    assert layout.status_code == 200
    assert 'lang="ko"' in layout.text
    assert 'class="doclayout-body"' in layout.text

    dl = client.get(f"/api/jobs/{jid}/layout.html?lang=ko")
    assert dl.status_code == 200
    assert dl.text.startswith("<!doctype html>")
    assert 'lang="ko"' in dl.text
    assert ".ko.layout.html" in dl.headers["content-disposition"]


# ── 4. 400 잘못된 lang / 409 미완료 잡 / 503 env 미설정 ────────────────────
def test_translate_validation_errors(client, sample_pdf, monkeypatch):
    jid = _done_job(client, sample_pdf)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_MODEL", "m")

    # 400 — 지원하지 않는 언어
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "fr"}).status_code == 400

    # 409 — 완료되지 않은 잡 (상태를 강제로 비-done 처리)
    job = client.app.state.store.get(jid)
    old = job.status
    job.status = "running"
    try:
        assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 409
    finally:
        job.status = old

    # 503 — 프로바이더 env 미설정
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATE_MODEL", raising=False)
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 503


# ── 5. cancel → error 이벤트 canceled:true ────────────────────────────────
def test_translate_cancel(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake(wait_cancel=True))
    jid = _done_job(client, sample_pdf)

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_until_status(client, jid, "running")

    c = client.post(f"/api/jobs/{jid}/translate/cancel?lang=ko")
    assert c.status_code == 202
    assert c.json()["status"] == "canceling"

    assert _wait_until_status(client, jid, "canceled")["status"] == "canceled"
    _wait_no_task(client, jid)

    evs = _collect_sse(client, f"/api/jobs/{jid}/translate/events?lang=ko")
    canceled_errors = [d for e, d in evs if e == "error" and d.get("canceled") is True]
    assert canceled_errors, evs


# ── 6. stale 조정: running인데 태스크 없음 → error 재기록 ──────────────────
def test_translate_state_stale_adjusted(client, sample_pdf, settings):
    jid = _done_job(client, sample_pdf)
    tdir = settings.jobs_dir / jid / "translations" / "ko"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "state.json").write_text(json.dumps({
        "lang": "ko", "status": "running", "current": 1, "total": 3, "error": None,
    }), encoding="utf-8")

    body = client.get(f"/api/jobs/{jid}/translate/state?lang=ko").json()
    assert body["status"] == "error"
    assert "서버가 재시작" in body["error"]
    # 원자적으로 재기록되어 재조회해도 error 유지
    assert _tstate(client, jid)["status"] == "error"


# ── 7. /archive에 result.ko.md 포함 (캐시 삭제 후 재생성) ──────────────────
def test_archive_includes_translation(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)

    ar0 = client.get(f"/api/jobs/{jid}/archive")
    assert ar0.status_code == 200
    names0 = set(zipfile.ZipFile(io.BytesIO(ar0.content)).namelist())
    assert "result.md" in names0
    assert "result.ko.md" not in names0

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)  # 스레드가 archive.zip 캐시를 삭제할 때까지
    assert _tstate(client, jid)["status"] == "done"

    ar1 = client.get(f"/api/jobs/{jid}/archive")
    assert ar1.status_code == 200
    names1 = set(zipfile.ZipFile(io.BytesIO(ar1.content)).namelist())
    assert "result.md" in names1
    assert "result.ko.md" in names1
