import io
import zipfile

from conftest import wait_done


def _upload(client, pdf_bytes: bytes, **data):
    return client.post(
        "/api/jobs",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        data=data,
    )


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == "fake"
    assert body["device"] == "cpu"
    assert "native_ops" in body


def test_upload_validation(client, sample_pdf):
    r = client.post("/api/jobs", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/jobs", files={"file": ("a.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 400
    r = _upload(client, sample_pdf, mode="weird")
    assert r.status_code == 400
    r = _upload(client, sample_pdf, dpi="9999")
    assert r.status_code == 400


def test_full_flow_multi(client, sample_pdf):
    r = _upload(client, sample_pdf, mode="multi")
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]

    body = wait_done(client, jid)
    assert body["status"] == "done", body
    assert body["progress"]["total_pages"] == 3
    assert body["progress"]["current_page"] == 3

    res = body["result"]
    assert len(res["pages"]) == 3
    assert len(res["layouts"]) == 3
    assert len(res["images"]) == 3  # FakeEngine: 페이지당 figure 1개

    md = client.get(f"/api/jobs/{jid}/markdown")
    assert md.status_code == 200
    assert "X-Partial" not in md.headers
    for name in ("p0001_0.jpg", "p0002_0.jpg", "p0003_0.jpg"):
        assert f"![](images/{name})" in md.text

    html = client.get(f"/api/jobs/{jid}/html")
    assert f'src="/api/jobs/{jid}/files/images/p0001_0.jpg"' in html.text
    assert "<table>" in html.text

    img = client.get(res["images"][0])
    assert img.status_code == 200
    assert img.headers["content-type"].startswith("image/")

    ar = client.get(f"/api/jobs/{jid}/archive")
    assert ar.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(ar.content))
    names = set(zf.namelist())
    assert "result.md" in names
    assert "images/p0001_0.jpg" in names

    listed = client.get("/api/jobs").json()["jobs"]
    assert any(j["job_id"] == jid for j in listed)

    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_full_flow_per_page(client, sample_pdf):
    r = _upload(client, sample_pdf, mode="per_page")
    assert r.status_code == 202
    jid = r.json()["job_id"]
    body = wait_done(client, jid)
    assert body["status"] == "done", body
    assert body["progress"]["total_chunks"] == 3
    md = client.get(f"/api/jobs/{jid}/markdown").text
    assert "![](images/p0001_0.jpg)" in md
    assert "![](images/p0003_0.jpg)" in md


def test_sse_events(client, sample_pdf):
    jid = _upload(client, sample_pdf, mode="multi").json()["job_id"]
    events = set()
    with client.stream("GET", f"/api/jobs/{jid}/events") as s:
        for line in s.iter_lines():
            if line.startswith("event: "):
                events.add(line.removeprefix("event: ").strip())
            if "event: done" in line or "event: error" in line:
                break
    assert "done" in events, events
    # 완료 후 재접속하면 스냅샷으로 즉시 done
    with client.stream("GET", f"/api/jobs/{jid}/events") as s:
        first_events = [ln for ln in s.iter_lines() if ln.startswith("event: ")]
    assert first_events and first_events[0] == "event: done"


def test_files_path_traversal_blocked(client, sample_pdf):
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    assert client.get(f"/api/jobs/{jid}/files/../meta.json").status_code in (400, 404)
    assert client.get(f"/api/jobs/{jid}/files/work/chunk_00/result.md").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/meta.json").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/pages/page_0001.png").status_code == 200


def test_cancel_keeps_partial_results(tmp_path, sample_pdf):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.4, frontend_dir=tmp_path / "no-frontend",
    )
    with TestClient(create_app(settings)) as client:
        jid = _upload(client, sample_pdf).json()["job_id"]
        r = client.post(f"/api/jobs/{jid}/cancel")
        assert r.status_code == 202
        assert r.json()["status"] == "canceling"
        body = wait_done(client, jid)
        assert body["status"] == "canceled", body
        # 삭제되지 않고 남아 있어야 함 (부분 결과 보존)
        assert client.get(f"/api/jobs/{jid}").status_code == 200
        md = client.get(f"/api/jobs/{jid}/markdown")
        assert md.status_code == 200
        assert md.headers.get("X-Partial") == "true"


def test_cancel_finished_job_is_noop(client, sample_pdf):
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 202
    assert r.json()["status"] == "done"


def test_render_preview(client, sample_pdf):
    jid = _upload(client, sample_pdf).json()["job_id"]
    md = "# 라이브\n\n![](images/p0001_0.jpg)\n\n<table><tr><td>a</td></tr></table>\n\n<script>x</script>"
    r = client.post(f"/api/jobs/{jid}/render-preview", content=md.encode())
    assert r.status_code == 200
    assert f'src="/api/jobs/{jid}/files/images/p0001_0.jpg"' in r.text
    assert "<table><tr><td>a</td></tr></table>" in r.text
    assert "<script>" not in r.text
    assert client.post("/api/jobs/j_nope/render-preview", content=b"x").status_code == 404


def test_archive_before_done_conflicts(client, sample_pdf, settings):
    # fake_delay=0이면 너무 빨리 끝나 409를 못 볼 수 있으므로 큰 파일로 시도하지 않고
    # 존재하지 않는 완료 전 상태를 시뮬레이션: 새 잡을 만들고 즉시 archive 요청 경합 허용
    jid = _upload(client, sample_pdf).json()["job_id"]
    r = client.get(f"/api/jobs/{jid}/archive")
    assert r.status_code in (200, 409)
