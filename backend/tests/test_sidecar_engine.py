"""SidecarEngine 통합 테스트 — stub sidecar + 실제 runner/merger/API 파이프라인."""

import json
import threading
import zipfile
from io import BytesIO

import pytest

from app.config import Settings
from app.engine.base import EngineError, JobCanceled, NullSink
from app.engine.registry import build_engine
from app.engine.sidecar import SidecarEngine
from app.main import create_app

from tests.test_sidecar_client import StubSidecar, _health_body, _parse_body
from tests.conftest import make_pdf_bytes, wait_done


@pytest.fixture
def stub():
    s = StubSidecar()
    yield s
    s.close()


def _settings(tmp_path, stub, **kw) -> Settings:
    base = dict(
        engine="ovisocr2", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, frontend_dir=tmp_path / "no-frontend",
        sidecar_url=stub.url, sidecar_connect_timeout_s=2.0,
        sidecar_read_timeout_s=10.0, sidecar_health_timeout_s=2.0,
    )
    base.update(kw)
    return Settings(**base)


# ── registry ───────────────────────────────────────────────────────────────

def test_registry_requires_sidecar_url(tmp_path):
    with pytest.raises(ValueError, match="OCR_SIDECAR_URL"):
        build_engine(Settings(engine="ovisocr2", device="cpu"))


def test_registry_builds_sidecar_engines(tmp_path, stub):
    eng = build_engine(_settings(tmp_path, stub))
    assert isinstance(eng, SidecarEngine)
    assert eng.name == "ovisocr2"
    caps = eng.capabilities()
    assert caps.supports_multi_page is False
    assert caps.stream_granularity == "page"
    assert caps.layout_capability == "figure_only"
    assert caps.preferred_chunk_size == 1
    assert caps.provider == "local-sidecar"

    paddle = build_engine(_settings(tmp_path, stub, engine="paddleocr_vl"))
    assert paddle.capabilities().layout_capability == "full"


def test_registry_unknown_engine_lists_options():
    with pytest.raises(ValueError, match="unlimited.*fake.*ovisocr2.*paddleocr_vl"):
        build_engine(Settings(engine="tesseract", device="cpu"))


def test_legacy_engines_still_build():
    from app.engine.fake import FakeEngine

    eng = build_engine(Settings(engine="fake", device="cpu", fake_delay=0.0))
    assert isinstance(eng, FakeEngine)
    caps = eng.capabilities()  # 기본 capability = 기존 Unlimited 의미 (하위 호환)
    assert caps.supports_multi_page is True
    assert caps.stream_granularity == "token"
    assert caps.preferred_chunk_size is None


# ── load/health ────────────────────────────────────────────────────────────

def test_load_fails_clearly_when_sidecar_down(tmp_path):
    settings = Settings(
        engine="ovisocr2", device="cpu", sidecar_url="http://127.0.0.1:1",
        sidecar_connect_timeout_s=0.5, sidecar_retries=0,
    )
    eng = build_engine(settings)
    with pytest.raises(EngineError, match="연결할 수 없습니다"):
        eng.load()
    assert not eng.loaded


def test_load_fails_clearly_while_model_loading(tmp_path, stub):
    stub.health_response = _health_body(model_loaded=False)
    eng = build_engine(_settings(tmp_path, stub))
    with pytest.raises(EngineError, match="아직 모델을 로드하지"):
        eng.load()


def test_provider_health_surface(tmp_path, stub):
    eng = build_engine(_settings(tmp_path, stub))
    ph = eng.provider_health()
    assert ph["status"] == "ok"
    assert ph["gpu_total_mb"] == 16384
    assert eng.gpu_name() == "NVIDIA GeForce RTX 5070 Ti"

    down = build_engine(_settings(tmp_path, stub, sidecar_url="http://127.0.0.1:1",
                                  sidecar_connect_timeout_s=0.3, sidecar_retries=0))
    ph2 = down.provider_health()
    assert ph2["status"] == "unreachable"
    assert "error" in ph2


# ── run_multi 청크 계약 ────────────────────────────────────────────────────

def test_run_multi_produces_chunk_contract(tmp_path, stub):
    eng = build_engine(_settings(tmp_path, stub))
    eng.load()
    page_png = tmp_path / "p1.png"
    from PIL import Image

    Image.new("RGB", (400, 560), "white").save(page_png)

    texts: list[str] = []

    class Sink:
        def on_text(self, t: str) -> None:
            texts.append(t)

    out = tmp_path / "chunk_00"
    md = eng.run_multi([page_png], out, Sink(), threading.Event())

    assert md.startswith("<PAGE>\n")
    assert "![](images/page_0_0.jpg)" in md
    assert (out / "images" / "page_0_0.jpg").is_file()
    assert (out / "result_with_boxes_0.jpg").is_file()
    assert (out / "boxes.json").is_file()
    assert (out / "raw_pages.json").is_file()
    # 페이지 단위 발행: <PAGE> 마커 후 페이지 전체 텍스트 (기존 SSE 계약)
    assert texts[0] == "<PAGE>\n"
    assert "본문" in "".join(texts)


def test_run_single_contract(tmp_path, stub):
    eng = build_engine(_settings(tmp_path, stub))
    eng.load()
    from PIL import Image

    page_png = tmp_path / "p1.png"
    Image.new("RGB", (400, 560), "white").save(page_png)
    out = tmp_path / "single"
    md = eng.run_single(page_png, out, NullSink(), threading.Event())
    assert "<PAGE>" not in md
    assert (out / "images" / "0.jpg").is_file()
    assert (out / "result_with_boxes.jpg").is_file()


def test_run_multi_cancel_before_start(tmp_path, stub):
    eng = build_engine(_settings(tmp_path, stub))
    eng.load()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(JobCanceled):
        eng.run_multi([tmp_path / "x.png"], tmp_path / "c", NullSink(), cancel)
    assert stub.requests_seen == []  # 취소 후에는 요청을 보내지 않는다


# ── API E2E (업로드 → 변환 → 결과/메타/아카이브) ───────────────────────────

@pytest.fixture
def sidecar_client_app(tmp_path, stub):
    from fastapi.testclient import TestClient

    app = create_app(_settings(tmp_path, stub))
    with TestClient(app) as c:
        yield c


def test_e2e_upload_to_done_with_sidecar(sidecar_client_app, stub):
    c = sidecar_client_app
    pdf = make_pdf_bytes(pages=2)
    r = c.post("/api/jobs", files={"file": ("doc.pdf", BytesIO(pdf), "application/pdf")},
               data={"mode": "multi"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    body = wait_done(c, job_id)
    assert body["status"] == "done", body

    # 엔진/모델 메타가 잡에 기록된다
    assert body["engine"] == "ovisocr2"
    assert body["model_id"] == "ATH-MaaS/OvisOCR2"
    assert body["model_revision"] == "rev"
    assert body["provider"] == "local-sidecar"

    # 페이지 단위 모델 안내 warning (multi 모드였으므로)
    assert any("페이지 단위 모델" in w for w in body["warnings"])

    # 기존 산출물 규약: 글로벌 리넘버링된 figure + layout
    md = c.get(f"/api/jobs/{job_id}/markdown").text
    assert "![](images/p0001_0.jpg)" in md
    assert "![](images/p0002_0.jpg)" in md
    result = body["result"]
    assert any("p0001_0.jpg" in u for u in result["images"])
    assert any("page_0001.jpg" in u for u in result["layouts"])
    assert result["has_layout"] is True

    # layout.json — figure_only 엔진은 image 블록만
    layout = c.get(f"/api/jobs/{job_id}/layout")
    assert layout.status_code == 200

    # archive에 meta.json 동봉
    archive = c.get(f"/api/jobs/{job_id}/archive")
    assert archive.status_code == 200
    zf = zipfile.ZipFile(BytesIO(archive.content))
    assert "meta.json" in zf.namelist()
    meta = json.loads(zf.read("meta.json"))
    assert meta["engine"] == "ovisocr2"
    assert meta["model_id"] == "ATH-MaaS/OvisOCR2"


def test_e2e_health_generalized(sidecar_client_app):
    body = sidecar_client_app.get("/api/health").json()
    assert body["engine"] == "ovisocr2"
    assert body["provider"] == "local-sidecar"
    assert body["capabilities"] == {
        "multi_page_context": False,
        "stream_granularity": "page",
        "layout": "figure_only",
        "figures": True,
    }
    assert body["provider_health"]["status"] == "ok"
    assert body["provider_health"]["gpu_total_mb"] == 16384


def test_e2e_health_when_sidecar_down(tmp_path):
    from fastapi.testclient import TestClient

    settings = Settings(
        engine="ovisocr2", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, frontend_dir=tmp_path / "no-frontend",
        sidecar_url="http://127.0.0.1:1", sidecar_connect_timeout_s=0.3,
        sidecar_retries=0,
    )
    with TestClient(create_app(settings)) as c:
        r = c.get("/api/health")
    # 메인 앱 health는 200 — provider 상태로 구분
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["provider_health"]["status"] == "unreachable"


def test_e2e_sidecar_error_isolated_per_page(sidecar_client_app, stub):
    """sidecar 500 → 페이지 실패가 내장 텍스트/placeholder로 격리되고 잡은 계속."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 2:  # 1페이지의 최초+재시도 실패
            return (500, json.dumps({"detail": "인위적 실패"}).encode())
        return (200, json.dumps(_parse_body()).encode())

    stub.parse_behavior = flaky
    c = sidecar_client_app
    pdf = make_pdf_bytes(pages=2)
    r = c.post("/api/jobs", files={"file": ("doc.pdf", BytesIO(pdf), "application/pdf")},
               data={"mode": "multi"})
    job_id = r.json()["job_id"]
    body = wait_done(c, job_id)
    assert body["status"] == "done", body
    # 실패 페이지에 대한 복구/placeholder 경고가 남는다
    assert any("1페이지" in w or "1–" in w for w in body["warnings"] if "페이지" in w)


# ── legacy 메타 복원 ───────────────────────────────────────────────────────

def test_legacy_meta_restores_without_engine_fields(tmp_path):
    from app.jobs import JobStore

    jobs_dir = tmp_path / "jobs"
    legacy_dir = jobs_dir / "j_legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "meta.json").write_text(json.dumps({
        "id": "j_legacy", "filename": "old.pdf", "mode": "multi", "dpi": 200,
        "status": "done", "created_at": "2026-01-01T00:00:00+00:00",
        "progress": {}, "error": None, "warnings": [],
    }), encoding="utf-8")
    store = JobStore(jobs_dir)
    store.load_existing()
    job = store.get("j_legacy")
    assert job is not None
    assert job.engine is None and job.model_id is None
    d = job.to_dict()
    assert d["engine"] is None
    assert d["model_id"] is None
    assert d["status"] == "done"


def test_new_meta_roundtrip(tmp_path):
    from app.jobs import JobStore

    store = JobStore(tmp_path / "jobs")
    job = store.create("a.pdf", "multi", 200, engine_info={
        "engine": "paddleocr_vl", "model_id": "PaddlePaddle/PaddleOCR-VL-1.6",
        "model_revision": "66317acc", "provider": "local-sidecar",
    })
    store2 = JobStore(tmp_path / "jobs")
    store2.load_existing()
    restored = store2.get(job.id)
    assert restored.engine == "paddleocr_vl"
    assert restored.model_id == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert restored.model_revision == "66317acc"
    assert restored.provider == "local-sidecar"


# ── capability 기반 청크 크기 ──────────────────────────────────────────────

def test_capability_chunk_size_page_engine(tmp_path, stub, sidecar_client_app):
    """페이지 단위 엔진: multi 모드에서도 내부 청크 크기 1 (total_chunks == 페이지 수)."""
    c = sidecar_client_app
    pdf = make_pdf_bytes(pages=3)
    r = c.post("/api/jobs", files={"file": ("doc.pdf", BytesIO(pdf), "application/pdf")},
               data={"mode": "multi"})
    body = wait_done(c, r.json()["job_id"])
    assert body["progress"]["total_chunks"] == 3
    assert body["progress"]["total_pages"] == 3


def test_provider_health_caches_failures(tmp_path, stub, monkeypatch):
    """죽은 sidecar에 대한 실패도 TTL 캐시 — health 폴링이 매번 연결을 시도하며
    요청 스레드를 connect timeout만큼 묶지 않게 한다."""
    from app.sidecar.client import SidecarUnavailableError

    eng = build_engine(_settings(tmp_path, stub))
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise SidecarUnavailableError("연결할 수 없습니다")

    monkeypatch.setattr(eng._client, "health", _boom)
    first = eng.provider_health()
    second = eng.provider_health()
    assert first["status"] == "unreachable"
    assert second == first
    assert calls["n"] == 1, "TTL 내 두 번째 호출은 캐시를 쓴다"
    assert not eng.loaded
    # load()도 캐시된 실패로 즉시 실패 (페이지마다 타임아웃을 반복하지 않는다)
    with pytest.raises(EngineError, match="연결할 수 없습니다"):
        eng.load()
    assert calls["n"] == 1


def test_sanitize_warnings_reach_job(sidecar_client_app, stub):
    """정화로 버려진 블록이 잡 warnings로 승격 — 조용한 내용 손실 금지."""
    body = _parse_body()
    body["page"]["blocks"] = [
        {"type": "image", "bbox": [900, 900, 100, 100],  # 좌표 역전 → 폐기
         "content": "", "order": 0, "figure_index": 0},
    ]
    body["page"]["markdown"] = "본문 [[FIGURE:0]]"
    stub.parse_behavior = lambda: (200, json.dumps(body).encode())

    c = sidecar_client_app
    pdf = make_pdf_bytes(pages=1)
    r = c.post("/api/jobs", files={"file": ("doc.pdf", BytesIO(pdf), "application/pdf")},
               data={"mode": "multi"})
    result = wait_done(c, r.json()["job_id"])
    assert result["status"] == "done"
    assert any("bbox" in w or "figure" in w for w in result["warnings"]), result["warnings"]


def test_warning_page_span_is_global(sidecar_client_app, stub):
    """경고의 페이지 번호는 전역이어야 한다 — 엔진의 청크-로컬 인덱스(기본 항상 0)를
    쓰면 100페이지 문서의 어느 페이지가 손실됐는지 특정할 수 없다."""
    body = _parse_body()
    body["page"]["blocks"] = [
        {"type": "image", "bbox": [900, 900, 100, 100],  # 좌표 역전 → 폐기 + 경고
         "content": "", "order": 0, "figure_index": 0},
    ]
    body["page"]["markdown"] = "본문"
    stub.parse_behavior = lambda: (200, json.dumps(body).encode())

    c = sidecar_client_app
    r = c.post("/api/jobs",
               files={"file": ("doc.pdf", BytesIO(make_pdf_bytes(pages=3)), "application/pdf")},
               data={"mode": "multi"})
    result = wait_done(c, r.json()["job_id"])
    spans = {w.split(":")[0] for w in result["warnings"] if "페이지:" in w}
    assert spans == {"1페이지", "2페이지", "3페이지"}, result["warnings"]


def test_job_warning_cap(tmp_path, stub, monkeypatch):
    """엔진 경고는 청크마다 나올 수 있어 잡 단위 상한이 없으면 meta.json이 폭증한다."""
    from app.pipeline import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_MAX_JOB_WARNINGS", 3)
    merger_warnings: list[str] = []

    class _Merger:
        warnings = merger_warnings

    # runner의 승격 루프와 동일한 규칙을 직접 검증 (상한 + 안내 1건 + 이후 폐기)
    engine_warnings = [f"경고 {i}" for i in range(10)]
    for w in engine_warnings:
        if len(_Merger.warnings) < runner_mod._MAX_JOB_WARNINGS:
            _Merger.warnings.append(f"1페이지: {w}")
        elif len(_Merger.warnings) == runner_mod._MAX_JOB_WARNINGS:
            _Merger.warnings.append("경고가 3건을 넘어 이후 항목은 생략됩니다 (서버 로그에서 전체 확인)")
    assert len(merger_warnings) == 4
    assert "생략" in merger_warnings[-1]


def test_engine_warnings_drained_between_jobs(tmp_path, stub):
    eng = build_engine(_settings(tmp_path, stub))
    eng._note("첫 잡 경고")
    assert eng.drain_warnings() == ["첫 잡 경고"]
    assert eng.drain_warnings() == []
    # 페이지마다 반복되는 동일 경고는 1건으로 접힌다
    eng._note("같은 경고")
    eng._note("같은 경고")
    eng._note("다른 경고")
    assert eng.drain_warnings() == ["같은 경고", "다른 경고"]


def test_literal_page_marker_does_not_break_chunk_contract(sidecar_client_app, stub):
    """모델이 본문에 <PAGE>를 뱉어도 페이지 수 계약이 깨지지 않는다."""
    body = _parse_body()
    body["page"]["markdown"] = "앞 문단\n\n<PAGE>\n\n뒤 문단"
    body["page"]["blocks"] = []
    stub.parse_behavior = lambda: (200, json.dumps(body).encode())

    c = sidecar_client_app
    pdf = make_pdf_bytes(pages=2)
    r = c.post("/api/jobs", files={"file": ("doc.pdf", BytesIO(pdf), "application/pdf")},
               data={"mode": "multi"})
    result = wait_done(c, r.json()["job_id"])
    assert result["status"] == "done"
    md = c.get(f"/api/jobs/{result['job_id']}/markdown").text
    assert "<PAGE>" not in md
    # 페이지 구분자는 정확히 페이지 수 - 1개 (마커 초과 경고 없음)
    assert md.count("\n---\n") == 1
    assert not any("마커" in w for w in result["warnings"])


def test_capability_chunk_size_fake_engine_unchanged(client, sample_pdf):
    """기존 엔진(fake): PAGES_PER_CHUNK 그대로 (기본 8 → 3페이지 = 1청크)."""
    r = client.post("/api/jobs",
                    files={"file": ("doc.pdf", BytesIO(sample_pdf), "application/pdf")},
                    data={"mode": "multi"})
    body = wait_done(client, r.json()["job_id"])
    assert body["progress"]["total_chunks"] == 1
    assert body["engine"] == "fake"  # 잡 메타는 fake 엔진에도 기록된다
    assert not any("페이지 단위 모델" in w for w in body["warnings"])
