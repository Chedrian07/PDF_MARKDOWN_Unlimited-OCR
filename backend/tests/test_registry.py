"""디바이스/엔진 registry 테스트 — Metal(MPS) 포함. 계약: docs/ARCHITECTURE.md §6"""

import pytest

from app.config import Settings
from app.engine.fake import FakeEngine
from app.engine.registry import build_engine


def _fake_settings(**kw) -> Settings:
    return Settings(engine="fake", preload_model=False, fake_delay=0.0, **kw)


def test_metal_device_builds_fake_engine():
    eng = build_engine(_fake_settings(device="metal"))
    assert isinstance(eng, FakeEngine)
    assert eng.device == "metal"


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="OCR_DEVICE"):
        build_engine(_fake_settings(device="tpu"))


def test_metal_unlimited_engine_constructs_without_torch():
    # 생성 시점엔 torch가 필요 없다 — MPS 가용성 검증은 load()에서 수행
    eng = build_engine(Settings(engine="unlimited", device="metal"))
    assert eng.device == "metal"
    assert eng.torch_device == "mps"
    assert not eng.loaded


def test_torch_device_mapping():
    from app.engine.unlimited import torch_device_name

    assert torch_device_name("cpu") == "cpu"
    assert torch_device_name("cuda") == "cuda"
    assert torch_device_name("metal") == "mps"


def test_env_mps_alias(monkeypatch):
    monkeypatch.setenv("OCR_DEVICE", "mps")
    assert Settings.from_env().device == "metal"


def test_page_output_limits_from_env(monkeypatch):
    monkeypatch.setenv("MAX_PAGE_OUTPUT_CHARS", "12345")
    monkeypatch.setenv("MAX_PAGE_OUTPUT_TOKENS", "4321")

    settings = Settings.from_env()

    assert settings.max_page_output_chars == 12_345
    assert settings.max_page_output_tokens == 4_321


def test_resolve_dtype():
    torch = pytest.importorskip("torch")
    from app.engine.unlimited import _resolve_dtype

    assert _resolve_dtype("metal", "auto") in (torch.bfloat16, torch.float32)
    assert _resolve_dtype("metal", "float16") is torch.float16
    assert _resolve_dtype("cpu", "auto") is torch.float32
    assert _resolve_dtype("cuda", "auto") is torch.bfloat16
    with pytest.raises(ValueError, match="OCR_DTYPE"):
        _resolve_dtype("metal", "int8")


def test_load_concurrent_calls_load_once(monkeypatch):
    # 프리로드 스레드와 워커 스레드가 load()에 동시 진입해도 from_pretrained는
    # 정확히 1회만 실행돼야 한다 (동시 로드 → meta 텐서 잔류 → .to() 실패 회귀 방지)
    pytest.importorskip("torch")
    import threading
    import time

    import transformers

    from app.engine.unlimited import UnlimitedEngine
    from app.vendor.unlimited_ocr import UnlimitedOCRForCausalLM

    calls: list[int] = []

    class _FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    def _fake_model_fp(*args, **kwargs):
        calls.append(1)
        time.sleep(0.2)  # 두 번째 스레드가 로딩 도중에 진입하도록 지연
        return _FakeModel()

    monkeypatch.setattr(UnlimitedOCRForCausalLM, "from_pretrained", _fake_model_fp)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: object())

    eng = UnlimitedEngine(Settings(engine="unlimited", device="cpu", preload_model=False))
    errors: list[Exception] = []

    def _load():
        try:
            eng.load()
        except Exception as e:  # noqa: BLE001 — 스레드 안 예외를 본문으로 전달
            errors.append(e)

    threads = [threading.Thread(target=_load) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert len(calls) == 1
    assert eng.loaded


def test_health_reports_metal(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = _fake_settings(
        device="metal",
        data_dir=tmp_path / "data",
        frontend_dir=tmp_path / "no-frontend",
    )
    with TestClient(create_app(settings)) as c:
        body = c.get("/api/health").json()
    assert body["device"] == "metal"
