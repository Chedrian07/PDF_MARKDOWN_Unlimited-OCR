import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as _config  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

# TestClient는 Host: testserver로 요청한다 — 프로덕션 기본값(ALLOWED_HOSTS)은 그대로 두고
# 테스트 프로세스의 기본 화이트리스트에만 추가한다 (Settings를 직접 생성하는 테스트 포함).
_config._DEFAULT_ALLOWED_HOSTS += ",testserver"


def make_pdf_bytes(pages: int = 3, with_image: bool = True) -> bytes:
    import fitz
    from PIL import Image, ImageDraw

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 80), f"Sample page {i + 1}", fontsize=24)
        page.insert_text((72, 120), "Unlimited-OCR pipeline test document.", fontsize=12)
        if with_image and i == 0:
            img = Image.new("RGB", (240, 140), (245, 245, 245))
            d = ImageDraw.Draw(img)
            for bi, h in enumerate((40, 90, 60, 110)):
                x = 20 + bi * 55
                d.rectangle((x, 130 - h, x + 40, 130), fill=(70, 90, 200))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            page.insert_image(fitz.Rect(72, 200, 372, 375), stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf_bytes()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        engine="fake",
        device="cpu",
        data_dir=tmp_path / "data",
        preload_model=False,
        fake_delay=0.0,
        frontend_dir=tmp_path / "no-frontend",  # 정적 마운트 비활성화
    )


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def wait_done(client, job_id: str, timeout: float = 15.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("done", "error", "canceled"):
            return body
        time.sleep(0.03)
    raise AssertionError(f"잡이 제한시간 내에 끝나지 않음: {client.get(f'/api/jobs/{job_id}').json()}")
