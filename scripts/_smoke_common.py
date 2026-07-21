"""smoke/benchmark 스크립트 공용 헬퍼 — 표준 라이브러리만 사용.

backend REST API(업로드/폴링/결과)와 nvidia-smi VRAM 샘플러를 감싼다.
"""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# 전송 계층에서 올 수 있는 예외 묶음. URLError·TimeoutError는 OSError의 서브클래스라
# OSError 하나로 덮이지만, 서버가 응답 도중 끊는 경우(RemoteDisconnected·
# IncompleteRead)는 http.client.HTTPException이라 별도로 잡아야 한다 —
# 기동 직후 backend에 붙으면 ConnectionResetError로 스크립트가 죽었다(실측).
_TRANSPORT_ERRORS = (OSError, http.client.HTTPException)


class SmokeError(RuntimeError):
    pass


def http_json(url: str, timeout: float = 10.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8")
    except _TRANSPORT_ERRORS as e:
        raise SmokeError(f"요청 실패 {url}: {e.__class__.__name__}: {e}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise SmokeError(f"JSON 응답이 아닙니다 {url}: {body[:120]}") from e


def http_text(url: str, timeout: float = 30.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8")
    except _TRANSPORT_ERRORS as e:
        raise SmokeError(f"요청 실패 {url}: {e.__class__.__name__}: {e}") from e


def upload_pdf(base_url: str, pdf_path: Path, mode: str = "multi", timeout: float = 120.0) -> str:
    """multipart/form-data 업로드 (stdlib) → job_id."""
    boundary = f"----smoke{uuid.uuid4().hex}"
    body = bytearray()

    def field(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{pdf_path.name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n".encode()
    )
    body.extend(pdf_path.read_bytes())
    body.extend(b"\r\n")
    field("mode", mode)
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{base_url}/api/jobs",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SmokeError(f"업로드 실패 HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    except _TRANSPORT_ERRORS as e:
        raise SmokeError(f"업로드 실패: {e.__class__.__name__}: {e}") from e
    return resp["job_id"]


def wait_job(base_url: str, job_id: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = http_json(f"{base_url}/api/jobs/{job_id}")
        if body["status"] in ("done", "error", "canceled"):
            return body
        time.sleep(2.0)
    raise SmokeError(f"잡이 {timeout_s:.0f}s 안에 끝나지 않았습니다 (job={job_id})")


def wait_model_loaded(base_url: str, timeout_s: float) -> dict:
    """health의 model_loaded가 true가 될 때까지 대기 (최초 모델 다운로드 감안)."""
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            last = http_json(f"{base_url}/api/health")
        except SmokeError:
            time.sleep(3.0)
            continue
        if last.get("model_loaded"):
            return last
        ph = last.get("provider_health") or {}
        print(f"  … 모델 로딩 대기 (provider={ph.get('status', 'n/a')}, "
              f"load_error={last.get('model_load_error')})")
        time.sleep(5.0)
    raise SmokeError(f"모델이 {timeout_s:.0f}s 안에 로드되지 않았습니다: {last}")


class VramSampler:
    """nvidia-smi 폴링으로 peak VRAM(MB)을 기록한다. GPU가 없으면 no-op."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5.0,
            )
            if out.returncode == 0 and out.stdout.strip():
                return int(float(out.stdout.strip().splitlines()[0]))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        return 0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.peak_mb = max(self.peak_mb, self._sample())

    def __enter__(self) -> "VramSampler":
        if self.available:
            self.peak_mb = self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


def ensure_sample_pdf(path: Path) -> Path:
    """샘플 PDF 확보 — 없으면 scripts/make_sample_pdf.py로 생성 (pymupdf 필요)."""
    if path.is_file():
        return path
    import sys

    script = Path(__file__).resolve().parent / "make_sample_pdf.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, str(script), str(path)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not path.is_file():
        raise SmokeError(
            f"샘플 PDF 생성 실패 — pymupdf가 있는 환경(backend venv)에서 실행하거나 "
            f"--pdf로 기존 PDF를 지정하세요. stderr: {proc.stderr.strip()[:300]}"
        )
    return path


def count_markers(markdown: str) -> dict:
    """결과 markdown의 구조 요소 집계 (정확도 점수가 아니라 존재 확인용)."""
    import re

    return {
        "chars": len(markdown),
        "figures": len(re.findall(r"!\[\]\(images/", markdown)),
        "tables": markdown.count("<table") + len(re.findall(r"^\|.+\|$", markdown, re.M)),
        "formulas": len(re.findall(r"\\\(|\\\[|\$\$", markdown)),
        "failed_pages": markdown.count("이 페이지는 변환에 실패했습니다"),
    }
