#!/usr/bin/env python3
"""OvisOCR2 스택 실 GPU smoke test (RTX 5070 Ti 검증 절차의 1단계).

전제: docker compose --profile ovis up -d --build ovisocr2 ocr-ovis  (backend: :8002)

    cd backend && uv run python ../scripts/smoke_ovisocr2_5070ti.py
    # 또는: python scripts/smoke_ovisocr2_5070ti.py --url http://127.0.0.1:8002 --pdf my.pdf

확인 항목: health(엔진/provider/GPU) → 모델 로드 대기 → 샘플 PDF 변환 →
Markdown/figure/table/formula/layout 산출물 → peak VRAM/처리 시간/OOM 경고.
종료 코드 0 = 전 항목 통과.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import (  # noqa: E402
    SmokeError, VramSampler, count_markers, ensure_sample_pdf,
    http_json, http_text, upload_pdf, wait_job, wait_model_loaded,
)

ENGINE = "ovisocr2"


def run(url: str, pdf: Path, model_wait_s: float, job_timeout_s: float) -> int:
    failures: list[str] = []

    print(f"[1/5] health 확인: {url}")
    health = http_json(f"{url}/api/health")
    if health.get("engine") != ENGINE:
        raise SmokeError(f"engine={health.get('engine')} — {ENGINE} 스택(:8002)이 아닙니다")
    print(f"  engine={health['engine']} model={health.get('model_id')} "
          f"provider={health.get('provider')}")

    print(f"[2/5] 모델 로드 대기 (최대 {model_wait_s:.0f}s — 최초 실행은 다운로드 포함)")
    health = wait_model_loaded(url, model_wait_s)
    ph = health.get("provider_health") or {}
    print(f"  로드 완료: gpu={health.get('gpu_name')} "
          f"free={ph.get('gpu_free_mb')}MB/{ph.get('gpu_total_mb')}MB "
          f"runtime={ph.get('runtime')} {ph.get('version')}")
    caps = health.get("capabilities") or {}
    if caps.get("stream_granularity") != "page":
        failures.append(f"capabilities.stream_granularity={caps.get('stream_granularity')} (기대: page)")

    print(f"[3/5] 샘플 PDF 변환: {pdf}")
    t0 = time.monotonic()
    with VramSampler() as vram:
        job_id = upload_pdf(url, pdf)
        body = wait_job(url, job_id, job_timeout_s)
    elapsed = time.monotonic() - t0
    if body["status"] != "done":
        raise SmokeError(f"잡 실패: status={body['status']} error={body.get('error')}")
    pages = body["progress"]["total_pages"]
    print(f"  완료: {pages}페이지 · {elapsed:.1f}s ({elapsed / max(pages, 1):.1f}s/페이지)"
          + (f" · peak VRAM {vram.peak_mb}MB" if vram.available else " · (nvidia-smi 없음)"))
    if body.get("engine") != ENGINE:
        failures.append(f"잡 메타 engine={body.get('engine')}")
    for w in body.get("warnings", []):
        print(f"  ⚠ warning: {w}")
    oom = [w for w in body.get("warnings", []) if "메모리" in w or "OOM" in w or "강등" in w]
    if oom:
        failures.append(f"OOM/해상도 강등 발생: {oom[0]}")

    print("[4/5] 산출물 검증")
    md = http_text(f"{url}/api/jobs/{job_id}/markdown")
    stats = count_markers(md)
    print(f"  markdown {stats['chars']}자 · figure {stats['figures']} · "
          f"table {stats['tables']} · formula {stats['formulas']}")
    if stats["chars"] < 100:
        failures.append("markdown이 비정상적으로 짧음")
    if stats["figures"] < 1:
        failures.append("figure crop 없음 (샘플에는 이미지 2개가 있음)")
    if stats["tables"] < 1:
        failures.append("표 없음 (샘플에는 표 1개가 있음)")
    if stats["failed_pages"]:
        failures.append(f"실패 페이지 {stats['failed_pages']}개")
    result = body.get("result") or {}
    if not result.get("images"):
        failures.append("result.images 비어 있음")
    if not result.get("has_layout"):
        failures.append("layout.json 없음 (figure_only layout이라도 있어야 함)")

    print("[5/5] 판정")
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("  ✓ OvisOCR2 스택 smoke test 통과 — 이 결과로 문서의 '실제 검증' 항목을 갱신하세요")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--pdf", type=Path, default=Path("sample/sample.pdf"))
    ap.add_argument("--model-wait", type=float, default=1800.0,
                    help="모델 로드 대기 상한(초) — 최초 다운로드 포함 기본 30분")
    ap.add_argument("--timeout", type=float, default=900.0, help="잡 완료 대기 상한(초)")
    args = ap.parse_args()
    try:
        pdf = ensure_sample_pdf(args.pdf)
        return run(args.url.rstrip("/"), pdf, args.model_wait, args.timeout)
    except SmokeError as e:
        print(f"✗ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
