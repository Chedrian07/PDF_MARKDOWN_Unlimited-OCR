#!/usr/bin/env python3
"""PaddleOCR-VL-1.6 스택 실 GPU smoke test (RTX 5070 Ti 검증 절차의 2단계).

전제: docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle  (backend: :8003)
      (⚠ ovis/cuda 스택과 동시 기동 금지 — 단일 GPU VRAM 경쟁)

    cd backend && uv run python ../scripts/smoke_paddleocr_vl_5070ti.py
    # 한국어 문서 검증: --pdf ko-doc.pdf 로 실제 한국어 PDF를 지정 권장

확인 항목: health → 모델 로드 대기 → 변환 → Markdown/layout(전체 블록)/figure/
table/formula → peak VRAM/처리 시간/OOM. 종료 코드 0 = 통과.
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

ENGINE = "paddleocr_vl"


def run(url: str, pdf: Path, model_wait_s: float, job_timeout_s: float) -> int:
    failures: list[str] = []

    print(f"[1/5] health 확인: {url}")
    health = http_json(f"{url}/api/health")
    if health.get("engine") != ENGINE:
        raise SmokeError(f"engine={health.get('engine')} — {ENGINE} 스택(:8003)이 아닙니다")

    print(f"[2/5] 모델 로드 대기 (최대 {model_wait_s:.0f}s)")
    health = wait_model_loaded(url, model_wait_s)
    ph = health.get("provider_health") or {}
    print(f"  로드 완료: gpu={health.get('gpu_name')} "
          f"free={ph.get('gpu_free_mb')}MB/{ph.get('gpu_total_mb')}MB "
          f"runtime={ph.get('runtime')} {ph.get('version')}")
    caps = health.get("capabilities") or {}
    if caps.get("layout") != "full":
        failures.append(f"capabilities.layout={caps.get('layout')} (기대: full)")

    print(f"[3/5] PDF 변환: {pdf}")
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
    for w in body.get("warnings", []):
        print(f"  ⚠ warning: {w}")
    oom = [w for w in body.get("warnings", []) if "메모리" in w or "강등" in w]
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
        failures.append("figure crop 없음")
    if stats["tables"] < 1:
        failures.append("표 없음")
    if stats["failed_pages"]:
        failures.append(f"실패 페이지 {stats['failed_pages']}개")
    result = body.get("result") or {}
    if not result.get("has_layout"):
        failures.append("layout.json 없음 (full layout 엔진)")
    # 한국어 보존: 한글 음절이 하나라도 있는 문서라면 결과에도 있어야 한다
    # (기본 영문 샘플에서는 스킵 — --pdf로 한국어 문서를 지정해 검증하세요)
    if any("가" <= ch <= "힣" for ch in md):
        print("  ✓ 한글 음절 보존 확인")
    else:
        print("  ⓘ 결과에 한글 없음 — 한국어 검증은 --pdf로 한국어 문서를 지정하세요")

    print("[5/5] 판정")
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("  ✓ PaddleOCR-VL 스택 smoke test 통과 — 문서의 '실제 검증' 항목을 갱신하세요")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8003")
    ap.add_argument("--pdf", type=Path, default=Path("sample/sample.pdf"))
    ap.add_argument("--model-wait", type=float, default=1800.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    try:
        pdf = ensure_sample_pdf(args.pdf)
        return run(args.url.rstrip("/"), pdf, args.model_wait, args.timeout)
    except SmokeError as e:
        print(f"✗ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
