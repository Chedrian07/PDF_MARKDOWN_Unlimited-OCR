#!/usr/bin/env python3
"""OCR 엔진 비교 벤치마크 — 같은 PDF들을 실행 중인 backend들에 순차 제출한다.

    python scripts/benchmark_ocr_engines.py \
      --endpoint unlimited=http://127.0.0.1:8001 \
      --endpoint ovis=http://127.0.0.1:8002 \
      --endpoint paddle=http://127.0.0.1:8003 \
      --input benchmark_docs/ --out bench_out/

⚠ 단일 GPU에서는 한 시점에 한 스택만 기동할 수 있으므로, 보통 스택을 하나씩
띄워 같은 --out 디렉터리에 이어서 실행한다 (결과는 endpoint별로 병합·누적).

출력: {out}/results.json · results.csv · summary.md, 엔진·문서별 markdown 사본.

정확도: --ground-truth DIR에 {문서stem}.md가 있을 때만 normalized edit distance /
CER / 구조 일치도를 계산한다. GT가 없으면 어떤 "정확도 점수"도 만들지 않는다.
figure IoU는 {문서stem}.boxes.json([[x1,y1,x2,y2] 0–999 목록])이 있을 때만.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import (  # noqa: E402
    SmokeError, VramSampler, count_markers, http_json, http_text,
    upload_pdf, wait_job,
)

_EDIT_DISTANCE_CAP = 20_000  # DP O(nm) 상한 — 초과분은 앞부분만 비교하고 표기


def edit_distance(a: str, b: str) -> int:
    """고전 Levenshtein DP (외부 의존성 없음). 호출자가 길이 상한을 보장한다."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _normalize_for_gt(text: str) -> str:
    text = re.sub(r"!\[\]\([^)]*\)", "", text)      # 이미지 참조 제거 (경로는 엔진별)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def gt_metrics(result_md: str, gt_md: str) -> dict:
    a = _normalize_for_gt(result_md)
    b = _normalize_for_gt(gt_md)
    truncated = len(a) > _EDIT_DISTANCE_CAP or len(b) > _EDIT_DISTANCE_CAP
    a_c, b_c = a[:_EDIT_DISTANCE_CAP], b[:_EDIT_DISTANCE_CAP]
    dist = edit_distance(a_c, b_c)
    denom = max(len(a_c), len(b_c), 1)
    rs, gs = count_markers(result_md), count_markers(gt_md)

    def _ratio(x: int, y: int) -> float:
        return round(min(x, y) / max(x, y, 1), 3) if max(x, y) else 1.0

    return {
        "normalized_edit_distance": round(dist / denom, 4),
        "cer": round(dist / max(len(b_c), 1), 4),
        "gt_truncated_compare": truncated,
        "structure_match": {
            "tables": _ratio(rs["tables"], gs["tables"]),
            "formulas": _ratio(rs["formulas"], gs["formulas"]),
            "figures": _ratio(rs["figures"], gs["figures"]),
        },
    }


def figure_iou(result_boxes: list, gt_boxes: list) -> float | None:
    """greedy 매칭 mean IoU (0–999 정규화 bbox 목록끼리)."""
    def iou(p, q):
        ix1, iy1 = max(p[0], q[0]), max(p[1], q[1])
        ix2, iy2 = min(p[2], q[2]), min(p[3], q[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter)
        return inter / area if area > 0 else 0.0

    if not gt_boxes:
        return None
    remaining = list(result_boxes)
    scores = []
    for g in gt_boxes:
        best_i, best = -1, 0.0
        for i, r in enumerate(remaining):
            s = iou(r, g)
            if s > best:
                best_i, best = i, s
        scores.append(best)
        if best_i >= 0:
            remaining.pop(best_i)
    return round(sum(scores) / len(scores), 3)


def bench_one(name: str, url: str, pdf: Path, timeout_s: float, out_dir: Path,
              gt_dir: Path | None) -> dict:
    health = http_json(f"{url}/api/health")
    t0 = time.monotonic()
    with VramSampler() as vram:
        job_id = upload_pdf(url, pdf)
        body = wait_job(url, job_id, timeout_s)
    elapsed = time.monotonic() - t0
    row: dict = {
        "engine": name,
        "backend_engine": health.get("engine"),
        "model": body.get("model_id") or health.get("model_id"),
        "model_revision": body.get("model_revision"),
        "document": pdf.name,
        "status": body["status"],
        "pages": body["progress"].get("total_pages", 0),
        "total_s": round(elapsed, 1),
        "pages_per_s": round(body["progress"].get("total_pages", 0) / elapsed, 3) if elapsed else 0,
        "avg_page_s": round(elapsed / max(body["progress"].get("total_pages", 1), 1), 1),
        "warnings": len(body.get("warnings", [])),
        "peak_vram_mb": vram.peak_mb if vram.available else None,
        "error": body.get("error"),
    }
    if body["status"] != "done":
        return row

    md = http_text(f"{url}/api/jobs/{job_id}/markdown")
    stats = count_markers(md)
    row.update({
        "md_chars": stats["chars"], "tables": stats["tables"],
        "formulas": stats["formulas"], "figures": stats["figures"],
        "failed_pages": stats["failed_pages"],
    })
    dest = out_dir / name / f"{pdf.stem}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    row["output_path"] = str(dest)

    if gt_dir is not None:
        gt_md = gt_dir / f"{pdf.stem}.md"
        if gt_md.is_file():
            row["gt"] = gt_metrics(md, gt_md.read_text(encoding="utf-8"))
        gt_boxes_f = gt_dir / f"{pdf.stem}.boxes.json"
        if gt_boxes_f.is_file():
            try:
                gt_boxes = json.loads(gt_boxes_f.read_text(encoding="utf-8"))
                # figure crop 픽셀 좌표(images/boxes.json — 공개 files API) → 0–999 정규화
                boxes_raw = json.loads(
                    http_text(f"{url}/api/jobs/{job_id}/files/images/boxes.json")
                )
                result_boxes = []
                for meta in boxes_raw.values():
                    w = max(int(meta.get("image_width", 0)), 1)
                    hgt = max(int(meta.get("image_height", 0)), 1)
                    result_boxes.append([
                        round(meta["x1"] / w * 999), round(meta["y1"] / hgt * 999),
                        round(meta["x2"] / w * 999), round(meta["y2"] / hgt * 999),
                    ])
                row["figure_iou"] = figure_iou(result_boxes, gt_boxes)
            except (SmokeError, json.JSONDecodeError, KeyError, OSError, ValueError):
                row["figure_iou"] = None
    return row


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    results_json = out_dir / "results.json"
    # 스택을 하나씩 띄워 여러 번 실행하는 워크플로 — 기존 결과에 병합 누적
    existing: list[dict] = []
    if results_json.is_file():
        try:
            existing = json.loads(results_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    keyed = {(r.get("engine"), r.get("document")): r for r in existing}
    for r in rows:
        keyed[(r.get("engine"), r.get("document"))] = r
    merged = list(keyed.values())
    results_json.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    cols = ["engine", "model", "model_revision", "document", "status", "pages",
            "total_s", "pages_per_s", "avg_page_s", "md_chars", "tables",
            "formulas", "figures", "warnings", "failed_pages", "peak_vram_mb",
            "output_path"]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)

    lines = ["# OCR 엔진 벤치마크 요약", "",
             "| engine | model | doc | pages | total(s) | s/page | md chars | tables | formulas | figures | warn | peak VRAM(MB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(merged, key=lambda r: (r.get("document") or "", r.get("engine") or "")):
        lines.append(
            f"| {r.get('engine')} | {r.get('model')} | {r.get('document')} | {r.get('pages')} "
            f"| {r.get('total_s')} | {r.get('avg_page_s')} | {r.get('md_chars', '-')} "
            f"| {r.get('tables', '-')} | {r.get('formulas', '-')} | {r.get('figures', '-')} "
            f"| {r.get('warnings')} | {r.get('peak_vram_mb') or '-'} |")
    gts = [r for r in merged if "gt" in r]
    if gts:
        lines += ["", "## Ground truth 지표", "",
                  "| engine | doc | norm. edit dist | CER | tables≈ | formulas≈ | figures≈ | fig IoU |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in gts:
            g = r["gt"]
            s = g["structure_match"]
            lines.append(
                f"| {r['engine']} | {r['document']} | {g['normalized_edit_distance']} "
                f"| {g['cer']} | {s['tables']} | {s['formulas']} | {s['figures']} "
                f"| {r.get('figure_iou', '-')} |")
    else:
        lines += ["", "> ground truth 미제공 — 정확도 지표 없음 (구조 집계는 존재 확인용이지 점수가 아님)"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", action="append", required=True,
                    metavar="NAME=URL", help="예: ovis=http://127.0.0.1:8002 (반복 가능)")
    ap.add_argument("--input", type=Path, required=True, help="PDF 파일 또는 디렉터리")
    ap.add_argument("--out", type=Path, default=Path("bench_out"))
    ap.add_argument("--ground-truth", type=Path, default=None,
                    help="{stem}.md / {stem}.boxes.json이 있는 디렉터리 (선택)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="문서당 대기 상한(초)")
    args = ap.parse_args()

    endpoints: list[tuple[str, str]] = []
    for e in args.endpoint:
        if "=" not in e:
            ap.error(f"--endpoint 형식은 name=url: {e!r}")
        name, url = e.split("=", 1)
        endpoints.append((name.strip(), url.strip().rstrip("/")))

    pdfs = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if not pdfs:
        print(f"✗ PDF 없음: {args.input}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, url in endpoints:
        try:
            health = http_json(f"{url}/api/health")
        except SmokeError as e:
            print(f"✗ {name} ({url}) 접속 불가 — 건너뜀: {e}")
            print("  (단일 GPU에서는 스택을 하나씩 띄워 같은 --out으로 재실행해 병합하세요)")
            continue
        print(f"● {name}: engine={health.get('engine')} model={health.get('model_id')}")
        for pdf in pdfs:
            print(f"  - {pdf.name} …", flush=True)
            try:
                row = bench_one(name, url, pdf, args.timeout, args.out, args.ground_truth)
            except SmokeError as e:
                row = {"engine": name, "document": pdf.name, "status": "error",
                       "error": str(e)[:300], "pages": 0, "total_s": None,
                       "warnings": 0}
                print(f"    ✗ {e}")
            else:
                print(f"    {row['status']} · {row.get('total_s')}s · "
                      f"figure {row.get('figures', '-')} · peak {row.get('peak_vram_mb')}MB")
            rows.append(row)

    if not rows:
        print("✗ 실행된 벤치마크가 없습니다")
        return 1
    write_outputs(rows, args.out)
    print(f"\n결과: {args.out}/results.json · results.csv · summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
