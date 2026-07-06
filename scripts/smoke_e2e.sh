#!/usr/bin/env bash
# E2E 스모크: 서버에 샘플 PDF를 업로드하고 결과 마크다운/이미지를 검증한다.
#
#   ./scripts/smoke_e2e.sh                      # http://localhost:8000 (cpu)
#   ./scripts/smoke_e2e.sh http://localhost:8001  # cuda
#   TIMEOUT_SECS=3600 ./scripts/smoke_e2e.sh    # CPU 실모델처럼 오래 걸릴 때

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PDF="$ROOT/sample/sample.pdf"
OUT_DIR="${OUT_DIR:-$ROOT/sample/e2e-out}"

[ -f "$PDF" ] || { echo "샘플 PDF가 없습니다: $PDF (scripts/make_sample_pdf.py 실행)"; exit 1; }
mkdir -p "$OUT_DIR"

echo "── 1. 헬스 체크 ($BASE_URL)"
health=$(curl -sf "$BASE_URL/api/health")
echo "$health" | python3 -m json.tool
engine=$(echo "$health" | python3 -c "import json,sys;print(json.load(sys.stdin)['engine'])")

echo "── 2. 모델 로딩 대기 (engine=$engine)"
if [ "$engine" = "unlimited" ]; then
  start=$(date +%s)
  until curl -sf "$BASE_URL/api/health" | python3 -c "import json,sys;d=json.load(sys.stdin);exit(0 if d['model_loaded'] else 1)"; do
    err=$(curl -sf "$BASE_URL/api/health" | python3 -c "import json,sys;print(json.load(sys.stdin).get('model_load_error') or '')")
    [ -n "$err" ] && { echo "모델 로드 실패: $err"; exit 1; }
    now=$(date +%s); [ $((now - start)) -gt "$TIMEOUT_SECS" ] && { echo "모델 로딩 타임아웃"; exit 1; }
    echo "  ... 로딩 중 ($((now - start))s)"; sleep 10
  done
fi
echo "  모델 준비 완료"

echo "── 3. 업로드: $PDF"
job=$(curl -sf -X POST "$BASE_URL/api/jobs" -F "file=@$PDF;type=application/pdf" -F "mode=multi")
echo "  $job"
job_id=$(echo "$job" | python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])")

echo "── 4. 완료 대기 (job=$job_id, timeout=${TIMEOUT_SECS}s)"
start=$(date +%s)
while :; do
  body=$(curl -sf "$BASE_URL/api/jobs/$job_id")
  status=$(echo "$body" | python3 -c "import json,sys;print(json.load(sys.stdin)['status'])")
  prog=$(echo "$body" | python3 -c "import json,sys;p=json.load(sys.stdin)['progress'];print(f\"{p['phase']} {p['current_page']}/{p['total_pages']} (청크 {p['chunk']}/{p['total_chunks']})\")")
  now=$(date +%s)
  echo "  [$((now - start))s] $status — $prog"
  case "$status" in
    done) break ;;
    error|canceled) echo "실패: $(echo "$body" | python3 -c "import json,sys;print(json.load(sys.stdin)['error'])")"; exit 1 ;;
  esac
  [ $((now - start)) -gt "$TIMEOUT_SECS" ] && { echo "타임아웃"; exit 1; }
  sleep 5
done

echo "── 5. 결과 검증"
curl -sf "$BASE_URL/api/jobs/$job_id/markdown" -o "$OUT_DIR/result.md"
curl -sf "$BASE_URL/api/jobs/$job_id/archive" -o "$OUT_DIR/result.zip"
wc -c "$OUT_DIR/result.md" "$OUT_DIR/result.zip"

python3 - "$OUT_DIR" "$BASE_URL" "$job_id" <<'PY'
import json, sys, urllib.request, zipfile
out, base, jid = sys.argv[1], sys.argv[2], sys.argv[3]
md = open(f"{out}/result.md", encoding="utf-8").read()
assert md.strip(), "마크다운이 비어 있음"
zf = zipfile.ZipFile(f"{out}/result.zip")
names = zf.namelist()
assert "result.md" in names, names
images = [n for n in names if n.startswith("images/")]
print(f"  markdown {len(md)}자, zip 항목 {len(names)}개 (figure {len(images)}개)")
if "![](images/" in md:
    assert images, "마크다운은 이미지를 참조하는데 zip에 이미지가 없음"
    ref = md.split("![](images/", 1)[1].split(")", 1)[0]
    assert f"images/{ref}" in names, f"참조 {ref} 가 zip에 없음"
    print(f"  figure 추출 확인: images/{ref}")
else:
    print("  경고: 마크다운에 이미지 참조 없음 (모델이 figure를 감지하지 못함)")
body = json.load(urllib.request.urlopen(f"{base}/api/jobs/{jid}"))
res = body["result"]
assert res["pages"] and res["layouts"], "pages/layout 산출물 누락"
print(f"  페이지 {len(res['pages'])}개 · 레이아웃 {len(res['layouts'])}개 · figure {len(res['images'])}개")
PY

echo "── E2E 성공 ✔  (결과: $OUT_DIR)"
