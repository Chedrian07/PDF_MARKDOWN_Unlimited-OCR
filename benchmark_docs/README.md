# benchmark_docs — 엔진 비교용 평가 문서 모음

`scripts/benchmark_ocr_engines.py --input benchmark_docs/`가 이 디렉터리의 PDF를
순차 변환해 엔진별 처리 시간·구조 집계를 비교한다.

**저작권이 있는 문서를 이 저장소에 커밋하지 마세요.** 이 디렉터리는 로컬 평가용
자리이며, PDF는 `.gitignore` 대상이 아니더라도 직접 관리 책임은 사용자에게 있다.

## 바로 쓸 수 있는 생성 입력 (저작권 문제 없음)

저장소 스크립트로 두 종을 즉시 만들 수 있다 — `docs/OCR_BENCHMARK.md`의 실측표가
이 두 문서로 측정한 것이다:

```bash
cd backend
uv run python ../scripts/make_sample_pdf.py ../benchmark_docs/en-mixed.pdf
uv run python ../scripts/make_sample_pdf.py ../benchmark_docs/ko-report.pdf --korean
```

`--korean`은 한글 음절·자모·한자 혼용·숫자와 단위·표(셀 줄바꿈)·수식 주변 한글·
figure·각주·페이지 번호를 한 페이지에 담아 한국어 보존을 검증한다.

## 권장 평가 문서 유형

엔진별 강점이 갈리는 지점을 모두 덮으려면 다음 유형을 하나씩 준비하는 것을 권장:

1. **한국어 일반 문서** — 보고서/공문 (한글 음절·자모·문단 구조)
2. **한국어 표 중심 문서** — 셀 병합·셀 내 줄바꿈 포함
3. **영문 논문** — 2단 레이아웃·각주·참고문헌
4. **수식 교재/논문** — display/inline LaTeX 밀도 높은 페이지
5. **다단(3단+) 문서** — 신문/뉴스레터류 읽기 순서
6. **스캔 문서** — 기울어짐/노이즈 (텍스트 레이어 없음)
7. **차트·figure 중심 문서** — 그림 추출(crop)·캡션
8. **영수증/양식** — 키-값 쌍, 도장(seal)
9. **한·영 혼용 기술 문서** — 코드 블록/식별자 혼재
10. **긴 PDF (30p+)** — 장문 안정성·반복 루프 내성

## Ground truth (선택)

`--ground-truth DIR`에 다음 파일이 있으면 정확도 지표가 추가된다:

- `{문서stem}.md` — 사람이 검수한 정답 Markdown
  → normalized edit distance · CER · 표/수식/figure 수 일치도
- `{문서stem}.boxes.json` — figure bbox 정답 `[[x1,y1,x2,y2], …]` (0–999 정규화)
  → figure bbox mean IoU

GT가 없으면 벤치마크는 **속도·구조 집계만** 출력한다 — 임의의 "정확도 점수"를
만들지 않는다.

## 단일 GPU 실행 순서 (RTX 5070 Ti)

```bash
docker compose up -d --build ocr-cuda        # ① Unlimited
python scripts/benchmark_ocr_engines.py --endpoint unlimited=http://127.0.0.1:8001 \
  --input benchmark_docs/ --out bench_out/
docker compose stop ocr-cuda

docker compose --profile ovis up -d --build ovisocr2 ocr-ovis  # ② OvisOCR2
python scripts/benchmark_ocr_engines.py --endpoint ovis=http://127.0.0.1:8002 \
  --input benchmark_docs/ --out bench_out/
docker compose stop ovisocr2 ocr-ovis

docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle  # ③ PaddleOCR-VL
python scripts/benchmark_ocr_engines.py --endpoint paddle=http://127.0.0.1:8003 \
  --input benchmark_docs/ --out bench_out/
docker compose stop paddleocr-vl ocr-paddle
```

같은 `--out`을 쓰면 `results.json`에 엔진별 결과가 병합 누적되고 `summary.md`가
전체 비교표로 갱신된다.
