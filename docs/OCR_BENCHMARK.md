# OCR 엔진 벤치마크 — 절차와 후보 조사

## 벤치마크 실행

도구: `scripts/benchmark_ocr_engines.py` (사용법·단일 GPU 순차 실행 절차는
[benchmark_docs/README.md](../benchmark_docs/README.md) 참조).

측정값: 엔진/모델/문서명/페이지 수/전체 시간/초당 페이지/페이지 평균 시간/
Markdown 문자 수/표·수식·figure 수/warning 수/실패 페이지 수/peak VRAM/출력 경로.
ground truth 제공 시에만 normalized edit distance·CER·구조 일치도·figure IoU 추가.
**GT 없이 정확도 점수를 만들지 않는다** — 구조 집계는 존재 확인용이다.

### 실측 (2026-07-20, RTX 5070 Ti 16GB · WSL2 · driver 591.86)

`scripts/benchmark_ocr_engines.py`로 **동일 입력·동일 절차**로 순차 측정한 결과다
(스택을 하나씩 기동 — 단일 GPU). 입력은 저장소에서 재생성 가능한 2종:

```bash
cd backend
uv run python ../scripts/make_sample_pdf.py <dir>/en-mixed.pdf            # 영문 2p (표·차트·수식표기)
uv run python ../scripts/make_sample_pdf.py <dir>/ko-report.pdf --korean  # 한국어 1p (표·수식·figure·각주)
```

| engine | doc | pages | total(s) | s/page | md chars | tables | formulas | figures | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| ovis | en-mixed | 2 | 2.2 | **1.1** | 738 | 1 | 0 | 2 | 13,419MB |
| paddle | en-mixed | 2 | 6.8 | 3.4 | 733 | 1 | 0 | 2 | 8,893MB |
| ovis | ko-report | 1 | 2.1 | **2.1** | 579 | 1 | 2 | 1 | 13,419MB |
| paddle | ko-report | 1 | 12.1 | 12.1 | 587 | 1 | 2 | 1 | 10,079MB |

**⚠ 콜드 스타트를 시간에서 반드시 분리할 것**: OvisOCR2의 **첫 요청**은 vLLM
그래프 컴파일 때문에 2페이지에 40~43초가 걸린다(실측 42.8s). 위 표는 그 이후의
정상 상태 수치다 — 컨테이너를 새로 띄운 직후 한 번 측정하고 "20초/페이지"라고
적으면 20배 틀린 값이 된다(이 문서의 초판이 그 오류를 냈다).

### 한국어 텍스트 정확도 (같은 입력, 실측 출력 대조)

| 원문 | OvisOCR2 | PaddleOCR-VL-1.6 |
|---|---|---|
| 혼용된 | **훈련된** ✗ | 혼용된 ✓ |
| 한글 자모 ㄱㄴㄷ | **한국 자료 717** ✗ | 한글 자모 ㄱㄴㄷ ✓ |
| 硏究報告書 | 研究**报**告書 (간체 혼입) | 研究報告書 ✓ |
| 보존 | **보준** ✗ | 보존 ✓ |
| 표 셀 `1,234` | `1,234` ✓ | `1, 2 3 4` (숫자 사이 공백) |

→ **한국어 본문 정확도는 PaddleOCR-VL이 명확히 우수**하고, 표 셀의 숫자 붙임은
OvisOCR2가 정확하다. 두 엔진 모두 제목 계층·표 HTML·figure·각주 구조는 보존한다.
(숫자 공백은 이 합성 PDF의 CJK 폰트 자간 특성일 수 있어 실문서로 재확인 권장.)

### 실문서 검증 (2026-07-21, 저장소 내 실제 arxiv 논문 + 스캔 시뮬)

합성 샘플 외에 **실제 문서 유형**으로 확장 검증했다 (입력은 저장소에 있거나
재생성 가능):

| 문서 | 유형 | 엔진 | 결과 |
|---|---|---|---|
| `sample/2504.19874v1.pdf` (25p) | 영문 논문·**2단**·수식 밀집·긴 PDF | Ovis | done 82s(3.3s/p)·figure 27·실패 0 · **저자 블록 읽기순서·display/inline LaTeX·Lemma 구조 정확** |
| 〃 | 〃 | Paddle | done 441s(**17.6s/p**)·figure 14·실패 0 · 저자·초록·인라인 수식·2단 읽기순서 정확, 미지 라벨 경고 1(algorithm) |
| `sample/unlimited-ocr-paper.pdf` (14p) | 영문 논문·figure | Ovis / Paddle | done·실패 0 · figure 2~3 · 표 4 |
| `scan-ko.pdf` (`make_sample_pdf.py --scan`) | **스캔 시뮬**(텍스트 레이어 0·1.4° 기울임·노이즈·JPEG q55) | Ovis / Paddle | **제목·표·수식 구조는 정확**하게 OCR하지만 열화가 심한 CJK 줄은 오독(예: "ㄱㄴㄷ"→"그늘", "硏究報告書"→"컨혈갑"). 내장 텍스트 fallback이 아니라 진짜 OCR(텍스트 레이어 0) — 스캔 견고성의 한계를 보여준다 |

**다단(2-column) 읽기 순서**: OvisOCR2는 2504.19874의 2단 저자 블록·본문을 읽기
순서대로 평탄화하고 수식을 보존했다(발췌: `$$ D(p_X,B):=\inf\{…\} $$`,
`$I(x;y)=h(x)-h(x|y)$`). 실측 출력은 위 표의 판정 근거다.

**속도 격차(실문서에서 확대)**: 밀집 학술 텍스트에서 Ovis는 3.3s/p인데 Paddle은
블록별 layout+VL 비용으로 **~17s/p**까지 느려진다(14p 244s). 짧은/한국어 문서는
Paddle, 길고 밀집한 영문 문서는 Ovis가 유리하다는 앞의 결론이 실문서에서도 유지된다.

> ground truth가 없어 편집거리·CER은 계산하지 않았다(구조·읽기순서·수식 보존은
> 실측 출력 대조로 확인). 정량 정확도가 필요하면 정답 md를 만들어
> `--ground-truth`로 측정할 것.

## 확정 엔진 요약

| | Unlimited-OCR (유지) | OvisOCR2 | PaddleOCR-VL-1.6 |
|---|---|---|---|
| 모델 | baidu/Unlimited-OCR 3.3B MoE | ATH-MaaS/OvisOCR2 0.9B | PaddlePaddle/PaddleOCR-VL-1.6 0.9B |
| 라이선스 | MIT | Apache-2.0 | Apache-2.0 |
| 실행 | in-process torch | vLLM 0.22.1 sidecar | paddle 3.3.1 sidecar |
| 강점 | 멀티페이지 문맥·토큰 스트리밍 | 페이지 정밀 파싱·figure bbox | 한국어·layout 블록·읽기 순서 |
| layout | full (그라운딩 토큰) | figure_only | full (블록+순서) |
| 16GB 적합성 | 검증됨 (~7GB) | 여유 큼 (util 0.80) | 여유 큼 |

## 후보 조사 (구현 제외 — 2026-07-20 공식 소스 기준)

| 후보 | 크기/라이선스 | CUDA·Blackwell | 16GB BF16 | 한국어 근거 | bbox/layout | 통합 비용 | 판정 |
|---|---|---|---|---|---|---|---|
| `zai-org/GLM-OCR` | 1.33B(safetensors)/MIT | 공식 언급 없음 | 여유 (~2.7GB) | HF 태그에 `ko` (본문 근거·벤치 없음) | 모델 자체 bbox 없음 — 별도 PP-DocLayoutV3 2단 파이프라인 | 높음 (vLLM nightly + transformers git HEAD 요구) | **제외** — OmniDocBench 1위(94.62)지만 bbox가 Paddle layout 의존이라 PaddleOCR-VL과 역할 중복, 스택이 nightly 의존 |
| `baidu/Qianfan-OCR` | 4.74B/Apache-2.0 | A100 벤치만, Blackwell 언급 없음 | **빠듯** (~9.5GB 가중치 + thinking 16K KV + 4K 비전) | 없음 ("192 languages"에 한국어 미명명) | Layout-as-Thought — 좌표 형식 미문서화 | 중간 (trust_remote_code) | **제외** — 16GB 헤드룸 부족, 공식 저비트 경로 없음 |
| `deepseek-ai/DeepSeek-OCR-2` | 3.39B/Apache-2.0 | **공식 스택이 CUDA 11.8 + torch 2.6 + 커스텀 vllm-0.8.5 wheel — sm_120 불가** | 여유 (~6.8GB, 돌릴 수 있다면) | 없음 | grounding 프롬프트 존재하나 출력 형식 미문서화 | 높음 (레거시 고정 스택) | **제외** — Blackwell 공식 경로 부재가 결정적 |
| `rednote-hilab/dots.mocr` | 3.04B/MIT | cu128/cu130 스택 권장 (사실상 Blackwell 가능, 공식 RTX50 언급은 없음) | 여유 (~6.1GB) | 없음 (showcase에 한국어 부재) | **가장 우수** — 페이지 JSON(11개 카테고리+bbox+content) | 중저 (vLLM 0.11+ 통합, trust_remote_code) | **제외** — olmOCR-bench 83.9로 매력적이나 한국어 근거가 전무해 이번 목적(한국어 문서) 대비 이점 불명확. 차기 후보 1순위 |

공통 결론: 네 후보 모두 "공식 Blackwell 지원 + 한국어 근거 + 16GB 여유"를 동시에
만족하지 못한다. OvisOCR2(페이지 파싱·figure)와 PaddleOCR-VL-1.6(한국어·layout)의
역할을 명백히 대체하는 후보가 없어 구현 범위에 추가하지 않았다.
"지원 완료"로 표기된 엔진은 fake/unlimited/ovisocr2/paddleocr_vl 4종뿐이다.
