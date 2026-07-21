# OvisOCR2 sidecar — RTX 5070 Ti (CUDA, Blackwell)

## 역할

페이지 단위 **정밀 문서 파싱 + figure bbox** 엔진. 한 번의 생성으로 Markdown·
표 HTML·LaTeX·figure bbox(`<img src="images/bbox_l_t_r_b.jpg" />`, [0,1000))를
출력하는 0.9B 경량 모델이다. 멀티페이지 문맥·토큰 스트리밍이 필요하면
Unlimited-OCR을, 한국어 레이아웃 중심이면 PaddleOCR-VL을 선택한다.

## 고정값 (2026-07-20 공식 소스 재확인)

| 항목 | 값 |
|---|---|
| 모델 ID | `ATH-MaaS/OvisOCR2` |
| 모델 revision | `65c619d374b55d4152e85150fc1b003700bc1f0c` (코드 기본값 — env로 오버라이드 가능) |
| 라이선스 | Apache-2.0 |
| 파라미터 | 852.9M (BF16 safetensors 단일 파일, ~1.7GB) |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` (GDN 하이브리드 어텐션) |
| runtime | vLLM **0.22.1** (모델 카드 공식 권장 — `pip install "vllm==0.22.1"`) |
| Docker base | `vllm/vllm-openai:v0.22.1-cu129` (Blackwell=CUDA≥12.8, cu129 태그 고정) |
| dtype | bfloat16 |
| 프롬프트 | 모델 카드 공식 OCR 프롬프트 원문 (`services/ovisocr2/app/model.py::OFFICIAL_PROMPT`) |

## RTX 5070 Ti 16GB VRAM 정책 (기본값 근거)

| 설정 | 기본값 | 근거 |
|---|---|---|
| `OVIS_GPU_MEMORY_UTILIZATION` | 0.80 | 데스크톱/WSL 디스플레이·런타임 몫 확보. 0.92 초과는 설정 거부 |
| `OVIS_MAX_MODEL_LEN` | 24576 | max_pixels 2880²에서 비전 토큰 ~8K + 출력 8192 + 프롬프트 여유. 하이브리드 어텐션이라 KV 비용 미미 |
| `OVIS_MAX_OUTPUT_TOKENS` | 8192 | 페이지별 생성 상한 (폭주 방지) |
| `OVIS_MAX_NUM_SEQS` | 1 | 단일 GPU 예측 가능성 — 동시 시퀀스 없음 |
| `OVIS_MIN_PIXELS` / `OVIS_MAX_PIXELS` | 448² / 2880² | 모델 카드 공식 값 |
| `OVIS_GDN_PREFILL_BACKEND` | `triton` | **sm_120 필수** — FlashInfer GDN 경로는 데이터센터 아키텍처 게이트. 모델 카드 공식 예제와 동일 |

## 실행

```bash
docker compose --profile ovis up -d --build ovisocr2 ocr-ovis   # sidecar(GPU) + backend(:8002)
# 진행 로그(모델 다운로드 ~1.7GB 포함):
docker compose --profile ovis logs -f ovisocr2
```

health 확인:

```bash
curl -s http://127.0.0.1:8002/api/health | python3 -m json.tool   # backend 관점
docker compose --profile ovis exec ovisocr2 \
  python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8080/health').read().decode())"
```

smoke test (실 GPU):

```bash
cd backend && uv run python ../scripts/smoke_ovisocr2_5070ti.py
```

종료 / 전환:

```bash
docker compose stop ovisocr2 ocr-ovis         # 대상만 정지 (⚠ paddle/cuda와 동시 기동 금지)
docker compose --profile ovis down            # 볼륨 외 전체 정리 (ocr-cpu도 함께 내려감)
```

캐시 삭제 (모델 재다운로드):

```bash
docker volume rm pdf_markdown_unlimited-ocr_ovis-hf-cache
```

## 출력 파싱 (strict)

`services/ovisocr2/app/parser.py` — 다음 형식만 figure로 인정:

```
<img src="images/bbox_{l}_{t}_{r}_{b}.jpg" />     (공백·slash 생략 변형만 허용)
```

- 좌표는 각각 ≤4자리 정수, [0,1000] 범위(1000은 999로 clamp), x1<x2·y1<y2,
  최소 변 2, 중복 제거, 페이지당 64개 상한
- 그 밖의 모든 `<img …>` 태그(외부 URL·경로 탈출·비정상 속성·닫히지 않은 태그)는
  **제거**되고 warning으로 기록
- 유효 태그는 순서대로 `[[FIGURE:n]]` placeholder로 치환 — 파일명은 절대 모델
  출력에서 오지 않는다
- 반복 suffix 정리: 모델 카드의 `_clean_truncated_repeats` 알고리즘을 의미론
  그대로 구현 (독립 unit test: `services/ovisocr2/tests/test_parser.py`)

## OOM 완화 순서

1. `OCR_REMOTE_PAGE_CONCURRENCY=1` 확인 (기본값)
2. `OVIS_MAX_PIXELS` 감소 (예: `4194304`=2048² — sidecar도 OOM 시 자동 1회 강등)
3. `OVIS_MAX_OUTPUT_TOKENS` 감소 (예: 4096)
4. `OVIS_MAX_MODEL_LEN` 감소 (max_pixels를 줄였다면 함께: 픽셀/1024 ≈ 비전 토큰)
5. `OVIS_GPU_MEMORY_UTILIZATION` 조정 (0.75 → 0.70)
6. (해당 없음 — 이 sidecar는 layout detector가 없다)
7. 마지막 수단: 더 작은 입력(RENDER_DPI 150) 또는 다른 엔진 선택

CPU offload는 사용하지 않는다.

## 문제 해결

| 증상 | 확인 |
|---|---|
| health `status:error` | `docker compose --profile ovis logs ovisocr2` — load_error 필드에 요약 |
| `엔진 서버 연결 안 됨` 배지 | sidecar 컨테이너 기동/healthcheck 상태 (`docker compose ps`) |
| 첫 잡이 `아직 모델을 로드하지` 오류 | 최초 다운로드 중 — 로그로 진행 확인 후 재시도 |
| Triton/GDN 관련 크래시 | `OVIS_GDN_PREFILL_BACKEND=triton` 유지 확인 (sm_120에서 flashinfer 불가) |
| 응답 시간 초과 | 페이지 해상도↓(`RENDER_DPI`) 또는 `OCR_SIDECAR_READ_TIMEOUT_S`↑ |

## Known limitations

- **layout_capability = figure_only**: 텍스트 블록 bbox를 제공하지 않는다.
  레이아웃 뷰에는 figure 박스만 표시된다 (UI가 안내 문구 표시).
- 스트리밍은 페이지 단위 — 토큰 델타 스트림 없음 (가짜 스트리밍 미구현이 의도).
- 취소 시 진행 중인 페이지의 GPU 추론은 완주 후 폐기된다 (프로토콜 문서 §취소).
- **한국어 본문 정확도가 PaddleOCR-VL보다 낮다(실측)**: 자체 한국어 샘플에서
  "혼용된→훈련된", "한글 자모 ㄱㄴㄷ→한국 자료 717", "보존→보준" 오독과 한자
  간체 혼입이 확인됐다. 한국어 중심 문서는 PaddleOCR-VL을 권장한다
  (반대로 표 셀 숫자는 Ovis가 정확했다 — docs/OCR_BENCHMARK.md 대조표).
- 첫 요청 컴파일 비용(~40초/2페이지)이 크다 — 짧은 문서를 가끔 처리하는 용도라면
  체감이 나쁠 수 있다(컨테이너를 계속 띄워 두면 해소).

## 검증 상태

- 구현·파서 fixture 테스트(28)·backend 통합 테스트·`docker compose config`: **완료**
- **RTX 5070 Ti 실 runtime 검증 완료 (2026-07-20, `scripts/smoke_ovisocr2_5070ti.py` exit 0)**:
  - 환경: WSL2 · driver 591.86 · `vllm/vllm-openai:v0.22.1-cu129` ·
    revision `65c619d3…` 고정 로드 확인 (`gdn_prefill_backend=triton`)
  - 모델 로드: 가중치 1.72GiB, health `gpu=NVIDIA GeForce RTX 5070 Ti`
  - 샘플 PDF(2페이지, 이미지 2·표 1): figure crop 2/2 · 표 1 · layout 오버레이 생성,
    실패 페이지 0
  - **성능은 콜드/웜을 분리해야 한다**: 컨테이너 기동 후 **첫 요청**은 vLLM 그래프
    컴파일로 2페이지에 **40~43초**(실측 42.8s), 이후 정상 상태는 **2페이지 2.2초
    (1.1초/페이지)** 로 20배 이상 빨라진다. 벤치마크·체감 성능을 논할 때 첫 요청
    수치를 쓰면 안 된다 (docs/OCR_BENCHMARK.md 실측표 참조)
  - **peak VRAM 12,969MB / 16,302MB** — `OVIS_GPU_MEMORY_UTILIZATION=0.80` 예산 내, OOM 없음
  - 잡 메타에 "페이지 단위 모델" 안내 warning 정상 기록
- **실문서 확장 검증 (2026-07-21)**: 실제 arxiv 논문 2504.19874v1(25p, 2단, 수식
  밀집)을 82s(3.3s/p)에 처리 — 2단 저자 블록 읽기순서·display/inline LaTeX·Lemma
  구조 정확, figure 27개 추출, 실패 0. 스캔 시뮬 PDF(텍스트 레이어 0)도 정확히
  OCR. `OCR_REMOTE_PAGE_CONCURRENCY` 1↔4 출력 바이트 동일(순서 보존 확인).
- 참고: 이 머신은 5070 Ti + 4060 Ti 2-GPU 구성 — vLLM이 두 GPU를 감지하지만
  CUDA 기본(fastest-first) 순서로 device 0 = 5070 Ti가 선택된다. 결정적 선택이
  필요하면 compose에 `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 추가하고 `GPU_DEVICE`를
  PCI 순서 기준으로 지정하라 (단일 GPU 원칙은 불변 — TP=1, 분산 없음).
