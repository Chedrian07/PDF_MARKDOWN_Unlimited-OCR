# PaddleOCR-VL-1.6 sidecar — RTX 5070 Ti (NVIDIA Blackwell)

## 역할

**한국어·다국어 + 완전한 layout** 엔진. layout detection(PP-DocLayout 계열) +
0.9B VL 인식기의 2단 파이프라인으로 페이지의 블록(제목/본문/표/수식/차트/도장/
각주/머리글…)·읽기 순서·bbox·Markdown을 함께 낸다. 시리즈 공식 문서 기준
109개 언어(한국어 명시) 지원.

## 고정값 (2026-07-20 공식 소스 재확인)

| 항목 | 값 |
|---|---|
| 모델 ID | `PaddlePaddle/PaddleOCR-VL-1.6` (파이프라인명 `PaddleOCR-VL-1.6-0.9B`) |
| 모델 revision | `66317acc4c9fc17bd154591ce650735cd2855f3e` (코드 기본값 — 기동 시 `snapshot_download`로 캐시 선점) |
| 라이선스 | Apache-2.0 |
| 파라미터 | 0.9B (BF16, ~959MB) + layout detector |
| paddlepaddle-gpu | **3.3.1** (공식 cu129 인덱스 — Blackwell 가이드 최소 3.2.1) |
| paddleocr | **3.6.0** (`paddleocr[doc-parser]` — VL-1.6 동봉 버전, paddlex 3.6.x 고정) |
| Docker base | `python:3.12-slim-bookworm` + 위 고정 wheel (자체 빌드 — 아래 §설치 경로) |
| dtype | bfloat16 (공식 기본) |

## Blackwell 설치 경로 (공식 가이드 기준)

공식 [PaddleOCR-VL NVIDIA Blackwell 가이드]가 제시하는 두 경로 중 **wheel 경로**를
기본으로 채택했다:

```
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install "paddleocr[doc-parser]==3.6.0"
```

- 가이드는 RTX 5090/5080/**5070 Ti**/5070/5060(Ti)/5050을 대상 목록에 두지만
  **공식 검증 장비는 RTX 5070**이다 — 5070 Ti에서는 반드시
  `scripts/smoke_paddleocr_vl_5070ti.py`로 실측 검증한다.
- 요구 조건: **CUDA 12.9+를 지원하는 NVIDIA 드라이버**, nvidia-container-toolkit.
- 대안: 공식 sm120 Docker 이미지
  (`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:<버전>-nvidia-gpu-sm120`)
  — 중국 레지스트리 접근성 문제로 자체 빌드를 기본으로 한다. `latest` 태그 금지.
- ⚠ `paddleocr install_genai_server_deps`(vLLM 서버 경로)의 사전 빌드 휠은 CUDA
  12.6 대상이라 Blackwell에서 쓰지 않는다 — in-process transformers/paddle 경로 사용.

### revision 고정의 잔여 위험

`PaddleOCRVL` 생성자는 HF revision 인자를 받지 않는다. sidecar는 기동 시
`huggingface_hub.snapshot_download(repo, revision=고정 SHA)`로 캐시를 선점하고
`PADDLE_PDX_MODEL_SOURCE=huggingface`로 HF 경로를 강제한다. 캐시가 비어 있고
스냅샷 다운로드가 실패하면 paddlex가 다른 스냅샷을 받을 수 있다 — health의
`model_revision`은 **의도된 고정값**이며, 엄밀한 검증은 스냅샷 로그로 확인한다.

## 디바이스 정책

- `PADDLEOCR_DEVICE=gpu:0` (기본): layout detector + VL 모두 GPU.
  0.9B BF16 + layout 모델은 16GB에서 여유가 크다 (커뮤니티 실측 vLLM 기준 ~3.3GB,
  in-process 경로는 그보다 높지만 16GB 내 — smoke test의 peak VRAM으로 실측).
- **컴포넌트별 디바이스 분리(layout=CPU, VL=GPU)는 공식 in-process 파이프라인이
  지원하지 않는다.** 공식 분리 경로는 `device="cpu"` + 별도 genai-server인데,
  이는 컨테이너 2개·CUDA 12.6 휠 문제로 이번 범위에서 제외 (VRAM 여유가 커서
  실익도 없음). 따라서 `PADDLEOCR_LAYOUT_DEVICE` 같은 변수는 **의도적으로 없다**.
- 전체 CPU 실행이 필요하면 `PADDLEOCR_DEVICE=cpu` (느림 — 비상용).

## 실행

```bash
docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle   # sidecar(GPU) + backend(:8003)
docker compose --profile paddle logs -f paddleocr-vl
```

health / smoke / 종료:

```bash
curl -s http://127.0.0.1:8003/api/health | python3 -m json.tool
cd backend && uv run python ../scripts/smoke_paddleocr_vl_5070ti.py   # 한국어 문서는 --pdf 지정
docker compose stop paddleocr-vl ocr-paddle    # 대상만 정지 (⚠ ovis/cuda와 동시 기동 금지)
docker compose --profile paddle down           # 전체 정리 (ocr-cpu도 함께 내려감)
```

캐시 삭제:

```bash
docker volume rm pdf_markdown_unlimited-ocr_paddle-hf-cache \
                 pdf_markdown_unlimited-ocr_paddle-x-cache
```

## 결과 스키마 → 프로토콜 변환

`services/paddleocr_vl/app/adapter.py`가 공식 결과(`res.json`)의
`parsing_res_list[].block_bbox(픽셀)/block_label/block_content/block_order`를
내부 프로토콜로 변환한다:

- bbox: 픽셀 → [0,999] 정규화 (2%+2px 초과 이상치는 폐기+warning)
- 라벨 → 정규화 타입 (`doc_title/paragraph_title→title`, `chart/seal→image`,
  `number→page_number` 등)
- markdown 재조립: `block_order` 순. 공식 기본과 동일하게 `number/footnote/
  header(+image)/footer(+image)/aside_text`는 markdown 제외·블록 보존
- 수식은 구분자 없으면 `\[ … \]`로 감싼다. 표 HTML·셀 내 줄바꿈·한글 음절/자모/
  한자/영문 혼용은 무변형 보존 (`tests/test_adapter.py`가 고정)
- 공식 markdown dict(base64 이미지 포함)는 **사용하지 않는다** — 이미지 바이너리
  금지 원칙. figure는 bbox로 backend가 원본 페이지에서 직접 crop
- fixture: `services/paddleocr_vl/tests/fixtures/official_page.json` — 실 GPU에서
  스키마 드리프트 발견 시 실측 결과로 교체하고 어댑터를 함께 갱신할 것

## OOM 완화 순서

1. `OCR_REMOTE_PAGE_CONCURRENCY=1` 확인 (기본값)
2. `PADDLEOCR_MAX_PIXELS` 설정/감소 (예: `4194304` — sidecar도 OOM 시 자동 1회 강등)
3. (출력 토큰 상한 — 공식 파이프라인 노출 옵션 없음, 해당 없음)
4. `RENDER_DPI` 감소 (backend 측 입력 축소)
5. (gpu_memory_utilization — paddle in-process 경로에는 해당 옵션 없음)
6. layout GPU 해제 = `PADDLEOCR_DEVICE=cpu` (전체 CPU — 최후에만)
7. 더 작은 엔진(OvisOCR2) 선택

## 문제 해결

| 증상 | 확인 |
|---|---|
| health `status:error` | `docker compose --profile paddle logs paddleocr-vl` (paddle import/CUDA 오류가 흔함) |
| `The GPU architecture is not supported` 류 | 드라이버가 CUDA 12.9+ 지원인지, wheel이 cu129 인덱스인지 확인 |
| 모델 다운로드 실패 | `PADDLE_PDX_MODEL_SOURCE=huggingface`, HF_TOKEN(프라이빗 미러 시), 네트워크 |
| 한글 깨짐/누락 | smoke를 `--pdf 한국어문서.pdf`로 실행해 재현 — adapter는 무변형 보존이므로 모델/렌더 단 확인 |

## 실행 스레드 정책 (중요 — 실측 기반)

`PaddleModel`은 파이프라인 **생성·워밍업·추론·캐시 해제를 전용 스레드 하나
(`max_workers=1` 실행기)** 에서만 수행한다. 이유는 실측 회귀다:

> 파이프라인을 만든 스레드에서 추론하면 정상인데, **같은 객체를 FastAPI 요청
> 스레드에서 호출하면** 파이프라인의 `vlm` 워커가 static graph 모드로 올라와
> `RuntimeError: Exception from the 'vlm' worker: int(Tensor) is not supported in
> static graph mode`로 **모든** 추론이 실패했다 (2026-07-20 RTX 5070 Ti, paddleocr
> 3.6.0 / paddlepaddle-gpu 3.3.1). 소유 스레드로 고정한 뒤 4연속 실행 모두 정상.

따라서 다음을 지켜야 한다 (수정 시 회귀 위험):

- paddle API를 요청 경로에서 직접 호출하지 않는다 — health의 GPU 이름/총량은
  로드 시 1회 수집한 캐시값이고, 가용량만 `nvidia-smi`로 읽는다.
- 로드 직후 소유 스레드에서 워밍업 추론을 1회 수행해 `vlm` 워커 생성 시점을
  통제한다(첫 사용자 요청 지연도 함께 제거된다).
- 이 실행기가 곧 직렬화 장치라 별도 추론 락은 두지 않는다 (단일 GPU 정책과 일치).

## Known limitations

- 공식 검증 GPU는 RTX 5070 — 5070 Ti는 목록 포함이지만 자체 smoke 필수.
- 페이지 단위 스트리밍 (토큰 델타 없음).
- 취소 시 진행 중 페이지의 추론은 완주 후 폐기.
- chart 인식(`use_chart_recognition`) 등 부가 옵션은 기본 비활성 (공식 기본값).

## 검증 상태

- 구현·어댑터 fixture 테스트(20)·backend 통합 테스트·`docker compose config`: **완료**
- **RTX 5070 Ti 실 runtime 검증 완료 (2026-07-20, `scripts/smoke_paddleocr_vl_5070ti.py` exit 0)**:
  - 환경: WSL2 · driver 591.86 · paddlepaddle-gpu 3.3.1(cu129) · paddleocr 3.6.0 ·
    HF 스냅샷이 고정 revision `66317acc…`로 내려오는 것을 로그로 확인
  - health: `gpu=NVIDIA GeForce RTX 5070 Ti`, `gpu_total_mb=16302`, `gpu_free_mb` 정상 보고
  - 영문 샘플 2페이지: **6.8s (3.4s/페이지)**, figure 2 · 표 1 · layout 정상
  - **한국어 문서 1페이지: 8.1~12.1s**, 한글 음절·자모(ㄱㄴㄷ)·한자·영문 혼용·제목
    계층·표 HTML·figure crop·각주 모두 보존 확인 (3회 연속 재현)
  - **한국어 정확도는 OvisOCR2보다 명확히 우수**(같은 입력 대조): Ovis가
    "혼용된→훈련된", "한글 자모 ㄱㄴㄷ→한국 자료 717"로 오독한 구간을 정확히
    인식했다. 반면 속도는 워밍업 후 Ovis가 2~6배 빠르다 — docs/OCR_BENCHMARK.md
  - **peak VRAM 8,285MB / 16,302MB** — layout+VL을 모두 GPU에 올린 기본 설정에서 OOM 없음
  - 실측으로 발견해 수정한 항목: `vlm` 워커 스레드 이슈(위 §실행 스레드 정책),
    `paddle.device.cuda.mem_get_info` 부재(→ nvidia-smi 조회), 미지 라벨
    `figure_title`·`display_formula`(→ LABEL_MAP 추가)
- **실문서 확장 검증 (2026-07-21)**: 실제 arxiv 논문 2504.19874v1(25p, 2단, 수식
  밀집)을 441s(**17.6s/p** — 밀집 학술 텍스트에서 Ovis의 5배 느림)에 처리, 저자·
  초록·2단 읽기순서·인라인 수식 정확, figure 14 추출, 실패 0. 스캔 시뮬 PDF는
  제목·표·수식 구조는 OCR하나 열화 심한 CJK 줄은 오독(스캔 견고성 한계).
  `OCR_REMOTE_PAGE_CONCURRENCY` 1↔4 출력 바이트 동일(순서 보존 확인).
- 실측으로 보강한 라벨: `figure_title`·`table_title`·`chart_title`·`display_formula`·
  `formula_number`·`reference_content`·`algorithm` → 매핑 추가. **미지 라벨은 내용
  손실이 아니라 경고 로그일 뿐**(content는 markdown에 text로 보존).

[PaddleOCR-VL NVIDIA Blackwell 가이드]: https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html
