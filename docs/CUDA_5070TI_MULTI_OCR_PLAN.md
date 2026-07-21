# RTX 5070 Ti (16GB, Blackwell) 단일 GPU — 멀티 OCR 엔진 계획서

> 이 문서는 OvisOCR2·PaddleOCR-VL-1.6 sidecar 통합의 설계 SSOT다.
> 구현과 문서가 어긋나면 이 문서를 갱신한다. 기존 전체 아키텍처는
> [ARCHITECTURE.md](ARCHITECTURE.md) 참조.

## 0. 목표와 비목표

**목표**: 기존 `baidu/Unlimited-OCR` 기반 PDF→Markdown 서비스의 모든 기능을
보존하면서, RTX 5070 Ti 16GB 단일 GPU에서 사용자가 **선택적으로** 더 높은 문서
파싱 정확도의 CUDA OCR 엔진(OvisOCR2, PaddleOCR-VL-1.6)을 쓸 수 있게 한다.

**비목표**: 다중 GPU / TP·PP / 자동 모델 라우팅 / 요청별 모델 스왑 /
클라우드 OCR API / 프론트엔드 재작성 / Kubernetes / 이기종 분산.

**단일 GPU 원칙**: 한 시점에 GPU OCR 모델은 정확히 하나만 활성화된다.
Compose profile로 엔진을 선택하고, 신규 엔진의 메인 backend는 GPU를 쓰지
않는다(전용 CUDA sidecar만 GPU 사용). cross-model 자동 fallback은 만들지
않는다 — Ovis가 죽었을 때 Unlimited를 자동 로드하면 값비싼 이중 로드·VRAM
경쟁이 생기므로, 사용자가 profile을 바꿔 명시적으로 전환한다.

## 1. 현재 구조 (baseline, 2026-07-20 확인)

- baseline 테스트: backend `uv run pytest` **328 passed, 1 skipped** ·
  `ruff check` 통과 · `npm test --prefix frontend` 통과 · `docker compose config` 통과.
- 워킹 트리 클린 (보존할 미커밋 변경 없음).

### 1.1 엔진 로딩 흐름

```
Settings.from_env() → build_engine(settings)   # app/engine/registry.py
  OCR_ENGINE=fake      → FakeEngine
  OCR_ENGINE=unlimited → UnlimitedEngine (torch 지연 임포트, 벤더링 코드)
main.create_app() → preload 스레드가 engine.load() (실패는 health.model_load_error)
Worker(단일 스레드) → execute_job() → engine.run_multi()/run_single()
```

### 1.2 runner/merger 파일 계약 (신규 엔진이 지켜야 하는 것)

- 청크 작업 디렉터리 (`work/chunk_XX/`):
  - multi: `images/page_{local}_{k}.jpg`, `result_with_boxes_{local}.jpg`,
    `boxes.json`, `raw_pages.json`, 반환 마크다운은 `<PAGE>` 마커로 구분
  - single: `images/{k}.jpg`, `result_with_boxes.jpg`, `boxes.json`, `raw_pages.json`
- `boxes.json`: `{crop파일명: {x1,y1,x2,y2(픽셀), image_width, image_height}}`
- `raw_pages.json`: `{"pages": [그라운딩 태그 원문 문자열, …]}` —
  `pipeline/layout.py::parse_page_blocks`가 `<|det|>label [x1,y1,x2,y2]<|/det|>텍스트`
  (0–999 정규화)와 `<|ref|>label<|/ref|><|det|>[[…]]<|/det|>` 두 문법을 파싱.
  image 라벨의 crop_index는 (ref류 전체 → det류) 순서로 증가하며 저장된 crop
  파일 순서와 일치해야 한다.
- 최종 산출물(merge가 생성): `result.md`, `images/p{글로벌:04d}_{k}.jpg`,
  `images/boxes.json`, `layout/page_{글로벌:04d}.jpg`, `layout.json`, `archive.zip`.

### 1.3 프론트엔드 스트리밍 계약

- SSE `token` 이벤트의 텍스트 델타. 각 청크 스트림은 `<PAGE>`로 시작하고
  이후 `<PAGE>`마다 페이지 +1 (첫 마커는 no-op 재확인).
- `<|det|>label [x1,y1,x2,y2]<|/det|>` 좌표로 라이브 박스 오버레이.
- 신규 페이지 단위 엔진도 이 문법을 그대로 쓰되, **페이지 완료 시점에 한 번에**
  텍스트를 발행한다(가짜 토큰 스트리밍 금지). `<PAGE>` 진행 계약은 유지되므로
  진행률·미리보기·오버레이가 그대로 동작한다.

## 2. 신규 아키텍처

```
                    ┌────────────────────────────────────────────┐
   브라우저 ──────► │ main backend (GPU 미사용, 기존 이미지 재사용) │
                    │  OCR_ENGINE=ovisocr2|paddleocr_vl           │
                    │  SidecarEngine ── SidecarClient(HTTP)       │
                    └───────────────┬────────────────────────────┘
                                    │ POST /v1/parse (페이지 이미지 1장)
                                    ▼
                    ┌────────────────────────────────────────────┐
                    │ CUDA sidecar (GPU 독점, services/…)          │
                    │  ovisocr2: vLLM 직접 로드                    │
                    │  paddleocr_vl: 공식 Blackwell 경로           │
                    └────────────────────────────────────────────┘

sidecar 응답(NormalizedPageResult: markdown + [[FIGURE:n]] + blocks)
        ↓
공통 Artifact Materializer (backend/app/sidecar/materializer.py)
  - 원본 렌더 페이지에서 figure bbox crop → images/{…}.jpg
  - boxes.json / raw_pages.json(그라운딩 문법 합성) / result_with_boxes 오버레이
  - [[FIGURE:n]] → ![](images/…) 치환
        ↓
기존 엔진 산출물 규약 (§1.2) → 기존 IncrementalMerger 무수정 재사용
```

핵심 결정:

1. **merger는 수정하지 않는다.** materializer가 기존 청크 계약을 100% 재현하므로
   Unlimited 경로 회귀 위험이 0에 수렴한다.
2. **raw_pages.json은 normalized block에서 합성한다** (계획서 Phase 4의 옵션 1).
   layout.py의 두 문법 중 inline det(`<|det|>label [x…]<|/det|>내용`)만 사용해
   문서 순서 = crop_index 순서를 보장한다. 합성 전에 내용의 `<|…|>` 특수 토큰
   패턴을 제거해 grammar 오염을 차단한다. Unlimited의 진짜 raw 출력 경로는 불변.
3. **sidecar 응답에는 이미지 바이너리·파일 경로가 없다.** bbox와 텍스트만 받고
   crop은 메인 backend가 원본 페이지 PNG에서 직접 수행한다(경로 신뢰 문제 원천 차단).
4. **backend Python 환경 불변.** 신규 의존성은 sidecar 컨테이너에만 존재.
   backend가 추가로 쓰는 것은 이미 의존성인 `requests`·`pydantic`(FastAPI 동봉)뿐.
5. **cross-model fallback 없음.** provider 실패는 명확한 오류로 표면화한다.

## 3. 모델 선정

### 3.1 확정 엔진

| | OvisOCR2 | PaddleOCR-VL-1.6 | Unlimited-OCR (유지) |
|---|---|---|---|
| 역할 | 페이지 정밀 파싱·figure | 한국어·표·수식·레이아웃 | 장문 멀티페이지 문맥·토큰 스트리밍 |
| 크기 | ~0.8B급 페이지 파서 | 경량 VL 파이프라인 | 3.3B MoE |
| 출력 | MD+표HTML+LaTeX+figure bbox | MD+구조화 블록(JSON) | MD+그라운딩 토큰 |
| 16GB 적합성 | 여유 큼 (util 0.80) | 여유 큼 (layout CPU 옵션) | 검증됨 (기존) |
| 실행 | vLLM sidecar | 공식 Blackwell 경로 sidecar | 기존 in-process |

정확한 revision·runtime 버전 고정값은 §8(고정값 표)에 기록한다 — 공식 모델
카드·공식 Blackwell 가이드를 재확인해 채운다(추측 금지).

### 3.2 조사만 수행한 후보 (구현 제외)

비교표는 [OCR_BENCHMARK.md](OCR_BENCHMARK.md) §후보 조사에 기록.
결론 요약: GLM-OCR·Qianfan-OCR·DeepSeek-OCR-2·dots.mocr 모두 이번 대상 환경에서
OvisOCR2/PaddleOCR-VL 대비 명백한 이점이 확인되지 않거나(한국어 근거 부족,
16GB 헤드룸 부족, Blackwell 공식 지원 불명) 통합 비용이 커서 제외.

## 4. 구현 단계 (실제 커밋 단위)

- **P1 capability**: `OCREngine`에 `EngineCapabilities`(model_id/revision/provider/
  supports_multi_page/preferred_chunk_size/stream_granularity/layout_capability/
  figure_capability) 추가 — 기본값은 기존 Unlimited 의미를 보존해 fake/기존 테스트
  무수정 통과. runner가 capability로 실제 chunk size 결정(페이지 단위 엔진은
  multi 모드에서도 chunk=1, 경고 1회 기록).
- **P2 job metadata + health**: Job에 engine/model_id/model_revision/provider
  저장·복원(기존 meta.json은 필드 부재 시 None 안전 복원), 목록·상세 API 노출,
  archive에 meta 포함. `/api/health`에 model_revision/provider/capabilities/
  provider_health 추가(기존 필드 불변). sidecar 다운 시에도 health는 200 —
  provider_health.status로 구분.
- **P3 normalized**: `backend/app/sidecar/protocol.py` — pydantic 모델
  (NormalizedBlock/NormalizedPageResult/ParseResponse/HealthResponse) + bbox 검증
  (0–999, x1<x2, y1<y2, 경미한 초과만 clamp, 심각 이상은 폐기+warning, NaN/inf/
  문자열/거대값 거부, 순서 보존, block type 정규화).
- **P4 materializer**: `backend/app/sidecar/materializer.py` — §2의 합성 규약.
  crop 최소 크기·out-of-bounds 방지·원자적 기록.
- **P5-6 client**: `backend/app/sidecar/client.py` — requests 기반 동기 클라이언트.
  connect/read/health timeout, 응답 크기 상한(스트리밍 계수), 1회 retry,
  취소 관측 시 호출자 즉시 해제(별도 스레드 + 0.2초 폴링 — 실제 소켓 절단은
  추론 중에는 불가, 계약 문서 §취소 참조), HTTP/모델/프로토콜 오류 구분,
  로그에 문서 내용·이미지 미기록.
- **P7 SidecarEngine**: `backend/app/engine/sidecar.py` — OCREngine 구현.
  load()=health 확인, run_multi/run_single → parse → materialize → sink에
  페이지 단위 발행(`<PAGE>` 계약 유지).
- **P8 ovisocr2 sidecar**: `services/ovisocr2/` — vLLM 직접 로드(OpenAI 서버
  이중 구조 회피), strict figure `<img src="images/bbox_l_t_r_b.jpg" />` 파서,
  반복 suffix 정리, OOM 시 max_pixels 강등 1회 후 실패, FastAPI /health·/v1/parse.
- **P9 paddleocr_vl sidecar**: `services/paddleocr_vl/` — 공식 파이프라인 결과
  schema→protocol 어댑터(fixture 보존), 한국어 보존 테스트, layout device 선택.
- **P10 registry/compose/env**: OCR_ENGINE 4종, profile `ovis`/`paddle`,
  기존 `ocr-cuda` 불변, sidecar는 내부 expose만, backend 포트 8002/8003 루프백.
- **P11 UI**: health capabilities 기반 배지(스트리밍 단위·provider), 잡 모델
  메타 표시, figure_only 표시. Vanilla JS 유지, 외부 의존성 0.
- **P12-14 테스트·보안**: 악성 bbox/경로/response bomb 테스트, stub HTTP 서버로
  client 테스트, sidecar parser fixture 테스트, frontend 테스트 확장.
- **P15-16 스크립트**: check_cuda_environment / smoke 2종 / benchmark.

## 5. 단일 GPU 리소스 정책

- sidecar만 `gpus` 예약, backend(ocr-ovis/ocr-paddle)는 GPU 미예약.
- `tensor_parallel_size=1`, `max_num_seqs=1`, page concurrency 기본 1.
- Ovis vLLM `gpu_memory_utilization=0.80` (WSL2/데스크톱 디스플레이 오버헤드 감안,
  0.95+ 금지), **`max_model_len=24576`**(초안의 16384에서 상향 — max_pixels 2880²의
  비전 토큰 ~8K + 출력 8192 + 프롬프트가 16384에 들어가지 않는다), 출력 토큰 8192.
- Paddle: BF16, 파이프라인 전체 디바이스는 `PADDLEOCR_DEVICE` 하나로 정한다.
  **컴포넌트별 디바이스 분리(layout=CPU, VL=GPU)는 공식 in-process 파이프라인이
  지원하지 않아 구현하지 않았다** — 따라서 `PADDLEOCR_LAYOUT_DEVICE` 같은 변수는
  존재하지 않는다(초안 철회, 근거: PADDLEOCR_VL_BLACKWELL_5070TI.md §디바이스 정책).
- OOM 완화 순서: concurrency 1 확인 → max_pixels 감소 → output token 감소 →
  max_model_len 감소 → gpu_memory_utilization 조정 → (Paddle은) 전체 CPU 전환 →
  소형 엔진. 엔진별 정확한 순서는 각 엔진 문서의 §OOM 완화 순서가 SSOT다.
  CPU offload는 기본 비활성.
- 동시 profile 기동은 VRAM 경쟁을 일으키므로 문서로 금지하고, 포트(8001/8002/8003)를
  분리해 실수로 겹쳐도 즉시 드러나게 한다.

## 6. 오류·취소 정책 (sidecar 엔진)

- provider 연결 실패(`SidecarUnavailableError`) vs 모델/프로토콜 오류
  (`SidecarProtocolError`) vs 타임아웃을 구분해 잡 warning/error 메시지에 반영.
- 재시도는 runner의 기존 1회 재시도에 위임 — client 자체 retry는 연결 수립
  실패에만 1회. 무한 재시도 없음.
- 취소: 요청 전·후 + 대기 중 0.2초 주기 확인 → 관측 즉시 호출자 해제.
  **이미 전송된 요청은 sidecar에서 완주한다**(추론 중에는 응답 헤더가 없어 소켓
  절단 수단이 없음 — 실측 확인). 취소 이후 도착한 결과는 병합하지 않는다.
  자세한 계약은 [OCR_ENGINE_PROTOCOL.md](OCR_ENGINE_PROTOCOL.md) §취소 참조.
- 페이지 단위 격리: 한 페이지 실패는 기존 runner의 placeholder/내장 텍스트
  fallback 경로를 그대로 탄다.

## 7. 테스트 전략

- 모델/GPU 없이: 프로토콜·파서·materializer·client(stub HTTP 서버)·registry·
  job metadata·보안(경로 탈출/거대 응답/악성 bbox) 전부 fixture 기반 유닛 테스트.
- sidecar 파서는 모델 로드 없이 임포트 가능하게 분리(`parser.py`/`adapter.py`).
- 실 GPU 검증은 scripts/smoke_*_5070ti.py — 이 세션에서는 GPU가 없으므로
  "구현·fixture·compose config 검증 완료, 실 runtime 검증 필요"로 명시한다.

## 8. 모델·런타임 고정값 (2026-07-20 공식 소스 재확인 완료)

| 항목 | OvisOCR2 | PaddleOCR-VL-1.6 |
|---|---|---|
| 모델 ID | `ATH-MaaS/OvisOCR2` | `PaddlePaddle/PaddleOCR-VL-1.6` |
| revision | `65c619d374b55d4152e85150fc1b003700bc1f0c` | `66317acc4c9fc17bd154591ce650735cd2855f3e` |
| runtime | vLLM **0.22.1** (모델 카드 공식 권장) | paddlepaddle-gpu **3.3.1**(cu129) + paddleocr **3.6.0** |
| Docker base | `vllm/vllm-openai:v0.22.1-cu129` | `python:3.12-slim-bookworm` + 고정 wheel (공식 Blackwell wheel 경로) |
| dtype | bfloat16 | bfloat16 |
| 라이선스 | Apache-2.0 | Apache-2.0 |
| 핵심 특이사항 | sm_120은 `gdn_prefill_backend=triton` 필수 | 공식 검증 GPU는 RTX 5070 — 5070 Ti는 smoke 필수 |

상세 근거: [OVISOCR2_CUDA_5070TI.md](OVISOCR2_CUDA_5070TI.md) ·
[PADDLEOCR_VL_BLACKWELL_5070TI.md](PADDLEOCR_VL_BLACKWELL_5070TI.md) ·
후보 비교 [OCR_BENCHMARK.md](OCR_BENCHMARK.md)

## 9. 위험 요소와 rollback

| 위험 | 대응 |
|---|---|
| vLLM Blackwell 미지원 조합 | 공식 cu128+ 이미지 태그 고정, preflight 스크립트가 sm_120 확인 |
| Paddle 공식 schema 변동 | 공식 fixture를 저장소에 보존, adapter가 필수 필드 결측 시 명확한 오류 |
| sidecar 다운 | health에 provider 상태 분리 노출, 잡은 명확한 provider 오류 |
| 기존 경로 회귀 | merger·unlimited.py 무수정, 전체 기존 테스트 유지 |
| rollback | 신규 코드는 전부 추가 파일 + 소규모 훅 — `OCR_ENGINE=unlimited`로 즉시 복귀 |

## 10. 실제 5070 Ti 검증 절차 (사용자 실행)

```bash
python scripts/check_cuda_environment.py            # 드라이버/sm_120/BF16/도커 GPU
docker compose --profile ovis up -d --build ovisocr2 ocr-ovis   # Ovis 스택(서비스명 명시)
python scripts/smoke_ovisocr2_5070ti.py              # health→1페이지→PDF→VRAM
docker compose stop ovisocr2 ocr-ovis
docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle  # Paddle 스택
python scripts/smoke_paddleocr_vl_5070ti.py
docker compose stop paddleocr-vl ocr-paddle
python scripts/benchmark_ocr_engines.py --endpoint … # 순차 비교 (동시 기동 금지)
```
