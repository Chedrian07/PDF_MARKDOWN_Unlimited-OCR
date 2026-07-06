# Unlimited-OCR PDF → Markdown 변환 서비스 — 아키텍처 & API 계약

> 이 문서는 본 프로젝트의 **단일 진실 공급원(SSOT)** 입니다.
> 백엔드/프론트엔드/네이티브 모듈은 모두 이 문서의 계약을 따릅니다.

## 1. 개요

웹에서 PDF를 업로드하면 [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)
(3.3B MoE VLM, DeepSeek-OCR 계열, MIT)로 파싱하여 **이미지(figure)까지 포함된 Markdown**으로
변환해 주는 셀프호스팅 애플리케이션.

- 디바이스 백엔드: **CPU**, **CUDA** (구현 완료) / **Metal** (로드맵, 스텁만 존재)
- 배포: `docker compose up` 한 번으로 실행 (CPU 기본, `--profile cuda`로 GPU)
- 개발 스택: Python 3.12 (uv 관리) + C++17 (pybind11 네이티브 모듈)

## 2. 모델 사용 방식 (리서치 결과 요약)

| 항목 | 내용 |
|---|---|
| 모델 | `baidu/Unlimited-OCR`, revision `ee63731b6461c8afcdcc7b15352e7d2ffecc2ead` 고정 |
| 로딩 | 벤더링된 모델 코드(`backend/app/vendor/unlimited_ocr/`)의 `UnlimitedOCRForCausalLM.from_pretrained()` — `trust_remote_code` 불필요 |
| 단일 이미지 | `model.infer(tokenizer, prompt='<image>document parsing.', ...)` — gundam(1024/640/crop) 또는 base(1024/1024) |
| PDF/멀티페이지 | `model.infer_multi(tokenizer, prompt='<image>Multi page parsing.', image_files=[...], image_size=1024, max_length=32768, no_repeat_ngram_size=35, ngram_window=1024, save_results=True)` |
| 페이지 구분 | 출력 텍스트에 `<PAGE>` 마커 |
| 이미지(figure) 추출 | 모델이 `<|ref|>image<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` (0–999 정규화 좌표) 출력 → 원본 페이지에서 크롭하여 `{out}/images/page_{i}_{k}.jpg` 저장, 마크다운에는 `![](images/page_{i}_{k}.jpg)` 치환 |
| 레이아웃 시각화 | 페이지별 `result_with_boxes_{i}.jpg` 저장 (GIF 데모의 박스 오버레이) |
| 고정 의존성 | torch==2.10.0, torchvision==0.25.0, transformers==4.57.1, pymupdf==1.27.2.2 등 (모델 README 기준) |
| CUDA 휠 | cu129 (README 테스트 환경 CUDA 12.9, Blackwell sm_120 포함) |
| flash-attn | 선택 사항 (미설치 시 eager attention) — 본 프로젝트는 미사용 |

### 벤더링 패치 (backend/app/vendor/unlimited_ocr/)
업스트림 코드는 `.cuda()` 및 `torch.autocast("cuda")`가 하드코딩되어 CPU에서 동작 불가.
다음 최소 패치를 적용하며, 전체 내역은 `PROVENANCE.md`에 기록:
1. `.cuda()` → 모델 파라미터 디바이스 기준 `.to(dev)`
2. `torch.autocast("cuda", bfloat16)` → 디바이스/디타입 조건부 autocast
3. `masked_scatter_`의 마스크 디바이스 수정 (modeling_unlimitedocr.py:582)
4. 이미지 텐서 `.to(torch.bfloat16)` 하드코딩 → 모델 dtype 추종
5. `infer`/`infer_multi`에 `streamer=None`, `stopping_criteria=None` 파라미터 추가 (SSE 스트리밍/취소용, 기본값이면 업스트림과 동일 동작)

## 3. 디렉터리 구조

```
├── docker-compose.yml          # ocr-cpu(기본), ocr-cuda(프로필)
├── .env                        # COMPOSE_PROFILES=cpu (기본값, 커밋됨)
├── docs/ARCHITECTURE.md        # 이 문서
├── backend/
│   ├── pyproject.toml          # uv 프로젝트, extras: cpu / cu129
│   ├── uv.lock
│   ├── Dockerfile              # ARG TORCH_VARIANT=cpu|cu129
│   ├── app/
│   │   ├── main.py             # FastAPI 앱 팩토리 + 정적 프론트엔드 서빙
│   │   ├── config.py           # 환경변수 설정(Settings)
│   │   ├── api.py              # REST + SSE 라우트
│   │   ├── jobs.py             # JobStore + 단일 워커 큐
│   │   ├── engine/
│   │   │   ├── base.py         # OCREngine 프로토콜, StreamCallback
│   │   │   ├── registry.py     # cpu/cuda 선택, metal → NotImplementedError
│   │   │   ├── unlimited.py    # 실모델 엔진 (벤더링 코드 사용)
│   │   │   └── fake.py         # 테스트/데모용 가짜 엔진 (torch 불필요)
│   │   ├── pipeline/
│   │   │   ├── pdf.py          # PDF → 페이지 PNG (pymupdf)
│   │   │   ├── merge.py        # <PAGE> 분리, figure 글로벌 리넘버링, md 병합
│   │   │   └── render.py       # markdown → HTML (markdown-it-py)
│   │   ├── native_ops.py       # uocr_native 로더 + 순수 파이썬 폴백
│   │   └── vendor/unlimited_ocr/   # 벤더링 모델 코드 (MIT) + PROVENANCE.md
│   └── tests/
├── native/                     # C++ pybind11 모듈 (uocr_native)
│   ├── pyproject.toml          # scikit-build-core
│   ├── CMakeLists.txt
│   ├── src/uocr_native.cpp
│   └── tests/test_parity.py
├── frontend/                   # 정적 SPA (빌드스텝/외부 의존성 0)
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── scripts/
    ├── make_sample_pdf.py      # 텍스트+표+차트이미지 포함 샘플 PDF 생성
    └── smoke_e2e.sh            # compose 기동 → 업로드 → 결과 검증
```

## 4. 처리 파이프라인

```
업로드(PDF) ──► JobStore(queued) ──► 워커(단일 스레드)
  1. render : pymupdf로 페이지별 PNG (RENDER_DPI, 기본 200) → pages/page_%04d.png
  2. ocr    : PAGES_PER_CHUNK(기본 8)개씩 infer_multi() 호출
              - 각 청크는 work/chunk_%02d/ 를 output_path로 사용
              - 커스텀 streamer가 토큰 델타를 SSE 큐로 전달
              - StoppingCriteria로 취소(cancel) 지원
  3. merge  : 청크 산출물 병합
              - <PAGE> 마커 분리 → 페이지 단위 마크다운
              - work/chunk_*/images/page_{i}_{k}.jpg → images/p{글로벌페이지:04d}_{k}.jpg 리네임
              - 마크다운 내 ![](images/page_{i}_{k}.jpg) 참조를 새 경로로 재작성
              - result_with_boxes_{i}.jpg → layout/page_{글로벌:04d}.jpg
              - 페이지 사이 PAGE_SEPARATOR(기본 "\n\n---\n\n")로 join → result.md
  4. done   : meta.json 갱신, SSE done 이벤트
```

- `mode=per_page`일 때는 2단계가 페이지당 `infer()`(gundam) 호출로 대체된다
  (`ngram_window=128`). 이미지 프리픽스는 페이지 디렉터리로 격리 후 동일하게 병합.
- 워커는 프로세스당 1개(모델 메모리 때문). 잡은 FIFO.

### 잡 디렉터리 레이아웃 (`{DATA_DIR}/jobs/{job_id}/`)

```
source.pdf                  # 업로드 원본
meta.json                   # 상태/진행/파라미터 (재시작 시 복원)
pages/page_0001.png ...     # 렌더된 입력 페이지 (1-based)
work/chunk_00/ ...          # 모델 원시 출력 (디버깅용, zip 제외)
result.md                   # 최종 병합 마크다운
images/p0001_0.jpg ...      # figure 크롭 (글로벌 페이지 번호, 1-based)
layout/page_0001.jpg ...    # 레이아웃 박스 오버레이
```

## 5. REST / SSE API 계약 (v1)

모든 경로는 `/api` 프리픽스. 프론트엔드는 같은 오리진에서 서빙되므로 CORS 불필요.

### GET /api/health
```json
{
  "status": "ok",
  "engine": "unlimited",            // 또는 "fake"
  "device": "cuda",                 // cpu | cuda
  "dtype": "bfloat16",
  "model_id": "baidu/Unlimited-OCR",
  "model_loaded": true,             // false면 첫 잡에서 로딩
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",  // cpu면 null
  "native_ops": true                // C++ 모듈 사용 여부
}
```

### POST /api/jobs — PDF 업로드
- `multipart/form-data`: `file`(필수, PDF), `mode`(`multi`|`per_page`, 기본 `multi`),
  `dpi`(72–400, 기본 200), `max_pages`(선택)
- 202 → `{"job_id": "j_1a2b3c4d5e6f", "status": "queued"}`
- 400(비PDF/손상), 413(MAX_UPLOAD_MB 초과)

### GET /api/jobs — 잡 목록 (최신순, 최대 50)
```json
{"jobs": [ { …GET /api/jobs/{id}와 동일 요약… } ]}
```

### GET /api/jobs/{id} — 상태
```json
{
  "job_id": "j_1a2b3c4d5e6f",
  "filename": "sample.pdf",
  "status": "running",              // queued|running|done|error|canceled
  "mode": "multi",
  "created_at": "2026-07-06T10:00:00+00:00",
  "progress": {
    "phase": "ocr",                 // render|ocr|merge
    "current_page": 3,              // 1-based, 처리 중/완료된 페이지
    "total_pages": 12,
    "chunk": 1, "total_chunks": 2
  },
  "error": null,
  "result": {                       // status=done일 때만
    "markdown_url": "/api/jobs/{id}/markdown",
    "html_url": "/api/jobs/{id}/html",
    "archive_url": "/api/jobs/{id}/archive",
    "images": ["/api/jobs/{id}/files/images/p0001_0.jpg"],
    "layouts": ["/api/jobs/{id}/files/layout/page_0001.jpg"],
    "pages": ["/api/jobs/{id}/files/pages/page_0001.png"]
  }
}
```

### GET /api/jobs/{id}/events — SSE
- `Content-Type: text/event-stream`, `retry: 3000`, 15초마다 `: ping` 주석
- 접속 시 현재 상태 스냅샷(progress) 1회 즉시 발행, 종료 잡이면 done/error 즉시 발행
- 이벤트:
  - `event: progress` `data: {"phase":"ocr","current_page":3,"total_pages":12,"chunk":1,"total_chunks":2,"status":"running"}`
    — `current_page`의 의미는 phase에 따라 다름: `render`=래스터화된 페이지 수,
    `ocr`=파싱 중 페이지(청크 시작 시 점프, `<PAGE>` 마커마다 증가), `merge`=총 페이지.
    **레이아웃 박스의 페이지 추적은 반드시 `phase==="ocr"`인 이벤트만 사용할 것**
  - `event: token`    `data: {"text":"델타 텍스트"}`   ← 모델 생성 토큰 실시간 (GIF 스타일)

    **토큰 스트림 문법 (실캡처로 확정, frontend/tests/fixtures/*.sse.txt):**
    각 청크의 스트림은 `<PAGE>` 마커로 **시작**한다 — 마커는 "지금 시작하는 페이지의 선언"이다.
    `청크k 스트림 = <PAGE> + page(start_k) 내용 + <PAGE> + page(start_k+1) 내용 + …`
    청크 시작 직전에 `progress(phase=ocr, current_page=start_k, chunk=k)`가 먼저 발행되므로,
    **각 청크의 첫 마커는 이미 선언된 페이지의 재확인(no-op)** 이고 이후 마커만 +1이다.
    블록 문법: `<|det|>label [x1,y1,x2,y2]<|/det|>텍스트…` (label: title/text/table/equation/
    image/page_number 등, 좌표 0–999 정규화) 또는 `<|ref|>label<|/ref|><|det|>[[…]]<|/det|>`.
    표는 블록 텍스트 안에 HTML `<table>`로 온다.
  - `event: done`     `data: {"markdown_url":"...","archive_url":"..."}`
  - `event: error`    `data: {"message":"..."}`  (취소 시 `"canceled": true` 포함)

### GET /api/jobs/{id}/markdown
- `text/markdown; charset=utf-8`. 실행 중이면 완료된 청크까지의 부분 결과 + `X-Partial: true`

### GET /api/jobs/{id}/html
- 최종(또는 부분) 마크다운을 서버에서 HTML 프래그먼트로 렌더 (markdown-it-py, GFM 테이블 지원)
- `<img src="images/...">` → `src="/api/jobs/{id}/files/images/..."`로 재작성됨

### GET /api/jobs/{id}/files/{path}
- 잡 디렉터리 하위 정적 파일 (pages/, images/, layout/ 만 허용, 경로 탈출 차단)

### GET /api/jobs/{id}/archive
- `result.md` + `images/`를 담은 zip (`{원본이름}.md.zip`). 미완료 시 409

### POST /api/jobs/{id}/cancel
- 실행/대기 중 잡을 **삭제 없이** 중단. 202 `{"job_id","status":"canceling"}`
  (이미 종료된 잡이면 현재 status 반환). 잡은 `canceled` 상태로 남고
  완료된 청크까지의 부분 결과는 /markdown 등에서 계속 접근 가능

### POST /api/jobs/{id}/render-preview
- 요청 본문(text/plain, ≤2MB)의 마크다운을 /html과 동일한 안전 렌더러로
  HTML 프래그먼트 렌더 (라이브 미리보기용 — 프론트가 정리한 스트림 텍스트를 debounce 전송)

### DELETE /api/jobs/{id}
- 실행 중이면 취소(cancel) 후 삭제, 완료면 디렉터리 삭제. 204

## 6. 디바이스 백엔드

| 백엔드 | 상태 | 선택 방법 | 비고 |
|---|---|---|---|
| CPU | ✅ 구현 | `OCR_DEVICE=cpu` | 기본 dtype float32 (`OCR_DTYPE`로 변경 가능) |
| CUDA | ✅ 구현 | `OCR_DEVICE=cuda` | bf16, cu129 휠, sm_89/sm_120 확인 |
| Metal | 🗺️ 로드맵 | `OCR_DEVICE=metal` | 현재 `NotImplementedError` — torch MPS 지원 시 registry에 추가 예정 |

`app/engine/registry.py`가 단일 진입점: 환경 검증(cuda 가용성 등) 후 엔진 생성.

## 7. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OCR_DEVICE` | `cpu` | `cpu`\|`cuda`\|`metal`(스텁) |
| `OCR_DTYPE` | `auto` | `auto`(cuda→bfloat16, cpu→float32)\|`bfloat16`\|`float32` |
| `OCR_ENGINE` | `unlimited` | `unlimited`\|`fake`(모델 없이 파이프라인 데모/테스트) |
| `MODEL_ID` | `baidu/Unlimited-OCR` | HF 모델 ID |
| `MODEL_REVISION` | `ee63731b…` | HF revision 고정 (README의 검증 커밋) |
| `PRELOAD_MODEL` | `1` | 기동 시 모델 로드 (0이면 첫 잡에서 lazy) |
| `DATA_DIR` | `/data` | 잡 저장소 루트 (`{DATA_DIR}/jobs`) |
| `HF_HOME` | `/data/hf` | HF 캐시 (compose 볼륨) |
| `RENDER_DPI` | `200` | 요청별 `dpi`로 오버라이드 가능 |
| `PAGES_PER_CHUNK` | `8` | infer_multi 청크 크기 |
| `MAX_PAGES` | `200` | 페이지 상한 |
| `MAX_UPLOAD_MB` | `100` | 업로드 상한 |
| `MAX_LENGTH` | `32768` | 생성 총 길이 상한 |
| `PAGE_SEPARATOR` | `\n\n---\n\n` | 병합 시 페이지 구분자 |
| `HOST`/`PORT` | `0.0.0.0`/`8000` | 서버 바인드 |

## 8. docker-compose

- `ocr-cpu`: 프로필 없음(기본), 포트 **8000**, `OCR_DEVICE=cpu`
- `ocr-cuda`: 프로필 `cuda`, 포트 **8001**, `OCR_DEVICE=cuda`, `gpus: all`
- 공유 볼륨: `hf-cache`(모델 가중치 ~6.7GB, 최초 1회 다운로드), `ocr-data`(잡 결과)
- `.env`의 `COMPOSE_PROFILES=cpu` 덕분에 `docker compose up` = CPU 서비스만 기동
- GPU: `docker compose --profile cuda up -d ocr-cuda`

## 9. C++ 네이티브 모듈 (`native/`, 모듈명 `uocr_native`)

목적: 토큰 생성 핫패스(no-repeat-ngram 배닝)와 figure 크롭의 C++ 가속.
**없어도 앱은 동작해야 한다** — `app/native_ops.py`가 임포트 실패 시 순수 파이썬 폴백 사용.

### 9.1 `banned_ngram_tokens(sequence, ngram_size, window) -> ndarray[int64]`
- 입력: `sequence` 1-D `int64` C-contiguous ndarray (지금까지 생성된 토큰열),
  `ngram_size >= 1`, `window >= 1`
- 의미론 (아래 파이썬 레퍼런스와 **완전 동일**해야 함, 반환은 오름차순 유니크):

```python
def banned_ngram_tokens_ref(sequence: list[int], ngram_size: int, window: int) -> list[int]:
    if len(sequence) < ngram_size:
        return []
    search_start = max(0, len(sequence) - window)
    search_end = len(sequence) - ngram_size + 1
    if search_end <= search_start:
        return []
    current_prefix = tuple(sequence[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    banned = set()
    for idx in range(search_start, search_end):
        ngram = sequence[idx:idx + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            banned.add(ngram[-1])
    return sorted(banned)
```

### 9.2 `crop_regions(image, boxes) -> list[ndarray | None]`
- 입력: `image` HxWx3 `uint8` C-contiguous, `boxes` Nx4 `int64` (x1,y1,x2,y2 — **0–999 정규화**)
- 각 박스에 대해 `x1p=int(x1/999*W)`, `y1p=int(y1/999*H)` … (파이썬 `int()` 절삭과 동일),
  `x2p=min(x2p,W)`, `y2p=min(y2p,H)`, `x1p=max(x1p,0)`, `y1p=max(y1p,0)`
- `x2p<=x1p or y2p<=y1p`면 해당 항목 `None`, 아니면 `(y2p-y1p, x2p-x1p, 3)` uint8 크롭 반환
- 반환 리스트 길이는 항상 N (박스와 1:1)

### 9.3 빌드/테스트
- scikit-build-core + pybind11 + CMake(C++17, `-O3`), Python 3.12
- `native/tests/test_parity.py`: 랜덤 케이스에서 레퍼런스와 완전 일치 검증 (경계: 빈 시퀀스,
  window > len, ngram_size=1, 좌표 0/999, 퇴화 박스)

## 10. 프론트엔드 (frontend/, 정적 SPA)

- **외부 네트워크 리소스 0** (CDN/폰트/트래커 금지), 빌드 스텝 없음, 바닐라 JS(ES modules)
- 한국어 UI, 다크/라이트 자동(`prefers-color-scheme`) + 수동 토글(localStorage)
- 구성:
  - 헤더: 앱명 "Unlimited-OCR — PDF → Markdown", `/api/health` 기반 디바이스/엔진 배지
  - 좌측: PDF 드롭존(+파일선택, 확장자/크기 검증) · 옵션(mode, dpi) · 잡 히스토리(5초 폴링)
  - 메인(활성 잡, **공식 데모 GIF 재현 3-패널 라이브 뷰**):
    1. 원본+레이아웃 — 현재 페이지 이미지 위에 스트림의 `<|det|>label [x1,y1,x2,y2]<|/det|>`
       (0–999 정규화) 좌표로 컬러 박스를 실시간 오버레이, `<PAGE>` 마커로 페이지 자동 전환
    2. RAW OUTPUT — SSE `token` 델타 모노스페이스 append (자동 스크롤, 청크 경계 holdback)
    3. 실시간 미리보기 — 정리된 스트림 텍스트를 600ms debounce로
       `POST /render-preview`에 보내 렌더된 HTML 표시
    - 실행 중 STOP(정지) 버튼 → `POST /cancel` (부분 결과 보존) · 진행 바(phase + 페이지 n/N)
    - 완료 시 탭 [미리보기(HTML)] [Markdown] [레이아웃] [원본 페이지] · [.md] [.zip] 다운로드 · 삭제
  - 미리보기 탭은 `/api/jobs/{id}/html` 응답을 주입 (클라이언트 md 렌더러 불필요)
  - SSE 불가 환경 폴백: 1초 상태 폴링(+부분 markdown 주기 조회)
- 성능: token append는 rAF 배칭, 히스토리 50개 제한

## 11. 테스트 전략

- `backend/tests/` (FakeEngine, torch 불필요 — CI/로컬 빠른 실행):
  - merge 로직(리넘버링/참조 재작성/`<PAGE>` 분리) 단위 테스트
  - API 플로우: 업로드→상태→SSE→markdown/html/zip→삭제 (httpx + TestClient)
  - pdf.py 렌더 테스트(생성 PDF), render.py img src 재작성
- `native/tests/`: C++ ↔ 파이썬 레퍼런스 패리티
- 실모델 E2E: `scripts/smoke_e2e.sh` — compose 기동 후 샘플 PDF 변환, figure 파일 존재 검증

## 12. 로드맵

1. **Metal 백엔드**: torch `mps` 디바이스 + 벤더 코드의 autocast 분기 확장 (registry에 등록만 하면 되는 구조)
2. vLLM/SGLang 서빙 엔진 옵션 (모델 repo가 공식 지원 — 대량 처리용)
3. 동시 워커 (GPU 멀티 인스턴스 / 페이지 병렬)
