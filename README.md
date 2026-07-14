# Unlimited-OCR — PDF → Markdown

[![CI](https://github.com/Chedrian07/PDF_MARKDOWN_Unlimited-OCR/actions/workflows/ci.yml/badge.svg)](https://github.com/Chedrian07/PDF_MARKDOWN_Unlimited-OCR/actions/workflows/ci.yml) — push/PR마다 backend pytest·ruff·frontend·native 패리티 테스트를 실행합니다.

웹에서 PDF를 업로드하면 [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
(3.3B MoE 비전-언어 모델, MIT)로 **이미지(figure)까지 추출된 Markdown**을 만들어 주는
셀프호스팅 서비스입니다.

- **PDF 속 이미지 완벽 처리**: 모델의 그라운딩 박스(`<|ref|>image<|/ref|><|det|>…`)로
  figure를 원본에서 크롭해 `images/`에 저장하고 마크다운에 `![](images/…)`로 연결
- **GIF 스타일 3-패널 라이브 뷰**: 변환 중 ① 원본 페이지 위 실시간 레이아웃
  박스 오버레이(그라운딩 좌표) ② RAW OUTPUT 토큰 스트림 ③ 실시간 렌더 미리보기가
  동시에 흐르고, STOP 버튼으로 중단해도 부분 결과가 보존됨
  (공식 데모 GIF의 long-horizon 파싱 경험 재현)
- **디바이스 백엔드**: CPU / CUDA / Metal(Apple Silicon, torch MPS) 지원
- **한국어 번역 (선택)**: 변환 결과를 OpenAI 호환 API로 한국어 번역 — §한국어 번역 참조
- **원커맨드 배포**: `docker compose up` 하나로 끝 (Metal은 로컬 실행 — 아래 참조)

## 빠른 시작 (Docker)

```bash
# CPU (기본 서비스 — 클론 직후 .env 없이 이 한 줄로 기동)
docker compose up -d --build
# → http://localhost:8000

# CUDA (NVIDIA GPU + nvidia container toolkit 필요)
docker compose up -d --build ocr-cuda   # 서비스명 지정 → cuda 프로필 자동 활성화
# → http://localhost:8001
```

- 최초 실행 시 모델 가중치(~6.7GB)를 `hf-cache` 볼륨에 1회 다운로드합니다
  (진행 상황: `docker compose logs -f`). CPU/CUDA 서비스가 캐시를 공유합니다.
- 모델 로딩 여부는 헤더 배지 또는 `GET /api/health`의 `model_loaded`로 확인.
- 한국어 번역 기능을 쓰려면 `cp .env.example .env` 후 API 키를 설정 — 아래 §한국어 번역 참조.

### 모델 없이 UI/파이프라인만 체험

```bash
OCR_ENGINE=fake docker compose up -d --build
```

## 보안

이 서비스는 **인증이 없습니다** — 접근 가능한 사람은 누구나 문서 열람·삭제·변환·(설정 시)
유료 번역 트리거가 가능합니다. 그래서 compose는 기본적으로 **루프백(127.0.0.1)에만**
포트를 바인딩하고, `ALLOWED_HOSTS`(기본 `localhost,127.0.0.1`) 밖의 Host 헤더는
400으로 거부합니다(DNS rebinding 방어).

LAN이나 공개 네트워크에 노출하려면 **반드시 인증을 제공하는 리버스 프록시**
(예: nginx + basic auth, Tailscale/VPN) 뒤에 두세요. 그 후:

1. `docker-compose.yml`의 ports를 `"8000:8000"`으로 변경 (또는 프록시만 컨테이너에 접근)
2. 접속에 쓸 호스트명/IP를 `.env`의 `ALLOWED_HOSTS`에 추가 — 예:
   `ALLOWED_HOSTS=localhost,127.0.0.1,ocr.example.com` (포트는 비교 시 무시됨).
   compose가 컨테이너로 전달합니다.

## 한국어 번역

변환이 끝난 문서(`result.md` + 레이아웃)를 OpenAI 호환 API로 한국어 번역해
번역본 미리보기/레이아웃/다운로드를 제공합니다 (수식·이미지·표는 마스킹으로 보존).

```bash
cp .env.example .env   # 키 설정 후 docker compose up -d 로 재기동
```

`.env`에 아래 값을 설정하면 활성화됩니다:

- `OPENAI_BASE_URL` — 버전 경로 포함 (예: `https://api.openai.com/v1`, `http://localhost:11434/v1`)
- `OPENAI_API_KEY` — 로컬 서버는 생략 가능
- `OPENAI_MODEL` — 번역에 쓸 모델 ID

미설정이어도 나머지 기능은 그대로 동작합니다 — 번역 요청 시에만 503과 함께
"번역 프로바이더가 설정되지 않았습니다" 안내가 표시됩니다.
동시성/재시도/reasoning 등 세부 옵션과 파이프라인 설계는
[docs/ARCHITECTURE.md §13](docs/ARCHITECTURE.md#13-한국어-번역-translation) 참조.

## macOS에서 Metal(MPS)로 실행

Docker(맥의 Linux VM)에서는 GPU 패스스루가 없어 Metal을 쓸 수 없습니다.
Apple Silicon Mac에서는 로컬(uv)로 실행하세요:

```bash
cd backend
uv sync --extra metal        # macOS arm64 torch 휠 (MPS 내장)
uv pip install ../native     # 선택 — C++ 가속
OCR_DEVICE=metal uv run uvicorn app.main:app   # http://localhost:8000
```

- dtype은 `auto`면 bfloat16(macOS 14+), 미지원 조합이면 float32로 자동 폴백
- 첫 청크는 Metal 셰이더 컴파일 때문에 이후 청크보다 느릴 수 있습니다
- 청크가 끝날 때마다 `torch.mps.empty_cache()`로 유니파이드 메모리를 반환합니다

## E2E 스모크 테스트

```bash
cd backend && uv run python ../scripts/make_sample_pdf.py ../sample/sample.pdf && cd ..
./scripts/smoke_e2e.sh                      # CPU (8000)
./scripts/smoke_e2e.sh http://localhost:8001  # CUDA (8001)
```

## 로컬 개발 (uv)

```bash
# 백엔드 (Python 3.12, torch CPU)
cd backend
uv sync --extra cpu          # CUDA: --extra cu129 · macOS Metal: --extra metal
uv pip install ../native     # C++ 가속 모듈 (선택 — 없어도 동작)
uv run pytest                # 유닛/통합 테스트 (FakeEngine, 모델 불필요)
uv run uvicorn app.main:app --reload   # http://localhost:8000

# 네이티브 모듈 단독 테스트
cd native && uv venv --python 3.12 .venv \
  && uv pip install -p .venv/bin/python -e . pytest numpy \
  && .venv/bin/python -m pytest tests/ -v

# 프론트엔드 테스트 (Node 22 필요 — 의존성 설치 불필요, 리포 루트에서 실행)
npm test --prefix frontend
```

환경변수 전체 목록: [docs/ARCHITECTURE.md §7](docs/ARCHITECTURE.md) —
`OCR_DEVICE`(cpu/cuda/metal), `OCR_DTYPE`, `OCR_ENGINE`(unlimited/fake),
`PAGES_PER_CHUNK`, `RENDER_DPI`, `MAX_UPLOAD_MB` 등.

## 동작 방식

```
PDF 업로드 → pymupdf로 페이지 PNG 렌더(기본 200dpi)
          → infer_multi()가 8페이지 청크 단위 one-shot 파싱 (<PAGE> 마커로 페이지 구분)
          → figure 크롭(images/) · 레이아웃 오버레이(layout/) · 참조 재작성
          → 페이지 병합 result.md → 미리보기/다운로드(.md, .zip)
```

- 모델 코드는 `backend/app/vendor/unlimited_ocr/`에 **벤더링**되어 있습니다
  (revision 고정, `trust_remote_code` 불필요). 업스트림은 CUDA 전용이라
  CPU 지원 패치 + `eval()` 보안 패치를 적용했습니다 — 내역:
  [PROVENANCE.md](backend/app/vendor/unlimited_ocr/PROVENANCE.md)
- `per_page` 모드(요청 옵션)는 페이지별 gundam 프리셋(1024/640/crop)으로 처리합니다.
- **수식 렌더링**: 모델의 `\(…\)`/`\[…\]` LaTeX를 렌더 레이어에서 정규화해
  (mdit-py-plugins dollarmath) 로컬 벤더링된 **KaTeX**(`frontend/vendor/katex/`,
  외부 CDN 없음)로 타이포셋합니다. 다운로드되는 `result.md`에는 원본 LaTeX가
  그대로 유지됩니다.
- **렌더 충실도**: figure는 그라운딩 bbox로 계산한 **원본 페이지 대비 상대
  폭**으로 표시되고(좁으면 센터링), 최종 미리보기는 페이지별
  `<section class="doc-page">`로 구분됩니다. 결과 탭의 **레이아웃** 뷰는
  전 블록의 좌표로 다단 배치까지 근사 재구성합니다 (best-effort — 텍스트
  리플로우/검색은 마크다운 뷰 담당). 이 모든 변형은 렌더 레이어 전용이며
  `result.md`는 순수 마크다운으로 유지됩니다.
- C++ 모듈(`native/`)은 토큰 생성 핫패스(no-repeat-ngram)를 가속합니다.
  없으면 순수 파이썬 폴백으로 동일하게 동작합니다.
- no-repeat-ngram 검사는 디바이스별 최적 경로를 탑니다: CUDA/MPS는 **GPU 상주
  torch 구현**(토큰마다 발생하던 시퀀스 D2H 복사·동기화 제거), CPU는 마지막
  window 토큰만 슬라이스해 C++/파이썬으로 스캔 — 세 구현 모두 레퍼런스와
  패리티 테스트로 검증됩니다. 참고: batch=1 자기회귀 디코드 특성상 GPU
  사용률은 원래 낮습니다(HF generate 루프가 지배) — 대량 처리 스루풋이
  필요하면 모델이 공식 지원하는 vLLM/SGLang 서빙을 고려하세요.
- CPU 스레드 수는 `OCR_CPU_THREADS`, CUDA GPU 선택은 `GPU_DEVICE`(compose)로
  제어합니다.

## 디바이스 백엔드 현황

| 백엔드 | 상태 | 비고 |
|---|---|---|
| CPU | ✅ | 기본 float32 (`OCR_DTYPE=bfloat16` 가능) |
| CUDA | ✅ | bf16, torch 2.10 cu129 (sm_89·sm_120 확인) |
| Metal | ✅ | torch MPS, `OCR_DEVICE=metal`(별칭 `mps`) — bf16, Apple Silicon 로컬 실행 전용 (Docker 불가) |

## 문서

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 아키텍처, REST/SSE API 계약, 설계 결정
- [backend/app/vendor/unlimited_ocr/PROVENANCE.md](backend/app/vendor/unlimited_ocr/PROVENANCE.md) — 벤더링/패치 내역

## 라이선스

프로젝트 코드는 [MIT](LICENSE)입니다. 벤더링된 코드는 각자의 라이선스를 따릅니다:

- 모델 가중치·벤더링된 모델 코드 — Baidu MIT
  ([backend/app/vendor/unlimited_ocr/LICENSE](backend/app/vendor/unlimited_ocr/LICENSE))
- KaTeX — MIT ([frontend/vendor/katex/LICENSE](frontend/vendor/katex/LICENSE))
