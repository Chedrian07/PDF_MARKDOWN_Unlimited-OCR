# Unlimited-OCR — PDF → Markdown

웹에서 PDF를 업로드하면 [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
(3.3B MoE 비전-언어 모델, MIT)로 **이미지(figure)까지 추출된 Markdown**을 만들어 주는
셀프호스팅 서비스입니다.

- **PDF 속 이미지 완벽 처리**: 모델의 그라운딩 박스(`<|ref|>image<|/ref|><|det|>…`)로
  figure를 원본에서 크롭해 `images/`에 저장하고 마크다운에 `![](images/…)`로 연결
- **GIF 스타일 3-패널 라이브 뷰**: 변환 중 ① 원본 페이지 위 실시간 레이아웃
  박스 오버레이(그라운딩 좌표) ② RAW OUTPUT 토큰 스트림 ③ 실시간 렌더 미리보기가
  동시에 흐르고, STOP 버튼으로 중단해도 부분 결과가 보존됨
  (공식 데모 GIF의 long-horizon 파싱 경험 재현)
- **디바이스 백엔드**: CPU / CUDA 지원, Metal은 로드맵 (아래 참조)
- **원커맨드 배포**: `docker compose up` 하나로 끝

## 빠른 시작 (Docker)

```bash
# CPU (기본 프로필)
docker compose up -d --build
# → http://localhost:8000

# CUDA (NVIDIA GPU + nvidia container toolkit 필요)
docker compose --profile cuda up -d --build ocr-cuda
# → http://localhost:8001
```

- 최초 실행 시 모델 가중치(~6.7GB)를 `hf-cache` 볼륨에 1회 다운로드합니다
  (진행 상황: `docker compose logs -f`). CPU/CUDA 서비스가 캐시를 공유합니다.
- 모델 로딩 여부는 헤더 배지 또는 `GET /api/health`의 `model_loaded`로 확인.

### 모델 없이 UI/파이프라인만 체험

```bash
OCR_ENGINE=fake docker compose up -d --build
```

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
uv sync --extra cpu          # CUDA 개발: --extra cu129
uv pip install ../native     # C++ 가속 모듈 (선택 — 없어도 동작)
uv run pytest                # 유닛/통합 테스트 (FakeEngine, 모델 불필요)
uv run uvicorn app.main:app --reload   # http://localhost:8000

# 네이티브 모듈 단독 테스트
cd native && uv venv --python 3.12 .venv \
  && uv pip install -p .venv/bin/python -e . pytest numpy \
  && .venv/bin/python -m pytest tests/ -v
```

환경변수 전체 목록: [docs/ARCHITECTURE.md §7](docs/ARCHITECTURE.md) —
`OCR_DEVICE`(cpu/cuda), `OCR_DTYPE`, `OCR_ENGINE`(unlimited/fake),
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
- C++ 모듈(`native/`)은 토큰 생성 핫패스(no-repeat-ngram)를 가속합니다.
  없으면 순수 파이썬 폴백으로 동일하게 동작합니다.

## 디바이스 백엔드 현황

| 백엔드 | 상태 | 비고 |
|---|---|---|
| CPU | ✅ | 기본 float32 (`OCR_DTYPE=bfloat16` 가능) |
| CUDA | ✅ | bf16, torch 2.10 cu129 (sm_89·sm_120 확인) |
| Metal | 🗺️ 로드맵 | torch MPS로 `engine/registry.py`에 추가 예정 — 구조상 registry 등록 + dtype 분기만 필요 |

## 문서

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 아키텍처, REST/SSE API 계약, 설계 결정
- [backend/app/vendor/unlimited_ocr/PROVENANCE.md](backend/app/vendor/unlimited_ocr/PROVENANCE.md) — 벤더링/패치 내역

## 라이선스 관련

모델 가중치와 벤더링된 모델 코드는 Baidu의 MIT 라이선스를 따릅니다.
