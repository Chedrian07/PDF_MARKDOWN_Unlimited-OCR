# 벤더링 출처 및 패치 내역

- 출처: https://huggingface.co/baidu/Unlimited-OCR
- Revision: `ee63731b6461c8afcdcc7b15352e7d2ffecc2ead` (2026-07-03)
- 가져온 날짜: 2026-07-06
- 라이선스: MIT (동봉된 LICENSE)
- 파일: `modeling_unlimitedocr.py`, `modeling_deepseekv2.py`,
  `configuration_deepseek_v2.py`, `deepencoder.py`, `conversation.py`

업스트림 코드는 CUDA 전용(`.cuda()`/`torch.autocast("cuda")` 하드코딩)이라
CPU 백엔드 지원을 위해 벤더링 후 아래 패치를 적용했다.
**`modeling_unlimitedocr.py`만 수정**했으며 나머지 파일은 원본 그대로다.

## 패치 목록 (modeling_unlimitedocr.py)

| # | 내용 | 이유 |
|---|---|---|
| P1 | `_autocast_ctx(device, dtype)` 헬퍼 추가 | `torch.autocast("cuda", bf16)` 하드코딩을 디바이스/디타입 인식으로 대체 (fp32면 no-op) |
| P2 | `UnlimitedOCRModel.forward`의 `masked_scatter_` 마스크 `.cuda()` → `.to(inputs_embeds.device)` | CPU/기타 디바이스 동작 |
| P3 | `infer()`/`infer_multi()`에 `streamer=None`, `stopping_criteria=None`, `logits_processor=None` 파라미터 추가 (기본값이면 업스트림과 동일 동작) | SSE 토큰 스트리밍, 잡 취소, C++ 가속 ngram 프로세서 주입 |
| P4 | `infer()`/`infer_multi()` 내부의 모든 `.cuda()` → `.to(모델 파라미터 디바이스)`, autocast → P1 헬퍼 | CPU/CUDA 공용 |
| P5 | 이미지 텐서 `.to(torch.bfloat16)` 하드코딩 → 모델 파라미터 dtype 추종 | fp32(CPU 기본) 로딩 시 dtype 불일치 방지 |
| P6 | 디코드 시 `input_ids.unsqueeze(0).cuda().shape[1]` → `input_ids.shape[0]` | 불필요한 디바이스 전송 제거 (동일 값) |
| P7 | `infer()`의 `save_results` 분기 끝에 `return outputs` 추가 | 업스트림은 파일만 쓰고 None 반환 — 호출자가 처리된 마크다운을 받도록 (다른 분기 동작 불변) |
| P8 | `eval()` 기반 geo 플로팅 분기 비활성화 (`if False and ...`) | **보안**: 모델 출력(=문서 내용에 의해 조작 가능)을 `eval()`에 전달 — PDF 내용을 통한 코드 실행 벡터. 본 앱은 geo 모드 미사용 |
| P9 | det 좌표 파싱 `eval()` → `ast.literal_eval()` | **보안**: 동일 벡터. 리터럴 좌표만 허용, 이상 입력은 기존 try/except로 스킵 |
| P10 | `UnlimitedOCRForCausalLM`에 `GenerationMixin` 명시 상속 추가 | transformers 4.50+는 커스텀 모델에 `generate()`를 자동 제공하지 않음. 업스트림은 trust_remote_code 로딩 시 transformers의 하위호환 심이 자동 부착하지만, 벤더링 직접 로드는 심이 없어 명시 상속 필요 |

업스트림 갱신 시: 새 revision을 받아 이 패치들을 재적용하고 이 문서를 갱신할 것.
