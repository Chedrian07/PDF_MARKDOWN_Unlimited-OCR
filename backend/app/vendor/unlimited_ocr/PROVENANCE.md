# 벤더링 출처 및 패치 내역

- 출처: https://huggingface.co/baidu/Unlimited-OCR
- Revision: `ee63731b6461c8afcdcc7b15352e7d2ffecc2ead` (2026-07-03)
- 가져온 날짜: 2026-07-06
- 라이선스: MIT (동봉된 LICENSE)
- 파일: `modeling_unlimitedocr.py`, `modeling_deepseekv2.py`,
  `configuration_deepseek_v2.py`, `deepencoder.py`, `conversation.py`

업스트림 코드는 CUDA 전용(`.cuda()`/`torch.autocast("cuda")` 하드코딩)이라
CPU 백엔드 지원을 위해 벤더링 후 아래 패치를 적용했다.
**P1–P15는 `modeling_unlimitedocr.py`**, **P16–P20은 `modeling_deepseekv2.py`**를
수정했으며 나머지 파일은 원본 그대로다.

## 패치 목록

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
| P11 | 이미지 임베딩 주입의 `masked_scatter_`(브로드캐스트 (L,1) 마스크) → bool 인덱싱 대입 (`inputs_embeds[idx][mask] = …`) | **MPS 정합성**: torch 2.10.0 MPS에서 브로드캐스트 마스크 `masked_scatter_`가 조용히 오동작(뷰/expand 무관, 소스 첫 원소만 기록)해 이미지 임베딩이 주입되지 않음 → 모델이 즉시 EOS 출력. CPU/CUDA는 결과 동일 (P2의 마스크 디바이스 수정 의도 포함) |
| P12 | `_autocast_ctx`가 `mps`에서는 항상 nullcontext 반환 | **MPS 정합성**: torch 2.10.0의 `torch.autocast("mps", bf16)`가 로짓을 오염시켜 생성이 반복 루프(`"), ), "` 무한 반복)로 퇴화. 가중치를 bf16으로 로딩하므로 autocast 없이도 bf16 연산 — 성능 손실 없음. CPU/CUDA 경로 불변 |
| P13 | `draw_bounding_boxes`가 figure 크롭 bbox(픽셀)+페이지 크기를 `{output_path}/boxes.json`으로 export | 업스트림은 크롭 후 좌표를 버림 — 렌더 레이어가 원본 페이지 대비 상대 폭으로 figure를 표시하는 데 필요 (마크다운 출력 계약 불변, 파일 추가만) |
| P14 | `infer`/`infer_multi`의 save_results가 치환 전 페이지 원문을 `{output_path}/raw_pages.json`으로 export | 좌표 기반 레이아웃 뷰(전 블록 type/bbox/텍스트)가 필요 — 구조 파싱은 앱 코드(pipeline/layout.py)에서 수행해 벤더 변경을 dump 한 줄로 최소화 (마크다운 계약 불변) |
| P15 | `infer`/`infer_multi`에 `generate_fn=None` 파라미터 추가 — 있으면 `self.generate` 대신 호출 | 커스텀 디코드 루프(app/engine/fast_decode.py — 호스트 동기화 블록 배칭, cpu/cuda/mps 공용) 주입용. 기본값 None이면 업스트림과 동일 동작. HF generate의 토큰당 동기화·기계장치가 측정된 디코드 병목이었음 |
| P16 | (`modeling_deepseekv2.py`) `SlidingWindowLlamaAttention`의 prefill·디코드 어텐션을 SDPA 융합 커널로 — **CUDA 전용 기본**, `OCR_SDPA=1/0`으로 전 디바이스 강제 on/off | eager(matmul→fp32 softmax→matmul)의 어텐션 행렬 물질화·다중 디스패치 제거 (CUDA 골든 파리티 검증). **MPS 실측(M1 Max, torch 2.10.0): SDPA fused 커널이 로짓을 오염시켜 디코드가 반복 붕괴**(P12 autocast 오염과 같은 계열) → MPS/CPU는 원본 eager(fp32 softmax) 유지 |
| P17 | (`modeling_deepseekv2.py`) `DeepseekV2MoE`에 융합 소배치 추론 경로(`_moe_infer_fused`) 추가 — 첫 호출 시 전 expert의 gate/up/down weight를 `[E, out, in]` 텐서로 `torch.stack`하고 각 expert Linear의 `.weight`를 그 스택의 뷰로 재지정(개별 텐서 참조 해제 → VRAM 중복 없음, 기존 `moe_infer` 루프도 뷰로 계속 동작). 디코드에선 `index_select`+`bmm` 3방으로 라우팅·MLP를 융합. **legacy 바이트 파리티는 불가**(가중치 재스택만으로 cuBLAS 커널 선택이 바뀜 — 동일 커널 mm 변형 실측으로 확인) → 채택 게이트: 동일 프로세스 내 결정성(같은 컨테이너 2회 변환 바이트 동일 실측 — 프로세스 재시작 간에는 cuBLASLt 알고리즘 휴리스틱으로 공백/동점 토큰 수준 변동이 legacy 포함 원래 존재)·구조 지표 등가(표/셀/이미지/수식 수 재빌드 간 동일 실측)·`OCR_MOE_FUSED=0` 완전 legacy 복원. **CUDA·디코드(`N==1`)·`ep_size==1` 한정 기본 on** (N>1 프리필 조각은 mm↔bmm 누적차로 근접 argmax 플립 실측 → 제외), `OCR_MOE_FUSED=0/false/no/off`로 킬스위치(env는 forward 1회 조회·캐시) | eval 디코드 병목이 `moe_infer`의 `tokens_per_expert.cpu().numpy()` — MoE 레이어(~27개)마다 GPU→CPU 동기화(WSL2 왕복 큼) + expert별 파이썬 루프의 소형 커널 난사(토큰 1개에 6 expert × 3 linear). 융합 경로는 **CPU 동기화 0회**. 실측 배경: RTX 5070 Ti batch-1 디코드 ~13 tok/s·sm 21%. dtype 캐스트/최종 가중합 순서를 upstream과 동일하게 맞춰 수치 정합 — N=1(디코드) bitwise 동일, N>1은 mm↔bmm 누적 순서차 fp32 round-off(~2e-7)뿐. 발동 조건 밖·`ep_size>1`은 기존 `moe_infer` 그대로 |
| P18 | (`modeling_deepseekv2.py`) `DeepseekV2MoE.moe_infer`에 단일 토큰(seq==1)·`ep_size==1` 디코드 패스트패스 추가 — **MPS 전용 기본**, `OCR_MOE_FAST=1/0`으로 전 디바이스 강제 on/off (P16 게이트 패턴). P17(CUDA 융합)과 상보 — P17이 발동하지 않는 디바이스에서 `moe_infer` 안에서 동작 | 배치=1 디코드에서 argsort/scatter/cnts 기계장치와 레이어당 호스트 동기화(`tokens_per_expert.cpu()`, 토큰당 11회)를 제거하고 topk 전문가만 직접 실행. P17과 달리 **가중치 재스택 없이** 기존 expert 뷰 루프만 사용 → 전문가 연산과 최종 가중합(`view→type→mul_→sum→type`)의 연산 순서가 원본과 동일해 **결과 비트 동일**(M4 Max 골든 실측: 2p·25p result.md sha256 동일, 합성 벤치 `torch.equal`=True). MPS 디스패치 바운드 완화용(2p 2.0x·25p 1.86x). 원본(비패스트패스) 경로는 불변 |
| P19 | (`modeling_deepseekv2.py`) `SlidingWindowLlamaAttention`에 rotary cos/sin 스텝 캐시(`_rope_cached`) 추가 — layer 0가 계산해 캐시 객체(past_kv)에 스태시, 레이어 1+는 **동일 텐서를 재사용** | rotary 출력은 position_ids·dtype에만 의존해 한 스텝의 전 레이어가 같은 값을 중복 계산 → 스텝당 rotary 계산을 레이어 수회에서 1회로 축소. **동일 텐서 재사용이라 결과 비트 동일**. layer 0는 키 일치와 무관하게 항상 재계산·갱신하므로 스텝 간 `id()` 재사용 충돌에도 안전(레이어 실행 순서 항상 0→N). 게이트 없이 전 디바이스 적용 |

| P20 | (`modeling_deepseekv2.py`) `SlidingWindowLlamaAttention.forward`의 **디코드 정상상태(링) 분기**에서 링 위치 상태를 파이썬 int(`past_kv._ring_pos` dict)에서 **디바이스 상주 0-dim int64 텐서**(`past_kv._ring_pos_t` dict)로 전환. 캐시 슬롯 쓰기를 슬라이스 대입(`kcache[:, :, slot:slot+1] = …`)에서 `kcache.index_copy_(2, (ring_pos_t+prefill_len).view(1), …)`로, 슬롯 갱신을 `ring_pos_t.add_(1).remainder_(W)`(텐서 in-place)로 재구성. 파이썬 int 경로는 **폴백 없이 폐기·단일화**(이중 상태 = 버그 온상). `_prefill_length`는 캡처 시점 고정 상수라 int dict 유지 | app/engine/fast_decode.py의 **CUDA Graph 디코드 캡처(U2)**가 성립하려면 링 슬롯 인덱싱·갱신이 캡처 안에서 재생 가능한 텐서 연산이어야 함(파이썬 int 갱신은 캡처에 기록되지 않아 리플레이 시 같은 슬롯만 덮어씀). `index_copy_`는 슬라이스 대입과 **저장 값 동일**, `remainder_`는 `%`와 동일 → **CPU/MPS/CUDA 전 백엔드 공통 적용, 출력 불변**. P16(SDPA)·P17/P18(MoE)·P19(rotary 캐시)와 무접촉 — P19의 `_rope_cached` 호출·어텐션 스코어 경로 그대로. 그래프 캡처 자체는 CUDA·정상상태 한정(`OCR_CUDA_GRAPHS` 킬스위치, 실패 시 eager 폴백)이라 이 텐서화만으로 비CUDA/비그래프 동작은 값 불변 |

업스트림 갱신 시: 새 revision을 받아 이 패치들을 재적용하고 이 문서를 갱신할 것.
