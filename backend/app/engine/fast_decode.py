"""범용(CPU/CUDA/MPS) 고속 그리디 디코드 루프 — HF ``generate()`` 대체.

측정된 병목: HF generate는 **토큰마다** EOS 체크(``.max()`` 디바이스 동기화),
스트리머 D2H 전송, 프로세서/크라이테리아 기계장치를 수행한다. 이 루프는
모델의 ``prepare_inputs_for_generation``/``forward``를 그대로 재사용해
**연산 경로(수치)는 HF와 동일**하게 유지하면서 그 기계장치만 제거하고,
피할 수 없는 호스트 동기화를 ``block`` 토큰마다 1회로 배칭한다.
순수 torch 연산만 사용하므로 cpu/cuda/mps 세 백엔드에서 같은 코드가 돈다.

트레이드오프: EOS를 블록 경계에서만 확인하므로 최대 block-1개 토큰을
초과 생성할 수 있다 — 반환 전에 EOS 위치로 잘라내므로 결과는 동일하고,
버려질 토큰의 생성 비용(수십 ms)만 소모된다. 라이브 스트림 체감 지연은
block=8 기준 0.1~0.2초 수준.

제약: 그리디(do_sample=False) 전용 — 샘플링 요청은 HF generate로 폴백.
전역 비활성: OCR_FAST_DECODE=0 (엔진이 generate_fn을 주입하지 않음).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fast_greedy_decode(model, gen_kwargs: dict, block: int = 8):
    """벤더 infer/infer_multi의 gen_kwargs(P15의 generate_fn 계약)를 받아
    output_ids [1, prompt+생성] 텐서를 돌려준다 — self.generate와 동일 형태."""
    import torch
    from transformers.cache_utils import DynamicCache

    if gen_kwargs.get("do_sample"):
        logger.info("do_sample 요청 — HF generate로 폴백")
        return model.generate(**gen_kwargs)

    input_ids = gen_kwargs["input_ids"]
    eos_id = gen_kwargs.get("eos_token_id")
    streamer = gen_kwargs.get("streamer")
    max_length = int(gen_kwargs.get("max_length") or 32768)
    processors = list(gen_kwargs.get("logits_processor") or [])
    criteria = list(gen_kwargs.get("stopping_criteria") or [])
    block = max(1, int(block))

    model_kwargs = {
        "images": gen_kwargs.get("images"),
        "images_seq_mask": gen_kwargs.get("images_seq_mask"),
        "images_spatial_crop": gen_kwargs.get("images_spatial_crop"),
        "attention_mask": torch.ones_like(input_ids),
        "past_key_values": DynamicCache(),
        "use_cache": True,
    }
    if streamer is not None:
        streamer.put(input_ids.cpu())  # skip_prompt 스트리머의 프롬프트 소비 (HF와 동일)

    finished_at = None  # EOS 포함 절단 위치 (prompt+생성 기준 절대 인덱스)
    pending = 0         # 아직 호스트로 내리지 않은 신규 토큰 수
    stopped = False

    while input_ids.shape[1] < max_length and finished_at is None and not stopped:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        out = model(**model_inputs, return_dict=True)
        logits = out.logits[:, -1, :]
        for proc in processors:
            logits = proc(input_ids, logits)
        next_tok = logits.argmax(dim=-1, keepdim=True)  # 디바이스에 상주 — 동기화 없음
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        model_kwargs["past_key_values"] = out.past_key_values
        am = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([am, am.new_ones(1, 1)], dim=-1)
        pending += 1

        if pending >= block or input_ids.shape[1] >= max_length:
            new = input_ids[0, -pending:].tolist()  # ← 블록당 단 1회의 디바이스 동기화
            if eos_id is not None and eos_id in new:
                cut = new.index(eos_id) + 1  # EOS 포함 (HF generate와 동일)
                finished_at = input_ids.shape[1] - pending + cut
                new = new[:cut]
            if streamer is not None and new:
                streamer.put(torch.tensor([new]))
            pending = 0
            if finished_at is None and criteria:
                stopped = any(bool(c(input_ids, None)) for c in criteria)

    if pending:  # 방어적 잔여 flush (정상 흐름에선 위 블록이 항상 소진)
        new = input_ids[0, -pending:].tolist()
        if eos_id is not None and eos_id in new:
            cut = new.index(eos_id) + 1
            finished_at = input_ids.shape[1] - pending + cut
            new = new[:cut]
        if streamer is not None and new:
            streamer.put(torch.tensor([new]))

    if streamer is not None:
        streamer.end()
    if finished_at is not None:
        input_ids = input_ids[:, :finished_at]
    return input_ids
