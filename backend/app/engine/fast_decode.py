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

── CUDA Graph 경로 (U2) ────────────────────────────────────────────────
디코드는 링캐시(고정 W) 정상상태에서 shape이 완전 정적 — 남은 병목은 토큰당
수십 개 소형 커널의 launch 갭 + 파이썬 디스패치다. CUDA 입력이고
``OCR_CUDA_GRAPHS`` 가 꺼지지 않았으며(기본 on) 프로세서가 window/ngram_size를
가진 단일 ngram이면, 프리필+워밍업(eager, 디코드 W+block 스텝)으로 링을
정상상태에 올린 뒤 디코드 1스텝을 ``torch.cuda.CUDAGraph`` 로 캡처해 리플레이한다.

  - 그래프 **안**: model.forward(임베딩→레이어들(P20 링 index_copy_·P17 융합 MoE·
    P19 rotary 캐시)→lm_head→float) + GraphNgram.step(밴) + argmax +
    push/cur_tok.copy_/pos_ids.add_ (전부 in-place, 정적 주소).
  - 그래프 **밖**: 블록당 1회 ngram 링에서 최근 block개 토큰 D2H 회수 +
    EOS/stop/스트리머 처리 + input_ids 조립 (eager와 동일한 동기화 cadence).
  - forward kwargs는 ``prepare_inputs_for_generation`` 이 디코드 스텝에 만드는
    구성(input_ids [1,1]·position_ids [1,1]·images*=None)을 정적 버퍼로 재현.
    attention_mask는 이 루프 전체가 쓰지 않으며(위 eager와 동일) 모델도
    디코드(q_len=1, past>0)에서 None 처리한다 — 동적 길이 텐서 없음.
  - position_ids는 정적 버퍼를 in-place 증분한다(id 불변). P19 rotary 캐시는
    layer 0가 키 일치와 무관하게 **항상 재계산·재스태시**하므로 안전하고, 그
    재계산이 캡처에 기록되어 리플레이마다 새 위치의 cos/sin이 계산된다.
    안전장치로 캡처 직전 ``_rope_key`` 를 리셋해 miss 시작을 코드로 고정한다.
  - 캡처는 커널을 **기록만 하고 실행하지 않는다** — 캡처 자체는 토큰을
    생성하지 않으며 상태(pos_ids·ngram·링)도 변하지 않는다.
  - 캡처/리플레이 실패는 try로 감싸 로그 + 같은 캐시로 eager 마무리
    (변환이 절대 죽지 않게). EOS 오버런 ≤ block-1 계약은 기존 그대로.
  - 킬스위치: ``OCR_CUDA_GRAPHS=0/false/no/off`` → 전 구간 eager.
    통합자의 0/1 파리티 검증 기준.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _cuda_graphs_enabled() -> bool:
    """OCR_CUDA_GRAPHS 환경변수 — 미설정(기본)·기타 값이면 on, 0/false/no/off면 off.
    디코드 호출당 1회만 조회된다(토큰 루프 밖)."""
    return os.environ.get("OCR_CUDA_GRAPHS", "").strip().lower() not in ("0", "false", "no", "off")


def _should_try_cuda_graph(input_ids, processors, model, block: int) -> bool:
    """그래프 경로 발동 조건 — 하나라도 어긋나면 False(→ eager, 예외 아님).

    CUDA 입력 · env on · 프로세서가 ngram_size/window를 제공하는 **단일** ngram
    (우리 ngram 하나라는 가정을 검증 — 다르면 폴백) · 링 윈도우 존재(정상상태
    판정 기준) · block ≤ ngram window(블록 회수가 ngram 링 용량을 넘지 않게).
    순수 파이썬 검사만 수행하므로 CPU 스텁으로 단위 검증 가능."""
    if not getattr(input_ids, "is_cuda", False):
        return False
    if not _cuda_graphs_enabled():
        return False
    if len(processors) != 1:
        return False
    proc = processors[0]
    if not (hasattr(proc, "ngram_size") and hasattr(proc, "window")):
        return False
    try:
        if int(block) > int(proc.window):
            return False
    except (TypeError, ValueError):
        return False
    W = getattr(getattr(model, "config", None), "_ring_window", None)
    if not W:
        return False
    return True


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
        "past_key_values": DynamicCache(),
        "use_cache": True,
        # attention_mask 대신 명시적 position_ids — 벤더 prepare_inputs가 위치 복원용으로
        # 마스크 전체에 토큰마다 cumsum을 돌던 것(O(길이) 커널 체인)을 제거한다.
        # 배치=1 그리디에서 마스크는 전부 1이라 정보가 없고, 모델 forward는 q_len=1에서
        # 마스크를 무시하며 프리필은 None이어도 동일한 causal 마스크를 만든다.
        "position_ids": torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
    }
    if streamer is not None:
        streamer.put(input_ids.cpu())  # skip_prompt 스트리머의 프롬프트 소비 (HF와 동일)

    # [U2] CUDA Graph 경로 — 조건 만족 시 캡처+리플레이 (실패해도 내부에서 eager 마무리).
    if _should_try_cuda_graph(input_ids, processors, model, block):
        return _cuda_graph_greedy_decode(
            model, input_ids, model_kwargs, processors, criteria,
            eos_id, streamer, max_length, block,
        )

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
        # 다음 스텝 위치 — 매번 새 텐서를 만든다 (P19 rotary 캐시가 id(position_ids)로
        # 스텝을 식별하므로 in-place 증가는 금지)
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
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


def _cuda_graph_greedy_decode(model, input_ids, model_kwargs, processors, criteria,
                              eos_id, streamer, max_length, block):
    """CUDA Graph 캡처 디코드 본체. 반환 형태·스트리머·EOS 계약은 eager와 동일.

    단계: ① eager 워밍업(프리필 + 디코드 W+block 스텝 — 링 정상상태 진입 +
    cuBLASLt/P17 스택 정착) → ② 정적 버퍼·GraphNgram prime → ③ 사이드스트림
    실행 워밍업(그래프 풀 예열, 실토큰 block개) → ④ 캡처(기록만, 토큰 0개) →
    ⑤ 리플레이 루프(block회 replay → 1회 D2H 회수 → EOS/stop 처리).
    ③~⑤ 실패 시 로그 + 같은 캐시로 eager 마무리(best-effort 토큰 회수 포함)."""
    import torch

    device = input_ids.device
    prompt_len = input_ids.shape[1]
    proc = processors[0]
    ngram_size, ngram_window = int(proc.ngram_size), int(proc.window)
    W = int(model.config._ring_window)

    # 종료/스트림/절단 공유 상태 (아래 클로저들이 갱신)
    state = {"input_ids": input_ids, "finished_at": None, "stopped": False}

    def _flush(new_tokens):
        """호스트 int 리스트(이번 블록 신규 토큰)로 EOS 절단·스트리밍·중단 판정.
        state['input_ids']는 이미 이 토큰들을 포함한 상태에서 호출된다."""
        ii = state["input_ids"]
        start_abs = ii.shape[1] - len(new_tokens)  # 이 블록 첫 토큰의 절대 인덱스
        emit = new_tokens
        if eos_id is not None and eos_id in new_tokens:
            cut = new_tokens.index(eos_id) + 1  # EOS 포함 (HF generate와 동일)
            state["finished_at"] = start_abs + cut
            emit = new_tokens[:cut]
        if streamer is not None and emit:
            streamer.put(torch.tensor([emit]))
        if state["finished_at"] is None and criteria:
            state["stopped"] = any(bool(c(ii, None)) for c in criteria)

    def _eager_step():
        """eager 1스텝 — 위 eager 루프와 동일 수치 경로(prepare_inputs+forward+proc+argmax).
        position_ids는 매번 새 텐서(P19 계약)."""
        ii = state["input_ids"]
        model_inputs = model.prepare_inputs_for_generation(ii, **model_kwargs)
        out = model(**model_inputs, return_dict=True)
        logits = out.logits[:, -1, :]
        for p in processors:
            logits = p(ii, logits)
        nxt = logits.argmax(dim=-1, keepdim=True)
        state["input_ids"] = torch.cat([ii, nxt], dim=-1)
        model_kwargs["past_key_values"] = out.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1

    def _eager_until(stop_len):
        """input_ids가 stop_len/max_length/EOS/취소에 이를 때까지 eager 블록 루프."""
        pending = 0
        while (state["input_ids"].shape[1] < stop_len
               and state["input_ids"].shape[1] < max_length
               and state["finished_at"] is None and not state["stopped"]):
            _eager_step()
            pending += 1
            if pending >= block:
                _flush(state["input_ids"][0, -pending:].tolist())
                pending = 0
        if pending:  # 경계 잔여 — 스트리밍·EOS 절단을 위해 flush (출력 계약 불변)
            _flush(state["input_ids"][0, -pending:].tolist())

    def _done():
        return (state["finished_at"] is not None or state["stopped"]
                or state["input_ids"].shape[1] >= max_length)

    # ── ① eager 워밍업: 링 정상상태(캐시 길이 = prefill+W, 이후 불변)까지 W 디코드
    #    스텝 + 정상상태 eager block-1스텝. 정상상태 진입 전에는 캐시가 매 스텝
    #    재할당(cat)되므로 캡처 불가 — 진입 후에는 캐시 주소가 고정된다.
    _eager_until(prompt_len + W + block)
    if _done():  # 워밍업 중 종료 — 그래프 불필요
        return _finalize(state, streamer)

    # ── ② 정적 상태 준비 ──
    from ..native_ops import GraphSlidingWindowNoRepeatNgram  # 지연 임포트 (torch 로드 후)

    pkv = model_kwargs["past_key_values"]  # 정상상태 — 링 캐시 텐서 주소 고정
    cur_tok = state["input_ids"][:, -1:].clone().contiguous()  # [1,1] 마지막(미투입) 토큰
    pos_val = state["input_ids"].shape[1] - 1                  # 그 토큰의 위치(=인덱스)
    pos_ids = torch.full((1, 1), pos_val, dtype=torch.long, device=device)
    ngram = GraphSlidingWindowNoRepeatNgram(ngram_size, ngram_window, device)
    ngram.prime(state["input_ids"][0])  # cur_tok까지 포함한 전체 시퀀스로 초기화
    base_pos = pos_val   # 실행된 그래프 스텝 수 산출 기준 (pos_ids가 커밋 마커)
    drained = 0          # 그래프 단계에서 input_ids로 회수 완료된 토큰 수

    def _fwd_step():
        # prepare_inputs_for_generation이 디코드 스텝에 만드는 구성을 정적으로 재현:
        # input_ids=[1,1]·position_ids=[1,1]·images*=None·use_cache=True.
        # attention_mask는 eager 경로도 안 쓰며 모델이 디코드에서 None 처리 — 생략과 동치.
        out = model(input_ids=cur_tok, position_ids=pos_ids, past_key_values=pkv,
                    use_cache=True, attention_mask=None, images=None,
                    images_seq_mask=None, images_spatial_crop=None, return_dict=True)
        logits = out.logits[:, -1, :]
        scores = ngram.step(logits)                 # 밴 (eager의 proc(input_ids, logits)와 동치)
        nxt = scores.argmax(dim=-1, keepdim=True)   # [1,1]
        ngram.push(nxt.view(()))                    # 링 기록 — tail이 항상 최신 토큰까지 반영
        cur_tok.copy_(nxt)                          # 다음 스텝 입력
        pos_ids.add_(1)                             # 마지막: 스텝 커밋 마커(진전량의 진리원)

    def _drain(k):
        """직전 k개 생성 토큰을 ngram 링에서 1회 D2H로 회수 → input_ids 조립 + flush."""
        nonlocal drained
        if k <= 0:
            return
        toks = ngram.recent(k).tolist()  # ← 블록당 단 1회의 디바이스 동기화 (eager와 동일 cadence)
        state["input_ids"] = torch.cat(
            [state["input_ids"], torch.tensor([toks], device=device, dtype=torch.long)], dim=-1)
        drained += k
        _flush(toks)

    try:
        # ── ③ 사이드스트림 실행 워밍업(캡처 전 필수 예열 — 표준 절차) — 실토큰 block개 ──
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(block):
                _fwd_step()
        torch.cuda.current_stream().wait_stream(s)
        _drain(block)
        if _done():
            return _finalize(state, streamer)

        # ── ④ 캡처 — 커널을 기록만 한다(실행 없음 → 토큰 0개, 상태 불변) ──
        # [P19 안전장치] 캡처 첫 forward가 반드시 rotary 캐시 miss로 시작하게 리셋 —
        # layer 0는 설계상 키와 무관하게 항상 재계산하지만, 스테일 _rope_val 텐서가
        # 그래프에 상수로 박히는 사고를 코드 수준에서 원천 차단한다. 리셋 후 캡처 중
        # layer 0가 그래프 풀 텐서로 재스태시 → 레이어 1+는 그 풀 텐서를 읽도록 기록됨.
        pkv._rope_key = None
        pkv._rope_val = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fwd_step()

        # ── ⑤ 리플레이 루프: block회 replay → block개 1회 회수 → EOS/stop 처리 ──
        while not _done():
            n_rep = min(block, max_length - state["input_ids"].shape[1])
            for _ in range(n_rep):
                graph.replay()
            _drain(n_rep)
        return _finalize(state, streamer)

    except Exception as exc:  # 캡처/리플레이 실패 — 변환이 죽지 않게 eager로 마무리
        logger.warning("CUDA Graph 디코드 실패 — eager 폴백: %r", exc, exc_info=True)
        # 아직 input_ids로 안 내려간 그래프 스텝 토큰을 best-effort 회수.
        # pos_ids는 각 스텝의 마지막 연산에서만 증가하므로 (완료 스텝 수)의 진리원이다.
        # (캡처는 실행이 아니라서 증가시키지 않음. 스텝 도중 실패는 커널 사이 창이라
        #  실질 불가능하고, 발생해도 ±1 토큰 회수 오차 — 가용성 우선.)
        try:
            advanced = int(pos_ids.item()) - base_pos
            missing = advanced - drained
            if 0 < missing <= ngram_window:
                _drain(missing)
        except Exception:  # pragma: no cover - 회수 실패해도 진행 (크래시 방지 우선)
            pass
        if not _done():
            # eager 재개: past는 그대로, position_ids는 새 텐서(P19 계약)로 재구성.
            model_kwargs["past_key_values"] = pkv
            model_kwargs["position_ids"] = torch.full(
                (1, 1), state["input_ids"].shape[1] - 1, dtype=torch.long, device=device)
            _eager_until(max_length)
        return _finalize(state, streamer)


def _finalize(state, streamer):
    """스트리머 종료 + finished_at 절단 후 input_ids 반환 (그래프 경로 공통 마무리)."""
    if streamer is not None:
        streamer.end()
    ii = state["input_ids"]
    if state["finished_at"] is not None:
        ii = ii[:, :state["finished_at"]]
    return ii
