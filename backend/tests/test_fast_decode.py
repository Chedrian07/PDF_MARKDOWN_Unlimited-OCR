from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from transformers.cache_utils import DynamicCache  # noqa: E402

from app.engine.fast_decode import fast_greedy_decode  # noqa: E402

VOCAB = 16
EOS = 9


class StubModel:
    """(마지막 토큰+1) % 8 을 다음 토큰으로 내는 결정적 스텁.
    eos_after 스텝을 넘기면 EOS(9)를 낸다. 캐시 소비량을 DynamicCache 속성으로 추적."""

    def __init__(self, eos_after=None):
        self.eos_after = eos_after
        self.steps = 0
        self.generate_called = False

    def generate(self, **kw):
        self.generate_called = True
        return kw["input_ids"]

    def prepare_inputs_for_generation(self, input_ids, **kw):
        past = kw.get("past_key_values")
        seen = getattr(past, "seen", 0)
        return {"input_ids": input_ids[:, seen:], "past_key_values": past}

    def __call__(self, input_ids=None, past_key_values=None, return_dict=True, **kw):
        past_key_values.seen = getattr(past_key_values, "seen", 0) + input_ids.shape[1]
        self.steps += 1
        last = int(input_ids[0, -1])
        if self.eos_after is not None and self.steps > self.eos_after:
            target = EOS
        else:
            target = (last + 1) % 8
        logits = torch.full((1, input_ids.shape[1], VOCAB), -10.0)
        logits[0, -1, target] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class StubStreamer:
    def __init__(self):
        self.puts: list[list[int]] = []
        self.ended = False

    def put(self, value):
        v = value[0] if value.dim() > 1 else value
        self.puts.append(v.tolist())

    def end(self):
        self.ended = True


def _run(model, prompt=(0, 1), block=4, **overrides):
    gen_kwargs = {
        "input_ids": torch.tensor([list(prompt)], dtype=torch.long),
        "eos_token_id": EOS,
        "max_length": 64,
        "do_sample": False,
        **overrides,
    }
    return fast_greedy_decode(model, gen_kwargs, block=block)


def test_generates_until_eos_and_trims_overrun():
    model = StubModel(eos_after=10)
    out = _run(model, block=4)
    # 10 스텝 체인 + 11번째 EOS. 블록 경계(12)까지 1개 초과 생성됐다가 절단됨
    assert out.shape == (1, 13)
    assert int(out[0, -1]) == EOS
    assert EOS not in out[0, :-1].tolist()
    expected_chain = [2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
    assert out[0, 2:-1].tolist() == expected_chain


def test_streamer_receives_prompt_then_tokens_in_order():
    model = StubModel(eos_after=10)
    streamer = StubStreamer()
    out = _run(model, block=4, streamer=streamer)
    assert streamer.ended
    assert streamer.puts[0] == [0, 1]  # 프롬프트 (skip_prompt 스트리머 소비용)
    streamed = [t for chunk in streamer.puts[1:] for t in chunk]
    assert streamed == out[0, 2:].tolist()  # EOS 포함, 초과분/중복/누락 없음


def test_logits_processors_applied_every_step():
    calls = []

    class CountingProc:
        def __call__(self, ids, scores):
            calls.append(ids.shape[1])
            return scores

    model = StubModel(eos_after=6)
    out = _run(model, block=4, logits_processor=[CountingProc()])
    generated_steps = model.steps
    assert len(calls) == generated_steps
    assert int(out[0, -1]) == EOS


def test_stopping_criteria_checked_at_block_boundary():
    model = StubModel(eos_after=None)  # EOS 없음

    class StopAt8:
        def __call__(self, ids, scores, **kw):
            return ids.shape[1] >= 2 + 8

    out = _run(model, block=4, stopping_criteria=[StopAt8()])
    assert out.shape == (1, 2 + 8)  # 블록 경계(8토큰)에서 정지


def test_max_length_cap_flushes_pending():
    model = StubModel(eos_after=None)
    streamer = StubStreamer()
    out = _run(model, block=4, streamer=streamer, max_length=9)
    assert out.shape == (1, 9)
    streamed = [t for chunk in streamer.puts[1:] for t in chunk]
    assert streamed == out[0, 2:].tolist()  # 7토큰(블록 4+3) 전부 스트림됨


def test_do_sample_falls_back_to_hf_generate():
    model = StubModel()
    out = _run(model, do_sample=True)
    assert model.generate_called
    assert out.tolist() == [[0, 1]]


def test_block_one_equals_larger_block_output():
    a = _run(StubModel(eos_after=9), block=1)
    b = _run(StubModel(eos_after=9), block=7)
    assert a.tolist() == b.tolist()  # 블록 크기는 결과에 영향 없음 (동기화 빈도만)


class RecordingModel(StubModel):
    """prepare_inputs_for_generation에 전달된 kwargs를 스텝별로 기록하는 스텁.
    attention_mask 전달 여부, position_ids 값 스냅샷, position_ids 텐서 객체를 남긴다."""

    def __init__(self, eos_after=None):
        super().__init__(eos_after=eos_after)
        self.saw_attention_mask = False
        self.position_id_values: list[list[int]] = []
        self.position_id_tensors: list = []  # 참조 유지 → id() 재사용(GC) 방지

    def prepare_inputs_for_generation(self, input_ids, **kw):
        if "attention_mask" in kw:
            self.saw_attention_mask = True
        pos = kw.get("position_ids")
        self.position_id_values.append(pos[0].tolist())  # 호출 시점 값 스냅샷
        self.position_id_tensors.append(pos)             # 객체 자체 보관 (아래 id 검사용)
        return super().prepare_inputs_for_generation(input_ids, **kw)


def test_passes_position_ids_and_no_attention_mask():
    """attention_mask를 넘기지 않고, position_ids가 프리필 [0..L-1] → 이후 [L],[L+1],…로 증가."""
    model = RecordingModel(eos_after=6)
    prompt = (0, 1, 2)  # L=3
    _run(model, prompt=prompt, block=4)

    assert model.saw_attention_mask is False  # (a) attention_mask 미전달
    assert model.position_id_values[0] == [0, 1, 2]  # (b) 프리필: 전체 위치
    # 이후 디코드 스텝: [3], [4], [5], … 단조 증가
    for step, value in enumerate(model.position_id_values[1:], start=len(prompt)):
        assert value == [step]


def test_position_ids_is_a_fresh_tensor_each_step():
    """(c) 매 스텝 position_ids는 직전과 다른 새 텐서 객체 — P19 rotary 캐시가
    id(position_ids)로 스텝을 식별하므로 in-place 증가(동일 id 재사용)는 회귀다."""
    model = RecordingModel(eos_after=6)
    _run(model, prompt=(0, 1, 2), block=4)

    tensors = model.position_id_tensors
    assert len(tensors) >= 3  # 프리필 + 최소 몇 스텝은 돌아야 의미 있는 검사
    ids = [id(t) for t in tensors]  # 전부 살아있어 id 재사용 불가
    assert len(set(ids)) == len(ids)  # 모든 스텝이 서로 다른 객체
