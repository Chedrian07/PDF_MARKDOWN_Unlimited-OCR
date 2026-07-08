"""[vendor patch P19] rotary cos/sin 스텝 캐시 — 스텝당 rotary 계산 12회→1회 검증.

rotary 출력은 position_ids와 dtype에만 의존하므로 한 스텝의 모든 레이어에서 동일하다.
layer 0가 계산해 캐시 객체(past_kv)에 스태시하고 layer 1+가 동일 텐서를 재사용한다
(동일 텐서 재사용 → 비트 동일). SlidingWindowLlamaAttention(mha_eager)이 선택되도록
use_mla=False, first_k_dense_replace를 크게 줘 전 레이어 dense MLP(MoE 무관)로 구성한다.
CI는 CPU라 전부 CPU에서 돈다.
"""

import pytest

torch = pytest.importorskip("torch")

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.llama import modeling_llama  # noqa: E402

from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config  # noqa: E402
from app.vendor.unlimited_ocr.modeling_deepseekv2 import DeepseekV2Model  # noqa: E402

NUM_LAYERS = 2


def _tiny_config() -> DeepseekV2Config:
    return DeepseekV2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=99,  # 전 레이어 dense MLP → MoE 경로와 무관
        moe_layer_freq=1,
        topk_method="greedy",
        scoring_func="softmax",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        aux_loss_alpha=0.0,
        use_mla=False,  # mha_eager → SlidingWindowLlamaAttention (P19 대상)
        max_position_embeddings=64,
        rope_theta=10000.0,
    )


def _build_model() -> DeepseekV2Model:
    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg._attn_implementation = "eager"
    model = DeepseekV2Model(cfg).eval()
    # mha_eager 계열(SlidingWindowLlamaAttention)이 실제로 선택됐는지 확인
    assert type(model.layers[0].self_attn).__name__ == "SlidingWindowLlamaAttention"
    return model


@pytest.fixture
def rope_counter(monkeypatch):
    """LlamaRotaryEmbedding.forward 호출 횟수 카운터 (레이어별 인스턴스가 모두 이 클래스)."""
    calls = {"n": 0}
    real = modeling_llama.LlamaRotaryEmbedding.forward

    def counting(self, x, position_ids):
        calls["n"] += 1
        return real(self, x, position_ids)

    monkeypatch.setattr(modeling_llama.LlamaRotaryEmbedding, "forward", counting)
    return calls


def test_prefill_computes_rope_once_regardless_of_layers(rope_counter):
    """prefill([1,4]) 1회 forward에 rotary 계산이 레이어 수(=2)와 무관하게 1회."""
    model = _build_model()
    assert model.config.num_hidden_layers == NUM_LAYERS
    ids = torch.randint(0, 64, (1, 4))
    rope_counter["n"] = 0
    with torch.no_grad():
        model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
    assert rope_counter["n"] == 1


def test_each_decode_step_computes_rope_once(rope_counter):
    """이어지는 디코드 스텝([1,1], 명시적 position_ids)마다 rotary 계산 1회."""
    model = _build_model()
    ids = torch.randint(0, 64, (1, 4))
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, past_key_values=cache)
    cache = out.past_key_values
    for step in range(4, 8):
        tok = torch.randint(0, 64, (1, 1))
        pos = torch.tensor([[step]], dtype=torch.long)
        rope_counter["n"] = 0
        with torch.no_grad():
            out = model(
                input_ids=tok, use_cache=True, past_key_values=cache, position_ids=pos
            )
        cache = out.past_key_values
        assert rope_counter["n"] == 1, f"step {step}"


def test_incremental_decode_matches_full_forward():
    """캐시 없이 한 번에 forward한 출력과 prefill+디코드 출력이 allclose (fp32, atol=1e-5).
    rotary 캐시가 값을 바꾸지 않음을 KV 캐시 인과 일관성으로 검증한다."""
    model = _build_model()
    seq = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        full = model(input_ids=seq, use_cache=False).last_hidden_state  # [1,8,H]

    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=seq[:, :4], use_cache=True, past_key_values=cache)
    cache = out.past_key_values
    inc = [out.last_hidden_state[:, -1, :]]  # 위치 3
    for step in range(4, 8):  # 위치 4..7을 티처포싱으로 디코드
        pos = torch.tensor([[step]], dtype=torch.long)
        with torch.no_grad():
            out = model(
                input_ids=seq[:, step : step + 1],
                use_cache=True,
                past_key_values=cache,
                position_ids=pos,
            )
        cache = out.past_key_values
        inc.append(out.last_hidden_state[:, -1, :])
    inc = torch.cat(inc, dim=0)  # 위치 3..7
    assert torch.allclose(inc, full[0, 3:8, :], atol=1e-5)


def test_ring_path_runs_and_caches_rope(rope_counter):
    """_ring_window=2 링 경로(워밍업→링 상주)를 5스텝 돌려 예외 없이 동작하고
    각 스텝 rotary 계산이 1회인지 검증."""
    model = _build_model()
    model.config._ring_window = 2  # 링 버퍼 활성화
    ids = torch.randint(0, 64, (1, 4))
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, past_key_values=cache)
    cache = out.past_key_values
    for step in range(4, 9):  # 5스텝: prefill_len(4)+W(2) 경계 넘어 워밍업 2 + 링 3
        tok = torch.randint(0, 64, (1, 1))
        pos = torch.tensor([[step]], dtype=torch.long)
        rope_counter["n"] = 0
        with torch.no_grad():
            out = model(
                input_ids=tok, use_cache=True, past_key_values=cache, position_ids=pos
            )
        cache = out.past_key_values
        assert rope_counter["n"] == 1, f"ring step {step}"


def test_deterministic_across_runs():
    """동일 시드 모델·동일 입력의 2회 실행이 torch.equal (결정론)."""
    seq = torch.randint(0, 64, (1, 6))

    def _run():
        model = _build_model()
        with torch.no_grad():
            return model(input_ids=seq, use_cache=False).last_hidden_state

    assert torch.equal(_run(), _run())
