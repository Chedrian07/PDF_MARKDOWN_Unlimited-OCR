"""[vendor patch P17] DeepseekV2MoE 융합 소배치 경로 수치/게이트 검증.

CUDA 없이 CPU에서 검증한다. 융합 경로는 발동 조건에 x.is_cuda를 포함하지만,
수치 검증은 내부 함수(_moe_infer_fused)를 직접 호출해 그 게이트를 우회한다.

수치 동일성 수준 (요구 1):
- N=1(디코드 소배치)은 bitwise 동일(torch.equal). 토큰 1개당 expert별 단일행
  matmul이라 eager의 mm과 fused의 bmm이 완전히 같은 값을 낸다(전 시드 실측 확인).
- N>1은 eager가 같은 expert에 배정된 여러 토큰을 하나의 mm으로 묶는 반면 fused는
  (token,k) 쌍마다 [1,h] bmm을 돌려 BLAS 누적 순서가 달라진다. dtype 캐스트/최종
  가중합 순서는 moe_infer와 bitwise 동일하게 맞췄으므로 이 차이는 순수 matmul 누적
  round-off(fp32, 실측 max_rel ~2e-7)뿐 → rtol=1e-3으로 완화(대여유 마진).
"""

import pytest

torch = pytest.importorskip("torch")

from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config  # noqa: E402
from app.vendor.unlimited_ocr.modeling_deepseekv2 import DeepseekV2MoE  # noqa: E402

HIDDEN = 32


def _build_moe(seed: int = 0, **overrides) -> DeepseekV2MoE:
    # 미니 config: n_routed_experts=8, num_experts_per_tok=2, hidden 32,
    # moe_intermediate 64, n_shared_experts=1. topk_method는 config 기본값이
    # 오타('gready')라 greedy 분기를 타도록 명시한다.
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(
        hidden_size=HIDDEN,
        moe_intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        topk_method="greedy",
        norm_topk_prob=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        ep_size=1,
        **overrides,
    )
    return DeepseekV2MoE(cfg).eval()


@torch.no_grad()
def _route(moe: DeepseekV2MoE, n_tokens: int, seed: int = 0):
    # 게이트로 실제 라우팅을 뽑아 eager/fused에 동일 입력·동일 라우팅을 먹인다.
    torch.manual_seed(seed)
    h = torch.randn(1, n_tokens, moe.config.hidden_size)
    topk_idx, topk_weight, _ = moe.gate(h)
    x = h.view(-1, moe.config.hidden_size)
    return x, topk_idx, topk_weight


class _FakeTensor:
    """_should_use_fused는 x.is_cuda / x.shape / topk_ids.shape만 본다.

    CPU에는 CUDA 텐서가 없으므로, is_cuda=True 상황의 크기/ep 게이트를 검증하려면
    이 덕타이핑 스텁으로 대신한다(실제 연산은 하지 않음).
    """

    is_cuda = True

    def __init__(self, *shape):
        self.shape = shape


# ── 요구 1·2: 수치 동일성 (N=1 bitwise, N>1 fp round-off) ──


@pytest.mark.parametrize("n_tokens", [1, 4, 16])
def test_fused_matches_eager(n_tokens):
    moe = _build_moe()
    x, topk_idx, topk_weight = _route(moe, n_tokens)
    ref = moe.moe_infer(x, topk_idx, topk_weight)
    got = moe._moe_infer_fused(x, topk_idx, topk_weight)
    assert ref.shape == got.shape == (n_tokens, HIDDEN)
    if n_tokens == 1:
        # 디코드 소배치: bitwise 동일 (모듈 독스트링 참조)
        assert torch.equal(ref, got)
    else:
        # N>1: eager mm vs fused per-pair bmm의 fp32 누적차(실측 ~2e-7)만 허용
        assert torch.allclose(ref, got, rtol=1e-3, atol=1e-5)


def test_fused_bitwise_decode_multiple_seeds():
    # N=1 bitwise 동일성이 시드에 무관함(라우팅이 바뀌어도)을 확인.
    for seed in range(8):
        moe = _build_moe(seed=seed)
        x, topk_idx, topk_weight = _route(moe, 1, seed=seed)
        ref = moe.moe_infer(x, topk_idx, topk_weight)
        got = moe._moe_infer_fused(x, topk_idx, topk_weight)
        assert torch.equal(ref, got), f"seed={seed}"


# ── 요구 3: 뷰 재지정 후 기존 경로 불변 + 스택 스토리지 공유 ──


def test_view_rebind_preserves_eager_and_shares_storage():
    moe = _build_moe()
    x, topk_idx, topk_weight = _route(moe, 4)

    ref_before = moe.moe_infer(x, topk_idx, topk_weight)  # 개별 weight 텐서
    moe._build_fused_experts()  # gate/up/down weight를 스택 뷰로 재지정
    ref_after = moe.moe_infer(x, topk_idx, topk_weight)  # 뷰(동일 데이터)로 재실행

    # 데이터가 그대로 복사된 뷰이므로 기존 루프 결과는 bitwise 불변
    assert torch.equal(ref_before, ref_after)

    g, u, d = moe._fused_w
    for i, e in enumerate(moe.experts):
        # 각 expert weight가 스택 텐서 i번째 슬라이스와 같은 원소 포인터
        assert e.gate_proj.weight.data_ptr() == g[i].data_ptr()
        assert e.up_proj.weight.data_ptr() == u[i].data_ptr()
        assert e.down_proj.weight.data_ptr() == d[i].data_ptr()
        # 그리고 스택 텐서 내부(동일 스토리지)를 가리킴 = VRAM 중복 없음
        assert (
            e.gate_proj.weight.untyped_storage().data_ptr()
            == g.untyped_storage().data_ptr()
        )


def test_build_fused_idempotent():
    # 두 번째 호출은 재스택하지 않고 같은 스택 텐서를 유지(디바이스/dtype 일치 시).
    moe = _build_moe()
    moe._build_fused_experts()
    first = moe._fused_w
    moe._build_fused_experts()
    assert moe._fused_w is first


# ── 요구 4: 발동 조건(킬스위치·CUDA·크기·ep) ──


def test_env_default_on(monkeypatch):
    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    assert _build_moe()._fused_env_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF", "False", " no "])
def test_env_killswitch_off(monkeypatch, val):
    monkeypatch.setenv("OCR_MOE_FUSED", val)
    moe = _build_moe()
    assert moe._fused_env_enabled() is False
    # 킬스위치가 우선 — 다른 조건과 무관하게 발동 안 함
    assert moe._should_use_fused(_FakeTensor(1, HIDDEN), _FakeTensor(1, 2)) is False


def test_should_use_fused_requires_cuda(monkeypatch):
    # CPU 텐서(is_cuda=False)는 기본 CUDA 전용 정책상 발동 안 함.
    monkeypatch.setenv("OCR_MOE_FUSED", "1")
    moe = _build_moe()
    x, topk_idx, _ = _route(moe, 1)
    assert x.is_cuda is False
    assert moe._should_use_fused(x, topk_idx) is False


def test_should_use_fused_size_gate(monkeypatch):
    # is_cuda=True(스텁)에서 **N==1(디코드)만** 발동 검증 — N>1은 mm↔bmm 누적
    # 순서차로 근접 argmax가 뒤집혀 E2E가 갈라진 실측(표 colspan 손상) 때문에 제외.
    monkeypatch.setenv("OCR_MOE_FUSED", "1")
    moe = _build_moe()
    assert moe._should_use_fused(_FakeTensor(1, HIDDEN), _FakeTensor(1, 2)) is True
    assert moe._should_use_fused(_FakeTensor(2, HIDDEN), _FakeTensor(2, 2)) is False
    assert moe._should_use_fused(_FakeTensor(32, HIDDEN), _FakeTensor(32, 2)) is False


def test_should_use_fused_ep_gate(monkeypatch):
    # ep_size>1(전문가 병렬)에서는 융합 경로 미발동 → 기존 all-to-all moe_infer.
    monkeypatch.setenv("OCR_MOE_FUSED", "1")
    moe = _build_moe()
    moe.ep_size = 2
    assert moe._should_use_fused(_FakeTensor(1, HIDDEN), _FakeTensor(1, 2)) is False


# ── forward 통합: CPU eval은 기존 eager 경로로 정상 동작(엣지 회귀 가드) ──


def test_forward_eval_cpu_runs():
    moe = _build_moe()
    h = torch.randn(1, 4, moe.config.hidden_size)
    with torch.no_grad():
        y = moe(h)
    assert y.shape == h.shape
    assert torch.isfinite(y).all()
