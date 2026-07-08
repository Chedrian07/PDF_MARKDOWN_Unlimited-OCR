"""[vendor patch P18] MoE 단일 토큰 디코드 패스트패스 — 골든 패리티(비트 동일) 검증.

배치=1 디코드에서 argsort/scatter/cnts 기계장치와 레이어당 호스트 동기화를 건너뛰고
topk 전문가만 직접 실행한다. 연산 순서(view→type→mul_→sum→type)가 원본과 동일해
패스트패스(on)와 원본 경로(off) 결과가 torch.equal 이어야 한다. CI는 CPU라 게이트를
OCR_MOE_FAST=1/0으로 강제해 두 경로를 CPU에서 비교한다.
"""

import pytest

torch = pytest.importorskip("torch")

from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config  # noqa: E402
from app.vendor.unlimited_ocr import modeling_deepseekv2 as mod  # noqa: E402
from app.vendor.unlimited_ocr.modeling_deepseekv2 import DeepseekV2MoE  # noqa: E402


def _tiny_config() -> DeepseekV2Config:
    return DeepseekV2Config(
        hidden_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        n_shared_experts=1,
        topk_method="greedy",
        scoring_func="softmax",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        aux_loss_alpha=0.0,
    )


def _build_moe(dtype=torch.float32) -> DeepseekV2MoE:
    # 고정 시드 → 결정적 가중치. eval()이라야 forward가 moe_infer(추론 경로)를 탄다.
    torch.manual_seed(0)
    moe = DeepseekV2MoE(_tiny_config()).eval()
    if dtype is not torch.float32:
        moe = moe.to(dtype)
    return moe


def _run(moe, x, monkeypatch, flag):
    monkeypatch.setenv("OCR_MOE_FAST", flag)
    with torch.no_grad():
        return moe(x)


# ── 게이트 로직 (P16 강제 레버 패턴) ──

def test_moe_fast_gate_env_override(monkeypatch):
    """OCR_MOE_FAST=1은 전 디바이스 강제 on, 0은 강제 off, 미설정은 MPS 전용."""
    monkeypatch.setenv("OCR_MOE_FAST", "1")
    assert mod._moe_fast_enabled("cpu") is True
    assert mod._moe_fast_enabled("cuda") is True
    assert mod._moe_fast_enabled("mps") is True

    monkeypatch.setenv("OCR_MOE_FAST", "0")
    assert mod._moe_fast_enabled("mps") is False
    assert mod._moe_fast_enabled("cpu") is False

    monkeypatch.delenv("OCR_MOE_FAST", raising=False)
    assert mod._moe_fast_enabled("mps") is True  # 기본 MPS 전용
    assert mod._moe_fast_enabled("cpu") is False
    assert mod._moe_fast_enabled("cuda") is False


# ── 비트 동일 골든 패리티 ──

def test_single_token_fastpath_bit_identical_fp32(monkeypatch):
    """seq==1 디코드: 패스트패스(on)와 원본 경로(off) 결과가 비트 동일 (fp32)."""
    moe = _build_moe()
    torch.manual_seed(1)
    x = torch.randn(1, 1, 16)
    fast = _run(moe, x, monkeypatch, "1")
    slow = _run(moe, x, monkeypatch, "0")
    assert torch.equal(fast, slow)


def test_single_token_fastpath_bit_identical_bf16(monkeypatch):
    """bfloat16에서도 연산 순서가 동일하므로 비트 동일이어야 한다."""
    moe = _build_moe(torch.bfloat16)
    torch.manual_seed(1)
    x = torch.randn(1, 1, 16, dtype=torch.bfloat16)
    fast = _run(moe, x, monkeypatch, "1")
    slow = _run(moe, x, monkeypatch, "0")
    assert torch.equal(fast, slow)


def test_multi_token_ignores_fastpath(monkeypatch):
    """seq>1 (x.shape[0]!=1)은 env=1이어도 패스트패스를 안 타므로 두 설정 결과 동일."""
    moe = _build_moe()
    torch.manual_seed(2)
    x = torch.randn(1, 3, 16)
    on = _run(moe, x, monkeypatch, "1")
    off = _run(moe, x, monkeypatch, "0")
    assert torch.equal(on, off)


def test_fastpath_branch_is_actually_exercised(monkeypatch):
    """CPU·seq1·env=1에서 실제로 패스트패스 분기를 타는지 확인 — 위 패리티 테스트의
    비공허성(둘 다 원본 경로로 빠져 우연히 같아지는 것 아님)을 보장한다."""
    moe = _build_moe()
    monkeypatch.setenv("OCR_MOE_FAST", "1")
    took_fast = []
    real = mod.DeepseekV2MoE.moe_infer

    def spy(self, x, topk_ids, topk_weight):
        took_fast.append(
            x.shape[0] == 1 and self.ep_size == 1 and mod._moe_fast_enabled(x.device.type)
        )
        return real(self, x, topk_ids, topk_weight)

    monkeypatch.setattr(mod.DeepseekV2MoE, "moe_infer", spy)
    torch.manual_seed(1)
    with torch.no_grad():
        moe(torch.randn(1, 1, 16))
    assert took_fast == [True]
