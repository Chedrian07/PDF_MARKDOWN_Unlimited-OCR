import random

import pytest

from app import native_ops


def _brute(seq, n, w):
    return native_ops.banned_ngram_tokens_py(seq, n, w)


CASES = [
    ([], 3, 10),
    ([1, 2], 3, 10),
    ([1, 2, 3], 3, 10),
    ([1, 2, 3, 1, 2, 4, 1, 2], 3, 100),
    ([5] * 40, 1, 8),
    (list(range(10)) * 3, 2, 7),
]


@pytest.mark.parametrize("seq,n,w", CASES)
def test_fallback_reference_cases(seq, n, w):
    out = native_ops.banned_ngram_tokens(seq, n, w)
    assert out == _brute(seq, n, w)


def test_fallback_semantics_repeat_blocking():
    # "1 2 3" 이 윈도우 안에 있고 현재 프리픽스가 "1 2"면 3이 금지된다
    assert 3 in _brute([1, 2, 3, 9, 9, 1, 2], 3, 100)
    # 윈도우 밖이면 금지되지 않는다
    assert _brute([1, 2, 3] + [9] * 50 + [1, 2], 3, 4) == []


@pytest.mark.skipif(not native_ops.HAVE_NATIVE, reason="uocr_native 미설치")
def test_native_matches_python_randomized():
    rng = random.Random(42)
    for _ in range(100):
        seq = [rng.randrange(6) for _ in range(rng.randrange(0, 200))]
        n = rng.randrange(1, 6)
        w = rng.randrange(1, 64)
        assert native_ops.banned_ngram_tokens(seq, n, w) == _brute(seq, n, w)


# ── 디바이스별 로짓 프로세서 패리티 (torch 필요 — cpu 텐서로 로직 검증) ──

def _processor_banned(proc_cls, seq, n, w, vocab):
    torch = pytest.importorskip("torch")
    ids = torch.tensor([list(seq)], dtype=torch.long)
    scores = torch.zeros(1, vocab)
    proc_cls(n, w)(ids, scores)
    return sorted(i for i in range(vocab) if scores[0, i] == float("-inf"))


@pytest.mark.parametrize("proc_name", ["TorchSlidingWindowNoRepeatNgram", "HostSlidingWindowNoRepeatNgram"])
@pytest.mark.parametrize("seq,n,w", CASES)
def test_processors_reference_cases(proc_name, seq, n, w):
    proc_cls = getattr(native_ops, proc_name)
    vocab = 16
    assert _processor_banned(proc_cls, seq, n, w, vocab) == _brute(seq, n, w)


@pytest.mark.parametrize("proc_name", ["TorchSlidingWindowNoRepeatNgram", "HostSlidingWindowNoRepeatNgram"])
def test_processors_randomized_parity(proc_name):
    proc_cls = getattr(native_ops, proc_name)
    rng = random.Random(7)
    vocab = 8
    for _ in range(200):
        seq = [rng.randrange(vocab) for _ in range(rng.randrange(0, 160))]
        n = rng.randrange(1, 6)
        w = rng.randrange(1, 48)
        assert _processor_banned(proc_cls, seq, n, w, vocab) == _brute(seq, n, w), (seq, n, w)


@pytest.mark.parametrize("proc_name", ["TorchSlidingWindowNoRepeatNgram", "HostSlidingWindowNoRepeatNgram"])
def test_processors_window_slice_boundaries(proc_name):
    # 슬라이스 동치가 위험한 경계: L == w, L == w±1, L == n, w < n
    proc_cls = getattr(native_ops, proc_name)
    vocab = 6
    rng = random.Random(11)
    for L, n, w in [(64, 3, 64), (65, 3, 64), (63, 3, 64), (5, 5, 64), (40, 4, 2), (40, 1, 8)]:
        for _ in range(20):
            seq = [rng.randrange(vocab) for _ in range(L)]
            assert _processor_banned(proc_cls, seq, n, w, vocab) == _brute(seq, n, w), (L, n, w)


def test_processor_production_shape():
    # 실사용 파라미터 (n=35, window=1024) 대형 시퀀스
    rng = random.Random(3)
    vocab = 24
    seq = [rng.randrange(vocab) for _ in range(3000)]
    # 반복 유도: 마지막 34개 프리픽스를 윈도우 중간에 복제
    seq[1500:1534] = seq[-34:]
    ref = _brute(seq, 35, 1024)
    assert _processor_banned(native_ops.TorchSlidingWindowNoRepeatNgram, seq, 35, 1024, vocab) == ref
    assert _processor_banned(native_ops.HostSlidingWindowNoRepeatNgram, seq, 35, 1024, vocab) == ref


def test_make_processor_tiers():
    torch_proc = native_ops.make_ngram_logits_processor(35, 1024, "cuda")
    assert isinstance(torch_proc[0], native_ops.TorchSlidingWindowNoRepeatNgram)
    mps_proc = native_ops.make_ngram_logits_processor(35, 1024, "mps")
    assert isinstance(mps_proc[0], native_ops.TorchSlidingWindowNoRepeatNgram)
    cpu_proc = native_ops.make_ngram_logits_processor(35, 128, "cpu")
    assert isinstance(cpu_proc[0], native_ops.HostSlidingWindowNoRepeatNgram)


def test_ngram_host_강제_레버(monkeypatch):
    """OCR_NGRAM_HOST=1이면 GPU/MPS에서도 호스트 티어 — MPS scatter 이슈 절연용."""
    from app.native_ops import (
        HostSlidingWindowNoRepeatNgram,
        TorchSlidingWindowNoRepeatNgram,
        make_ngram_logits_processor,
    )

    monkeypatch.delenv("OCR_NGRAM_HOST", raising=False)
    assert isinstance(make_ngram_logits_processor(30, 128, "mps")[0], TorchSlidingWindowNoRepeatNgram)
    monkeypatch.setenv("OCR_NGRAM_HOST", "1")
    assert isinstance(make_ngram_logits_processor(30, 128, "mps")[0], HostSlidingWindowNoRepeatNgram)
    assert isinstance(make_ngram_logits_processor(30, 128, "cuda")[0], HostSlidingWindowNoRepeatNgram)
