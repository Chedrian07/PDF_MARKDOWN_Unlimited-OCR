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
