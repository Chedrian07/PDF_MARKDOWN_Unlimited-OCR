"""C++ 네이티브 모듈(uocr_native) 로더 + 순수 파이썬 폴백.

네이티브 모듈이 없어도 앱은 완전히 동작한다 (docs/ARCHITECTURE.md §9).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import uocr_native as _native

    HAVE_NATIVE = True
except ImportError:  # pragma: no cover - 환경 의존
    _native = None
    HAVE_NATIVE = False
    logger.info("uocr_native 미설치 — 순수 파이썬 폴백 사용")


def banned_ngram_tokens_py(sequence, ngram_size: int, window: int) -> list[int]:
    """벤더 코드 SlidingWindowNoRepeatNgramProcessor와 동일한 의미론 (레퍼런스)."""
    seq = list(sequence)
    if len(seq) < ngram_size:
        return []
    search_start = max(0, len(seq) - window)
    search_end = len(seq) - ngram_size + 1
    if search_end <= search_start:
        return []
    current_prefix = tuple(seq[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    banned = set()
    for idx in range(search_start, search_end):
        ngram = seq[idx : idx + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            banned.add(ngram[-1])
    return sorted(banned)


def banned_ngram_tokens(sequence, ngram_size: int, window: int) -> list[int]:
    if HAVE_NATIVE:
        import numpy as np

        arr = np.ascontiguousarray(sequence, dtype=np.int64)
        return _native.banned_ngram_tokens(arr, ngram_size, window).tolist()
    return banned_ngram_tokens_py(sequence, ngram_size, window)


def make_ngram_logits_processor(ngram_size: int, window: int):
    """네이티브 가속 no-repeat-ngram 프로세서. 네이티브 미설치 시 None을 돌려
    호출측이 벤더 기본(순수 파이썬) 프로세서를 쓰게 한다."""
    if not HAVE_NATIVE:
        return None

    import numpy as np
    import torch

    class _NativeSlidingWindowNoRepeatNgram:
        def __init__(self) -> None:
            self.ngram_size = ngram_size
            self.window = window

        def __call__(self, input_ids: "torch.Tensor", scores: "torch.Tensor") -> "torch.Tensor":
            for b in range(input_ids.shape[0]):
                seq = np.ascontiguousarray(input_ids[b].detach().cpu().numpy(), dtype=np.int64)
                banned = _native.banned_ngram_tokens(seq, self.ngram_size, self.window)
                if banned.size:
                    idx = torch.from_numpy(banned).to(scores.device)
                    scores[b, idx] = float("-inf")
            return scores

    return [_NativeSlidingWindowNoRepeatNgram()]
