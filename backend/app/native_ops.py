"""no-repeat-ngram 로짓 프로세서 — 디바이스별 최적 구현 + 순수 파이썬 폴백.

`make_ngram_logits_processor()`가 디바이스에 맞는 티어를 고른다:
- cuda/mps: torch 텐서 연산만으로 밴 마스크 계산(`TorchSlidingWindowNoRepeatNgram`)
  — 기존에는 토큰마다 시퀀스 전체를 GPU→CPU로 복사·동기화했고 시퀀스가 길수록
  비용이 커졌다(측정: 디코드 병목의 주 요인). GPU 상주 구현은 동기화가 없다.
- cpu(+C++): uocr_native 스캔. 마지막 window 토큰만 잘라 넘겨 복사를 상수화.
- cpu(폴백): 동일 슬라이스를 쓰는 파이썬 구현.

의미론은 전부 벤더 SlidingWindowNoRepeatNgramProcessor(레퍼런스 = 아래
banned_ngram_tokens_py)와 동일하며 tests/test_native_ops.py가 패리티를 검증한다.

슬라이스 동치 증명: 레퍼런스는 ngram 시작 위치 idx ∈ [max(0, L−w), L−n]만 본다.
이 구간의 ngram은 전부 마지막 w개 토큰 안에 있고(시작 ≥ L−w, 끝 ≤ L−1),
현재 프리픽스도 마지막 n−1개 토큰이므로 seq[−w:]만으로 결과가 같다.
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


class TorchSlidingWindowNoRepeatNgram:
    """CUDA/MPS 상주 no-repeat-ngram — 호스트 동기화·전송 없이 인그래프로 동작."""

    def __init__(self, ngram_size: int, window: int) -> None:
        if ngram_size < 1 or window < 1:
            raise ValueError("ngram_size와 window는 1 이상이어야 합니다")
        self.ngram_size = ngram_size
        self.window = window

    def __call__(self, input_ids, scores):
        import torch

        n, w = self.ngram_size, self.window
        vocab = scores.shape[-1]
        for b in range(input_ids.shape[0]):
            seq = input_ids[b]
            if seq.shape[0] < n:
                continue
            seg = seq[-w:]
            m = seg.shape[0] - n + 1
            if m <= 0:
                continue
            unf = seg.unfold(0, n, 1)  # [m, n] 뷰
            if n > 1:
                match = (unf[:, :-1] == seg[-(n - 1):]).all(dim=1)
            else:
                match = torch.ones(m, dtype=torch.bool, device=seq.device)
            # 비매치 후보는 vocab 번째(버림 슬롯)로 스캐터 → 분기/동기화 없이 마스크 구성
            cand = unf[:, -1]
            idx = torch.where(match, cand, torch.full_like(cand, vocab))
            ban = torch.zeros(vocab + 1, dtype=torch.bool, device=scores.device)
            ban[idx] = True
            scores[b] = scores[b].masked_fill(ban[:vocab], float("-inf"))
        return scores


class HostSlidingWindowNoRepeatNgram:
    """CPU 경로 — 마지막 window 토큰만 잘라 C++(가능 시) 또는 파이썬으로 스캔."""

    def __init__(self, ngram_size: int, window: int) -> None:
        if ngram_size < 1 or window < 1:
            raise ValueError("ngram_size와 window는 1 이상이어야 합니다")
        self.ngram_size = ngram_size
        self.window = window

    def __call__(self, input_ids, scores):
        import torch

        for b in range(input_ids.shape[0]):
            seq = input_ids[b, -self.window:].tolist()
            if len(seq) < self.ngram_size:
                continue
            banned = banned_ngram_tokens(seq, self.ngram_size, self.window)
            if len(banned):
                idx = torch.as_tensor(banned, dtype=torch.long, device=scores.device)
                scores[b, idx] = float("-inf")
        return scores


def make_ngram_logits_processor(ngram_size: int, window: int, device_type: str = "cpu") -> list:
    """generate()에 넘길 logits_processor 리스트를 디바이스에 맞게 생성.
    device_type: torch 디바이스 문자열 (cpu | cuda | mps)"""
    if device_type in ("cuda", "mps"):
        return [TorchSlidingWindowNoRepeatNgram(ngram_size, window)]
    return [HostSlidingWindowNoRepeatNgram(ngram_size, window)]
