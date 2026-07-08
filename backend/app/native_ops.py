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


class GraphSlidingWindowNoRepeatNgram:
    """CUDA Graph 캡처용 정적-shape no-repeat-ngram (``TorchSliding…``의 그래프 대응).

    ``TorchSliding…``는 매 호출 input_ids 전체(동적 길이)에서 ``seq[-w:]``를 슬라이스한다.
    그래프 안에서는 동적 슬라이스가 불가하므로, 마지막 ``window`` 토큰을 고정 크기 링
    버퍼에 상주시키고 shape이 완전히 정적인 연산만으로 밴 마스크를 만든다.

    상태(전부 디바이스 상주, 캡처 중 고정 주소):
      - ``tail`` [window] int64 링 버퍼(생성 토큰 저장)
      - ``pos``   0-dim int64  다음 쓰기 슬롯
      - ``count`` 0-dim int64  유효 길이(window로 saturate)
      - ``_ar_W``/``_ar_M`` arange 상수(캡처 중 읽기 전용)

    사용 절차:
      1. ``prime(seq)``      — 프리필+워밍업 토큰으로 초기화(캡처 전 1회, 동적 OK)
      2. ``step(scores)``    — 정적 연산만으로 밴 마스크 계산·masked_fill(캡처 대상)
      3. ``push(next_tok)``  — 생성 토큰을 링에 기록(캡처 대상)
      ``recent(k)``          — 최근 k개 토큰 회수(호스트 D2H용, 비캡처)

    의미론은 ``banned_ngram_tokens_py``(레퍼런스)와 동일하며, count < ngram_size면
    밴이 비도록 마스크 산술로 처리한다(파이썬 분기 없음). tests/test_native_ops.py가
    레퍼런스와의 패리티를 검증한다.
    """

    def __init__(self, ngram_size: int, window: int, device) -> None:
        import torch

        if ngram_size < 1 or window < 1:
            raise ValueError("ngram_size와 window는 1 이상이어야 합니다")
        self.ngram_size = int(ngram_size)
        self.window = int(window)
        self.device = device
        W, n = self.window, self.ngram_size
        self.tail = torch.zeros(W, dtype=torch.long, device=device)
        self.pos = torch.zeros((), dtype=torch.long, device=device)
        self.count = torch.zeros((), dtype=torch.long, device=device)
        self._ar_W = torch.arange(W, device=device)
        self._ar_M = torch.arange(max(W - n + 1, 0), device=device)

    def prime(self, token_ids_1d) -> None:
        """프리필+워밍업 토큰 1-D 텐서로 링을 초기화(마지막 window개만 보존).
        캡처 전 1회 — 동적 길이 연산 허용."""
        import torch

        seq = token_ids_1d
        L = int(seq.shape[0])
        W = self.window
        k = min(L, W)
        self.tail.zero_()
        if k > 0:
            self.tail[:k].copy_(seq[-k:].to(device=self.device, dtype=torch.long))
        # tail[:k]=마지막 k개(오래된→최신), pos=k%W, count=k 로 두면
        # win = tail[(pos+arange(W))%W] 이 항상 최신을 win[-1]에 정렬한다(recent/step 공통).
        self.pos.fill_(k % W)
        self.count.fill_(k)

    def _window_ordered(self):
        """링을 오래된→최신 순 [window] 텐서로 복원(win[-1]=최신)."""
        roll = (self.pos + self._ar_W).remainder(self.window)  # [W]
        return self.tail.index_select(0, roll)

    def step(self, scores):
        """scores [1, vocab]에 밴(-inf) 적용 후 반환. 정적 shape 연산만."""
        import torch

        W, n = self.window, self.ngram_size
        if W < n:  # 윈도우가 ngram보다 작으면 ngram 자체가 불가 — no-op(상수 분기)
            return scores
        vocab = scores.shape[-1]
        win = self._window_ordered()          # [W], win[-1]=최신
        unf = win.unfold(0, n, 1)             # [M, n] 뷰, M = W-n+1
        # 유효 ngram: 시작 인덱스 idx >= W-count (그 앞은 프라임 패딩). count는 텐서 —
        # 파이썬 분기 없이 마스크 산술로. count<n이면 valid가 전부 False → 밴 없음(레퍼런스와 동일).
        valid = self._ar_M >= (W - self.count)  # [M] bool
        if n > 1:
            prefix = win[W - n + 1:]            # [n-1] 현재 (n-1)-그램(고정 슬라이스)
            pmatch = unf[:, :-1].eq(prefix).all(dim=1)  # [M]
        else:
            pmatch = torch.ones(unf.shape[0], dtype=torch.bool, device=win.device)
        match = valid & pmatch                 # [M]
        cand = unf[:, -1]                       # [M] 각 ngram의 마지막 토큰(밴 후보)
        # 비매치 후보는 vocab 번째(버림 슬롯)로 스캐터 → 분기/동기화 없이 마스크 구성.
        # 주의: `ban[idx] = True`(파이썬 스칼라 index_put_)는 동기식 H2D 스칼라 복사를
        # 유발해 CUDA Graph 캡처를 깨뜨린다(실측: cudaErrorStreamCaptureUnsupported,
        # native_ops.py:181). index_fill_은 값이 커널 인자로 박혀 캡처 안전.
        idx = torch.where(match, cand, torch.full_like(cand, vocab))
        ban = torch.zeros(vocab + 1, dtype=torch.bool, device=scores.device)
        ban.index_fill_(0, idx, True)
        return scores.masked_fill(ban[:vocab], float("-inf"))

    def push(self, next_token) -> None:
        """생성 토큰(0-dim 또는 1개 원소 텐서)을 링에 기록. 전부 in-place 텐서 연산."""
        W = self.window
        self.tail.index_copy_(0, self.pos.view(1), next_token.reshape(1).to(self.tail.dtype))
        self.pos.add_(1).remainder_(W)         # 다음 슬롯 (in-place, 캡처 재생 가능)
        self.count.add_(1).clamp_(max=W)       # 유효 길이 saturate

    def recent(self, k: int):
        """최근 k개 토큰을 [k] 텐서(오래된→최신)로 반환 — 호스트 회수(.tolist())용."""
        k = max(0, min(int(k), self.window))
        return self._window_ordered()[self.window - k:]


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
    device_type: torch 디바이스 문자열 (cpu | cuda | mps)

    OCR_NGRAM_HOST=1이면 GPU/MPS에서도 호스트(C++/파이썬) 티어를 강제한다 —
    MPS scatter류 이슈(P11 전례) 절연용 디버그 레버."""
    import os

    force_host = os.environ.get("OCR_NGRAM_HOST", "").strip().lower() in ("1", "true", "yes", "on")
    if device_type in ("cuda", "mps") and not force_host:
        return [TorchSlidingWindowNoRepeatNgram(ngram_size, window)]
    return [HostSlidingWindowNoRepeatNgram(ngram_size, window)]
