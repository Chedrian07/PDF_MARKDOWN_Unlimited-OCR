"""Parity tests: uocr_native (C++) must match the pure-Python references exactly.

The reference implementations are embedded here on purpose (self-contained, no
imports from the rest of the repo) so this test file fully specifies the C++
module's contract.
"""

import numpy as np
import pytest

import uocr_native


# ---------------------------------------------------------------------------
# Reference implementations (the contract the C++ code must match exactly)
# ---------------------------------------------------------------------------
def banned_ngram_tokens_ref(sequence, ngram_size, window):
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
        ngram = seq[idx:idx + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            banned.add(ngram[-1])
    return sorted(banned)


def crop_regions_ref(image, boxes):
    H, W = int(image.shape[0]), int(image.shape[1])
    out = []
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1p = int(x1 / 999 * W)
        y1p = int(y1 / 999 * H)
        x2p = int(x2 / 999 * W)
        y2p = int(y2 / 999 * H)
        x1p = max(x1p, 0)
        y1p = max(y1p, 0)
        x2p = min(x2p, W)
        y2p = min(y2p, H)
        if x2p <= x1p or y2p <= y1p:
            out.append(None)
        else:
            out.append(np.array(image[y1p:y2p, x1p:x2p], copy=True))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seq(values):
    return np.array(values, dtype=np.int64)


def _assert_banned_matches(sequence, ngram_size, window):
    seq_arr = _seq(sequence)
    expected = banned_ngram_tokens_ref(list(sequence), ngram_size, window)
    got = uocr_native.banned_ngram_tokens(seq_arr, ngram_size, window)

    assert isinstance(got, np.ndarray)
    assert got.dtype == np.int64
    assert got.ndim == 1
    # ascending + unique
    assert list(got) == sorted(set(got.tolist()))
    assert got.tolist() == expected


def _assert_crop_matches(image, boxes):
    boxes_arr = np.array(boxes, dtype=np.int64).reshape(-1, 4)
    expected = crop_regions_ref(image, boxes_arr)
    got = uocr_native.crop_regions(image, boxes_arr)

    assert isinstance(got, list)
    assert len(got) == len(expected) == boxes_arr.shape[0]
    for g, e in zip(got, expected):
        if e is None:
            assert g is None
        else:
            assert g is not None
            assert isinstance(g, np.ndarray)
            assert g.dtype == np.uint8
            assert g.flags["C_CONTIGUOUS"]
            assert g.shape == e.shape
            assert np.array_equal(g, e)
            # Must be a fresh copy, not a view into the source image.
            assert not np.shares_memory(g, image)


# ---------------------------------------------------------------------------
# banned_ngram_tokens — explicit boundary cases
# ---------------------------------------------------------------------------
def test_banned_empty_sequence():
    _assert_banned_matches([], 1, 10)
    _assert_banned_matches([], 3, 10)


def test_banned_seq_shorter_than_ngram():
    _assert_banned_matches([5], 3, 10)
    _assert_banned_matches([1, 2], 3, 10)
    _assert_banned_matches([1, 2, 3], 5, 100)


def test_banned_window_larger_than_seq():
    _assert_banned_matches([1, 2, 3, 1, 2, 4], 3, 1000)
    _assert_banned_matches([7, 7, 7, 7], 2, 999999)


def test_banned_window_smaller_than_seq():
    # Only the trailing window participates.
    seq = [1, 2, 3, 4, 1, 2, 3, 9, 1, 2, 3, 5]
    _assert_banned_matches(seq, 4, 4)
    _assert_banned_matches(seq, 4, 6)
    _assert_banned_matches(seq, 3, 5)
    _assert_banned_matches(seq, 2, 3)


def test_banned_ngram_size_one():
    # ngram_size == 1 bans every distinct token inside the trailing window.
    _assert_banned_matches([1, 2, 2, 3, 3, 3, 4], 1, 3)
    _assert_banned_matches([1, 2, 2, 3, 3, 3, 4], 1, 100)
    _assert_banned_matches([9, 9, 9, 9, 9], 1, 1)
    _assert_banned_matches([], 1, 5)


def test_banned_window_equals_one():
    _assert_banned_matches([1, 2, 3, 4, 5], 1, 1)
    # window == 1 with ngram_size > 1 => search_end <= search_start => empty.
    _assert_banned_matches([1, 2, 3, 4, 5], 3, 1)


def test_banned_prefix_match_and_collisions():
    # Repeated "1,2,3" prefix: completing token after prefix (1,2) should be
    # banned from earlier occurrences.
    seq = [1, 2, 3, 0, 1, 2, 7, 0, 1, 2]  # current prefix is (1, 2)
    _assert_banned_matches(seq, 3, 100)


# ---------------------------------------------------------------------------
# banned_ngram_tokens — randomized parity (seeded)
# ---------------------------------------------------------------------------
def test_banned_random_small_vocab_broad():
    rng = np.random.default_rng(12345)
    for _ in range(400):
        vocab = int(rng.integers(1, 5))          # tiny vocab -> forces collisions
        length = int(rng.integers(0, 80))
        seq = rng.integers(0, vocab + 1, size=length).tolist()
        ngram_size = int(rng.integers(1, 8))
        window = int(rng.integers(1, 60))
        _assert_banned_matches(seq, ngram_size, window)


def test_banned_ngram35_window1024_randomized():
    """>=200 seeded cases at the production config (no_repeat_ngram_size=35,
    ngram_window=1024). Sequences are built by tiling short random patterns so
    that 34-token prefixes actually repeat and the banning path is exercised."""
    rng = np.random.default_rng(2026)
    for _ in range(250):
        vocab = int(rng.integers(2, 7))
        pattern_len = int(rng.integers(1, 50))
        pattern = rng.integers(0, vocab, size=pattern_len)
        total = int(rng.integers(0, 1500))
        if total == 0:
            seq = []
        else:
            reps = total // pattern_len + 1
            seq = np.tile(pattern, reps)[:total].tolist()
            # add a little random noise on top to vary prefixes
            noise = int(rng.integers(0, min(len(seq) + 1, 20)))
            for _ in range(noise):
                j = int(rng.integers(0, len(seq)))
                seq[j] = int(rng.integers(0, vocab))
        _assert_banned_matches(seq, 35, 1024)


def test_banned_pure_random_large_ngram():
    # Pure random (collisions unlikely) — still must match (both empty-ish).
    rng = np.random.default_rng(777)
    for _ in range(200):
        length = int(rng.integers(0, 1200))
        seq = rng.integers(0, 5000, size=length).tolist()
        _assert_banned_matches(seq, 35, 1024)


# ---------------------------------------------------------------------------
# banned_ngram_tokens — input handling
# ---------------------------------------------------------------------------
def test_banned_forcecast_and_noncontiguous():
    # Non-contiguous int64 slice — forcecast should accept it and match.
    base = np.arange(0, 40, dtype=np.int64)
    seq = base[::2]  # non-contiguous view
    assert not seq.flags["C_CONTIGUOUS"]
    _assert_banned_matches(seq.tolist(), 2, 10)


def test_banned_invalid_ngram_or_window_raises():
    seq = _seq([1, 2, 3])
    with pytest.raises(ValueError):
        uocr_native.banned_ngram_tokens(seq, 0, 10)
    with pytest.raises(ValueError):
        uocr_native.banned_ngram_tokens(seq, -1, 10)
    with pytest.raises(ValueError):
        uocr_native.banned_ngram_tokens(seq, 2, 0)
    with pytest.raises(ValueError):
        uocr_native.banned_ngram_tokens(seq, 2, -5)


def test_banned_wrong_ndim_raises():
    bad = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(ValueError):
        uocr_native.banned_ngram_tokens(bad, 2, 10)


# ---------------------------------------------------------------------------
# crop_regions — explicit boundary cases
# ---------------------------------------------------------------------------
def _img(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_crop_full_frame_box_0_to_999():
    img = _img(50, 80, seed=1)
    # (0,0,999,999) should map to the (almost) full frame.
    _assert_crop_matches(img, [[0, 0, 999, 999]])


def test_crop_box_at_extremes():
    img = _img(30, 40, seed=2)
    _assert_crop_matches(img, [
        [0, 0, 500, 500],
        [500, 500, 999, 999],
        [0, 0, 0, 0],        # degenerate -> None
        [999, 999, 999, 999],
    ])


def test_crop_degenerate_boxes():
    img = _img(20, 20, seed=3)
    _assert_crop_matches(img, [
        [100, 100, 100, 300],   # x1 == x2 -> None
        [100, 100, 300, 100],   # y1 == y2 -> None
        [400, 400, 300, 500],   # x2 < x1 -> None
        [400, 400, 500, 300],   # y2 < y1 -> None
    ])


def test_crop_tiny_images():
    _assert_crop_matches(_img(1, 1, seed=4), [[0, 0, 999, 999], [0, 0, 500, 500]])
    _assert_crop_matches(_img(3, 7, seed=5), [[0, 0, 999, 999],
                                              [0, 0, 400, 400],
                                              [500, 500, 999, 999]])
    _assert_crop_matches(_img(7, 3, seed=6), [[0, 0, 999, 999],
                                              [200, 200, 800, 800]])


def test_crop_empty_boxes():
    img = _img(10, 10, seed=7)
    got = uocr_native.crop_regions(img, np.zeros((0, 4), dtype=np.int64))
    assert got == []


def test_crop_random_boxes_seeded():
    rng = np.random.default_rng(99)
    for _ in range(200):
        h = int(rng.integers(1, 60))
        w = int(rng.integers(1, 60))
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        nboxes = int(rng.integers(1, 8))
        boxes = []
        for _ in range(nboxes):
            x1 = int(rng.integers(0, 1000))
            y1 = int(rng.integers(0, 1000))
            x2 = int(rng.integers(0, 1000))
            y2 = int(rng.integers(0, 1000))
            boxes.append([x1, y1, x2, y2])
        _assert_crop_matches(img, boxes)


def test_crop_random_boxes_including_0_and_999():
    rng = np.random.default_rng(4242)
    for _ in range(200):
        h = int(rng.integers(1, 40))
        w = int(rng.integers(1, 40))
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        # Bias coordinates toward the 0 / 999 extremes.
        choices = np.array([0, 0, 999, 999, 1, 998,
                            int(rng.integers(0, 1000)),
                            int(rng.integers(0, 1000))])
        boxes = []
        for _ in range(int(rng.integers(1, 6))):
            picked = rng.choice(choices, size=4)
            boxes.append([int(v) for v in picked])
        _assert_crop_matches(img, boxes)


# ---------------------------------------------------------------------------
# crop_regions — input handling
# ---------------------------------------------------------------------------
def test_crop_forcecast_noncontiguous_boxes():
    img = _img(20, 20, seed=8)
    boxes64 = np.array([[0, 0, 999, 999], [100, 100, 800, 800]], dtype=np.int64)
    # Non-contiguous boxes view (transpose round-trip) still must work.
    boxes_view = np.asfortranarray(boxes64)
    assert not boxes_view.flags["C_CONTIGUOUS"]
    _assert_crop_matches(img, boxes_view)


def test_crop_wrong_shape_raises():
    img = _img(10, 10, seed=9)
    with pytest.raises(ValueError):
        uocr_native.crop_regions(img, np.zeros((3, 3), dtype=np.int64))  # not Nx4
    with pytest.raises(ValueError):
        # image not HxWx3
        uocr_native.crop_regions(np.zeros((10, 10), dtype=np.uint8),
                                 np.zeros((1, 4), dtype=np.int64))
    with pytest.raises(ValueError):
        uocr_native.crop_regions(np.zeros((10, 10, 4), dtype=np.uint8),
                                 np.zeros((1, 4), dtype=np.int64))


# ---------------------------------------------------------------------------
# Smoke benchmark — 10k calls on a 1024 window (no speed assertion, no crash)
# ---------------------------------------------------------------------------
def test_smoke_benchmark_10k_calls():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 8, size=1200, dtype=np.int64)  # small vocab, > window
    for _ in range(10_000):
        out = uocr_native.banned_ngram_tokens(seq, 35, 1024)
        assert out.dtype == np.int64
    # Final call parity sanity check.
    _assert_banned_matches(seq.tolist(), 35, 1024)
