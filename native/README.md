# uocr_native

C++17 (pybind11) native acceleration for the **Unlimited-OCR PDF → Markdown**
service. Optional: `backend/app/native_ops.py` falls back to pure Python if this
module is not importable, so the app works with or without it.

Module name: `uocr_native`. Distribution name: `uocr-native`.

## Functions

### `banned_ngram_tokens(sequence, ngram_size, window) -> np.ndarray[int64]`
Sliding-window no-repeat-ngram token banning, called once per generated token
during decoding (`no_repeat_ngram_size=35`, `ngram_window=1024` in production).

- `sequence`: 1-D `int64` ndarray of tokens generated so far (forcecast; a
  non-contiguous or castable array is accepted and converted).
- `ngram_size >= 1`, `window >= 1` (else `ValueError`).
- Returns the **ascending, unique** int64 array of tokens that would complete a
  repeated n-gram inside the trailing `window`.

### `crop_regions(image, boxes) -> list[np.ndarray | None]`
Crops figure regions out of a rendered page.

- `image`: `HxWx3` `uint8` C-contiguous ndarray (forcecast accepted).
- `boxes`: `Nx4` `int64` ndarray, rows `(x1, y1, x2, y2)` in 0–999 normalized
  coordinates (forcecast accepted).
- Per box: `x1p = int(x1/999*W)` (truncation toward zero) etc., then clamp to
  the frame. Degenerate/empty boxes yield `None`; otherwise a **new,
  C-contiguous** `uint8` crop of shape `(y2p-y1p, x2p-x1p, 3)`.
- The returned list is always length `N` (1:1 with `boxes`).

## Build & test (Python 3.12 via uv)

The host Python may be newer than 3.12; use a uv-managed 3.12 interpreter. The
`.venv/`, `build/`, and `*.so` artifacts are already git-ignored at the repo
root.

```bash
cd native
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e . pytest numpy
.venv/bin/python -m pytest tests/ -v
```

Toolchain: CMake ≥ 3.15 and a C++17 compiler (GCC/Clang). `scikit-build-core`
uses the Ninja generator (pulled in as a build requirement).

## Building a wheel

```bash
uv build --wheel        # writes dist/uocr_native-0.1.0-*.whl
# or, equivalently:
pip wheel . -w dist
```

The build is standalone (PEP 517, isolated): `scikit-build-core` + `pybind11` +
`ninja` are resolved automatically as build requirements. Only `numpy` is a
runtime dependency.
