// uocr_native — C++ native acceleration for the Unlimited-OCR PDF -> Markdown service.
//
// Two hot-path helpers, each with an exact pure-Python reference (see
// docs/ARCHITECTURE.md section 9 and native/tests/test_parity.py):
//
//   1. banned_ngram_tokens(sequence, ngram_size, window)
//        Sliding-window no-repeat-ngram logits processor. Called once per
//        generated token during LLM decoding, so it stays allocation-light:
//        a single pass over the window, deduping into a std::set (already
//        sorted) which is copied straight into the returned ndarray.
//
//   2. crop_regions(image, boxes)
//        Crops figure regions out of a rendered page given 0-999 normalized
//        boxes, matching numpy slicing byte-for-byte.
//
// The module is optional: app/native_ops.py falls back to pure Python if the
// import fails, so correctness (parity) matters far more than raw speed here.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Function 1: banned_ngram_tokens
// ---------------------------------------------------------------------------
//
// Exact semantics of the Python reference:
//
//   seq = list(sequence)
//   if len(seq) < ngram_size: return []
//   search_start = max(0, len(seq) - window)
//   search_end   = len(seq) - ngram_size + 1
//   if search_end <= search_start: return []
//   current_prefix = tuple(seq[-(ngram_size - 1):]) if ngram_size > 1 else ()
//   banned = set()
//   for idx in range(search_start, search_end):
//       ngram = seq[idx:idx + ngram_size]
//       if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
//           banned.add(ngram[-1])
//   return sorted(banned)
//
// Returned as an ascending, unique 1-D int64 ndarray.
static py::array_t<std::int64_t> banned_ngram_tokens(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sequence,
    std::int64_t ngram_size,
    std::int64_t window) {
    if (ngram_size < 1) {
        throw std::invalid_argument("ngram_size must be >= 1");
    }
    if (window < 1) {
        throw std::invalid_argument("window must be >= 1");
    }
    if (sequence.ndim() != 1) {
        throw std::invalid_argument("sequence must be a 1-D int64 array");
    }

    const std::int64_t n = static_cast<std::int64_t>(sequence.shape(0));
    const std::int64_t* data = sequence.data();

    std::set<std::int64_t> banned;

    if (n >= ngram_size) {
        const std::int64_t search_start = std::max<std::int64_t>(0, n - window);
        const std::int64_t search_end = n - ngram_size + 1;  // exclusive
        if (search_end > search_start) {
            const std::int64_t prefix_len = ngram_size - 1;
            if (ngram_size == 1) {
                // ngram[-1] == seq[idx]; prefix is empty so every idx qualifies.
                for (std::int64_t idx = search_start; idx < search_end; ++idx) {
                    banned.insert(data[idx]);
                }
            } else {
                // current_prefix == seq[n - prefix_len : n]
                const std::int64_t* prefix = data + (n - prefix_len);
                const std::size_t prefix_bytes =
                    static_cast<std::size_t>(prefix_len) * sizeof(std::int64_t);
                for (std::int64_t idx = search_start; idx < search_end; ++idx) {
                    // tuple(ngram[:-1]) == current_prefix
                    if (std::memcmp(data + idx, prefix, prefix_bytes) == 0) {
                        banned.insert(data[idx + prefix_len]);  // ngram[-1]
                    }
                }
            }
        }
    }

    py::array_t<std::int64_t> result(static_cast<py::ssize_t>(banned.size()));
    std::int64_t* out = result.mutable_data();
    py::ssize_t i = 0;
    for (std::int64_t v : banned) {  // std::set iterates in ascending order
        out[i++] = v;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Function 2: crop_regions
// ---------------------------------------------------------------------------
//
// image: HxWx3 uint8 C-contiguous.
// boxes: Nx4 int64, rows (x1, y1, x2, y2) in 0-999 normalized coordinates.
//
// Per box (matching Python int() truncation toward zero exactly; a C++
// double->int64 cast truncates toward zero the same way):
//   x1p = int(x1 / 999 * W);  x2p = int(x2 / 999 * W)
//   y1p = int(y1 / 999 * H);  y2p = int(y2 / 999 * H)
//   x1p = max(x1p, 0); y1p = max(y1p, 0); x2p = min(x2p, W); y2p = min(y2p, H)
// If x2p <= x1p or y2p <= y1p -> None, else a NEW C-contiguous uint8 copy of
// image[y1p:y2p, x1p:x2p] with shape (y2p-y1p, x2p-x1p, 3).
// Returned list length is always N.
static py::list crop_regions(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> image,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> boxes) {
    if (image.ndim() != 3 || image.shape(2) != 3) {
        throw std::invalid_argument("image must be an HxWx3 uint8 array");
    }
    if (boxes.ndim() != 2 || boxes.shape(1) != 4) {
        throw std::invalid_argument("boxes must be an Nx4 int64 array");
    }

    const std::int64_t H = static_cast<std::int64_t>(image.shape(0));
    const std::int64_t W = static_cast<std::int64_t>(image.shape(1));
    const std::uint8_t* img = image.data();

    const std::int64_t N = static_cast<std::int64_t>(boxes.shape(0));
    const std::int64_t* bx = boxes.data();

    const double Wd = static_cast<double>(W);
    const double Hd = static_cast<double>(H);

    py::list out;

    for (std::int64_t i = 0; i < N; ++i) {
        const std::int64_t x1 = bx[i * 4 + 0];
        const std::int64_t y1 = bx[i * 4 + 1];
        const std::int64_t x2 = bx[i * 4 + 2];
        const std::int64_t y2 = bx[i * 4 + 3];

        // Same operation order as Python: (v / 999) * dim, then truncate.
        std::int64_t x1p = static_cast<std::int64_t>(static_cast<double>(x1) / 999.0 * Wd);
        std::int64_t y1p = static_cast<std::int64_t>(static_cast<double>(y1) / 999.0 * Hd);
        std::int64_t x2p = static_cast<std::int64_t>(static_cast<double>(x2) / 999.0 * Wd);
        std::int64_t y2p = static_cast<std::int64_t>(static_cast<double>(y2) / 999.0 * Hd);

        x1p = std::max<std::int64_t>(x1p, 0);
        y1p = std::max<std::int64_t>(y1p, 0);
        x2p = std::min<std::int64_t>(x2p, W);
        y2p = std::min<std::int64_t>(y2p, H);

        if (x2p <= x1p || y2p <= y1p) {
            out.append(py::none());
            continue;
        }

        const std::int64_t ch = y2p - y1p;
        const std::int64_t cw = x2p - x1p;

        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(ch),
                                          static_cast<py::ssize_t>(cw),
                                          static_cast<py::ssize_t>(3)};
        py::array_t<std::uint8_t> crop(shape);
        std::uint8_t* dst = crop.mutable_data();

        const std::int64_t dst_row_bytes = cw * 3;
        for (std::int64_t r = 0; r < ch; ++r) {
            // Source row (y1p + r), starting at column x1p, in the H x W x 3
            // C-contiguous image buffer.
            const std::uint8_t* src = img + (((y1p + r) * W + x1p) * 3);
            std::memcpy(dst + r * dst_row_bytes, src,
                        static_cast<std::size_t>(dst_row_bytes));
        }
        out.append(std::move(crop));
    }

    return out;
}

PYBIND11_MODULE(uocr_native, m) {
    m.doc() =
        "Native acceleration for baidu/Unlimited-OCR: no-repeat-ngram token "
        "banning and figure region cropping.";

    m.def("banned_ngram_tokens", &banned_ngram_tokens, py::arg("sequence"),
          py::arg("ngram_size"), py::arg("window"),
          "Return the ascending, unique int64 array of tokens that would "
          "complete a repeated n-gram within the trailing `window` tokens of "
          "`sequence` (sliding-window no-repeat-ngram banning).");

    m.def("crop_regions", &crop_regions, py::arg("image"), py::arg("boxes"),
          "Crop HxWx3 uint8 `image` at each Nx4 int64 box (0-999 normalized "
          "coords). Returns a length-N list of new C-contiguous uint8 crops, "
          "or None for degenerate/empty boxes.");
}
