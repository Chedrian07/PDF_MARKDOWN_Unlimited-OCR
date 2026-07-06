"""baidu/Unlimited-OCR 실엔진 (벤더링 코드 사용, CPU/CUDA/Metal 공용).

공식 사용 파라미터 (모델 README):
- 멀티페이지: prompt='<image>Multi page parsing.', image_size=1024,
  no_repeat_ngram_size=35, ngram_window=1024
- 단일(gundam): prompt='<image>document parsing.', base_size=1024,
  image_size=640, crop_mode=True, ngram_window=128
"""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import threading
from pathlib import Path

from ..config import Settings
from ..native_ops import make_ngram_logits_processor
from .base import EngineError, OCREngine, StreamSink

logger = logging.getLogger(__name__)

MULTI_PROMPT = "<image>Multi page parsing."
SINGLE_PROMPT = "<image>document parsing."
NGRAM_SIZE = 35
MULTI_NGRAM_WINDOW = 1024
SINGLE_NGRAM_WINDOW = 128

# 사용자 노출 디바이스명(OCR_DEVICE) → torch 디바이스 문자열
_TORCH_DEVICES = {"cpu": "cpu", "cuda": "cuda", "metal": "mps"}


def torch_device_name(device: str) -> str:
    return _TORCH_DEVICES.get(device, device)


def _mps_bf16_supported() -> bool:
    # bf16은 macOS 14+ 에서만 지원 — 실제 할당으로 프로브 (torch 버전별 API 차이 회피)
    import torch

    try:
        torch.zeros(1, dtype=torch.bfloat16, device="mps")
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _apple_chip_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return out or "Apple Silicon"
    except Exception:
        return "Apple Silicon"


def _resolve_dtype(device: str, dtype_name: str):
    import torch

    if dtype_name == "auto":
        if device == "cuda":
            return torch.bfloat16
        if device == "metal":
            if _mps_bf16_supported():
                return torch.bfloat16
            logger.warning("이 macOS/torch 조합은 MPS bfloat16을 지원하지 않아 float32로 동작합니다 "
                           "(느림 — OCR_DTYPE=float16 시도 가능)")
            return torch.float32
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"알 수 없는 OCR_DTYPE: {dtype_name!r} (auto|bfloat16|float16|float32)")


class UnlimitedEngine(OCREngine):
    name = "unlimited"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.device = settings.device
        self.torch_device = torch_device_name(settings.device)
        self._model = None
        self._tokenizer = None
        self.dtype_name = ""
        if self.device == "metal":
            # MPS 미구현 op를 CPU로 폴백 — torch 첫 임포트 전에 설정돼야 적용된다
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer

        from ..vendor.unlimited_ocr import UnlimitedOCRForCausalLM

        s = self._settings
        if self.device == "cuda" and not torch.cuda.is_available():
            raise EngineError(
                "OCR_DEVICE=cuda 이지만 CUDA를 사용할 수 없습니다. "
                "GPU/드라이버와 컨테이너의 gpus 설정을 확인하세요."
            )
        if self.device == "metal" and not torch.backends.mps.is_available():
            hint = (
                "Apple Silicon Mac + macOS 12.3 이상이 필요합니다."
                if torch.backends.mps.is_built()
                else "설치된 torch가 MPS 없이 빌드되었습니다 — macOS arm64용 휠로 재설치하세요 "
                     "(backend에서 `uv sync --extra metal`). Docker/Linux에서는 Metal을 쓸 수 없습니다."
            )
            raise EngineError(f"OCR_DEVICE=metal 이지만 MPS를 사용할 수 없습니다. {hint}")
        dtype = _resolve_dtype(self.device, s.dtype)
        self.dtype_name = str(dtype).replace("torch.", "")
        logger.info("모델 로딩 시작: %s@%s (device=%s/%s dtype=%s)",
                    s.model_id, s.model_revision[:8], self.device, self.torch_device, self.dtype_name)
        tokenizer = AutoTokenizer.from_pretrained(s.model_id, revision=s.model_revision)
        # 이 아키텍처는 mha_eager(SlidingWindowLlamaAttention)만 구현 → eager 필수
        model = UnlimitedOCRForCausalLM.from_pretrained(
            s.model_id,
            revision=s.model_revision,
            dtype=dtype,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model = model.eval().to(self.torch_device)
        self._model = model
        self._tokenizer = tokenizer
        logger.info("모델 로딩 완료")

    def gpu_name(self) -> str | None:
        if self.device == "metal":
            return _apple_chip_name()
        if self.device != "cuda":
            return None
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - 방어적
            return None

    # ── 내부 ───────────────────────────────────────────────────

    def _release_device_cache(self) -> None:
        """청크 사이 MPS 캐시 반환 — 유니파이드 메모리라 장문서 잡의 시스템 메모리 압박을 줄인다."""
        if self.torch_device != "mps":
            return
        try:
            import torch

            torch.mps.empty_cache()
        except Exception:  # pragma: no cover - 방어적
            pass

    def _gen_extras(self, sink: StreamSink, cancel: threading.Event, ngram_window: int) -> dict:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        tokenizer = self._tokenizer
        eos_text = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)

        class _SinkStreamer(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                text = text.replace(eos_text, "\n")
                if text:
                    sink.on_text(text)

        class _CancelCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return cancel.is_set()

        extras: dict = {
            "streamer": _SinkStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
            "stopping_criteria": StoppingCriteriaList([_CancelCriteria()]),
        }
        native_lp = make_ngram_logits_processor(NGRAM_SIZE, ngram_window)
        if native_lp is not None:
            extras["logits_processor"] = native_lp
        return extras

    # ── OCREngine 구현 ─────────────────────────────────────────

    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        self.load()
        out_dir.mkdir(parents=True, exist_ok=True)
        s = self._settings
        try:
            outputs, _tokens = self._model.infer_multi(
                self._tokenizer,
                prompt=MULTI_PROMPT,
                image_files=[str(p) for p in image_paths],
                output_path=str(out_dir),
                image_size=1024,
                max_length=s.max_length,
                no_repeat_ngram_size=NGRAM_SIZE,
                ngram_window=MULTI_NGRAM_WINDOW,
                save_results=True,
                **self._gen_extras(sink, cancel, MULTI_NGRAM_WINDOW),
            )
        finally:
            self._release_device_cache()
        # 취소 시에도 부분 출력을 반환한다 — 병합 후 취소 처리는 runner 몫
        return outputs

    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        self.load()
        out_dir.mkdir(parents=True, exist_ok=True)
        s = self._settings
        try:
            outputs = self._model.infer(
                self._tokenizer,
                prompt=SINGLE_PROMPT,
                image_file=str(image_path),
                output_path=str(out_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=s.max_length,
                no_repeat_ngram_size=NGRAM_SIZE,
                ngram_window=SINGLE_NGRAM_WINDOW,
                save_results=True,
                **self._gen_extras(sink, cancel, SINGLE_NGRAM_WINDOW),
            )
        finally:
            self._release_device_cache()
        return outputs or ""
