"""vLLM 기반 OvisOCR2 로더/추론 — 모델 카드 공식 예제(vLLM 0.22.1)를 따른다.

OpenAI 호환 서버를 별도로 띄우지 않고 sidecar 프로세스가 vLLM Python API로
모델을 직접 로드한다 (불필요한 이중 구조 회피 — docs/OVISOCR2_CUDA_5070TI.md).
"""

from __future__ import annotations

import logging
import threading

from .config import OvisConfig

logger = logging.getLogger(__name__)

# 모델 카드 공식 OCR 프롬프트 (선행 개행 포함, 원문 그대로)
OFFICIAL_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_'
    '{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as "
    "HTML: <table>...</table>. Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)


class OvisModel:
    """단일 프로세스·단일 모델. 추론은 락으로 직렬화한다 (max_num_seqs=1)."""

    def __init__(self, cfg: OvisConfig) -> None:
        self.cfg = cfg
        self._llm = None
        self._prompt: str | None = None
        self._sampling_cls = None
        self._infer_lock = threading.Lock()
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        """vLLM 엔진 로드 — 실패 시 load_error에 기록하고 예외 전파."""
        try:
            from vllm import LLM, SamplingParams

            cfg = self.cfg
            logger.info(
                "OvisOCR2 로딩: %s@%s (util=%.2f, max_len=%d, gdn=%s)",
                cfg.model_id, cfg.model_revision[:8] or "latest",
                cfg.gpu_memory_utilization, cfg.max_model_len, cfg.gdn_prefill_backend,
            )
            llm = LLM(
                model=cfg.model_id,
                revision=cfg.model_revision or None,
                tokenizer_revision=cfg.model_revision or None,
                tensor_parallel_size=1,
                dtype=cfg.dtype,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_model_len=cfg.max_model_len,
                max_num_seqs=cfg.max_num_seqs,
                # 컨슈머 Blackwell(sm_120)에서는 triton(FLA) 경로가 동작 경로 —
                # 모델 카드 공식 예제와 동일 인자
                gdn_prefill_backend=cfg.gdn_prefill_backend,
            )
            self._prompt = llm.get_tokenizer().apply_chat_template(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": OFFICIAL_PROMPT},
                    ],
                }],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self._sampling_cls = SamplingParams
            self._llm = llm
            self.load_error = None
            logger.info("OvisOCR2 로딩 완료")
        except Exception as e:
            self.load_error = f"{e.__class__.__name__}: {e}"[:500]
            logger.exception("OvisOCR2 로딩 실패")
            raise

    def infer(
        self, image, max_pixels: int | None = None, max_output_tokens: int | None = None
    ) -> str:
        """PIL 이미지 1장 → raw 마크다운 텍스트. OOM 처리(1회 해상도 강등)는 호출자 몫."""
        if self._llm is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        cfg = self.cfg
        params = self._sampling_cls(
            max_tokens=max_output_tokens or cfg.max_output_tokens, temperature=0.0
        )
        request = {
            "prompt": self._prompt,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "images_kwargs": {
                    "min_pixels": cfg.min_pixels,
                    "max_pixels": max_pixels or cfg.max_pixels,
                }
            },
        }
        with self._infer_lock:
            outputs = self._llm.generate([request], params, use_tqdm=False)
        return outputs[0].outputs[0].text

    @staticmethod
    def release_cache() -> None:
        """OOM 후 CUDA 캐시 반환 (best-effort)."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - 방어적
            pass
