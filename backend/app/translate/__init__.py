"""문서 한국어 번역 — OpenAI 호환 API (Chat Completions·Responses).

공개 API:
    from app.translate import run_translation, TranslateConfig, TranslateError
"""

from .engine import run_translation  # noqa: F401
from .types import (  # noqa: F401
    SUPPORTED_LANGS,
    TranslateAPIError,
    TranslateConfig,
    TranslateError,
    TranslateResult,
)
