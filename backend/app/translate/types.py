"""번역 파이프라인 공용 계약. 문서: docs/ARCHITECTURE.md §번역

디렉터리 계약 (job_dir 기준, lang은 BCP-47 소문자 예: "ko"):
  translations/{lang}/state.json    진행 상태 — 아래 write_state() 스키마
  translations/{lang}/glossary.json 문서 용어집 [{"src","ko","policy","first_unit"}]
  translations/{lang}/units.json    유닛 캐시 {cache_key: 번역문}
  translations/{lang}/report.json   품질 리포트 {"kept_original":[...],"retried":n,"skipped":n,...}
  result.{lang}.md                  번역된 마크다운 — page_separator 구조·페이지 수 보존
  layout.{lang}.json                blocks[].content만 교체된 layout.json (그 외 필드 동일)

설계 불변식:
  * 플레이스홀더(<m1 v="…"/> 형식) 복원 실패 유닛은 **원문 유지** — 내용을 잃지 않는다.
  * 캐시 키에 model·PROMPT_V·용어집 부분집합이 들어가 설정 변경 시 자동 재번역된다.
  * 이 패키지는 OCR 엔진/torch에 의존하지 않는다 (requests + 표준 라이브러리만).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

# 프롬프트/마스킹 규칙 개정 시 올린다 → 캐시 키가 바뀌어 자동 재번역 (fonts_v 패턴)
# v2: 문체 few-shot 예시 추가 (4B급 모델 합쇼체 이탈 실측 → 예시로 0건)
PROMPT_V = "2"

SUPPORTED_LANGS = ("ko",)


class TranslateError(RuntimeError):
    """번역 실패 — message는 사용자에게 그대로 보여줄 수 있는 한국어 문장."""


class TranslateAPIError(TranslateError):
    """업스트림 API 오류 (상태코드·본문 요약 포함)."""


def _clean(v: str | None) -> str:
    """env 값 정리 — 공백/따옴표 제거 (.env를 셸/compose 밖에서 읽었을 때 대비)."""
    return (v or "").strip().strip("'\"").strip()


@dataclass(frozen=True)
class TranslateConfig:
    base_url: str
    api_key: str
    model: str
    api_mode: str = "auto"  # auto | chat | responses
    concurrency: int = 4
    timeout_s: float = 180.0
    max_retries: int = 3
    temperature: str = "0"  # "none"이면 파라미터 생략
    max_tokens_param: str = "max_tokens"  # max_tokens | max_completion_tokens | none
    context: bool = True  # 직전 유닛 꼬리를 참고 컨텍스트로 프롬프트에 포함

    @classmethod
    def from_env(cls, env: dict | None = None) -> "TranslateConfig":
        e = os.environ if env is None else env
        base_url = _clean(e.get("OPENAI_BASE_URL"))
        model = _clean(e.get("TRANSLATE_MODEL")) or _clean(e.get("OPENAI_MODEL"))
        if not base_url or not model:
            raise TranslateError(
                "번역 프로바이더가 설정되지 않았습니다 — .env에 OPENAI_BASE_URL과 "
                "OPENAI_MODEL을 지정하세요 (.env.example 참조)"
            )
        mode = (_clean(e.get("TRANSLATE_API_MODE")) or "auto").lower()
        if mode not in ("auto", "chat", "responses"):
            raise TranslateError("TRANSLATE_API_MODE는 auto|chat|responses 중 하나여야 합니다")
        mt_param = (_clean(e.get("TRANSLATE_MAX_TOKENS_PARAM")) or "max_tokens").lower()
        if mt_param not in ("max_tokens", "max_completion_tokens", "none"):
            raise TranslateError(
                "TRANSLATE_MAX_TOKENS_PARAM은 max_tokens|max_completion_tokens|none 중 하나여야 합니다"
            )
        return cls(
            base_url=base_url,
            api_key=_clean(e.get("OPENAI_API_KEY")),
            model=model,
            api_mode=mode,
            concurrency=max(1, int(_clean(e.get("TRANSLATE_CONCURRENCY")) or 4)),
            timeout_s=max(5.0, float(_clean(e.get("TRANSLATE_TIMEOUT_S")) or 180)),
            max_retries=max(0, int(_clean(e.get("TRANSLATE_MAX_RETRIES")) or 3)),
            temperature=(_clean(e.get("TRANSLATE_TEMPERATURE")) or "0").lower(),
            max_tokens_param=mt_param,
            context=(_clean(e.get("TRANSLATE_CONTEXT")) or "1") not in ("0", "false", "no"),
        )


@dataclass
class TranslateResult:
    """run_translation 반환값 — report.json에도 같은 내용이 남는다."""

    status: str  # done | canceled
    total: int = 0  # 번역 대상 유닛 수 (skip 제외)
    translated: int = 0  # 이번 실행에서 API로 번역된 유닛
    cached: int = 0  # 캐시 적중 유닛
    kept_original: list[str] = field(default_factory=list)  # 복원 실패 → 원문 유지 유닛 id
    skipped: int = 0  # 정책상 번역 제외 (references·수식뿐인 블록 등)
    api_mode: str = ""  # 실제 사용된 모드 (auto가 확정된 결과)


def cache_key(masked_src: str, model: str, glossary_pairs: list[tuple[str, str]]) -> str:
    """유닛 캐시 키 — 원문(마스킹 후)·모델·프롬프트 버전·해당 유닛 용어집에 민감.

    용어집이 바뀌면 영향받는 유닛만 자연 무효화된다. glossary_pairs는
    (src, ko) 튜플 목록이며 순서 무관하도록 정렬해 해시한다.
    """
    h = hashlib.sha256()
    h.update(PROMPT_V.encode())
    h.update(b"\x1f")
    h.update(model.encode())
    h.update(b"\x1f")
    for s, k in sorted(glossary_pairs):
        h.update(s.encode())
        h.update(b"\x1e")
        h.update(k.encode())
        h.update(b"\x1f")
    h.update(masked_src.encode())
    return h.hexdigest()
