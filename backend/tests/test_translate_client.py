"""클라이언트 — chat/responses 파싱·auto 폴백·재시도·오류·후처리·URL 정규화."""

import pytest

from app.translate.client import OpenAICompatClient, _normalize_base_url
from app.translate.types import TranslateAPIError, TranslateConfig


def _cfg(**kw) -> TranslateConfig:
    base = dict(
        base_url="https://host/v1", api_key="sk-x", model="m",
        api_mode="auto", max_retries=3, temperature="0", max_tokens_param="max_tokens",
    )
    base.update(kw)
    return TranslateConfig(**base)


def test_base_url_정규화():
    assert _normalize_base_url("https://host:/v1") == "https://host/v1"   # 빈 포트 교정
    assert _normalize_base_url("  https://host/v1/ ") == "https://host/v1"  # strip + 끝 /
    assert _normalize_base_url("https://host:8080/v1") == "https://host:8080/v1"  # 실 포트 보존


def test_chat_파싱():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": "안녕하세요"}}]}, {})
    assert c.complete("s", "u", max_tokens=100) == "안녕하세요"
    assert c.api_mode_used == "chat"


def test_responses_output_text():
    c = OpenAICompatClient(_cfg(api_mode="responses"))
    c._post = lambda p, pl: (200, {"output_text": "응답 텍스트"}, {})
    assert c.complete("s", "u", max_tokens=100) == "응답 텍스트"


def test_responses_output_배열_reasoning_스킵():
    c = OpenAICompatClient(_cfg(api_mode="responses"))
    body = {"output": [
        {"type": "reasoning", "content": [{"type": "text", "text": "무시"}]},
        {"type": "message", "content": [
            {"type": "output_text", "text": "앞"},
            {"type": "text", "text": "뒤"},
        ]},
    ]}
    c._post = lambda p, pl: (200, body, {})
    assert c.complete("s", "u", max_tokens=100) == "앞뒤"


def test_responses_output_text_빈문자열이면_배열로():
    c = OpenAICompatClient(_cfg(api_mode="responses"))
    body = {"output_text": "   ", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "배열본문"}]},
    ]}
    c._post = lambda p, pl: (200, body, {})
    assert c.complete("s", "u", max_tokens=100) == "배열본문"


def test_auto_404_chat_래치():
    paths = []

    def post(p, pl):
        paths.append(p)
        if p == "responses":
            return (404, "not found", {})
        return (200, {"choices": [{"message": {"content": "챗"}}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="auto"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "챗"
    assert paths == ["responses", "chat/completions"]
    assert c.api_mode_used == "chat"
    # 이후 호출은 chat 직행 (영구 래치)
    paths.clear()
    c.complete("s", "u", max_tokens=100)
    assert paths == ["chat/completions"]


def test_auto_responses_성공시_래치():
    c = OpenAICompatClient(_cfg(api_mode="auto"))
    c._post = lambda p, pl: (200, {"output_text": "ok"}, {})
    c.complete("s", "u", max_tokens=100)
    assert c.api_mode_used == "responses"


def test_429_retry_after_재시도():
    seq = iter([
        (429, "느림", {"Retry-After": "0"}),
        (200, {"choices": [{"message": {"content": "성공"}}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    assert c.complete("s", "u", max_tokens=100) == "성공"


def test_401_인증실패_메시지():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (401, "unauthorized", {})
    with pytest.raises(TranslateAPIError, match="인증 실패"):
        c.complete("s", "u", max_tokens=100)


def test_chat_404_엔드포인트_메시지():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (404, "x", {})
    with pytest.raises(TranslateAPIError, match="엔드포인트 없음"):
        c.complete("s", "u", max_tokens=100)


def test_think_스트립_코드펜스_벗기기():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    content = "<think>추론 과정</think>\n```\n최종 번역문\n```"
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": content}}]}, {})
    assert c.complete("s", "u", max_tokens=100) == "최종 번역문"


def test_빈응답_오류():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": "   "}}]}, {})
    with pytest.raises(TranslateAPIError, match="빈 응답"):
        c.complete("s", "u", max_tokens=100)


def test_temperature_max_tokens_생략():
    c = OpenAICompatClient(_cfg(api_mode="chat", temperature="none", max_tokens_param="none"))
    captured = {}

    def post(p, pl):
        captured.update(pl)
        return (200, {"choices": [{"message": {"content": "x"}}]}, {})

    c._post = post
    c.complete("s", "u", max_tokens=100)
    assert "temperature" not in captured
    assert "max_tokens" not in captured and "max_completion_tokens" not in captured


def test_max_completion_tokens_파라미터():
    c = OpenAICompatClient(_cfg(api_mode="chat", max_tokens_param="max_completion_tokens"))
    captured = {}

    def post(p, pl):
        captured.update(pl)
        return (200, {"choices": [{"message": {"content": "x"}}]}, {})

    c._post = post
    c.complete("s", "u", max_tokens=512)
    assert captured["max_completion_tokens"] == 512 and "max_tokens" not in captured


def test_reasoning_effort별_max_tokens_예산():
    """effort별 요청 max_tokens 테이블 (사용자 확정값) + xhigh 모드 지원."""
    from app.translate.types import REASONING_MAX_TOKENS, TranslateConfig

    expect = {"": 8192, "off": 8192, "low": 10240, "medium": 20480, "high": 40960, "xhigh": 81920}
    assert REASONING_MAX_TOKENS == expect
    for mode, budget in expect.items():
        cfg = TranslateConfig(base_url="https://h/v1", api_key="", model="m", reasoning=mode)
        assert cfg.max_output_tokens == budget

    # from_env가 xhigh를 허용하고 payload에 effort로 실림
    cfg = TranslateConfig.from_env({
        "OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m",
        "TRANSLATE_REASONING": "xhigh", "TRANSLATE_API_MODE": "chat",
    })
    assert cfg.reasoning == "xhigh" and cfg.max_output_tokens == 81920
    from app.translate.client import OpenAICompatClient
    p = OpenAICompatClient(cfg)._build_payload("chat", "s", "u", cfg.max_output_tokens)
    assert p["reasoning"] == {"effort": "xhigh"} and p["max_tokens"] == 81920
