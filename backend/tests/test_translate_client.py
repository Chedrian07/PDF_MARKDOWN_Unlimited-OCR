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


def test_잘림_chat_finish_reason_length_예산2배_재시도():
    """chat 출력이 length로 잘리면 max_tokens 2배로 1회 재시도한다."""
    calls = []

    def post(p, pl):
        calls.append(pl.get("max_tokens"))
        if len(calls) == 1:
            return (200, {"choices": [{"message": {"content": "잘린 절반"},
                                       "finish_reason": "length"}]}, {})
        return (200, {"choices": [{"message": {"content": "완전한 번역"},
                                   "finish_reason": "stop"}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "완전한 번역"
    assert calls == [100, 200]


def test_잘림_responses_incomplete_재시도():
    calls = []

    def post(p, pl):
        calls.append(pl.get("max_output_tokens"))
        if len(calls) == 1:
            return (200, {"status": "incomplete", "output_text": "부분"}, {})
        return (200, {"status": "completed", "output_text": "전체 번역"}, {})

    c = OpenAICompatClient(_cfg(api_mode="responses"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "전체 번역"
    assert calls == [100, 200]


def test_잘림_재시도도_잘리면_재시도_출력_반환():
    """2배 예산 후에도 잘리면 그 출력을 그대로 쓴다 — 이후는 래더가 흡수."""
    seq = iter([
        (200, {"choices": [{"message": {"content": "A"}, "finish_reason": "length"}]}, {}),
        (200, {"choices": [{"message": {"content": "AB"}, "finish_reason": "length"}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    assert c.complete("s", "u", max_tokens=100) == "AB"


def test_잘림_재시도_API오류면_잘린_첫출력_반환(caplog):
    """2배 재시도가 TranslateAPIError로 실패해도 잘린 첫 출력이 있으면 그것을 반환 — 래더가 흡수."""
    import logging

    seq = iter([
        (200, {"choices": [{"message": {"content": "잘린 절반"}, "finish_reason": "length"}]}, {}),
        (400, "bad request", {}),  # 2배 재시도 — 비재시도 상태코드로 즉시 TranslateAPIError
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "잘린 절반"
    assert any("2배 재시도 실패" in r.message for r in caplog.records)


def test_잘림_재시도_API오류_첫출력도_비면_예외전파():
    """첫 출력이 비어 있으면(전부 잘림) 재시도 실패 예외를 그대로 전파한다."""
    seq = iter([
        (200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, {}),
        (400, "bad request", {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with pytest.raises(TranslateAPIError, match="HTTP 400"):
        c.complete("s", "u", max_tokens=100)


def test_잘림_빈출력_reasoning_예산소진_재시도로_회복():
    """thinking이 예산을 다 먹어 content가 비어도 '빈 응답' 오류 대신 재시도."""
    seq = iter([
        (200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, {}),
        (200, {"choices": [{"message": {"content": "본문"}, "finish_reason": "stop"}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    assert c.complete("s", "u", max_tokens=100) == "본문"


def test_잘림_전부_빈출력이면_오류():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": ""},
                                                "finish_reason": "length"}]}, {})
    with pytest.raises(TranslateAPIError, match="잘렸습니다"):
        c.complete("s", "u", max_tokens=100)


def test_잘림_max_tokens_param_none이면_재시도_안함():
    """max_tokens를 안 보내는 설정에선 재시도해도 같은 요청 — 1회로 끝낸다."""
    calls = []

    def post(p, pl):
        calls.append(1)
        return (200, {"choices": [{"message": {"content": "부분 출력"},
                                   "finish_reason": "length"}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat", max_tokens_param="none"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "부분 출력"
    assert len(calls) == 1


def test_재시도_경로_warning_로그(caplog):
    """429 백오프·잘림 2배 재시도가 서버 로그에 warning으로 남는다 — 무기록 재시도 금지.
    본문·API 키는 로그에 남기지 않는다."""
    import logging

    seq = iter([
        (429, "느림", {"Retry-After": "0"}),
        (200, {"choices": [{"message": {"content": "부분"}, "finish_reason": "length"}]}, {}),
        (200, {"choices": [{"message": {"content": "완전"}, "finish_reason": "stop"}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "완전"
    msgs = [r.message for r in caplog.records]
    assert any("HTTP 429" in m and "Retry-After" in m for m in msgs)   # 백오프 warning
    assert any("잘림" in m and "100→200" in m for m in msgs)           # 잘림 재시도 warning
    joined = "\n".join(msgs)
    assert "sk-x" not in joined and "부분" not in joined               # 키·본문 무기록


def test_연결오류_재시도_warning_로그(caplog):
    import logging

    import requests as _requests

    calls = []

    def post(p, pl):
        calls.append(1)
        if len(calls) == 1:
            raise _requests.ConnectionError("boom")
        return (200, {"choices": [{"message": {"content": "성공"}}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    c._backoff = lambda headers, attempt: 0.0                          # 테스트 대기 제거
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "성공"
    msgs = [r.message for r in caplog.records]
    assert any("연결 오류(ConnectionError)" in m for m in msgs)


def test_연결오류_ChunkedEncodingError_재시도로_회복(caplog):
    """본문 수신 중 끊김(RequestException 계열, ConnectionError 비상속)도 재시도한다."""
    import logging

    import requests as _requests

    calls = []

    def post(p, pl):
        calls.append(1)
        if len(calls) == 1:
            raise _requests.exceptions.ChunkedEncodingError("본문 수신 중 끊김")
        return (200, {"choices": [{"message": {"content": "성공"}}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    c._backoff = lambda headers, attempt: 0.0                          # 테스트 대기 제거
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "성공"
    assert len(calls) == 2
    assert any("연결 오류(ChunkedEncodingError)" in r.message for r in caplog.records)


def test_연결오류_계속_실패면_TranslateAPIError_래핑():
    """재시도 예산 소진 시 기존 ConnectionError 경로와 동일하게 TranslateAPIError로 전파."""
    import requests as _requests

    calls = []

    def post(p, pl):
        calls.append(1)
        raise _requests.exceptions.ContentDecodingError("깨진 응답")

    c = OpenAICompatClient(_cfg(api_mode="chat", max_retries=1))
    c._post = post
    c._backoff = lambda headers, attempt: 0.0                          # 테스트 대기 제거
    with pytest.raises(TranslateAPIError, match="연결 실패"):
        c.complete("s", "u", max_tokens=100)
    assert len(calls) == 2                                             # 최초 1 + 재시도 1


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
