"""OpenAI 호환 클라이언트 — Chat Completions·Responses API 양쪽 지원.

로컬 서버(vLLM·Ollama·llama.cpp 등)는 어느 한쪽만 지원하는 경우가 많아 api_mode
"auto"는 첫 호출에 responses를 시도하고 404/405/501이면 chat으로 영구 래치한다.
requests만 쓰며(런타임 기존 의존성), 실제 전송은 _post 한 메서드로 모아 테스트가
그것만 몽키패치하도록 한다.
"""

from __future__ import annotations

import re
import time

import requests

from .types import TranslateAPIError, TranslateConfig

# 재시도 대상 상태코드 (일시적 오류)
_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})
# auto 모드에서 responses → chat 폴백을 유발하는 상태코드
_FALLBACK = frozenset({404, 405, 501})

_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)```\s*$", re.DOTALL)


class _NeedsFallback(Exception):
    """내부용 — auto 모드에서 responses가 미지원일 때 chat 폴백을 신호."""


def _normalize_base_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    # 빈 포트 교정: "https://host:/v1" → "https://host/v1" (사용자 .env 실측)
    return re.sub(r"^(https?://[^/]+?):(?=/|$)", r"\1", url)


class OpenAICompatClient:
    def __init__(self, cfg: TranslateConfig) -> None:
        self.cfg = cfg
        self.base_url = _normalize_base_url(cfg.base_url)
        self.session = requests.Session()
        self._latched: str | None = None  # auto 확정 모드 (인스턴스 수명 동안 유지)
        self.api_mode_used = "" if cfg.api_mode == "auto" else cfg.api_mode

    # ── 전송 심(seam) — 테스트는 이 메서드만 몽키패치 ──────────────────
    def _post(self, path: str, payload: dict) -> tuple[int, dict | str, dict]:
        """(status, body(json이면 dict 아니면 str), headers) 반환."""
        url = f"{self.base_url}/{path}"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        resp = self.session.post(url, json=payload, headers=headers, timeout=self.cfg.timeout_s)
        try:
            body: dict | str = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body, dict(resp.headers)

    # ── 공개 API ───────────────────────────────────────────────────
    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        mode, allow_fallback = self._mode_for_call()
        try:
            text = self._send(mode, system, user, max_tokens, allow_fallback)
        except _NeedsFallback:
            # responses 미지원 → chat으로 영구 래치, 같은 요청 재전송(재시도 미소모)
            self._latched = "chat"
            self.api_mode_used = "chat"
            return self._send("chat", system, user, max_tokens, allow_fallback=False)
        if self.cfg.api_mode == "auto" and self._latched is None:
            self._latched = mode
            self.api_mode_used = mode
        return text

    # ── 내부 ────────────────────────────────────────────────────────
    def _mode_for_call(self) -> tuple[str, bool]:
        if self.cfg.api_mode != "auto":
            return self.cfg.api_mode, False
        if self._latched is not None:
            return self._latched, False
        return "responses", True  # 첫 auto 호출 — responses 시도, 폴백 허용

    def _build_payload(self, mode: str, system: str, user: str, max_tokens: int) -> dict:
        cfg = self.cfg
        temp_ok = cfg.temperature != "none"
        if mode == "responses":
            p: dict = {"model": cfg.model, "instructions": system, "input": user}
            if temp_ok:
                p["temperature"] = float(cfg.temperature)
            if cfg.max_tokens_param != "none":
                p["max_output_tokens"] = max_tokens
            return p
        p = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if temp_ok:
            p["temperature"] = float(cfg.temperature)
        if cfg.max_tokens_param == "max_tokens":
            p["max_tokens"] = max_tokens
        elif cfg.max_tokens_param == "max_completion_tokens":
            p["max_completion_tokens"] = max_tokens
        return p

    def _send(self, mode: str, system: str, user: str, max_tokens: int, allow_fallback: bool) -> str:
        path = "responses" if mode == "responses" else "chat/completions"
        payload = self._build_payload(mode, system, user, max_tokens)
        attempt = 0
        while True:
            try:
                status, body, headers = self._post(path, payload)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.cfg.max_retries:
                    time.sleep(self._backoff({}, attempt))
                    attempt += 1
                    continue
                raise TranslateAPIError(f"번역 API 연결 실패: {e}") from e

            if status == 200:
                return self._parse(mode, body)
            if allow_fallback and status in _FALLBACK:
                raise _NeedsFallback()
            if status in (401, 403):
                raise TranslateAPIError("번역 API 인증 실패 — OPENAI_API_KEY를 확인하세요")
            if status == 404 and mode == "chat":
                raise TranslateAPIError(
                    "번역 API 엔드포인트 없음 — OPENAI_BASE_URL이 /v1까지 포함하는지 확인하세요"
                )
            if status in _RETRYABLE and attempt < self.cfg.max_retries:
                time.sleep(self._backoff(headers, attempt))
                attempt += 1
                continue
            raise TranslateAPIError(f"번역 API 오류 (HTTP {status}): {_body_preview(body)}")

    def _backoff(self, headers: dict, attempt: int) -> float:
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra is not None:
            try:
                return max(0.0, float(ra))
            except (TypeError, ValueError):
                pass
        return min(30.0, float(3 ** attempt))  # 1 → 3 → 9 → 27 → 30

    def _parse(self, mode: str, body: dict | str) -> str:
        if not isinstance(body, dict):
            raise TranslateAPIError(f"번역 API 응답 파싱 실패: {_body_preview(body)}")
        try:
            if mode == "responses":
                ot = body.get("output_text")
                if isinstance(ot, str) and ot.strip():
                    text = ot
                else:
                    text = _parse_responses_output(body.get("output", []))
            else:
                text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise TranslateAPIError(f"번역 API 응답 파싱 실패: {_body_preview(body)}") from e
        text = _postprocess(text)
        if not text:
            raise TranslateAPIError("번역 API가 빈 응답을 반환했습니다")
        return text


def _parse_responses_output(output) -> str:
    """responses output[] 순회 — reasoning은 건너뛰고 message의 output_text/text를 잇는다."""
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") == "reasoning":
            continue
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    t = c.get("text")
                    if isinstance(t, str):
                        parts.append(t)
    return "".join(parts)


def _postprocess(text) -> str:
    """선두 <think> 블록 제거, 전체 감싼 코드펜스 벗기기, strip."""
    if not isinstance(text, str):
        return ""
    text = _THINK_RE.sub("", text).strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text.strip()


def _body_preview(body: dict | str) -> str:
    s = body if isinstance(body, str) else str(body)
    return s[:200]
