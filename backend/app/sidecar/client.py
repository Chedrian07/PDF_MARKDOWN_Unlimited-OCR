"""sidecar HTTP 클라이언트 — 동기(requests), 워커 스레드 전용.

설계 원칙:
- 타임아웃: 연결/읽기/health 분리. 무한 대기 없음.
- 응답 크기 상한: 스트리밍으로 계수하며 초과 즉시 중단 (response bomb 방어).
- 재시도: **연결 수립 실패에만** OCR_SIDECAR_RETRIES회. 읽기 중 실패·모델 오류는
  재시도하지 않는다 — 상위 runner의 청크 1회 재시도가 담당 (이중 재시도 방지).
- 취소: 요청을 헬퍼 스레드에서 돌리고 cancel 이벤트를 폴링, 취소 시 HTTP 연결을
  닫는다. sidecar 내부에서 이미 시작된 추론은 즉시 멈추지 않을 수 있다(한계 —
  docs/OCR_ENGINE_PROTOCOL.md §취소). 취소 후 도착한 결과는 폐기된다.
- 로그: 요청 id·크기·상태코드만. 문서 내용·이미지 바이트는 절대 남기지 않는다.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from pathlib import Path
from typing import Protocol

import requests
import urllib3

from ..engine.base import EngineError, JobCanceled
from .protocol import PROTOCOL_VERSION, ParseResponse, SidecarHealth

logger = logging.getLogger(__name__)

_HEALTH_MAX_BYTES = 1024 * 1024  # health 응답 상한 (1MB — 정상 응답은 수백 바이트)
_CHUNK = 64 * 1024
_RETRY_DELAY_S = 0.5

# 연결이 **수립되기 전** 실패한 urllib3 원인들 — 이때만 요청이 전달되지 않았음이 보장된다
_PRE_SEND_CAUSES = (
    urllib3.exceptions.NewConnectionError,     # ECONNREFUSED · DNS 실패(NameResolutionError 포함)
    urllib3.exceptions.ConnectTimeoutError,    # 3-way handshake 타임아웃
)


class CancelSignal(Protocol):
    """취소 신호 — threading.Event가 구조적으로 만족한다.

    SidecarEngine은 잡 취소와 청크 내부 실패를 OR로 합친 신호를 넘기므로
    구체 타입(threading.Event)이 아니라 이 프로토콜을 받는다."""

    def is_set(self) -> bool: ...


def is_pre_send_failure(error: BaseException) -> bool:
    """요청이 전달되기 **전** 실패인가 — 이 경우에만 재시도가 안전하다.

    이미 전달된 뒤의 실패(서버가 응답 도중 끊음 등)를 재시도하면 sidecar가 같은
    페이지를 두 번 추론한다(단일 GPU 직렬 처리라 그만큼 잡이 지연된다).
    requests의 ConnectTimeout은 ConnectionError의 서브클래스이기도 해서
    분류를 클래스 하나로 할 수 없다 — urllib3 원인 사슬을 따라간다.
    """
    if isinstance(error, requests.exceptions.ConnectTimeout):
        return True
    if not isinstance(error, requests.exceptions.ConnectionError):
        return False
    node: BaseException | None = (
        error.args[0] if error.args and isinstance(error.args[0], BaseException) else None
    )
    for _ in range(4):  # MaxRetryError → NewConnectionError 정도의 얕은 사슬
        if node is None:
            return False
        if isinstance(node, _PRE_SEND_CAUSES):
            return True
        node = getattr(node, "reason", None)
    return False


class SidecarError(EngineError):
    """sidecar 관련 실패의 공통 부모 (사용자 노출 가능 메시지)."""


class SidecarUnavailableError(SidecarError):
    """연결 실패/타임아웃 — provider가 죽었거나 재시작 중."""


class SidecarProtocolError(SidecarError):
    """응답이 프로토콜 계약을 위반 (잘못된 JSON/스키마/크기 초과/버전 불일치)."""


class SidecarClient:
    def __init__(
        self,
        base_url: str,
        *,
        engine_name: str,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 600.0,
        health_timeout_s: float = 5.0,
        max_response_mb: int = 20,
        retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.engine_name = engine_name
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.health_timeout_s = health_timeout_s
        self.max_response_bytes = max_response_mb * 1024 * 1024
        self.retries = retries
        self._session = requests.Session()
        self._session_lock = threading.Lock()

    # ── 세션 관리 ───────────────────────────────────────────────

    def _get_session(self) -> requests.Session:
        with self._session_lock:
            return self._session

    def _reset_session(self) -> None:
        """취소로 연결을 강제 종료한 뒤 재사용 가능한 새 세션으로 교체."""
        with self._session_lock:
            try:
                self._session.close()
            except Exception:  # pragma: no cover - 방어적
                pass
            self._session = requests.Session()

    def close(self) -> None:
        with self._session_lock:
            self._session.close()

    # ── health ─────────────────────────────────────────────────

    def health(self) -> SidecarHealth:
        url = f"{self.base_url}/health"
        try:
            resp = self._get_session().get(
                url, timeout=(self.connect_timeout_s, self.health_timeout_s), stream=True
            )
        except requests.exceptions.Timeout as e:
            raise SidecarUnavailableError("sidecar health 응답 시간 초과") from e
        except requests.exceptions.RequestException as e:
            raise SidecarUnavailableError(
                f"sidecar({self.base_url})에 연결할 수 없습니다 — 컨테이너 기동/프로필을 확인하세요"
            ) from e
        with resp:
            body = self._read_capped(resp, _HEALTH_MAX_BYTES)
            if resp.status_code != 200:
                raise SidecarUnavailableError(
                    f"sidecar health가 HTTP {resp.status_code}를 반환했습니다"
                )
        data = self._parse_json(body, context="health")
        try:
            h = SidecarHealth.model_validate(data)
        except Exception as e:
            raise SidecarProtocolError(f"sidecar health 스키마 위반: {e}") from e
        self._check_identity(h.protocol_version, h.engine, context="health")
        return h

    # ── parse ──────────────────────────────────────────────────

    def parse_page(
        self,
        image_path: Path,
        page_index: int,
        request_id: str,
        options: dict | None,
        cancel: CancelSignal | None,
    ) -> ParseResponse:
        """페이지 이미지 1장을 파싱. 실패 종류별로 구분된 예외를 던진다."""
        url = f"{self.base_url}/v1/parse"
        image_bytes = image_path.read_bytes()
        data = {
            "page_index": str(page_index),
            "request_id": request_id,
            "options": json.dumps(options or {}, ensure_ascii=False),
        }

        for attempt in range(self.retries + 1):
            if cancel is not None and cancel.is_set():
                raise JobCanceled()
            try:
                status, body = self._cancellable_post(
                    url, image_bytes, image_path.name, data, cancel
                )
                break
            except requests.exceptions.RequestException as e:
                # 요청 전달 **전** 실패만 재시도 — 전달 후 실패를 재시도하면 같은
                # 페이지가 sidecar에서 두 번 추론된다. 읽기 타임아웃·중도 절단은
                # 즉시 전파하고 상위 runner의 청크 재시도에 맡긴다.
                if not is_pre_send_failure(e) or attempt >= self.retries:
                    raise self._as_transport_error(e) from e
                logger.warning(
                    "sidecar 연결 실패 (req=%s, 시도 %d/%d) — 재시도",
                    request_id, attempt + 1, self.retries + 1,
                )
                time.sleep(_RETRY_DELAY_S)

        if status != 200:
            detail = self._error_detail(body)
            if status >= 500:
                raise SidecarError(f"sidecar 추론 실패 (HTTP {status}): {detail}")
            raise SidecarProtocolError(f"sidecar 요청 거부 (HTTP {status}): {detail}")

        parsed = self._parse_json(body, context="parse")
        try:
            resp = ParseResponse.model_validate(parsed)
        except Exception as e:
            raise SidecarProtocolError(f"sidecar 응답 스키마 위반: {e}") from e
        self._check_identity(resp.protocol_version, resp.engine, context="parse")
        logger.info(
            "sidecar parse 완료 (req=%s, page=%d, %.1fKB)",
            request_id, page_index, len(body) / 1024,
        )
        return resp

    # ── 내부 ───────────────────────────────────────────────────

    def _cancellable_post(
        self,
        url: str,
        image_bytes: bytes,
        filename: str,
        data: dict,
        cancel: CancelSignal | None,
    ) -> tuple[int, bytes]:
        """POST를 헬퍼 스레드에서 실행하고 취소를 0.2초 주기로 관측한다.

        취소가 관측되면 호출자를 즉시 풀어 주고(JobCanceled) 결과를 폐기한다.
        응답 헤더를 이미 받은 뒤라면 연결도 닫지만, **추론 중에는 헤더가 없어
        실제 절단이 불가능**하다 — 그 요청은 sidecar에서 완주한다
        (docs/OCR_ENGINE_PROTOCOL.md §취소 의미론과 한계)."""
        holder: dict = {}
        done = threading.Event()

        def _worker() -> None:
            try:
                resp = self._get_session().post(
                    url,
                    files={"file": (filename, image_bytes, "image/png")},
                    data=data,
                    stream=True,
                    timeout=(self.connect_timeout_s, self.read_timeout_s),
                )
                holder["resp"] = resp
                with resp:
                    holder["result"] = (
                        resp.status_code,
                        self._read_capped(resp, self.max_response_bytes),
                    )
            except BaseException as e:  # noqa: BLE001 — 본 스레드로 전달
                holder["error"] = e
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True, name="sidecar-request")
        t.start()
        while not done.wait(timeout=0.2):
            if cancel is not None and cancel.is_set():
                resp = holder.get("resp")
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:  # pragma: no cover - 방어적
                        pass
                self._reset_session()
                raise JobCanceled()
        if "error" in holder:
            e = holder["error"]
            # 전송 계층 예외(RequestException)는 원본 그대로 전파한다 — 재시도 가능
            # 여부(요청 전달 전/후) 판정과 메시지 변환은 parse_page가 한 곳에서 담당.
            raise e
        return holder["result"]

    def _as_transport_error(self, error: BaseException) -> SidecarError:
        """전송 계층 예외 → 사용자 노출 가능한 sidecar 오류 (원인 구분 유지)."""
        if isinstance(error, requests.exceptions.ConnectTimeout):
            return SidecarUnavailableError(
                f"sidecar 연결 시간 초과({self.connect_timeout_s:.0f}s) — "
                "컨테이너 기동/프로필을 확인하세요"
            )
        if isinstance(error, requests.exceptions.ReadTimeout):
            return SidecarUnavailableError(
                f"sidecar 응답 시간 초과({self.read_timeout_s:.0f}s) — "
                "페이지가 지나치게 크거나 provider가 멈췄을 수 있습니다"
            )
        if isinstance(error, requests.exceptions.ConnectionError):
            if is_pre_send_failure(error):
                return SidecarUnavailableError(
                    f"sidecar({self.base_url})에 연결할 수 없습니다 — "
                    "컨테이너 기동/프로필을 확인하세요"
                )
            return SidecarUnavailableError(
                "sidecar 연결이 요청 도중 끊겼습니다 (provider 재시작/크래시 가능) — "
                "docker compose logs로 확인하세요"
            )
        return SidecarUnavailableError(f"sidecar 요청 실패: {error.__class__.__name__}")

    @staticmethod
    def _read_capped(resp: requests.Response, limit: int) -> bytes:
        buf = bytearray()
        for chunk in resp.iter_content(_CHUNK):
            buf.extend(chunk)
            if len(buf) > limit:
                raise SidecarProtocolError(
                    f"sidecar 응답이 상한({limit // (1024 * 1024)}MB)을 초과했습니다"
                )
        return bytes(buf)

    @staticmethod
    def _parse_json(body: bytes, context: str) -> dict:
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SidecarProtocolError(
                f"sidecar {context} 응답이 유효한 JSON이 아닙니다 (잘린 응답일 수 있음)"
            ) from e
        if not isinstance(data, dict):
            raise SidecarProtocolError(f"sidecar {context} 응답이 JSON 객체가 아닙니다")
        return data

    def _check_identity(self, protocol_version: int, engine: str, context: str) -> None:
        if protocol_version != PROTOCOL_VERSION:
            raise SidecarProtocolError(
                f"sidecar 프로토콜 버전 불일치 ({context}: {protocol_version} ≠ {PROTOCOL_VERSION})"
            )
        if engine != self.engine_name:
            raise SidecarProtocolError(
                f"sidecar 엔진 불일치 ({context}: '{engine}' ≠ '{self.engine_name}') — "
                "OCR_SIDECAR_URL이 다른 엔진 컨테이너를 가리키고 있습니다"
            )

    @staticmethod
    def _error_detail(body: bytes) -> str:
        try:
            data = json.loads(body.decode("utf-8"))
            if isinstance(data, dict):
                d = data.get("detail") or data.get("error") or ""
                if isinstance(d, str) and d:
                    return d[:300]
        except Exception:  # noqa: BLE001 — 오류 본문은 best-effort
            pass
        return "(상세 없음)"
