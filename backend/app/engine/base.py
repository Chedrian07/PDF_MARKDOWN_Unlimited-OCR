"""OCR 엔진 인터페이스. 계약: docs/ARCHITECTURE.md §4

엔진 출력 형식(모델 업스트림 save_results 규약과 동일):
- run_multi: `<PAGE>` 마커로 페이지가 구분된 처리 완료 마크다운을 반환.
  figure는 out_dir/images/page_{청크내idx}_{k}.jpg 로 저장되고 마크다운에는
  ![](images/page_{i}_{k}.jpg) 참조가 들어감. 페이지별 레이아웃 오버레이는
  out_dir/result_with_boxes_{i}.jpg
- run_single: 단일 페이지 마크다운 반환. figure는 out_dir/images/{k}.jpg,
  오버레이는 out_dir/result_with_boxes.jpg
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class EngineError(RuntimeError):
    """엔진 실행 실패 (사용자에게 노출 가능한 메시지)."""


class RepetitiveOutputError(EngineError):
    """모델 생성이 반복에 빠지거나 페이지별 출력 상한을 넘어 조기 중단됨."""


class JobCanceled(Exception):
    """사용자 취소로 중단됨."""


class StreamSink(Protocol):
    def on_text(self, text: str) -> None:
        """모델이 생성한 텍스트 델타 (SSE token 이벤트로 전달됨)."""
        ...


class NullSink:
    def on_text(self, text: str) -> None:  # pragma: no cover - trivial
        pass


@dataclass(frozen=True)
class EngineCapabilities:
    """엔진 메타데이터·능력 선언 — runner의 청크 크기 결정과 health/Job 메타에 쓰인다.

    기본값은 기존 Unlimited-OCR 의미를 보존한다(하위 호환): 멀티페이지 문맥 지원,
    토큰 단위 스트리밍, 완전한 layout. 페이지 단위 sidecar 엔진은 이를 오버라이드한다.

    - preferred_chunk_size: None이면 settings.pages_per_chunk 사용.
    - stream_granularity: "token"(생성 토큰 델타) | "page"(페이지 완료 시 일괄).
    - layout_capability: "full"(텍스트+figure bbox) | "figure_only" | "none".
    """

    model_id: str = ""
    model_revision: str = ""
    provider: str = "in-process"
    supports_multi_page: bool = True
    preferred_chunk_size: int | None = None
    stream_granularity: str = "token"
    layout_capability: str = "full"
    figure_capability: bool = True


class OCREngine(abc.ABC):
    name: str = "base"
    device: str = "cpu"
    dtype_name: str = "float32"

    @property
    @abc.abstractmethod
    def loaded(self) -> bool: ...

    def capabilities(self) -> EngineCapabilities:
        """기본값 = 기존 Unlimited 의미 — fake/기존 테스트가 깨지지 않는다."""
        return EngineCapabilities()

    def provider_health(self) -> dict | None:
        """외부 provider(sidecar) 상태 — in-process 엔진은 None."""
        return None

    def drain_warnings(self) -> list[str]:
        """직전 실행에서 쌓인 사용자 노출용 경고를 꺼내고 비운다 (기본: 없음).

        runner가 청크마다 호출해 잡 warnings에 합친다 — 정화/절단으로 내용이
        빠졌는데 잡이 조용히 'done'이 되는 것을 막는다."""
        return []

    @abc.abstractmethod
    def load(self) -> None:
        """모델/리소스 로드 (멱등·스레드 세이프 — 동시 호출 시 한쪽만 로드, 나머지는 완료 대기).

        프리로드 스레드(main)와 워커 스레드(jobs)가 동시에 호출할 수 있다.
        """

    @abc.abstractmethod
    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str: ...

    @abc.abstractmethod
    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str: ...

    def gpu_name(self) -> str | None:
        return None
