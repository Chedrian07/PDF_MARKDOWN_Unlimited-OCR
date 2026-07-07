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
from pathlib import Path
from typing import Protocol


class EngineError(RuntimeError):
    """엔진 실행 실패 (사용자에게 노출 가능한 메시지)."""


class JobCanceled(Exception):
    """사용자 취소로 중단됨."""


class StreamSink(Protocol):
    def on_text(self, text: str) -> None:
        """모델이 생성한 텍스트 델타 (SSE token 이벤트로 전달됨)."""
        ...


class NullSink:
    def on_text(self, text: str) -> None:  # pragma: no cover - trivial
        pass


class OCREngine(abc.ABC):
    name: str = "base"
    device: str = "cpu"
    dtype_name: str = "float32"

    @property
    @abc.abstractmethod
    def loaded(self) -> bool: ...

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
