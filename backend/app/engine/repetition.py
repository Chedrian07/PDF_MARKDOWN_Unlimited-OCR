"""OCR 생성 스트림의 숫자 변형 의미 반복 감지.

토큰 단위 no-repeat n-gram은 행 번호나 bbox 좌표가 매번 달라지면 동일한 문장을
새 시퀀스로 취급한다. 이 감지기는 완성된 행의 숫자를 정규화해 그 사각지대만
보완한다. 일반 표/목록을 끊지 않도록 한 페이지 안의 *연속된* 긴 행만 센다.
"""

from __future__ import annotations

import re

SEMANTIC_REPEAT_THRESHOLD = 24
_MIN_TEMPLATE_CHARS = 24
_MIN_ALPHA_CHARS = 8
_MAX_NUMBER_FIELDS = 2
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")
_LAYOUT_BLOCK = re.compile(
    r"<\|(?:ref|det)\|>.*?<\|/(?:ref|det)\|>",
    flags=re.IGNORECASE,
)
_PAGE_MARKER = "<PAGE>"


class SemanticRepetitionDetector:
    """증분 텍스트에서 숫자만 달라지는 동일 행의 연속 반복을 찾는다."""

    def __init__(self, threshold: int = SEMANTIC_REPEAT_THRESHOLD) -> None:
        if threshold < 2:
            raise ValueError("threshold는 2 이상이어야 합니다")
        self.threshold = threshold
        self.detected = False
        self.repeat_count = 0
        self.example = ""
        self._pending = ""
        self._template: str | None = None

    def feed(self, text: str, *, stream_end: bool = False) -> bool:
        """스트림 델타를 추가하고 반복 감지 여부를 반환한다.

        행이 여러 델타에 걸쳐 들어오는 TextStreamer 계약을 위해 마지막 미완성
        행을 보관한다. ``stream_end``일 때만 개행 없는 마지막 행도 검사한다.
        """
        if self.detected:
            return True
        if text:
            self._pending += text.replace("\r\n", "\n").replace("\r", "\n")
        lines = self._pending.split("\n")
        self._pending = lines.pop() if lines else ""
        for line in lines:
            self._observe_line(line)
            if self.detected:
                return True
        if stream_end and self._pending:
            final = self._pending
            self._pending = ""
            self._observe_line(final)
        return self.detected

    def _observe_line(self, line: str) -> None:
        # 한 델타/행 안에 페이지 마커와 다음 내용이 붙어 오는 경우도 처리한다.
        segments = line.split(_PAGE_MARKER)
        for index, segment in enumerate(segments):
            if index:
                self._reset_run()
            self._observe_segment(segment)
            if self.detected:
                return

    def _observe_segment(self, line: str) -> None:
        # 레이아웃 메타데이터의 ``text``/bbox는 실제 문서 내용이 아니다. 이를
        # 길이·문자 수에 포함하면 숫자뿐인 정상 표도 긴 문장으로 오인할 수 있다.
        content = _LAYOUT_BLOCK.sub("", line)
        compact = _WHITESPACE.sub(" ", content.strip()).casefold()
        if not compact:
            return

        # 이 방어선은 증가하는 행 번호/bbox처럼 기존 n-gram을 피하는 패턴만
        # 겨냥한다. 숫자가 없는 일반 반복 문장은 기존 no-repeat가 담당한다.
        if not _DIGITS.search(compact):
            self._reset_run()
            return
        # 여러 수치 열을 가진 실험 로그/표는 동일한 문장 틀이 정상이다. 이번
        # guard는 행 번호처럼 소수의 값만 바꿔 n-gram을 피하는 루프에 한정한다.
        if len(_DIGITS.findall(compact)) > _MAX_NUMBER_FIELDS:
            self._reset_run()
            return
        template = _DIGITS.sub("<n>", compact)
        alpha_chars = sum(char.isalpha() for char in template)
        if len(template) < _MIN_TEMPLATE_CHARS or alpha_chars < _MIN_ALPHA_CHARS:
            self._reset_run()
            return

        if template == self._template:
            self.repeat_count += 1
        else:
            self._template = template
            self.repeat_count = 1
        self.example = compact[:160]
        if self.repeat_count >= self.threshold:
            self.detected = True

    def _reset_run(self) -> None:
        self._template = None
        self.repeat_count = 0

    @property
    def message(self) -> str:
        detail = f" ({self.example})" if self.example else ""
        return f"의미상 동일한 OCR 행이 {self.repeat_count}회 반복되어 생성을 중단했습니다{detail}"
