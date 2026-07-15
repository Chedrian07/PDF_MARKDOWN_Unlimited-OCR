"""OCR 생성 스트림의 의미 반복과 페이지별 출력 폭주 감지.

토큰 단위 no-repeat n-gram은 행 번호나 bbox 좌표가 매번 달라지거나 같은 짧은
패턴이 하나의 긴 행 안에서 이어지면 루프를 놓칠 수 있다. 이 감지기는 기존의
행 템플릿 검사에 개행과 무관한 bounded rolling 검사를 더하고, 페이지별 문자·
토큰 예산을 마지막 안전장치로 둔다. 모든 버퍼는 고정 상한을 가져 장시간 생성
자체가 감지기의 메모리를 늘리지 않는다.
"""

from __future__ import annotations

import re

# 행 템플릿 임계 — 숫자만 다른 동일 행이 이만큼 이어져야 루프로 판정한다.
# 24는 합법 열거형 문서(설문지·법령 서식 등 '1. 매우 그렇다' 류 24행)가 걸리는
# 오탐이 실증되어 64로 상향 — 폭주 루프는 수백 회 반복하므로 재현율 손실이 없고,
# 한 페이지에 64행 이상 동일 템플릿은 사실상 병리적이다.
SEMANTIC_REPEAT_THRESHOLD = 64
MAX_PAGE_OUTPUT_CHARS = 16_384
MAX_PAGE_OUTPUT_TOKENS = 6_144
# 마커 폭주 여유 배수 — 기대 페이지 수의 이 배수(최소 +2)를 넘는 <PAGE> 마커는
# 루프로 판정한다. 마커가 낀 반복 루프는 마커마다 페이지 예산이 리셋되어 다른
# 채널을 전부 우회하므로 마커 수 자체를 예산에 포함해야 한다.
_PAGE_FLOOD_FACTOR = 2

_MIN_TEMPLATE_CHARS = 24
_MIN_ALPHA_CHARS = 8
_MAX_NUMBER_FIELDS = 2
_ROLLING_BUFFER_CHARS = 4_096
_ROLLING_CHECK_INTERVAL = 32
_ROLLING_MAX_UNIT_ATOMS = 64
_ROLLING_MIN_REPEATS = 24
_ROLLING_MIN_SPAN_CHARS = 3_072
_COUNTER_REPEAT_THRESHOLD = 20
_COUNTER_MAX_ALPHA_RATIO = 0.05

_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")
_ATOM = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_LAYOUT_BLOCK = re.compile(
    r"<\|(?:ref|det)\|>.*?<\|/(?:ref|det)\|>",
    flags=re.IGNORECASE | re.DOTALL,
)
_INCOMPLETE_LAYOUT_BLOCK = re.compile(
    r"<\|(?:ref|det)\|>[^\n]*\Z",
    flags=re.IGNORECASE,
)
_COUNTER = re.compile(r"#\s*(\d{1,7})\s*\.")
_PAGE_MARKER = "<PAGE>"


class SemanticRepetitionDetector:
    """증분 텍스트에서 반복 생성과 페이지 출력 예산 초과를 찾는다."""

    def __init__(
        self,
        threshold: int = SEMANTIC_REPEAT_THRESHOLD,
        *,
        max_page_chars: int | None = MAX_PAGE_OUTPUT_CHARS,
        max_page_tokens: int | None = MAX_PAGE_OUTPUT_TOKENS,
        expected_pages: int | None = None,
    ) -> None:
        if threshold < 2:
            raise ValueError("threshold는 2 이상이어야 합니다")
        if max_page_chars is not None and max_page_chars < 1:
            raise ValueError("max_page_chars는 1 이상이어야 합니다")
        if max_page_tokens is not None and max_page_tokens < 1:
            raise ValueError("max_page_tokens는 1 이상이어야 합니다")
        if expected_pages is not None and expected_pages < 1:
            raise ValueError("expected_pages는 1 이상이어야 합니다")
        self.threshold = threshold
        self.max_page_chars = max_page_chars
        self.max_page_tokens = max_page_tokens
        self.expected_pages = expected_pages
        # 마커 폭주 상한 — 정상 변이(마커 1~2개 초과 생성)는 merge의 불일치 보정이
        # 흡수하므로 여유를 두고, 그 이상은 마커가 낀 루프로 판정한다.
        self._max_markers = (
            None
            if expected_pages is None
            else max(expected_pages + 2, expected_pages * _PAGE_FLOOD_FACTOR)
        )
        self.detected = False
        self.reason: str | None = None
        self.repeat_count = 0
        self.example = ""
        self.page_index = 0
        self.page_chars = 0
        self.page_tokens = 0

        self._pending_line = ""
        self._marker_tail = ""
        self._template: str | None = None
        self._rolling = ""
        self._rolling_since_check = 0
        self._saw_page_marker = False
        self._message = ""

    def feed_tokens(self, count: int) -> bool:
        """스트리머가 받은 신규 생성 토큰 수를 현재 페이지 예산에 반영한다."""
        if self.detected or count <= 0:
            return self.detected
        self.page_tokens += count
        if self.max_page_tokens is not None and self.page_tokens > self.max_page_tokens:
            self._trip_page_limit("page_token_limit", self.page_tokens, "토큰", self.max_page_tokens)
        return self.detected

    def feed(self, text: str, *, stream_end: bool = False) -> bool:
        """스트림 델타를 추가하고 반복/출력 상한 감지 여부를 반환한다.

        ``<PAGE>``가 델타 경계에서 쪼개져도 페이지별 상태를 정확히 초기화한다.
        행 기반 검사는 완성 행을 보고, rolling 검사는 개행을 제거한 atom suffix를
        보므로 마지막 행이 끝나기 전에도 반복을 중단할 수 있다.
        """
        if self.detected:
            return True
        data = self._marker_tail + (text or "").replace("\r\n", "\n").replace("\r", "\n")
        self._marker_tail = ""

        while not self.detected:
            marker_at = data.find(_PAGE_MARKER)
            if marker_at < 0:
                break
            self._consume(data[:marker_at])
            if self.detected:
                return True
            self._finish_line()
            self._start_page()
            data = data[marker_at + len(_PAGE_MARKER) :]

        if not self.detected:
            keep = self._partial_marker_suffix(data)
            if keep:
                self._marker_tail = data[-keep:]
                data = data[:-keep]
            self._consume(data)

        if stream_end and not self.detected:
            if self._marker_tail:
                tail = self._marker_tail
                self._marker_tail = ""
                self._consume(tail)
            self._finish_line()
            self._check_rolling(force=True)
        return self.detected

    @staticmethod
    def _partial_marker_suffix(text: str) -> int:
        upper = min(len(text), len(_PAGE_MARKER) - 1)
        for size in range(upper, 0, -1):
            if text.endswith(_PAGE_MARKER[:size]):
                return size
        return 0

    def _consume(self, content: str) -> None:
        if not content or self.detected:
            return
        self.page_chars += len(content)
        self._rolling = (self._rolling + content)[-_ROLLING_BUFFER_CHARS:]
        self._rolling_since_check += len(content)

        line_data = self._pending_line + content
        lines = line_data.split("\n")
        self._pending_line = lines.pop()[-_ROLLING_BUFFER_CHARS:] if lines else ""
        for line in lines:
            self._observe_line(line)
            if self.detected:
                return

        self._check_rolling()
        if (
            not self.detected
            and self.max_page_chars is not None
            and self.page_chars > self.max_page_chars
        ):
            self._trip_page_limit(
                "page_char_limit", self.page_chars, "문자", self.max_page_chars
            )

    def _finish_line(self) -> None:
        if self._pending_line and not self.detected:
            final = self._pending_line
            self._pending_line = ""
            self._observe_line(final)

    def _observe_line(self, line: str) -> None:
        # 레이아웃 메타데이터의 ``text``/bbox는 실제 문서 내용이 아니다.
        content = _LAYOUT_BLOCK.sub("", line)
        compact = _WHITESPACE.sub(" ", content.strip()).casefold()
        if not compact:
            return

        # 증가하는 행 번호처럼 기존 n-gram을 피하는 패턴에 한정한다. 여러 수치
        # 열을 가진 정상 표/실험 로그는 rolling exact 검사에서도 단위가 변하므로
        # 여기서는 기존처럼 제외한다.
        numbers = _DIGITS.findall(compact)
        if not numbers or len(numbers) > _MAX_NUMBER_FIELDS:
            self._reset_line_run()
            return
        template = _DIGITS.sub("<n>", compact)
        alpha_chars = sum(char.isalpha() for char in template)
        if len(template) < _MIN_TEMPLATE_CHARS or alpha_chars < _MIN_ALPHA_CHARS:
            self._reset_line_run()
            return

        if template == self._template:
            self.repeat_count += 1
        else:
            self._template = template
            self.repeat_count = 1
        self.example = compact[:160]
        if self.repeat_count >= self.threshold:
            self.reason = "line_repeat"
            self.detected = True
            self._message = (
                f"의미상 동일한 OCR 행이 {self.repeat_count}회 반복되어 생성을 중단했습니다"
                f" ({self.example})"
            )

    def _check_rolling(self, *, force: bool = False) -> None:
        if self.detected:
            return
        if not force and self._rolling_since_check < _ROLLING_CHECK_INTERVAL:
            return
        self._rolling_since_check = 0
        content = _LAYOUT_BLOCK.sub("", self._rolling)
        content = _INCOMPLETE_LAYOUT_BLOCK.sub("", content)
        if not content.strip():
            return

        counter = self._counter_loop(content)
        if counter is not None:
            count, example = counter
            self._trip_rolling("순차 번호", count, example)
            return

        tandem = self._exact_tandem(content)
        if tandem is not None:
            count, example = tandem
            self._trip_rolling("token", count, example)

    @staticmethod
    def _counter_loop(content: str) -> tuple[int, str] | None:
        matches = list(_COUNTER.finditer(content))
        if len(matches) < _COUNTER_REPEAT_THRESHOLD:
            return None
        run_start = 0
        for index in range(1, len(matches) + 1):
            continues = (
                index < len(matches)
                and int(matches[index].group(1)) == int(matches[index - 1].group(1)) + 1
            )
            if continues:
                continue
            run_count = index - run_start
            if run_count >= _COUNTER_REPEAT_THRESHOLD:
                start = matches[run_start].start()
                end = matches[index - 1].end()
                snippet = content[start:end]
                alpha = sum(char.isalpha() for char in snippet)
                if alpha / max(len(snippet), 1) <= _COUNTER_MAX_ALPHA_RATIO:
                    return run_count, _WHITESPACE.sub(" ", snippet[:160]).strip()
            run_start = index
        return None

    @staticmethod
    def _exact_tandem(content: str) -> tuple[int, str] | None:
        # 완성 행 반복은 semantic line guard/no-repeat n-gram이 담당한다. exact
        # rolling 채널은 개행을 기다리지 못하는 긴 현재 행의 suffix만 검사해 정상
        # 표의 반복 셀/상태값(N/A, Yes 등)을 페이지 전체로 이어 붙이지 않는다.
        content = content.rsplit("\n", 1)[-1]
        atoms = _ATOM.findall(content)
        max_unit = min(_ROLLING_MAX_UNIT_ATOMS, len(atoms) // _ROLLING_MIN_REPEATS)
        for unit_size in range(1, max_unit + 1):
            unit = atoms[-unit_size:]
            has_alnum = any(any(char.isalnum() for char in atom) for atom in unit)
            has_alpha = any(any(char.isalpha() for char in atom) for atom in unit)
            has_punctuation = any(not atom.isalnum() and atom != "_" for atom in unit)
            # 숫자만 반복되는 벡터/행렬은 정상 데이터일 수 있다. 실제 p27의
            # ``3 .``처럼 구조 구두점이 붙거나 단어가 반복될 때만 exact 채널을 쓴다.
            if "|" in unit or not has_alnum or not (has_alpha or has_punctuation):
                continue
            repeats = 1
            cursor = len(atoms) - 2 * unit_size
            while cursor >= 0 and atoms[cursor : cursor + unit_size] == unit:
                repeats += 1
                cursor -= unit_size
            if repeats < _ROLLING_MIN_REPEATS:
                continue
            span = " ".join(atoms[-repeats * unit_size :])
            if len(span) < _ROLLING_MIN_SPAN_CHARS:
                continue
            return repeats, " ".join(unit)[:160]
        return None

    def _trip_rolling(self, label: str, count: int, example: str) -> None:
        self.reason = "rolling_repeat"
        self.detected = True
        self.repeat_count = count
        self.example = example
        detail = f" ({example})" if example else ""
        self._message = (
            f"개행과 무관한 rolling {label} 패턴이 {count}회 반복되어 생성을 중단했습니다"
            f"{detail}"
        )

    def _trip_page_limit(self, reason: str, amount: int, unit: str, limit: int) -> None:
        self.reason = reason
        self.detected = True
        self._message = (
            f"청크 내 {self.page_index + 1}페이지 생성량이 {unit} 상한({limit:,})을 "
            f"초과해 중단했습니다 (관측 {amount:,}{unit})"
        )

    def _start_page(self) -> None:
        # multi 출력은 보통 leading <PAGE>로 시작한다. 첫 leading marker는 1페이지
        # 선언이므로 index를 올리지 않고, 이후 marker부터 다음 페이지로 이동한다.
        if self._saw_page_marker:
            self.page_index += 1
        self._saw_page_marker = True
        # 마커 폭주 검사 — 마커가 낀 루프는 페이지 예산이 마커마다 리셋되어 아래
        # 채널을 전부 우회하므로, 마커 수 자체가 기대치를 크게 넘으면 중단한다.
        if self._max_markers is not None and self.page_index + 1 > self._max_markers:
            self.reason = "page_flood"
            self.detected = True
            self._message = (
                f"<PAGE> 마커가 기대 페이지 수({self.expected_pages})를 크게 초과해 "
                f"{self.page_index + 1}개째 생성되어 중단했습니다"
            )
            return
        self.page_chars = 0
        self.page_tokens = 0
        self._pending_line = ""
        self._rolling = ""
        self._rolling_since_check = 0
        self._reset_line_run()

    def _reset_line_run(self) -> None:
        self._template = None
        self.repeat_count = 0

    @property
    def message(self) -> str:
        return self._message or "OCR 반복 출력이 감지되어 생성을 중단했습니다"
