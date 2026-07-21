"""OvisOCR2 raw 출력 파서 — 모델·vLLM 없이 임포트 가능한 순수 모듈 (표준 라이브러리만).

모델 카드 공식 출력 규약:
- figure: `<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />`,
  좌표는 [0, 1000) 정규화 정수 (공식 파서 정규식과 동일 형태만 인정)
- 표: HTML `<table>…</table>` / 수식: LaTeX / 나머지: 표준 Markdown
- 잘린 반복 suffix 정리: 모델 카드의 `_clean_truncated_repeats` 알고리즘

이 모듈은 모델 출력을 **비신뢰 입력**으로 취급한다:
- 좌표는 자릿수 제한(≤4자리) 정규식으로만 추출 — 경로 문자열은 절대 만들지 않는다
- 유효 figure 태그는 `[[FIGURE:n]]` placeholder로 치환 (파일명 결정은 메인 backend 몫)
- 그 외 모든 `<img …>` 태그는 제거 (외부 URL·경로 탈출·비정상 속성 무력화)
- figure 수·태그 길이·좌표 범위·중복·퇴화 bbox 검증
"""

from __future__ import annotations

import re

BBOX_MAX = 999          # 내부 프로토콜 정규화 상한 (0–999)
COORD_LIMIT = 1000      # 모델 출력 좌표 상한 [0, 1000) — 1000은 999로 clamp
MAX_FIGURES = 64
MIN_BBOX_SIDE = 2       # 정규화 단위 최소 변 — 미만은 퇴화 bbox로 거부
MAX_RAW_CHARS = 400_000

# 공식 규약 + 안전한 변형만: 공백 유연화, self-closing slash 생략 허용.
# \d{1,4}로 태그 길이가 상한되고, 경로는 숫자 4개 외 어떤 문자열도 매치되지 않는다.
FIGURE_TAG_RE = re.compile(
    r'<img\s+src="images/bbox_(\d{1,4})_(\d{1,4})_(\d{1,4})_(\d{1,4})\.jpg"\s*/?>'
)
# 유효 figure 추출 후 남은 모든 img 태그(외부 URL·트래버설·비정상 속성)는 제거
_ANY_IMG_TAG_RE = re.compile(r"<img\b[^>]{0,500}?/?>", re.IGNORECASE)
# 닫히지 않은 <img … (태그 종결 없이 줄이 끝나는 잔여물)도 정리
_UNCLOSED_IMG_RE = re.compile(r"<img\b[^>\n]{0,500}", re.IGNORECASE)


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """모델 카드 공식 반복 suffix 정리 알고리즘 (의미론 동일 구현).

    텍스트 끝에서 주기 1–200의 반복을 찾아, 5회 이상 & 100자 이상 반복이면
    한 주기 + 잘린 꼬리만 남긴다. 8000자 미만 텍스트는 건드리지 않는다.
    """
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]

    return text


def _validate_bbox(
    left: int, top: int, right: int, bottom: int
) -> tuple[int, int, int, int] | None:
    """[0,1000) 좌표 → [0,999] 정규화. 위반 시 None."""
    coords = (left, top, right, bottom)
    if any(v < 0 or v > COORD_LIMIT for v in coords):
        return None
    # [0,1000) 규약의 경계값 1000만 999로 clamp (0.1% 오차)
    left, top, right, bottom = (min(v, BBOX_MAX) for v in coords)
    if right - left < MIN_BBOX_SIDE or bottom - top < MIN_BBOX_SIDE:
        return None
    return (left, top, right, bottom)


def parse_page(raw: str) -> dict:
    """raw 모델 출력 → 프로토콜 page dict (markdown/blocks/warnings).

    markdown의 유효 figure 태그는 순서대로 `[[FIGURE:n]]`으로 치환되고,
    같은 순서로 image 블록(figure_index=n, bbox [0,999])이 생성된다.
    heading/본문/HTML 표/LaTeX/코드/목록은 그대로 보존된다.
    """
    warnings: list[str] = []
    if len(raw) > MAX_RAW_CHARS:
        warnings.append(f"모델 출력이 상한({MAX_RAW_CHARS}자)을 초과해 절단됨")
        raw = raw[:MAX_RAW_CHARS]

    blocks: list[dict] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    counter = {"n": 0}

    def _replace(m: re.Match) -> str:
        try:
            left, top, right, bottom = (int(g) for g in m.groups())
        except ValueError:  # pragma: no cover — \d 정규식상 불가, 방어적
            warnings.append("figure 태그 좌표 파싱 실패 — 제거")
            return ""
        bbox = _validate_bbox(left, top, right, bottom)
        if bbox is None:
            warnings.append(f"figure bbox 좌표 이상({left},{top},{right},{bottom}) — 제거")
            return ""
        if bbox in seen_boxes:
            warnings.append(f"중복 figure bbox{bbox} — 제거")
            return ""
        if counter["n"] >= MAX_FIGURES:
            warnings.append(f"figure 수 상한({MAX_FIGURES}) 초과 — 이후 태그 제거")
            return ""
        seen_boxes.add(bbox)
        n = counter["n"]
        counter["n"] += 1
        blocks.append({
            "type": "image",
            "bbox": list(bbox),
            "content": "",
            "order": n,
            "figure_index": n,
            "confidence": None,
        })
        return f"[[FIGURE:{n}]]"

    markdown = FIGURE_TAG_RE.sub(_replace, raw)

    # 유효 figure 외의 img 태그는 전부 제거 — 어떤 경로/URL도 통과시키지 않는다
    stripped = _ANY_IMG_TAG_RE.subn("", markdown)
    markdown = stripped[0]
    if stripped[1]:
        warnings.append(f"비정상 img 태그 {stripped[1]}개 제거")
    unclosed = _UNCLOSED_IMG_RE.subn("", markdown)
    markdown = unclosed[0]
    if unclosed[1]:
        warnings.append(f"닫히지 않은 img 태그 잔여물 {unclosed[1]}개 제거")

    # 반복 suffix 정리는 placeholder 치환 후 적용 (공식 순서: 태그 필터 → 정리).
    cleaned = clean_truncated_repeats(markdown)
    if len(cleaned) != len(markdown):
        warnings.append("잘린 반복 suffix 정리됨 (모델 카드 알고리즘)")
        markdown = cleaned
        # 정리로 placeholder가 사라진 figure는 본문 참조 없는 crop이 된다 —
        # 메인 backend materializer가 페이지 끝에 붙인다 (내용 손실 없음)

    return {
        "markdown": markdown.strip(),
        "blocks": blocks,
        "warnings": warnings,
    }
