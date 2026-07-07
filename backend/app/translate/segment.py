"""번역 단위(유닛) 분리·재조립 — 마크다운과 레이아웃 두 소스.

마크다운은 페이지 구분자로 나눈 뒤 페이지별로 markdown-it 블록 토큰의 줄 범위를
유닛으로 삼는다. 재조립은 **원문 바이트를 최대한 보존**한다: 유닛 줄 범위만
번역문으로 교체하고 나머지(빈 줄·수평선 등)는 그대로 둔다.

핵심 골든 불변식:
  translations가 모든 유닛을 unit.src 그대로 매핑하면
  assemble_markdown 출력은 원본 md와 **바이트 동일**하다.

references 섹션은 skip_reason="references"로 표시해 번역에서 제외한다(문서 끝까지,
같은 레벨 이하의 다음 heading 전까지).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from markdown_it import MarkdownIt

# 세그먼트 전용 파서 — commonmark + table (render.py와 별개 인스턴스, dollarmath 불필요:
# 수식은 마스킹이 처리하고 여기선 블록 줄 범위만 필요).
_md = MarkdownIt("commonmark").enable("table")

# level 0 블록 오프너 → 유닛 kind
_OPENERS = {
    "paragraph_open": "paragraph",
    "heading_open": "heading",
    "table_open": "table",
    "fence": "fence",
    "html_block": "html",
    "blockquote_open": "blockquote",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
}

_REF_HEADING_RE = re.compile(r"(?i)^(references?|bibliography|acknowledg\w*)$")
_HR_LINE_RE = re.compile(r"^\s*-{3,}\s*$")


@dataclass
class Unit:
    id: str  # "md:{page}:{i}" | "lay:{page}:{i}"
    kind: str
    page: int
    src: str
    skip_reason: str = ""


def _page_blocks(page_text: str) -> list[dict]:
    """한 페이지의 level-0 블록들 → [{i, kind, s, e, level?, text?}] (문서 순서).

    i는 페이지 내 블록 인덱스(유닛 id에 사용), [s,e)는 0-based 줄 반열림 범위.
    heading은 level(int)과 inline 텍스트를 함께 싣는다(references 판별용).
    """
    tokens = _md.parse(page_text)
    blocks: list[dict] = []
    i = 0
    for idx, t in enumerate(tokens):
        if t.level != 0 or t.type not in _OPENERS or not t.map:
            continue
        b = {"i": i, "kind": _OPENERS[t.type], "s": t.map[0], "e": t.map[1]}
        if t.type == "heading_open":
            tag = t.tag[1:]
            b["level"] = int(tag) if tag.isdigit() else 1
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            b["text"] = nxt.content if nxt is not None and nxt.type == "inline" else ""
        blocks.append(b)
        i += 1
    return blocks


def _mark_references(annotated: list[tuple[Unit, dict]]) -> None:
    """references/bibliography/acknowledgments heading부터 같은 레벨 이하의 다음
    heading 전까지 skip_reason="references"로 표시(문서 전역, 페이지 넘나듦)."""
    ref_level: int | None = None
    for unit, b in annotated:
        if unit.kind == "heading":
            level = b.get("level", 1)
            # 활성 references 구간을 닫는 heading(같은 레벨 이하 = 레벨 번호 ≤ 기준)
            if ref_level is not None and level <= ref_level:
                ref_level = None
            htext = (b.get("text") or "").strip().strip("#").strip()
            if _REF_HEADING_RE.match(htext):
                ref_level = level
                unit.skip_reason = "references"
                continue
        if ref_level is not None:
            unit.skip_reason = "references"


def split_markdown(md_text: str, page_separator: str) -> list[Unit]:
    """result.md를 페이지별 블록 유닛으로 분리(문서 순서)."""
    pages = md_text.split(page_separator)
    annotated: list[tuple[Unit, dict]] = []
    for page_idx, page in enumerate(pages):
        lines = page.split("\n")
        for b in _page_blocks(page):
            src = "\n".join(lines[b["s"]:b["e"]])
            unit = Unit(id=f"md:{page_idx}:{b['i']}", kind=b["kind"], page=page_idx, src=src)
            annotated.append((unit, b))
    _mark_references(annotated)
    return [u for u, _ in annotated]


def _sanitize_unit(text: str) -> str:
    """번역문 새니타이즈 — 페이지 구분자 오염 방지.

    유닛 내부의 `---`(3+ 대시만 있는 줄)를 "⸻"로 바꾸고 앞뒤 빈 줄을 제거한다.
    (identity 케이스에서 유닛 src는 대시 전용 줄·앞뒤 빈 줄을 포함하지 않으므로 무변화.)
    """
    lines = text.split("\n")
    lines = ["⸻" if _HR_LINE_RE.match(ln) else ln for ln in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def assemble_markdown(md_text: str, page_separator: str, translations: dict[str, str]) -> str:
    """원문에서 유닛 줄 범위만 번역문으로 교체(페이지별 뒤→앞), 나머지 보존.

    페이지 수가 원본과 달라지면 ValueError(최후 방어). 새니타이즈와 유닛 단위
    page_separator 검사가 선방어한다.
    """
    pages = md_text.split(page_separator)
    out_pages: list[str] = []
    for page_idx, page in enumerate(pages):
        lines = page.split("\n")
        # 뒤에서 앞으로 교체 → 앞선 유닛의 줄 인덱스가 밀리지 않는다
        for b in sorted(_page_blocks(page), key=lambda x: x["s"], reverse=True):
            uid = f"md:{page_idx}:{b['i']}"
            if uid not in translations:
                continue
            new_text = _sanitize_unit(translations[uid])
            if page_separator and page_separator in new_text:
                continue  # 유닛 단위 선방어 — 구분자 유발 유닛은 원문 유지
            lines[b["s"]:b["e"]] = new_text.split("\n")
        out_pages.append("\n".join(lines))
    result = page_separator.join(out_pages)
    if len(result.split(page_separator)) != len(pages):
        raise ValueError(
            f"조립 후 페이지 수 불일치: {len(result.split(page_separator))} != {len(pages)}"
        )
    return result


def layout_units(pages: list) -> list[Unit]:
    """layout.json 페이지들에서 번역 대상 블록 유닛만 (content 있고 image 키 없음)."""
    units: list[Unit] = []
    for page in pages:
        pno = page.get("page")
        for i, block in enumerate(page.get("blocks", [])):
            if "image" in block:
                continue
            content = block.get("content")
            if not content or not str(content).strip():
                continue
            units.append(
                Unit(id=f"lay:{pno}:{i}", kind=str(block.get("type") or "text"), page=pno, src=content)
            )
    return units


def apply_layout(pages: list, translations: dict[str, str]) -> list:
    """deep copy 후 content만 교체 — bbox/fs/bold/vertical/fonts_v 등은 그대로."""
    out = copy.deepcopy(pages)
    for page in out:
        pno = page.get("page")
        for i, block in enumerate(page.get("blocks", [])):
            uid = f"lay:{pno}:{i}"
            if uid in translations:
                block["content"] = translations[uid]
    return out
