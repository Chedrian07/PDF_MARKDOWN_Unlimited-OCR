"""문서 용어집 — 시드(큐레이션) + 자동 약어(policy A) + LLM 후보 판정.

용어집은 문서 전체에서 용어를 일관되게 번역하기 위한 것이다. 각 유닛 번역 시
그 유닛에 실제 등장하는 용어쌍만 프롬프트에 실어 비대화를 막는다(for_unit).
policy D 용어는 문서 첫 등장 유닛에서만 "역어(원어)" 병기한다.

policy 분류: A=원문 유지(약어·고유명사) B=확립 학술어 C=음차 D=신조어·병기.
매칭은 단어 경계 + 대소문자 무시("cat"이 "category"에 걸리지 않는다). policy A는
프롬프트/캐시 키에서 제외한다(시스템 프롬프트 5번 규칙이 처리).

LLM 실패 시 시드+약어만으로 진행한다(예외를 삼키고 warnings에 남김).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import prompts

_SEED_PATH = Path(__file__).parent / "data" / "seed_ko.json"

# 대문자 n그램 후보에서 제외할 문두 일반어
_STOPWORDS = {
    "the", "we", "in", "this", "a", "an", "it", "our", "however", "these", "those",
    "for", "and", "of", "to", "is", "are", "was", "were", "that", "with", "as",
    "by", "on", "or", "from", "at", "which", "their", "they", "its", "also",
    "such", "can", "more", "most", "than", "then", "thus", "here", "there",
    "when", "while", "where", "if", "but", "not", "all", "each", "both", "we",
    "figure", "table", "section", "equation", "appendix", "algorithm", "fig", "eq",
}

_CAP_NGRAM_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b")
_HYPHEN_RE = re.compile(r"\b[a-z]+(?:-[a-z]+)+\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\d?\b")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"<[mkgucft]\d+\b[^>]*>")


@dataclass
class GlossaryEntry:
    src: str
    ko: str
    policy: str
    first_unit: str = ""


def _strip_tokens(md_text: str) -> str:
    """수식·코드·태그 등을 공백으로 치환한 스캔용 텍스트(마스킹 재사용)."""
    from .masking import mask

    masked, _ = mask(md_text)
    return _PLACEHOLDER_RE.sub(" ", masked)


class Glossary:
    def __init__(self, entries: list[GlossaryEntry] | None = None) -> None:
        self.entries: list[GlossaryEntry] = entries or []
        self.warnings: list[str] = []
        self._re_cache: dict[str, re.Pattern] = {}

    def _matcher(self, src: str) -> re.Pattern:
        pat = self._re_cache.get(src)
        if pat is None:
            body = r"\s+".join(re.escape(w) for w in src.split())
            pat = re.compile(r"\b" + body + r"\b", re.IGNORECASE)
            self._re_cache[src] = pat
        return pat

    def for_unit(self, text: str, unit_id: str = "") -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """이 유닛에 등장하는 (일반 용어쌍 B/C/D, 첫 등장 병기쌍 D)를 반환.

        unit_id가 엔트리의 first_unit과 일치하는 D 항목만 병기 목록에 들어간다.
        """
        general: list[tuple[str, str]] = []
        first: list[tuple[str, str]] = []
        for e in self.entries:
            if e.policy == "A":
                continue
            if self._matcher(e.src).search(text):
                general.append((e.src, e.ko))
                if e.policy == "D" and unit_id and e.first_unit == unit_id:
                    first.append((e.src, e.ko))
        return general, first

    def compute_first_units(self, ordered_units) -> None:
        """문서 순서의 유닛들을 스캔해 각 엔트리의 first_unit(첫 등장 유닛 id)을 채운다."""
        for e in self.entries:
            e.first_unit = ""
        pats = [(e, self._matcher(e.src)) for e in self.entries]
        for unit in ordered_units:
            for e, pat in pats:
                if not e.first_unit and pat.search(unit.src):
                    e.first_unit = unit.id

    def save(self, path) -> None:
        data = [
            {"src": e.src, "ko": e.ko, "policy": e.policy, "first_unit": e.first_unit}
            for e in self.entries
        ]
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Glossary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([
            GlossaryEntry(d["src"], d["ko"], d["policy"], d.get("first_unit", ""))
            for d in data
        ])


def load_seed() -> list[GlossaryEntry]:
    data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    return [GlossaryEntry(d["src"].lower(), d["ko"], d["policy"]) for d in data]


def extract_candidates(md_text: str) -> list[str]:
    """용어 후보 — 대문자 1-3그램(빈도≥3)·하이픈 소문자 합성어(빈도≥3). 상한 80.

    전대문자 약어는 여기서 제외한다(build_glossary가 자동 policy A로 처리).
    """
    text = _strip_tokens(md_text)
    cap = Counter()
    for m in _CAP_NGRAM_RE.finditer(text):
        phrase = re.sub(r"\s+", " ", m.group()).strip()
        if phrase.split()[0].lower() in _STOPWORDS:
            continue
        cap[phrase] += 1
    hyph = Counter()
    for m in _HYPHEN_RE.finditer(text):
        hyph[m.group()] += 1
    cands = [p for p, c in cap.items() if c >= 3]
    cands += [p for p, c in hyph.items() if c >= 3]
    # 안정적 순서: 빈도 내림차순 → 알파벳
    freq = {**cap, **hyph}
    cands.sort(key=lambda p: (-freq[p], p))
    return cands[:80]


def _parse_glossary_json(raw: str) -> list:
    """LLM 응답의 JSON 배열을 관용 파싱 — 코드펜스 벗기고 배열만 추출."""
    s = raw.strip()
    m = re.match(r"^```[^\n]*\n(.*?)```\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j < i:
        return []
    data = json.loads(s[i:j + 1])
    return data if isinstance(data, list) else []


def build_glossary(md_text: str, ordered_units, client, cfg) -> Glossary:
    """시드 + 자동 약어(A) + (client 있으면) LLM 판정으로 용어집을 만든다.

    ordered_units는 md 유닛(문서 순서) — 마지막에 각 엔트리 first_unit을 계산한다.
    LLM 호출·파싱 실패는 삼키고 warnings에 남긴 뒤 시드만으로 진행한다.
    """
    g = Glossary(load_seed())
    seen = {e.src for e in g.entries}

    # 자동 약어 → policy A (프롬프트/캐시엔 안 들어가지만 문서 참조용으로 기록)
    scan = _strip_tokens(md_text)
    for ac in sorted(set(_ACRONYM_RE.findall(scan))):
        if ac.lower() in seen:
            continue
        g.entries.append(GlossaryEntry(ac.lower(), ac, "A"))
        seen.add(ac.lower())

    cands = [c for c in extract_candidates(md_text) if c.lower() not in seen]

    if client is not None and cands:
        try:
            headings = _HEADING_RE.findall(md_text)
            prompt = prompts.build_glossary_prompt(md_text[:2000], headings, cands)
            raw = client.complete(prompts.SYSTEM_GLOSSARY, prompt, max_tokens=4000)
            for it in _parse_glossary_json(raw):
                if not isinstance(it, dict):
                    continue
                src = str(it.get("src", "")).strip().lower()
                ko = str(it.get("ko", "")).strip()
                pol = str(it.get("policy", "")).strip().upper()
                if not src or not ko or pol not in ("A", "B", "C", "D"):
                    continue
                if src in seen:
                    continue
                g.entries.append(GlossaryEntry(src, ko, pol))
                seen.add(src)
        except Exception as e:  # noqa: BLE001 — LLM 실패는 치명적이지 않다
            g.warnings.append(f"용어집 LLM 판정 실패 — 시드로 진행: {e}")

    g.compute_first_units(ordered_units)
    return g
