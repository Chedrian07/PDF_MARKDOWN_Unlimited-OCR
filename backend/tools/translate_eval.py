"""한국어 번역 품질 평가 하네스 — 번역 잡 산출물을 오프라인으로 채점한다.

사용법:
    PYTHONPATH=backend uv run python backend/tools/translate_eval.py JOB_DIR \\
        [--lang ko] [--json OUT.json] [--judge] [--judge-model MODEL] [--sample 12]

세 층위로 평가한다.
  1. 하드 불변식 — 구조 보존(페이지/블록 수, 플레이스홀더 잔재, 수식·이미지·표
     카운트, 레이아웃 필드 보존). 하나라도 실패하면 프로세스 종료코드 1.
  2. 프로세스 지표 — report.json에서 그대로 읽는다.
  3. 품질 지표 — units.json 캐시에서 유닛 번역을 재구성해(엔진과 동일한
     cache_key 재현법) 합쇼체·용어집 준수·한글 비율을 측정한다.
  4. (--judge) LLM 채점 — 표본 유닛을 원문/번역 쌍으로 평가자 모델에 보낸다.

설계 원칙: 평가 로직은 순수 함수로 분리하고 main은 얇게 유지한다(테스트가
함수를 직접 import). PROMPT_V·마스킹 규칙은 하드코딩하지 않고 임포트한 상수와
state.json 값을 그대로 쓴다 — 구버전 캐시로 cache_key 재현이 안 되는 유닛은
오류가 아니라 "미확인"으로 따로 센다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# app.translate는 순수 파이썬(torch/OCR 비의존) — 계약 API만 사용한다.
from app.translate.client import OpenAICompatClient
from app.translate.glossary import Glossary
from app.translate.masking import mask, should_skip
from app.translate.segment import layout_units, split_markdown
from app.translate.types import PROMPT_V, TranslateConfig, cache_key

# 페이지 구분자 — 파이프라인 고정 계약.
PAGE_SEPARATOR = "\n\n---\n\n"

# 플레이스홀더 잔재 패턴 (과제 명세 그대로). 마스킹 산출물 `<m1 v="…"/>`도 매칭한다.
PLACEHOLDER_RE = re.compile(r"<[mckuftg]\d+[^>]*>")

# 합쇼체 종결어미 — 명세 목록 + 흔한 형태 몇 개(정보 지표, 하드 불변식 아님).
# 접미 공유 형태(있습니다⊃습니다)는 정규식이 최좌단에서 각 시작음절로만 매칭하므로
# 이중 계수되지 않는다. 안전하게 긴 형태를 앞에 둔다.
_HAPSYO_ENDINGS = (
    "있습니다", "없습니다", "습니다", "합니다", "입니다", "됩니다",
    "갑니다", "옵니다", "봅니다", "십니까", "습니까", "입니까", "합니까",
)
HAPSYO_RE = re.compile("(?:" + "|".join(_HAPSYO_ENDINGS) + ")")


# ── 잡 로딩 ──────────────────────────────────────────────────────────────

@dataclass
class Job:
    """번역 잡 디렉터리에서 읽은 원자료(누락 파일은 None)."""

    job_dir: Path
    lang: str
    md_text: str | None
    ko_text: str | None
    layout_pages: list | None
    layout_ko_pages: list | None
    glossary: Glossary
    cache: dict
    state: dict
    report: dict


def _read_text(p: Path) -> str | None:
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _read_json(p: Path, default):
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_json_list(p: Path) -> list | None:
    val = _read_json(p, None)
    return val if isinstance(val, list) else None


def load_job(job_dir, lang: str = "ko") -> Job:
    """잡 디렉터리에서 평가에 필요한 모든 산출물을 관용적으로 읽는다.

    번역이 진행 중이어서 일부 파일이 없을 수 있으므로 누락은 None/기본값으로 둔다.
    """
    job_dir = Path(job_dir)
    tdir = job_dir / "translations" / lang
    gpath = tdir / "glossary.json"
    glossary = Glossary.load(gpath) if gpath.is_file() else Glossary()
    return Job(
        job_dir=job_dir,
        lang=lang,
        md_text=_read_text(job_dir / "result.md"),
        ko_text=_read_text(job_dir / f"result.{lang}.md"),
        layout_pages=_read_json_list(job_dir / "layout.json"),
        layout_ko_pages=_read_json_list(job_dir / f"layout.{lang}.json"),
        glossary=glossary,
        cache=_read_json(tdir / "units.json", {}) or {},
        state=_read_json(tdir / "state.json", {}) or {},
        report=_read_json(tdir / "report.json", {}) or {},
    )


# ── 유닛 번역 재구성 (엔진과 동일한 cache_key 재현법) ─────────────────────

def unit_cache_key(unit, glossary: Glossary, model: str) -> str:
    """엔진의 유닛 캐시 키를 그대로 재현한다.

    masked,_=mask(src) → pairs,first=glossary.for_unit(src, id) + keep_terms(src)
    → cache_key(masked, model, pairs+first+[(k,k)…]). PROMPT_V는 cache_key가 임포트한
    상수를 쓰므로 여기서 주입하지 않는다(엔진 버전 상향에 자동 추종).
    """
    masked, _ = mask(unit.src)
    pairs, first = glossary.for_unit(unit.src, unit.id)
    keep = glossary.keep_terms(unit.src)
    return cache_key(masked, model, pairs + first + [(k, k) for k in keep])


@dataclass
class Recon:
    """유닛 번역 재구성 결과."""

    units: list = field(default_factory=list)          # 전체 유닛(md+lay, 문서순)
    md_ids: list = field(default_factory=list)         # md 유닛 id(문서순)
    by_id: dict = field(default_factory=dict)          # id → Unit
    found: dict = field(default_factory=dict)          # id → 번역문(캐시 적중, 복원본)
    skipped_ids: list = field(default_factory=list)    # skip_reason/should_skip
    kept_ids: list = field(default_factory=list)       # report의 kept_original(원문유지)
    unverified_ids: list = field(default_factory=list)  # 캐시 미적중·비skip·비kept = 구버전


def reconstruct(job: Job) -> Recon:
    """md·layout 유닛을 분리하고 각 유닛의 번역을 캐시에서 되찾는다."""
    rec = Recon()
    if job.md_text is None:
        return rec
    md_units = split_markdown(job.md_text, PAGE_SEPARATOR)
    lay_units = layout_units(job.layout_pages) if job.layout_pages else []
    rec.units = md_units + lay_units
    rec.md_ids = [u.id for u in md_units]
    rec.by_id = {u.id: u for u in rec.units}

    model = job.state.get("model", "")
    kept = set(job.report.get("kept_original", []) or [])
    for u in rec.units:
        if u.skip_reason or should_skip(u.src):
            rec.skipped_ids.append(u.id)
            continue
        key = unit_cache_key(u, job.glossary, model)
        if key in job.cache:
            rec.found[u.id] = job.cache[key]
        elif u.id in kept:
            rec.kept_ids.append(u.id)         # 원문 유지 유닛 — 캐시에 없는 게 정상
        else:
            rec.unverified_ids.append(u.id)   # 구버전 캐시/용어집 표류 → 재현 불가
    return rec


# ── 하드 불변식 ──────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    na: bool = False   # 해당 없음(예: layout 미존재) — 통과/실패로 세지 않음

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "na": self.na}


def _count_placeholders(text: str | None) -> int:
    return len(PLACEHOLDER_RE.findall(text)) if text else 0


def _layout_field_diffs(orig_pages: list, ko_pages: list) -> list[str]:
    """content를 제외한 모든 레이아웃 필드(페이지·블록 수준)의 불일치 목록."""
    diffs: list[str] = []
    for pi, (op, kp) in enumerate(zip(orig_pages, ko_pages)):
        op_meta = {k: v for k, v in op.items() if k != "blocks"}
        kp_meta = {k: v for k, v in kp.items() if k != "blocks"}
        if op_meta != kp_meta:
            diffs.append(f"page[{pi}] 페이지 필드")
        ob, kb = op.get("blocks", []) or [], kp.get("blocks", []) or []
        for bi, (o, k) in enumerate(zip(ob, kb)):
            o2 = {kk: vv for kk, vv in o.items() if kk != "content"}
            k2 = {kk: vv for kk, vv in k.items() if kk != "content"}
            if o2 != k2:
                diffs.append(f"page[{pi}].block[{bi}] 필드")
    return diffs


def check_hard_invariants(job: Job) -> list[Check]:
    """구조 보존 하드 불변식. na가 아닌 항목이 하나라도 실패하면 FAIL."""
    checks: list[Check] = []
    md, ko = job.md_text, job.ko_text

    # 1) 페이지 수 (result.md vs result.ko.md)
    if md is None:
        checks.append(Check("페이지 수(md)", False, "result.md 없음"))
    elif ko is None:
        checks.append(Check("페이지 수(md)", False, f"result.{job.lang}.md 없음"))
    else:
        a = len(md.split(PAGE_SEPARATOR))
        b = len(ko.split(PAGE_SEPARATOR))
        checks.append(Check("페이지 수(md)", a == b, f"원문 {a} · 번역 {b}"))

    # 2) 레이아웃 페이지·블록 수
    lp, lk = job.layout_pages, job.layout_ko_pages
    if lp is None:
        checks.append(Check("페이지·블록 수(layout)", True, "layout.json 없음", na=True))
    elif lk is None:
        checks.append(Check("페이지·블록 수(layout)", False, f"layout.{job.lang}.json 없음"))
    else:
        page_ok = len(lp) == len(lk)
        block_mismatch = [
            i for i, (p, q) in enumerate(zip(lp, lk))
            if len(p.get("blocks", []) or []) != len(q.get("blocks", []) or [])
        ]
        ok = page_ok and not block_mismatch
        detail = f"페이지 {len(lp)}·{len(lk)}"
        if block_mismatch:
            detail += f", 블록 수 불일치 페이지 {block_mismatch}"
        checks.append(Check("페이지·블록 수(layout)", ok, detail))

    # 3) 플레이스홀더 잔재 (result.ko.md + layout.ko.json)
    if ko is None and lk is None:
        checks.append(Check("플레이스홀더 잔재", True, "번역 산출물 없음", na=True))
    else:
        n_md = _count_placeholders(ko)
        lk_dump = json.dumps(lk, ensure_ascii=False) if lk is not None else ""
        n_lay = _count_placeholders(lk_dump)
        checks.append(Check(
            "플레이스홀더 잔재", (n_md + n_lay) == 0,
            f"ko.md {n_md}건 · layout {n_lay}건",
        ))

    # 4) 수식 카운트 \( \[ (초과·부족 모두 실패)
    if md is not None and ko is not None:
        a1, b1 = md.count("\\("), ko.count("\\(")
        a2, b2 = md.count("\\["), ko.count("\\[")
        checks.append(Check(
            "수식 카운트", a1 == b1 and a2 == b2,
            f"\\( {a1}={b1} · \\[ {a2}={b2}",
        ))
        # 5) 이미지 ![
        i1, i2 = md.count("!["), ko.count("![")
        checks.append(Check("이미지 ![ 카운트", i1 == i2, f"{i1}={i2}"))
        # 6) 표 <td / <table
        t1, t2 = md.count("<td"), ko.count("<td")
        b1t, b2t = md.count("<table"), ko.count("<table")
        checks.append(Check(
            "표 <td/<table 카운트", t1 == t2 and b1t == b2t,
            f"<td {t1}={t2} · <table {b1t}={b2t}",
        ))
    else:
        for nm in ("수식 카운트", "이미지 ![ 카운트", "표 <td/<table 카운트"):
            checks.append(Check(nm, False, "result.md 또는 번역 md 없음"))

    # 7) 레이아웃 필드 보존 (content 제외 전부 동일)
    if lp is None:
        checks.append(Check("레이아웃 필드 보존", True, "layout.json 없음", na=True))
    elif lk is None:
        checks.append(Check("레이아웃 필드 보존", False, f"layout.{job.lang}.json 없음"))
    else:
        diffs = _layout_field_diffs(lp, lk)
        checks.append(Check(
            "레이아웃 필드 보존", not diffs,
            "동일" if not diffs else f"{len(diffs)}건 불일치: {diffs[:5]}",
        ))
    return checks


# ── 프로세스 지표 ────────────────────────────────────────────────────────

def process_metrics(job: Job) -> dict:
    """report.json 기반 프로세스 지표. total은 파생(translated+cached+kept)."""
    r = job.report
    kept = list(r.get("kept_original", []) or [])
    m = {
        "translated": int(r.get("translated", 0) or 0),
        "cached": int(r.get("cached", 0) or 0),
        "kept_original": kept,
        "kept_original_count": len(kept),
        "retried": int(r.get("retried", 0) or 0),
        "skipped": int(r.get("skipped", 0) or 0),
    }
    m["total"] = m["translated"] + m["cached"] + m["kept_original_count"]
    # 동시 작업 에이전트가 추가할 수 있는 선택 필드 — 있으면 그대로 싣는다.
    for opt in ("repaired", "split", "sanitized"):
        if opt in r:
            m[opt] = r[opt]
    return m


# ── 품질 지표 ────────────────────────────────────────────────────────────

def _term_matcher(src: str) -> re.Pattern:
    """glossary._matcher와 동일 — 단어 경계 + 대소문자 무시, 다단어는 \\s+ 결합."""
    body = r"\s+".join(re.escape(w) for w in src.split())
    return re.compile(r"\b" + body + r"\b", re.IGNORECASE)


def _strip_placeholders(text: str) -> str:
    """비언어 토큰(수식·코드·URL·태그 등)을 공백으로 치환 — 한글 비율 왜곡 방지."""
    masked, _ = mask(text)
    return PLACEHOLDER_RE.sub(" ", masked)


def count_hapsyo(rec: Recon) -> tuple[int, list[str]]:
    """번역된 유닛에서 합쇼체 종결어미 등장 수와 해당 유닛 id 목록."""
    total = 0
    ids: list[str] = []
    for uid, trans in rec.found.items():
        n = len(HAPSYO_RE.findall(trans))
        if n:
            total += n
            ids.append(uid)
    return total, sorted(ids)


def hangul_ratio(rec: Recon) -> tuple[float, int, int]:
    """번역 유닛(캐시 적중분)에서 한글 문자 / 전체 문자(토큰 제외, 공백 제외).

    (비율, 한글수, 전체수) 반환. 전체가 0이면 비율 0.0.
    """
    hangul = total = 0
    for trans in rec.found.values():
        stripped = _strip_placeholders(trans)
        hangul += len(re.findall(r"[가-힣]", stripped))
        total += len(re.findall(r"\S", stripped))
    return (hangul / total if total else 0.0), hangul, total


def check_glossary(rec: Recon, glossary: Glossary) -> list[dict]:
    """용어집 준수 위반 목록.

    B/C/D: src가 단어 경계로 등장하는 원문 유닛의 대응 번역에 ko가 부분 문자열로
           없으면 위반. A: src(원형)가 번역에 부분 문자열로 남아있지 않으면 위반
           (조사 부착 허용 → 단어 경계 대신 부분 문자열, 대소문자 무시).
    번역이 없는 유닛(skip/kept/미확인)은 검사 대상에서 제외한다.
    """
    from app.translate.glossary import _strip_tokens  # 엔진과 동일한 스캔 규칙

    violations: list[dict] = []
    scan_cache: dict[str, str] = {}
    for e in glossary.entries:
        matcher = _term_matcher(e.src)
        for u in rec.units:
            trans = rec.found.get(u.id)
            if trans is None:
                continue
            # 수식·코드·인용 내부 매칭 방지 — 번역 대상 텍스트에서만 검사 (엔진 for_unit과 동일)
            scan = scan_cache.get(u.id)
            if scan is None:
                scan = scan_cache[u.id] = _strip_tokens(u.src)
            if not matcher.search(scan):
                continue
            if e.policy == "A":
                bad = e.src.lower() not in trans.lower()
                reason = "원형 미유지"
            else:
                bad = e.ko not in trans
                reason = "역어 누락"
            if bad:
                violations.append({
                    "unit_id": u.id, "src": e.src, "ko": e.ko,
                    "policy": e.policy, "reason": reason,
                })
    return violations


def quality_metrics(rec: Recon, glossary: Glossary) -> dict:
    hap_count, hap_ids = count_hapsyo(rec)
    ratio, hangul_n, total_n = hangul_ratio(rec)
    return {
        "found": len(rec.found),
        "skipped": len(rec.skipped_ids),
        "kept_original": len(rec.kept_ids),
        "unverified": len(rec.unverified_ids),
        "unverified_ids": rec.unverified_ids,
        "hapsyo_count": hap_count,
        "hapsyo_units": hap_ids,
        "hangul_ratio": ratio,
        "hangul_chars": hangul_n,
        "total_chars": total_n,
        "glossary_violations": check_glossary(rec, glossary),
    }


# ── LLM 채점 (--judge) ───────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "당신은 영어 학술/기술 문서의 한국어 번역을 채점하는 전문 평가자입니다.\n"
    "주어진 [원문]과 [번역]을 비교해, 아래 스키마의 JSON 객체 '하나만' 출력하세요.\n"
    "설명·머리말·코드펜스를 절대 덧붙이지 말고 순수 JSON만 반환합니다.\n"
    "{\n"
    '  "fidelity": 1-5 정수 (의미 충실도),\n'
    '  "fluency": 1-5 정수 (한국어 유창성),\n'
    '  "terminology": 1-5 정수 (전문용어 적절성),\n'
    '  "name_errors": ["원문표기→번역표기", ...],\n'
    '  "omission": true/false,\n'
    '  "note": "한 줄 총평"\n'
    "}\n"
    "인명·고유명사는 원문 표기 유지가 정책입니다 — 음차되거나 번역됐으면 "
    "name_errors에 \"원문→번역\" 형식으로 기록하세요(없으면 빈 배열).\n"
    "<m1 .../> <k2 .../> 같은 태그는 번역 대상이 아닌 보호 토큰이니 채점에서 제외합니다."
)


def parse_judge_json(raw: str) -> dict:
    """관용 JSON 파싱 — 코드펜스 벗기고 첫 { ~ 마지막 } 만 추출."""
    s = raw.strip()
    m = re.match(r"^```[^\n]*\n(.*?)```\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError("JSON 객체를 찾지 못함")
    obj = json.loads(s[i:j + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON 객체가 아님")
    return obj


def _as_score(v):
    """1-5 정수로 관용 변환 — 실패 시 None."""
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def build_judge_sample(rec: Recon, sample_size: int = 12, seed: int = 1234) -> list[str]:
    """채점 표본 — 번역된 md 유닛 중 최장 4개 + 고정 시드 랜덤 (sample-4)개."""
    md_found = [uid for uid in rec.md_ids if uid in rec.found]
    by_len = sorted(md_found, key=lambda uid: len(rec.by_id[uid].src), reverse=True)
    longest = by_len[:4]
    chosen = set(longest)
    rest = [uid for uid in md_found if uid not in chosen]
    random.Random(seed).shuffle(rest)
    extra = rest[: max(0, sample_size - len(longest))]
    return longest + extra


def run_judge(
    rec: Recon,
    cfg: TranslateConfig,
    *,
    sample_size: int = 12,
    judge_model: str | None = None,
    client=None,
) -> dict:
    """표본 유닛을 평가자 모델로 채점하고 집계한다.

    client가 None이면 cfg(있으면 judge_model로 교체)로 OpenAICompatClient 생성.
    파싱/호출 실패 유닛은 건너뛰고 failed로 센다.
    """
    if judge_model:
        cfg = dataclasses.replace(cfg, model=judge_model)
    if client is None:
        client = OpenAICompatClient(cfg)

    sample = build_judge_sample(rec, sample_size)
    per_unit: list[dict] = []
    failed = 0
    for uid in sample:
        src = rec.by_id[uid].src
        trans = rec.found[uid]
        user = f"[원문]\n{src}\n\n[번역]\n{trans}"
        try:
            raw = client.complete(JUDGE_SYSTEM, user, max_tokens=1000)
            data = parse_judge_json(raw)
        except Exception:
            failed += 1
            continue
        name_errors = data.get("name_errors") or []
        if not isinstance(name_errors, list):
            name_errors = [str(name_errors)]
        per_unit.append({
            "unit_id": uid,
            "fidelity": _as_score(data.get("fidelity")),
            "fluency": _as_score(data.get("fluency")),
            "terminology": _as_score(data.get("terminology")),
            "name_errors": [str(x) for x in name_errors],
            "omission": bool(data.get("omission")),
            "note": str(data.get("note", "")),
        })

    def _avg(key: str):
        vals = [u[key] for u in per_unit if u[key] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    all_name_errors = [ne for u in per_unit for ne in u["name_errors"]]
    return {
        "sample_size": len(sample),
        "judged": len(per_unit),
        "failed": failed,
        "avg_fidelity": _avg("fidelity"),
        "avg_fluency": _avg("fluency"),
        "avg_terminology": _avg("terminology"),
        "name_errors": all_name_errors,
        "omission_count": sum(1 for u in per_unit if u["omission"]),
        "per_unit": per_unit,
    }


# ── 상위 평가 (하드+프로세스+품질) ──────────────────────────────────────

def evaluate(job_dir, lang: str = "ko") -> dict:
    """하드 불변식·프로세스·품질을 평가한 리포트 dict를 반환(judge 미포함)."""
    job = load_job(job_dir, lang)
    checks = check_hard_invariants(job)
    hard_ok = all(c.ok for c in checks if not c.na)
    rec = reconstruct(job)
    return {
        "job_dir": str(job.job_dir),
        "lang": lang,
        "prompt_v_current": PROMPT_V,
        "prompt_v_state": job.state.get("prompt_v"),
        "model": job.state.get("model", ""),
        "status": job.state.get("status", ""),
        "hard_ok": hard_ok,
        "hard_invariants": [c.as_dict() for c in checks],
        "process": process_metrics(job),
        "quality": quality_metrics(rec, job.glossary),
    }


# ── 사람이 읽는 요약 ─────────────────────────────────────────────────────

def format_summary(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"번역 품질 평가 — {report['job_dir']}  (lang={report['lang']}, "
                 f"status={report.get('status') or '?'})")
    if report.get("prompt_v_state") and report["prompt_v_state"] != report["prompt_v_current"]:
        lines.append(f"  ⚠ PROMPT_V 불일치: state={report['prompt_v_state']} "
                     f"vs 현재={report['prompt_v_current']} (일부 유닛 미확인 가능)")
    lines.append("=" * 66)

    lines.append("\n[하드 불변식]")
    for c in report["hard_invariants"]:
        mark = "—" if c["na"] else ("✓" if c["ok"] else "✗")
        lines.append(f"  {mark} {c['name']:<20} {c['detail']}")

    p = report["process"]
    lines.append("\n[프로세스 지표]")
    lines.append(f"  total {p['total']} · translated {p['translated']} · "
                 f"cached {p['cached']} · retried {p['retried']} · skipped {p['skipped']}")
    kept = p["kept_original"]
    lines.append(f"  kept_original {p['kept_original_count']}건"
                 + (f": {kept}" if kept else ""))
    for opt in ("repaired", "split", "sanitized"):
        if opt in p:
            lines.append(f"  {opt} {p[opt]}")

    q = report["quality"]
    lines.append("\n[품질 지표]")
    lines.append(f"  재구성: 번역 {q['found']} · 스킵 {q['skipped']} · "
                 f"원문유지 {q['kept_original']} · 미확인 {q['unverified']}"
                 + (f" {q['unverified_ids']}" if q["unverified_ids"] else ""))
    lines.append(f"  합쇼체 등장 {q['hapsyo_count']}회 (유닛 {len(q['hapsyo_units'])}개)")
    lines.append(f"  한글 비율 {q['hangul_ratio'] * 100:.1f}% "
                 f"({q['hangul_chars']}/{q['total_chars']}자)")
    viols = q["glossary_violations"]
    lines.append(f"  용어집 위반 {len(viols)}건")
    for v in viols[:12]:
        lines.append(f"    - [{v['policy']}] {v['unit_id']}  "
                     f"\"{v['src']}\" → \"{v['ko']}\" {v['reason']}")
    if len(viols) > 12:
        lines.append(f"    … 외 {len(viols) - 12}건")

    if report.get("judge"):
        j = report["judge"]
        lines.append(f"\n[LLM 채점]  (표본 {j['sample_size']} · 채점 {j['judged']} · "
                     f"실패 {j['failed']})")
        lines.append(f"  fidelity {j['avg_fidelity']} · fluency {j['avg_fluency']} · "
                     f"terminology {j['avg_terminology']}")
        lines.append(f"  누락(omission) {j['omission_count']}건 · "
                     f"인명/고유명사 오류 {len(j['name_errors'])}건")
        for ne in j["name_errors"][:12]:
            lines.append(f"    - {ne}")

    total = sum(1 for c in report["hard_invariants"] if not c["na"])
    passed = sum(1 for c in report["hard_invariants"] if not c["na"] and c["ok"])
    verdict = "PASS" if report["hard_ok"] else "FAIL"
    lines.append("\n" + "=" * 66)
    lines.append(f"결과: {verdict}  (하드 불변식 {passed}/{total} 통과)")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="translate_eval",
        description="한국어 번역 품질 평가 하네스",
    )
    ap.add_argument("job_dir", help="번역 잡 디렉터리 경로")
    ap.add_argument("--lang", default="ko", help="번역 언어 코드 (기본 ko)")
    ap.add_argument("--json", dest="json_out", default=None, help="리포트 JSON 저장 경로")
    ap.add_argument("--judge", action="store_true", help="LLM 채점 활성화")
    ap.add_argument("--judge-model", default=None, help="채점에 쓸 모델(기본 번역 모델)")
    ap.add_argument("--sample", type=int, default=12, help="채점 표본 유닛 수 (기본 12)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = evaluate(args.job_dir, args.lang)

    if args.judge:
        try:
            cfg = TranslateConfig.from_env()
            job = load_job(args.job_dir, args.lang)
            rec = reconstruct(job)
            if rec.found:
                report["judge"] = run_judge(
                    rec, cfg, sample_size=args.sample, judge_model=args.judge_model,
                )
            else:
                print("경고: 재구성된 번역 유닛이 없어 채점을 건너뜁니다.", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — 채점 실패가 불변식 결과를 가리면 안 됨
            print(f"경고: LLM 채점 실패 — {e}", file=sys.stderr)

    print(format_summary(report))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"\n(JSON 저장: {args.json_out})", file=sys.stderr)

    return 0 if report["hard_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
