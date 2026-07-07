"""translate_eval 하네스 테스트 — 네트워크 금지.

엔진(run_translation) + 결정적 가짜 client로 tmp_path에 미니 잡을 실제 생성한 뒤
하드 불변식·용어집 위반·합쇼체 검출과 judge 스모크를 검증한다. 가짜 client는
기존 tests/test_translate_engine.py의 마커/에코 패턴을 재사용한다.
"""

import json

import pytest

from app.translate.engine import run_translation
from app.translate.glossary import Glossary
from app.translate.segment import split_markdown
from app.translate.types import TranslateConfig

from tools.translate_eval import (
    build_judge_sample,
    check_glossary,
    check_hard_invariants,
    evaluate,
    load_job,
    parse_judge_json,
    reconstruct,
    run_judge,
    unit_cache_key,
)

SEP = "\n\n---\n\n"

# 시드 용어집은 전부 ML 용어 → 중립 문서(여행 노트)로 위반 0을 달성한다.
NEUTRAL_MD = (
    "# Weekend Trip Notes\n\n"
    "We visited the old harbor and watched the boats at sunrise.\n\n"
    "![](images/harbor.jpg)\n\n"
    "Table 1. Prices near the pier\n\n"
    "<table><tr><td>Item</td><td>Cost</td></tr>"
    "<tr><td>Coffee</td><td>3</td></tr></table>\n\n"
    "The sign showed \\( a + b = c \\) beside the dock.\n\n"
    "---\n\n"
    "## Second Day\n\n"
    "- We walked along the beach in the morning.\n"
    "- Later we found a small shop downtown.\n\n"
    "The trip ended with a quiet dinner by the river.\n"
)

NEUTRAL_LAYOUT = [
    {"page": 1, "width": 1000, "height": 1400, "fonts_v": "2", "blocks": [
        {"type": "title", "bbox": [0, 0, 999, 80], "content": "Weekend Trip Notes",
         "fs": 2.5, "bold": True},
        {"type": "text", "bbox": [0, 100, 999, 300], "content": "We visited the old harbor.",
         "fs": 1.8},
        {"type": "image", "bbox": [0, 320, 500, 700], "content": "", "image": "harbor.jpg"},
    ]},
    {"page": 2, "width": 1000, "height": 1400, "blocks": [
        {"type": "title", "bbox": [0, 0, 999, 80], "content": "Second Day", "fs": 2.5},
        {"type": "text", "bbox": [0, 100, 999, 300], "content": "We walked along the beach."},
    ]},
]


# ── 가짜 client (test_translate_engine.py 패턴 재사용) ────────────────────

def _marker(user: str):
    tag = "[번역할 원문]\n"
    return user.split(tag, 1)[1] if tag in user else None


class EchoClient:
    """[번역할 원문] 섹션을 그대로 반환 → 마스킹 왕복 후 원문과 동일(구조 보존)."""

    def __init__(self):
        self.calls = 0
        self.api_mode_used = "chat"

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        src = _marker(user)
        return src if src is not None else ""  # 용어집 프롬프트 → 빈 응답(시드 폴백)


class JudgeStub:
    """judge 스모크 — 코드펜스로 감싼 JSON 1건 반환(펜스 벗기기까지 검증)."""

    def __init__(self):
        self.calls = 0
        self.api_mode_used = "chat"

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        return (
            '```json\n'
            '{"fidelity": 4, "fluency": 5, "terminology": 4, '
            '"name_errors": ["Vaswani→바스와니"], "omission": false, "note": "자연스러움"}\n'
            '```'
        )


@pytest.fixture
def cfg() -> TranslateConfig:
    return TranslateConfig(
        base_url="https://host/v1", api_key="k", model="test-model",
        api_mode="chat", concurrency=2, temperature="0",
        max_tokens_param="max_tokens", context=False,
    )


def _build(job_dir, cfg, md_text, layout=None, client=None):
    (job_dir / "result.md").write_text(md_text, encoding="utf-8")
    if layout is not None:
        (job_dir / "layout.json").write_text(
            json.dumps(layout, ensure_ascii=False), encoding="utf-8")
    run_translation(job_dir, "ko", cfg, client=client or EchoClient())
    return job_dir


@pytest.fixture
def job(tmp_path, cfg):
    return _build(tmp_path, cfg, NEUTRAL_MD, NEUTRAL_LAYOUT)


def _md_units(job_dir):
    md = (job_dir / "result.md").read_text(encoding="utf-8")
    return split_markdown(md, SEP)


def _inject(job_dir, unit, translation, lang="ko"):
    """엔진과 동일한 cache_key로 units.json에 특정 번역을 주입한다."""
    tdir = job_dir / "translations" / lang
    glossary = Glossary.load(tdir / "glossary.json")
    model = json.loads((tdir / "state.json").read_text(encoding="utf-8"))["model"]
    key = unit_cache_key(unit, glossary, model)
    cache = json.loads((tdir / "units.json").read_text(encoding="utf-8"))
    cache[key] = translation
    (tdir / "units.json").write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return unit.id


# ── 1. 전부 정상 ─────────────────────────────────────────────────────────

def test_정상잡_하드불변식_통과_위반0(job):
    report = evaluate(job)

    assert report["hard_ok"] is True
    for c in report["hard_invariants"]:
        assert c["na"] or c["ok"], f"불변식 실패: {c['name']} — {c['detail']}"

    q = report["quality"]
    assert q["found"] > 0                    # 모든 대상 유닛이 캐시에서 재구성됨
    assert q["unverified"] == 0              # 같은 PROMPT_V → 미확인 없음
    assert q["glossary_violations"] == []    # 중립 문서 → 용어집 위반 없음


def test_정상잡_프로세스지표_report와_일치(job):
    report = evaluate(job)
    rep = json.loads(
        (job / "translations" / "ko" / "report.json").read_text(encoding="utf-8"))
    p = report["process"]
    assert p["translated"] == rep["translated"]
    assert p["total"] == rep["translated"] + rep["cached"] + len(rep["kept_original"])


# ── 2. 플레이스홀더 잔재 주입 → 하드 불변식 실패 ──────────────────────────

def test_플레이스홀더_잔재_감지(job):
    ko = job / "result.ko.md"
    ko.write_text(ko.read_text(encoding="utf-8") + '\n\n<m1 v="leak"/>\n', encoding="utf-8")

    report = evaluate(job)
    residue = next(c for c in report["hard_invariants"] if c["name"] == "플레이스홀더 잔재")
    assert residue["ok"] is False
    assert report["hard_ok"] is False


def test_수식_초과_감지(job):
    """모델이 없던 수식을 발명한 경우(초과)도 실패로 잡는다."""
    ko = job / "result.ko.md"
    ko.write_text(ko.read_text(encoding="utf-8") + "\n\n추가 수식 \\( z \\) 발명.\n",
                  encoding="utf-8")
    report = evaluate(job)
    math = next(c for c in report["hard_invariants"] if c["name"] == "수식 카운트")
    assert math["ok"] is False
    assert report["hard_ok"] is False


# ── 3. 용어집 위반 주입 → 검출 ───────────────────────────────────────────

def test_용어집_위반_검출(tmp_path, cfg):
    _build(tmp_path, cfg, "The transformer processes each sentence quickly.\n")
    u = next(x for x in _md_units(tmp_path) if "transformer" in x.src.lower())

    # 역어(트랜스포머) 누락 → 위반
    _inject(tmp_path, u, "이 문장은 아주 빠르게 처리된다.")
    viols = evaluate(tmp_path)["quality"]["glossary_violations"]
    assert any(v["unit_id"] == u.id and v["src"] == "transformer" and v["policy"] == "C"
               for v in viols)

    # 양성 대조 — 역어 포함 시 위반 없음
    _inject(tmp_path, u, "이 트랜스포머는 문장을 빠르게 처리한다.")
    viols2 = evaluate(tmp_path)["quality"]["glossary_violations"]
    assert not any(v["src"] == "transformer" for v in viols2)


# ── 4. 합쇼체 주입 → 검출 ────────────────────────────────────────────────

def test_합쇼체_검출(tmp_path, cfg):
    _build(tmp_path, cfg, "We saw the calm harbor at dawn.\n")
    u = _md_units(tmp_path)[0]
    _inject(tmp_path, u, "우리는 새벽에 고요한 항구를 보았습니다. 정말 평화로웠습니다.")

    q = evaluate(tmp_path)["quality"]
    assert q["hapsyo_count"] >= 2            # 보았습니다 · 평화로웠습니다
    assert u.id in q["hapsyo_units"]


# ── 미확인(구버전 캐시) 카운트 ───────────────────────────────────────────

def test_캐시_비면_전부_미확인(job):
    before = evaluate(job)["quality"]
    (job / "translations" / "ko" / "units.json").write_text("{}", encoding="utf-8")

    after = evaluate(job)["quality"]
    assert after["found"] == 0
    assert after["unverified"] == before["found"]   # 대상 유닛 전부 미확인으로 이동
    assert after["skipped"] == before["skipped"]     # 스킵 분류는 캐시와 무관


# ── judge 스모크 (몽키패치 client) ───────────────────────────────────────

def test_judge_스모크(job, cfg):
    rec = reconstruct(load_job(job))
    stub = JudgeStub()
    result = run_judge(rec, cfg, client=stub, sample_size=12)

    assert result["judged"] >= 1
    assert result["failed"] == 0
    assert stub.calls == result["judged"]
    assert result["avg_fidelity"] == 4
    assert result["avg_fluency"] == 5
    assert result["avg_terminology"] == 4
    assert "Vaswani→바스와니" in result["name_errors"]
    assert result["omission_count"] == 0

    # 표본은 번역된 md 유닛에서만, 상한 sample_size.
    sample = build_judge_sample(rec, sample_size=12)
    assert 1 <= len(sample) <= 12
    assert all(uid in rec.found and uid in rec.md_ids for uid in sample)


def test_parse_judge_json_관용파싱():
    plain = '{"fidelity": 3, "omission": true}'
    assert parse_judge_json(plain)["fidelity"] == 3
    fenced = '```json\n{"fluency": 5}\n```'
    assert parse_judge_json(fenced)["fluency"] == 5
    noisy = '설명입니다.\n{"terminology": 2}\n끝.'
    assert parse_judge_json(noisy)["terminology"] == 2
    with pytest.raises(ValueError):
        parse_judge_json("JSON이 전혀 없음")


# ── 함수 단위 재사용성 (main 없이 import 가능) ───────────────────────────

def test_check_함수_직접호출(job):
    j = load_job(job)
    checks = check_hard_invariants(j)
    assert all(c.ok for c in checks if not c.na)

    rec = reconstruct(j)
    assert check_glossary(rec, j.glossary) == []
