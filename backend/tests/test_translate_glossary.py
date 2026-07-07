"""용어집 — 시드 검증·단어 경계·first_unit 순서·LLM 관용 파싱·폴백."""

from app.translate.glossary import (
    Glossary,
    GlossaryEntry,
    build_glossary,
    extract_candidates,
    load_seed,
)
from app.translate.segment import split_markdown

SEP = "\n\n---\n\n"


def test_시드_로드_정책과_소문자():
    seed = load_seed()
    assert 90 <= len(seed) <= 140
    assert all(e.policy in ("A", "B", "C", "D") for e in seed)
    assert all(e.src == e.src.lower() for e in seed)
    # normalization과 regularization이 구분되어 있어야 함(정규화 vs 정칙화)
    m = {e.src: e.ko for e in seed}
    assert m["normalization"] == "정규화" and m["regularization"] == "정칙화"


def test_단어_경계_매칭():
    g = Glossary([GlossaryEntry("cat", "고양이", "C")])
    gen, _ = g.for_unit("A cat sat.")
    assert gen == [("cat", "고양이")]
    gen2, _ = g.for_unit("The category is broad.")  # "cat"이 "category"에 안 걸림
    assert gen2 == []


def test_policy_A_는_제외():
    g = Glossary([
        GlossaryEntry("bert", "BERT", "A"),
        GlossaryEntry("attention", "어텐션", "C"),
    ])
    gen, first = g.for_unit("BERT uses attention.")
    assert ("attention", "어텐션") in gen
    assert all(s != "bert" for s, _ in gen)  # A는 프롬프트에 안 들어감


def test_first_unit_순서와_병기():
    md = "# Intro\n\nWe use sparse attention first.\n\nThen more sparse attention later."
    units = split_markdown(md, SEP)
    g = Glossary([GlossaryEntry("sparse attention", "스파스 어텐션", "D")])
    g.compute_first_units(units)
    entry = g.entries[0]
    assert entry.first_unit == "md:0:1"  # 첫 등장 유닛
    # 첫 등장 유닛에서만 병기
    _, first_here = g.for_unit("We use sparse attention first.", "md:0:1")
    assert ("sparse attention", "스파스 어텐션") in first_here
    _, first_later = g.for_unit("Then more sparse attention later.", "md:0:2")
    assert first_later == []


def test_extract_candidates_빈도와_약어제외():
    md = (
        "Graph Attention operates on nodes. Graph Attention is useful. "
        "We study Graph Attention again. The BERT and GPT models. "
        "meta-learning helps. meta-learning again. meta-learning thrice."
    )
    cands = extract_candidates(md)
    assert "Graph Attention" in cands       # 대문자 3회+
    assert "meta-learning" in cands          # 하이픈 합성어 3회+
    assert "BERT" not in cands and "GPT" not in cands  # 약어는 후보 제외


def test_build_glossary_LLM_관용파싱():
    md = "We propose Foo Bar. Foo Bar works. Foo Bar again. Uses widget-thing widget-thing widget-thing."
    units = split_markdown(md, SEP)

    class FakeLLM:
        def complete(self, system, user, *, max_tokens):
            # 코드펜스로 감싼 JSON — 관용 파싱이 벗겨내야 함
            return '```json\n[{"src": "Foo Bar", "ko": "푸 바", "policy": "D"},' \
                   ' {"src": "", "ko": "버림", "policy": "C"},' \
                   ' {"src": "widget-thing", "ko": "위젯", "policy": "X"}]\n```'

    g = build_glossary(md, units, FakeLLM(), None)
    srcs = {e.src: e for e in g.entries}
    assert "foo bar" in srcs and srcs["foo bar"].ko == "푸 바"  # src 소문자화
    assert "widget-thing" not in srcs   # policy 이상 → 버림
    assert g.warnings == []


def test_build_glossary_LLM_실패_시드폴백():
    md = "Some Model Name appears. Some Model Name twice. Some Model Name thrice."
    units = split_markdown(md, SEP)

    class BadLLM:
        def complete(self, system, user, *, max_tokens):
            raise RuntimeError("서버 오류")

    seed_n = len(load_seed())
    g = build_glossary(md, units, BadLLM(), None)
    assert len(g.entries) >= seed_n          # 시드는 보존
    assert len(g.warnings) == 1              # 실패 보고


def test_save_load_왕복(tmp_path):
    g = Glossary([GlossaryEntry("attention", "어텐션", "C", "md:0:1")])
    p = tmp_path / "glossary.json"
    g.save(p)
    g2 = Glossary.load(p)
    assert g2.entries[0].src == "attention"
    assert g2.entries[0].first_unit == "md:0:1"


def test_전대문자_제목줄은_약어_추출에서_제외():
    """실측(PubLayNet): 올캡 제목의 OF·LAYOUT·TABLE이 약어로 오등재 → 전역 위반 161건.
    소리치는 줄은 빼고, 혼합 대소문자 산문의 진짜 약어(CNN)는 살린다."""
    from app.translate.glossary import build_glossary

    md = ("PUBLAYNET: LARGEST DATASET EVER FOR DOCUMENT LAYOUT ANALYSIS\n\n"
          "We train a CNN on the dataset. The CNN performs well.\n")
    g = build_glossary(md, [], None, None)
    srcs = {e.src for e in g.entries if e.policy == "A"}
    assert "cnn" in srcs
    for junk in ("of", "layout", "table", "dataset", "analysis", "ever", "largest"):
        assert junk not in srcs, junk


def test_LLM_엔트리_스테밍_중복_가드(monkeypatch):
    """시드 fine-tuning(파인튜닝)이 있으면 LLM의 fine-tuned(미세 조정된)는 거부 —
    같은 용어의 표기 분열 방지. 스톱워드(let)도 거부."""
    from app.translate.glossary import build_glossary

    class FakeClient:
        def complete(self, system, user, *, max_tokens):
            return ('[{"src":"fine-tuned","ko":"미세 조정된","policy":"B"},'
                    '{"src":"let","ko":"Let","policy":"A"},'
                    '{"src":"resampler","ko":"리샘플러","policy":"C"}]')

    md = ("We fine-tuned the model with a Resampler stage. "
          "It uses a Resampler twice, and adds a Resampler at the end.\n")
    g = build_glossary(md, [], FakeClient(), None)
    srcs = {e.src for e in g.entries}
    assert "resampler" in srcs           # 정상 엔트리는 통과
    assert "fine-tuned" not in srcs      # 시드 fine-tuning과 스테밍 충돌 → 거부
    assert "let" not in srcs             # 스톱워드 → 거부


def test_LLM_A엔트리_형태_검증():
    """단일 title-case 일반어(Long·Model·to-end)는 A로 못 들어온다 —
    전대문자·내부대문자·다단어 고유명사만 통과."""
    from app.translate.glossary import build_glossary

    class FakeClient:
        def complete(self, system, user, *, max_tokens):
            return ('[{"src":"long","ko":"Long","policy":"A"},'
                    '{"src":"to-end","ko":"to-end","policy":"A"},'
                    '{"src":"publaynet","ko":"PubLayNet","policy":"A"},'
                    '{"src":"unlimited ocr","ko":"Unlimited OCR","policy":"A"}]')

    md = ("It has a Long horizon. We add a Long window and a Long buffer. "
          "It uses a PubLayNet split, trains on PubLayNet data, evals on PubLayNet too. "
          "Also end to-end here, more to-end there, again to-end now. "
          "See also Unlimited Ocr v1, an Unlimited Ocr v2, and Unlimited Ocr v3.\n")
    g = build_glossary(md, [], FakeClient(), None)
    a = {e.src: e.ko for e in g.entries if e.policy == "A"}
    assert "publaynet" in a and "unlimited ocr" in a
    assert "long" not in a and "to-end" not in a
