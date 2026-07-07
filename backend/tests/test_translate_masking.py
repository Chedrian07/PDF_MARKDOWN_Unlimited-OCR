"""마스킹 — 토큰 왕복·관용 복원·검증·should_skip."""

import pytest

from app.translate.masking import mask, should_skip, unmask


def _roundtrip(text: str) -> tuple[str, list, list]:
    masked, mapping = mask(text)
    return unmask(masked, mapping)


@pytest.mark.parametrize("text", [
    "The energy is $E=mc^2$ here.",                       # 인라인 수식
    "Display $$\\sum_{i=1}^n x_i$$ end.",                 # 디스플레이 수식
    "Inline latex \\( a^2 + b^2 \\) here.",                # LaTeX 인라인 (render.py 실측 형태)
    "Block \\[ E = mc^2 \\] shown.",                       # LaTeX 디스플레이
    "See ```py\nx = 1\ny = 2\n``` block.",                # 펜스 코드
    "Use `inline_code()` please.",                        # 인라인 코드
    "Figure ![cap](images/p0001_0.jpg) shown.",           # 이미지
    "Visit https://example.com/path?a=1 now.",            # URL
    "DOI 10.1145/1234567.890 reference.",                 # DOI
    "Mail me at user.name@example.co.kr today.",          # 이메일
    "Bold <b>text</b> and <br/> tag.",                    # HTML 태그
    "As in [1] and [2, 3] and [4-6].",                    # 인용
    "See Figure 2 and Table 1 and Eq. (5).",              # 참조
])
def test_각_토큰_종류_왕복(text):
    restored, missing, dup = _roundtrip(text)
    assert restored == text
    assert missing == [] and dup == []


def test_수식_혼합_인라인_디스플레이():
    text = "식 $a$와 $$b$$ 혼합"
    masked, mapping = mask(text)
    # 인라인·디스플레이 각각 다른 플레이스홀더로 마스킹됨 (v 미리보기엔 원문 조각이 남음)
    assert set(mapping.values()) == {"$a$", "$$b$$"}
    assert len(mapping) == 2
    assert "$a$와" not in masked  # 구조상 원문 수식이 본문에서 치환됨
    restored, missing, dup = unmask(masked, mapping)
    assert restored == text and not missing and not dup


def test_플레이스홀더_전역_번호():
    masked, mapping = mask("$x$ and Figure 1 and [2]")
    # 종류를 가로질러 1,2,3 전역 증가
    assert set(mapping) == {"m1", "f2", "c3"}


def test_관용_복원_슬래시_속성_공백():
    _, mapping = mask("value $x$ end")
    pid = next(iter(mapping))
    for variant in (f"값 <{pid}> 끝", f"값 < {pid} /> 끝", f"값 <{pid} v=\"바뀜\"/> 끝"):
        restored, missing, dup = unmask(variant, mapping)
        assert restored == "값 $x$ 끝"
        assert not missing and not dup


def test_missing_보고():
    _, mapping = mask("$x$ and $y$")
    restored, missing, dup = unmask("플레이스홀더 없는 번역", mapping)
    assert set(missing) == {"m1", "m2"} and dup == []


def test_dup_보고_전부_복원():
    masked, mapping = mask("only $x$")
    pid = next(iter(mapping))
    restored, missing, dup = unmask(f"<{pid}/> 그리고 <{pid}/>", mapping)
    assert restored == "$x$ 그리고 $x$"  # 전부 복원
    assert pid in dup and missing == []


def test_잔여_플레이스홀더_dup():
    _, mapping = mask("text $x$ here")
    # LLM이 만들어낸 존재하지 않는 플레이스홀더가 남음
    restored, missing, dup = unmask("복원됨 $x$ 그런데 <m9/> 잔여", mapping)
    assert any("m9" in d for d in dup)


def test_should_skip_수식뿐():
    assert should_skip("$E = mc^2$") == "non-linguistic"
    assert should_skip("[1, 2, 3]") == "non-linguistic"


def test_should_skip_식별자():
    assert should_skip("arXiv:2504.19874") == "identifier"
    assert should_skip("arXiv:1908.07836v1") == "identifier"


def test_should_skip_한국어_과반():
    assert should_skip("이것은 이미 한국어로 된 문장이다.") == "already-korean"


def test_should_skip_번역대상():
    assert should_skip("This is a normal English sentence about models.") == ""
    # 수식 섞였어도 자연어가 있으면 번역 대상
    assert should_skip("The loss is $L = \\sum x$ over samples.") == ""
