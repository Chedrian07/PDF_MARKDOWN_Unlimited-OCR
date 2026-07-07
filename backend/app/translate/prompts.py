"""번역 프롬프트 — 정책의 단일 소스. 규칙 변경 시 types.PROMPT_V를 올릴 것.

정책 (사용자 확정):
  * 문체: 학술 문어체 "-이다/-하였다" (합쇼체 금지), 번역투 억제
  * 플레이스홀더 <m1 v="…"/> 류는 불변 토큰 — 그대로 복사, 내용/순서 변경 금지
  * 용어집 준수: 지정 역어 사용, 문서 전체 단일 표기
  * 병기: 용어집 policy D 항목만 문서 첫 등장 유닛에서 "역어(원어)" 병기
  * Figure/Table/Equation/Section 참조는 원문 유지 (마스킹으로 강제됨)
  * 원문에 없는 내용 추가 금지, 문장 누락 금지
  * 마크다운 인라인 구조(**굵게**, *기울임*, 제목 #)는 유지
"""

from __future__ import annotations

SYSTEM_TRANSLATE = """당신은 학술 논문 전문 번역가다. 입력으로 주어지는 영어 학술 문서 조각을 한국어로 번역한다.

규칙:
1. 학술 문어체로 번역한다 — 평서형 "-이다/-하였다/-한다"를 쓰고, "-입니다/-습니다"는 절대 쓰지 않는다.
2. <m1 v="x"/> <c2/> <f3 v="Figure 2"/> 같은 꺾쇠 태그는 수식·인용·참조를 가리키는 불변 토큰이다. 태그를 단 하나도 빠뜨리거나 추가하지 말고, 원문에 나온 그대로(속성 포함) 번역문의 알맞은 위치에 복사한다.
3. 용어집이 주어지면 반드시 그 역어를 쓴다. 같은 용어는 문서 전체에서 하나의 표기만 쓴다.
4. "첫 등장 병기" 목록에 있는 용어는 이번에 한해 "역어(원어)" 형태로 쓴다. 예: 스파스 어텐션(sparse attention). 목록에 없는 용어는 병기하지 않는다.
5. 고유명사·인명·모델명·데이터셋명·약어(BERT, ImageNet, CNN, mAP 등)는 번역·음차하지 말고 원문 그대로 둔다. 인명도 한글로 음차하지 않는다 — Shannon, Vaswani, Hinton은 '섀넌·바스와니·힌턴'이 아니라 Shannon·Vaswani·Hinton 그대로 쓴다. 소유격은 원문에 조사만 붙인다: "Shannon's theorem" → "Shannon의 정리"(O), "섀넌의 정리"(X).
6. 마크다운 표기(# 제목, **굵게**, *기울임*, | 표 |, - 목록)는 구조를 그대로 유지한 채 텍스트만 번역한다.
7. 내용을 추가·요약·생략하지 않는다. 문장 수를 유지하려 애쓰되 자연스러운 한국어가 우선이다.
8. 외래어는 국립국어원 외래어 표기법을 따른다 (데이터, 애플리케이션, 콘텐츠, 메시지).
9. 출력은 번역문만 낸다 — 설명, 주석, 인사말, 코드펜스 금지.

번역 예시 (문체 기준 — 소형 모델의 합쇼체 이탈 방지용 few-shot, 실측 A/B로 검증됨):
원문: This paper proposes a new method. We evaluate it on three datasets.
번역: 본 논문은 새로운 방법을 제안한다. 우리는 이를 세 개의 데이터셋에서 평가하였다."""

SYSTEM_GLOSSARY = """당신은 학술 논문 번역을 위한 용어집 편집자다. 논문의 개요와 용어 후보 목록을 보고, 각 용어의 한국어 번역 정책을 정한다.

정책 분류:
- "A": 고유명사·모델명·데이터셋명·약어 — 원문 그대로 유지 (ko = 원문과 동일하게)
- "B": 확립된 한국어 학술 용어가 있음 — 그 용어 사용 (예: gradient descent → 경사 하강법)
- "C": 관례상 음차 표기 — 외래어 표기법에 맞는 음차 (예: attention → 어텐션)
- "D": 신조어·해당 분야 밖에서 낯선 용어 — 음차 또는 번역 + 첫 등장 병기 대상

JSON 배열만 출력한다. 다른 텍스트 금지:
[{"src": "원어", "ko": "역어", "policy": "A|B|C|D"}, ...]"""


def build_unit_prompt(
    masked_src: str,
    glossary_pairs: list[tuple[str, str]],
    first_terms: list[tuple[str, str]],
    context_tail: str | None = None,
    keep_terms: list[str] | None = None,
) -> str:
    """유닛 하나의 user 메시지. glossary_pairs/first_terms/keep_terms는 이 유닛에
    실제로 등장하는 용어만 추려서 넘긴다 (프롬프트 비대화 방지).

    keep_terms(policy A 원형): 규칙 5만으로는 산문 속 약어를 풀어쓰는 사례가
    실측됨(MSE→"평균 제곱 오차") — 유닛별 명시 목록이 강제력을 만든다."""
    parts: list[str] = []
    if keep_terms:
        lines = "\n".join(f"- {t}" for t in keep_terms)
        parts.append(f"[원문 유지 — 다음 표기는 번역·풀어쓰기·음차 없이 그대로 쓴다]\n{lines}")
    if glossary_pairs:
        lines = "\n".join(f"- {s} → {k}" for s, k in glossary_pairs)
        parts.append(f"[용어집 — 반드시 이 역어 사용]\n{lines}")
    if first_terms:
        lines = "\n".join(f"- {s} → {k}({s})" for s, k in first_terms)
        parts.append(f"[첫 등장 병기 — 이번에만 역어(원어) 형태로]\n{lines}")
    if context_tail:
        parts.append(f"[직전 문맥 — 참고만 하고 번역하지 말 것]\n{context_tail}")
    parts.append(f"[번역할 원문]\n{masked_src}")
    return "\n\n".join(parts)


def build_repair_prompt(masked_src: str, broken_output: str, bad_tags: list[str]) -> str:
    """플레이스홀더 누락·중복 수정 패스 user 메시지 — SYSTEM_TRANSLATE와 함께 쓴다.

    재번역이 아니라 태그 수리다: 원문(태그 포함)과 방금 나온 깨진 번역문을 함께 주고,
    번역 내용은 유지한 채 태그만 원문과 일치시키도록 요구한다. 첫 줄 헤더는 한 줄로
    유지한다(테스트·파서가 마스킹 원문을 안정적으로 되뽑을 수 있게)."""
    tags = " ".join(bad_tags[:12])
    return (
        f"[원문 — 아래 꺾쇠 태그가 정답이다]\n{masked_src}\n\n"
        f"[수정할 번역문]\n{broken_output}\n\n"
        f"다음 꺾쇠 태그가 누락되었거나 중복되었다: {tags}\n"
        "번역 내용은 그대로 두고 태그만 바로잡아, 모든 꺾쇠 태그가 원문과 똑같이 "
        "(속성 포함) 각각 정확히 한 번씩 나타나게 하라. 수정한 번역문만 출력한다."
    )


def build_glossary_prompt(
    title_and_abstract: str,
    headings: list[str],
    candidates: list[str],
) -> str:
    heads = "\n".join(f"- {h}" for h in headings[:40])
    cands = "\n".join(f"- {c}" for c in candidates[:80])
    return (
        f"[논문 개요]\n{title_and_abstract[:2000]}\n\n"
        f"[섹션 제목]\n{heads}\n\n"
        f"[용어 후보 — 각각 policy와 역어를 정하라]\n{cands}"
    )
