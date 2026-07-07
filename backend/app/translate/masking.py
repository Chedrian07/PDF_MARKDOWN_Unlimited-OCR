"""비언어 토큰 마스킹 — 번역 전 치환, 번역 후 복원·검증.

수식·코드·이미지·URL·인용·참조·HTML 태그는 번역해서는 안 되는 불변 토큰이다.
이들을 `<m1 v="…"/>` 형태의 플레이스홀더로 바꿔 LLM에 넘기고(v는 조사 선택용
미리보기), 번역문에서 원문으로 복원한다. 복원 실패(누락·중복)는 검증에서 보고되며
엔진이 해당 유닛을 원문 유지로 처리한다.

설계 핵심:
  * **단일 패스 결합 정규식** — 우선순위 순서의 alternation 하나로 스캔한다.
    re.sub는 치환 결과를 재스캔하지 않으므로 플레이스홀더가 자기 자신을
    다시 매칭하는 문제가 원천적으로 없다.
  * 플레이스홀더 번호는 종류를 가로질러 전역 1-based로 증가한다 (예: m1 c2 f3).
  * 복원은 관용적 — 슬래시 누락·속성 변형·공백 삽입을 모두 허용한다.
"""

from __future__ import annotations

import re

# ── 결합 토큰 정규식 (우선순위 = alternation 순서) ────────────────────────
# 그룹명 첫 글자가 종류 코드다: m=수식 k=코드 g=이미지 t=태그 u=URL/DOI/이메일
# c=인용 f=참조. DOTALL은 `.`에만 영향 → 펜스/디스플레이 수식만 개행을 넘는다.
_TOKEN_RE = re.compile(
    r"(?P<k1>```.*?```)"                                    # 1 펜스 코드
    r"|(?P<m1>\$\$.*?\$\$)"                                 # 2 디스플레이 수식
    r"|(?P<m2>\$[^$\n]+(?:\n[^$\n]+)?\$)"                   # 3 인라인 수식(개행 1개 허용, 비어있지 않음)
    # 3b LaTeX 델리미터 — render.py 포터빌리티 계약상 result.md는 \(..\)/\[..\]를
    #    원본 그대로 유지하므로 $-정규화 여부와 무관하게 수식으로 보호한다.
    r"|(?P<m3>\\\[.*?\\\])"                                 # 디스플레이 \[..\]
    r"|(?P<m4>\\\(.*?\\\))"                                 # 인라인 \(..\)
    r"|(?P<k2>`[^`\n]+`)"                                   # 4 인라인 코드
    r"|(?P<g1>!\[[^\]]*\]\([^)\s]*\))"                      # 5 이미지
    r"|(?P<t1></?[a-zA-Z][^>]*>)"                           # 6 HTML 태그
    r"|(?P<u1>https?://\S+|\b10\.\d{4,}/\S+|[\w.+%-]+@[\w-]+\.[\w.-]+)"  # 7 URL/DOI/이메일
    r"|(?P<c1>\[\d+(?:\s*[,–-]\s*\d+)*\])"             # 8 인용 [1] [1, 2] [3-5]
    r"|(?P<f1>\b(?:Figure|Fig\.?|Table|Tab\.?|Equation|Eqs?\.?|Section|Sec\.?"
    r"|Appendix|Algorithm|Alg\.?)\s*\(?\d+(?:\.\d+)*\)?)",  # 9 Fig/Table/Eq/Sec 참조
    re.DOTALL,
)

# 복원·잔여 검사에 쓰는 플레이스홀더 인식 패턴 (id 접두 = 종류 코드)
_PLACEHOLDER_RE = re.compile(r"<[mkgucft]\d+\b[^>]*>")
_RESIDUAL_RE = re.compile(r"<[mkgucft]\d+")


def _preview(s: str) -> str:
    """원문 앞 12자 미리보기 — 따옴표·개행·꺾쇠 제거(플레이스홀더 문법 보호)."""
    s = re.sub(r"[\"'\n\r<>]", "", s)
    return s.strip()[:12]


def mask(text: str) -> tuple[str, dict[str, str]]:
    """비언어 토큰을 플레이스홀더로 치환. (masked, {placeholder_id → 원문}) 반환."""
    mapping: dict[str, str] = {}
    counter = [0]

    def _repl(m: re.Match) -> str:
        counter[0] += 1
        kind = m.lastgroup[0]  # 그룹명 첫 글자 = 종류 코드
        original = m.group()
        pid = f"{kind}{counter[0]}"
        mapping[pid] = original
        return f'<{pid} v="{_preview(original)}"/>'

    return _TOKEN_RE.sub(_repl, text), mapping


def _lenient_re(pid: str) -> re.Pattern:
    """관용 복원 패턴 — `<m1>`, `< m1 />`, 속성 변형·슬래시 누락 모두 허용."""
    return re.compile(r"<\s*" + re.escape(pid) + r"\b[^>]*?/?\s*>")


def unmask(translated: str, mapping: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """플레이스홀더를 원문으로 복원. (복원문, missing_ids, dup_ids) 반환.

    각 id는 정확히 1회 등장이 정상 — 0회는 missing, 2회 이상은 dup(전부 복원하되
    실패로 보고). 복원 후에도 남은 `<m1`류 잔여물이 있으면 dup에 추가한다.
    """
    missing: list[str] = []
    dup: list[str] = []
    out = translated
    for pid, original in mapping.items():
        pat = _lenient_re(pid)
        n = len(pat.findall(out))
        if n == 0:
            missing.append(pid)
        elif n >= 2:
            dup.append(pid)
        # 개수와 무관하게 전부 복원 (lambda로 원문의 백슬래시·그룹참조를 리터럴 취급)
        out = pat.sub(lambda _m, _o=original: _o, out)
    for m in _RESIDUAL_RE.finditer(out):
        dup.append(m.group())
    return out, missing, dup


# ── 모델 발명 딜리미터·페이지 마커 새니타이즈 (unmask 이전 원출력에 적용) ─────
# 마스킹된 원문에는 수식 딜리미터가 전혀 없다(전부 플레이스홀더). 따라서 모델
# 출력에 나타나는 `\(` `\[` `$$` 등은 전부 모델이 만들어낸 것이며, 남겨두면
# 수식 카운트 불일치·원문 오염을 일으킨다. 리터럴 <PAGE>도 모델 발명품이다.
# 플레이스홀더 태그의 v 속성 내부에서 치환돼도 무해하다 — unmask는 id로 매칭하고
# 이 치환들은 `<` `>` `/` `=` `"`를 건드리지 않아 태그 구조를 깨지 않는다.
_SANITIZE_SUBS: tuple[tuple[str, str], ...] = (
    ("\\(", "("),
    ("\\)", ")"),
    ("\\[", "["),
    ("\\]", "]"),
    ("$$", ""),      # 이중 달러 제거 (실 수식은 마스킹으로 보호됨)
    ("<PAGE>", ""),  # 리터럴 페이지 마커 제거
)


def sanitize_translation(raw: str) -> tuple[str, int]:
    """모델 발명 수식 딜리미터·리터럴 <PAGE>를 제거. (정리문, 치환 건수) 반환.

    엔진의 모든 complete() 출력 경로(최초·repair·분할 반쪽)에 unmask 직전 적용한다.
    치환들은 서로 겹치는 문자열을 만들지 않으므로 순서·연쇄 재매칭 문제가 없다.
    치환은 전체에 적용하되 **카운트는 플레이스홀더 태그 밖만** 센다 — v 속성의
    수식 미리보기(v="\\( E=mc…")까지 세면 리포트가 실측(25p 784건)처럼 부풀려진다.
    """
    outside = _PLACEHOLDER_RE.sub("", raw)  # 카운트 전용 — 태그(속성 포함) 제거본
    count = 0
    out = raw
    for needle, repl in _SANITIZE_SUBS:
        if needle in out:
            count += outside.count(needle)
            out = out.replace(needle, repl)
    return out, count


# 참고문헌 항목 줄 — "[12] Gersho, A. …" 형태. 헤딩("References") 기반 스킵은
# OCR 출력이 헤딩을 안 뽑으면(실측 2504.19874v1 — 굵은 텍스트/부재) 무력하므로,
# 항목 모양으로 페이지·뷰(md/layout) 무관하게 감지한다.
_REF_LINE_RE = re.compile(r"^\s*\[\d+\]\s+\S")
# 단일 항목일 때 서지 정보 증거 — 연도·arXiv·이니셜("Smith, J.")·권/쪽 표기
_REF_EVIDENCE_RE = re.compile(r"\b(?:19|20)\d{2}\b|arXiv|,\s*[A-Z]\.|\bpp\.\s*\d|\bvol\.\s*\d", re.I)


def _looks_reference_list(text: str) -> bool:
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return False
    hits = sum(1 for ln in lines if _REF_LINE_RE.match(ln))
    if hits >= 2:
        return True  # [n] 항목이 여럿 → 참고문헌 목록
    # 단일 [n] 줄은 서지 증거가 있을 때만 (본문 "[1] shows that…" 오탐 방지)
    return hits == len(lines) == 1 and bool(_REF_EVIDENCE_RE.search(text))


def should_skip(text: str) -> str:
    """번역 불필요 사유를 반환(빈 문자열이면 번역 대상).

    already-korean: 한글이 비공백 문자의 과반 → 이미 번역됨.
    identifier: 전체가 arXiv id 패턴뿐.
    references: 참고문헌 항목 목록 — 서지 정보는 원문 유지가 정책.
    non-linguistic: 마스킹 후 잔여에 2자+ 알파벳 단어가 없거나 영문자 비율 < 0.3.
    """
    stripped = text.strip()
    if not stripped:
        return "non-linguistic"

    if _looks_reference_list(text):
        return "references"

    non_ws = len(re.findall(r"\S", text))
    hangul = len(re.findall(r"[가-힣]", text))
    if non_ws and hangul / non_ws > 0.5:
        return "already-korean"

    if re.fullmatch(r"(?:arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s*)+", stripped):
        return "identifier"

    masked, _ = mask(text)
    residual = _PLACEHOLDER_RE.sub(" ", masked)
    if not re.search(r"[A-Za-z]{2,}", residual):
        return "non-linguistic"
    letters = len(re.findall(r"[A-Za-z]", residual))
    non_ws_r = len(re.findall(r"\S", residual))
    if non_ws_r == 0 or letters / non_ws_r < 0.3:
        return "non-linguistic"
    return ""
