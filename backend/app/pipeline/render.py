"""마크다운 → HTML 프래그먼트 (서버사이드 렌더).

raw HTML은 비활성(html=False)이라 OCR 결과에 악성 태그가 섞여도 이스케이프된다.
예외적으로 **신뢰 경로에서 서버가 생성**하는 것만 복원/주입한다:
- 표: 모델이 HTML `<table>`로 출력 → 구조 태그만(숫자 colspan/rowspan 포함) 복원
- 수식: 모델의 `\\( … \\)` / `\\[ … \\]`(실측 형태, 아래 참조)를 $-델리미터로
  정규화한 뒤 dollarmath로 파싱하고, tex를 **이스케이프한** `.math-inline` /
  `.math-display` 요소로 출력 — 최종 타이포셋은 클라이언트 KaTeX가 수행한다.

정규화는 렌더 레이어에서만 일어난다. result.md(다운로드 소스)는 모델 원본
LaTeX 델리미터를 그대로 유지한다 (포터빌리티 계약, ARCHITECTURE.md 전역 제약).
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from mdit_py_plugins.dollarmath import dollarmath_plugin


def _render_math_inline(self, tokens, idx, options, env) -> str:
    return f'<span class="math-inline">{escapeHtml(tokens[idx].content)}</span>'


def _render_math_block(self, tokens, idx, options, env) -> str:
    return f'<div class="math-display">{escapeHtml(tokens[idx].content)}</div>'


_md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
_md.enable(["table", "strikethrough"])
# allow_space=True: 모델이 `\( [10, 30] \)`처럼 공백을 끼워 넣는 실측 케이스 허용
_md.use(dollarmath_plugin, allow_space=True, double_inline=False)
_md.add_render_rule("math_inline", _render_math_inline)
_md.add_render_rule("math_block", _render_math_block)

# 표 구조 태그만 복원한다. 여는 태그의 **임의 속성**(border/style/class/onclick 등)은
# 전부 버리고 colspan/rowspan(숫자)만 유지한다 — OvisOCR2처럼 모델이 `<table border="1">`
# 로 속성을 붙여도 여는 태그가 통째로 이스케이프돼 표가 깨지던 것을 고친다.
# 태그명 **직후에 경계**(공백/`/`/`&gt;`)를 룩어헤드로 강제한다 — 이게 없으면 `<threshold>`·
# `<trace>` 같은 본문/코드 플레이스홀더의 접두 `th`/`tr`가 표 태그로 오인돼 가운데 텍스트가
# 소리없이 삭제된다. 속성은 태그 경계(&gt;/&lt;)를 넘지 않는 tempered-dot으로 300자까지
# 소거 대상으로 잡는다(백트래킹·폭탄 방어). 원본 속성이 그대로 통과하지 않으므로 XSS-safe.
_TABLE_TAG = re.compile(
    r"&lt;(/?)(table|thead|tbody|tr|th|td)(?=[\s/]|&gt;)"
    r"((?:(?!&gt;|&lt;).){0,300}?)"
    r"\s*/?&gt;"
)
# 진짜 속성은 앞에 공백이 있다 — data-colspan/x-rowspan 같은 접미 속성을 오승격하지 않게
# 선행 공백을 요구한다(구분자 뒤 워드경계만으로는 `-colspan`도 매칭됐다).
_SAFE_TABLE_ATTR = re.compile(r"(?<=\s)(colspan|rowspan)=&quot;(\d{1,3})&quot;")


def _restore_table_tags(html: str) -> str:
    def _repl(m: re.Match) -> str:
        slash, tag, attrs = m.groups()
        safe = ""
        if not slash and attrs:  # 닫는 태그엔 속성이 없다
            for name, val in _SAFE_TABLE_ATTR.findall(attrs):
                safe += f' {name}="{val}"'
        return f"<{slash}{tag}{safe}>"

    return _TABLE_TAG.sub(_repl, html)


# ── 수식 델리미터 정규화 (렌더 전용) ──────────────────────────────────
# 엔진별 수식 표기가 다르다 (둘 다 지원해야 한다):
#   Unlimited-OCR: 인라인 `\( … \)`, 디스플레이 `\[ … \]`
#   OvisOCR2·PaddleOCR-VL: 인라인 `$ … $`, 디스플레이 `$$ … $$` (표준 LaTeX)
# `$`는 통화($5)와 수식이 모두 쓰는 모호한 문자다 — `$$`는 항상 수식으로, `$…$`는
# 내용이 LaTeX스러울 때(\^_{} 포함)만 수식으로, 그 외 bare `$`는 통화로 이스케이프한다.
# 결과 md(result.md)는 원본 표기를 그대로 보존하고, 변환은 렌더에서만 한다.
_CODE_REGION = re.compile(
    r"^```.*?^```[ \t]*$|^~~~.*?^~~~[ \t]*$|`[^`\n]+`",
    re.DOTALL | re.MULTILINE,
)
_MATH_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_MATH_DOLLAR_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_MATH_DOLLAR_INLINE = re.compile(r"\$([^$\n]+?)\$")
_MATH_LIKE = re.compile(r"[\\^_{}]")  # LaTeX 명령/첨자
_MASK_FMT = "\x00MDMASK{}\x00"


def _is_inline_dollar_math(tex: str) -> bool:
    """`$…$` 내용이 수식인가 — LaTeX스럽거나(단항 포함), 통화가 아닌 짧은 변수식.

    구분: 수식 변수($T·$x·$\\tau)는 문자/기호로 시작하고, 통화($5·$10·$5 그리고·
    $5 million)는 숫자로 시작한다. 숫자+변수($2x, LaTeX 없음)는 드물어 리터럴로 두는
    편이 안전하다 — KaTeX 오류보다 원문 텍스트가 낫다. 긴 산문은 길이로 배제."""
    tex = tex.strip()
    if _MATH_LIKE.search(tex):
        return True
    return bool(tex) and not tex[0].isdigit() and len(tex) <= 40


def _normalize_math_delimiters(md_text: str) -> str:
    """엔진별 수식 표기(`\\(..\\)`/`\\[..\\]` 및 `$..$`/`$$..$$`)를 dollarmath 대상으로
    정규화한다. 코드 구간은 마스킹, 통화용 bare `$`는 이스케이프해 오탐을 막는다."""
    masked: list[str] = []

    def _mask_literal(text: str) -> str:
        masked.append(text)
        return _MASK_FMT.format(len(masked) - 1)

    # 1) 코드펜스/인라인 코드 보호
    md_text = _CODE_REGION.sub(lambda m: _mask_literal(m.group(0)), md_text)

    # 2) 모델이 `$$`/`$`로 낸 수식을 **통화 이스케이프 전에** 마스킹(Ovis/Paddle).
    #    $$는 항상 수식, $…$는 LaTeX스러운 내용일 때만(그 외는 통화로 남겨 이스케이프).
    def _mask_dollar_display(m: re.Match) -> str:
        tex = m.group(1).strip()
        return _mask_literal(f"\n\n$$\n{tex}\n$$\n\n") if tex else ""

    def _mask_dollar_inline(m: re.Match) -> str:
        tex = m.group(1).strip()
        if not tex or not _is_inline_dollar_math(tex):
            return m.group(0)  # 통화 등 — 마스킹하지 않고 아래에서 이스케이프되게 둔다
        return _mask_literal(f"${tex}$")

    md_text = _MATH_DOLLAR_DISPLAY.sub(_mask_dollar_display, md_text)
    md_text = _MATH_DOLLAR_INLINE.sub(_mask_dollar_inline, md_text)

    # 3) 남은 bare `$`(통화)는 이스케이프 — 이 함수가 만든 $-델리미터만 수식이 된다
    md_text = md_text.replace("$", "\\$")

    # 4) Unlimited의 `\(..\)`/`\[..\]` → `$..$`/`$$..$$`
    def _display(m: re.Match) -> str:
        tex = m.group(1).strip()
        return f"\n\n$$\n{tex}\n$$\n\n" if tex else ""

    def _inline(m: re.Match) -> str:
        tex = m.group(1).strip()
        return f"${tex}$" if tex else ""

    md_text = _MATH_DISPLAY.sub(_display, md_text)
    md_text = _MATH_INLINE.sub(_inline, md_text)

    # 5) 마스킹 복원 (코드 + $$/$ 수식)
    for i, original in enumerate(masked):
        md_text = md_text.replace(_MASK_FMT.format(i), original)
    return md_text


# ── 플레인 텍스트 + 수식 스팬 (마크다운이 아닌 문맥용 — 레이아웃 뷰 등) ──
_MATH_ANY = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.DOTALL)


def text_with_math_html(text: str) -> str:
    """플레인 텍스트를 전부 이스케이프하되 `\\(..\\)`/`\\[..\\]` 구간은
    KaTeX 대상 `.math-inline`/`.math-display` 스팬으로 변환한다."""
    out: list[str] = []
    pos = 0
    for m in _MATH_ANY.finditer(text):
        out.append(escapeHtml(text[pos:m.start()]))
        display_tex, inline_tex = m.group(1), m.group(2)
        tex = (display_tex if display_tex is not None else inline_tex).strip()
        if tex:
            cls = "math-display" if display_tex is not None else "math-inline"
            out.append(f'<span class="{cls}">{escapeHtml(tex)}</span>')
        pos = m.end()
    out.append(escapeHtml(text[pos:]))
    return "".join(out)


# ── figure 상대 폭 주입 (렌더 후처리 — result.md/원문 불변) ───────────
# 벤더 P13이 export한 boxes.json(픽셀 bbox + 페이지 크기)으로 각 figure를
# 원본 페이지 대비 상대 폭으로 표시. 값은 전부 서버가 계산한 숫자라 안전하다.
_IMG_TAG = re.compile(r'<img src="([^"]+/images/([^"/]+))" alt="([^"]*)"\s*/?>')
_CENTER_THRESHOLD = 0.6
_MIN_REL_W = 0.08


def _inject_figure_widths(html: str, figure_boxes: dict) -> str:
    def _repl(m: re.Match) -> str:
        src, name, alt = m.groups()
        meta = figure_boxes.get(name)
        if not isinstance(meta, dict):
            return m.group(0)
        try:
            rel_w = (float(meta["x2"]) - float(meta["x1"])) / float(meta["image_width"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return m.group(0)
        if not (0 < rel_w <= 1.5):  # 비정상 메타는 무시하고 풀폭 폴백
            return m.group(0)
        rel_w = min(max(rel_w, _MIN_REL_W), 1.0)
        style = f"width:{rel_w * 100:.1f}%;height:auto;"
        if rel_w < _CENTER_THRESHOLD:
            style += "display:block;margin-left:auto;margin-right:auto;"
        return f'<img src="{src}" alt="{alt}" style="{style}">'

    return _IMG_TAG.sub(_repl, html)


def render_markdown_html(
    markdown_text: str, files_base_url: str, figure_boxes: dict | None = None
) -> str:
    """`![](images/...)` 상대 참조를 잡 파일 서빙 URL로 재작성해 렌더.
    figure_boxes(images/boxes.json)가 있으면 figure에 원본 상대 폭을 주입한다."""
    html = _md.render(_normalize_math_delimiters(markdown_text))
    html = _restore_table_tags(html)
    html = html.replace('src="images/', f'src="{files_base_url}/images/')
    if figure_boxes:
        html = _inject_figure_widths(html, figure_boxes)
    return html


def render_document_html(
    markdown_text: str,
    files_base_url: str,
    figure_boxes: dict | None = None,
    page_separator: str = "\n\n---\n\n",
) -> str:
    """최종 문서 렌더(/html 전용): 페이지 경계를 `<section class="doc-page">`로 승격.

    소스(result.md)는 포터빌리티를 위해 `---` 구분자를 유지하고, 경계 해석은
    렌더에서만 한다. 본문이 우연히 구분자와 동일한 텍스트를 포함하면 초과
    분할될 수 있는 best-effort 휴리스틱 (실측 코퍼스에서 미관측).
    라이브 프리뷰(/render-preview)는 기존 flat 렌더를 그대로 쓴다.
    """
    if not markdown_text.strip():
        return ""
    segments = markdown_text.split(page_separator) if page_separator else [markdown_text]
    if len(segments) == 1:
        return render_markdown_html(markdown_text, files_base_url, figure_boxes)
    parts = []
    for i, seg in enumerate(segments, start=1):
        inner = render_markdown_html(seg, files_base_url, figure_boxes)
        parts.append(f'<section class="doc-page" data-page="{i}">\n{inner}</section>')
    return "\n".join(parts)
