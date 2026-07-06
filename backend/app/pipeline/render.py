"""마크다운 → HTML 프래그먼트 (서버사이드 렌더).

raw HTML은 비활성(html=False)이라 OCR 결과에 악성 태그가 섞여도 이스케이프된다.
단, Unlimited-OCR은 표를 HTML `<table>`로 출력하므로 구조적 표 태그만
(colspan/rowspan 숫자 속성 포함) 선별적으로 복원한다 — 그 외 태그/속성은
이스케이프 상태를 유지해 XSS를 차단한다.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
_md.enable(["table", "strikethrough"])

_TABLE_TAG = re.compile(
    r"&lt;(/?)(table|thead|tbody|tr|th|td)"
    r"((?:\s+(?:colspan|rowspan)=&quot;\d{1,3}&quot;)*)\s*&gt;"
)


def _restore_table_tags(html: str) -> str:
    def _repl(m: re.Match) -> str:
        slash, tag, attrs = m.groups()
        attrs = attrs.replace("&quot;", '"') if attrs else ""
        return f"<{slash}{tag}{attrs}>"

    return _TABLE_TAG.sub(_repl, html)


def render_markdown_html(markdown_text: str, files_base_url: str) -> str:
    """`![](images/...)` 상대 참조를 잡 파일 서빙 URL로 재작성해 렌더."""
    html = _md.render(markdown_text)
    html = _restore_table_tags(html)
    return html.replace('src="images/', f'src="{files_base_url}/images/')
