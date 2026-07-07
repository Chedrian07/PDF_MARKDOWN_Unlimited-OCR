"""번역 오케스트레이터 — 공개 진입점 run_translation().

api.py는 이 함수만 안다. 워커 스레드에서 블로킹 호출되며, 진행률은
progress 콜백으로, 중단은 cancel 이벤트로 통신한다 (OCR 워커와 같은 패턴).

상태 전이: state.json을 이 함수가 직접 기록한다 —
  running(current/total 갱신) → done | error(message) | canceled
호출자는 예외를 SSE error 이벤트로만 중계하면 된다.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import re

from . import prompts
from .client import OpenAICompatClient
from .glossary import Glossary, build_glossary
from .masking import mask, sanitize_translation, should_skip, unmask
from .segment import apply_layout, assemble_markdown, layout_units, split_markdown
from .types import PROMPT_V, TranslateConfig, TranslateError, TranslateResult, cache_key

# 문장 경계 — 종결부호 뒤 공백. 분할 지점·분할 가능 판정에 함께 쓴다.
_SENT_BOUND_RE = re.compile(r"(?<=[.!?…])\s+")
# 구조 유닛 감지 — 줄 시작이 표(|)·목록(-, *)·인용(>)·번호목록(숫자.)·펜스(```)인 줄.
_STRUCT_LINE_RE = re.compile(r"^\s*(?:[|>*-]|\d+\.|```)")


def _splittable(src: str) -> bool:
    """문장 분할 재시도 대상인가 — 여러 문장이고 구조 유닛(표·목록·인용·펜스)이 아니면 True."""
    if not _SENT_BOUND_RE.search(src):
        return False  # 문장 경계가 없으면 나눌 수 없다
    return not any(_STRUCT_LINE_RE.match(ln) for ln in src.split("\n"))


def _split_two(src: str) -> tuple[str, str] | None:
    """문장 경계 중 중앙에 가장 가까운 지점에서 2분할. (앞, 뒤) 또는 None(경계 없음/한쪽 공백)."""
    bounds = [m.end() for m in _SENT_BOUND_RE.finditer(src)]
    if not bounds:
        return None
    mid = len(src) / 2
    cut = min(bounds, key=lambda b: abs(b - mid))
    left, right = src[:cut].rstrip(), src[cut:].lstrip()
    if not left or not right:
        return None
    return left, right


# HTML 표 유닛 — 문장 분할 대신 행(</tr>) 경계에서 자른다. 초대형 표(실측 6.4KB,
# 플레이스홀더 수십 개)는 한 번에 번역·repair가 모두 실패하는 유일한 유형이었다.
_TABLE_ROW_END_RE = re.compile(r"</tr\s*>", re.I)


def _is_table_unit(src: str) -> bool:
    return src.lstrip().lower().startswith("<table")


def _split_table(src: str) -> tuple[str, str] | None:
    """`</tr>` 경계 중 중앙에 가장 가까운 지점에서 2분할. 재결합은 단순 이어붙임 —
    행 사이 공백은 HTML 렌더에 무의미하므로 구조가 정확히 보존된다."""
    bounds = [m.end() for m in _TABLE_ROW_END_RE.finditer(src)]
    if len(bounds) < 2:
        return None  # 행이 하나뿐이면 나눠도 의미 없음
    mid = len(src) / 2
    # 마지막 </tr> 뒤에서 자르면 오른쪽이 </table>뿐이 된다 — 마지막 경계는 제외
    cut = min(bounds[:-1], key=lambda b: abs(b - mid))
    left, right = src[:cut], src[cut:]
    if not left.strip() or not right.strip():
        return None
    return left, right


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1))


def run_translation(
    job_dir: Path,
    lang: str,
    cfg: TranslateConfig,
    *,
    page_separator: str = "\n\n---\n\n",
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    force: bool = False,  # True면 유닛 캐시를 읽지 않고 전부 재번역 (쓰기는 함)
    client=None,  # 테스트 주입용 — None이면 cfg로 OpenAICompatClient 생성
) -> TranslateResult:
    """job_dir의 result.md(+ 있으면 layout.json)를 lang으로 번역한다.

    산출물·캐시·상태 파일 계약은 types.py 모듈 docstring 참조.
    실패 시 TranslateError(사용자 표시용 한국어 메시지)를 던지고
    state.json에 error를 남긴다. 취소 시 부분 캐시는 보존된다.
    """
    job_dir = Path(job_dir)
    tdir = job_dir / "translations" / lang
    tdir.mkdir(parents=True, exist_ok=True)  # error 상태도 기록할 수 있도록 선행 생성
    started = _now()
    total = 0
    done = 0

    def write_state(status: str, current: int, total_: int, error: str | None = None) -> None:
        mode = getattr(client, "api_mode_used", "") or (cfg.api_mode if cfg.api_mode != "auto" else "")
        _atomic_write_json(tdir / "state.json", {
            "lang": lang,
            "status": status,
            "current": current,
            "total": total_,
            "error": error,
            "model": cfg.model,
            "api_mode": mode,
            "prompt_v": PROMPT_V,
            "started_at": started,
            "finished_at": _now() if status in ("done", "error", "canceled") else None,
        })

    try:
        # POST 직후 SSE 접속(404 방지)을 위해 유닛 집계 전이라도 즉시 running을 남긴다.
        write_state("running", 0, 0)
        result_md = job_dir / "result.md"
        if not result_md.is_file():
            raise TranslateError("번역할 결과가 없습니다 — 변환이 완료된 잡인지 확인하세요")
        md_text = result_md.read_text(encoding="utf-8")

        layout_path = job_dir / "layout.json"
        layout_pages = None
        if layout_path.is_file():
            try:
                loaded = json.loads(layout_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    layout_pages = loaded
            except Exception:
                layout_pages = None

        # 유닛 분리 — md(문서 순서) + layout
        md_units = split_markdown(md_text, page_separator)
        lay_units = layout_units(layout_pages) if layout_pages else []
        all_units = md_units + lay_units

        targets = []
        skipped = 0
        for u in all_units:
            if u.skip_reason or should_skip(u.src):
                skipped += 1
            else:
                targets.append(u)
        total = len(targets)

        # 직전 유닛 꼬리 컨텍스트 (같은 소스 내에서만; 프롬프트 참고용, 캐시 키엔 무영향)
        context_map: dict[str, str] = {}
        if cfg.context:
            for seq in (md_units, lay_units):
                prev = None
                for u in seq:
                    if prev is not None:
                        context_map[u.id] = prev.src[-200:]
                    prev = u

        if client is None:
            client = OpenAICompatClient(cfg)
        write_state("running", 0, total)

        # 용어집 — 있으면 로드(캐시 안정), 없거나 force면 빌드 후 저장
        gpath = tdir / "glossary.json"
        if gpath.is_file() and not force:
            glossary = Glossary.load(gpath)
        else:
            glossary = build_glossary(md_text, md_units, client, cfg)
            glossary.save(gpath)

        # 유닛 캐시 (dict: cache_key → 번역문)
        upath = tdir / "units.json"
        cache: dict[str, str] = {}
        if upath.is_file():
            try:
                loaded = json.loads(upath.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cache = loaded
            except Exception:
                cache = {}

        warnings = list(glossary.warnings)
        results: dict[str, str] = {}
        kept_original: list[str] = []
        translated_n = 0
        cached_n = 0
        retried_n = 0    # 최초 패스 실패로 래더(repair/분할)에 진입한 유닛 수
        repaired_n = 0   # repair 패스로 복구된 유닛 수
        split_n = 0      # 문장 분할로 복구된 유닛 수
        sanitized_n = 0  # sanitize 치환 총건수 (모든 complete 출력 경로 합산)

        def _max_toks(masked: str) -> int:
            return min(8000, max(384, len(masked) // 2 + 300))

        def _run_pass(prompt: str, max_toks: int, mapping: dict) -> tuple[str, list, list, int, str]:
            """complete → sanitize → unmask 한 번. (복원문, missing, dup, 치환건수, 정리된_원출력)."""
            raw = client.complete(prompts.SYSTEM_TRANSLATE, prompt, max_tokens=max_toks)
            clean, sc = sanitize_translation(raw)
            restored, missing, dup = unmask(clean, mapping)
            return restored, missing, dup, sc, clean

        def _translate_fragment(src, pairs, first, ctx, stats, keep=None) -> str | None:
            """분할된 반쪽 하나를 독립 번역 — mask→complete→sanitize→unmask + repair 1회(추가 분할 없음).

            성공 시 복원문, 실패 시 None. sanitize 건수만 stats에 누적한다. 반쪽 단계의
            API 오류는 무손실 원칙상 치명적이지 않으므로 None 처리(원 유닛 원문 유지로 귀결)."""
            masked, mapping = mask(src)
            max_toks = _max_toks(masked)
            prompt = prompts.build_unit_prompt(masked, pairs, first, context_tail=ctx, keep_terms=keep)
            try:
                restored, missing, dup, sc, clean = _run_pass(prompt, max_toks, mapping)
            except TranslateError:
                return None
            stats["sanitized"] += sc
            if not missing and not dup and restored.strip():
                return restored
            try:
                rprompt = prompts.build_repair_prompt(masked, clean, missing + dup)
                r_restored, r_missing, r_dup, r_sc, _ = _run_pass(rprompt, max_toks, mapping)
                stats["sanitized"] += r_sc
                if not r_missing and not r_dup and r_restored.strip():
                    return r_restored
            except TranslateError:
                pass
            return None

        def translate_unit(u):
            masked, mapping = mask(u.src)
            pairs, first = glossary.for_unit(u.src, u.id)
            keep = glossary.keep_terms(u.src)
            # keep(A 원형)도 출력 정책을 바꾸므로 캐시 키에 포함 — (k, k) 쌍으로 해시
            key = cache_key(masked, cfg.model, pairs + first + [(k, k) for k in keep])
            stats = {"retried": 0, "repaired": 0, "split": 0, "sanitized": 0}
            if not force and key in cache:
                return u, cache[key], "cached", key, stats
            ctx = context_map.get(u.id) if cfg.context else None
            max_toks = _max_toks(masked)
            prompt = prompts.build_unit_prompt(masked, pairs, first, context_tail=ctx, keep_terms=keep)

            # 0) 최초 패스 — complete→sanitize→unmask. 태그 완전하면 즉시 성공.
            #    (step 0의 API 오류는 잡 전체 실패로 전파 — 기존 계약 유지)
            restored, missing, dup, sc, clean = _run_pass(prompt, max_toks, mapping)
            stats["sanitized"] += sc
            if not missing and not dup and restored.strip():
                return u, restored, "translated", key, stats

            # 여기부터 신뢰도 래더 — 태그 누락·중복 또는 빈 출력
            stats["retried"] = 1

            # 1) repair 패스 — 원문(태그 포함)+깨진 번역문을 주고 태그만 바로잡게 한다.
            try:
                rprompt = prompts.build_repair_prompt(masked, clean, missing + dup)
                r_restored, r_missing, r_dup, r_sc, _ = _run_pass(rprompt, max_toks, mapping)
                stats["sanitized"] += r_sc
                if not r_missing and not r_dup and r_restored.strip():
                    stats["repaired"] = 1
                    return u, r_restored, "translated", key, stats
            except TranslateError:
                pass

            # 2a) HTML 표 유닛 — </tr> 행 경계 분할 (깊이 2 = 최대 4분할).
            #     초대형 표는 반쪽도 실패할 수 있어 재귀 한 단계를 더 허용한다.
            if _is_table_unit(u.src):
                def _table_part(src: str, depth: int) -> str | None:
                    got = _translate_fragment(src, pairs, first, None, stats, keep)
                    if got is not None or depth <= 0:
                        return got
                    sub = _split_table(src)
                    if sub is None:
                        return None
                    a = _table_part(sub[0], depth - 1)
                    if a is None:
                        return None
                    b = _table_part(sub[1], depth - 1)
                    return None if b is None else a + b

                ts = _split_table(u.src)
                if ts is not None:
                    left = _table_part(ts[0], 1)
                    right = _table_part(ts[1], 1) if left is not None else None
                    if left is not None and right is not None:
                        stats["split"] = 1
                        return u, left + right, "translated", key, stats

            # 2b) 문장 분할 재시도(깊이 1) — 여러 문장·비구조 유닛만. 양쪽 성공 시 " "로 결합.
            elif _splittable(u.src):
                halves = _split_two(u.src)
                if halves is not None:
                    left_src, right_src = halves
                    left = _translate_fragment(left_src, pairs, first, ctx, stats, keep)
                    if left is not None:
                        # 뒷반 컨텍스트: 앞반 src의 꼬리 200자 (컨텍스트 비활성 시 생략)
                        right_ctx = left_src[-200:] if cfg.context else None
                        right = _translate_fragment(right_src, pairs, first, right_ctx, stats, keep)
                        if right is not None:
                            stats["split"] = 1
                            return u, left + " " + right, "translated", key, stats

            # 3) 최종 실패 → 원문 유지 (무손실 원칙 불변)
            return u, u.src, "kept", key, stats

        # 취소: 디스패치 전 선체크
        if cancel is not None and cancel.is_set():
            _atomic_write_json(upath, cache)
            write_state("canceled", done, total)
            return TranslateResult(
                status="canceled", total=total, translated=0, cached=0,
                kept_original=[], skipped=skipped,
                api_mode=getattr(client, "api_mode_used", "") or cfg.api_mode,
            )

        canceled = False
        with cf.ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as ex:
            futures = {}
            for u in targets:
                if cancel is not None and cancel.is_set():
                    canceled = True
                    break
                futures[ex.submit(translate_unit, u)] = u

            if not canceled:
                for fut in cf.as_completed(futures):
                    if cancel is not None and cancel.is_set():
                        canceled = True
                        for f in futures:
                            f.cancel()
                        break
                    u, text, status, key, stats = fut.result()
                    results[u.id] = text
                    retried_n += stats["retried"]
                    repaired_n += stats["repaired"]
                    split_n += stats["split"]
                    sanitized_n += stats["sanitized"]
                    if status == "cached":
                        cached_n += 1
                    elif status == "translated":
                        translated_n += 1
                        cache[key] = text
                    elif status == "kept":
                        kept_original.append(u.id)
                    done += 1
                    if progress is not None:
                        progress(done, total)
                    if done % 10 == 0:
                        _atomic_write_json(upath, cache)
                        # SSE 없이 state 폴링으로 보는 클라이언트(프런트 폴백)용 진행률
                        write_state("running", done, total)

        _atomic_write_json(upath, cache)

        if canceled:
            write_state("canceled", done, total)
            return TranslateResult(
                status="canceled", total=total, translated=translated_n, cached=cached_n,
                kept_original=kept_original, skipped=skipped,
                api_mode=getattr(client, "api_mode_used", "") or cfg.api_mode,
            )

        # 조립 — 번역된 유닛만 교체(나머지 원문 보존)
        md_trans = {u.id: results[u.id] for u in md_units if u.id in results}
        assembled = assemble_markdown(md_text, page_separator, md_trans)
        _atomic_write(job_dir / f"result.{lang}.md", assembled)

        if layout_pages is not None:
            lay_trans = {u.id: results[u.id] for u in lay_units if u.id in results}
            new_pages = apply_layout(layout_pages, lay_trans)
            _atomic_write(job_dir / f"layout.{lang}.json", json.dumps(new_pages, ensure_ascii=False))

        api_mode = getattr(client, "api_mode_used", "") or cfg.api_mode
        _atomic_write_json(tdir / "report.json", {
            "kept_original": kept_original,
            "retried": retried_n,
            "repaired": repaired_n,
            "split": split_n,
            "sanitized": sanitized_n,
            "skipped": skipped,
            "cached": cached_n,
            "translated": translated_n,
            "api_mode": api_mode,
            "warnings": warnings,
        })
        write_state("done", total, total)
        return TranslateResult(
            status="done", total=total, translated=translated_n, cached=cached_n,
            kept_original=kept_original, skipped=skipped, api_mode=api_mode,
        )

    except TranslateError as e:
        write_state("error", done, total, error=str(e))
        raise
    except Exception as e:  # noqa: BLE001 — 사용자용 메시지로 감싸 재발생
        write_state("error", done, total, error=f"번역 중 오류: {e}")
        raise TranslateError(f"번역 중 오류: {e}") from e
