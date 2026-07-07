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

from . import prompts
from .client import OpenAICompatClient
from .glossary import Glossary, build_glossary
from .masking import mask, should_skip, unmask
from .segment import apply_layout, assemble_markdown, layout_units, split_markdown
from .types import PROMPT_V, TranslateConfig, TranslateError, TranslateResult, cache_key


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
        retried_n = 0

        def translate_unit(u):
            masked, mapping = mask(u.src)
            pairs, first = glossary.for_unit(u.src, u.id)
            key = cache_key(masked, cfg.model, pairs + first)
            if not force and key in cache:
                return u, cache[key], "cached", key, False
            ctx = context_map.get(u.id) if cfg.context else None
            prompt = prompts.build_unit_prompt(masked, pairs, first, context_tail=ctx)
            max_toks = min(8000, max(384, len(masked) // 2 + 300))
            out = client.complete(prompts.SYSTEM_TRANSLATE, prompt, max_tokens=max_toks)
            restored, missing, dup = unmask(out, mapping)
            if missing or dup:
                # 플레이스홀더 소실 → 강조 지시 붙여 1회 재시도
                suffix = prompts.build_retry_suffix(missing + dup)
                try:
                    out2 = client.complete(prompts.SYSTEM_TRANSLATE, prompt + suffix, max_tokens=max_toks)
                    restored2, missing2, dup2 = unmask(out2, mapping)
                    if not missing2 and not dup2 and restored2.strip():
                        return u, restored2, "translated", key, True
                except TranslateError:
                    pass
                return u, u.src, "kept", key, True  # 원문 유지
            if not restored.strip():
                return u, u.src, "kept", key, False
            return u, restored, "translated", key, False

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
                    u, text, status, key, was_retried = fut.result()
                    results[u.id] = text
                    if was_retried:
                        retried_n += 1
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
