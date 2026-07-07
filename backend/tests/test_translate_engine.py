"""엔진 — 미니 잡 디렉터리에서 조립·캐시·재시도·취소·상태 전이 검증.

torch/OCR 없이 requests+표준 라이브러리만으로 도는지도 함께 확인한다(스텁 클라이언트).
"""

import json
import threading

import pytest

from app.translate.engine import run_translation
from app.translate.types import TranslateConfig, TranslateResult

SEP = "\n\n---\n\n"

RESULT_MD = (
    "# Deep Learning\n\n"
    "We train a model with loss $L = \\sum_i x_i$ over the dataset.\n\n"
    "---\n\n"
    "## Results\n\n"
    "The accuracy improved on the benchmark dataset.\n"
)

LAYOUT = [
    {"page": 1, "width": 1000, "height": 1400, "fonts_v": "2", "blocks": [
        {"type": "title", "bbox": [0, 0, 999, 80], "content": "Deep Learning", "fs": 2.5, "bold": True},
        {"type": "text", "bbox": [0, 100, 999, 300], "content": "We train a model over data.", "fs": 1.78},
        {"type": "image", "bbox": [0, 320, 500, 700], "content": "", "image": "p0001_0.jpg"},
    ]},
    {"page": 2, "width": 1000, "height": 1400, "blocks": [
        {"type": "title", "bbox": [0, 0, 999, 80], "content": "Results", "fs": 2.5},
        {"type": "text", "bbox": [0, 100, 999, 300], "content": "The accuracy improved a lot."},
    ]},
]


def _marker(user: str) -> str | None:
    tag = "[번역할 원문]\n"
    return user.split(tag, 1)[1] if tag in user else None


class EchoClient:
    """[번역할 원문] 섹션을 그대로 반환 → 마스킹 왕복 후 원문과 동일."""

    def __init__(self):
        self.calls = 0
        self.api_mode_used = "chat"

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        src = _marker(user)
        return src if src is not None else ""  # 용어집 프롬프트 → 빈 응답(시드 폴백)


class MarkerClient(EchoClient):
    """각 줄 앞에 § 를 붙임 — 번역 반영 확인용(플레이스홀더는 보존)."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        src = _marker(user)
        if src is None:
            return ""
        return "\n".join("§" + ln for ln in src.split("\n"))


class FaultyClient(EchoClient):
    """플레이스홀더를 떨어뜨림 → 재시도 후 원문 유지되어야 함."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        src = _marker(user)
        if src is None:
            return ""
        import re
        return re.sub(r"<[mkgucft]\d+\b[^>]*>", "", src)


@pytest.fixture
def cfg() -> TranslateConfig:
    return TranslateConfig(
        base_url="https://host/v1", api_key="k", model="test-model",
        api_mode="chat", concurrency=2, temperature="0", max_tokens_param="max_tokens",
        context=False,  # 결정성 위해 컨텍스트 비활성(캐시 키엔 무영향)
    )


@pytest.fixture
def job(tmp_path):
    (tmp_path / "result.md").write_text(RESULT_MD, encoding="utf-8")
    (tmp_path / "layout.json").write_text(json.dumps(LAYOUT, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def _state(job) -> dict:
    return json.loads((job / "translations" / "ko" / "state.json").read_text(encoding="utf-8"))


def test_echo_결과_바이트동일_및_레이아웃_content동일(job, cfg):
    res = run_translation(job, "ko", cfg, client=EchoClient())
    assert isinstance(res, TranslateResult) and res.status == "done"

    # result.ko.md == result.md (바이트 동일)
    assert (job / "result.ko.md").read_text(encoding="utf-8") == RESULT_MD

    # layout.ko.json: content는 원본과 동일, 그 외 필드도 완전 동일
    out = json.loads((job / "layout.ko.json").read_text(encoding="utf-8"))
    assert out == LAYOUT  # Echo가 content를 원문 그대로 되돌리므로 전체 동일

    st = _state(job)
    assert st["status"] == "done" and st["current"] == st["total"] == res.total
    from app.translate.types import PROMPT_V
    assert st["model"] == "test-model" and st["prompt_v"] == PROMPT_V
    assert res.kept_original == [] and res.translated > 0


def test_marker_md와_layout에_반영_필드보존(job, cfg):
    res = run_translation(job, "ko", cfg, client=MarkerClient())
    assert res.status == "done"

    md = (job / "result.ko.md").read_text(encoding="utf-8")
    assert "§" in md
    assert len(md.split(SEP)) == 2                      # 페이지 수 보존
    assert md.count("$L = \\sum_i x_i$") == 1           # 플레이스홀더 복원됨

    out = json.loads((job / "layout.ko.json").read_text(encoding="utf-8"))
    title = out[0]["blocks"][0]
    assert title["content"].startswith("§") and title["content"].endswith("Deep Learning")
    assert title["bbox"] == [0, 0, 999, 80] and title["fs"] == 2.5 and title["bold"] is True
    assert out[0]["fonts_v"] == "2"
    # 이미지 블록은 손대지 않음
    assert out[0]["blocks"][2]["content"] == "" and out[0]["blocks"][2]["image"] == "p0001_0.jpg"


def test_캐시_2회차_호출없음(job, cfg):
    run_translation(job, "ko", cfg, client=EchoClient())
    echo2 = EchoClient()
    res2 = run_translation(job, "ko", cfg, client=echo2)
    assert echo2.calls == 0                    # 용어집 로드 + 전 유닛 캐시 적중
    assert res2.cached == res2.total and res2.translated == 0
    assert (job / "result.ko.md").read_text(encoding="utf-8") == RESULT_MD


def test_faulty_재시도후_원문유지(job, cfg):
    faulty = FaultyClient()
    res = run_translation(job, "ko", cfg, client=faulty)
    assert res.status == "done"
    # 수식 플레이스홀더가 있는 유닛(md:0:1)은 복원 실패 → 원문 유지
    assert "md:0:1" in res.kept_original
    # retried는 report.json에 기록된다 (TranslateResult엔 없음)
    report = json.loads((job / "translations" / "ko" / "report.json").read_text(encoding="utf-8"))
    assert report["retried"] >= 1
    assert report["kept_original"] == res.kept_original
    assert "md:0:1" in report["kept_original"]
    # 결과 md에는 원문 수식이 그대로 남아있어야 함
    assert "$L = \\sum_i x_i$" in (job / "result.ko.md").read_text(encoding="utf-8")


def test_취소_사전set_canceled(job, cfg):
    ev = threading.Event()
    ev.set()
    res = run_translation(job, "ko", cfg, client=EchoClient(), cancel=ev)
    assert res.status == "canceled"
    assert not (job / "result.ko.md").exists()   # 조립 전에 중단
    assert _state(job)["status"] == "canceled"


def test_force_재번역(job, cfg):
    run_translation(job, "ko", cfg, client=EchoClient())
    forced = EchoClient()
    res = run_translation(job, "ko", cfg, client=forced, force=True)
    assert forced.calls > 0                       # 캐시 무시하고 재번역
    assert res.translated == res.total and res.cached == 0
    assert (job / "result.ko.md").read_text(encoding="utf-8") == RESULT_MD


def test_result_md_없으면_에러(tmp_path, cfg):
    from app.translate.types import TranslateError
    with pytest.raises(TranslateError, match="번역할 결과가 없습니다"):
        run_translation(tmp_path, "ko", cfg, client=EchoClient())
    assert _state(tmp_path)["status"] == "error"


def test_progress_콜백(job, cfg):
    seen = []
    run_translation(job, "ko", cfg, client=EchoClient(), progress=lambda c, t: seen.append((c, t)))
    assert seen and seen[-1][0] == seen[-1][1]     # 마지막 current == total
    assert all(t == seen[-1][1] for _, t in seen)


def test_layout_없어도_동작(tmp_path, cfg):
    (tmp_path / "result.md").write_text(RESULT_MD, encoding="utf-8")
    res = run_translation(tmp_path, "ko", cfg, client=EchoClient())
    assert res.status == "done"
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == RESULT_MD
    assert not (tmp_path / "layout.ko.json").exists()
