import hashlib
import threading
from pathlib import Path

import pytest

from app.config import Settings
from app.engine.base import RepetitiveOutputError
from app.engine.repetition import SEMANTIC_REPEAT_THRESHOLD, SemanticRepetitionDetector
from app.engine.unlimited import UnlimitedEngine


def _loop_line(index: int) -> str:
    return (
        f"<|det|>text [{100 + index}, 260, 820, 274]<|/det|>"
        f'# {index}. SYSTEM: "ENVIRONMENT LOOP"\n'
    )


def test_detects_number_and_coordinate_variant_loop_at_threshold():
    detector = SemanticRepetitionDetector(threshold=4)

    for index in range(1, 4):
        assert detector.feed(_loop_line(index)) is False
    assert detector.feed(_loop_line(4)) is True
    assert detector.repeat_count == 4
    assert "4회 반복" in detector.message


def test_fragmented_line_is_counted_only_after_newline():
    detector = SemanticRepetitionDetector(threshold=3)
    line = _loop_line(1)

    for character in line[:-1]:
        assert detector.feed(character) is False
    assert detector.repeat_count == 0
    assert detector.feed("\n") is False
    assert detector.repeat_count == 1

    payload = _loop_line(2) + _loop_line(3)
    for start in range(0, len(payload), 7):
        detected = detector.feed(payload[start : start + 7])
    assert detected is True


def test_page_marker_and_substantive_content_reset_the_run():
    detector = SemanticRepetitionDetector(threshold=4)

    detector.feed(_loop_line(1) + _loop_line(2) + _loop_line(3))
    detector.feed("<PA")
    detector.feed("GE>\n")
    detector.feed(_loop_line(4) + _loop_line(5) + _loop_line(6))
    assert detector.detected is False

    detector.feed("Preprint. Under review. 23\n")
    detector.feed("실제 본문은 매 페이지마다 달라집니다.\n")
    detector.feed("Preprint. Under review. 24\n")
    assert detector.detected is False


def test_short_numeric_rows_and_distinct_numbered_items_are_ignored():
    detector = SemanticRepetitionDetector(threshold=3)

    detector.feed("1 | 100 | 200\n2 | 110 | 210\n3 | 120 | 220\n")
    detector.feed(
        "1. Install the package and verify its checksum.\n"
        "2. Start the service after loading its configuration.\n"
        "3. Inspect the logs and confirm the result.\n"
    )
    assert detector.detected is False


def test_layout_metadata_does_not_turn_numeric_table_into_long_text():
    detector = SemanticRepetitionDetector()

    for index in range(1, SEMANTIC_REPEAT_THRESHOLD + 5):
        detector.feed(
            f"<|ref|>text<|/ref|><|det|>[[{index},260,820,274]]<|/det|>"
            f"{index} | {index * 10} | {index * 20}\n"
        )

    assert detector.detected is False


def test_repeated_multi_metric_rows_are_treated_as_legitimate_data():
    detector = SemanticRepetitionDetector()

    for index in range(1, 41):
        detector.feed(
            f"Experiment {index}: accuracy {800 + index}, latency {120 + index} ms\n"
        )

    assert detector.detected is False


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 511])
def test_actual_page27_single_line_fixture_is_detected_without_newline(chunk_size):
    """j_9ac80d37f2d4의 실제 27페이지 폭주 1행을 delta 크기별로 재생한다."""
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "j_9ac80d37f2d4_page27_numeric_loop.txt"
    )
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "7dc1a9a2582ccf24c0d443b736736f424ea3934349b6adf08c117775c493e9e4"
    )
    fixture = fixture.read_text(encoding="utf-8").rstrip("\n")
    assert "\n" not in fixture
    detector = SemanticRepetitionDetector(
        max_page_chars=100_000,
        max_page_tokens=None,
    )

    consumed = 0
    for start in range(0, len(fixture), chunk_size):
        delta = fixture[start : start + chunk_size]
        consumed += len(delta)
        if detector.feed(delta):
            break

    assert detector.detected is True
    assert detector.reason == "rolling_repeat"
    assert consumed < len(fixture)
    assert len(detector._rolling) <= 4_096
    assert len(detector._pending_line) <= 4_096


def test_rolling_buffer_stays_bounded_for_long_non_repetitive_line():
    detector = SemanticRepetitionDetector(
        max_page_chars=None,
        max_page_tokens=None,
    )
    payload = " ".join(f"unique_word_{index}" for index in range(2_000))

    detector.feed(payload)

    assert detector.detected is False
    assert len(detector._rolling) <= 4_096
    assert len(detector._pending_line) <= 4_096


def test_page_markers_reset_character_budget_even_when_fragmented():
    detector = SemanticRepetitionDetector(
        max_page_chars=5,
        max_page_tokens=None,
    )

    assert detector.feed("<PA") is False
    assert detector.feed("GE>12345") is False
    assert detector.page_index == 0 and detector.page_chars == 5
    assert detector.feed("<PAGE>abcde") is False
    assert detector.page_index == 1 and detector.page_chars == 5
    assert detector.feed("x") is True
    assert detector.reason == "page_char_limit"
    assert "문자 상한(5)" in detector.message


def test_leading_page_marker_does_not_advance_index_after_whitespace():
    detector = SemanticRepetitionDetector(
        max_page_chars=10,
        max_page_tokens=None,
    )

    detector.feed(" \n<PA")
    detector.feed("GE>first")

    assert detector.page_index == 0
    assert detector.page_chars == 5


def test_token_budget_is_hard_limit_without_waiting_for_decoded_text():
    detector = SemanticRepetitionDetector(
        max_page_chars=None,
        max_page_tokens=4,
    )

    assert detector.feed_tokens(4) is False
    assert detector.feed_tokens(1) is True
    assert detector.reason == "page_token_limit"
    assert "토큰 상한(4)" in detector.message


def test_plain_numeric_vector_is_not_treated_as_rolling_loop():
    detector = SemanticRepetitionDetector(
        max_page_chars=None,
        max_page_tokens=None,
    )

    detector.feed(" ".join(["0"] * 100))

    assert detector.detected is False


@pytest.mark.parametrize(
    "row",
    [
        "N/A",
        "Yes",
        "Preprint. Under review.",
        "- Not applicable",
    ],
)
def test_normal_repeated_rows_do_not_trigger_generic_rolling_guard(row):
    detector = SemanticRepetitionDetector(
        max_page_chars=None,
        max_page_tokens=None,
    )

    detector.feed((row + "\n") * 40)

    assert detector.detected is False


@pytest.mark.parametrize("value", ["N/A", "Yes", "0.0"])
def test_normal_repeated_values_on_one_long_row_are_not_cut_off(value):
    detector = SemanticRepetitionDetector(
        max_page_chars=None,
        max_page_tokens=None,
    )

    detector.feed((value + " ") * 200)

    assert detector.detected is False


class _Sink:
    def __init__(self) -> None:
        self.text = ""

    def on_text(self, text: str) -> None:
        self.text += text


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids, **kwargs):
        return "<eos>"


class _BudgetTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, **kwargs):
        pieces = {
            0: "prompt",
            10: "<PAGE>\n",
            11: "x ",
            12: "y ",
            99: "<eos>",
        }
        return "".join(pieces.get(int(token), "z ") for token in token_ids)


class _LoopingModel:
    def __init__(self, cancel: threading.Event | None = None) -> None:
        self.cancel = cancel

    def _emit_loop(self, **kwargs) -> None:
        streamer = kwargs["streamer"]
        criteria = kwargs["stopping_criteria"]
        for index in range(1, SEMANTIC_REPEAT_THRESHOLD + 1):
            streamer.on_finalized_text(_loop_line(index))
        assert bool(criteria[0](None, None)) is True
        if self.cancel is not None:
            self.cancel.set()

    def infer_multi(self, tokenizer, **kwargs):
        self._emit_loop(**kwargs)
        return "<PAGE>\npartial", 10

    def infer(self, tokenizer, **kwargs):
        self._emit_loop(**kwargs)
        return "partial"


class _OverBudgetModel:
    def infer_multi(self, tokenizer, **kwargs):
        kwargs["streamer"].on_finalized_text("<PAGE>\n123456")
        assert bool(kwargs["stopping_criteria"][0](None, None)) is True
        return "<PAGE>\npartial", 10


def _engine_with_model(monkeypatch, model) -> UnlimitedEngine:
    settings = Settings(
        engine="unlimited",
        device="cpu",
        preload_model=False,
        fast_decode=False,
    )
    engine = UnlimitedEngine(settings)
    engine._model = model
    engine._tokenizer = _Tokenizer()
    monkeypatch.setattr("app.engine.unlimited.make_ngram_logits_processor", lambda *args: [])
    return engine


def test_engine_turns_guard_stop_into_dedicated_error(tmp_path, monkeypatch):
    engine = _engine_with_model(monkeypatch, _LoopingModel())

    with pytest.raises(
        RepetitiveOutputError,
        match=rf"{SEMANTIC_REPEAT_THRESHOLD}회 반복",
    ):
        engine.run_multi(
            [Path("page.png")], tmp_path / "out", _Sink(), threading.Event()
        )


def test_engine_turns_page_character_limit_into_dedicated_error(tmp_path, monkeypatch):
    engine = _engine_with_model(monkeypatch, _OverBudgetModel())
    engine._settings.max_page_output_chars = 5

    with pytest.raises(RepetitiveOutputError, match=r"문자 상한\(5\)"):
        engine.run_multi(
            [Path("page.png")], tmp_path / "out", _Sink(), threading.Event()
        )


def test_single_engine_uses_the_same_generation_guard(tmp_path, monkeypatch):
    engine = _engine_with_model(monkeypatch, _LoopingModel())

    with pytest.raises(RepetitiveOutputError):
        engine.run_single(
            Path("page.png"), tmp_path / "out", _Sink(), threading.Event()
        )


def test_user_cancel_takes_precedence_over_repetition_error(tmp_path, monkeypatch):
    cancel = threading.Event()
    engine = _engine_with_model(monkeypatch, _LoopingModel(cancel))

    result = engine.run_multi([Path("page.png")], tmp_path / "out", _Sink(), cancel)

    assert result == "<PAGE>\npartial"
    assert cancel.is_set()


def test_streamer_counts_tokens_after_page_marker_reset(monkeypatch):
    import torch

    settings = Settings(
        engine="unlimited",
        device="cpu",
        preload_model=False,
        fast_decode=False,
    )
    engine = UnlimitedEngine(settings)
    engine._tokenizer = _BudgetTokenizer()
    monkeypatch.setattr("app.engine.unlimited.make_ngram_logits_processor", lambda *args: [])
    detector = SemanticRepetitionDetector(
        max_page_chars=None,
        max_page_tokens=4,
    )
    extras = engine._gen_extras(_Sink(), threading.Event(), 128, detector)
    streamer = extras["streamer"]

    streamer.put(torch.tensor([[0]]))  # prompt: 생성 예산에서 제외
    streamer.put(torch.tensor([[10, 11]]))  # leading <PAGE> + 첫 내용
    streamer.put(torch.tensor([[12, 12]]))
    assert detector.page_index == 0 and detector.page_tokens == 4
    assert detector.detected is False

    # 이전 페이지가 상한에 정확히 닿았어도 marker가 든 다음 block을 먼저
    # 디코딩/reset하고 새 페이지에 계수하므로 경계 오탐이 없어야 한다.
    streamer.put(torch.tensor([[10, 11]]))
    assert detector.page_index == 1 and detector.page_tokens == 2
    assert detector.detected is False


# ── 병합 후 리뷰 반영 회귀 테스트 (page_flood·열거형 오탐) ──────────────


def test_enumeration_form_does_not_false_positive_at_default_threshold():
    """숫자만 다른 동일 행 40개(설문지·법령 서식류) — 리뷰에서 실증된 오탐 회귀 방지.

    threshold 기본값을 24→64로 올린 근거: 합법 열거형 문서가 24행에서 걸려
    per_page 강등→텍스트 레이어(구조 소실)/플레이스홀더(스캔 문서 전손)로
    이어지는 품질 회귀가 실증됐다. 폭주 루프는 수백 회 반복하므로 64로도 잡힌다."""
    detector = SemanticRepetitionDetector()
    for index in range(1, 41):
        assert detector.feed(f"{index}. 매우 그렇다 / 그렇다 / 보통이다 / 아니다\n") is False
    assert detector.feed("", stream_end=True) is False


def test_marker_flood_loop_is_detected_as_page_flood():
    """<PAGE> 마커가 낀 루프는 마커마다 페이지 예산이 리셋되어 문자·토큰 예산을
    전부 우회한다(리뷰 실증: 2,000회 반복에도 미감지) — 마커 수 자체를 예산에
    포함해 기대 페이지 수를 크게 넘으면 중단한다."""
    detector = SemanticRepetitionDetector(expected_pages=2)
    unit = "<|ref|>text<|/ref|><|det|>[[1,2,3,4]]<|/det|># 3. LOOP BODY HERE\n<PAGE>"
    tripped = False
    for _ in range(50):
        if detector.feed(unit):
            tripped = True
            break
    assert tripped is True
    assert detector.reason == "page_flood"
    assert "기대 페이지 수(2)" in detector.message


def test_expected_marker_count_with_slack_does_not_trip():
    """정상 변이(마커 1~2개 초과)는 merge의 불일치 보정이 흡수하므로 트립 금지."""
    detector = SemanticRepetitionDetector(expected_pages=2)
    md = (
        "<PAGE>\n1페이지 본문입니다. 내용이 이어집니다.\n"
        "<PAGE>\n2페이지 본문입니다. 내용이 이어집니다.\n"
        "<PAGE>\n모델이 가끔 내는 여분 마커 하나까지는 정상 변이로 본다.\n"
    )
    assert detector.feed(md) is False
    assert detector.feed("", stream_end=True) is False
    assert detector.reason is None
