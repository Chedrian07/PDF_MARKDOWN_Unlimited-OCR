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


class _Sink:
    def __init__(self) -> None:
        self.text = ""

    def on_text(self, text: str) -> None:
        self.text += text


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids, **kwargs):
        return "<eos>"


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
