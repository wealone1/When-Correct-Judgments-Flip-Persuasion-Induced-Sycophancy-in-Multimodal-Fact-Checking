"""Regression tests for formal MMFakeBench parsing and correctness upgrades."""

from __future__ import annotations

import json
from pathlib import Path

from prototype_induction.io import read_jsonl
from prototype_induction.mmfakebench_formal import (
    parse_task_prediction,
    reparse_prediction_file,
    upgrade_prediction_file,
)


def test_binary_parser_ignores_known_model_control_tokens() -> None:
    cases = {
        "Real<turn|>": ("Real", "REAL"),
        "Fake<|im_end|>\n<|endoftext|>": ("Fake", "FAKE"),
        "Finish[REAL].<end_of_turn>": ("Real", "REAL"),
        "Finish[FAKE].</s>": ("Fake", "FAKE"),
    }
    for raw, expected in cases.items():
        assert parse_task_prediction(raw, "binary") == expected


def test_binary_parser_remains_strict_for_explanatory_prose() -> None:
    assert parse_task_prediction("The answer is Real.<turn|>", "binary") == (
        "Unparseable",
        None,
    )


def test_reparse_recovers_stored_nonthinking_final_answer(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({
        "sample_id": "s1",
        "task": "binary",
        "gt": "Real",
        "prediction": "Unparseable",
        "raw_response": "Real<turn|>",
        "final_answer": "Real<turn|>",
        "parseable": False,
        "error": None,
    }) + "\n")
    assert reparse_prediction_file(path, "binary") == 1
    row = next(read_jsonl(path))
    assert row["prediction"] == "Real"
    assert row["is_correct"] is True
    assert row["reparsed_from_raw"] is True
    assert reparse_prediction_file(path, "binary") == 0


def test_binary_uppercase_output_class_does_not_break_correctness(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({
        "sample_id": "s1",
        "task": "binary",
        "gt": "Real",
        "prediction": "Real",
        "prediction_class": "REAL",
        "target_class": "Real",
        "is_correct": False,
        "task_correct": False,
        "binary_correct": True,
        "correctness": "Incorrect",
    }) + "\n")
    assert upgrade_prediction_file(path, "binary") == 1
    row = next(read_jsonl(path))
    assert row["is_correct"] is True
    assert row["task_correct"] is True
    assert row["binary_correct"] is True
    assert row["correctness"] == "Correct"
