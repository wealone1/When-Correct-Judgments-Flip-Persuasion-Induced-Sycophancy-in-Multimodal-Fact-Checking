"""Regression tests for teacher-forced MMFakeBench margin analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch
from PIL import Image

from prototype_induction.io import read_jsonl
from prototype_induction.mmfakebench_margin import (
    CandidateScore,
    PairScore,
    _splice_generation_prefix,
    _template_alignment,
    active_models,
    aggregate_model,
    baseline_path,
    protocol_path,
    run_baselines,
    run_protocol,
    validate_model,
)
from prototype_induction.mmfakebench_modality_persuasion import CONDITIONS


class FakeGemmaTokenizer:
    """Expose only the decoding operations used by template alignment."""

    pieces = {
        1: "context-a",
        2: "context-b",
        10: "Finish[REAL].",
        11: "Finish[FAKE].",
        90: "<|channel>thought\n",
        91: "<channel|>",
        99: "unsafe-template-change",
    }

    def decode(self, token_ids: Sequence[int], **_: Any) -> str:
        return "".join(self.pieces[int(item)] for item in token_ids)

    def convert_ids_to_tokens(self, token_ids: Sequence[int]) -> list[str]:
        return [self.pieces[int(item)] for item in token_ids]


class FakeMarginBackend:
    """Return fixed sequence scores based on the current follow-up marker."""

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake", "actual_generation": False}

    def score_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        candidates: Mapping[str, str],
    ) -> PairScore:
        current = str(messages[-1]["content"][-1]["text"])
        real_logprob, fake_logprob = (
            (-0.1, -3.0) if "CROSS" in current else (-3.0, -0.1)
        )

        def candidate(label: str, value: float) -> CandidateScore:
            return CandidateScore(
                label=label,
                text=str(candidates[label]),
                total_logprob=value,
                token_ids=[1, 2],
                tokens=["one", "two"],
                token_logprobs=[value / 2, value / 2],
                excluded_eos_ids=[],
            )

        return PairScore(
            candidates={
                "Real": candidate("Real", real_logprob),
                "Fake": candidate("Fake", fake_logprob),
            },
            input_tokens=20,
            latency_seconds=0.01,
            template_prefix_sha256="fixture",
        )


def test_gemma_empty_thought_channel_is_safely_inserted() -> None:
    tokenizer = FakeGemmaTokenizer()
    prefix_ids = [1, 2, 90, 91]
    full_ids = [1, 2, 10]
    alignment = _template_alignment(
        prefix_ids,
        full_ids,
        "Finish[REAL].",
        tokenizer,
    )
    assert alignment.strategy == "insert_empty_thought_channel"
    assert alignment.full_answer_start == 2
    assert alignment.inserted_token_ids == [90, 91]

    prefix = {
        "input_ids": torch.tensor([prefix_ids]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "pixel_values": torch.zeros((1, 3, 2, 2)),
    }
    full = {
        "input_ids": torch.tensor([full_ids]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.ones((1, 3, 2, 2)),
    }
    aligned = _splice_generation_prefix(
        prefix,
        full,
        alignment.full_answer_start,
        torch,
    )
    assert aligned["input_ids"].tolist() == [[1, 2, 90, 91, 10]]
    assert aligned["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert torch.equal(aligned["pixel_values"], torch.ones((1, 3, 2, 2)))


def test_splice_ignores_image_width_that_matches_token_length() -> None:
    """A 768-token Gemma context must not splice a 768-pixel image tensor."""
    prefix = {
        "input_ids": torch.arange(768).unsqueeze(0),
        "attention_mask": torch.ones((1, 768), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 2, 768)),
    }
    full = {
        "input_ids": torch.arange(768).unsqueeze(0),
        "attention_mask": torch.ones((1, 768), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 2, 768)),
    }

    aligned = _splice_generation_prefix(prefix, full, 764, torch)

    assert aligned["input_ids"].shape[-1] == 772
    assert aligned["attention_mask"].shape[-1] == 772
    assert aligned["pixel_values"].shape == (1, 3, 2, 768)
    assert torch.equal(aligned["pixel_values"], torch.ones((1, 3, 2, 768)))


def test_unrecognized_template_mismatch_remains_rejected() -> None:
    with pytest.raises(RuntimeError, match="does not preserve"):
        _template_alignment(
            [1, 2, 99],
            [1, 2, 10],
            "Finish[REAL].",
            FakeGemmaTokenizer(),
        )


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    raw = tmp_path / "raw"
    image = raw / "MMFakeBench_test/fake/group/image.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image)
    prompt: dict[str, Any] = {
        "sample_id": "sample-1",
        "gt": "Fake",
        "persuasion_target": "Real",
        "original_news_text": "fixture caption",
        "source_annotation_image_path": "/fake/group/image.png",
        "split": "test",
        "fake_cls": "textual_veracity_distortion",
        "fake_subcategory": "group",
        "analysis_cohort": "S1+",
        "selector_model": "model-a",
        "matched_pair_id": "pair-1",
        "selector_successful_conditions": [CONDITIONS[0]],
        "selector_n_successful_conditions": 1,
        "credibility_available": True,
        "baseline_predictions": {"model-a": "Fake"},
        "baseline_assistant_answers": {"model-a": "Finish[FAKE]."},
    }
    for index, condition in enumerate(CONDITIONS):
        prompt[condition] = f"{condition} {'CROSS' if index == 0 else 'KEEP'}"
    formal_template = (
        "Caption: [News caption]\n\n"
        "Answer with exactly one of the following and nothing else:\n"
        "Finish[REAL].\nFinish[FAKE]."
    )
    config: dict[str, Any] = {
        "dataset": {
            "source_root": str(raw),
            "seed": 7,
            "expected_samples": 1,
            "expected_category_counts": {"textual_veracity_distortion": 1},
        },
        "baseline": {"formal_config": "fixture"},
        "scoring": {
            "candidates": {"Real": "Finish[REAL].", "Fake": "Finish[FAKE]."},
            "final_probe": (
                "Answer with exactly one of the following and nothing else:\n"
                "Finish[REAL].\nFinish[FAKE]."
            ),
            "tie_tolerance": 1e-8,
        },
        "reporting": {
            "results_root": str(tmp_path / "margin"),
            "persuasion_config": str(tmp_path / "missing.yaml"),
        },
        "inference": {"log_every": 100},
        "models": {
            "model-a": {
                "display_name": "Model A",
                "hub_id": "fixture/model-a",
                "output_name": "model_a",
                "enabled": True,
                "native_multimodal": True,
                "thinking": False,
            }
        },
        "_formal_config": {
            "prompts": {"binary": {"template": formal_template}},
            "reporting": {"results_root": str(tmp_path / "formal")},
            "models": {"model-a": {"output_name": "model_a"}},
        },
        "_formal_config_path": "fixture.yaml",
        "_prompts_sha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    return config, [prompt], raw


def test_active_models_excludes_native_thinking() -> None:
    config = {
        "models": {
            "direct": {"enabled": True, "native_multimodal": True, "thinking": False},
            "thinking": {"enabled": True, "native_multimodal": True, "thinking": True},
        }
    }
    assert active_models(config) == ["direct"]
    try:
        validate_model(config, "thinking")
    except ValueError as exc:
        assert "Non-thinking only" in str(exc)
    else:
        raise AssertionError("thinking model should be rejected")


def test_independent_records_full_sequence_margin_and_displacement(tmp_path: Path) -> None:
    config, prompts, raw = _fixture(tmp_path)
    backend = FakeMarginBackend()
    baseline_status = run_baselines(
        prompts, config, "model-a", backend, source_root=raw, project_root=tmp_path
    )
    assert baseline_status["written"] == 1
    status = run_protocol(
        prompts,
        config,
        "model-a",
        "independent",
        backend,
        source_root=raw,
        project_root=tmp_path,
    )
    assert status["written"] == len(CONDITIONS)
    baseline = list(read_jsonl(baseline_path(config, "model-a", tmp_path)))[0]
    rows = list(read_jsonl(protocol_path(config, "model-a", "independent", tmp_path)))
    first = next(row for row in rows if row["condition"] == CONDITIONS[0])
    assert baseline["margin"] == 2.9
    assert first["margin"] == -2.9
    assert first["margin_shift"] == -5.8
    assert first["persuasion_displacement"] == 5.8
    assert first["boundary_crossed_from_baseline"] is True
    assert first["candidate_scores"]["Fake"]["token_ids"] == [1, 2]
    assert first["eos_scored"] is False
    assert first["analysis_cohort"] == "S1+"


def test_cascade_continues_after_crossing_and_can_recover(tmp_path: Path) -> None:
    config, prompts, raw = _fixture(tmp_path)
    backend = FakeMarginBackend()
    run_baselines(
        prompts, config, "model-a", backend, source_root=raw, project_root=tmp_path
    )
    status = run_protocol(
        prompts,
        config,
        "model-a",
        "cascade",
        backend,
        source_root=raw,
        project_root=tmp_path,
    )
    assert status["written"] == len(CONDITIONS)
    rows = list(read_jsonl(protocol_path(config, "model-a", "cascade", tmp_path)))
    assert len(rows) == len(CONDITIONS)
    assert rows[0]["selected_prediction"] == "Real"
    assert rows[1]["selected_prediction"] == "Fake"
    assert rows[1]["stage_recovered_correct_side"] is True
    assert rows[-1]["history_conditions"] == list(CONDITIONS[:-1])

    summary = aggregate_model(prompts, config, "model-a", project_root=tmp_path)
    assert summary["cascade"]["ever_crossed_correct_to_wrong"] == 1
    assert summary["cascade"]["recovered_after_first_crossing"] == 1
    assert summary["cascade"]["by_analysis_cohort_trajectory"]["S1+"][
        "ever_crossed_correct_to_wrong"
    ] == 1
    assert summary["integrity"]["formal_complete"] is False  # independent is not run here


def test_resume_does_not_duplicate_completed_rows(tmp_path: Path) -> None:
    config, prompts, raw = _fixture(tmp_path)
    backend = FakeMarginBackend()
    run_baselines(
        prompts, config, "model-a", backend, source_root=raw, project_root=tmp_path
    )
    run_protocol(
        prompts, config, "model-a", "independent", backend,
        source_root=raw, project_root=tmp_path,
    )
    status = run_protocol(
        prompts, config, "model-a", "independent", backend,
        source_root=raw, project_root=tmp_path,
    )
    assert status["written"] == 0
    assert len(list(read_jsonl(protocol_path(config, "model-a", "independent", tmp_path)))) == len(CONDITIONS)
