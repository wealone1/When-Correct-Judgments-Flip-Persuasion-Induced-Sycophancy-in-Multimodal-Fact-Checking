"""Regression tests for MMFakeBench modality contribution scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
from PIL import Image

from prototype_induction.io import read_jsonl
from prototype_induction.mmfakebench_margin import CandidateScore, PairScore
from prototype_induction.mmfakebench_modality_contribution import (
    EVIDENCE_CONDITIONS,
    FOLLOWUP_CONDITIONS,
    aggregate_model,
    build_messages,
    derived_path,
    run_model,
    scores_path,
)


class FakeContributionBackend:
    """Return known margins for every evidence/follow-up combination."""

    margins = {
        ("full", "neutral"): 4.0,
        ("text_only", "neutral"): 3.0,
        ("image_only", "neutral"): 2.0,
        ("evidence_free", "neutral"): 1.0,
        ("full", "image_grounded_persuasion"): -2.0,
        ("text_only", "image_grounded_persuasion"): 2.0,
        ("image_only", "image_grounded_persuasion"): -1.0,
        ("evidence_free", "image_grounded_persuasion"): 0.5,
        ("full", "text_grounded_persuasion"): 0.0,
        ("text_only", "text_grounded_persuasion"): 1.0,
        ("image_only", "text_grounded_persuasion"): 1.5,
        ("evidence_free", "text_grounded_persuasion"): 0.5,
    }

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake", "actual_generation": False}

    def score_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        candidates: Mapping[str, str],
    ) -> PairScore:
        initial = str(messages[0]["content"][-1]["text"])
        has_image = any(
            item["type"] == "image_path" for item in messages[0]["content"]
        )
        has_text = "fixture caption" in initial
        evidence = (
            "full"
            if has_image and has_text
            else "image_only"
            if has_image
            else "text_only"
            if has_text
            else "evidence_free"
        )
        followup_text = str(messages[-1]["content"][-1]["text"])
        followup = (
            "image_grounded_persuasion"
            if "IMAGE-PERSUASION" in followup_text
            else "text_grounded_persuasion"
            if "TEXT-PERSUASION" in followup_text
            else "neutral"
        )
        margin = self.margins[(evidence, followup)]
        if margin >= 0:
            fake_logprob = -0.1
            real_logprob = -0.1 - margin
        else:
            real_logprob = -0.1
            fake_logprob = -0.1 + margin

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


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    raw = tmp_path / "raw"
    image = raw / "MMFakeBench_test/fake/group/image.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image)
    prompt = {
        "sample_id": "sample-1",
        "gt": "Fake",
        "persuasion_target": "Real",
        "original_news_text": "fixture caption",
        "source_annotation_image_path": "/fake/group/image.png",
        "split": "test",
        "fake_cls": "textual_veracity_distortion",
        "fake_subcategory": "group",
        "analysis_cohort": "S1+",
        "matched_pair_id": "pair-1",
        "baseline_assistant_answers": {"model-a": "Finish[FAKE]."},
        "text_grounded_persuasion": "TEXT-PERSUASION",
        "image_grounded_persuasion": "IMAGE-PERSUASION",
    }
    final_probe = (
        "Answer with exactly one of the following and nothing else:\n"
        "Finish[REAL].\nFinish[FAKE]."
    )
    config: dict[str, Any] = {
        "dataset": {
            "source_root": str(raw),
            "seed": 7,
            "expected_samples": 1,
            "expected_category_counts": {"textual_veracity_distortion": 1},
            "expected_cohort_counts": {"S1+": 1},
        },
        "prompts": {
            "initial": {
                "full": "FULL [News caption]",
                "text_only": "TEXT [News caption]",
                "image_only": "IMAGE NO-CAPTION",
                "evidence_free": "NO EVIDENCE",
            },
            "neutral_recheck": f"NEUTRAL\n{final_probe}",
        },
        "scoring": {
            "candidates": {"Real": "Finish[REAL].", "Fake": "Finish[FAKE]."},
            "final_probe": final_probe,
            "tie_tolerance": 1e-8,
        },
        "reporting": {
            "results_root": str(tmp_path / "results"),
            "report": str(tmp_path / "report.md"),
            "bootstrap_repetitions": 50,
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
        "_prompts_sha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    return config, [prompt], raw


def test_messages_remove_only_requested_news_evidence(tmp_path: Path) -> None:
    config, prompts, raw = _fixture(tmp_path)
    image = raw / "MMFakeBench_test/fake/group/image.png"
    for evidence in EVIDENCE_CONDITIONS:
        messages = build_messages(prompts[0], config, image, evidence, "neutral")
        first_content = messages[0]["content"]
        has_image = any(item["type"] == "image_path" for item in first_content)
        initial_text = str(first_content[-1]["text"])
        assert has_image is (evidence in {"full", "image_only"})
        assert ("fixture caption" in initial_text) is (
            evidence in {"full", "text_only"}
        )
        assert messages[1]["content"][0]["text"] == "Finish[FAKE]."


def test_run_records_all_matched_contexts_and_resumes(tmp_path: Path) -> None:
    config, prompts, raw = _fixture(tmp_path)
    backend = FakeContributionBackend()
    first = run_model(
        prompts,
        config,
        "model-a",
        backend,
        source_root=raw,
        project_root=tmp_path,
    )
    assert first["written"] == len(EVIDENCE_CONDITIONS) * len(FOLLOWUP_CONDITIONS)
    rows = list(read_jsonl(scores_path(config, "model-a", tmp_path)))
    assert len(rows) == 12
    image_full = next(
        row
        for row in rows
        if row["evidence_condition"] == "full"
        and row["followup_condition"] == "image_grounded_persuasion"
    )
    assert image_full["margin"] == -2.0
    assert image_full["persuasion_displacement"] == 6.0
    assert image_full["boundary_crossed_from_neutral"] is True

    resumed = run_model(
        prompts,
        config,
        "model-a",
        backend,
        source_root=raw,
        project_root=tmp_path,
    )
    assert resumed["written"] == 0
    assert len(list(read_jsonl(scores_path(config, "model-a", tmp_path)))) == 12


def test_aggregate_computes_cross_modal_trust_transfer(tmp_path: Path) -> None:
    config, prompts, raw = _fixture(tmp_path)
    run_model(
        prompts,
        config,
        "model-a",
        FakeContributionBackend(),
        source_root=raw,
        project_root=tmp_path,
    )
    summary = aggregate_model(prompts, config, "model-a", project_root=tmp_path)
    rows = list(read_jsonl(derived_path(config, "model-a", tmp_path)))
    image = next(
        row
        for row in rows
        if row["persuasion_condition"] == "image_grounded_persuasion"
    )
    text = next(
        row
        for row in rows
        if row["persuasion_condition"] == "text_grounded_persuasion"
    )
    assert image["displacements"]["full"] == 6.0
    assert image["displacements"]["text_only"] == 1.0
    assert image["image_ctt"] == 5.0
    assert text["displacements"]["full"] == pytest.approx(4.0)
    assert text["displacements"]["image_only"] == pytest.approx(0.5)
    assert text["text_ctt"] == pytest.approx(3.5)
    primary = summary["primary_cross_modal_tests"][
        "textual_veracity_distortion/image_grounded_persuasion"
    ]
    assert primary["metric"] == "image_ctt"
    assert primary["summary"]["mean"] == 5.0
    assert summary["integrity"]["formal_complete"] is True

