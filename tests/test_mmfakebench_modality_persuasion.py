"""Regression tests for modality-specific persuasion and its core FR metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from prototype_induction.io import read_jsonl
from prototype_induction.mmfakebench_baseline import Generation
from prototype_induction.mmfakebench_modality_persuasion import (
    COT_MODES,
    CONDITIONS,
    aggregate_model,
    dynamic_baseline_path,
    modality_specific_metrics,
    prediction_path,
    reparse_prediction_file,
    reasoning_token_limit,
    run_model,
    separate_cot_generation,
    split_visible_cot,
    thinking_analysis,
)


class FakeBackend:
    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake"}

    def generate_messages(
        self, messages: Sequence[Mapping[str, Any]], seed: int
    ) -> Generation:
        text = str(messages[-1]["content"][-1]["text"])
        answer = "Real" if "FLIP" in text else "Fake"
        return Generation(answer, None, answer, 1, 1, 0.01, {})


class ThinkingFakeBackend:
    def __init__(self) -> None:
        self.calls: list[Sequence[Mapping[str, Any]]] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake", "thinking": True, "max_new_tokens": 4096}

    def generate_messages(
        self, messages: Sequence[Mapping[str, Any]], seed: int
    ) -> Generation:
        self.calls.append(messages)
        if len(messages) == 1:
            reasoning = "baseline reasoning"
        else:
            assistant_history = str(messages[1]["content"][0]["text"])
            assert "<think>\nbaseline reasoning\n</think>" in assistant_history
            reasoning = "persuasion reasoning"
        return Generation(
            f"<think>{reasoning}</think>\nFinish[FAKE].",
            reasoning,
            "Finish[FAKE].",
            10,
            12,
            0.1,
            {},
        )


class CotFakeBackend:
    def __init__(self) -> None:
        self.calls: list[Sequence[Mapping[str, Any]]] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake", "thinking": False, "max_new_tokens": 1024}

    def generate_messages(
        self, messages: Sequence[Mapping[str, Any]], seed: int
    ) -> Generation:
        self.calls.append(messages)
        raw = "The follow-up adds no verifiable evidence.\nFinish[FAKE].<|im_end|>"
        return Generation(raw, None, raw, 10, 18, 0.1, {})


def test_visible_cot_split_requires_a_final_line() -> None:
    reasoning, final = split_visible_cot(
        "Text is plausible, but the image conflicts.\n**Final: Finish[FAKE].**<|im_end|>"
    )
    assert reasoning == "Text is plausible, but the image conflicts."
    assert final == "**Final: Finish[FAKE].**"

    reasoning, final = split_visible_cot(
        "I currently lean Fake, but the response ended before a final line"
    )
    assert reasoning is not None
    assert final == ""


def test_separate_cot_generation_preserves_raw_and_metadata() -> None:
    raw = "Short rationale.\nFinish[REAL].<turn|>"
    source = Generation(raw, None, raw, 10, 12, 0.2, {"fixture": True})
    result = separate_cot_generation(source, "generic_cot")
    assert result.raw_response == raw
    assert result.reasoning == "Short rationale."
    assert result.final_answer == "Finish[REAL]."
    assert result.metadata["visible_cot_split"] is True


def test_reparse_recovers_gemma_and_qwen_persuasion_outputs(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    fixtures = [
        {
            "sample_id": "gemma",
            "condition": "simple_disagreement",
            "status": "completed",
            "gt": "Real",
            "persuasion_target": "Fake",
            "final_prediction": "Unparseable",
            "parseable": False,
            "flipped": False,
            "no_flip": False,
            "raw_response": "Real<turn|>",
            "final_answer": "Real<turn|>",
            "error": None,
        },
        {
            "sample_id": "qwen",
            "condition": "repetition",
            "status": "completed",
            "gt": "Real",
            "persuasion_target": "Fake",
            "final_prediction": "Unparseable",
            "parseable": False,
            "flipped": False,
            "no_flip": False,
            "raw_response": "Fake<|im_end|>\n<|endoftext|>",
            "final_answer": "Fake<|im_end|>\n<|endoftext|>",
            "error": None,
        },
        {
            "sample_id": "cot",
            "condition": "logical_appeal",
            "reasoning_mode": "generic_cot",
            "status": "completed",
            "gt": "Fake",
            "persuasion_target": "Real",
            "final_prediction": "Unparseable",
            "parseable": False,
            "flipped": False,
            "no_flip": False,
            "raw_response": "The follow-up adds no evidence.\nFinish[FAKE].<turn|>",
            "reasoning": None,
            "final_answer": "",
            "error": None,
        },
        {
            "sample_id": "legacy-gemma-channel",
            "condition": "simple_disagreement",
            "status": "completed",
            "gt": "Real",
            "persuasion_target": "Fake",
            "final_prediction": "Unparseable",
            "parseable": False,
            "flipped": False,
            "no_flip": False,
            "raw_response": "thought\n<channel|>Fake<turn|>",
            "final_answer": "thought\n<channel|>Fake<turn|>",
            "error": None,
        },
        {
            "sample_id": "legacy-gemma-prefix",
            "condition": "simple_disagreement",
            "status": "completed",
            "gt": "Fake",
            "persuasion_target": "Real",
            "final_prediction": "Unparseable",
            "parseable": False,
            "flipped": False,
            "no_flip": False,
            "raw_response": ".<|channel>thought\n<channel|>Fake<turn|>",
            "final_answer": ".<|channel>thought\n<channel|>Fake<turn|>",
            "error": None,
        },
        {
            "sample_id": "truncated-rationale",
            "condition": "image_grounded_persuasion",
            "status": "completed",
            "gt": "Fake",
            "persuasion_target": "Real",
            "final_prediction": "Unparseable",
            "parseable": False,
            "flipped": False,
            "no_flip": False,
            "raw_response": "The item is Fake.\n<|channel>thought\n<channel|>Finish[FAKE",
            "final_answer": "The item is Fake.\n<|channel>thought\n<channel|>Finish[FAKE",
            "error": None,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in fixtures))
    assert reparse_prediction_file(path) == 5
    rows = list(read_jsonl(path))
    assert rows[0]["final_prediction"] == "Real"
    assert rows[0]["no_flip"] is True
    assert rows[0]["flipped"] is False
    assert rows[1]["final_prediction"] == "Fake"
    assert rows[1]["no_flip"] is False
    assert rows[1]["flipped"] is True
    assert rows[2]["final_prediction"] == "Fake"
    assert rows[2]["reasoning"] == "The follow-up adds no evidence."
    assert rows[2]["final_answer"] == "Finish[FAKE]."
    assert rows[3]["final_prediction"] == "Fake"
    assert rows[3]["flipped"] is True
    assert rows[4]["final_prediction"] == "Fake"
    assert rows[4]["no_flip"] is True
    assert rows[5]["final_prediction"] == "Unparseable"
    assert not rows[5]["parseable"]
    assert all(row["reparsed_from_raw"] for row in rows[:5])
    assert reparse_prediction_file(path) == 0


def test_core_cross_modal_metrics() -> None:
    def row(fake_cls: str, condition: str, flipped: bool) -> dict[str, Any]:
        return {
            "sample_id": f"{fake_cls}:{condition}",
            "fake_cls": fake_cls,
            "condition": condition,
            "status": "completed",
            "parseable": True,
            "flipped": flipped,
            "error": None,
        }

    rows = [
        row("textual_veracity_distortion", "text_grounded_persuasion", False),
        row("textual_veracity_distortion", "image_grounded_persuasion", True),
        row("visual_veracity_distortion", "text_grounded_persuasion", True),
        row("visual_veracity_distortion", "image_grounded_persuasion", False),
        row("original", "text_grounded_persuasion", False),
        row("original", "image_grounded_persuasion", False),
    ]
    metrics = modality_specific_metrics(rows)
    assert metrics["FR_TextError_ImagePersuasion"]["reversal_rate"] == 1.0
    assert metrics["FR_ImageError_TextPersuasion"]["reversal_rate"] == 1.0
    assert metrics["cross_minus_same"]["TextError_ImageMinusText"] == 1.0
    assert metrics["cross_minus_same"]["ImageError_TextMinusImage"] == 1.0


def test_independent_run_uses_only_binary_baseline_correct_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    image_path = raw / "MMFakeBench_test/fake/group/image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)
    prompt_path = tmp_path / "prompts.jsonl"
    prompt: dict[str, Any] = {
        "sample_id": "sample-1",
        "gt": "Fake",
        "persuasion_target": "Real",
        "original_news_text": "fixture caption",
        "source_annotation_image_path": "/fake/group/image.png",
        "split": "test",
        "fake_cls": "textual_veracity_distortion",
        "fake_subcategory": "group",
        "credibility_available": True,
    }
    for condition in CONDITIONS:
        prompt[condition] = f"{condition} {'FLIP' if condition == 'image_grounded_persuasion' else 'KEEP'}"
    prompt_path.write_text(json.dumps(prompt) + "\n")
    binary_template = "Caption: [News caption]"
    config: dict[str, Any] = {
        "dataset": {"prompts": str(prompt_path), "seed": 7, "source_root": str(raw)},
        "baseline": {"require_complete": True},
        "prompt": {"response_instruction": "Answer only Real or Fake."},
        "reporting": {"results_root": str(tmp_path / "persuasion")},
        "models": {"model-a": {"enabled": True, "native_multimodal": True, "hub_id": "fixture", "output_name": "model_a"}},
        "_formal_config": {
            "prompts": {"binary": {"template": binary_template}},
            "reporting": {"results_root": str(tmp_path / "formal")},
            "models": {"model-a": {"output_name": "model_a"}},
        },
        "_formal_config_path": "fixture.yaml",
        "_prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    }
    baseline = tmp_path / "formal/binary/model_a/predictions.jsonl"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({
        "sample_id": "sample-1", "task": "binary", "gt": "Fake",
        "prediction": "Fake", "prompt_sha256": hashlib.sha256(binary_template.encode()).hexdigest(),
        "error": None,
    }) + "\n")
    status = run_model(
        [prompt], config, "model-a", "independent", FakeBackend(),
        source_root=raw, project_root=tmp_path,
    )
    assert status["written"] == len(CONDITIONS)
    rows = list(read_jsonl(prediction_path(config, "independent", "model-a", tmp_path)))
    assert len(rows) == len(CONDITIONS)
    assert sum(bool(row["flipped"]) for row in rows) == 1
    assert next(row for row in rows if row["flipped"])["condition"] == "image_grounded_persuasion"


def test_cot_run_uses_separate_path_budget_and_reasoning_fields(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    image_path = raw / "MMFakeBench_test/fake/group/image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)
    prompt_path = tmp_path / "prompts.jsonl"
    prompt: dict[str, Any] = {
        "sample_id": "cot-1",
        "gt": "Fake",
        "persuasion_target": "Real",
        "original_news_text": "fixture caption",
        "source_annotation_image_path": "/fake/group/image.png",
        "split": "test",
        "fake_cls": "textual_veracity_distortion",
        "fake_subcategory": "group",
        "credibility_available": True,
    }
    for condition in CONDITIONS:
        prompt[condition] = f"{condition} claim"
    prompt_path.write_text(json.dumps(prompt) + "\n")
    binary_template = "Caption: [News caption]"
    cot_config = {
        mode: {
            "max_new_tokens": 1024,
            "response_instruction": f"Use {mode}; end with Finish[REAL]. or Finish[FAKE].",
        }
        for mode in COT_MODES
    }
    config: dict[str, Any] = {
        "dataset": {"prompts": str(prompt_path), "seed": 7, "source_root": str(raw)},
        "baseline": {"require_complete": True},
        "prompt": {"response_instruction": "Answer only Real or Fake."},
        "cot": {"modes": cot_config},
        "inference": {"max_new_tokens": 128},
        "reporting": {"results_root": str(tmp_path / "persuasion")},
        "models": {
            "model-a": {
                "enabled": True,
                "native_multimodal": True,
                "hub_id": "fixture",
                "output_name": "model_a",
                "display_name": "Model A",
                "thinking": False,
            }
        },
        "_formal_config": {
            "prompts": {"binary": {"template": binary_template}},
            "reporting": {"results_root": str(tmp_path / "formal")},
            "models": {"model-a": {"output_name": "model_a"}},
        },
        "_formal_config_path": "fixture.yaml",
        "_prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    }
    baseline = tmp_path / "formal/binary/model_a/predictions.jsonl"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({
        "sample_id": "cot-1", "task": "binary", "gt": "Fake",
        "prediction": "Fake", "prompt_sha256": hashlib.sha256(binary_template.encode()).hexdigest(),
        "error": None,
    }) + "\n")

    backend = CotFakeBackend()
    status = run_model(
        [prompt], config, "model-a", "independent", backend,
        reasoning_mode="generic_cot", source_root=raw, project_root=tmp_path,
    )
    assert status["max_new_tokens"] == 1024
    assert reasoning_token_limit(config, "structured_multimodal_cot", "model-a") == 1024
    path = prediction_path(
        config, "independent", "model-a", tmp_path,
        reasoning_mode="generic_cot",
    )
    assert "/cot/generic_cot/independent/" in path.as_posix()
    rows = list(read_jsonl(path))
    assert len(rows) == len(CONDITIONS)
    assert all(row["reasoning_mode"] == "generic_cot" for row in rows)
    assert all(row["reasoning"] == "The follow-up adds no verifiable evidence." for row in rows)
    assert all(row["final_answer"] == "Finish[FAKE]." for row in rows)
    assert all(row["final_prediction"] == "Fake" for row in rows)
    assert all(row["generation_metadata"]["visible_cot_split"] for row in rows)
    assert "Use generic_cot" in str(backend.calls[0][-1]["content"][-1]["text"])
    summary = aggregate_model(
        [prompt], config, "model-a", "independent",
        reasoning_mode="generic_cot", project_root=tmp_path,
    )
    assert summary["cot_analysis"]["overall"]["reasoning_present"] == len(CONDITIONS)
    reasoning_path = path.parent / "reasoning_analysis.jsonl"
    reasoning_rows = list(read_jsonl(reasoning_path))
    assert len(reasoning_rows) == len(CONDITIONS)
    assert all(row["reasoning_mode"] == "generic_cot" for row in reasoning_rows)
    assert all(row["final_prediction"] == "Fake" for row in reasoning_rows)


def test_thinking_model_screens_then_immediately_reuses_reasoning_context(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    image_path = raw / "MMFakeBench_test/fake/group/image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)
    prompt_path = tmp_path / "prompts.jsonl"
    prompt: dict[str, Any] = {
        "sample_id": "thinking-1",
        "gt": "Fake",
        "persuasion_target": "Real",
        "original_news_text": "fixture caption",
        "source_annotation_image_path": "/fake/group/image.png",
        "split": "test",
        "fake_cls": "textual_veracity_distortion",
        "fake_subcategory": "group",
        "credibility_available": True,
    }
    for condition in CONDITIONS:
        prompt[condition] = f"{condition} KEEP"
    prompt_path.write_text(json.dumps(prompt) + "\n")
    binary_template = "Caption: [News caption]"
    config: dict[str, Any] = {
        "dataset": {"prompts": str(prompt_path), "seed": 7, "source_root": str(raw)},
        "baseline": {
            "require_complete": True,
            "allow_dynamic_screening": True,
            "dynamic_results_root": str(tmp_path / "dynamic"),
        },
        "prompt": {"response_instruction": "Answer only Real or Fake."},
        "inference": {"max_new_tokens": 128},
        "reporting": {"results_root": str(tmp_path / "persuasion")},
        "models": {
            "thinking-model": {
                "enabled": True,
                "native_multimodal": True,
                "hub_id": "fixture",
                "output_name": "thinking_model",
                "thinking": True,
                "max_new_tokens": 4096,
            }
        },
        "_formal_config": {
            "prompts": {"binary": {"name": "fixture", "template": binary_template}},
            "reporting": {"results_root": str(tmp_path / "formal")},
            "models": {"thinking-model": {"output_name": "thinking_model"}},
        },
        "_formal_config_path": "fixture.yaml",
        "_prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    }
    backend = ThinkingFakeBackend()
    status = run_model(
        [prompt],
        config,
        "thinking-model",
        "independent",
        backend,
        source_root=raw,
        project_root=tmp_path,
    )
    assert status["baseline_generated"] == 1
    assert status["baseline_correct_eligible"] == 1
    assert len(backend.calls) == 1 + len(CONDITIONS)
    baseline_rows = list(
        read_jsonl(dynamic_baseline_path(config, "thinking-model", tmp_path))
    )
    assert baseline_rows[0]["reasoning"] == "baseline reasoning"
    persuasion_rows = list(
        read_jsonl(prediction_path(config, "independent", "thinking-model", tmp_path))
    )
    assert len(persuasion_rows) == len(CONDITIONS)
    assert all(row["baseline_source"] == "dynamic_screening" for row in persuasion_rows)
    assert all(row["baseline_context_includes_reasoning"] for row in persuasion_rows)


def test_thinking_analysis_separates_flipped_and_resisted_traces() -> None:
    baseline = [{
        "prediction": "Fake", "is_correct": True, "reasoning": "base trace",
        "output_tokens": 20, "latency_seconds": 1.0, "error": None,
    }]
    persuasion = [
        {
            "condition": "simple_disagreement", "parseable": True, "flipped": True,
            "no_flip": False, "reasoning": "flip trace", "output_tokens": 30,
            "latency_seconds": 2.0, "error": None, "status": "completed",
        },
        {
            "condition": "simple_disagreement", "parseable": True, "flipped": False,
            "no_flip": True, "reasoning": "resist trace", "output_tokens": 40,
            "latency_seconds": 3.0, "error": None, "status": "completed",
        },
    ]
    result = thinking_analysis(persuasion, baseline, 4096)
    assert result["baseline"]["by_outcome"]["correct"]["rows"] == 1
    assert result["persuasion"]["by_outcome"]["flipped"]["rows"] == 1
    assert result["persuasion"]["by_outcome"]["resisted"]["rows"] == 1
