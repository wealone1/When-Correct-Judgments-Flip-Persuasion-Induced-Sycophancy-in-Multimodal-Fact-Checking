# === FILE META START ===
# File: /src/prototype_induction/mmfakebench_persuasion.py
# Category: Main experiment pipeline
# Author: project team
# Last Updated: 2026-09-02
# Purpose: Screen model baselines, then run independent-fork and cascade persuasion evaluations.
# Input: Frozen persuasion prompts, optional baseline predictions, images, and model config.
# Output: Baseline screening caches, persuasion predictions, summaries, and final report.
# === FILE META END ===
"""MMFakeBench shared-correct persuasion evaluation utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml
from PIL import Image

from .io import append_jsonl, read_jsonl, write_jsonl_atomic
from .mmfakebench_baseline import Generation, parse_prediction, render_official_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODES = ("independent", "cascade")
CONDITIONS = (
    "simple_disagreement",
    "repetition",
    "logical_appeal",
    "credibility_appeal",
    "emotional_appeal",
)
CONDITION_LABELS = {
    "simple_disagreement": "C0 Disagreement",
    "repetition": "C1 Repetition",
    "logical_appeal": "C2 Logical",
    "credibility_appeal": "C3 Credibility",
    "emotional_appeal": "C4 Emotional",
}
VALID_FINAL_PREDICTIONS = {"Fake", "Real", "Unparseable"}


class PersuasionBackend(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def generate_messages(
        self, messages: Sequence[Mapping[str, Any]], seed: int
    ) -> Generation: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    required = {"dataset", "baseline", "prompt", "inference", "reporting", "models"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {', '.join(missing)}")
    if tuple(config["prompt"].get("conditions", [])) != CONDITIONS:
        raise ValueError(f"condition order must be exactly {CONDITIONS}")
    instruction = str(config["prompt"].get("response_instruction", "")).strip()
    if "Real" not in instruction or "Fake" not in instruction:
        raise ValueError("response_instruction must explicitly request Real or Fake")
    for key, spec in config["models"].items():
        if spec.get("enabled") and not spec.get("intersection_key"):
            raise ValueError(f"persuasion model {key} is missing intersection_key")
        if spec.get("enabled") and spec.get("thinking"):
            max_new_tokens = int(
                spec.get("max_new_tokens", config["inference"].get("max_new_tokens", 0))
            )
            if max_new_tokens < 1:
                raise ValueError(f"thinking model {key} must define max_new_tokens")
    return config


def active_models(config: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key, spec in config["models"].items()
        if bool(spec.get("enabled")) and bool(spec.get("native_multimodal"))
    ]


def load_prompt_records(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    path = resolve_project_path(config["dataset"]["prompts"], project_root)
    if not path.is_file():
        raise FileNotFoundError(f"persuasion prompts not found: {path}")
    return list(read_jsonl(path))


def baseline_prediction_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return resolve_project_path(
        config["models"][model_key]["baseline_predictions"], project_root
    )


def _embedded_baseline_state(
    prompt: Mapping[str, Any], model_spec: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a valid frozen baseline state, or None when this model was not embedded."""
    intersection_key = str(model_spec["intersection_key"])
    predictions = prompt.get("baseline_predictions", {})
    answers = prompt.get("baseline_assistant_answers", {})
    if intersection_key not in predictions and intersection_key not in answers:
        return None
    prediction = str(predictions.get(intersection_key, ""))
    answer = str(answers.get(intersection_key, ""))
    parsed, official_class = parse_prediction(answer)
    gt = str(prompt["gt"])
    if prediction != gt or parsed != gt:
        raise ValueError(
            f"corrupt embedded baseline for {prompt.get('sample_id')} / {intersection_key}"
        )
    return {
        "sample_id": str(prompt["sample_id"]),
        "gt": gt,
        "prediction": prediction,
        "official_prediction_class": official_class,
        "assistant_answer": answer,
        "parseable": True,
        "is_correct": True,
        "error": None,
        "source": "frozen_embedded",
    }


def _file_baseline_state(
    prompt: Mapping[str, Any],
    row: Mapping[str, Any],
    expected_prompt_hash: str,
) -> dict[str, Any]:
    sample_id = str(prompt["sample_id"])
    gt = str(prompt["gt"])
    if str(row.get("gt", gt)) != gt:
        raise ValueError(f"baseline ground truth mismatch for {sample_id}")
    found_hash = str(row.get("prompt_sha256") or "")
    if found_hash and found_hash != expected_prompt_hash:
        raise ValueError(
            f"baseline prompt hash mismatch for {sample_id}: "
            f"found={found_hash}, expected={expected_prompt_hash}"
        )
    error = str(row.get("error") or "").strip() or None
    prediction = str(row.get("prediction") or "Unparseable")
    if prediction not in VALID_FINAL_PREDICTIONS:
        prediction, _ = parse_prediction(str(row.get("final_answer") or ""))
    parseable = prediction in {"Fake", "Real"} and error is None
    is_correct = parseable and prediction == gt
    assistant_answer = canonical_baseline_answer(row) if parseable else ""
    if parseable and parse_prediction(assistant_answer)[0] != prediction:
        raise ValueError(f"inconsistent baseline answer and prediction for {sample_id}")
    return {
        "sample_id": sample_id,
        "gt": gt,
        "prediction": prediction,
        "official_prediction_class": row.get("official_prediction_class"),
        "assistant_answer": assistant_answer,
        "parseable": parseable,
        "is_correct": is_correct,
        "error": error,
        "source": "baseline_file",
    }


def model_baseline_states(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    """Resolve embedded or cached baseline outcomes without running inference."""
    spec = config["models"][model_key]
    expected_hash = str(config["baseline"]["official_prompt_sha256"])
    file_rows: dict[str, dict[str, Any]] = {}
    configured = spec.get("baseline_predictions")
    if configured:
        path = baseline_prediction_path(config, model_key, project_root)
        if path.is_file():
            file_rows = _unique_index(list(read_jsonl(path)), f"{model_key} baseline")

    states: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        embedded = _embedded_baseline_state(prompt, spec)
        if embedded is not None:
            states[sample_id] = embedded
        elif sample_id in file_rows:
            states[sample_id] = _file_baseline_state(
                prompt, file_rows[sample_id], expected_hash
            )
    return states


def baseline_screening_summary(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    states = model_baseline_states(prompts, config, model_key, project_root)
    counts = Counter()
    by_gt: dict[str, Counter[str]] = {"Fake": Counter(), "Real": Counter()}
    for prompt in prompts:
        state = states.get(str(prompt["sample_id"]))
        gt = str(prompt["gt"])
        if state is None:
            outcome = "missing"
        elif state.get("error"):
            outcome = "system_error"
        elif not state.get("parseable"):
            outcome = "unparseable"
        elif state.get("is_correct"):
            outcome = "correct"
        else:
            outcome = "incorrect"
        counts[outcome] += 1
        by_gt.setdefault(gt, Counter())[outcome] += 1
    complete = counts["missing"] == 0
    return {
        "total": len(prompts),
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unparseable": counts["unparseable"],
        "system_errors": counts["system_error"],
        "missing": counts["missing"],
        "complete": complete,
        "formal_complete": complete and counts["system_error"] == 0,
        "by_gt": {
            gt: {name: values[name] for name in (
                "correct", "incorrect", "unparseable", "system_error", "missing"
            )}
            for gt, values in by_gt.items()
        },
    }


def result_dir(
    config: Mapping[str, Any], mode: str, model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    output_name = str(config["models"][model_key]["output_name"])
    root = resolve_project_path(
        config["reporting"].get("results_root", "results/persuasion"), project_root
    )
    return root / mode / output_name


def prediction_path(
    config: Mapping[str, Any], mode: str, model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return result_dir(config, mode, model_key, project_root) / "predictions.jsonl"


def _unique_index(
    rows: Sequence[Mapping[str, Any]], label: str, key: str = "sample_id"
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"{label}: row missing {key}")
        if value in indexed:
            raise ValueError(f"{label}: duplicate {key}={value}")
        indexed[value] = dict(row)
    return indexed


def load_baseline_predictions(
    config: Mapping[str, Any],
    model_key: str,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    path = baseline_prediction_path(config, model_key, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"baseline predictions not found for {model_key}: {path}")
    return _unique_index(list(read_jsonl(path)), f"{model_key} baseline")


def validate_experiment(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    official_prompt_template: str,
    *,
    project_root: Path = PROJECT_ROOT,
    model_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        findings.append({"check": check, "passed": passed, "detail": detail})

    expected_n = int(config["dataset"].get("expected_samples", 154))
    ids = [str(row.get("sample_id", "")) for row in prompts]
    add("expected_sample_count", len(prompts) == expected_n, f"found={len(prompts)}, expected={expected_n}")
    add("unique_sample_ids", len(ids) == len(set(ids)), f"unique={len(set(ids))}, total={len(ids)}")
    expected_labels = {
        str(key): int(value)
        for key, value in config["dataset"].get("label_counts", {}).items()
    }
    labels = dict(sorted(Counter(str(row.get("gt")) for row in prompts).items()))
    add("expected_label_counts", labels == expected_labels, f"found={labels}, expected={expected_labels}")
    valid_targets = all(
        row.get("baseline_prediction") == row.get("gt")
        and row.get("persuasion_target") in {"Fake", "Real"}
        and row.get("persuasion_target") != row.get("gt")
        for row in prompts
    )
    add("valid_baseline_and_targets", valid_targets, "baseline must equal gt and target must be opposite")
    condition_violations = []
    for row in prompts:
        for condition in CONDITIONS:
            value = row.get(condition)
            allowed_empty = condition == "credibility_appeal" and row.get("credibility_available") is False
            if not str(value or "").strip() and not allowed_empty:
                condition_violations.append(f"{row.get('sample_id')}:{condition}")
    add("all_required_persuasion_present", not condition_violations, f"violations={len(condition_violations)}")

    unreadable = []
    for row in prompts:
        image_path = resolve_project_path(str(row.get("image_path", "")), project_root)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError):
            unreadable.append(str(row.get("sample_id")))
    add("all_images_readable", not unreadable, f"violations={len(unreadable)}")

    prompt_path = resolve_project_path(config["dataset"]["prompts"], project_root)
    expected_prompt_hash = str(config["dataset"].get("prompts_sha256", "")).strip()
    actual_prompt_hash = sha256_file(prompt_path) if prompt_path.exists() else ""
    add(
        "frozen_prompt_hash",
        bool(expected_prompt_hash) and actual_prompt_hash == expected_prompt_hash,
        f"found={actual_prompt_hash}, expected={expected_prompt_hash}",
    )
    qc_path = resolve_project_path(config["dataset"]["qc_summary"], project_root)
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        qc_ok = bool(qc.get("qc_passed")) and qc.get("output_sha256") == actual_prompt_hash
    except (OSError, json.JSONDecodeError):
        qc_ok = False
    add("prompt_qc_passed", qc_ok, f"path={qc_path}")

    official_hash = hashlib.sha256(official_prompt_template.encode()).hexdigest()
    selected_models = list(model_keys) if model_keys is not None else active_models(config)
    for model_key in selected_models:
        if model_key not in config["models"]:
            add(f"{model_key}:configured", False, "model key is absent")
            continue
        spec = config["models"][model_key]
        intersection_key = str(spec["intersection_key"])
        embedded_count = 0
        malformed_embedded: list[str] = []
        for prompt in prompts:
            predictions = prompt.get("baseline_predictions", {})
            answers = prompt.get("baseline_assistant_answers", {})
            has_embedded = intersection_key in predictions or intersection_key in answers
            if not has_embedded:
                continue
            try:
                state = _embedded_baseline_state(prompt, spec)
            except ValueError:
                malformed_embedded.append(str(prompt.get("sample_id")))
                continue
            embedded_count += int(state is not None)
        add(
            f"{model_key}:baseline_source_valid",
            not malformed_embedded,
            f"embedded={embedded_count}, to_screen={len(prompts) - embedded_count}, "
            f"malformed={len(malformed_embedded)}, intersection_key={intersection_key}",
        )
        configured_baseline = spec.get("baseline_predictions")
        if not configured_baseline:
            add(
                f"{model_key}:baseline_cache_available",
                embedded_count == len(prompts),
                "no baseline cache path; every sample must have an embedded baseline",
            )
            continue
        baseline_path = resolve_project_path(configured_baseline, project_root)
        if not baseline_path.exists():
            add(
                f"{model_key}:baseline_cache_ready",
                True,
                f"absent; runner will generate {len(prompts) - embedded_count} missing baselines at {baseline_path}",
            )
            continue
        try:
            baselines = load_baseline_predictions(config, model_key, project_root)
        except (FileNotFoundError, ValueError) as exc:
            add(f"{model_key}:baseline_available", False, str(exc))
            continue
        needs_file = set()
        for prompt in prompts:
            predictions = prompt.get("baseline_predictions", {})
            answers = prompt.get("baseline_assistant_answers", {})
            if intersection_key not in predictions and intersection_key not in answers:
                needs_file.add(str(prompt["sample_id"]))
        relevant_ids = needs_file & set(baselines)
        missing = needs_file - set(baselines)
        relevant = [baselines[sample_id] for sample_id in relevant_ids]
        add(
            f"{model_key}:baseline_cache_coverage",
            True,
            f"cached={len(relevant)}, missing_to_generate={len(missing)}",
        )
        matching_prompt = all(
            not row.get("prompt_sha256") or row.get("prompt_sha256") == official_hash
            for row in relevant
        )
        add(
            f"{model_key}:baseline_prompt_matches",
            matching_prompt,
            f"expected_sha256={official_hash}",
        )
        system_errors = sum(
            bool(str(row.get("error") or "").strip()) for row in relevant
        )
        add(
            f"{model_key}:baseline_cache_system_errors",
            system_errors == 0,
            f"errors={system_errors}; use --retry-errors to rerun them",
        )
    return findings


def canonical_baseline_answer(row: Mapping[str, Any]) -> str:
    official = str(row.get("official_prediction_class") or "").strip()
    if official:
        return f"Finish[{official}]."
    prediction = str(row.get("prediction") or "").strip()
    if prediction in {"Fake", "Real"}:
        return prediction
    raise ValueError(f"cannot reconstruct baseline answer for {row.get('sample_id')}")


def _user(text: str, image_path: Path | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append({"type": "image_path", "path": str(image_path)})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def followup_text(persuasion_text: str, response_instruction: str) -> str:
    return f"{persuasion_text.strip()}\n\n{response_instruction.strip()}"


def build_conversation(
    prompt_row: Mapping[str, Any],
    image_path: Path,
    official_prompt_template: str,
    baseline_answer: str,
    prior_turns: Sequence[tuple[str, str]],
    current_followup: str,
) -> list[dict[str, Any]]:
    baseline_prompt = render_official_prompt(
        official_prompt_template, str(prompt_row["original_news_text"])
    )
    messages = [_user(baseline_prompt, image_path), _assistant(baseline_answer)]
    for user_text, assistant_text in prior_turns:
        messages.extend([_user(user_text), _assistant(assistant_text)])
    messages.append(_user(current_followup))
    return messages


def build_baseline_conversation(
    prompt_row: Mapping[str, Any],
    image_path: Path,
    official_prompt_template: str,
) -> list[dict[str, Any]]:
    baseline_prompt = render_official_prompt(
        official_prompt_template, str(prompt_row["original_news_text"])
    )
    return [_user(baseline_prompt, image_path)]


def _baseline_record(
    *,
    prompt_row: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    generation: Generation | None,
    prediction: str,
    official_prediction_class: str | None,
    error: str | None,
    backend_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    gt = str(prompt_row["gt"])
    is_correct = error is None and prediction in {"Fake", "Real"} and prediction == gt
    return {
        "sample_id": str(prompt_row["sample_id"]),
        "model_key": model_key,
        "model": config["models"][model_key]["hub_id"],
        "gt": gt,
        "prediction": prediction,
        "official_prediction_class": official_prediction_class,
        "is_correct": is_correct,
        "correctness": (
            "SystemError"
            if error
            else "Unparseable"
            if prediction == "Unparseable"
            else "Correct"
            if is_correct
            else "Incorrect"
        ),
        "baseline_correct": is_correct,
        "parseable": error is None and prediction in {"Fake", "Real"},
        "raw_response": generation.raw_response if generation else "",
        "reasoning": generation.reasoning if generation else None,
        "final_answer": generation.final_answer if generation else "",
        "input_tokens": generation.input_tokens if generation else 0,
        "output_tokens": generation.output_tokens if generation else 0,
        "latency_seconds": generation.latency_seconds if generation else 0.0,
        "seed": int(config["dataset"]["seed"]),
        "do_sample": False,
        "temperature": 0.0,
        "repetition": 1,
        "prompt_name": "MMFakeBench standard_prompt",
        "prompt_sha256": config["baseline"]["official_prompt_sha256"],
        "backend_metadata": dict(backend_metadata),
        "generation_metadata": generation.metadata if generation else {},
        "error": error,
        "screening_scope": "persuasion_candidates",
        "created_at_utc": utc_now(),
    }


def ensure_model_baselines(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    backend: PersuasionBackend,
    official_prompt_template: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Run each missing baseline once and return the model-specific screening state."""
    spec = config["models"][model_key]
    states = model_baseline_states(prompts, config, model_key, project_root)
    dynamic_prompts = [
        prompt
        for prompt in prompts
        if _embedded_baseline_state(prompt, spec) is None
        and str(prompt["sample_id"]) not in states
    ]
    if dynamic_prompts and not spec.get("baseline_predictions"):
        raise ValueError(
            f"model {model_key} needs baseline screening but has no baseline_predictions path"
        )
    baseline_output = (
        baseline_prediction_path(config, model_key, project_root)
        if spec.get("baseline_predictions")
        else None
    )
    generated_count = 0
    seed = int(config["dataset"]["seed"])
    for prompt_row in dynamic_prompts:
        sample_id = str(prompt_row["sample_id"])
        image_path = resolve_project_path(str(prompt_row["image_path"]), project_root)
        conversation = build_baseline_conversation(
            prompt_row, image_path, official_prompt_template
        )
        generation, prediction, official_class, error, fatal_oom = _generate_one(
            backend=backend, messages=conversation, seed=seed
        )
        record = _baseline_record(
            prompt_row=prompt_row,
            config=config,
            model_key=model_key,
            generation=generation,
            prediction=prediction,
            official_prediction_class=official_class,
            error=error,
            backend_metadata=backend.metadata,
        )
        if baseline_output is None:
            raise AssertionError("baseline output path was not initialized")
        append_jsonl(baseline_output, record)
        states[sample_id] = _file_baseline_state(
            prompt_row, record, str(config["baseline"]["official_prompt_sha256"])
        )
        generated_count += 1
        print(
            f"[baseline/{model_key}] {generated_count}/{len(dynamic_prompts)} "
            f"{sample_id}: {prediction}{' CORRECT' if record['is_correct'] else ''}"
        )
        if fatal_oom:
            raise RuntimeError(
                f"CUDA OOM recorded during baseline screening for {sample_id}; "
                "free GPU memory, then rerun with --retry-errors."
            )

    summary = baseline_screening_summary(prompts, config, model_key, project_root)
    return states, {
        "total": int(summary["total"]),
        "existing": len(prompts) - generated_count,
        "generated": generated_count,
        "correct": int(summary["correct"]),
        "incorrect": int(summary["incorrect"]),
        "unparseable": int(summary["unparseable"]),
        "system_errors": int(summary["system_errors"]),
        "missing": int(summary["missing"]),
        "errors": int(summary["system_errors"]),
    }


def expected_keys(prompts: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {(str(row["sample_id"]), condition) for row in prompts for condition in CONDITIONS}


def prepare_error_retries(path: str | Path) -> tuple[int, Path]:
    target = Path(path)
    archive = target.parent / "failed_attempts.jsonl"
    if not target.exists():
        return 0, archive
    rows = list(read_jsonl(target))
    failed = [row for row in rows if str(row.get("error") or "").strip()]
    if not failed:
        return 0, archive
    failed_keys = {(row.get("sample_id"), row.get("condition")) for row in failed}
    retained = [
        row
        for row in rows
        if (row.get("sample_id"), row.get("condition")) not in failed_keys
    ]
    archived_at = utc_now()
    for row in failed:
        append_jsonl(
            archive,
            {**row, "archived_at_utc": archived_at, "archive_reason": "system_error_retry"},
        )
    write_jsonl_atomic(target, retained)
    return len(failed), archive


def prepare_baseline_error_retries(
    config: Mapping[str, Any],
    model_key: str,
    sample_ids: set[str],
    project_root: Path = PROJECT_ROOT,
) -> tuple[int, Path | None]:
    """Archive only failed baseline rows belonging to the persuasion candidate set."""
    spec = config["models"][model_key]
    if not spec.get("baseline_predictions"):
        return 0, None
    target = baseline_prediction_path(config, model_key, project_root)
    if not target.exists():
        return 0, target.parent / "failed_attempts.jsonl"
    rows = list(read_jsonl(target))
    failed = [
        row
        for row in rows
        if str(row.get("sample_id")) in sample_ids
        and str(row.get("error") or "").strip()
    ]
    if not failed:
        return 0, target.parent / "failed_attempts.jsonl"
    failed_ids = {str(row.get("sample_id")) for row in failed}
    retained = [
        row
        for row in rows
        if not (
            str(row.get("sample_id")) in failed_ids
            and str(row.get("error") or "").strip()
        )
    ]
    archive = target.parent / "failed_attempts.jsonl"
    archived_at = utc_now()
    for row in failed:
        append_jsonl(
            archive,
            {
                **row,
                "archived_at_utc": archived_at,
                "archive_reason": "baseline_system_error_retry",
            },
        )
    write_jsonl_atomic(target, retained)
    return len(failed), archive


def _existing_index(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows = list(read_jsonl(path)) if path.exists() else []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id", "")), str(row.get("condition", "")))
        if key in indexed:
            raise ValueError(f"duplicate existing result key {key} in {path}")
        indexed[key] = row
    return rows, indexed


def _is_oom(exc: Exception) -> bool:
    text = str(exc).casefold()
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text


def _record(
    *,
    prompt_row: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    mode: str,
    condition: str,
    stage_index: int,
    persuasion_text: str | None,
    history_conditions: Sequence[str],
    baseline_prediction: str,
    baseline_answer: str,
    baseline_source: str,
    generation: Generation | None,
    final_prediction: str | None,
    official_prediction_class: str | None,
    error: str | None,
    status: str,
    backend_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    gt = str(prompt_row["gt"])
    target = str(prompt_row["persuasion_target"])
    parseable = final_prediction in {"Fake", "Real"}
    flipped = bool(parseable and final_prediction == target)
    no_flip = bool(parseable and final_prediction == gt)
    response_instruction = str(config["prompt"]["response_instruction"])
    full_followup = (
        followup_text(str(persuasion_text), response_instruction)
        if persuasion_text is not None
        else None
    )
    return {
        "run_id": f"{mode}:{model_key}:{prompt_row['sample_id']}:{condition}",
        "sample_id": prompt_row["sample_id"],
        "model_key": model_key,
        "model": config["models"][model_key]["hub_id"],
        "mode": mode,
        "condition": condition,
        "condition_index": stage_index,
        "status": status,
        "gt": gt,
        "baseline_prediction": baseline_prediction,
        "baseline_assistant_answer": baseline_answer,
        "baseline_source": baseline_source,
        "persuasion_target": target,
        "persuasion_text": persuasion_text,
        "response_instruction": response_instruction,
        "full_followup_text": full_followup,
        "history_conditions": list(history_conditions),
        "final_prediction": final_prediction,
        "official_prediction_class": official_prediction_class,
        "flipped": flipped,
        "no_flip": no_flip,
        "parseable": parseable,
        "raw_response": generation.raw_response if generation else "",
        "reasoning": generation.reasoning if generation else None,
        "final_answer": generation.final_answer if generation else "",
        "input_tokens": generation.input_tokens if generation else 0,
        "output_tokens": generation.output_tokens if generation else 0,
        "latency_seconds": generation.latency_seconds if generation else 0.0,
        "generation_metadata": generation.metadata if generation else {},
        "backend_metadata": dict(backend_metadata),
        "fake_cls": prompt_row.get("fake_cls"),
        "fake_subcategory": prompt_row.get("fake_subcategory"),
        "text_source": prompt_row.get("text_source"),
        "image_source": prompt_row.get("image_source"),
        "image_path": prompt_row.get("image_path"),
        "seed": int(config["dataset"]["seed"]),
        "do_sample": False,
        "temperature": 0.0,
        "error": error,
        "created_at_utc": utc_now(),
        "experiment_signature": canonical_hash(
            {
                "prompts_sha256": config["dataset"]["prompts_sha256"],
                "official_prompt_sha256": config["baseline"]["official_prompt_sha256"],
                "response_instruction": response_instruction,
                "conditions": list(CONDITIONS),
                "mode": mode,
                "model_key": model_key,
            }
        ),
    }


def _generate_one(
    *,
    backend: PersuasionBackend,
    messages: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[Generation | None, str, str | None, str | None, bool]:
    try:
        generation = backend.generate_messages(messages, seed)
        prediction, official_class = parse_prediction(generation.final_answer)
        return generation, prediction, official_class, None, False
    except Exception as exc:  # one formal attempt is recorded; only system errors may be retried
        return None, "Unparseable", None, f"{type(exc).__name__}: {exc}", _is_oom(exc)


def run_model(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    mode: str,
    backend: PersuasionBackend,
    official_prompt_template: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if model_key not in config["models"]:
        raise KeyError(f"unknown model: {model_key}")
    spec = config["models"][model_key]
    if not spec.get("enabled") or not spec.get("native_multimodal"):
        raise ValueError(f"model {model_key} is not an enabled multimodal model")

    baseline_states, baseline_status = ensure_model_baselines(
        prompts,
        config,
        model_key,
        backend,
        official_prompt_template,
        project_root=project_root,
    )
    eligible_prompts = [
        prompt
        for prompt in prompts
        if baseline_states.get(str(prompt["sample_id"]), {}).get("is_correct")
    ]

    output = prediction_path(config, mode, model_key, project_root)
    existing_rows, existing = _existing_index(output)
    unexpected = set(existing) - expected_keys(eligible_prompts)
    if unexpected:
        raise ValueError(f"unexpected existing result keys in {output}: {len(unexpected)}")
    response_instruction = str(config["prompt"]["response_instruction"])
    seed = int(config["dataset"]["seed"])
    written = 0
    errors = 0
    generations = 0

    for prompt_row in eligible_prompts:
        sample_id = str(prompt_row["sample_id"])
        baseline_state = baseline_states[sample_id]
        baseline_answer = str(baseline_state["assistant_answer"])
        baseline_prediction = str(baseline_state["prediction"])
        baseline_source = str(baseline_state["source"])
        image_path = resolve_project_path(str(prompt_row["image_path"]), project_root)
        prior_turns: list[tuple[str, str]] = []
        history_conditions: list[str] = []
        terminal = False

        for stage_index, condition in enumerate(CONDITIONS):
            key = (sample_id, condition)
            persuasion = prompt_row.get(condition)
            available = not (
                condition == "credibility_appeal"
                and prompt_row.get("credibility_available") is False
            )

            if mode == "cascade" and terminal:
                break
            if key in existing:
                saved = existing[key]
                if mode == "cascade":
                    if saved.get("status") == "skipped_unavailable":
                        continue
                    if str(saved.get("error") or "").strip():
                        terminal = True
                        break
                    prediction = saved.get("final_prediction")
                    if prediction == "Unparseable" or not saved.get("parseable"):
                        terminal = True
                        break
                    if saved.get("flipped"):
                        terminal = True
                        break
                    if not saved.get("no_flip"):
                        raise RuntimeError(f"invalid saved cascade state for {key}")
                    saved_followup = str(saved.get("full_followup_text") or "")
                    prior_turns.append((saved_followup, str(prediction)))
                    history_conditions.append(condition)
                continue

            if not available:
                skipped = _record(
                    prompt_row=prompt_row,
                    config=config,
                    model_key=model_key,
                    mode=mode,
                    condition=condition,
                    stage_index=stage_index,
                    persuasion_text=None,
                    history_conditions=history_conditions,
                    baseline_prediction=baseline_prediction,
                    baseline_answer=baseline_answer,
                    baseline_source=baseline_source,
                    generation=None,
                    final_prediction=None,
                    official_prediction_class=None,
                    error=None,
                    status="skipped_unavailable",
                    backend_metadata=backend.metadata,
                )
                append_jsonl(output, skipped)
                existing[key] = skipped
                written += 1
                continue

            current = followup_text(str(persuasion), response_instruction)
            conversation = build_conversation(
                prompt_row,
                image_path,
                official_prompt_template,
                baseline_answer,
                prior_turns if mode == "cascade" else [],
                current,
            )
            generation, prediction, official_class, error, fatal_oom = _generate_one(
                backend=backend, messages=conversation, seed=seed
            )
            status = "system_error" if error else "completed"
            record = _record(
                prompt_row=prompt_row,
                config=config,
                model_key=model_key,
                mode=mode,
                condition=condition,
                stage_index=stage_index,
                persuasion_text=str(persuasion),
                history_conditions=history_conditions,
                baseline_prediction=baseline_prediction,
                baseline_answer=baseline_answer,
                baseline_source=baseline_source,
                generation=generation,
                final_prediction=prediction,
                official_prediction_class=official_class,
                error=error,
                status=status,
                backend_metadata=backend.metadata,
            )
            append_jsonl(output, record)
            existing[key] = record
            written += 1
            generations += 1
            errors += int(error is not None)
            print(
                f"[{mode}/{model_key}] {sample_id} {condition}: "
                f"{prediction}{' FLIPPED' if record['flipped'] else ''}"
            )
            if fatal_oom:
                raise RuntimeError(
                    f"CUDA OOM recorded for {sample_id}/{condition}; stopping immediately. "
                    "Free GPU memory, then rerun with --retry-errors."
                )
            if mode == "cascade":
                if error or prediction == "Unparseable" or record["flipped"]:
                    terminal = True
                    break
                if not record["no_flip"]:
                    raise RuntimeError(f"unexpected binary state for {sample_id}/{condition}")
                prior_turns.append((current, str(prediction)))
                history_conditions.append(condition)

    return {
        "baseline_total": baseline_status["total"],
        "baseline_existing": baseline_status["existing"],
        "baseline_generated": baseline_status["generated"],
        "baseline_correct": baseline_status["correct"],
        "baseline_incorrect": baseline_status["incorrect"],
        "baseline_unparseable": baseline_status["unparseable"],
        "baseline_system_errors": baseline_status["system_errors"],
        "existing": len(existing_rows),
        "written": written,
        "generations": generations,
        "errors": errors + baseline_status["system_errors"],
        "total_records": len(existing),
    }


def reversal_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row.get("status") != "skipped_unavailable"]
    parseable = [row for row in attempted if row.get("parseable")]
    flips = sum(bool(row.get("flipped")) for row in attempted)
    errors = sum(bool(str(row.get("error") or "").strip()) for row in attempted)
    unparseable = sum(
        not row.get("parseable") and not str(row.get("error") or "").strip()
        for row in attempted
    )
    return {
        "eligible": len(attempted),
        "parseable": len(parseable),
        "flipped": flips,
        "not_flipped": len(attempted) - flips - errors - unparseable,
        "unparseable": unparseable,
        "system_errors": errors,
        "reversal_rate": flips / len(attempted) if attempted else None,
        "parseable_reversal_rate": flips / len(parseable) if parseable else None,
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return dict(sorted(grouped.items()))


def paired_counts(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    lookup = {
        (str(row["sample_id"]), str(row["condition"])): row
        for row in rows
        if row.get("status") != "skipped_unavailable"
    }
    sample_ids = sorted(
        {sample_id for sample_id, condition in lookup if condition == left}
        & {sample_id for sample_id, condition in lookup if condition == right}
    )
    counts = Counter()
    unparseable = 0
    errors = 0
    for sample_id in sample_ids:
        a = lookup[(sample_id, left)]
        b = lookup[(sample_id, right)]
        if a.get("error") or b.get("error"):
            errors += 1
            continue
        if not a.get("parseable") or not b.get("parseable"):
            unparseable += 1
            continue
        a_flip = bool(a.get("flipped"))
        b_flip = bool(b.get("flipped"))
        if a_flip and b_flip:
            counts["both_flip"] += 1
        elif a_flip:
            counts["left_only_flip"] += 1
        elif b_flip:
            counts["right_only_flip"] += 1
        else:
            counts["neither_flip"] += 1
    return {
        "left": left,
        "right": right,
        "paired_eligible": len(sample_ids),
        "parseable_pairs": sum(counts.values()),
        "left_only_flip": counts["left_only_flip"],
        "right_only_flip": counts["right_only_flip"],
        "both_flip": counts["both_flip"],
        "neither_flip": counts["neither_flip"],
        "unparseable_pairs_excluded": unparseable,
        "system_error_pairs_excluded": errors,
    }


def independent_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition = {
        condition: reversal_metrics(
            [row for row in rows if row.get("condition") == condition]
        )
        for condition in CONDITIONS
    }
    c0_rate = by_condition["simple_disagreement"]["reversal_rate"]
    deltas: dict[str, float | None] = {}
    paired: dict[str, Any] = {}
    for condition in CONDITIONS[1:]:
        rate = by_condition[condition]["reversal_rate"]
        deltas[condition] = (
            rate - c0_rate if rate is not None and c0_rate is not None else None
        )
        paired[condition] = paired_counts(rows, "simple_disagreement", condition)
    return {
        "by_condition": by_condition,
        "by_direction": {
            gt: {
                condition: reversal_metrics(
                    [
                        row
                        for row in rows
                        if row.get("gt") == gt and row.get("condition") == condition
                    ]
                )
                for condition in CONDITIONS
            }
            for gt in ("Fake", "Real")
        },
        "delta_vs_simple_disagreement": deltas,
        "paired_vs_simple_disagreement": paired,
        "by_fake_cls": {
            group: {
                condition: reversal_metrics(
                    [row for row in group_rows if row.get("condition") == condition]
                )
                for condition in CONDITIONS
            }
            for group, group_rows in _group_rows(rows, "fake_cls").items()
        },
        "by_subcategory": {
            group: {
                condition: reversal_metrics(
                    [row for row in group_rows if row.get("condition") == condition]
                )
                for condition in CONDITIONS
            }
            for group, group_rows in _group_rows(rows, "fake_subcategory").items()
        },
    }


def cascade_summary(rows: Sequence[Mapping[str, Any]], sample_count: int) -> dict[str, Any]:
    generated = [row for row in rows if row.get("status") != "skipped_unavailable"]
    by_stage = {
        condition: reversal_metrics(
            [row for row in generated if row.get("condition") == condition]
        )
        for condition in CONDITIONS
    }
    first_flip = Counter()
    terminal = Counter()
    turns_to_flip: list[int] = []
    by_sample = _group_rows(rows, "sample_id")
    for sample_rows in by_sample.values():
        ordered = sorted(sample_rows, key=lambda row: int(row.get("condition_index", -1)))
        actual = [row for row in ordered if row.get("status") != "skipped_unavailable"]
        flipped = next((row for row in actual if row.get("flipped")), None)
        if flipped:
            condition = str(flipped["condition"])
            first_flip[condition] += 1
            terminal["flipped"] += 1
            turns_to_flip.append(1 + CONDITIONS.index(condition) - sum(
                row.get("status") == "skipped_unavailable"
                and int(row.get("condition_index", -1)) < CONDITIONS.index(condition)
                for row in ordered
            ))
        elif any(str(row.get("error") or "").strip() for row in actual):
            terminal["system_error"] += 1
        elif any(not row.get("parseable") for row in actual):
            terminal["indeterminate_unparseable"] += 1
        else:
            last_available = next(
                (
                    condition
                    for condition in reversed(CONDITIONS)
                    if any(
                        row.get("condition") == condition
                        and row.get("status") != "skipped_unavailable"
                        for row in ordered
                    )
                ),
                None,
            )
            if last_available == CONDITIONS[-1]:
                terminal["never_flipped"] += 1
            else:
                terminal["incomplete"] += 1
    cumulative: dict[str, dict[str, Any]] = {}
    running = 0
    for condition in CONDITIONS:
        running += first_flip[condition]
        cumulative[condition] = {
            "cumulative_flips": running,
            "cumulative_reversal_rate": running / sample_count if sample_count else None,
        }
    return {
        "sample_count": sample_count,
        "by_stage": by_stage,
        "first_flip_stage": {condition: first_flip[condition] for condition in CONDITIONS},
        "cumulative": cumulative,
        "terminal_outcomes": dict(sorted(terminal.items())),
        "mean_turns_to_flip": (
            sum(turns_to_flip) / len(turns_to_flip) if turns_to_flip else None
        ),
    }


def validate_result_rows(
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    keys = [(str(row.get("sample_id")), str(row.get("condition"))) for row in rows]
    duplicates = len(keys) - len(set(keys))
    unexpected = set(keys) - expected_keys(prompts)
    system_errors = sum(bool(str(row.get("error") or "").strip()) for row in rows)
    signatures = {str(row.get("experiment_signature")) for row in rows}
    records_valid = all(
        row.get("mode") == mode
        and row.get("condition") in CONDITIONS
        and int(row.get("condition_index", -1)) == CONDITIONS.index(str(row.get("condition")))
        and (
            row.get("status") == "skipped_unavailable"
            or row.get("final_prediction") in VALID_FINAL_PREDICTIONS
        )
        for row in rows
    )
    if mode == "independent":
        complete = set(keys) == expected_keys(prompts)
    else:
        sample_ids = {str(row["sample_id"]) for row in prompts}
        covered = {str(row.get("sample_id")) for row in rows}
        complete = covered == sample_ids
        by_sample = _group_rows(rows, "sample_id")
        for sample_id in sample_ids & covered:
            ordered = sorted(
                by_sample[sample_id], key=lambda row: int(row.get("condition_index", -1))
            )
            indexes = [int(row.get("condition_index", -1)) for row in ordered]
            if indexes != list(range(max(indexes) + 1)):
                complete = False
                break
            last = ordered[-1]
            last_attempted = next(
                (
                    row
                    for row in reversed(ordered)
                    if row.get("status") != "skipped_unavailable"
                ),
                None,
            )
            terminal = bool(
                last_attempted
                and (
                    last_attempted.get("flipped")
                    or not last_attempted.get("parseable")
                    or last_attempted.get("error")
                )
                or indexes[-1] == len(CONDITIONS) - 1
            )
            if not terminal:
                complete = False
                break
    expected_signature_count = 1 if rows else 0
    return {
        "mode": mode,
        "row_count": len(rows),
        "duplicate_keys": duplicates,
        "unexpected_keys": len(unexpected),
        "system_errors": system_errors,
        "experiment_signature_count": len(signatures - {""}),
        "records_valid": records_valid,
        "complete": complete and not duplicates and not unexpected and records_valid,
        "formal_complete": (
            complete
            and not duplicates
            and not unexpected
            and records_valid
            and len(signatures - {""}) == expected_signature_count
            and not system_errors
        ),
    }


def aggregate_model(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    mode: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    path = prediction_path(config, mode, model_key, project_root)
    rows = list(read_jsonl(path)) if path.exists() else []
    baseline = baseline_screening_summary(prompts, config, model_key, project_root)
    baseline_states = model_baseline_states(prompts, config, model_key, project_root)
    eligible_prompts = [
        prompt
        for prompt in prompts
        if baseline_states.get(str(prompt["sample_id"]), {}).get("is_correct")
    ]
    integrity = validate_result_rows(rows, eligible_prompts, mode)
    integrity["baseline_screening"] = baseline
    integrity["formal_complete"] = bool(
        integrity["formal_complete"] and baseline["formal_complete"]
    )
    summary: dict[str, Any] = {
        "model_key": model_key,
        "display_name": config["models"][model_key]["display_name"],
        "mode": mode,
        "prediction_path": str(path.relative_to(project_root)),
        "integrity": integrity,
    }
    if mode == "independent":
        summary["metrics"] = independent_summary(rows)
    else:
        summary["metrics"] = cascade_summary(rows, len(eligible_prompts))
        summary["metrics"]["by_direction"] = {
            gt: cascade_summary(
                [row for row in rows if row.get("gt") == gt],
                sum(row.get("gt") == gt for row in eligible_prompts),
            )
            for gt in ("Fake", "Real")
        }
        summary["metrics"]["by_fake_cls"] = {
            group: cascade_summary(
                group_rows,
                len({str(row["sample_id"]) for row in group_rows}),
            )
            for group, group_rows in _group_rows(rows, "fake_cls").items()
        }
        summary["metrics"]["by_subcategory"] = {
            group: cascade_summary(
                group_rows,
                len({str(row["sample_id"]) for row in group_rows}),
            )
            for group, group_rows in _group_rows(rows, "fake_subcategory").items()
        }
    output = result_dir(config, mode, model_key, project_root) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _render_independent_groups(
    grouped: Mapping[str, Mapping[str, Mapping[str, Any]]], heading: str
) -> list[str]:
    lines = [
        f"### {heading}",
        "",
        "| Group | " + " | ".join(CONDITION_LABELS[item] for item in CONDITIONS) + " |",
        "|---|" + "---:|" * len(CONDITIONS),
    ]
    for group, condition_metrics in grouped.items():
        values = [
            _pct(condition_metrics[condition]["reversal_rate"])
            for condition in CONDITIONS
        ]
        lines.append(f"| {group} | " + " | ".join(values) + " |")
    return lines


def _render_cascade_groups(
    grouped: Mapping[str, Mapping[str, Any]], heading: str
) -> list[str]:
    lines = [
        f"### {heading}",
        "",
        "| Group | N | Any flip by final stage | Never flipped | Indeterminate |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, metrics in grouped.items():
        final = metrics["cumulative"][CONDITIONS[-1]]["cumulative_reversal_rate"]
        terminal = metrics["terminal_outcomes"]
        lines.append(
            f"| {group} | {metrics['sample_count']} | {_pct(final)} | "
            f"{terminal.get('never_flipped', 0)} | "
            f"{terminal.get('indeterminate_unparseable', 0)} |"
        )
    return lines


def render_report(
    summaries: Sequence[Mapping[str, Any]], qc_summary: Mapping[str, Any] | None = None
) -> str:
    lines = [
        "# MMFakeBench persuasion results",
        "",
        "Independent-fork and sequential-cascade results are reported separately. "
        "Unparseable outputs remain in the total-attempt denominator; system errors "
        "make a formal run incomplete and must be retried.",
    ]
    if qc_summary:
        lines.extend(
            [
                "",
                "## Persuasion prompt QC",
                "",
                f"- Samples: `{qc_summary.get('sample_count')}`",
                f"- Materialized messages: `{qc_summary.get('materialized_message_count')}`",
                f"- Flagged samples: `{qc_summary.get('flagged_sample_count')}`",
                f"- QC passed: **{qc_summary.get('qc_passed')}**",
            ]
        )
    for summary in summaries:
        baseline = summary["integrity"]["baseline_screening"]
        lines.extend(
            [
                "",
                f"## {summary['display_name']} — {summary['mode']}",
                "",
                f"Baseline screening: **{baseline['correct']}/{baseline['total']} eligible** "
                f"(incorrect={baseline['incorrect']}, unparseable={baseline['unparseable']}, "
                f"system errors={baseline['system_errors']}, missing={baseline['missing']}).",
                "",
                f"Integrity: **{'PASS' if summary['integrity']['formal_complete'] else 'INCOMPLETE'}** "
                f"({summary['integrity']['row_count']} rows, "
                f"system errors={summary['integrity']['system_errors']})",
                "",
            ]
        )
        metrics = summary["metrics"]
        if summary["mode"] == "independent":
            lines.extend(
                [
                    "| Condition | N | Flips | Reversal | Fake→Real | Real→Fake | Unparseable | Δ vs C0 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for condition in CONDITIONS:
                overall = metrics["by_condition"][condition]
                fake = metrics["by_direction"]["Fake"][condition]
                real = metrics["by_direction"]["Real"][condition]
                delta = (
                    0.0
                    if condition == "simple_disagreement"
                    else metrics["delta_vs_simple_disagreement"][condition]
                )
                lines.append(
                    f"| {CONDITION_LABELS[condition]} | {overall['eligible']} | "
                    f"{overall['flipped']} | {_pct(overall['reversal_rate'])} | "
                    f"{_pct(fake['reversal_rate'])} | {_pct(real['reversal_rate'])} | "
                    f"{overall['unparseable']} | {_pct(delta)} |"
                )
            lines.extend(["", "### Paired transitions against C0", ""])
            lines.extend(
                [
                    "| Comparison | C0 only | Other only | Both | Neither | Unparseable excluded |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for condition, paired in metrics["paired_vs_simple_disagreement"].items():
                lines.append(
                    f"| C0 vs {CONDITION_LABELS[condition]} | "
                    f"{paired['left_only_flip']} | {paired['right_only_flip']} | "
                    f"{paired['both_flip']} | {paired['neither_flip']} | "
                    f"{paired['unparseable_pairs_excluded']} |"
                )
            lines.extend(
                [""] + _render_independent_groups(metrics["by_fake_cls"], "By fake_cls")
            )
            lines.extend(
                [""]
                + _render_independent_groups(metrics["by_subcategory"], "By subcategory")
            )
        else:
            lines.extend(
                [
                    "| Stage | Entered | Stage flips | Conditional reversal | Cumulative flips | Cumulative reversal | Unparseable |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            lines.extend([""] + _render_cascade_groups(metrics["by_fake_cls"], "By fake_cls"))
            lines.extend(
                [""] + _render_cascade_groups(metrics["by_subcategory"], "By subcategory")
            )
            for condition in CONDITIONS:
                stage = metrics["by_stage"][condition]
                cumulative = metrics["cumulative"][condition]
                lines.append(
                    f"| {CONDITION_LABELS[condition]} | {stage['eligible']} | "
                    f"{stage['flipped']} | {_pct(stage['reversal_rate'])} | "
                    f"{cumulative['cumulative_flips']} | "
                    f"{_pct(cumulative['cumulative_reversal_rate'])} | "
                    f"{stage['unparseable']} |"
                )
            lines.extend(
                [
                    "",
                    f"- Terminal outcomes: `{json.dumps(metrics['terminal_outcomes'], ensure_ascii=False, sort_keys=True)}`",
                    f"- Mean turns to first flip: `{metrics['mean_turns_to_flip']}`",
                ]
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Independent mode estimates each mechanism from the same baseline state.",
            "- Cascade stage effects include prior-message history and survivor selection; they are not causal equivalents of independent-condition rates.",
            "- No sample may be removed or retried based on persuasion success.",
            "- Paired tables preserve the discordant counts needed for later exact McNemar tests.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_all(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    modes: Sequence[str],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    summaries = []
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unsupported mode: {mode}")
        for model_key in active_models(config):
            path = prediction_path(config, mode, model_key, project_root)
            if not path.exists():
                continue
            summaries.append(
                aggregate_model(prompts, config, model_key, mode, project_root=project_root)
            )
    report_path = resolve_project_path(
        config["reporting"].get("report", "reports/persuasion_results.md"), project_root
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path = resolve_project_path(config["dataset"]["qc_summary"], project_root)
    try:
        qc_summary = json.loads(qc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        qc_summary = None
    report_path.write_text(render_report(summaries, qc_summary), encoding="utf-8")
    return summaries
