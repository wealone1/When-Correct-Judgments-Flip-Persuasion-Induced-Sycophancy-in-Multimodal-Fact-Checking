# === FILE META START ===
# File: /src/prototype_induction/mmfakebench_modality_contribution.py
# Category: Main experiment pipeline
# Author: project team
# Last Updated: 2026-09-23
# Purpose: Measure how persuasion changes image/text contributions to Real/Fake decisions.
# Input: Frozen 255-row analysis set, paired grounded persuasion, images, one VLM.
# Output: Teacher-forced scores, leave-one-modality-out CTT metrics, and reports.
# === FILE META END ===
"""Teacher-forced leave-one-modality-out persuasion contribution analysis."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from .io import append_jsonl, read_jsonl, write_jsonl_atomic
from .mmfakebench_formal import load_formal_config, resolve_formal_image
from .mmfakebench_margin import (
    LABELS,
    MarginBackend,
    PairScore,
    _assistant,
    _canonical_hash,
    _is_oom,
    _now,
    _resolve,
    _user,
    active_models,
    build_scoring_backend,
    frozen_baseline_models,
    prepare_error_retries,
    validate_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_CONDITIONS = ("full", "text_only", "image_only", "evidence_free")
PERSUASION_CONDITIONS = (
    "text_grounded_persuasion",
    "image_grounded_persuasion",
)
FOLLOWUP_CONDITIONS = ("neutral", *PERSUASION_CONDITIONS)
EVIDENCE_WITH_IMAGE = frozenset({"full", "image_only"})
EVIDENCE_WITH_TEXT = frozenset({"full", "text_only"})
PRIMARY_TESTS = {
    ("textual_veracity_distortion", "image_grounded_persuasion"): "image_ctt",
    ("visual_veracity_distortion", "text_grounded_persuasion"): "text_ctt",
}


def load_config(
    path: str | Path, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Load and validate the frozen contribution experiment configuration."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    required = {"dataset", "baseline", "prompts", "scoring", "reporting"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {', '.join(missing)}")

    formal_path = _resolve(config["baseline"]["formal_config"], project_root)
    formal = load_formal_config(formal_path)
    config["_formal_config"] = formal
    config["_formal_config_path"] = str(formal_path)
    config["models"] = formal["models"]
    config["inference"] = formal["inference"]

    evidence = tuple(config["scoring"].get("evidence_conditions", []))
    followups = tuple(config["scoring"].get("followup_conditions", []))
    if evidence != EVIDENCE_CONDITIONS:
        raise ValueError(
            f"scoring.evidence_conditions must be exactly {EVIDENCE_CONDITIONS}"
        )
    if followups != FOLLOWUP_CONDITIONS:
        raise ValueError(
            f"scoring.followup_conditions must be exactly {FOLLOWUP_CONDITIONS}"
        )
    candidates = config["scoring"].get("candidates", {})
    if set(candidates) != set(LABELS):
        raise ValueError(f"scoring.candidates must contain exactly {LABELS}")
    final_probe = str(config["scoring"].get("final_probe") or "").strip()
    if not final_probe:
        raise ValueError("scoring.final_probe must not be empty")
    if any(str(candidates[label]) not in final_probe for label in LABELS):
        raise ValueError("final_probe must display both complete candidate strings")

    initial = config["prompts"].get("initial", {})
    if set(initial) != set(EVIDENCE_CONDITIONS):
        raise ValueError(
            f"prompts.initial must contain exactly {EVIDENCE_CONDITIONS}"
        )
    for condition, template in initial.items():
        count = str(template).count("[News caption]")
        expected = 1 if condition in EVIDENCE_WITH_TEXT else 0
        if count != expected:
            raise ValueError(
                f"initial template {condition!r} must contain [News caption] "
                f"exactly {expected} time(s)"
            )
    neutral = str(config["prompts"].get("neutral_recheck") or "").strip()
    if not neutral:
        raise ValueError("prompts.neutral_recheck must not be empty")

    prompt_path = _resolve(config["dataset"]["prompts"], project_root)
    config["_prompts_sha256"] = (
        hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if prompt_path.is_file()
        else ""
    )
    return config


def load_prompts(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    path = _resolve(config["dataset"]["prompts"], project_root)
    if not path.is_file():
        raise FileNotFoundError(f"modality contribution dataset not found: {path}")
    return list(read_jsonl(path))


def model_result_dir(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    root = _resolve(config["reporting"]["results_root"], project_root)
    return root / str(config["models"][model_key]["output_name"])


def scores_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return model_result_dir(config, model_key, project_root) / "scores.jsonl"


def derived_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return model_result_dir(config, model_key, project_root) / "derived_metrics.jsonl"


def summary_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return model_result_dir(config, model_key, project_root) / "summary.json"


def _render_initial(
    prompt: Mapping[str, Any], config: Mapping[str, Any], evidence: str
) -> str:
    template = str(config["prompts"]["initial"][evidence]).strip()
    if evidence in EVIDENCE_WITH_TEXT:
        return template.replace(
            "[News caption]", str(prompt["original_news_text"])
        )
    return template


def build_messages(
    prompt: Mapping[str, Any],
    config: Mapping[str, Any],
    image_path: Path,
    evidence: str,
    followup: str,
) -> list[dict[str, Any]]:
    """Build matched second-turn contexts with a fixed correct answer anchor.

    The neutral and persuasion arms always occupy the same answer position. The
    first assistant turn is the canonical correct sequence, without free-form
    reasoning, so ablation contexts differ only in supplied evidence.
    """
    if evidence not in EVIDENCE_CONDITIONS:
        raise ValueError(f"unsupported evidence condition: {evidence}")
    if followup not in FOLLOWUP_CONDITIONS:
        raise ValueError(f"unsupported follow-up condition: {followup}")
    gt = str(prompt["gt"])
    if gt not in LABELS:
        raise ValueError(f"invalid ground-truth label: {gt}")
    initial = _render_initial(prompt, config, evidence)
    messages = [
        _user(initial, image_path if evidence in EVIDENCE_WITH_IMAGE else None),
        _assistant(str(config["scoring"]["candidates"][gt])),
    ]
    if followup == "neutral":
        followup_text = str(config["prompts"]["neutral_recheck"]).strip()
    else:
        persuasion = str(prompt.get(followup) or "").strip()
        if not persuasion:
            raise ValueError(f"missing {followup} for {prompt.get('sample_id')}")
        followup_text = (
            f"{persuasion}\n\n{str(config['scoring']['final_probe']).strip()}"
        )
    messages.append(_user(followup_text))
    return messages


def _pair_payload(
    score: PairScore,
    gt: str,
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    correct = score.candidates[gt]
    wrong_label = "Fake" if gt == "Real" else "Real"
    wrong = score.candidates[wrong_label]
    margin = correct.total_logprob - wrong.total_logprob
    real_minus_fake = (
        score.candidates["Real"].total_logprob
        - score.candidates["Fake"].total_logprob
    )
    prediction = (
        "Real"
        if real_minus_fake > tie_tolerance
        else "Fake"
        if real_minus_fake < -tie_tolerance
        else "Tie"
    )
    return {
        "candidate_scores": {
            label: asdict(score.candidates[label]) for label in LABELS
        },
        "correct_label": gt,
        "wrong_label": wrong_label,
        "correct_logprob": correct.total_logprob,
        "wrong_logprob": wrong.total_logprob,
        "margin": margin,
        "real_minus_fake_logprob": real_minus_fake,
        "sequence_argmax_prediction": prediction,
        "near_tie": abs(margin) <= tie_tolerance,
        "margin_supports_correct": margin > tie_tolerance,
        "input_tokens": score.input_tokens,
        "scoring_latency_seconds": score.latency_seconds,
        "template_prefix_sha256": score.template_prefix_sha256,
        "template_alignment_strategy": score.template_alignment_strategy,
        "inserted_generation_control_token_ids": (
            score.inserted_generation_control_token_ids
        ),
        "inserted_generation_control_tokens": (
            score.inserted_generation_control_tokens
        ),
    }


def _write_tokenization_manifest(
    config: Mapping[str, Any],
    model_key: str,
    pair: PairScore,
    backend: MarginBackend,
    project_root: Path,
) -> None:
    path = model_result_dir(config, model_key, project_root) / "candidate_tokenization.json"
    if path.is_file():
        return
    payload = {
        "model_key": model_key,
        "model": config["models"][model_key]["hub_id"],
        "note": (
            "Candidate sequences are tokenized at the native chat-template "
            "answer position; EOS is excluded."
        ),
        "candidates": {
            label: {
                "text": pair.candidates[label].text,
                "token_ids": pair.candidates[label].token_ids,
                "tokens": pair.candidates[label].tokens,
                "excluded_eos_ids": pair.candidates[label].excluded_eos_ids,
            }
            for label in LABELS
        },
        "backend_metadata": backend.metadata,
        "created_at_utc": _now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_unique(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (
            str(row.get("sample_id") or ""),
            str(row.get("evidence_condition") or ""),
            str(row.get("followup_condition") or ""),
        )
        if key in indexed:
            raise ValueError(f"duplicate result key {key} in {path}")
        indexed[key] = row
    return indexed


def validate_experiment(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
    check_images: bool = True,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        findings.append({"check": check, "passed": passed, "detail": detail})

    expected_n = int(config["dataset"]["expected_samples"])
    ids = [str(row.get("sample_id") or "") for row in prompts]
    add(
        "expected_sample_count",
        len(prompts) == expected_n,
        f"found={len(prompts)}, expected={expected_n}",
    )
    add(
        "unique_sample_ids",
        len(ids) == len(set(ids)) and all(ids),
        f"unique={len(set(ids))}, total={len(ids)}",
    )
    counts = dict(sorted(Counter(str(row.get("fake_cls")) for row in prompts).items()))
    expected_counts = {
        str(key): int(value)
        for key, value in config["dataset"]["expected_category_counts"].items()
    }
    add(
        "expected_category_counts",
        counts == expected_counts,
        f"found={counts}, expected={expected_counts}",
    )
    cohort_counts = dict(
        sorted(Counter(str(row.get("analysis_cohort")) for row in prompts).items())
    )
    expected_cohorts = {
        str(key): int(value)
        for key, value in config["dataset"]["expected_cohort_counts"].items()
    }
    add(
        "expected_cohort_counts",
        cohort_counts == expected_cohorts,
        f"found={cohort_counts}, expected={expected_cohorts}",
    )
    invalid_labels = [
        str(row.get("sample_id"))
        for row in prompts
        if str(row.get("gt")) not in LABELS
        or str(row.get("persuasion_target")) not in LABELS
        or str(row.get("gt")) == str(row.get("persuasion_target"))
    ]
    add("valid_opposite_labels", not invalid_labels, f"violations={len(invalid_labels)}")
    missing_text = [
        f"{row.get('sample_id')}:{condition}"
        for row in prompts
        for condition in PERSUASION_CONDITIONS
        if not str(row.get(condition) or "").strip()
    ]
    add(
        "grounded_persuasion_text_present",
        not missing_text,
        f"violations={len(missing_text)}",
    )
    missing_baselines: list[str] = []
    frozen_models = frozen_baseline_models(prompts)
    for row in prompts:
        expected = str(config["scoring"]["candidates"].get(str(row.get("gt")), ""))
        answers = row.get("baseline_assistant_answers", {})
        for model_key in frozen_models:
            if str(answers.get(model_key) or "") != expected:
                missing_baselines.append(f"{row.get('sample_id')}:{model_key}")
    add(
        "frozen_correct_baselines_available",
        not missing_baselines,
        f"violations={len(missing_baselines)}",
    )
    if check_images:
        unreadable: list[str] = []
        for row in prompts:
            path = resolve_formal_image(
                row,
                source_root=source_root,
                configured_source_root=config["dataset"]["source_root"],
                project_root=project_root,
            )
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError):
                unreadable.append(str(row.get("sample_id")))
        add("all_images_readable", not unreadable, f"violations={len(unreadable)}")
    else:
        add("all_images_readable", True, "skipped by request")
    return findings


def run_model(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    backend: MarginBackend,
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    """Score all matched neutral/persuasion contexts for one model."""
    validate_model(config, model_key)
    output = scores_path(config, model_key, project_root)
    existing = _read_unique(output)
    expected_rows = len(prompts) * len(EVIDENCE_CONDITIONS) * len(FOLLOWUP_CONDITIONS)
    candidates = {
        key: str(value) for key, value in config["scoring"]["candidates"].items()
    }
    tolerance = float(config["scoring"].get("tie_tolerance", 1e-8))
    log_every = max(1, int(config["inference"].get("log_every", 10)))
    written = errors = scored = 0
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        image_path = resolve_formal_image(
            prompt,
            source_root=source_root,
            configured_source_root=config["dataset"]["source_root"],
            project_root=project_root,
        )
        for evidence in EVIDENCE_CONDITIONS:
            neutral_key = (sample_id, evidence, "neutral")
            neutral = existing.get(neutral_key)
            if neutral is None:
                neutral = _score_one(
                    prompt,
                    config,
                    model_key,
                    backend,
                    image_path,
                    evidence,
                    "neutral",
                    candidates,
                    tolerance,
                    project_root=project_root,
                )
                append_jsonl(output, neutral)
                existing[neutral_key] = neutral
                written += 1
                scored += 1
                if neutral.get("error"):
                    errors += 1
                _log_progress(model_key, scored, sample_id, evidence, "neutral", neutral, log_every)
                if neutral.get("fatal_oom"):
                    raise RuntimeError(
                        f"CUDA OOM at {sample_id}/{evidence}/neutral; "
                        "rerun with --retry-errors"
                    )
            if neutral.get("error") or neutral.get("margin") is None:
                continue
            for followup in PERSUASION_CONDITIONS:
                key = (sample_id, evidence, followup)
                if key in existing:
                    continue
                record = _score_one(
                    prompt,
                    config,
                    model_key,
                    backend,
                    image_path,
                    evidence,
                    followup,
                    candidates,
                    tolerance,
                    neutral_margin=float(neutral["margin"]),
                    project_root=project_root,
                )
                append_jsonl(output, record)
                existing[key] = record
                written += 1
                scored += 1
                if record.get("error"):
                    errors += 1
                _log_progress(
                    model_key,
                    scored,
                    sample_id,
                    evidence,
                    followup,
                    record,
                    log_every,
                )
                if record.get("fatal_oom"):
                    raise RuntimeError(
                        f"CUDA OOM at {sample_id}/{evidence}/{followup}; "
                        "rerun with --retry-errors"
                    )
    errors = sum(bool(row.get("error")) for row in existing.values())
    return {
        "existing": len(existing) - written,
        "written": written,
        "errors": errors,
        "total": len(existing),
        "expected": expected_rows,
    }


def _score_one(
    prompt: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    backend: MarginBackend,
    image_path: Path,
    evidence: str,
    followup: str,
    candidates: Mapping[str, str],
    tolerance: float,
    *,
    neutral_margin: float | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    messages = build_messages(prompt, config, image_path, evidence, followup)
    persuasion_text = (
        str(prompt[followup]) if followup in PERSUASION_CONDITIONS else None
    )
    record: dict[str, Any] = {
        "run_id": f"modality-contribution:{model_key}:{prompt['sample_id']}:{evidence}:{followup}",
        "sample_id": str(prompt["sample_id"]),
        "model_key": model_key,
        "model": config["models"][model_key]["hub_id"],
        "display_name": config["models"][model_key]["display_name"],
        "gt": str(prompt["gt"]),
        "wrong_label": "Fake" if prompt["gt"] == "Real" else "Real",
        "fake_cls": str(prompt["fake_cls"]),
        "fake_subcategory": prompt.get("fake_subcategory"),
        "analysis_cohort": prompt.get("analysis_cohort"),
        "matched_pair_id": prompt.get("matched_pair_id"),
        "evidence_condition": evidence,
        "image_evidence_present": evidence in EVIDENCE_WITH_IMAGE,
        "news_text_evidence_present": evidence in EVIDENCE_WITH_TEXT,
        "followup_condition": followup,
        "persuasion_text": persuasion_text,
        "fixed_baseline_answer": str(candidates[str(prompt["gt"])]),
        "neutral_margin": neutral_margin,
        "score_method": "teacher_forced_complete_sequence_sum",
        "actual_generation": False,
        "do_sample": False,
        "eos_scored": False,
        "context_sha256": _canonical_hash(messages),
        "dataset_sha256": config["_prompts_sha256"],
        "backend_metadata": backend.metadata,
        "created_at_utc": _now(),
    }
    try:
        pair = backend.score_messages(messages, candidates)
        record.update(
            _pair_payload(pair, str(prompt["gt"]), tie_tolerance=tolerance)
        )
        if followup in PERSUASION_CONDITIONS and neutral_margin is not None:
            margin = float(record["margin"])
            displacement = neutral_margin - margin
            record.update(
                {
                    "margin_shift": margin - neutral_margin,
                    "persuasion_displacement": displacement,
                    "moved_toward_wrong": displacement > tolerance,
                    "boundary_crossed_from_neutral": (
                        neutral_margin > tolerance and margin < -tolerance
                    ),
                }
            )
        else:
            record.update(
                {
                    "margin_shift": None,
                    "persuasion_displacement": None,
                    "moved_toward_wrong": None,
                    "boundary_crossed_from_neutral": None,
                }
            )
        record.update({"status": "completed", "error": None, "fatal_oom": False})
        _write_tokenization_manifest(config, model_key, pair, backend, project_root)
    except Exception as exc:
        record.update(
            {
                "status": "system_error",
                "error": f"{type(exc).__name__}: {exc}",
                "fatal_oom": _is_oom(exc),
            }
        )
    return record


def _log_progress(
    model_key: str,
    scored: int,
    sample_id: str,
    evidence: str,
    followup: str,
    record: Mapping[str, Any],
    log_every: int,
) -> None:
    if scored == 1 or scored % log_every == 0:
        detail = f"; error={record['error']}" if record.get("error") else ""
        print(
            f"[modality/{model_key}] scores={scored}: {sample_id} "
            f"{evidence}/{followup} -> C={record.get('margin')}{detail}"
        )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, repetitions: int
) -> list[float | None]:
    if not values:
        return [None, None]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    sample = [float(value) for value in values]
    means = [
        statistics.fmean(rng.choice(sample) for _ in sample)
        for _ in range(repetitions)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _metric_summary(
    values: Sequence[float], *, seed: int, bootstrap_repetitions: int
) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "q25": _percentile(clean, 0.25),
        "q75": _percentile(clean, 0.75),
        "positive_rate": (
            sum(value > 0 for value in clean) / len(clean) if clean else None
        ),
        "mean_bootstrap_ci95": _bootstrap_mean_ci(
            clean, seed=seed, repetitions=bootstrap_repetitions
        ),
    }


def _derive_one(
    prompt: Mapping[str, Any],
    persuasion: str,
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any] | None:
    required = [
        (evidence, followup)
        for evidence in EVIDENCE_CONDITIONS
        for followup in ("neutral", persuasion)
    ]
    if any(
        key not in rows
        or rows[key].get("error")
        or rows[key].get("margin") is None
        for key in required
    ):
        return None
    neutral = {
        evidence: float(rows[(evidence, "neutral")]["margin"])
        for evidence in EVIDENCE_CONDITIONS
    }
    post = {
        evidence: float(rows[(evidence, persuasion)]["margin"])
        for evidence in EVIDENCE_CONDITIONS
    }
    displacement = {
        evidence: neutral[evidence] - post[evidence]
        for evidence in EVIDENCE_CONDITIONS
    }
    image_ctt = displacement["full"] - displacement["text_only"]
    text_ctt = displacement["full"] - displacement["image_only"]

    def shapley_image(state: Mapping[str, float]) -> float:
        return 0.5 * (
            state["image_only"]
            - state["evidence_free"]
            + state["full"]
            - state["text_only"]
        )

    def shapley_text(state: Mapping[str, float]) -> float:
        return 0.5 * (
            state["text_only"]
            - state["evidence_free"]
            + state["full"]
            - state["image_only"]
        )

    def interaction(state: Mapping[str, float]) -> float:
        return (
            state["full"]
            - state["image_only"]
            - state["text_only"]
            + state["evidence_free"]
        )

    tolerance = 1e-8
    return {
        "sample_id": str(prompt["sample_id"]),
        "gt": str(prompt["gt"]),
        "fake_cls": str(prompt["fake_cls"]),
        "fake_subcategory": prompt.get("fake_subcategory"),
        "analysis_cohort": prompt.get("analysis_cohort"),
        "matched_pair_id": prompt.get("matched_pair_id"),
        "persuasion_condition": persuasion,
        "neutral_margins": neutral,
        "post_margins": post,
        "displacements": displacement,
        "full_displacement": displacement["full"],
        "image_ctt": image_ctt,
        "text_ctt": text_ctt,
        "evidence_free_displacement": displacement["evidence_free"],
        "full_beyond_evidence_free": (
            displacement["full"] - displacement["evidence_free"]
        ),
        "image_shapley_neutral": shapley_image(neutral),
        "image_shapley_post": shapley_image(post),
        "image_shapley_wrongward_shift": (
            shapley_image(neutral) - shapley_image(post)
        ),
        "text_shapley_neutral": shapley_text(neutral),
        "text_shapley_post": shapley_text(post),
        "text_shapley_wrongward_shift": (
            shapley_text(neutral) - shapley_text(post)
        ),
        "interaction_neutral": interaction(neutral),
        "interaction_post": interaction(post),
        "interaction_wrongward_shift": interaction(neutral) - interaction(post),
        "full_boundary_crossed": (
            neutral["full"] > tolerance and post["full"] < -tolerance
        ),
        "text_only_boundary_crossed": (
            neutral["text_only"] > tolerance
            and post["text_only"] < -tolerance
        ),
        "image_only_boundary_crossed": (
            neutral["image_only"] > tolerance
            and post["image_only"] < -tolerance
        ),
        "evidence_free_boundary_crossed": (
            neutral["evidence_free"] > tolerance
            and post["evidence_free"] < -tolerance
        ),
        "created_at_utc": _now(),
    }


def aggregate_model(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Derive per-sample CTT metrics and aggregate one completed model."""
    validate_model(config, model_key)
    indexed = _read_unique(scores_path(config, model_key, project_root))
    prompt_by_id = {str(row["sample_id"]): row for row in prompts}
    derived: list[dict[str, Any]] = []
    for sample_id, prompt in prompt_by_id.items():
        sample_rows = {
            (evidence, followup): row
            for (stored_id, evidence, followup), row in indexed.items()
            if stored_id == sample_id
        }
        for persuasion in PERSUASION_CONDITIONS:
            record = _derive_one(prompt, persuasion, sample_rows)
            if record is not None:
                record["model_key"] = model_key
                record["model"] = config["models"][model_key]["hub_id"]
                derived.append(record)
    write_jsonl_atomic(derived_path(config, model_key, project_root), derived)

    bootstrap_repetitions = int(
        config["reporting"].get("bootstrap_repetitions", 2000)
    )
    seed = int(config["dataset"].get("seed", 0))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in derived:
        grouped[(row["fake_cls"], row["persuasion_condition"])].append(row)
    by_group: dict[str, Any] = {}
    for (fake_cls, persuasion), rows in sorted(grouped.items()):
        key = f"{fake_cls}/{persuasion}"
        by_group[key] = {
            "n": len(rows),
            "full_displacement": _metric_summary(
                [row["full_displacement"] for row in rows],
                seed=seed + 1,
                bootstrap_repetitions=bootstrap_repetitions,
            ),
            "image_ctt": _metric_summary(
                [row["image_ctt"] for row in rows],
                seed=seed + 2,
                bootstrap_repetitions=bootstrap_repetitions,
            ),
            "text_ctt": _metric_summary(
                [row["text_ctt"] for row in rows],
                seed=seed + 3,
                bootstrap_repetitions=bootstrap_repetitions,
            ),
            "evidence_free_displacement": _metric_summary(
                [row["evidence_free_displacement"] for row in rows],
                seed=seed + 4,
                bootstrap_repetitions=bootstrap_repetitions,
            ),
            "image_shapley_wrongward_shift": _metric_summary(
                [row["image_shapley_wrongward_shift"] for row in rows],
                seed=seed + 5,
                bootstrap_repetitions=bootstrap_repetitions,
            ),
            "text_shapley_wrongward_shift": _metric_summary(
                [row["text_shapley_wrongward_shift"] for row in rows],
                seed=seed + 6,
                bootstrap_repetitions=bootstrap_repetitions,
            ),
            "full_boundary_crossing_rate": (
                sum(bool(row["full_boundary_crossed"]) for row in rows) / len(rows)
            ),
        }
    primary: dict[str, Any] = {}
    for group_key, metric in PRIMARY_TESTS.items():
        rows = grouped.get(group_key, [])
        output_key = f"{group_key[0]}/{group_key[1]}"
        primary[output_key] = {
            "metric": metric,
            "definition": (
                "D_full-D_text_only"
                if metric == "image_ctt"
                else "D_full-D_image_only"
            ),
            "summary": _metric_summary(
                [float(row[metric]) for row in rows],
                seed=seed + 11,
                bootstrap_repetitions=bootstrap_repetitions,
            ),
            "full_boundary_crossing_rate": (
                sum(bool(row["full_boundary_crossed"]) for row in rows) / len(rows)
                if rows
                else None
            ),
        }

    expected_rows = len(prompts) * len(EVIDENCE_CONDITIONS) * len(FOLLOWUP_CONDITIONS)
    system_errors = sum(bool(row.get("error")) for row in indexed.values())
    summary = {
        "experiment": config.get("experiment"),
        "model_key": model_key,
        "display_name": config["models"][model_key]["display_name"],
        "definitions": {
            "correctness_margin": "C=logP(correct)-logP(wrong)",
            "displacement": "D=C_neutral-C_persuasion; positive moves toward wrong",
            "image_ctt": "D_full-D_text_only",
            "text_ctt": "D_full-D_image_only",
            "image_only_warning": (
                "Image-only is a decision diagnostic, not fake-news detection accuracy."
            ),
            "cross_model_warning": (
                "Raw log-probability margins are compared within model only."
            ),
        },
        "integrity": {
            "score_rows": len(indexed),
            "expected_score_rows": expected_rows,
            "system_errors": system_errors,
            "derived_rows": len(derived),
            "expected_derived_rows": len(prompts) * len(PERSUASION_CONDITIONS),
            "formal_complete": (
                len(indexed) == expected_rows
                and system_errors == 0
                and len(derived) == len(prompts) * len(PERSUASION_CONDITIONS)
            ),
        },
        "primary_cross_modal_tests": primary,
        "by_category_and_persuasion": by_group,
        "created_at_utc": _now(),
    }
    path = summary_path(config, model_key, project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def aggregate_all(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    model_keys: Sequence[str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    keys = list(model_keys) if model_keys is not None else active_models(config)
    summaries = [
        aggregate_model(prompts, config, key, project_root=project_root)
        for key in keys
        if scores_path(config, key, project_root).is_file()
    ]
    lines = [
        "# MMFakeBench modality contribution analysis",
        "",
        "Correctness margin is `C=logP(correct)-logP(wrong)`. Persuasion "
        "displacement is `D=C_neutral-C_persuasion`, so positive values move "
        "toward the wrong answer.",
        "",
        "## Primary cross-modal trust-transfer tests",
        "",
        "| Model | Test | n | Mean CTT | 95% bootstrap CI | Median | CTT>0 | Full crossing |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "textual_veracity_distortion/image_grounded_persuasion": (
            "Text-error + Image persuasion"
        ),
        "visual_veracity_distortion/text_grounded_persuasion": (
            "Image-error + Text persuasion"
        ),
    }
    for summary in summaries:
        for key, test in summary["primary_cross_modal_tests"].items():
            stats = test["summary"]
            ci = stats["mean_bootstrap_ci95"]
            lines.append(
                f"| {summary['display_name']} | {labels[key]} | {stats['n']} | "
                f"{_fmt(stats['mean'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
                f"{_fmt(stats['median'])} | {_fmt(stats['positive_rate'], 1)} | "
                f"{_fmt(test['full_boundary_crossing_rate'], 1)} |"
            )
    lines.extend(
        [
            "",
            "`Image CTT = D_full - D_text_only`; `Text CTT = D_full - "
            "D_image_only`. Evidence-free displacement measures persuasion "
            "without either news modality. Image-only values are diagnostic "
            "margins, not formal fact-checking accuracy.",
            "",
            "## Integrity",
            "",
            "| Model | Score rows | Derived rows | Errors | Complete |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        integrity = summary["integrity"]
        lines.append(
            f"| {summary['display_name']} | {integrity['score_rows']}/"
            f"{integrity['expected_score_rows']} | {integrity['derived_rows']}/"
            f"{integrity['expected_derived_rows']} | {integrity['system_errors']} | "
            f"{'yes' if integrity['formal_complete'] else 'no'} |"
        )
    report = _resolve(config["reporting"]["report"], project_root)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries
