# === FILE META START ===
# File: /src/prototype_induction/mmfakebench_modality_persuasion.py
# Category: Main experiment pipeline
# Author: project team
# Last Updated: 2026-09-14
# Purpose: Evaluate direct and explicit-CoT persuasion after formal baseline screening.
# Input: Fixed 4,126-row prompts, embedded/dynamic binary baselines, images, model config.
# Output: Independent/cascade predictions, FR metrics, and separated reasoning analyses.
# === FILE META END ===
"""Modality-specific persuasion over the formal MMFakeBench evaluation set."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml
from PIL import Image

from .io import append_jsonl, read_jsonl, write_jsonl_atomic
from .mmfakebench_baseline import Generation, split_reasoning
from .mmfakebench_formal import (
    load_formal_config,
    parse_task_prediction,
    prediction_path as formal_prediction_path,
    render_prompt,
    resolve_formal_image,
    strip_output_control_tokens,
)
from .mmfakebench_persuasion import paired_counts, reversal_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODES = ("independent", "cascade")
DIRECT_REASONING_MODE = "direct"
COT_MODES = (
    "generic_cot",
    "structured_multimodal_cot",
    "persuasion_audit_cot",
)
REASONING_MODES = (DIRECT_REASONING_MODE, *COT_MODES)
CONDITIONS = (
    "simple_disagreement",
    "repetition",
    "logical_appeal",
    "credibility_appeal",
    "emotional_appeal",
    "text_grounded_persuasion",
    "image_grounded_persuasion",
)
CONDITION_LABELS = {
    "simple_disagreement": "C0 Disagreement",
    "repetition": "C1 Repetition",
    "logical_appeal": "C2 Logical",
    "credibility_appeal": "C3 Credibility",
    "emotional_appeal": "C4 Emotional",
    "text_grounded_persuasion": "C5 Text-grounded",
    "image_grounded_persuasion": "C6 Image-grounded",
}
EXPECTED_CATEGORIES = {
    "original": 3300,
    "textual_veracity_distortion": 3300,
    "visual_veracity_distortion": 1100,
}


class PersuasionBackend(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def generate_messages(
        self, messages: Sequence[Mapping[str, Any]], seed: int
    ) -> Generation: ...


def _resolve(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_config(
    path: str | Path, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    required = {"dataset", "baseline", "prompt", "reporting"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {', '.join(missing)}")
    if tuple(config["prompt"].get("conditions", [])) != CONDITIONS:
        raise ValueError(f"condition order must be exactly {CONDITIONS}")
    formal_path = _resolve(config["baseline"]["formal_config"], project_root)
    formal = load_formal_config(formal_path)
    if config["baseline"].get("task") != "binary":
        raise ValueError("modality persuasion requires the formal binary baseline")
    config["_formal_config"] = formal
    config["_formal_config_path"] = str(formal_path)
    config["models"] = formal["models"]
    config["inference"] = formal["inference"]
    cot_modes = config.get("cot", {}).get("modes", {})
    for reasoning_mode in COT_MODES:
        spec = cot_modes.get(reasoning_mode)
        if not isinstance(spec, Mapping):
            raise ValueError(f"missing cot mode config: {reasoning_mode}")
        if not str(spec.get("response_instruction") or "").strip():
            raise ValueError(f"cot mode {reasoning_mode} has no response_instruction")
        if int(spec.get("max_new_tokens", 0)) < 1:
            raise ValueError(f"cot mode {reasoning_mode} must define max_new_tokens")
    for key, spec in config["models"].items():
        if spec.get("enabled") and spec.get("thinking"):
            max_new_tokens = int(
                spec.get("max_new_tokens", config["inference"].get("max_new_tokens", 0))
            )
            if max_new_tokens < 1:
                raise ValueError(f"thinking model {key} must define max_new_tokens")
    prompt_path = _resolve(config["dataset"]["prompts"], project_root)
    config["_prompts_sha256"] = _sha256(prompt_path) if prompt_path.is_file() else ""
    return config


def active_models(
    config: Mapping[str, Any], reasoning_mode: str = DIRECT_REASONING_MODE
) -> list[str]:
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"unsupported reasoning mode: {reasoning_mode}")
    return [
        key
        for key, spec in config["models"].items()
        if bool(spec.get("enabled")) and bool(spec.get("native_multimodal"))
        and (reasoning_mode == DIRECT_REASONING_MODE or not bool(spec.get("thinking")))
    ]


def reasoning_mode_spec(
    config: Mapping[str, Any], reasoning_mode: str
) -> Mapping[str, Any]:
    """Return the response instruction and generation budget for one mode."""
    if reasoning_mode == DIRECT_REASONING_MODE:
        return {
            "response_instruction": config["prompt"]["response_instruction"],
            "max_new_tokens": None,
        }
    if reasoning_mode not in COT_MODES:
        raise ValueError(f"unsupported reasoning mode: {reasoning_mode}")
    spec = config.get("cot", {}).get("modes", {}).get(reasoning_mode)
    if not isinstance(spec, Mapping):
        raise ValueError(f"missing cot mode config: {reasoning_mode}")
    return spec


def reasoning_token_limit(
    config: Mapping[str, Any], reasoning_mode: str, model_key: str
) -> int:
    """Resolve the actual response-token cap recorded for an experiment arm."""
    spec = reasoning_mode_spec(config, reasoning_mode)
    configured = spec.get("max_new_tokens")
    if configured is not None:
        return int(configured)
    model = config["models"][model_key]
    return int(
        model.get(
            "max_new_tokens",
            config.get("inference", {}).get("max_new_tokens", 128),
        )
    )


def validate_reasoning_model(
    config: Mapping[str, Any], model_key: str, reasoning_mode: str
) -> None:
    """Keep explicit CoT experiments separate from native-Thinking models."""
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"unsupported reasoning mode: {reasoning_mode}")
    if reasoning_mode != DIRECT_REASONING_MODE and bool(
        config["models"][model_key].get("thinking")
    ):
        raise ValueError(
            f"{reasoning_mode} requires a Non-thinking model; "
            f"{model_key} is configured as native Thinking"
        )


def load_prompts(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    path = _resolve(config["dataset"]["prompts"], project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"modality persuasion prompts not found: {path}; run the generator first"
        )
    return list(read_jsonl(path))


def baseline_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return formal_prediction_path(
        config["_formal_config"], "binary", model_key, project_root
    )


def dynamic_baseline_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    configured = config["baseline"].get(
        "dynamic_results_root",
        str(Path(config["reporting"]["results_root"]) / "baseline"),
    )
    root = _resolve(configured, project_root)
    output_name = str(config["models"][model_key]["output_name"])
    return root / output_name / "predictions.jsonl"


def _index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{label}: missing sample_id")
        if sample_id in result:
            raise ValueError(f"{label}: duplicate sample_id={sample_id}")
        result[sample_id] = dict(row)
    return result


def _embedded_baseline_state(
    prompt: Mapping[str, Any], model_key: str
) -> dict[str, Any] | None:
    predictions = prompt.get("baseline_predictions") or {}
    answers = prompt.get("baseline_assistant_answers") or {}
    if model_key not in predictions and model_key not in answers:
        return None
    prediction = str(predictions.get(model_key) or "")
    answer = str(answers.get(model_key) or "")
    parsed, _ = parse_task_prediction(answer, "binary")
    if prediction not in {"Fake", "Real"} or parsed != prediction:
        raise ValueError(
            f"invalid embedded baseline for {prompt.get('sample_id')} / {model_key}"
        )
    if prediction != str(prompt.get("gt")):
        raise ValueError(
            f"embedded baseline is not correct for {prompt.get('sample_id')} / {model_key}"
        )
    return {
        "prediction": prediction,
        "assistant_answer": answer,
        "parseable": True,
        "is_correct": True,
        "error": None,
        "source": "embedded_shared_correct_intersection",
    }


def _baseline_assistant_text(
    row: Mapping[str, Any], *, thinking: bool
) -> str:
    """Reconstruct the model's own baseline turn for the follow-up context."""
    prediction = str(row.get("prediction") or "")
    if thinking:
        reasoning = str(row.get("reasoning") or "").strip()
        final = str(row.get("final_answer") or "").strip()
        final = re.sub(r"<\|(?:im_end|endoftext)\|>", "", final).strip()
        if reasoning and final:
            return f"<think>\n{reasoning}\n</think>\n{final}"
        if final:
            return final
    return f"Finish[{prediction.upper()}]." if prediction in {"Fake", "Real"} else ""


def _persuasion_assistant_text(
    row: Mapping[str, Any], *, thinking: bool, reasoning_mode: str
) -> str:
    if reasoning_mode in COT_MODES:
        return strip_output_control_tokens(str(row.get("raw_response") or "")).strip()
    normalized = dict(row)
    normalized["prediction"] = row.get("final_prediction")
    return _baseline_assistant_text(normalized, thinking=thinking)


def baseline_states(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    formal_path = baseline_path(config, model_key, project_root)
    cache_path = dynamic_baseline_path(config, model_key, project_root)
    formal_rows = list(read_jsonl(formal_path)) if formal_path.is_file() else []
    cache_rows = list(read_jsonl(cache_path)) if cache_path.is_file() else []
    formal_index = _index(formal_rows, f"{model_key} formal binary baseline")
    cache_index = _index(cache_rows, f"{model_key} dynamic binary baseline")
    expected_ids = {str(row["sample_id"]) for row in prompts}
    formal = config["_formal_config"]
    prompt_hash = hashlib.sha256(
        str(formal["prompts"]["binary"]["template"]).encode()
    ).hexdigest()
    states: dict[str, dict[str, Any]] = {}
    counts = Counter()
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        embedded = _embedded_baseline_state(prompt, model_key)
        if embedded is not None:
            counts["correct"] += 1
            counts["embedded"] += 1
            states[sample_id] = embedded
            continue
        candidates = [
            ("formal_binary_result", formal_index.get(sample_id)),
            ("dynamic_screening", cache_index.get(sample_id)),
        ]
        present = [(source, row) for source, row in candidates if row is not None]
        if not present:
            counts["missing"] += 1
            continue
        valid = [
            (source, row)
            for source, row in present
            if row.get("task") == "binary" and row.get("prompt_sha256") == prompt_hash
        ]
        if not valid:
            counts["invalid"] += 1
            continue
        source, row = valid[0]
        error = str(row.get("error") or "").strip() or None
        prediction = str(row.get("prediction") or "Unparseable")
        parseable = prediction in {"Fake", "Real"} and error is None
        correct = parseable and prediction == prompt["gt"]
        outcome = "system_error" if error else ("correct" if correct else ("incorrect" if parseable else "unparseable"))
        counts[outcome] += 1
        states[sample_id] = {
            "prediction": prediction,
            "assistant_answer": (
                _baseline_assistant_text(
                    row,
                    thinking=bool(config["models"][model_key].get("thinking")),
                )
                if parseable
                else ""
            ),
            "parseable": parseable,
            "is_correct": correct,
            "error": error,
            "source": source,
            "reasoning": row.get("reasoning"),
            "final_answer": row.get("final_answer"),
        }
    extra = (set(formal_index) | set(cache_index)) - expected_ids
    status = {
        "formal_path": str(formal_path),
        "dynamic_path": str(cache_path),
        "total_expected": len(prompts),
        "formal_file_rows": len(formal_rows),
        "dynamic_file_rows": len(cache_rows),
        "resolved": len(states),
        "embedded": counts["embedded"],
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unparseable": counts["unparseable"],
        "system_errors": counts["system_error"],
        "missing": counts["missing"],
        "invalid": counts["invalid"],
        "extra": len(extra),
        "complete": (
            counts["missing"] == 0
            and counts["invalid"] == 0
            and counts["system_error"] == 0
        ),
    }
    return states, status


def validate_prompts(
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
    add("expected_sample_count", len(prompts) == expected_n, f"found={len(prompts)}, expected={expected_n}")
    add("unique_sample_ids", len(ids) == len(set(ids)) and all(ids), f"unique={len(set(ids))}, total={len(ids)}")
    categories = dict(sorted(Counter(str(row.get("fake_cls")) for row in prompts).items()))
    expected_categories = {
        str(key): int(value)
        for key, value in config["dataset"]["expected_category_counts"].items()
    }
    add("expected_category_counts", categories == expected_categories, f"found={categories}, expected={expected_categories}")
    if "expected_cohort_counts" in config["dataset"]:
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
        expected_cell = int(config["dataset"].get("expected_class_cohort_count", 0))
        cells = Counter(
            (str(row.get("fake_cls")), str(row.get("analysis_cohort")))
            for row in prompts
        )
        expected_cells = {
            (fake_cls, cohort): expected_cell
            for fake_cls in expected_categories
            for cohort in expected_cohorts
        }
        add(
            "balanced_class_cohort_cells",
            expected_cell > 0 and dict(cells) == expected_cells,
            f"found={dict(sorted(cells.items()))}, expected_each={expected_cell}",
        )
    excluded = set(map(str, config["dataset"].get("excluded_subcategories", [])))
    excluded_present = [
        str(row["sample_id"])
        for row in prompts
        if str(row.get("fake_subcategory")) in excluded
    ]
    add(
        "excluded_subcategories_absent",
        not excluded_present,
        f"excluded={sorted(excluded)}, violations={len(excluded_present)}",
    )
    embedded_errors: list[str] = []
    for row in prompts:
        for model_key in config["dataset"].get("reference_models", []):
            try:
                if _embedded_baseline_state(row, str(model_key)) is None:
                    embedded_errors.append(f"{row.get('sample_id')}:{model_key}:missing")
            except ValueError as exc:
                embedded_errors.append(str(exc))
    add(
        "reference_baselines_embedded_and_correct",
        not embedded_errors,
        f"violations={len(embedded_errors)}",
    )
    target_errors = [
        row["sample_id"]
        for row in prompts
        if row.get("persuasion_target") != ("Real" if row.get("gt") == "Fake" else "Fake")
    ]
    add("opposite_persuasion_targets", not target_errors, f"violations={len(target_errors)}")
    missing_conditions = [
        f"{row.get('sample_id')}:{condition}"
        for row in prompts
        for condition in CONDITIONS
        if not str(row.get(condition) or "").strip()
        and not (condition == "credibility_appeal" and row.get("credibility_available") is False)
    ]
    add("all_required_conditions_present", not missing_conditions, f"violations={len(missing_conditions)}")
    prompt_path = _resolve(config["dataset"]["prompts"], project_root)
    audit_path = _resolve(config["dataset"]["prompt_audit"], project_root)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        hash_ok = bool(audit.get("qc_passed")) and audit.get("output_sha256") == _sha256(prompt_path)
    except (OSError, json.JSONDecodeError):
        hash_ok = False
    add("prompt_qc_and_hash", hash_ok, f"audit={audit_path}")
    if check_images:
        unreadable: list[str] = []
        configured_root = config["dataset"]["source_root"]
        for row in prompts:
            path = resolve_formal_image(
                row,
                source_root=source_root,
                configured_source_root=configured_root,
                project_root=project_root,
            )
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError):
                unreadable.append(str(row["sample_id"]))
        add("all_images_readable", not unreadable, f"violations={len(unreadable)}")
    else:
        add("all_images_readable", True, "skipped")
    return findings


def result_dir(
    config: Mapping[str, Any],
    mode: str,
    model_key: str,
    project_root: Path,
    *,
    reasoning_mode: str = DIRECT_REASONING_MODE,
) -> Path:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"unsupported reasoning mode: {reasoning_mode}")
    output_name = str(config["models"][model_key]["output_name"])
    root = _resolve(config["reporting"]["results_root"], project_root)
    if reasoning_mode != DIRECT_REASONING_MODE:
        root = root / "cot" / reasoning_mode
    return root / mode / output_name


def prediction_path(
    config: Mapping[str, Any],
    mode: str,
    model_key: str,
    project_root: Path = PROJECT_ROOT,
    *,
    reasoning_mode: str = DIRECT_REASONING_MODE,
) -> Path:
    return result_dir(
        config,
        mode,
        model_key,
        project_root,
        reasoning_mode=reasoning_mode,
    ) / "predictions.jsonl"


_COT_FINAL_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?(?:final(?:\s+(?:answer|verdict))?\s*:\s*)?"
    r"(?:\*\*)?\s*(?:finish\s*\[\s*(?:real|fake)\s*\]\s*\.?|"
    r"(?:real|fake)\s*[.!]?)\s*(?:\*\*)?\s*$",
    flags=re.IGNORECASE,
)


def split_visible_cot(raw_response: str) -> tuple[str | None, str]:
    """Separate a visible CoT rationale from its required final-line verdict.

    The split is intentionally anchored to the last non-empty line. Mentions of
    Real or Fake inside the rationale cannot become the stored final answer.
    """
    lines = strip_output_control_tokens(raw_response).strip().splitlines()
    final_index = next(
        (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
        None,
    )
    if final_index is None:
        return None, ""
    final = lines[final_index].strip()
    if not _COT_FINAL_LINE_RE.fullmatch(final):
        return raw_response.strip() or None, ""
    reasoning = "\n".join(lines[:final_index]).strip() or None
    return reasoning, final


def separate_cot_generation(
    generation: Generation, reasoning_mode: str
) -> Generation:
    """Return a generation whose visible rationale and verdict are separate."""
    if reasoning_mode not in COT_MODES:
        return generation
    reasoning, final = split_visible_cot(generation.raw_response)
    metadata = dict(generation.metadata)
    metadata.update({
        "reasoning_mode": reasoning_mode,
        "visible_cot_split": bool(final),
    })
    return Generation(
        raw_response=generation.raw_response,
        reasoning=reasoning,
        final_answer=final,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        latency_seconds=generation.latency_seconds,
        metadata=metadata,
    )


_LEGACY_CONTROL_FRAGMENT_RE = re.compile(
    r"<\|?(?:channel|turn)\|?>",
    flags=re.IGNORECASE,
)


def _recover_legacy_control_wrapped_verdict(row: Mapping[str, Any]) -> str | None:
    """Recover a standalone verdict wrapped in malformed Gemma control tokens.

    Older Gemma decodes sometimes produced forms such as
    ``thought\n<channel|>Fake<turn|>`` or
    ``.<|channel>thought\n<channel|>Fake<turn|>``.  This recovery remains
    deliberately narrow: after removing only known control-token fragments,
    the channel marker, and harmless edge punctuation, the entire remaining
    text must be one Real/Fake verdict.  A rationale or truncated ``Finish[``
    sequence is therefore not accepted.
    """
    raw = str(row.get("final_answer") or row.get("raw_response") or "").strip()
    if not raw:
        return None
    cleaned = re.sub(r"<\|[^>]+\|>", " ", raw)
    cleaned = _LEGACY_CONTROL_FRAGMENT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\bthought\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n.:;-")
    match = re.fullmatch(
        r"(?:Finish\s*\[\s*)?(Real|Fake)(?:\s*\])?\s*\.?",
        cleaned,
        flags=re.IGNORECASE,
    )
    return match.group(1).capitalize() if match else None


def reparse_prediction_file(path: str | Path) -> int:
    """Recover stored persuasion verdicts and derived fields without inference.

    Raw responses remain unchanged. This migration is deliberately strict: it
    accepts only the normal binary answer syntax after known output-control
    tokens have been removed by ``parse_task_prediction``. An unfinished
    reasoning trace is never interpreted as a final verdict.
    """
    target = Path(path)
    if not target.is_file():
        return 0
    rows = list(read_jsonl(target))
    changed = 0
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if str(row.get("error") or "").strip():
            continue
        if row.get("status") == "skipped_unavailable":
            continue
        stored_final = str(row.get("final_answer") or "").strip()
        final = stored_final
        cot_reasoning: str | None = None
        reasoning_mode = str(row.get("reasoning_mode") or DIRECT_REASONING_MODE)
        if reasoning_mode in COT_MODES:
            cot_reasoning, cot_final = split_visible_cot(
                str(row.get("raw_response") or "")
            )
            if cot_final:
                final = cot_final
        if not final:
            reasoning, parsed_final = split_reasoning(
                str(row.get("raw_response") or "")
            )
            if reasoning is not None and not parsed_final:
                continue
            final = parsed_final
        prediction, _ = parse_task_prediction(final, "binary")
        if prediction == "Unparseable":
            prediction = _recover_legacy_control_wrapped_verdict(row) or prediction
        if prediction == "Unparseable":
            continue
        updates = {
            "final_prediction": prediction,
            "parseable": True,
            "flipped": prediction == str(row.get("persuasion_target")),
            "no_flip": prediction == str(row.get("gt")),
        }
        if cot_reasoning is not None:
            updates["reasoning"] = cot_reasoning
            updates["final_answer"] = final
        if any(row.get(key) != value for key, value in updates.items()):
            was_unparseable = not bool(row.get("parseable"))
            row.update(updates)
            if was_unparseable:
                row["reparsed_from_raw"] = True
                row["reparsed_at_utc"] = now
            changed += 1
    if changed:
        write_jsonl_atomic(target, rows)
    return changed


def prepare_error_retries(path: str | Path) -> tuple[int, Path]:
    target = Path(path)
    archive = target.parent / "failed_attempts.jsonl"
    if not target.is_file():
        return 0, archive
    rows = list(read_jsonl(target))
    failed = [row for row in rows if str(row.get("error") or "").strip()]
    if not failed:
        return 0, archive
    now = datetime.now(timezone.utc).isoformat()
    for row in failed:
        append_jsonl(archive, {**row, "archived_at_utc": now, "archive_reason": "system_error_retry"})
    write_jsonl_atomic(target, [row for row in rows if not str(row.get("error") or "").strip()])
    return len(failed), archive


def _user(text: str, image_path: Path | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append({"type": "image_path", "path": str(image_path)})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _followup(persuasion: str, instruction: str) -> str:
    return f"{persuasion.strip()}\n\n{instruction.strip()}"


def _conversation(
    prompt: Mapping[str, Any],
    image_path: Path,
    baseline_template: str,
    baseline_answer: str,
    prior_turns: Sequence[tuple[str, str]],
    current: str,
) -> list[dict[str, Any]]:
    messages = [
        _user(render_prompt(baseline_template, str(prompt["original_news_text"])), image_path),
        _assistant(baseline_answer),
    ]
    for user_text, answer in prior_turns:
        messages.extend([_user(user_text), _assistant(answer)])
    messages.append(_user(current))
    return messages


def prepare_baseline_error_retries(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> tuple[int, Path]:
    """Archive and remove system-error rows from the dynamic baseline cache."""
    target = dynamic_baseline_path(config, model_key, project_root)
    archive = target.parent / "failed_attempts.jsonl"
    if not target.is_file():
        return 0, archive
    rows = list(read_jsonl(target))
    failed = [row for row in rows if str(row.get("error") or "").strip()]
    if not failed:
        return 0, archive
    now = datetime.now(timezone.utc).isoformat()
    for row in failed:
        append_jsonl(
            archive,
            {**row, "archived_at_utc": now, "archive_reason": "baseline_system_error_retry"},
        )
    write_jsonl_atomic(
        target,
        [row for row in rows if not str(row.get("error") or "").strip()],
    )
    return len(failed), archive


def _dynamic_baseline_record(
    *,
    prompt: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    generation: Generation | None,
    prediction: str,
    output_class: str | None,
    error: str | None,
    backend_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    gt = str(prompt["gt"])
    parseable = error is None and prediction in {"Fake", "Real"}
    correct = parseable and prediction == gt
    baseline_template = str(config["_formal_config"]["prompts"]["binary"]["template"])
    return {
        "sample_id": str(prompt["sample_id"]),
        "task": "binary",
        "model_key": model_key,
        "model": config["models"][model_key]["hub_id"],
        "gt": gt,
        "fake_cls": prompt["fake_cls"],
        "fake_subcategory": prompt.get("fake_subcategory"),
        "analysis_cohort": prompt.get("analysis_cohort"),
        "selector_model": prompt.get("selector_model"),
        "matched_pair_id": prompt.get("matched_pair_id"),
        "selector_successful_conditions": prompt.get(
            "selector_successful_conditions"
        ),
        "selector_n_successful_conditions": prompt.get(
            "selector_n_successful_conditions"
        ),
        "split": prompt.get("split"),
        "prediction": prediction,
        "prediction_class": output_class,
        "parseable": parseable,
        "is_correct": correct,
        "task_correct": correct,
        "binary_correct": correct,
        "correctness": (
            "SystemError"
            if error
            else "Unparseable"
            if prediction == "Unparseable"
            else "Correct"
            if correct
            else "Incorrect"
        ),
        "raw_response": generation.raw_response if generation else "",
        "reasoning": generation.reasoning if generation else None,
        "final_answer": generation.final_answer if generation else "",
        "input_tokens": generation.input_tokens if generation else 0,
        "output_tokens": generation.output_tokens if generation else 0,
        "latency_seconds": generation.latency_seconds if generation else 0.0,
        "seed": int(config["dataset"]["seed"]),
        "do_sample": False,
        "temperature": 0.0,
        "prompt_name": config["_formal_config"]["prompts"]["binary"].get(
            "name", "mmfakebench_binary"
        ),
        "prompt_sha256": hashlib.sha256(baseline_template.encode()).hexdigest(),
        "backend_metadata": dict(backend_metadata),
        "generation_metadata": generation.metadata if generation else {},
        "error": error,
        "screening_scope": "filtered_shared_correct_4126",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _generate_dynamic_baseline(
    prompt: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    backend: PersuasionBackend,
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], bool]:
    """Generate and append one baseline, returning its in-memory state."""
    baseline_template = str(config["_formal_config"]["prompts"]["binary"]["template"])
    image_path = resolve_formal_image(
        prompt,
        source_root=source_root,
        configured_source_root=config["dataset"]["source_root"],
        project_root=project_root,
    )
    messages = [
        _user(
            render_prompt(baseline_template, str(prompt["original_news_text"])),
            image_path,
        )
    ]
    fatal_oom = False
    try:
        generation = backend.generate_messages(
            messages, int(config["dataset"]["seed"])
        )
        prediction, output_class = parse_task_prediction(
            generation.final_answer, "binary"
        )
        error = None
    except Exception as exc:
        generation = None
        prediction, output_class = "Unparseable", None
        error = f"{type(exc).__name__}: {exc}"
        fatal_oom = (
            type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in str(exc).casefold()
        )
    record = _dynamic_baseline_record(
        prompt=prompt,
        config=config,
        model_key=model_key,
        generation=generation,
        prediction=prediction,
        output_class=output_class,
        error=error,
        backend_metadata=backend.metadata,
    )
    append_jsonl(dynamic_baseline_path(config, model_key, project_root), record)
    parseable = error is None and prediction in {"Fake", "Real"}
    state = {
        "prediction": prediction,
        "assistant_answer": (
            _baseline_assistant_text(record, thinking=bool(config["models"][model_key].get("thinking")))
            if parseable
            else ""
        ),
        "parseable": parseable,
        "is_correct": bool(record["is_correct"]),
        "error": error,
        "source": "dynamic_screening",
        "reasoning": record.get("reasoning"),
        "final_answer": record.get("final_answer"),
    }
    return state, fatal_oom


def ensure_model_baselines(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    backend: PersuasionBackend,
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], int]:
    """Resolve embedded/formal baselines and screen any missing model once."""
    states, status = baseline_states(prompts, config, model_key, project_root)
    missing = [
        prompt for prompt in prompts if str(prompt["sample_id"]) not in states
    ]
    if missing and not config["baseline"].get("allow_dynamic_screening", False):
        raise RuntimeError(
            f"{model_key} has {len(missing)} unresolved baselines and dynamic screening is disabled"
        )
    output = dynamic_baseline_path(config, model_key, project_root)
    existing_rows = list(read_jsonl(output)) if output.is_file() else []
    existing_ids = {str(row.get("sample_id")) for row in existing_rows}
    if len(existing_ids) != len(existing_rows):
        raise ValueError(f"duplicate sample_id in dynamic baseline cache: {output}")
    generated_count = 0
    for prompt in missing:
        sample_id = str(prompt["sample_id"])
        if sample_id in existing_ids:
            raise RuntimeError(f"unresolved cached baseline cannot be overwritten: {sample_id}")
        state, fatal_oom = _generate_dynamic_baseline(
            prompt,
            config,
            model_key,
            backend,
            source_root=source_root,
            project_root=project_root,
        )
        existing_ids.add(sample_id)
        generated_count += 1
        print(
            f"[baseline/{model_key}] {generated_count}/{len(missing)}: "
            f"{sample_id} -> {state['prediction']}"
            f"{' CORRECT' if state['is_correct'] else ''}"
        )
        if fatal_oom:
            raise RuntimeError(
                f"CUDA OOM at baseline/{sample_id}; free memory and rerun with --retry-errors"
            )
    states, status = baseline_states(prompts, config, model_key, project_root)
    return states, status, generated_count


def _relation(fake_cls: str, condition: str) -> str:
    if fake_cls == "textual_veracity_distortion":
        return "cross_modal_whitewashing_core" if condition == "image_grounded_persuasion" else "same_modality_defense"
    if fake_cls == "visual_veracity_distortion":
        return "cross_modal_whitewashing_core" if condition == "text_grounded_persuasion" else "same_modality_defense"
    return "original_false_rejection_control"


def _make_record(
    *,
    prompt: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    mode: str,
    reasoning_mode: str,
    max_new_tokens: int,
    condition: str,
    condition_index: int,
    persuasion_text: str | None,
    history: Sequence[str],
    baseline_answer: str,
    baseline_source: str,
    generation: Generation | None,
    prediction: str | None,
    error: str | None,
    status: str,
    backend_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    target = str(prompt["persuasion_target"])
    parseable = prediction in {"Fake", "Real"}
    relation = _relation(str(prompt["fake_cls"]), condition) if condition in CONDITIONS[-2:] else "generic"
    run_prefix = mode if reasoning_mode == DIRECT_REASONING_MODE else f"{reasoning_mode}:{mode}"
    return {
        "run_id": f"{run_prefix}:{model_key}:{prompt['sample_id']}:{condition}",
        "sample_id": prompt["sample_id"],
        "model_key": model_key,
        "model": config["models"][model_key]["hub_id"],
        "mode": mode,
        "reasoning_mode": reasoning_mode,
        "reasoning_kind": (
            "visible_cot" if reasoning_mode in COT_MODES else "native_or_none"
        ),
        "max_new_tokens": max_new_tokens,
        "condition": condition,
        "condition_index": condition_index,
        "condition_label": CONDITION_LABELS[condition],
        "modality_relation": relation,
        "status": status,
        "gt": prompt["gt"],
        "fake_cls": prompt["fake_cls"],
        "fake_subcategory": prompt.get("fake_subcategory"),
        "analysis_cohort": prompt.get("analysis_cohort"),
        "selector_model": prompt.get("selector_model"),
        "matched_pair_id": prompt.get("matched_pair_id"),
        "selector_successful_conditions": prompt.get(
            "selector_successful_conditions"
        ),
        "selector_n_successful_conditions": prompt.get(
            "selector_n_successful_conditions"
        ),
        "baseline_prediction": prompt["gt"],
        "baseline_assistant_answer": f"Finish[{str(prompt['gt']).upper()}].",
        "baseline_source": baseline_source,
        "baseline_context_includes_reasoning": "</think>" in baseline_answer,
        "baseline_context_sha256": hashlib.sha256(
            baseline_answer.encode("utf-8")
        ).hexdigest(),
        "persuasion_target": target,
        "persuasion_text": persuasion_text,
        "history_conditions": list(history),
        "final_prediction": prediction,
        "flipped": bool(parseable and prediction == target),
        "no_flip": bool(parseable and prediction == prompt["gt"]),
        "parseable": parseable,
        "raw_response": generation.raw_response if generation else "",
        "reasoning": generation.reasoning if generation else None,
        "final_answer": generation.final_answer if generation else "",
        "input_tokens": generation.input_tokens if generation else 0,
        "output_tokens": generation.output_tokens if generation else 0,
        "latency_seconds": generation.latency_seconds if generation else 0.0,
        "generation_metadata": generation.metadata if generation else {},
        "backend_metadata": dict(backend_metadata),
        "error": error,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_signature": _canonical_hash({
            "prompt_sha256": config["_prompts_sha256"],
            "baseline_config": config["_formal_config_path"],
            "conditions": CONDITIONS,
            "response_instruction": reasoning_mode_spec(
                config, reasoning_mode
            )["response_instruction"],
            "mode": mode,
            "reasoning_mode": reasoning_mode,
            "max_new_tokens": max_new_tokens,
            "model_key": model_key,
        }),
    }


def run_model(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    mode: str,
    backend: PersuasionBackend,
    *,
    reasoning_mode: str = DIRECT_REASONING_MODE,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    validate_reasoning_model(config, model_key, reasoning_mode)
    states, baseline_status = baseline_states(prompts, config, model_key, project_root)
    output = prediction_path(
        config,
        mode,
        model_key,
        project_root,
        reasoning_mode=reasoning_mode,
    )
    reparsed_existing = reparse_prediction_file(output)
    if reparsed_existing:
        print(
            f"[repair/{mode}/{model_key}] recovered "
            f"{reparsed_existing} stored verdicts without inference"
        )
    existing_rows = list(read_jsonl(output)) if output.is_file() else []
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing_rows:
        key = (str(row.get("sample_id")), str(row.get("condition")))
        if key in existing:
            raise ValueError(f"duplicate existing result: {key}")
        existing[key] = dict(row)
    baseline_cache = dynamic_baseline_path(config, model_key, project_root)
    baseline_cache_rows = (
        list(read_jsonl(baseline_cache)) if baseline_cache.is_file() else []
    )
    baseline_cache_ids = {
        str(row.get("sample_id")) for row in baseline_cache_rows
    }
    if len(baseline_cache_ids) != len(baseline_cache_rows):
        raise ValueError(f"duplicate sample_id in dynamic baseline cache: {baseline_cache}")
    baseline_template = str(config["_formal_config"]["prompts"]["binary"]["template"])
    instruction = str(
        reasoning_mode_spec(config, reasoning_mode)["response_instruction"]
    )
    max_new_tokens = reasoning_token_limit(config, reasoning_mode, model_key)
    seed = int(config["dataset"]["seed"])
    configured_root = config["dataset"]["source_root"]
    thinking = bool(config["models"][model_key].get("thinking"))
    cascade_stop_on_flip = bool(
        config.get("prompt", {}).get("cascade_stop_on_flip", True)
    )
    log_every = max(1, int(config.get("inference", {}).get("log_every", 10)))
    written = generations = errors = baseline_generated = 0
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        baseline_state = states.get(sample_id)
        if baseline_state is None:
            if not config["baseline"].get("allow_dynamic_screening", False):
                raise RuntimeError(
                    f"{model_key} has no valid baseline for {sample_id} and dynamic screening is disabled"
                )
            if sample_id in baseline_cache_ids:
                raise RuntimeError(
                    f"unresolved cached baseline cannot be overwritten: {sample_id}"
                )
            baseline_state, fatal_baseline_oom = _generate_dynamic_baseline(
                prompt,
                config,
                model_key,
                backend,
                source_root=source_root,
                project_root=project_root,
            )
            states[sample_id] = baseline_state
            baseline_cache_ids.add(sample_id)
            baseline_generated += 1
            if baseline_generated == 1 or baseline_generated % log_every == 0:
                print(
                    f"[baseline/{model_key}] {baseline_generated}: {sample_id} -> "
                    f"{baseline_state['prediction']}"
                    f"{' CORRECT' if baseline_state['is_correct'] else ''}"
                )
            if fatal_baseline_oom:
                raise RuntimeError(
                    f"CUDA OOM at baseline/{sample_id}; free memory and rerun with --retry-errors"
                )
        if baseline_state is None or not baseline_state.get("is_correct"):
            continue
        baseline_answer = str(baseline_state["assistant_answer"])
        image_path = resolve_formal_image(
            prompt,
            source_root=source_root,
            configured_source_root=configured_root,
            project_root=project_root,
        )
        history: list[str] = []
        prior_turns: list[tuple[str, str]] = []
        terminal = False
        for index, condition in enumerate(CONDITIONS):
            if mode == "cascade" and terminal:
                break
            key = (sample_id, condition)
            if key in existing:
                saved = existing[key]
                if mode == "cascade":
                    if saved.get("status") == "skipped_unavailable":
                        continue
                    if (
                        saved.get("error")
                        or not saved.get("parseable")
                        or (saved.get("flipped") and cascade_stop_on_flip)
                    ):
                        terminal = True
                        break
                    if not saved.get("no_flip") and not saved.get("flipped"):
                        raise RuntimeError(f"invalid saved cascade state: {key}")
                    prior_turns.append(
                        (
                            str(saved["full_followup_text"]),
                            _persuasion_assistant_text(
                                saved,
                                thinking=thinking,
                                reasoning_mode=reasoning_mode,
                            ),
                        )
                    )
                    history.append(condition)
                continue
            persuasion = prompt.get(condition)
            available = not (
                condition == "credibility_appeal"
                and prompt.get("credibility_available") is False
            )
            if not available:
                record = _make_record(
                    prompt=prompt, config=config, model_key=model_key, mode=mode,
                    reasoning_mode=reasoning_mode, max_new_tokens=max_new_tokens,
                    condition=condition, condition_index=index, persuasion_text=None,
                    history=history, baseline_answer=baseline_answer,
                    baseline_source=str(baseline_state["source"]), generation=None,
                    prediction=None, error=None, status="skipped_unavailable",
                    backend_metadata=backend.metadata,
                )
                record["full_followup_text"] = None
                append_jsonl(output, record)
                existing[key] = record
                written += 1
                continue
            current = _followup(str(persuasion), instruction)
            messages = _conversation(
                prompt, image_path, baseline_template, baseline_answer,
                prior_turns if mode == "cascade" else [], current,
            )
            fatal_oom = False
            try:
                generation = backend.generate_messages(messages, seed)
                generation = separate_cot_generation(generation, reasoning_mode)
                prediction, _ = parse_task_prediction(generation.final_answer, "binary")
                error = None
            except Exception as exc:
                generation = None
                prediction = "Unparseable"
                error = f"{type(exc).__name__}: {exc}"
                fatal_oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).casefold()
                errors += 1
            record = _make_record(
                prompt=prompt, config=config, model_key=model_key, mode=mode,
                reasoning_mode=reasoning_mode, max_new_tokens=max_new_tokens,
                condition=condition, condition_index=index, persuasion_text=str(persuasion),
                history=history, baseline_answer=baseline_answer,
                baseline_source=str(baseline_state["source"]), generation=generation,
                prediction=prediction, error=error,
                status="system_error" if error else "completed",
                backend_metadata=backend.metadata,
            )
            record["full_followup_text"] = current
            append_jsonl(output, record)
            existing[key] = record
            written += 1
            generations += 1
            if generations == 1 or generations % log_every == 0:
                print(
                    f"[{mode}/{model_key}] generations={generations}: "
                    f"{sample_id} {condition} -> {prediction}"
                    f"{' FLIPPED' if record['flipped'] else ''}"
                )
            if fatal_oom:
                raise RuntimeError(
                    f"CUDA OOM at {sample_id}/{condition}; free memory and rerun with --retry-errors"
                )
            if mode == "cascade":
                if (
                    error
                    or not record["parseable"]
                    or (record["flipped"] and cascade_stop_on_flip)
                ):
                    terminal = True
                else:
                    prior_turns.append(
                        (
                            current,
                            _persuasion_assistant_text(
                                record,
                                thinking=thinking,
                                reasoning_mode=reasoning_mode,
                            ),
                        )
                    )
                    history.append(condition)
    _, baseline_status = baseline_states(prompts, config, model_key, project_root)
    baseline_integrity_errors = (
        baseline_status["missing"]
        + baseline_status["invalid"]
        + baseline_status["system_errors"]
    )
    return {
        "baseline_total": baseline_status["total_expected"],
        "baseline_generated": baseline_generated,
        "baseline_correct_eligible": baseline_status["correct"],
        "baseline_incorrect": baseline_status["incorrect"],
        "baseline_unparseable": baseline_status["unparseable"],
        "baseline_system_errors": baseline_status["system_errors"],
        "baseline_missing": baseline_status["missing"],
        "reparsed_existing": reparsed_existing,
        "reasoning_mode": reasoning_mode,
        "max_new_tokens": max_new_tokens,
        "existing": len(existing_rows),
        "written": written,
        "generations": generations,
        "errors": errors + baseline_integrity_errors,
        "total_records": len(existing),
    }


def _subset(
    rows: Sequence[Mapping[str, Any]], fake_cls: str, condition: str
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row.get("fake_cls") == fake_cls and row.get("condition") == condition
    ]


def _rate(rows: Sequence[Mapping[str, Any]], fake_cls: str, condition: str) -> dict[str, Any]:
    return reversal_metrics(_subset(rows, fake_cls, condition))


def modality_specific_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text_same = _rate(rows, "textual_veracity_distortion", "text_grounded_persuasion")
    text_cross = _rate(rows, "textual_veracity_distortion", "image_grounded_persuasion")
    image_cross = _rate(rows, "visual_veracity_distortion", "text_grounded_persuasion")
    image_same = _rate(rows, "visual_veracity_distortion", "image_grounded_persuasion")
    original_text = _rate(rows, "original", "text_grounded_persuasion")
    original_image = _rate(rows, "original", "image_grounded_persuasion")

    def delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
        a, b = left.get("reversal_rate"), right.get("reversal_rate")
        return float(a) - float(b) if a is not None and b is not None else None

    text_rows = [row for row in rows if row.get("fake_cls") == "textual_veracity_distortion"]
    image_rows = [row for row in rows if row.get("fake_cls") == "visual_veracity_distortion"]
    original_rows = [row for row in rows if row.get("fake_cls") == "original"]
    return {
        "FR_TextError_ImagePersuasion": text_cross,
        "FR_ImageError_TextPersuasion": image_cross,
        "same_modality_controls": {
            "FR_TextError_TextPersuasion": text_same,
            "FR_ImageError_ImagePersuasion": image_same,
        },
        "original_false_rejection_controls": {
            "FR_Original_TextPersuasion": original_text,
            "FR_Original_ImagePersuasion": original_image,
        },
        "cross_minus_same": {
            "TextError_ImageMinusText": delta(text_cross, text_same),
            "ImageError_TextMinusImage": delta(image_cross, image_same),
        },
        "paired_text_error_image_vs_text": paired_counts(
            text_rows, "text_grounded_persuasion", "image_grounded_persuasion"
        ),
        "paired_image_error_text_vs_image": paired_counts(
            image_rows, "image_grounded_persuasion", "text_grounded_persuasion"
        ),
        "paired_original_text_vs_image": paired_counts(
            original_rows, "text_grounded_persuasion", "image_grounded_persuasion"
        ),
    }


def independent_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    subcategory_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cohort_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("fake_cls"))].append(row)
        subcategory_groups[str(row.get("fake_subcategory"))].append(row)
        if row.get("analysis_cohort") is not None:
            cohort_groups[str(row.get("analysis_cohort"))].append(row)
    return {
        "by_condition": {
            condition: reversal_metrics([row for row in rows if row.get("condition") == condition])
            for condition in CONDITIONS
        },
        "by_fake_cls": {
            fake_cls: {
                condition: reversal_metrics([row for row in group if row.get("condition") == condition])
                for condition in CONDITIONS
            }
            for fake_cls, group in sorted(groups.items())
        },
        "by_fake_subcategory": {
            subcategory: {
                condition: reversal_metrics(
                    [row for row in group if row.get("condition") == condition]
                )
                for condition in CONDITIONS
            }
            for subcategory, group in sorted(subcategory_groups.items())
        },
        "by_analysis_cohort": {
            cohort: {
                condition: reversal_metrics(
                    [row for row in group if row.get("condition") == condition]
                )
                for condition in CONDITIONS
            }
            for cohort, group in sorted(cohort_groups.items())
        },
        "by_fake_cls_and_cohort": {
            fake_cls: {
                cohort: {
                    condition: reversal_metrics([
                        row for row in rows
                        if row.get("fake_cls") == fake_cls
                        and row.get("analysis_cohort") == cohort
                        and row.get("condition") == condition
                    ])
                    for condition in CONDITIONS
                }
                for cohort in sorted(cohort_groups)
            }
            for fake_cls in sorted(groups)
        },
        "modality_specific": modality_specific_metrics(rows),
    }


def cascade_summary(
    rows: Sequence[Mapping[str, Any]],
    eligible_count: int,
    *,
    _include_cohorts: bool = True,
) -> dict[str, Any]:
    first_flip = Counter()
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row.get("sample_id"))].append(row)
    terminal = Counter()
    for sample_rows in by_sample.values():
        actual = [row for row in sample_rows if row.get("status") != "skipped_unavailable"]
        flipped = next((row for row in actual if row.get("flipped")), None)
        if flipped:
            first_flip[str(flipped["condition"])] += 1
            terminal["flipped"] += 1
        elif any(row.get("error") for row in actual):
            terminal["system_error"] += 1
        elif any(not row.get("parseable") for row in actual):
            terminal["indeterminate_unparseable"] += 1
        elif any(row.get("condition") == CONDITIONS[-1] for row in actual):
            terminal["never_flipped"] += 1
        else:
            terminal["incomplete"] += 1
    cumulative: dict[str, Any] = {}
    running = 0
    for condition in CONDITIONS:
        running += first_flip[condition]
        cumulative[condition] = {
            "cumulative_flips": running,
            "cumulative_reversal_rate": running / eligible_count if eligible_count else None,
        }
    result = {
        "eligible_samples": eligible_count,
        "by_stage": {
            condition: reversal_metrics([row for row in rows if row.get("condition") == condition])
            for condition in CONDITIONS
        },
        "by_fake_subcategory_stage": {
            subcategory: {
                condition: reversal_metrics([
                    row
                    for row in rows
                    if row.get("fake_subcategory") == subcategory
                    and row.get("condition") == condition
                ])
                for condition in CONDITIONS
            }
            for subcategory in sorted(
                {str(row.get("fake_subcategory")) for row in rows}
            )
        },
        "first_flip_stage": {condition: first_flip[condition] for condition in CONDITIONS},
        "cumulative": cumulative,
        "terminal_outcomes": dict(sorted(terminal.items())),
        "modality_specific_stage_metrics_exploratory": modality_specific_metrics(rows),
        "warning": "Modality-specific stage FR in cascade has survivor and history bias; use independent mode for primary FR.",
    }
    cohorts = sorted({
        str(row.get("analysis_cohort"))
        for row in rows
        if row.get("analysis_cohort") is not None
    })
    if _include_cohorts and cohorts:
        result["by_analysis_cohort"] = {}
        result["by_fake_cls_and_cohort"] = {}
        for cohort in cohorts:
            cohort_rows = [
                row for row in rows if row.get("analysis_cohort") == cohort
            ]
            cohort_ids = {str(row.get("sample_id")) for row in cohort_rows}
            result["by_analysis_cohort"][cohort] = cascade_summary(
                cohort_rows,
                len(cohort_ids),
                _include_cohorts=False,
            )
        for fake_cls in sorted({str(row.get("fake_cls")) for row in rows}):
            result["by_fake_cls_and_cohort"][fake_cls] = {}
            for cohort in cohorts:
                cell_rows = [
                    row for row in rows
                    if row.get("fake_cls") == fake_cls
                    and row.get("analysis_cohort") == cohort
                ]
                cell_ids = {str(row.get("sample_id")) for row in cell_rows}
                result["by_fake_cls_and_cohort"][fake_cls][cohort] = (
                    cascade_summary(
                        cell_rows,
                        len(cell_ids),
                        _include_cohorts=False,
                    )
                )
    return result


def _integrity(
    rows: Sequence[Mapping[str, Any]],
    eligible_ids: set[str],
    mode: str,
    *,
    cascade_stop_on_flip: bool = True,
) -> dict[str, Any]:
    keys = [(str(row.get("sample_id")), str(row.get("condition"))) for row in rows]
    duplicates = len(keys) - len(set(keys))
    errors = sum(bool(str(row.get("error") or "").strip()) for row in rows)
    unexpected = sum(sample_id not in eligible_ids or condition not in CONDITIONS for sample_id, condition in keys)
    if mode == "independent" or (mode == "cascade" and not cascade_stop_on_flip):
        expected = {(sample_id, condition) for sample_id in eligible_ids for condition in CONDITIONS}
        complete = set(keys) == expected and duplicates == 0 and errors == 0
    else:
        complete_samples = 0
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("sample_id"))].append(row)
        for sample_id in eligible_ids:
            sample_rows = sorted(grouped.get(sample_id, []), key=lambda row: int(row.get("condition_index", -1)))
            if not sample_rows:
                continue
            available = [row for row in sample_rows if row.get("status") != "skipped_unavailable"]
            terminal = any(row.get("flipped") or row.get("error") or not row.get("parseable") for row in available)
            exhausted = any(row.get("condition") == CONDITIONS[-1] for row in available)
            if terminal or exhausted:
                complete_samples += 1
        complete = complete_samples == len(eligible_ids) and duplicates == 0 and errors == 0
    return {
        "row_count": len(rows),
        "eligible_samples": len(eligible_ids),
        "duplicate_keys": duplicates,
        "unexpected_keys": unexpected,
        "system_errors": errors,
        "formal_complete": complete and unexpected == 0,
    }


def _percentile(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(quantile * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _trace_stats(
    rows: Sequence[Mapping[str, Any]], max_new_tokens: int
) -> dict[str, Any]:
    output_tokens = [int(row.get("output_tokens") or 0) for row in rows]
    reasoning = [str(row.get("reasoning") or "").strip() for row in rows]
    reasoning_words = [len(text.split()) for text in reasoning if text]
    reasoning_chars = [len(text) for text in reasoning if text]
    latencies = [float(row.get("latency_seconds") or 0.0) for row in rows]
    token_limit_hits = sum(value >= max_new_tokens for value in output_tokens)
    return {
        "rows": len(rows),
        "reasoning_present": sum(bool(text) for text in reasoning),
        "reasoning_present_rate": (
            sum(bool(text) for text in reasoning) / len(rows) if rows else None
        ),
        "mean_output_tokens": (
            sum(output_tokens) / len(output_tokens) if output_tokens else None
        ),
        "median_output_tokens": _percentile(output_tokens, 0.5),
        "p95_output_tokens": _percentile(output_tokens, 0.95),
        "mean_reasoning_words": (
            sum(reasoning_words) / len(reasoning_words) if reasoning_words else None
        ),
        "mean_reasoning_chars": (
            sum(reasoning_chars) / len(reasoning_chars) if reasoning_chars else None
        ),
        "mean_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "max_new_tokens": max_new_tokens,
        "token_limit_hits": token_limit_hits,
        "token_limit_hit_rate": token_limit_hits / len(rows) if rows else None,
    }


def _persuasion_outcome(row: Mapping[str, Any]) -> str:
    if str(row.get("error") or "").strip():
        return "system_error"
    if row.get("status") == "skipped_unavailable":
        return "skipped_unavailable"
    if not row.get("parseable"):
        return "unparseable"
    if row.get("flipped"):
        return "flipped"
    if row.get("no_flip"):
        return "resisted"
    return "other"


def _baseline_outcome(row: Mapping[str, Any]) -> str:
    if str(row.get("error") or "").strip():
        return "system_error"
    prediction = str(row.get("prediction") or "Unparseable")
    if prediction == "Unparseable":
        return "unparseable"
    return "correct" if row.get("is_correct") else "incorrect"


def thinking_analysis(
    persuasion_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Summarize saved Thinking traces without semantically judging hidden reasoning."""
    baseline_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_groups[_baseline_outcome(row)].append(row)
    persuasion_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    condition_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    class_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    subcategory_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in persuasion_rows:
        persuasion_groups[_persuasion_outcome(row)].append(row)
        condition_groups[str(row.get("condition"))].append(row)
        class_groups[str(row.get("fake_cls"))].append(row)
        subcategory_groups[str(row.get("fake_subcategory"))].append(row)
    return {
        "baseline": {
            "overall": _trace_stats(baseline_rows, max_new_tokens),
            "by_outcome": {
                key: _trace_stats(group, max_new_tokens)
                for key, group in sorted(baseline_groups.items())
            },
        },
        "persuasion": {
            "overall": _trace_stats(persuasion_rows, max_new_tokens),
            "by_outcome": {
                key: _trace_stats(group, max_new_tokens)
                for key, group in sorted(persuasion_groups.items())
            },
            "by_condition": {
                condition: _trace_stats(condition_groups.get(condition, []), max_new_tokens)
                for condition in CONDITIONS
            },
            "by_fake_cls": {
                key: _trace_stats(group, max_new_tokens)
                for key, group in sorted(class_groups.items())
            },
            "by_fake_subcategory": {
                key: _trace_stats(group, max_new_tokens)
                for key, group in sorted(subcategory_groups.items())
            },
        },
        "interpretation_boundary": (
            "These are trace availability, length, latency, truncation, and outcome-stratified "
            "descriptives. They do not treat chain-of-thought text as a faithful causal explanation."
        ),
    }


def _reasoning_export(
    baseline_rows: Sequence[Mapping[str, Any]],
    persuasion_rows: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for phase, rows in (("baseline", baseline_rows), ("persuasion", persuasion_rows)):
        for row in rows:
            reasoning = str(row.get("reasoning") or "").strip()
            exported.append({
                "phase": phase,
                "sample_id": row.get("sample_id"),
                "condition": row.get("condition") if phase == "persuasion" else None,
                "reasoning_mode": row.get("reasoning_mode", DIRECT_REASONING_MODE),
                "fake_cls": row.get("fake_cls"),
                "fake_subcategory": row.get("fake_subcategory"),
                "gt": row.get("gt"),
                "baseline_outcome": _baseline_outcome(row) if phase == "baseline" else None,
                "persuasion_outcome": _persuasion_outcome(row) if phase == "persuasion" else None,
                "final_prediction": (
                    row.get("prediction") if phase == "baseline" else row.get("final_prediction")
                ),
                "reasoning_present": bool(reasoning),
                "reasoning_words": len(reasoning.split()) if reasoning else 0,
                "reasoning_chars": len(reasoning),
                "output_tokens": int(row.get("output_tokens") or 0),
                "reached_max_new_tokens": int(row.get("output_tokens") or 0) >= max_new_tokens,
                "latency_seconds": float(row.get("latency_seconds") or 0.0),
                "reasoning": row.get("reasoning"),
                "final_answer": row.get("final_answer"),
            })
    return exported


def aggregate_model(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    mode: str,
    *,
    reasoning_mode: str = DIRECT_REASONING_MODE,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    validate_reasoning_model(config, model_key, reasoning_mode)
    states, baseline_status = baseline_states(prompts, config, model_key, project_root)
    eligible_ids = {
        sample_id for sample_id, state in states.items() if state.get("is_correct")
    }
    path = prediction_path(
        config,
        mode,
        model_key,
        project_root,
        reasoning_mode=reasoning_mode,
    )
    reparse_prediction_file(path)
    rows = list(read_jsonl(path)) if path.is_file() else []
    model_spec = config["models"][model_key]
    thinking = bool(model_spec.get("thinking"))
    max_new_tokens = reasoning_token_limit(config, reasoning_mode, model_key)
    selected_ids = {str(row["sample_id"]) for row in prompts}
    dynamic_path = dynamic_baseline_path(config, model_key, project_root)
    baseline_rows = [
        row
        for row in (list(read_jsonl(dynamic_path)) if dynamic_path.is_file() else [])
        if str(row.get("sample_id")) in selected_ids
    ]
    has_reasoning = thinking or reasoning_mode in COT_MODES
    reasoning_summary = (
        thinking_analysis(
            rows,
            baseline_rows if thinking else [],
            max_new_tokens,
        )
        if has_reasoning
        else None
    )
    reasoning_export_path: str | None = None
    if (
        has_reasoning
        and config.get("analysis", {}).get("thinking", {}).get(
            "export_reasoning_rows", True
        )
    ):
        export_path = result_dir(
            config,
            mode,
            model_key,
            project_root,
            reasoning_mode=reasoning_mode,
        ) / "reasoning_analysis.jsonl"
        write_jsonl_atomic(
            export_path,
            _reasoning_export(baseline_rows if thinking else [], rows, max_new_tokens),
        )
        reasoning_export_path = str(export_path.relative_to(project_root))
    integrity = _integrity(
        rows,
        eligible_ids,
        mode,
        cascade_stop_on_flip=bool(
            config.get("prompt", {}).get("cascade_stop_on_flip", True)
        ),
    )
    integrity["baseline_complete"] = bool(baseline_status["complete"])
    integrity["formal_complete"] = bool(
        integrity["formal_complete"] and baseline_status["complete"]
    )
    cascade_stop_on_flip = bool(
        config.get("prompt", {}).get("cascade_stop_on_flip", True)
    )
    metrics = (
        independent_summary(rows)
        if mode == "independent"
        else cascade_summary(rows, len(eligible_ids))
    )
    if mode == "cascade" and not cascade_stop_on_flip:
        metrics["warning"] = (
            "This focused Cascade scores all seven stages after a flip. Stage values "
            "have accumulated-history effects but no early-stop survivor bias."
        )
    summary = {
        "model_key": model_key,
        "display_name": config["models"][model_key]["display_name"],
        "mode": mode,
        "reasoning_mode": reasoning_mode,
        "baseline": baseline_status,
        "integrity": integrity,
        "metrics": metrics,
        "cascade_stop_on_flip": cascade_stop_on_flip if mode == "cascade" else None,
        "thinking": thinking,
        "reasoning_analysis": reasoning_summary,
        "thinking_analysis": reasoning_summary if thinking else None,
        "cot_analysis": (
            reasoning_summary["persuasion"]
            if reasoning_mode in COT_MODES and reasoning_summary
            else None
        ),
        "reasoning_analysis_path": reasoning_export_path,
    }
    output = result_dir(
        config,
        mode,
        model_key,
        project_root,
        reasoning_mode=reasoning_mode,
    ) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def render_report(summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# MMFakeBench Modality-Specific Persuasion",
        "",
        "Primary results use independent forks over the filtered 4,126-row six-model "
        "Non-thinking intersection. Thinking models are screened on this pool and only their "
        "own baseline-correct samples continue immediately to persuasion.",
        "",
        "## Core cross-modal whitewashing metrics",
        "",
        "| Model | Protocol | Response | Eligible | FR Text-error ← Image | FR Image-error ← Text | Text-error same Text | Image-error same Image | Complete |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary["metrics"].get("modality_specific") or summary["metrics"].get("modality_specific_stage_metrics_exploratory", {})
        core_text = metrics.get("FR_TextError_ImagePersuasion", {})
        core_image = metrics.get("FR_ImageError_TextPersuasion", {})
        same = metrics.get("same_modality_controls", {})
        lines.append(
            f"| {summary['display_name']} | {summary['mode']} | "
            f"{summary.get('reasoning_mode', DIRECT_REASONING_MODE)} | "
            f"{summary['integrity']['eligible_samples']} | "
            f"{_fmt(core_text.get('reversal_rate'))} | {_fmt(core_image.get('reversal_rate'))} | "
            f"{_fmt(same.get('FR_TextError_TextPersuasion', {}).get('reversal_rate'))} | "
            f"{_fmt(same.get('FR_ImageError_ImagePersuasion', {}).get('reversal_rate'))} | "
            f"{summary['integrity']['formal_complete']} |"
        )
    lines.extend([
        "",
        "`FR_TextError_ImagePersuasion` measures whether a correct image can whitewash an erroneous text.",
        "`FR_ImageError_TextPersuasion` measures whether a correct text can whitewash an erroneous image.",
        "All cascade modality-specific values are exploratory because later stages condition on survival and prior persuasion history.",
        "",
        "## Reasoning trace diagnostics",
        "",
        "| Model | Protocol | Response | Baseline traces | Persuasion traces | Reasoning present | Mean output tokens | Token-limit hits | Mean latency |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for summary in summaries:
        analysis = summary.get("reasoning_analysis") or summary.get("thinking_analysis")
        if not analysis:
            continue
        baseline = analysis["baseline"]["overall"]
        persuasion = analysis["persuasion"]["overall"]
        lines.append(
            f"| {summary['display_name']} | {summary['mode']} | "
            f"{summary.get('reasoning_mode', DIRECT_REASONING_MODE)} | "
            f"{baseline['rows']} | "
            f"{persuasion['rows']} | {_fmt(persuasion['reasoning_present_rate'])} | "
            f"{persuasion['mean_output_tokens'] if persuasion['mean_output_tokens'] is not None else 'N/A'} | "
            f"{persuasion['token_limit_hits']} | "
            f"{persuasion['mean_latency_seconds'] if persuasion['mean_latency_seconds'] is not None else 'N/A'} |"
        )
    lines.extend([
        "",
        "Reasoning diagnostics compare trace length and latency for flipped versus resisted "
        "responses and flag max-token truncation. Saved reasoning is not treated as a faithful "
        "causal explanation of the model's decision.",
        "",
    ])
    return "\n".join(lines)


def aggregate_all(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    modes: Sequence[str],
    *,
    model_keys: Sequence[str] | None = None,
    reasoning_mode: str = DIRECT_REASONING_MODE,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    selected_models = (
        list(model_keys)
        if model_keys is not None
        else active_models(config, reasoning_mode)
    )
    summaries = [
        aggregate_model(
            prompts,
            config,
            model_key,
            mode,
            reasoning_mode=reasoning_mode,
            project_root=project_root,
        )
        for mode in modes
        for model_key in selected_models
    ]
    if reasoning_mode == DIRECT_REASONING_MODE:
        report = _resolve(config["reporting"]["report"], project_root)
    else:
        report_root = _resolve(
            config.get("cot", {}).get("report_root", "reports/mmfakebench_cot_4126"),
            project_root,
        )
        report = report_root / f"{reasoning_mode}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(summaries), encoding="utf-8")
    return summaries
