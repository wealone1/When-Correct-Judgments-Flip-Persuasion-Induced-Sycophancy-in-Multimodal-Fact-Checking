# === FILE META START ===
# File: /src/prototype_induction/mmfakebench_formal.py
# Category: Main experiment pipeline
# Author: project team
# Last Updated: 2026-09-07
# Purpose: Build and evaluate the full MMFakeBench original/textual/visual set.
# Input: Formal YAML config, official MMFakeBench annotations/images, model outputs.
# Output: Portable dataset manifest, crash-safe predictions, task reports.
# === FILE META END ===
"""Formal MMFakeBench evaluation for binary detection and modality localization."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from .io import append_jsonl, read_jsonl, write_jsonl_atomic
from .mmfakebench_baseline import (
    BaselineBackend,
    Generation,
    load_label_candidates,
    resolve_project_path,
    split_reasoning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("binary", "modality")
TARGET_CATEGORIES = (
    "original",
    "textual_veracity_distortion",
    "visual_veracity_distortion",
)
CATEGORY_TO_BINARY = {
    "original": "Real",
    "textual_veracity_distortion": "Fake",
    "visual_veracity_distortion": "Fake",
}
CATEGORY_TO_MODALITY = {
    "original": "ORIGINAL",
    "textual_veracity_distortion": "TEXT REFUTES",
    "visual_veracity_distortion": "IMAGE REFUTES",
}


def load_formal_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    required = {"dataset", "prompts", "inference", "reporting", "models"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {', '.join(missing)}")
    configured_tasks = set(config["prompts"])
    if configured_tasks != set(TASKS):
        raise ValueError(f"prompts must contain exactly: {', '.join(TASKS)}")
    for task in TASKS:
        template = str(config["prompts"][task].get("template", ""))
        if template.count("[News caption]") != 1:
            raise ValueError(f"{task} prompt must contain [News caption] exactly once")
    return config


def _count(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "unknown")) for row in rows).items()))


def build_formal_dataset(
    config: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select every original/textual/visual row; no random sampling is performed."""
    spec = config["dataset"]
    root = Path(source_root or spec["source_root"]).expanduser().resolve()
    splits = list(spec.get("splits", ["val", "test"]))
    rows = load_label_candidates(root, "Real", splits) + load_label_candidates(
        root, "Fake", splits
    )
    selected = [row for row in rows if row["fake_cls"] in TARGET_CATEGORIES]
    selected.sort(key=lambda row: str(row["sample_id"]))
    for row in selected:
        row["formal_selection"] = "all_original_textual_visual"
        row["target_binary"] = CATEGORY_TO_BINARY[str(row["fake_cls"])]
        row["target_modality"] = CATEGORY_TO_MODALITY[str(row["fake_cls"])]
        # The annotation-relative path and split make the manifest portable.
        # source_image_path is retained only as a convenient local fallback.
        row["image_path"] = row["source_image_path"]
    output = resolve_project_path(spec["output"], project_root)
    write_jsonl_atomic(output, selected)
    audit = {
        "selection": "all rows in the requested categories; no sampling",
        "source_root_at_prepare_time": str(root),
        "splits": splits,
        "target_categories": list(TARGET_CATEGORIES),
        "sample_count": len(selected),
        "label_counts": _count(selected, "gt"),
        "category_counts": _count(selected, "fake_cls"),
        "subcategory_counts": _count(selected, "fake_subcategory"),
        "split_counts": _count(selected, "split"),
        "dataset_path": str(output),
        "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected, audit


def resolve_formal_image(
    row: Mapping[str, Any],
    *,
    source_root: str | Path | None,
    configured_source_root: str | Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a manifest image portably on local machines and GPU servers."""
    if source_root is not None:
        root = Path(source_root).expanduser()
        return root / f"MMFakeBench_{row['split']}" / str(
            row["source_annotation_image_path"]
        ).lstrip("/")
    configured = resolve_project_path(configured_source_root, project_root)
    candidate = configured / f"MMFakeBench_{row['split']}" / str(
        row["source_annotation_image_path"]
    ).lstrip("/")
    if candidate.is_file():
        return candidate
    for field in ("image_path", "source_image_path"):
        value = str(row.get(field) or "").strip()
        if value:
            path = Path(value).expanduser()
            path = path if path.is_absolute() else project_root / path
            if path.is_file():
                return path
    return candidate


def validate_dataset(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
    check_images: bool = True,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        findings.append({"check": check, "passed": passed, "detail": detail})

    spec = config["dataset"]
    expected_n = int(spec["expected_samples"])
    ids = [str(row.get("sample_id") or "") for row in rows]
    add("expected_sample_count", len(rows) == expected_n, f"found={len(rows)}, expected={expected_n}")
    add("unique_sample_ids", len(ids) == len(set(ids)) and all(ids), f"unique={len(set(ids))}, total={len(ids)}")
    actual_categories = _count(rows, "fake_cls")
    expected_categories = {
        str(key): int(value) for key, value in spec["expected_category_counts"].items()
    }
    add("expected_category_counts", actual_categories == expected_categories, f"found={actual_categories}, expected={expected_categories}")
    labels = _count(rows, "gt")
    expected_labels = {str(key): int(value) for key, value in spec["expected_label_counts"].items()}
    add("expected_label_counts", labels == expected_labels, f"found={labels}, expected={expected_labels}")
    unwanted = [str(row.get("sample_id")) for row in rows if row.get("fake_cls") not in TARGET_CATEGORIES]
    add("no_mismatch_or_unknown_categories", not unwanted, f"violations={len(unwanted)}")
    target_errors = [
        str(row.get("sample_id"))
        for row in rows
        if row.get("target_binary") != CATEGORY_TO_BINARY.get(str(row.get("fake_cls")))
        or row.get("target_modality") != CATEGORY_TO_MODALITY.get(str(row.get("fake_cls")))
    ]
    add("target_labels_match_fake_cls", not target_errors, f"violations={len(target_errors)}")
    empty_text = [str(row.get("sample_id")) for row in rows if not str(row.get("text") or "").strip()]
    add("all_text_present", not empty_text, f"violations={len(empty_text)}")
    if check_images:
        unreadable: list[str] = []
        for row in rows:
            path = resolve_formal_image(
                row,
                source_root=source_root,
                configured_source_root=spec["source_root"],
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


def render_prompt(template: str, news_text: str) -> str:
    if template.count("[News caption]") != 1:
        raise ValueError("prompt must contain [News caption] exactly once")
    return template.replace("[News caption]", news_text)


def _unique_finish_class(final_answer: str, classes: Sequence[str]) -> str | None:
    alternatives = "|".join(re.escape(value) for value in classes)
    matches = re.findall(
        rf"Finish\s*\[\s*({alternatives})\s*\.?\s*\]",
        final_answer,
        flags=re.IGNORECASE,
    )
    normalized = {re.sub(r"\s+", " ", item.upper()) for item in matches}
    return next(iter(normalized)) if len(normalized) == 1 else None


_OUTPUT_CONTROL_TOKEN_RE = re.compile(
    r"(?:"
    r"<turn\|>|"
    r"<\|(?:im_end|endoftext|eot_id|end|eom)\|>|"
    r"<end_of_turn>|"
    r"</s>|"
    r"<eos>"
    r")",
    flags=re.IGNORECASE,
)


def strip_output_control_tokens(final_answer: str) -> str:
    """Remove known generation-boundary tokens without relaxing label syntax.

    Some Hugging Face processors retain model-specific end-of-turn markers when
    decoding with ``skip_special_tokens=False``. Keeping those markers in the raw
    response is useful for reasoning audits, but they must not turn an otherwise
    exact ``Real``/``Fake`` verdict into a semantic parse failure.
    """
    return _OUTPUT_CONTROL_TOKEN_RE.sub("", final_answer).strip()


def parse_task_prediction(final_answer: str, task: str) -> tuple[str, str | None]:
    """Return normalized binary prediction plus the task-specific output class."""
    final_answer = strip_output_control_tokens(final_answer)
    if task == "binary":
        found = _unique_finish_class(final_answer, ("REAL", "FAKE"))
        if found:
            return found.capitalize(), found
        exact = [
            match.group(1).upper()
            for line in final_answer.splitlines()
            if (match := re.fullmatch(r"\s*(REAL|FAKE)\s*[.!]?\s*", line, re.IGNORECASE))
        ]
        return (exact[-1].capitalize(), exact[-1]) if exact and len(set(exact)) == 1 else ("Unparseable", None)
    if task == "modality":
        found = _unique_finish_class(
            final_answer, ("TEXT REFUTES", "IMAGE REFUTES", "ORIGINAL")
        )
        if found:
            return ("Real" if found == "ORIGINAL" else "Fake"), found
        return "Unparseable", None
    raise ValueError(f"unknown task: {task}")


def prediction_path(
    config: Mapping[str, Any],
    task: str,
    model_key: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    output_name = str(config["models"][model_key]["output_name"])
    root = resolve_project_path(config["reporting"]["results_root"], project_root)
    return root / task / output_name / "predictions.jsonl"


def reparse_prediction_file(path: str | Path, task: str) -> int:
    """Recover clear stored final labels without rerunning the model.

    Prefer the already separated ``final_answer``. Fall back to raw output only
    when it either has a completed reasoning boundary or contains no reasoning
    channel at all; an unfinished reasoning trace is never treated as a verdict.
    """
    target = Path(path)
    if not target.is_file():
        return 0
    rows = list(read_jsonl(target))
    changed = 0
    for row in rows:
        if row.get("prediction") != "Unparseable" or row.get("error"):
            continue
        raw = str(row.get("raw_response") or "")
        stored_final = str(row.get("final_answer") or "").strip()
        reasoning = row.get("reasoning")
        final = stored_final
        if not final:
            parsed_reasoning, parsed_final = split_reasoning(raw)
            if parsed_reasoning is not None and not parsed_final:
                continue
            reasoning, final = parsed_reasoning, parsed_final
        prediction, output_class = parse_task_prediction(final, task)
        if prediction == "Unparseable":
            continue
        target_class = str(row.get("target_class"))
        task_correct = (
            prediction == row.get("gt")
            if task == "binary"
            else output_class == target_class
        )
        binary_correct = prediction == row.get("gt")
        row.update({
            "prediction": prediction,
            "prediction_class": output_class,
            "official_prediction_class": output_class if task == "modality" else None,
            "reasoning": reasoning,
            "final_answer": final,
            "parseable": True,
            "is_correct": task_correct,
            "task_correct": task_correct,
            "binary_correct": binary_correct,
            "correctness": "Correct" if task_correct else "Incorrect",
            "reparsed_from_raw": True,
            "reparsed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        changed += 1
    if changed:
        write_jsonl_atomic(target, rows)
    return changed


def upgrade_prediction_file(path: str | Path, task: str) -> int:
    """Recompute correctness fields from immutable prediction and GT values.

    Binary output classes are uppercase (``REAL``/``FAKE``), while the portable
    manifest intentionally stores normalized labels as ``Real``/``Fake``. Older
    formal results compared those strings case-sensitively. This upgrade repairs
    only derived fields and never changes raw model output or parsed prediction.
    """
    target = Path(path)
    if not target.is_file():
        return 0
    rows = list(read_jsonl(target))
    changed = 0
    for row in rows:
        prediction = str(row.get("prediction") or "Unparseable")
        output_class = row.get("prediction_class")
        expected_class = (
            str(row.get("gt"))
            if task == "binary"
            else CATEGORY_TO_MODALITY.get(str(row.get("fake_cls")))
        )
        task_correct = (
            prediction == expected_class
            if task == "binary"
            else output_class == expected_class
        )
        binary_correct = prediction in {"Fake", "Real"} and prediction == row.get("gt")
        correctness = (
            "Unparseable"
            if prediction == "Unparseable"
            else ("Correct" if task_correct else "Incorrect")
        )
        updates = {
            "is_correct": task_correct,
            "task_correct": task_correct,
            "binary_correct": binary_correct,
            "correctness": correctness,
        }
        if any(row.get(key) != value for key, value in updates.items()):
            row.update(updates)
            row["correctness_recomputed_at_utc"] = datetime.now(timezone.utc).isoformat()
            changed += 1
    if changed:
        write_jsonl_atomic(target, rows)
    return changed


def run_model(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    task: str,
    backend: BaselineBackend,
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    """Run one task/model pair once per sample, with append-only crash recovery."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    model = config["models"][model_key]
    if not model.get("enabled") or not model.get("native_multimodal"):
        raise ValueError(f"model is disabled or not native multimodal: {model_key}")
    output = prediction_path(config, task, model_key, project_root)
    reparse_prediction_file(output, task)
    existing = list(read_jsonl(output)) if output.is_file() else []
    existing_ids = [str(row.get("sample_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError(f"duplicate sample_id already exists in {output}")
    completed = set(existing_ids)
    seed = int(config["dataset"]["seed"])
    prompt_spec = config["prompts"][task]
    template = str(prompt_spec["template"])
    prompt_hash = hashlib.sha256(template.encode()).hexdigest()
    remaining = sum(str(row["sample_id"]) not in completed for row in rows)
    written = errors = 0
    log_every = max(1, int(config["inference"].get("log_every", 10)))
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        image_path = resolve_formal_image(
            row,
            source_root=source_root,
            configured_source_root=config["dataset"]["source_root"],
            project_root=project_root,
        )
        prompt = render_prompt(template, str(row["text"]))
        fatal_oom = False
        try:
            generation = backend.generate(image_path, prompt, seed)
            prediction, output_class = parse_task_prediction(generation.final_answer, task)
            error = None
        except Exception as exc:
            generation = Generation("", None, "", 0, 0, 0.0, {})
            prediction, output_class = "Unparseable", None
            error = f"{type(exc).__name__}: {exc}"
            fatal_oom = type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)
            errors += 1
        target_class = str(row[f"target_{task}"])
        task_correct = (
            prediction == row["gt"]
            if task == "binary"
            else output_class == target_class
        )
        binary_correct = prediction in {"Fake", "Real"} and prediction == row["gt"]
        record = {
            "sample_id": sample_id,
            "task": task,
            "model_key": model_key,
            "model": model["hub_id"],
            "gt": row["gt"],
            "fake_cls": row["fake_cls"],
            "fake_subcategory": row["fake_subcategory"],
            "split": row["split"],
            "target_class": target_class,
            "prediction": prediction,
            "prediction_class": output_class,
            "official_prediction_class": output_class if task == "modality" else None,
            "is_correct": task_correct,
            "task_correct": task_correct,
            "binary_correct": binary_correct,
            "correctness": "Unparseable" if prediction == "Unparseable" else ("Correct" if task_correct else "Incorrect"),
            "parseable": prediction != "Unparseable",
            "raw_response": generation.raw_response,
            "reasoning": generation.reasoning,
            "final_answer": generation.final_answer,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "latency_seconds": generation.latency_seconds,
            "seed": seed,
            "do_sample": False,
            "temperature": 0.0,
            "repetition": 1,
            "prompt_name": prompt_spec["name"],
            "prompt_sha256": prompt_hash,
            "backend_metadata": backend.metadata,
            "generation_metadata": generation.metadata,
            "error": error,
        }
        append_jsonl(output, record)
        completed.add(sample_id)
        written += 1
        if written == 1 or written % log_every == 0 or written == remaining:
            print(f"[{task}/{model_key}] {written}/{remaining}: {sample_id} -> {output_class or prediction}")
        if fatal_oom:
            raise RuntimeError(
                f"CUDA OOM recorded for {sample_id}; rerun with --retry-errors after freeing memory"
            )
    return {"existing": len(existing), "written": written, "errors": errors, "total": len(completed)}


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(rows: Sequence[Mapping[str, Any]], task: str) -> dict[str, Any]:
    expected_classes = (
        ("REAL", "FAKE") if task == "binary" else ("ORIGINAL", "TEXT REFUTES", "IMAGE REFUTES")
    )
    parsed = [row for row in rows if row.get("prediction_class") in expected_classes]
    correct = sum(bool(row.get("task_correct")) for row in rows)
    binary_correct = sum(bool(row.get("binary_correct")) for row in rows)
    class_accuracy: dict[str, Any] = {}
    matrix: dict[str, dict[str, int]] = {}
    for target in expected_classes:
        subset = [row for row in rows if row.get("target_class") == target]
        class_correct = sum(row.get("prediction_class") == target for row in subset)
        class_accuracy[target] = {
            "count": len(subset),
            "correct": class_correct,
            "accuracy": _safe_div(class_correct, len(subset)),
        }
        matrix[target] = dict(sorted(Counter(
            str(row.get("prediction_class") or "Unparseable") for row in subset
        ).items()))
    available = [value["accuracy"] for value in class_accuracy.values() if value["accuracy"] is not None]
    return {
        "total": len(rows),
        "parseable": len(parsed),
        "unparseable": len(rows) - len(parsed),
        "system_errors": sum(bool(str(row.get("error") or "").strip()) for row in rows),
        "correct": correct,
        "task_accuracy_parseable": _safe_div(correct, len(parsed)),
        "task_accuracy_overall": _safe_div(correct, len(rows)),
        "macro_class_accuracy": sum(available) / len(available) if available else None,
        "binary_correct": binary_correct,
        "binary_accuracy_overall": _safe_div(binary_correct, len(rows)),
        "class_accuracy": class_accuracy,
        "confusion_matrix": matrix,
    }


def _group_metrics(rows: Sequence[Mapping[str, Any]], task: str, field: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return {key: _metrics(grouped[key], task) for key in sorted(grouped)}


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def _render_report(summary: Mapping[str, Any], config: Mapping[str, Any], task: str) -> str:
    task_name = "Binary Real/Fake" if task == "binary" else "Modality localization"
    lines = [
        f"# MMFakeBench Formal Evaluation: {task_name}",
        "",
        f"- Dataset: **{summary['dataset_count']}** samples",
        "- Included: `original`, `textual_veracity_distortion`, `visual_veracity_distortion`",
        "- Excluded: `mismatch`",
        "- Sampling: none; all val and test rows in the included categories",
        f"- Prompt source: {config['prompts'][task]['source']}",
        "",
        "## Results",
        "",
        "| Model | Complete | Total | Parseable | Correct | Overall accuracy | Parseable accuracy | Macro class accuracy | Binary accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, item in summary["models"].items():
        metrics = item["metrics"]
        lines.append(
            f"| {config['models'][model_key]['display_name']} | {item['complete']} | "
            f"{metrics['total']} | {metrics['parseable']} | {metrics['correct']} | "
            f"{_fmt(metrics['task_accuracy_overall'])} | {_fmt(metrics['task_accuracy_parseable'])} | "
            f"{_fmt(metrics['macro_class_accuracy'])} | {_fmt(metrics['binary_accuracy_overall'])} |"
        )
    lines.extend([
        "",
        "## Prompt",
        "",
        "```text",
        str(config["prompts"][task]["template"]),
        "```",
        "",
        "The modality task follows MMFakeBench's official text-first, image-second decision order, "
        "but removes MISMATCH because that category is intentionally absent from this formal set.",
        "",
    ])
    return "\n".join(lines)


def aggregate_task(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    task: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    expected_ids = {str(row["sample_id"]) for row in samples}
    models: dict[str, Any] = {}
    complete_models: list[str] = []
    checks: list[dict[str, Any]] = []
    prompt_hash = hashlib.sha256(str(config["prompts"][task]["template"]).encode()).hexdigest()
    for model_key, model in config["models"].items():
        if not model.get("enabled") or not model.get("native_multimodal"):
            continue
        path = prediction_path(config, task, model_key, project_root)
        upgrade_prediction_file(path, task)
        reparse_prediction_file(path, task)
        rows = list(read_jsonl(path)) if path.is_file() else []
        ids = [str(row.get("sample_id")) for row in rows]
        complete = (
            len(rows) == len(samples)
            and set(ids) == expected_ids
            and len(ids) == len(set(ids))
            and all(row.get("prompt_sha256") == prompt_hash for row in rows)
            and not any(str(row.get("error") or "").strip() for row in rows)
        )
        if complete:
            complete_models.append(model_key)
        checks.append({
            "check": f"{task}/{model_key}:complete",
            "passed": complete,
            "detail": f"found={len(rows)}, expected={len(samples)}, unique={len(set(ids))}",
        })
        models[model_key] = {
            "complete": complete,
            "prediction_path": str(path.relative_to(project_root)),
            "metrics": _metrics(rows, task),
            "by_fake_cls": _group_metrics(rows, task, "fake_cls"),
            "by_subcategory": _group_metrics(rows, task, "fake_subcategory"),
            "by_split": _group_metrics(rows, task, "split"),
        }
    report_root = resolve_project_path(config["reporting"]["report_root"], project_root) / task
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "summary.json"
    report_path = report_root / "report.md"
    summary: dict[str, Any] = {
        "task": task,
        "dataset_count": len(samples),
        "dataset_sha256": hashlib.sha256(
            resolve_project_path(config["dataset"]["output"], project_root).read_bytes()
        ).hexdigest(),
        "complete_models": complete_models,
        "models": models,
        "integrity_checks": checks,
        "integrity_passed": all(item["passed"] for item in checks),
        "summary_path": str(summary_path.relative_to(project_root)),
        "report_path": str(report_path.relative_to(project_root)),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary, config, task), encoding="utf-8")
    return summary
