#!/usr/bin/env python3
# === FILE META START ===
# File: /run_mmfakebench_formal.py
# Category: Main experiment runner
# Author: project team
# Last Updated: 2026-10-13
# Purpose: Run the full MMFakeBench original/textual/visual formal evaluation.
# Input: configs/mmfakebench_formal.yaml, MMFakeBench annotations/images, models.
# Output: Portable 7,700-row manifest, per-task predictions, JSON/Markdown reports.
# Run Example:
# python run_mmfakebench_formal.py prepare --source-root /path/to/MMFakeBench
# python run_mmfakebench_formal.py validate --source-root /path/to/MMFakeBench
# CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_formal.py run --model qwen3-vl-8b
# CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_formal.py run --model qwen3-vl-8b --task modality
# python run_mmfakebench_formal.py aggregate --task all
# === FILE META END ===
"""Command-line entry point for the two-task MMFakeBench formal evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prototype_induction.io import read_jsonl  # noqa: E402
from prototype_induction.mmfakebench_baseline import (  # noqa: E402
    active_multimodal_models,
    build_backend,
    prepare_error_retries,
    resolve_project_path,
)
from prototype_induction.mmfakebench_formal import (  # noqa: E402
    TASKS,
    aggregate_task,
    build_formal_dataset,
    load_formal_config,
    prediction_path,
    reparse_prediction_file,
    run_model,
    upgrade_prediction_file,
    validate_dataset,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/mmfakebench_formal.yaml"


def _load_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve_project_path(config["dataset"]["output"], PROJECT_ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"formal dataset not found: {path}; run `prepare` first")
    return list(read_jsonl(path))


def _print_findings(findings: list[dict[str, Any]]) -> bool:
    passed = True
    for item in findings:
        state = "PASS" if item["passed"] else "FAIL"
        print(f"[{state}] {item['check']}: {item['detail']}")
        passed = passed and bool(item["passed"])
    return passed


def _tasks(value: str) -> tuple[str, ...]:
    return TASKS if value in {"all", "both"} else (value,)


def command_prepare(args: argparse.Namespace, config: dict[str, Any]) -> int:
    rows, audit = build_formal_dataset(
        config,
        source_root=args.source_root,
        project_root=PROJECT_ROOT,
    )
    print(f"wrote {len(rows)} formal samples to {config['dataset']['output']}")
    print(json.dumps({
        "label_counts": audit["label_counts"],
        "category_counts": audit["category_counts"],
        "subcategory_counts": audit["subcategory_counts"],
        "split_counts": audit["split_counts"],
        "dataset_sha256": audit["dataset_sha256"],
    }, indent=2, ensure_ascii=False))
    findings = validate_dataset(
        rows,
        config,
        source_root=args.source_root,
        project_root=PROJECT_ROOT,
        check_images=not args.skip_images,
    )
    return 0 if _print_findings(findings) else 1


def command_validate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    rows = _load_rows(config)
    findings = validate_dataset(
        rows,
        config,
        source_root=args.source_root,
        project_root=PROJECT_ROOT,
        check_images=not args.skip_images,
    )
    for task in TASKS:
        for model_key in active_multimodal_models(config):
            path = prediction_path(config, task, model_key, PROJECT_ROOT)
            upgraded = upgrade_prediction_file(path, task)
            reparsed = reparse_prediction_file(path, task)
            predictions = list(read_jsonl(path)) if path.is_file() else []
            errors = sum(bool(str(row.get("error") or "").strip()) for row in predictions)
            unparseable = sum(
                row.get("prediction") == "Unparseable"
                and not str(row.get("error") or "").strip()
                for row in predictions
            )
            details = []
            if upgraded:
                details.append(f"correctness_recomputed={upgraded}")
            if reparsed:
                details.append(f"reparsed={reparsed}")
            suffix = f"; {'; '.join(details)}" if details else ""
            print(
                f"[INFO] {task}/{model_key}: {len(predictions)}/{len(rows)}; "
                f"system_errors={errors}; semantic_unparseable={unparseable}; "
                f"at {path}{suffix}"
            )
    return 0 if _print_findings(findings) else 1


def command_run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.model not in config["models"]:
        choices = ", ".join(config["models"])
        raise ValueError(f"unknown model {args.model!r}; choices: {choices}")
    rows = _load_rows(config)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
        print("WARNING: --limit is a smoke test and not a complete formal run")
    backend = build_backend(
        config,
        args.model,
        project_root=PROJECT_ROOT,
        model_path_override=args.model_path,
    )
    exit_code = 0
    for task in _tasks(args.task):
        output = prediction_path(config, task, args.model, PROJECT_ROOT)
        if args.retry_errors:
            count, archive = prepare_error_retries(output)
            print(f"[{task}] prepared {count} system-error retries; archive={archive}")
        status = run_model(
            rows,
            config,
            args.model,
            task,
            backend,
            source_root=args.source_root,
            project_root=PROJECT_ROOT,
        )
        print(json.dumps({"task": task, **status}, indent=2, ensure_ascii=False))
        if status["errors"]:
            exit_code = 1
    return exit_code


def command_aggregate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    rows = _load_rows(config)
    exit_code = 0
    for task in _tasks(args.task):
        summary = aggregate_task(rows, config, task, project_root=PROJECT_ROOT)
        print(json.dumps({
            "task": task,
            "complete_models": summary["complete_models"],
            "dataset_count": summary["dataset_count"],
            "integrity_passed": summary["integrity_passed"],
            "report": summary["report_path"],
        }, indent=2, ensure_ascii=False))
        if not summary["integrity_passed"]:
            exit_code = 1
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MMFakeBench formal 7,700-sample binary/modality evaluation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build the complete 7,700-row manifest")
    prepare.add_argument("--source-root", type=Path, help="MMFakeBench root override")
    prepare.add_argument("--skip-images", action="store_true", help="skip image readability checks")

    validate = subparsers.add_parser("validate", help="validate dataset and show both task statuses")
    validate.add_argument("--source-root", type=Path, help="MMFakeBench root override")
    validate.add_argument("--skip-images", action="store_true", help="skip image readability checks")

    run = subparsers.add_parser("run", help="run one model; binary is the default task")
    run.add_argument("--model", required=True)
    run.add_argument("--task", choices=(*TASKS, "both"), default="binary")
    run.add_argument("--model-path", type=Path, help="local model directory override")
    run.add_argument("--source-root", type=Path, help="MMFakeBench root override")
    run.add_argument("--limit", type=int, help="smoke-test only; run first N samples")
    run.add_argument("--retry-errors", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="aggregate one or both formal tasks")
    aggregate.add_argument("--task", choices=(*TASKS, "all"), default="all")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_formal_config(args.config)
        if args.command == "prepare":
            return command_prepare(args, config)
        if args.command == "validate":
            return command_validate(args, config)
        if args.command == "run":
            return command_run(args, config)
        if args.command == "aggregate":
            return command_aggregate(args, config)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
