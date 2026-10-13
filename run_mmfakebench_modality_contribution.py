#!/usr/bin/env python3
# === FILE META START ===
# File: /run_mmfakebench_modality_contribution.py
# Category: Main experiment runner
# Author: project team
# Last Updated: 2026-10-13
# Purpose: Run and aggregate MMFakeBench image/text persuasion contribution scores.
# Input: Frozen 300-row analysis set, grounded persuasion prompts, one non-thinking VLM.
# Output: Matched ablation margins, CTT metrics, model summaries, and one report.
# Run Example:
# python run_mmfakebench_modality_contribution.py validate --skip-images
# CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_modality_contribution.py run \
#   --model qwen3-vl-8b --source-root /path/to/MMFakeBench
# python run_mmfakebench_modality_contribution.py aggregate
# === FILE META END ===
"""Command-line entry point for MMFakeBench modality contribution analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prototype_induction.mmfakebench_modality_contribution import (  # noqa: E402
    EVIDENCE_CONDITIONS,
    FOLLOWUP_CONDITIONS,
    active_models,
    aggregate_all,
    build_scoring_backend,
    load_config,
    load_prompts,
    prepare_error_retries,
    run_model,
    scores_path,
    validate_experiment,
    validate_model,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/mmfakebench_modality_contribution.yaml"


def _print_findings(findings: list[dict[str, Any]]) -> bool:
    passed = True
    for finding in findings:
        state = "PASS" if finding["passed"] else "FAIL"
        print(f"[{state}] {finding['check']}: {finding['detail']}")
        passed = passed and bool(finding["passed"])
    return passed


def command_validate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    prompts = load_prompts(config, PROJECT_ROOT)
    findings = validate_experiment(
        prompts,
        config,
        source_root=args.source_root,
        project_root=PROJECT_ROOT,
        check_images=not args.skip_images,
    )
    expected = len(prompts) * len(EVIDENCE_CONDITIONS) * len(FOLLOWUP_CONDITIONS)
    for model_key in active_models(config):
        path = scores_path(config, model_key, PROJECT_ROOT)
        rows = errors = 0
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rows += 1
                errors += bool(json.loads(line).get("error"))
        print(f"[INFO] {model_key}: scores={rows}/{expected}; system_errors={errors}")
    return 0 if _print_findings(findings) else 1


def command_run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    validate_model(config, args.model)
    prompts = load_prompts(config, PROJECT_ROOT)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        prompts = prompts[: args.limit]
        print("WARNING: --limit is a smoke test; rerun without it for the formal experiment")
    path = scores_path(config, args.model, PROJECT_ROOT)
    if args.retry_errors:
        count, archive = prepare_error_retries(path)
        print(f"prepared {count} retries from {path.name}; archive={archive}")
    backend = build_scoring_backend(
        config,
        args.model,
        project_root=PROJECT_ROOT,
        model_path_override=args.model_path,
    )
    status = run_model(
        prompts,
        config,
        args.model,
        backend,
        source_root=args.source_root,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["errors"] == 0 else 1


def command_aggregate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    prompts = load_prompts(config, PROJECT_ROOT)
    model_keys = [args.model] if args.model else None
    if args.model:
        validate_model(config, args.model)
    summaries = aggregate_all(
        prompts,
        config,
        model_keys=model_keys,
        project_root=PROJECT_ROOT,
    )
    compact = [
        {
            "model": summary["model_key"],
            "score_rows": summary["integrity"]["score_rows"],
            "derived_rows": summary["integrity"]["derived_rows"],
            "system_errors": summary["integrity"]["system_errors"],
            "complete": summary["integrity"]["formal_complete"],
        }
        for summary in summaries
    ]
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"report: {config['reporting']['report']}")
    return 0 if compact and all(item["complete"] for item in compact) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate prompts and show contribution-result status"
    )
    validate.add_argument("--source-root", type=Path, help="MMFakeBench root override")
    validate.add_argument("--skip-images", action="store_true")

    run = subparsers.add_parser("run", help="score one non-thinking model")
    run.add_argument("--model", required=True)
    run.add_argument("--model-path", type=Path, help="override local model directory")
    run.add_argument("--source-root", type=Path, help="MMFakeBench root override")
    run.add_argument("--limit", type=int, help="smoke-test only: first N samples")
    run.add_argument("--retry-errors", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="aggregate completed scores")
    aggregate.add_argument("--model", help="aggregate one model; default: all available")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config, PROJECT_ROOT)
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

