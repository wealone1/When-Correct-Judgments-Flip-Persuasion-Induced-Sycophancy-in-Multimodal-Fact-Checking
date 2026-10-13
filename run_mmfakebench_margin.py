#!/usr/bin/env python3
# === FILE META START ===
# File: /run_mmfakebench_margin.py
# Category: Main experiment runner
# Author: project team
# Last Updated: 2026-10-13
# Purpose: Run and aggregate teacher-forced MMFakeBench persuasion margins.
# Input: configs/mmfakebench_analysis_margin.yaml, frozen prompts, images, one non-thinking VLM.
# Output: Token decompositions, baseline/independent/cascade margin JSONL, summaries.
# Run Example:
# python run_mmfakebench_margin.py validate --skip-images
# CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_margin.py run --model qwen3-vl-8b --protocol both --source-root /path/to/MMFakeBench
# python run_mmfakebench_margin.py aggregate
# === FILE META END ===
"""Command-line entry point for MMFakeBench teacher-forced margin analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prototype_induction.mmfakebench_margin import (  # noqa: E402
    PROTOCOLS,
    active_models,
    aggregate_all,
    baseline_path,
    build_scoring_backend,
    load_config,
    load_prompts,
    prepare_error_retries,
    protocol_path,
    run_baselines,
    run_protocol,
    validate_experiment,
    validate_model,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/mmfakebench_margin.yaml"


def _print_findings(findings: list[dict[str, Any]]) -> bool:
    passed = True
    for finding in findings:
        state = "PASS" if finding["passed"] else "FAIL"
        print(f"[{state}] {finding['check']}: {finding['detail']}")
        passed = passed and bool(finding["passed"])
    return passed


def _protocols(value: str) -> tuple[str, ...]:
    return PROTOCOLS if value == "both" else (value,)


def command_validate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    prompts = load_prompts(config, PROJECT_ROOT)
    findings = validate_experiment(
        prompts,
        config,
        source_root=args.source_root,
        project_root=PROJECT_ROOT,
        check_images=not args.skip_images,
    )
    for model_key in active_models(config):
        baseline = baseline_path(config, model_key, PROJECT_ROOT)
        baseline_rows = sum(1 for row in baseline.open(encoding="utf-8") if row.strip()) if baseline.is_file() else 0
        details = [f"baseline={baseline_rows}/{len(prompts)}"]
        for protocol in PROTOCOLS:
            path = protocol_path(config, model_key, protocol, PROJECT_ROOT)
            rows = sum(1 for row in path.open(encoding="utf-8") if row.strip()) if path.is_file() else 0
            details.append(f"{protocol}={rows}/{len(prompts) * 7}")
        print(f"[INFO] {model_key}: {'; '.join(details)}")
    return 0 if _print_findings(findings) else 1


def command_run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    validate_model(config, args.model)
    prompts = load_prompts(config, PROJECT_ROOT)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        prompts = prompts[: args.limit]
        print("WARNING: --limit is a smoke test; rerun without it for the formal experiment")
    if args.retry_errors:
        targets = [baseline_path(config, args.model, PROJECT_ROOT)] + [
            protocol_path(config, args.model, protocol, PROJECT_ROOT)
            for protocol in _protocols(args.protocol)
        ]
        for path in targets:
            count, archive = prepare_error_retries(path)
            print(f"prepared {count} retries from {path.name}; archive={archive}")
    backend = build_scoring_backend(
        config,
        args.model,
        project_root=PROJECT_ROOT,
        model_path_override=args.model_path,
    )
    statuses: dict[str, Any] = {}
    statuses["baseline"] = run_baselines(
        prompts,
        config,
        args.model,
        backend,
        source_root=args.source_root,
        project_root=PROJECT_ROOT,
    )
    for protocol in _protocols(args.protocol):
        statuses[protocol] = run_protocol(
            prompts,
            config,
            args.model,
            protocol,
            backend,
            source_root=args.source_root,
            project_root=PROJECT_ROOT,
        )
    print(json.dumps(statuses, indent=2, ensure_ascii=False))
    return 0 if all(status["errors"] == 0 for status in statuses.values()) else 1


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
            "baseline_rows": summary["integrity"]["baseline_rows"],
            "independent_rows": summary["integrity"]["independent_rows"],
            "cascade_rows": summary["integrity"]["cascade_rows"],
            "cascade_ever_crossed": summary["cascade"]["ever_crossed_correct_to_wrong"],
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

    validate = subparsers.add_parser("validate", help="validate frozen prompts and show result status")
    validate.add_argument("--source-root", type=Path, help="MMFakeBench root override")
    validate.add_argument("--skip-images", action="store_true")

    run = subparsers.add_parser("run", help="score one non-thinking model")
    run.add_argument("--model", required=True)
    run.add_argument("--protocol", choices=(*PROTOCOLS, "both"), default="both")
    run.add_argument("--model-path", type=Path, help="override local model directory")
    run.add_argument("--source-root", type=Path, help="MMFakeBench root override")
    run.add_argument("--limit", type=int, help="smoke-test only: first N samples")
    run.add_argument("--retry-errors", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="aggregate completed margin files")
    aggregate.add_argument("--model", help="aggregate one model; default: all non-thinking models")
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
