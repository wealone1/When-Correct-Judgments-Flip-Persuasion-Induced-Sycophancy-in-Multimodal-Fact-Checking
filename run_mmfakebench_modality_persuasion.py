#!/usr/bin/env python3
# === FILE META START ===
# File: /run_mmfakebench_modality_persuasion.py
# Category: Main experiment runner
# Author: project team
# Last Updated: 2026-10-13
# Purpose: Run direct or explicit-CoT seven-condition MMFakeBench persuasion.
# Input: Filtered 4,126-row pool, fixed prompts, images, and target models.
# Output: Independent/cascade predictions, FR metrics, and Thinking diagnostics.
# Run Example:
# python run_mmfakebench_modality_persuasion.py validate --source-root /path/to/MMFakeBench
# CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_modality_persuasion.py run --model qwen3-vl-8b --mode independent --source-root /path/to/MMFakeBench
# CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_modality_persuasion.py run --model qwen3-vl-8b --mode independent --reasoning-mode structured_multimodal_cot --source-root /path/to/MMFakeBench
# python run_mmfakebench_modality_persuasion.py aggregate --mode both
# === FILE META END ===
"""CLI for the formal MMFakeBench modality-specific persuasion experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prototype_induction.mmfakebench_baseline import build_backend  # noqa: E402
from prototype_induction.mmfakebench_modality_persuasion import (  # noqa: E402
    COT_MODES,
    DIRECT_REASONING_MODE,
    MODES,
    REASONING_MODES,
    active_models,
    aggregate_all,
    baseline_states,
    load_config,
    load_prompts,
    prediction_path,
    prepare_baseline_error_retries,
    prepare_error_retries,
    reparse_prediction_file,
    reasoning_token_limit,
    run_model,
    validate_reasoning_model,
    validate_prompts,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/mmfakebench_modality_persuasion.yaml"


def _print(findings: list[dict[str, Any]]) -> bool:
    passed = True
    for item in findings:
        state = "PASS" if item["passed"] else "FAIL"
        print(f"[{state}] {item['check']}: {item['detail']}")
        passed = passed and bool(item["passed"])
    return passed


def command_validate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    prompts = load_prompts(config, PROJECT_ROOT)
    findings = validate_prompts(
        prompts, config, source_root=args.source_root, project_root=PROJECT_ROOT,
        check_images=not args.skip_images,
    )
    for model_key in active_models(config, args.reasoning_mode):
        _, status = baseline_states(prompts, config, model_key, PROJECT_ROOT)
        print(f"[INFO] baseline/{model_key}: {json.dumps(status, ensure_ascii=False)}")
        for mode in MODES:
            path = prediction_path(
                config,
                mode,
                model_key,
                PROJECT_ROOT,
                reasoning_mode=args.reasoning_mode,
            )
            reparsed = reparse_prediction_file(path)
            count = sum(1 for line in path.open() if line.strip()) if path.is_file() else 0
            print(
                f"[INFO] {mode}/{model_key}: rows={count}; "
                f"reparsed_clear_verdicts={reparsed}; path={path}"
            )
    return 0 if _print(findings) else 1


def command_run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.model not in config["models"]:
        raise ValueError(f"unknown model {args.model!r}; choices: {', '.join(config['models'])}")
    validate_reasoning_model(config, args.model, args.reasoning_mode)
    prompts = load_prompts(config, PROJECT_ROOT)
    findings = validate_prompts(
        prompts, config, source_root=args.source_root, project_root=PROJECT_ROOT,
        check_images=False,
    )
    failed = [item for item in findings if not item["passed"]]
    if failed:
        raise RuntimeError("prompt validation failed: " + "; ".join(item["check"] for item in failed))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        prompts = prompts[: args.limit]
        print("WARNING: --limit is a smoke test; rerun without it for formal results")
    path = prediction_path(
        config,
        args.mode,
        args.model,
        PROJECT_ROOT,
        reasoning_mode=args.reasoning_mode,
    )
    if args.retry_errors:
        count, archive = prepare_error_retries(path)
        print(f"prepared {count} persuasion retries; archive={archive}")
        baseline_count, baseline_archive = prepare_baseline_error_retries(
            config, args.model, PROJECT_ROOT
        )
        print(
            f"prepared {baseline_count} baseline retries; "
            f"archive={baseline_archive}"
        )
    backend_config = copy.deepcopy(config["_formal_config"])
    if args.reasoning_mode in COT_MODES:
        backend_config["models"][args.model]["max_new_tokens"] = reasoning_token_limit(
            config, args.reasoning_mode, args.model
        )
    backend = build_backend(
        backend_config, args.model, project_root=PROJECT_ROOT,
        model_path_override=args.model_path,
    )
    status = run_model(
        prompts, config, args.model, args.mode, backend,
        reasoning_mode=args.reasoning_mode,
        source_root=args.source_root, project_root=PROJECT_ROOT,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["errors"] == 0 else 1


def command_aggregate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    prompts = load_prompts(config, PROJECT_ROOT)
    modes = MODES if args.mode == "both" else (args.mode,)
    if args.model is not None and args.model not in config["models"]:
        raise ValueError(
            f"unknown model {args.model!r}; choices: {', '.join(config['models'])}"
        )
    summaries = aggregate_all(
        prompts,
        config,
        modes,
        model_keys=[args.model] if args.model is not None else None,
        reasoning_mode=args.reasoning_mode,
        project_root=PROJECT_ROOT,
    )
    compact = [
        {
            "model": item["model_key"],
            "mode": item["mode"],
            "reasoning_mode": item["reasoning_mode"],
            "eligible": item["integrity"]["eligible_samples"],
            "rows": item["integrity"]["row_count"],
            "formal_complete": item["integrity"]["formal_complete"],
        }
        for item in summaries
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if args.reasoning_mode == DIRECT_REASONING_MODE:
        report = config["reporting"]["report"]
    else:
        report = str(
            Path(config["cot"]["report_root"]) / f"{args.reasoning_mode}.md"
        )
    print(f"report: {report}")
    return 0 if compact and all(item["formal_complete"] for item in compact) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--source-root", type=Path)
    validate.add_argument("--skip-images", action="store_true")
    validate.add_argument(
        "--reasoning-mode", choices=REASONING_MODES, default=DIRECT_REASONING_MODE
    )
    run = subparsers.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--mode", choices=MODES, default="independent")
    run.add_argument(
        "--reasoning-mode", choices=REASONING_MODES, default=DIRECT_REASONING_MODE
    )
    run.add_argument("--model-path", type=Path)
    run.add_argument("--source-root", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--retry-errors", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--mode", choices=(*MODES, "both"), default="both")
    aggregate.add_argument("--model", help="aggregate one model instead of all enabled models")
    aggregate.add_argument(
        "--reasoning-mode", choices=REASONING_MODES, default=DIRECT_REASONING_MODE
    )
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
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
