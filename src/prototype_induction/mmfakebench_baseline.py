# === FILE META START ===
# File: /src/prototype_induction/mmfakebench_baseline.py
# Category: Main experiment pipeline
# Author: project team
# Last Updated: 2026-09-24
# Purpose: Build and evaluate the reproducible MMFakeBench Fake-300 baseline.
# Input: Official MMFakeBench val/test annotations, images, config, model outputs.
# Output: Sample JSONL, prediction JSONL, correct subsets, audit report.
# === FILE META END ===
"""MMFakeBench baseline-only screening utilities.

This module deliberately stops at baseline screening. It contains no persuasion
prompt, attack generation, retry-to-success logic, or post-hoc resampling.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import yaml
from PIL import Image, ImageOps

from .io import append_jsonl, read_jsonl, write_jsonl_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CLASSES = {
    "TEXT REFUTES": "Fake",
    "IMAGE REFUTES": "Fake",
    "MISMATCH": "Fake",
    "ORIGINAL": "Real",
}
VALID_PREDICTIONS = {"Fake", "Real", "Unparseable"}


@dataclass(frozen=True)
class Generation:
    """One formal model generation and its runtime metadata."""

    raw_response: str
    reasoning: str | None
    final_answer: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    metadata: dict[str, Any]


class BaselineBackend(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def generate(self, image_path: Path, prompt: str, seed: int) -> Generation: ...


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML config and reject missing top-level sections."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    required = {"dataset", "prompt", "inference", "reporting", "models"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {', '.join(missing)}")
    template = str(config["prompt"].get("template", ""))
    if template.count("[News caption]") != 1:
        raise ValueError("official prompt must contain [News caption] exactly once")
    return config


def resolve_project_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


def _subcategory(image_path: str) -> tuple[str, str]:
    """Return normalized and source folder subcategory names.

    MMFakeBench exposes ``fake_cls`` as its category. Its next concrete official
    subdivision is the image parent folder. The split/count suffix is removed so
    val and test samples from the same source group share one stratum.
    """
    parts = Path(image_path.lstrip("/")).parts
    raw = parts[1] if len(parts) > 1 else "unknown"
    normalized = re.sub(r"_(?:val|test)_\d+$", "", raw, flags=re.IGNORECASE)
    return normalized, raw


def _source_image_path(source_root: Path, split: str, annotation_path: str) -> Path:
    return source_root / f"MMFakeBench_{split}" / annotation_path.lstrip("/")


def load_label_candidates(
    source_root: str | Path,
    target_label: str,
    splits: Sequence[str] = ("val", "test"),
) -> list[dict[str, Any]]:
    """Load one official GT class without any model-based filtering."""
    normalized_target = target_label.strip().casefold()
    if normalized_target not in {"fake", "real", "true"}:
        raise ValueError(f"unsupported MMFakeBench target label: {target_label}")
    expected_raw = "fake" if normalized_target == "fake" else "true"
    normalized_gt = "Fake" if expected_raw == "fake" else "Real"
    root = Path(source_root).expanduser().resolve()
    candidates: list[dict[str, Any]] = []
    for split in splits:
        annotation_path = root / f"MMFakeBench_{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"annotation file not found: {annotation_path}")
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"expected a JSON list: {annotation_path}")
        for source_index, row in enumerate(rows):
            raw_gt = str(row.get("gt_answers", "")).strip()
            if raw_gt.casefold() != expected_raw:
                continue
            annotation_image_path = str(row.get("image_path", ""))
            subcategory, raw_subcategory = _subcategory(annotation_image_path)
            category = str(row.get("fake_cls", "unknown")).strip() or "unknown"
            source_image = _source_image_path(root, split, annotation_image_path)
            candidates.append(
                {
                    "sample_id": f"mmfakebench:{split}:{source_index:05d}",
                    "split": split,
                    "source_index": source_index,
                    "text": str(row.get("text", "")),
                    "gt": normalized_gt,
                    "gt_raw": raw_gt,
                    "fake_category": category,
                    "fake_subcategory": subcategory,
                    "fake_subcategory_raw": raw_subcategory,
                    "fake_cls": category,
                    "text_source": row.get("text_source"),
                    "image_source": row.get("image_source"),
                    "source_annotation_image_path": annotation_image_path,
                    "source_image_path": str(source_image),
                }
            )
    return candidates


def load_fake_candidates(
    source_root: str | Path,
    splits: Sequence[str] = ("val", "test"),
) -> list[dict[str, Any]]:
    """Backward-compatible loader for the official GT=Fake candidate pool."""
    return load_label_candidates(source_root, "Fake", splits)


def _bounded_largest_remainder(
    sizes: Mapping[tuple[str, ...], int], target: int
) -> dict[tuple[str, ...], int]:
    """Allocate proportional integer quotas with deterministic redistribution."""
    total = sum(sizes.values())
    if target < 0 or target > total:
        raise ValueError(f"sample size {target} is outside [0, {total}]")
    if not sizes:
        return {}
    exact = {key: target * size / total for key, size in sizes.items()}
    quotas = {key: min(size, math.floor(exact[key])) for key, size in sizes.items()}
    remaining = target - sum(quotas.values())
    ordering = sorted(
        sizes,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    while remaining:
        progressed = False
        for key in ordering:
            if quotas[key] < sizes[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("unable to redistribute the remaining sample quota")
    return quotas


def stratified_sample(
    candidates: Sequence[dict[str, Any]],
    sample_size: int,
    seed: int,
    strata_fields: Sequence[str] = ("fake_category", "fake_subcategory"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Proportionally sample category/subcategory strata with a fixed seed."""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = tuple(str(row[field]) for field in strata_fields)
        grouped[key].append(row)
    sizes = {key: len(rows) for key, rows in grouped.items()}
    quotas = _bounded_largest_remainder(sizes, sample_size)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: item["sample_id"])
        chosen = rng.sample(rows, quotas[key])
        selected.extend(chosen)
    selected.sort(key=lambda item: item["sample_id"])
    audit = {
        "seed": seed,
        "sample_size": sample_size,
        "candidate_count": len(candidates),
        "strata_fields": list(strata_fields),
        "strata": [
            {
                **{field: value for field, value in zip(strata_fields, key)},
                "candidate_count": sizes[key],
                "sampled_count": quotas[key],
            }
            for key in sorted(grouped)
        ],
    }
    return selected, audit


def _safe_image_name(row: Mapping[str, Any]) -> str:
    suffix = Path(str(row["source_image_path"])).suffix.lower() or ".jpg"
    digest = hashlib.sha256(str(row["sample_id"]).encode()).hexdigest()[:12]
    sample = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(row["sample_id"]))
    return f"{sample}__{digest}{suffix}"


def materialize_sample_images(
    rows: Sequence[dict[str, Any]],
    image_output: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Copy sampled images into the project so server sync is self-contained."""
    output = Path(image_output)
    output.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        source = Path(str(row["source_image_path"]))
        target = output / _safe_image_name(row)
        if not target.exists():
            shutil.copy2(source, target)
        row["image_path"] = str(target.relative_to(project_root))
        materialized.append(row)
    return materialized


def count_by(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "unknown")) for row in rows).items()))


def build_sample_dataset(
    config: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build and save the exact Fake-only stratified sample declared in config."""
    dataset = config["dataset"]
    if dataset.get("class_sample_sizes"):
        return build_balanced_sample_dataset(
            config, source_root=source_root, project_root=project_root
        )
    root = Path(source_root or dataset["source_root"]).expanduser().resolve()
    candidates = load_fake_candidates(root, dataset.get("splits", ["val", "test"]))
    selected, audit = stratified_sample(
        candidates,
        int(dataset["sample_size"]),
        int(dataset["seed"]),
        tuple(dataset.get("stratify_by", ["fake_category", "fake_subcategory"])),
    )
    if bool(dataset.get("materialize_images", True)):
        image_output = resolve_project_path(dataset["image_output"], project_root)
        selected = materialize_sample_images(
            selected, image_output, project_root=project_root
        )
    else:
        for row in selected:
            row["image_path"] = row["source_image_path"]
    for row in selected:
        row["sampling_seed"] = int(dataset["seed"])
        row["sampling_strata"] = list(dataset["stratify_by"])
    output = resolve_project_path(dataset["output"], project_root)
    write_jsonl_atomic(output, selected)
    audit.update(
        {
            "source_root": str(root),
            "splits": list(dataset.get("splits", ["val", "test"])),
            "candidate_category_counts": count_by(candidates, "fake_category"),
            "sampled_category_counts": count_by(selected, "fake_category"),
            "sampled_subcategory_counts": count_by(selected, "fake_subcategory"),
            "dataset_output": str(output),
            "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
    audit_path = output.with_suffix(".sampling.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected, audit


def build_balanced_sample_dataset(
    config: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve Fake-300 and add a reproducible stratified GT=True/Real-300."""
    dataset = config["dataset"]
    class_sizes = {str(key): int(value) for key, value in dataset["class_sample_sizes"].items()}
    if set(class_sizes) != {"Fake", "Real"}:
        raise ValueError("class_sample_sizes must contain exactly Fake and Real")
    root = Path(source_root or dataset["source_root"]).expanduser().resolve()
    splits = list(dataset.get("splits", ["val", "test"]))
    seed = int(dataset["seed"])
    strata_fields = tuple(dataset.get("stratify_by", ["fake_category", "fake_subcategory"]))

    fake_candidates = load_label_candidates(root, "Fake", splits)
    real_candidates = load_label_candidates(root, "Real", splits)
    existing_fake_path = resolve_project_path(dataset["existing_fake_sample"], project_root)
    if not existing_fake_path.is_file():
        raise FileNotFoundError(
            f"preserved Fake sample not found: {existing_fake_path}; build Fake-300 first"
        )
    fake_rows = [dict(row) for row in read_jsonl(existing_fake_path)]
    if len(fake_rows) != class_sizes["Fake"]:
        raise ValueError(
            f"preserved Fake sample has {len(fake_rows)} rows; expected {class_sizes['Fake']}"
        )
    if any(row.get("gt") != "Fake" for row in fake_rows):
        raise ValueError("preserved Fake sample contains a non-Fake row")
    for row in fake_rows:
        row.setdefault("gt_raw", "Fake")

    real_rows, real_audit = stratified_sample(
        real_candidates,
        class_sizes["Real"],
        seed,
        strata_fields,
    )
    if bool(dataset.get("materialize_images", True)):
        real_output = resolve_project_path(dataset["real_image_output"], project_root)
        real_rows = materialize_sample_images(
            real_rows, real_output, project_root=project_root
        )
    else:
        for row in real_rows:
            row["image_path"] = row["source_image_path"]
    for row in real_rows:
        row["sampling_seed"] = seed
        row["sampling_strata"] = list(strata_fields)

    selected = fake_rows + real_rows
    output = resolve_project_path(dataset["output"], project_root)
    write_jsonl_atomic(output, selected)
    all_candidates = fake_candidates + real_candidates
    audit = {
        "seed": seed,
        "sample_size": len(selected),
        "class_sample_sizes": class_sizes,
        "candidate_count": len(all_candidates),
        "candidate_label_counts": count_by(all_candidates, "gt"),
        "sampled_label_counts": count_by(selected, "gt"),
        "candidate_category_counts": count_by(all_candidates, "fake_category"),
        "sampled_category_counts": count_by(selected, "fake_category"),
        "sampled_subcategory_counts": count_by(selected, "fake_subcategory"),
        "source_root": str(root),
        "splits": splits,
        "strata_fields": list(strata_fields),
        "strata": [dict(item, gt="Real") for item in real_audit["strata"]],
        "preserved_fake_sample": str(existing_fake_path),
        "dataset_output": str(output),
        "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    audit_path = output.with_suffix(".sampling.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected, audit


def resolve_sample_image(
    row: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> Path:
    image_path = Path(str(row.get("image_path", ""))).expanduser()
    return image_path if image_path.is_absolute() else project_root / image_path


def validate_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_n: int | None = None,
    expected_gt_counts: Mapping[str, int] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Run all sample-level integrity checks and return structured findings."""
    findings: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        findings.append({"check": name, "passed": passed, "detail": detail})

    ids = [str(row.get("sample_id", "")) for row in rows]
    add("expected_sample_count", expected_n is None or len(rows) == expected_n,
        f"found={len(rows)}, expected={expected_n}")
    add("unique_sample_ids", len(ids) == len(set(ids)) and all(ids),
        f"unique={len(set(ids))}, total={len(ids)}")
    invalid_gt = [
        row.get("sample_id") for row in rows if row.get("gt") not in {"Fake", "Real"}
    ]
    add("valid_ground_truth_labels", not invalid_gt, f"violations={len(invalid_gt)}")
    if expected_gt_counts is not None:
        actual_gt_counts = count_by(rows, "gt")
        expected = dict(sorted((str(key), int(value)) for key, value in expected_gt_counts.items()))
        add(
            "expected_ground_truth_balance",
            actual_gt_counts == expected,
            f"found={actual_gt_counts}, expected={expected}",
        )
    empty_text = [row.get("sample_id") for row in rows if not str(row.get("text", "")).strip()]
    add("all_text_present", not empty_text, f"violations={len(empty_text)}")
    unreadable: list[str] = []
    for row in rows:
        path = resolve_sample_image(row, project_root)
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            unreadable.append(str(row.get("sample_id")))
    add("all_images_readable", not unreadable, f"violations={len(unreadable)}")
    return findings


def render_official_prompt(template: str, news_text: str) -> str:
    """Insert only the caption into the official prompt's literal placeholder."""
    if template.count("[News caption]") != 1:
        raise ValueError("prompt template must contain [News caption] exactly once")
    return template.replace("[News caption]", news_text)


def split_reasoning(raw_response: str) -> tuple[str | None, str]:
    """Separate common Qwen/Gemma thought channels without changing content."""
    qwen = re.search(r"<think>(.*?)</think>(.*)$", raw_response, re.DOTALL | re.IGNORECASE)
    if qwen:
        return qwen.group(1).strip() or None, qwen.group(2).strip()
    if re.search(r"</think>", raw_response, re.IGNORECASE):
        reasoning_text, final_text = re.split(
            r"</think>", raw_response, maxsplit=1, flags=re.IGNORECASE
        )
        return reasoning_text.strip() or None, final_text.strip()
    if re.search(r"<think>", raw_response, re.IGNORECASE):
        unfinished = re.split(r"<think>", raw_response, maxsplit=1, flags=re.IGNORECASE)[1]
        return unfinished.strip() or None, ""
    gemma_modern = re.search(
        r"<\|channel\|>(?:analysis|thought)<\|message\|>(.*?)"
        r"(?:<\|end\|>.*?)?<\|channel\|>final<\|message\|>(.*)$",
        raw_response,
        re.DOTALL | re.IGNORECASE,
    )
    if gemma_modern:
        return gemma_modern.group(1).strip() or None, gemma_modern.group(2).strip()
    gemma = re.search(
        r"<\|channel>thought\s*(.*?)<channel\|>(.*)$",
        raw_response,
        re.DOTALL | re.IGNORECASE,
    )
    if gemma:
        return gemma.group(1).strip() or None, gemma.group(2).strip()
    if re.search(
        r"<\|channel(?:\|>)?(?:analysis|thought)", raw_response, re.IGNORECASE
    ):
        return raw_response.strip() or None, ""
    return None, raw_response.strip()


def parse_prediction(final_answer: str) -> tuple[str, str | None]:
    """Parse the final official class and normalize it to Fake/Real.

    The official MMFakeBench prompt emits four ``Finish[...]`` classes. Three
    classes are fake mechanisms; ``ORIGINAL`` is the real class. A direct final
    Fake/Real answer is accepted as a fallback. Ambiguous output is never fixed
    using ground truth.
    """
    official = re.findall(
        r"Finish\s*\[\s*(TEXT\s+REFUTES|IMAGE\s+REFUTES|MISMATCH|ORIGINAL)"
        r"\s*\.?\s*\]",
        final_answer,
        flags=re.IGNORECASE,
    )
    if official:
        normalized_classes = [re.sub(r"\s+", " ", item.upper()) for item in official]
        if len(set(normalized_classes)) == 1:
            official_class = normalized_classes[-1]
            return OFFICIAL_CLASSES[official_class], official_class
        return "Unparseable", None

    explicit = re.findall(
        r"(?:final\s+(?:answer|verdict)|answer|verdict|prediction|classification)"
        r"\s*[:=\-]\s*(fake|real)\b",
        final_answer,
        flags=re.IGNORECASE,
    )
    if explicit:
        labels = {item.casefold() for item in explicit}
        if len(labels) == 1:
            value = explicit[-1].capitalize()
            return value, None
        return "Unparseable", None

    exact_lines = [
        match.group(1).capitalize()
        for line in final_answer.splitlines()
        if (match := re.fullmatch(r"\s*(fake|real)\s*[.!]?\s*", line, re.IGNORECASE))
    ]
    if exact_lines and len(set(exact_lines)) == 1:
        return exact_lines[-1], None

    terms = {match.group(0).casefold() for match in re.finditer(r"\b(?:fake|real)\b", final_answer, re.IGNORECASE)}
    if len(terms) == 1:
        return next(iter(terms)).capitalize(), None
    return "Unparseable", None


class TransformersMultimodalBackend:
    """One loaded native multimodal Hugging Face model."""

    def __init__(
        self,
        model_path: str,
        *,
        model_class: str,
        template_kwargs: Mapping[str, Any],
        thinking: bool,
        dtype: str,
        device_map: str,
        attention: str,
        max_new_tokens: int,
        local_files_only: bool,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Install requirements-mmfakebench-baseline.txt in the GPU environment."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run inference on the GPU server")
        dtype_lookup = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_lookup:
            raise ValueError(f"unsupported dtype: {dtype}")
        load_kwargs: dict[str, Any] = {
            "dtype": dtype_lookup[dtype],
            "device_map": device_map,
            "local_files_only": local_files_only,
        }
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if attention != "auto":
            load_kwargs["attn_implementation"] = attention
        if model_class == "auto_multimodal":
            try:
                loader = transformers.AutoModelForMultimodalLM
            except AttributeError as exc:
                raise RuntimeError(
                    "This Transformers version lacks AutoModelForMultimodalLM."
                ) from exc
        elif model_class == "qwen3_vl":
            try:
                loader = transformers.Qwen3VLForConditionalGeneration
            except AttributeError as exc:
                raise RuntimeError(
                    "This Transformers version lacks Qwen3VLForConditionalGeneration."
                ) from exc
        else:
            raise ValueError(f"unsupported model_class: {model_class}")
        self.torch = torch
        self.model = loader.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            **({"trust_remote_code": True} if trust_remote_code else {}),
        )
        self.template_kwargs = dict(template_kwargs)
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        self._metadata = {
            "backend": "transformers",
            "model_path": str(Path(model_path).resolve()) if Path(model_path).exists() else model_path,
            "model_class": model_class,
            "trust_remote_code": trust_remote_code,
            "model_commit": getattr(self.model.config, "_commit_hash", None),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "dtype": dtype,
            "device_map": device_map,
            "attention": attention,
            "do_sample": False,
            "temperature": 0.0,
            "repetitions": 1,
            "thinking": thinking,
            "template_kwargs": dict(template_kwargs),
            "max_new_tokens": max_new_tokens,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def generate(self, image_path: Path, prompt: str, seed: int) -> Generation:
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.template_kwargs,
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated = output[0][inputs["input_ids"].shape[-1] :]
        raw = self.processor.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        reasoning, final = split_reasoning(raw) if self.thinking else (None, raw)
        return Generation(
            raw_response=raw,
            reasoning=reasoning,
            final_answer=final,
            input_tokens=input_tokens,
            output_tokens=int(generated.shape[-1]),
            latency_seconds=latency,
            metadata={},
        )

    def generate_messages(
        self, messages: Sequence[Mapping[str, Any]], seed: int
    ) -> Generation:
        """Generate from a complete multimodal conversation.

        Persuasion experiments use this method to reconstruct a model's own
        baseline answer and then append either one independent follow-up or a
        cumulative cascade. Image inputs use the project-level ``image_path``
        representation so no open PIL handle escapes this method.
        """
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        converted: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = []
            for item in message["content"]:
                if item["type"] == "image_path":
                    with Image.open(str(item["path"])) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB").copy()
                    content.append({"type": "image", "image": image})
                elif item["type"] == "text":
                    content.append({"type": "text", "text": str(item["text"])})
                else:
                    raise ValueError(f"unsupported message content type: {item['type']}")
            converted.append({"role": str(message["role"]), "content": content})
        inputs = self.processor.apply_chat_template(
            converted,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.template_kwargs,
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated = output[0][inputs["input_ids"].shape[-1] :]
        raw = self.processor.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        reasoning, final = split_reasoning(raw) if self.thinking else (None, raw)
        return Generation(
            raw_response=raw,
            reasoning=reasoning,
            final_answer=final,
            input_tokens=input_tokens,
            output_tokens=int(generated.shape[-1]),
            latency_seconds=latency,
            metadata={},
        )


def active_multimodal_models(config: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key, spec in config["models"].items()
        if bool(spec.get("enabled")) and bool(spec.get("native_multimodal"))
    ]


def prediction_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    output_name = config["models"][model_key]["output_name"]
    return project_root / "results" / "baseline" / output_name / "predictions.jsonl"


def upgrade_prediction_file(path: str | Path) -> int:
    """Add derived correctness fields to older Fake-only prediction records."""
    target = Path(path)
    if not target.exists():
        return 0
    records = list(read_jsonl(target))
    changed = 0
    for item in records:
        if "is_correct" in item and "correctness" in item:
            continue
        is_correct = (
            item.get("prediction") in {"Fake", "Real"}
            and item.get("prediction") == item.get("gt")
        )
        item["is_correct"] = is_correct
        item["correctness"] = (
            "Unparseable"
            if item.get("prediction") == "Unparseable"
            else ("Correct" if is_correct else "Incorrect")
        )
        item["baseline_correct"] = is_correct
        item["baseline_correct_fake"] = (
            item.get("gt") == "Fake" and item.get("prediction") == "Fake"
        )
        item.setdefault("gt_raw", item.get("gt"))
        changed += 1
    if changed:
        write_jsonl_atomic(target, records)
    return changed


def reparse_prediction_file(path: str | Path) -> int:
    """Recover clear final verdicts from stored raw output without inference."""
    target = Path(path)
    if not target.exists():
        return 0
    records = list(read_jsonl(target))
    changed = 0
    for item in records:
        if item.get("prediction") != "Unparseable" or item.get("error"):
            continue
        raw = str(item.get("raw_response") or "")
        # Require an explicit end-of-thinking marker. This avoids treating a
        # class mentioned inside a max-token-truncated reasoning trace as final.
        if not re.search(r"</think>", raw, re.IGNORECASE):
            continue
        reasoning, final = split_reasoning(raw)
        prediction, official_class = parse_prediction(final)
        if prediction == "Unparseable":
            continue
        gt = str(item.get("gt"))
        is_correct = prediction == gt
        item.update(
            {
                "prediction": prediction,
                "official_prediction_class": official_class,
                "parseable": True,
                "reasoning": reasoning,
                "final_answer": final,
                "is_correct": is_correct,
                "correctness": "Correct" if is_correct else "Incorrect",
                "baseline_correct": is_correct,
                "baseline_correct_fake": gt == "Fake" and prediction == "Fake",
                "reparsed_from_raw": True,
                "reparsed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        changed += 1
    if changed:
        write_jsonl_atomic(target, records)
    return changed


def prepare_error_retries(path: str | Path) -> tuple[int, Path]:
    """Archive and remove only failed system attempts so they can run once more.

    Semantic Unparseable outputs without an ``error`` value remain untouched.
    The archived attempt keeps the original record for a complete audit trail.
    """
    target = Path(path)
    archive = target.parent / "failed_attempts.jsonl"
    if not target.exists():
        return 0, archive
    records = list(read_jsonl(target))
    failed = [row for row in records if str(row.get("error") or "").strip()]
    if not failed:
        return 0, archive
    retained = [row for row in records if not str(row.get("error") or "").strip()]
    archived_at = datetime.now(timezone.utc).isoformat()
    for row in failed:
        append_jsonl(
            archive,
            {
                **row,
                "archived_at_utc": archived_at,
                "archive_reason": "system_error_retry",
            },
        )
    write_jsonl_atomic(target, retained)
    return len(failed), archive


def build_backend(
    config: Mapping[str, Any],
    model_key: str,
    *,
    project_root: Path = PROJECT_ROOT,
    model_path_override: str | Path | None = None,
) -> TransformersMultimodalBackend:
    spec = config["models"][model_key]
    inference = config["inference"]
    configured = resolve_project_path(spec["local_path"], project_root)
    model_path = str(model_path_override or (configured if configured.exists() else spec["hub_id"]))
    return TransformersMultimodalBackend(
        model_path,
        model_class=str(spec.get("model_class", "auto_multimodal")),
        template_kwargs=spec.get("template_kwargs", {}),
        thinking=bool(spec.get("thinking")),
        dtype=str(inference.get("dtype", "auto")),
        device_map=str(inference.get("device_map", "auto")),
        attention=str(inference.get("attention", "auto")),
        max_new_tokens=int(
            spec.get("max_new_tokens", inference.get("max_new_tokens", 1024))
        ),
        local_files_only=bool(inference.get("local_files_only", True)),
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
    )


def run_model(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    backend: BaselineBackend,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    """Run exactly one formal generation per unfinished sample, crash-safely."""
    if model_key not in config["models"]:
        raise KeyError(f"unknown model: {model_key}")
    spec = config["models"][model_key]
    if not spec.get("native_multimodal"):
        raise ValueError(f"{model_key} is skipped_non_multimodal")
    if not spec.get("enabled"):
        raise ValueError(f"{model_key} is disabled in config")
    output = prediction_path(config, model_key, project_root)
    upgrade_prediction_file(output)
    reparse_prediction_file(output)
    existing = list(read_jsonl(output)) if output.exists() else []
    existing_ids = [str(item.get("sample_id")) for item in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError(f"duplicate sample_id already exists in {output}")
    completed = set(existing_ids)
    seed = int(config["dataset"]["seed"])
    prompt_template = str(config["prompt"]["template"])
    written = 0
    errors = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        image_path = resolve_sample_image(row, project_root)
        prompt = render_official_prompt(prompt_template, str(row["text"]))
        fatal_system_error = False
        try:
            generation = backend.generate(image_path, prompt, seed)
            prediction, official_class = parse_prediction(generation.final_answer)
            error = None
        except Exception as exc:  # one failed formal attempt is recorded, never retried
            generation = Generation("", None, "", 0, 0, 0.0, {})
            prediction, official_class = "Unparseable", None
            error = f"{type(exc).__name__}: {exc}"
            fatal_system_error = (
                type(exc).__name__ == "OutOfMemoryError"
                or "CUDA out of memory" in str(exc)
            )
            errors += 1
        gt = str(row["gt"])
        is_correct = prediction in {"Fake", "Real"} and prediction == gt
        record = {
            "sample_id": sample_id,
            "model_key": model_key,
            "model": spec["hub_id"],
            "gt": gt,
            "gt_raw": row.get("gt_raw", gt),
            "prediction": prediction,
            "official_prediction_class": official_class,
            "is_correct": is_correct,
            "correctness": (
                "Unparseable"
                if prediction == "Unparseable"
                else ("Correct" if is_correct else "Incorrect")
            ),
            "baseline_correct": is_correct,
            "baseline_correct_fake": gt == "Fake" and prediction == "Fake",
            "parseable": prediction != "Unparseable",
            "raw_response": generation.raw_response,
            "reasoning": generation.reasoning,
            "final_answer": generation.final_answer,
            "fake_category": row["fake_category"],
            "fake_subcategory": row["fake_subcategory"],
            "split": row["split"],
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "latency_seconds": generation.latency_seconds,
            "seed": seed,
            "do_sample": False,
            "temperature": 0.0,
            "repetition": 1,
            "prompt_name": config["prompt"]["name"],
            "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
            "backend_metadata": backend.metadata,
            "generation_metadata": generation.metadata,
            "error": error,
        }
        append_jsonl(output, record)
        completed.add(sample_id)
        written += 1
        print(f"[{model_key}] {written}/{len(rows) - len(existing)} {sample_id}: {prediction}")
        if fatal_system_error:
            raise RuntimeError(
                "CUDA OOM was recorded for "
                f"{sample_id}; stopping immediately to prevent cascading failed "
                "records. Free GPU memory, then rerun with --retry-errors."
            )
    return {"existing": len(existing), "written": written, "errors": errors, "total": len(completed)}


def _group_metrics(
    predictions: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get(field, "unknown"))].append(row)
    return {key: prediction_metrics(grouped[key]) for key in sorted(grouped)}


def prediction_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("prediction")) for row in predictions)
    parseable = counts["Fake"] + counts["Real"]
    correct = sum(
        1
        for row in predictions
        if row.get("prediction") in {"Fake", "Real"}
        and row.get("prediction") == row.get("gt")
    )
    gt_fake = [row for row in predictions if row.get("gt") == "Fake"]
    gt_real = [row for row in predictions if row.get("gt") == "Real"]
    correct_fake = sum(row.get("prediction") == "Fake" for row in gt_fake)
    correct_real = sum(row.get("prediction") == "Real" for row in gt_real)
    fake_accuracy = correct_fake / len(gt_fake) if gt_fake else None
    real_accuracy = correct_real / len(gt_real) if gt_real else None
    class_accuracies = [value for value in (fake_accuracy, real_accuracy) if value is not None]
    return {
        "total_samples": len(predictions),
        "total_sampled_fake": len(gt_fake),
        "total_sampled_real": len(gt_real),
        "parseable": parseable,
        "predicted_fake": counts["Fake"],
        "predicted_real": counts["Real"],
        "unparseable": counts["Unparseable"],
        "correct": correct,
        "incorrect_parseable": parseable - correct,
        "baseline_correct_fake": correct_fake,
        "baseline_correct_real": correct_real,
        "baseline_accuracy": correct / parseable if parseable else None,
        "overall_accuracy": correct / len(predictions) if predictions else None,
        "fake_accuracy": fake_accuracy,
        "real_accuracy": real_accuracy,
        "balanced_accuracy": (
            sum(class_accuracies) / len(class_accuracies) if class_accuracies else None
        ),
    }


def _prediction_integrity(
    sample_ids: set[str],
    expected_gt_by_id: Mapping[str, str],
    predictions: Sequence[Mapping[str, Any]],
    model_key: str,
    prompt_sha256: str,
) -> tuple[list[dict[str, Any]], bool]:
    ids = [str(row.get("sample_id", "")) for row in predictions]
    id_set = set(ids)
    checks = [
        {
            "check": f"{model_key}: exactly_N_results",
            "passed": len(predictions) == len(sample_ids),
            "detail": f"found={len(predictions)}, expected={len(sample_ids)}",
        },
        {
            "check": f"{model_key}: no_duplicate_results",
            "passed": len(ids) == len(id_set),
            "detail": f"unique={len(id_set)}, total={len(ids)}",
        },
        {
            "check": f"{model_key}: same_sample_set",
            "passed": id_set == sample_ids,
            "detail": f"missing={len(sample_ids-id_set)}, extra={len(id_set-sample_ids)}",
        },
        {
            "check": f"{model_key}: valid_predictions",
            "passed": all(row.get("prediction") in VALID_PREDICTIONS for row in predictions),
            "detail": "allowed=Fake,Real,Unparseable",
        },
        {
            "check": f"{model_key}: no_system_errors",
            "passed": all(not str(row.get("error") or "").strip() for row in predictions),
            "detail": "system failures must be rerun with --retry-errors",
        },
        {
            "check": f"{model_key}: result_ground_truth_matches_dataset",
            "passed": all(
                row.get("gt") == expected_gt_by_id.get(str(row.get("sample_id")))
                for row in predictions
            ),
            "detail": "expected each result gt to match the balanced dataset",
        },
        {
            "check": f"{model_key}: identical_official_prompt",
            "passed": all(row.get("prompt_sha256") == prompt_sha256 for row in predictions),
            "detail": f"expected_sha256={prompt_sha256}",
        },
        {
            "check": f"{model_key}: deterministic_single_run_config",
            "passed": all(
                row.get("do_sample") is False
                and float(row.get("temperature", -1)) == 0.0
                and int(row.get("repetition", -1)) == 1
                for row in predictions
            ),
            "detail": "expected do_sample=False, temperature=0, repetition=1",
        },
    ]
    return checks, all(item["passed"] for item in checks)


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    counts = Counter(str(row.get(field, "unknown")) for row in rows)
    total = len(rows)
    return {
        key: {"count": count, "percentage": (100.0 * count / total if total else 0.0)}
        for key, count in sorted(counts.items())
    }


def _subset_path(config: Mapping[str, Any], model_key: str, project_root: Path) -> Path:
    output_name = config["models"][model_key]["output_name"]
    return project_root / "subsets" / f"{output_name}_baseline_correct_fake.jsonl"


def aggregate_results(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create model subsets, shared intersections, audits, and the final report."""
    sample_ids = {str(row["sample_id"]) for row in samples}
    expected_gt_by_id = {str(row["sample_id"]): str(row["gt"]) for row in samples}
    sample_lookup = {str(row["sample_id"]): dict(row) for row in samples}
    expected_n = int(config["dataset"]["sample_size"])
    findings = validate_samples(
        samples,
        expected_n=expected_n,
        expected_gt_counts=config["dataset"].get("class_sample_sizes"),
        project_root=project_root,
    )
    per_model: dict[str, Any] = {}
    correct_ids: dict[str, set[str]] = {}
    successful: list[str] = []
    prompt_sha256 = hashlib.sha256(
        str(config["prompt"]["template"]).encode()
    ).hexdigest()
    for model_key in active_multimodal_models(config):
        path = prediction_path(config, model_key, project_root)
        reparse_prediction_file(path)
        predictions = list(read_jsonl(path)) if path.exists() else []
        checks, complete = _prediction_integrity(
            sample_ids, expected_gt_by_id, predictions, model_key, prompt_sha256
        )
        findings.extend(checks)
        per_model[model_key] = {
            "status": "successful" if complete else "incomplete",
            "prediction_path": str(path.relative_to(project_root)),
            "metrics": prediction_metrics(predictions),
            "by_category": _group_metrics(predictions, "fake_category"),
            "by_subcategory": _group_metrics(predictions, "fake_subcategory"),
        }
        if complete:
            successful.append(model_key)
            correct_ids[model_key] = {
                str(row["sample_id"])
                for row in predictions
                if row.get("gt") == "Fake" and row.get("prediction") == "Fake"
            }
            subset = [sample_lookup[sample_id] for sample_id in sorted(correct_ids[model_key])]
            write_jsonl_atomic(_subset_path(config, model_key, project_root), subset)

    intersections: dict[str, dict[str, int]] = {}
    for left in successful:
        intersections[left] = {}
        for right in successful:
            intersections[left][right] = len(correct_ids[left] & correct_ids[right])
    shared_ids = set.intersection(*(correct_ids[key] for key in successful)) if successful else set()
    shared = [sample_lookup[sample_id] for sample_id in sorted(shared_ids)]
    shared_path = project_root / "subsets/shared_correct_fake.jsonl"
    write_jsonl_atomic(shared_path, shared)
    shared_valid = all(
        sample_id in correct_ids[model_key]
        for sample_id in shared_ids
        for model_key in successful
    )
    findings.append(
        {
            "check": "shared_rows_correct_for_all_successful_models",
            "passed": shared_valid,
            "detail": f"shared={len(shared_ids)}, successful_models={len(successful)}",
        }
    )
    successful_id_sets = []
    for model_key in successful:
        model_predictions = list(
            read_jsonl(prediction_path(config, model_key, project_root))
        )
        successful_id_sets.append(
            {str(row["sample_id"]) for row in model_predictions}
        )
    same_successful_sets = (
        all(ids == successful_id_sets[0] for ids in successful_id_sets[1:])
        if successful_id_sets
        else True
    )
    findings.append(
        {
            "check": "successful_models_use_identical_sample_ids",
            "passed": same_successful_sets,
            "detail": f"successful_models={len(successful)}",
        }
    )
    fake_samples = [row for row in samples if row.get("gt") == "Fake"]
    original_distributions = {
        field: _distribution(fake_samples, field)
        for field in ("fake_category", "fake_subcategory")
    }
    shared_distributions = {
        field: _distribution(shared, field)
        for field in ("fake_category", "fake_subcategory")
    }
    threshold = float(config["reporting"].get("selection_bias_threshold_percentage_points", 10))
    shifts: dict[str, dict[str, float]] = {}
    max_shift = 0.0
    for field in original_distributions:
        keys = set(original_distributions[field]) | set(shared_distributions[field])
        shifts[field] = {}
        for key in sorted(keys):
            before = float(original_distributions[field].get(key, {}).get("percentage", 0.0))
            after = float(shared_distributions[field].get(key, {}).get("percentage", 0.0))
            shifts[field][key] = after - before
            max_shift = max(max_shift, abs(after - before))
    sampling_audit_path = resolve_project_path(
        config["dataset"]["output"], project_root
    ).with_suffix(".sampling.json")
    sampling_audit = (
        json.loads(sampling_audit_path.read_text(encoding="utf-8"))
        if sampling_audit_path.exists()
        else {}
    )
    summary = {
        "sample_count": len(samples),
        "sampled_fake_count": len(fake_samples),
        "sampled_real_count": len(samples) - len(fake_samples),
        "successful_models": successful,
        "per_model": per_model,
        "intersection_matrix": intersections,
        "shared_correct": len(shared),
        "shared_correct_percentage": (
            len(shared) / len(fake_samples) if fake_samples else None
        ),
        "original_distributions": original_distributions,
        "shared_distributions": shared_distributions,
        "distribution_shifts_percentage_points": shifts,
        "selection_bias_threshold_percentage_points": threshold,
        "selection_bias_flag": max_shift >= threshold,
        "maximum_absolute_shift_percentage_points": max_shift,
        "integrity_checks": findings,
        "integrity_passed": all(item["passed"] for item in findings),
        "sampling_audit": sampling_audit,
    }
    summary_path = project_root / "reports/baseline_screening.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(samples, config, summary, project_root=project_root)
    report_path = resolve_project_path(config["reporting"]["output"], project_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return summary


def _fmt_accuracy(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def _distribution_table(
    original: Mapping[str, Mapping[str, Any]],
    shared: Mapping[str, Mapping[str, Any]],
    shifts: Mapping[str, float],
) -> list[str]:
    lines = ["| Group | Original n (%) | Shared n (%) | Shift (pp) |", "|---|---:|---:|---:|"]
    for key in sorted(set(original) | set(shared)):
        before = original.get(key, {"count": 0, "percentage": 0.0})
        after = shared.get(key, {"count": 0, "percentage": 0.0})
        lines.append(
            f"| {key} | {before['count']} ({before['percentage']:.2f}%) | "
            f"{after['count']} ({after['percentage']:.2f}%) | {shifts.get(key, 0.0):+.2f} |"
        )
    return lines


def render_report(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Render the complete human-readable baseline screening report."""
    dataset = config["dataset"]
    lines = [
        "# MMFakeBench Baseline Screening Report",
        "",
        "This report covers baseline screening only. No persuasion prompt, attack text, "
        "recovery, voting, or model-performance-based resampling was used.",
        "",
        "## Dataset and sampling",
        "",
        f"- Source: `{summary.get('sampling_audit', {}).get('source_root', dataset['source_root'])}`",
        f"- Splits: `{', '.join(dataset.get('splits', []))}`",
        "- Candidate classes: official `GT=Fake` and `GT=True` "
        "(`True` is normalized to `gt=Real` to match prediction labels)",
        f"- Full candidate pool: **{summary.get('sampling_audit', {}).get('candidate_count', 'unknown')}**",
        f"- Balanced sample size: **{len(samples)}** "
        f"(Fake={summary.get('sampled_fake_count')}, Real={summary.get('sampled_real_count')})",
        f"- Seed: `{dataset['seed']}`",
        "- Strategy: preserve the already evaluated Fake-300 exactly; sample Real-300 "
        "with proportional bounded largest-remainder allocation over official source "
        "subcategories, followed by seeded sampling within each stratum.",
        "",
        "### Label counts",
        "",
        "| Normalized GT | Candidate | Sampled |",
        "|---|---:|---:|",
    ]
    candidate_label_counts = summary.get("sampling_audit", {}).get(
        "candidate_label_counts", {}
    )
    sampled_label_counts = count_by(samples, "gt")
    for key in sorted(set(candidate_label_counts) | set(sampled_label_counts)):
        lines.append(
            f"| {key} | {candidate_label_counts.get(key, 0)} | "
            f"{sampled_label_counts.get(key, 0)} |"
        )
    lines.extend([
        "",
        "### Category counts",
        "",
        "| Category | Candidate | Sampled |",
        "|---|---:|---:|",
    ])
    candidate_counts = summary.get("sampling_audit", {}).get(
        "candidate_category_counts", {}
    )
    sampled_counts = count_by(samples, "fake_category")
    for key in sorted(candidate_counts):
        lines.append(f"| {key} | {candidate_counts[key]} | {sampled_counts.get(key, 0)} |")
    lines.extend(["", "### Sampled subcategory counts", "", "| Subcategory | Sampled |", "|---|---:|"])
    for key, value in count_by(samples, "fake_subcategory").items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Official baseline prompt",
            "",
            f"Source: {config['prompt']['source']}",
            "",
            "Only `[News caption]` is replaced with the sample text; every model receives "
            "the same resulting prompt, with the image placed before the text.",
            "",
            "```text",
            str(config["prompt"]["template"]),
            "```",
            "",
            "The official classes are normalized as follows: `TEXT REFUTES`, "
            "`IMAGE REFUTES`, and `MISMATCH` → `Fake`; `ORIGINAL` → `Real`.",
            "",
            "## Model configuration",
            "",
            "All formal runs use `do_sample=False`, `temperature=0`, and `repetition=1`. "
            "Thinking remains enabled where configured, and raw reasoning is stored separately when parsable.",
            "",
            "| Model | Native multimodal | Enabled | Status | Thinking | Local path |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    successful = set(summary["successful_models"])
    for key, spec in config["models"].items():
        if key in successful:
            status = "successful"
        elif not spec.get("native_multimodal"):
            status = "skipped_non_multimodal"
        elif not spec.get("enabled"):
            status = str(spec.get("status", "disabled"))
        else:
            status = "incomplete"
        lines.append(
            f"| {spec['display_name']} | {bool(spec.get('native_multimodal'))} | "
            f"{bool(spec.get('enabled'))} | {status} | {bool(spec.get('thinking'))} | "
            f"`{spec['local_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Baseline results",
            "",
            "`Parseable accuracy = correct / parseable`; `overall accuracy` treats "
            "Unparseable as incorrect. Balanced accuracy is the mean of Fake and Real accuracy.",
            "",
            "| Model | Total | GT Fake | GT Real | Correct | Wrong | Unparseable | Parseable Acc. | Overall Acc. | Fake Acc. | Real Acc. | Balanced Acc. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, spec in config["models"].items():
        if not spec.get("native_multimodal"):
            lines.append(
                f"| {spec['display_name']} | — | — | — | — | — | — | — | — | — | — | skipped_non_multimodal |"
            )
            continue
        if key not in summary["per_model"]:
            continue
        metrics = summary["per_model"][key]["metrics"]
        lines.append(
            f"| {spec['display_name']} | {metrics['total_samples']} | "
            f"{metrics['total_sampled_fake']} | {metrics['total_sampled_real']} | "
            f"{metrics['correct']} | {metrics['incorrect_parseable']} | "
            f"{metrics['unparseable']} | {_fmt_accuracy(metrics['baseline_accuracy'])} | "
            f"{_fmt_accuracy(metrics['overall_accuracy'])} | "
            f"{_fmt_accuracy(metrics['fake_accuracy'])} | "
            f"{_fmt_accuracy(metrics['real_accuracy'])} | "
            f"{_fmt_accuracy(metrics['balanced_accuracy'])} |"
        )

    for model_key in summary["successful_models"]:
        display_name = config["models"][model_key]["display_name"]
        for field, heading in (
            ("by_category", "category"),
            ("by_subcategory", "subcategory"),
        ):
            lines.extend(
                [
                    "",
                    f"### {display_name}: by {heading}",
                    "",
                    "| Group | Total | GT Fake | GT Real | Correct | Wrong | Unparseable | Parseable Acc. | Overall Acc. |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for group, metrics in summary["per_model"][model_key][field].items():
                lines.append(
                    f"| {group} | {metrics['total_samples']} | "
                    f"{metrics['total_sampled_fake']} | {metrics['total_sampled_real']} | "
                    f"{metrics['correct']} | {metrics['incorrect_parseable']} | "
                    f"{metrics['unparseable']} | {_fmt_accuracy(metrics['baseline_accuracy'])} | "
                    f"{_fmt_accuracy(metrics['overall_accuracy'])} |"
                )

    lines.extend(["", "## Correct-set intersections", ""])
    if successful:
        lines.append(f"Shared-correct intersection: **N = {summary['shared_correct']}**")
        percentage = summary["shared_correct_percentage"]
        lines.append(
            f"Shared-correct percentage among the preserved Fake-300: "
            f"**{100 * percentage:.2f}%**"
        )
        lines.extend(["", "### Pairwise intersection matrix", ""])
        ordered = list(summary["successful_models"])
        lines.append("| Model | " + " | ".join(ordered) + " |")
        lines.append("|---|" + "---:|" * len(ordered))
        for left in ordered:
            values = [str(summary["intersection_matrix"][left][right]) for right in ordered]
            lines.append(f"| {left} | " + " | ".join(values) + " |")
    else:
        lines.append("No model currently has a complete, valid prediction file; shared set is empty.")

    for field, heading in (
        ("fake_category", "Category distribution"),
        ("fake_subcategory", "Subcategory distribution"),
    ):
        lines.extend(["", f"## {heading}: Fake-300 vs shared", ""])
        lines.extend(
            _distribution_table(
                summary["original_distributions"][field],
                summary["shared_distributions"][field],
                summary["distribution_shifts_percentage_points"][field],
            )
        )
    lines.extend(
        [
            "",
            "## Selection-bias audit",
            "",
            f"- Reporting threshold: `{summary['selection_bias_threshold_percentage_points']:.2f}` percentage points",
            f"- Maximum absolute distribution shift: `{summary['maximum_absolute_shift_percentage_points']:.2f}` percentage points",
            f"- Obvious selection-bias flag: **{summary['selection_bias_flag']}**",
            "",
            "This flag is descriptive only. The sample and shared subset are not changed in response.",
            "",
            "## Integrity checks",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
    )
    for item in summary["integrity_checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"| {item['check']} | {status} | {item['detail']} |")
    lines.extend(
        [
            "",
            f"Overall integrity: **{'PASS' if summary['integrity_passed'] else 'FAIL'}**",
            "",
            "## Output files",
            "",
            f"- Sample: `{dataset['output']}`",
            "- Per-model predictions: `results/baseline/<model_name>/predictions.jsonl`",
            "- Model-specific subsets: `subsets/<model_name>_baseline_correct_fake.jsonl`",
            "- Shared subset: `subsets/shared_correct_fake.jsonl`",
            "- Machine-readable report: `reports/baseline_screening.json`",
            "",
        ]
    )
    return "\n".join(lines)
