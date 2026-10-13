# === FILE META START ===
# File: /src/prototype_induction/mmfakebench_margin.py
# Category: Main experiment pipeline
# Author: project team
# Last Updated: 2026-09-23
# Purpose: Measure teacher-forced Real/Fake correctness margins before and after persuasion.
# Input: Frozen 255-row diagnostic prompts, MMFakeBench images, non-thinking VLMs.
# Output: Baseline, independent, and full-cascade sequence-log-likelihood margins.
# === FILE META END ===
"""Teacher-forced sequence margins for the MMFakeBench persuasion experiment."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml
from PIL import Image, ImageOps

from .io import append_jsonl, read_jsonl, write_jsonl_atomic
from .mmfakebench_baseline import TransformersMultimodalBackend
from .mmfakebench_formal import load_formal_config, render_prompt, resolve_formal_image
from .mmfakebench_modality_persuasion import (
    CONDITIONS,
    CONDITION_LABELS,
    baseline_states,
    prediction_path as persuasion_prediction_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ("independent", "cascade")
LABELS = ("Real", "Fake")
SEQUENCE_ALIGNED_INPUT_KEYS = frozenset(
    {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
)


@dataclass(frozen=True)
class CandidateScore:
    """Teacher-forced score for every token in one complete candidate answer."""

    label: str
    text: str
    total_logprob: float
    token_ids: list[int]
    tokens: list[str]
    token_logprobs: list[float]
    excluded_eos_ids: list[int]


@dataclass(frozen=True)
class PairScore:
    """Scores for the two fixed binary answer sequences at one answer position."""

    candidates: dict[str, CandidateScore]
    input_tokens: int
    latency_seconds: float
    template_prefix_sha256: str
    template_alignment_strategy: str = "exact_generation_prefix"
    inserted_generation_control_token_ids: list[int] = field(default_factory=list)
    inserted_generation_control_tokens: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TemplateAlignment:
    """Safe alignment between generation-prefix and explicit-answer templates."""

    full_answer_start: int
    strategy: str
    inserted_token_ids: list[int]
    inserted_tokens: list[str]


class MarginBackend(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def score_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        candidates: Mapping[str, str],
    ) -> PairScore: ...


def _resolve(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_oom(exc: Exception) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).casefold()


def _user(text: str, image_path: Path | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append({"type": "image_path", "path": str(image_path)})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    return next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


def _template_alignment(
    prefix_ids: Sequence[int],
    full_ids: Sequence[int],
    candidate: str,
    tokenizer: Any,
) -> TemplateAlignment:
    """Locate the answer boundary without silently accepting template drift.

    Most processors preserve the complete `add_generation_prompt` prefix when
    an explicit assistant answer is rendered. Gemma 4 non-thinking is a known
    exception: the generation prefix ends in an empty thought-channel control
    block, while the explicit-answer template omits that block. In that one
    auditable case we splice the missing control block back before scoring.
    """
    prefix = [int(item) for item in prefix_ids]
    full = [int(item) for item in full_ids]
    if full[: len(prefix)] == prefix:
        return TemplateAlignment(
            full_answer_start=len(prefix),
            strategy="exact_generation_prefix",
            inserted_token_ids=[],
            inserted_tokens=[],
        )

    common = _common_prefix_length(prefix, full)
    inserted = prefix[common:]
    if common < 1 or not inserted or len(inserted) > 32:
        raise RuntimeError(
            "candidate chat template does not preserve the generation prefix; "
            f"first mismatch at token {common}"
        )
    inserted_text = str(tokenizer.decode(inserted, skip_special_tokens=False))
    compact_control = "".join(inserted_text.split())
    is_empty_thought_channel = (
        "<|channel>thought" in compact_control
        and "<channel|>" in compact_control
    )
    answer_text = str(
        tokenizer.decode(full[common:], skip_special_tokens=False)
    )
    if (
        not is_empty_thought_channel
        or "".join(candidate.split()) not in "".join(answer_text.split())
    ):
        raise RuntimeError(
            "candidate chat template does not preserve the generation prefix; "
            f"first mismatch at token {common}"
        )
    return TemplateAlignment(
        full_answer_start=common,
        strategy="insert_empty_thought_channel",
        inserted_token_ids=inserted,
        inserted_tokens=[
            str(item) for item in tokenizer.convert_ids_to_tokens(inserted)
        ],
    )


def _splice_generation_prefix(
    prefix: Any,
    full: Any,
    full_answer_start: int,
    torch_module: Any,
) -> Any:
    """Insert generation-only controls into explicit token-sequence inputs.

    Tensor shape alone cannot identify a sequence-aligned field. In particular,
    Gemma 4 images are processed at width 768, so a 768-token context can make
    ``pixel_values`` accidentally satisfy the same shape checks as ``input_ids``.
    Restricting the splice to known token-sequence keys prevents image tensors
    from being widened when the two unrelated dimensions happen to coincide.
    """
    prefix_length = int(prefix["input_ids"].shape[-1])
    full_length = int(full["input_ids"].shape[-1])
    for key, value in list(full.items()):
        if key not in SEQUENCE_ALIGNED_INPUT_KEYS:
            continue
        prefix_value = prefix.get(key)
        if (
            hasattr(value, "shape")
            and hasattr(prefix_value, "shape")
            and len(value.shape) >= 2
            and tuple(value.shape[:-1]) == tuple(prefix_value.shape[:-1])
            and int(value.shape[-1]) == full_length
            and int(prefix_value.shape[-1]) == prefix_length
        ):
            full[key] = torch_module.cat(
                [prefix_value, value[..., full_answer_start:]], dim=-1
            )
    return full


class TeacherForcedScoringBackend:
    """Score complete assistant continuations with a loaded native VLM.

    Candidate text is inserted through the same processor chat template as the
    context. This is safer than tokenizing labels in isolation because token
    boundaries can depend on the assistant prefix and preceding whitespace.
    """

    def __init__(self, backend: TransformersMultimodalBackend) -> None:
        if backend.thinking:
            raise ValueError("margin scoring v1 excludes native-Thinking models")
        self.backend = backend
        self.torch = backend.torch
        self.model = backend.model
        self.processor = backend.processor
        self.template_kwargs = dict(backend.template_kwargs)
        self.model_class = str(backend.metadata.get("model_class") or "")
        self._metadata = {
            **backend.metadata,
            "scoring": "teacher_forced_full_sequence_sum",
            "actual_generation": False,
            "eos_scored": False,
            "candidate_tokenization": "native_chat_template_contextual_suffix",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def _convert_messages(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
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
        return converted

    def _templated_inputs(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
        continue_final_message: bool = False,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": True,
            "return_tensors": "pt",
            **self.template_kwargs,
        }
        if continue_final_message:
            kwargs["continue_final_message"] = True
        return self.processor.apply_chat_template(messages, **kwargs).to(self.model.device)

    def _tokenizer(self) -> Any:
        tokenizer = getattr(self.processor, "tokenizer", None)
        return tokenizer if tokenizer is not None else self.processor

    def _forward(self, inputs: Mapping[str, Any]) -> Any:
        """Run one standard Transformers teacher-forced scoring forward."""
        return self.model(**inputs, use_cache=False)

    def score_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        candidates: Mapping[str, str],
    ) -> PairScore:
        if set(candidates) != set(LABELS):
            raise ValueError(f"candidates must contain exactly {LABELS}")
        torch = self.torch
        converted = self._convert_messages(messages)
        prefix = self._templated_inputs(converted, add_generation_prompt=True)
        prefix_ids = prefix["input_ids"][0].detach().cpu().tolist()
        prefix_length = len(prefix_ids)
        if prefix_length < 1:
            raise RuntimeError("chat template produced an empty assistant prefix")
        tokenizer = self._tokenizer()
        eos_ids = getattr(tokenizer, "eos_token_id", None)
        eos_set = (
            {int(item) for item in eos_ids}
            if isinstance(eos_ids, (list, tuple, set))
            else ({int(eos_ids)} if eos_ids is not None else set())
        )
        scored: dict[str, CandidateScore] = {}
        total_latency = 0.0
        shared_alignment: TemplateAlignment | None = None
        for label in LABELS:
            candidate = str(candidates[label])
            full_messages = [*converted, {
                "role": "assistant",
                "content": [{"type": "text", "text": candidate}],
            }]
            full = self._templated_inputs(
                full_messages,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            full_ids = full["input_ids"][0].detach().cpu().tolist()
            alignment = _template_alignment(
                prefix_ids,
                full_ids,
                candidate,
                tokenizer,
            )
            if shared_alignment is None:
                shared_alignment = alignment
            elif (
                alignment.full_answer_start != shared_alignment.full_answer_start
                or alignment.strategy != shared_alignment.strategy
                or alignment.inserted_token_ids
                != shared_alignment.inserted_token_ids
            ):
                raise RuntimeError(
                    "Real/Fake candidates do not share one template answer boundary"
                )
            if alignment.strategy == "insert_empty_thought_channel":
                full = _splice_generation_prefix(
                    prefix,
                    full,
                    alignment.full_answer_start,
                    torch,
                )
                full_ids = full["input_ids"][0].detach().cpu().tolist()
            if full_ids[:prefix_length] != prefix_ids:
                raise RuntimeError(
                    "internal template-alignment failure after control-token insertion"
                )
            suffix = [int(item) for item in full_ids[prefix_length:]]
            excluded_eos: list[int] = []
            while suffix and suffix[-1] in eos_set:
                excluded_eos.insert(0, suffix.pop())
            if not suffix:
                raise RuntimeError(f"candidate {candidate!r} produced no scoreable tokens")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self._forward(full)
                positions = output.logits[0, prefix_length - 1 : prefix_length + len(suffix) - 1]
                log_probs = torch.log_softmax(positions.float(), dim=-1)
                target = torch.tensor(suffix, device=log_probs.device, dtype=torch.long)
                token_logprobs_tensor = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
                token_logprobs = [float(value) for value in token_logprobs_tensor.detach().cpu()]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            total_latency += elapsed
            tokens = [str(item) for item in tokenizer.convert_ids_to_tokens(suffix)]
            scored[label] = CandidateScore(
                label=label,
                text=candidate,
                total_logprob=float(sum(token_logprobs)),
                token_ids=suffix,
                tokens=tokens,
                token_logprobs=token_logprobs,
                excluded_eos_ids=excluded_eos,
            )
            del output, positions, log_probs, token_logprobs_tensor, target, full
        return PairScore(
            candidates=scored,
            input_tokens=prefix_length,
            latency_seconds=total_latency,
            template_prefix_sha256=hashlib.sha256(
                json.dumps(prefix_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            template_alignment_strategy=(
                shared_alignment.strategy
                if shared_alignment is not None
                else "exact_generation_prefix"
            ),
            inserted_generation_control_token_ids=(
                shared_alignment.inserted_token_ids
                if shared_alignment is not None
                else []
            ),
            inserted_generation_control_tokens=(
                shared_alignment.inserted_tokens
                if shared_alignment is not None
                else []
            ),
        )


def build_scoring_backend(
    config: Mapping[str, Any],
    model_key: str,
    *,
    project_root: Path = PROJECT_ROOT,
    model_path_override: str | Path | None = None,
) -> TeacherForcedScoringBackend:
    """Load one formal model once and wrap it with teacher-forced scoring."""
    from .mmfakebench_baseline import build_backend

    backend = build_backend(
        config,
        model_key,
        project_root=project_root,
        model_path_override=model_path_override,
    )
    return TeacherForcedScoringBackend(backend)


def load_config(path: str | Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    required = {"dataset", "baseline", "scoring", "reporting"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {', '.join(missing)}")
    formal_path = _resolve(config["baseline"]["formal_config"], project_root)
    formal = load_formal_config(formal_path)
    config["_formal_config"] = formal
    config["_formal_config_path"] = str(formal_path)
    config["models"] = formal["models"]
    config["inference"] = formal["inference"]
    if tuple(config["scoring"].get("protocols", [])) != PROTOCOLS:
        raise ValueError(f"scoring.protocols must be exactly {PROTOCOLS}")
    candidates = config["scoring"].get("candidates", {})
    if set(candidates) != set(LABELS):
        raise ValueError(f"scoring.candidates must contain exactly {LABELS}")
    final_probe = str(config["scoring"].get("final_probe") or "").rstrip()
    if not final_probe:
        raise ValueError("scoring.final_probe must not be empty")
    baseline_template = str(formal["prompts"]["binary"]["template"]).rstrip()
    if not baseline_template.endswith(final_probe):
        raise ValueError(
            "formal binary baseline must end with scoring.final_probe exactly"
        )
    if str(candidates["Real"]) not in final_probe or str(candidates["Fake"]) not in final_probe:
        raise ValueError("final_probe must display both complete candidate strings")
    prompt_path = _resolve(config["dataset"]["prompts"], project_root)
    config["_prompts_sha256"] = (
        hashlib.sha256(prompt_path.read_bytes()).hexdigest() if prompt_path.is_file() else ""
    )
    return config


def active_models(config: Mapping[str, Any]) -> list[str]:
    """Return only enabled native multimodal Non-thinking models."""
    return [
        key
        for key, spec in config["models"].items()
        if bool(spec.get("enabled"))
        and bool(spec.get("native_multimodal"))
        and not bool(spec.get("thinking"))
    ]


def frozen_baseline_models(
    prompts: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return model keys embedded when the frozen cohort was constructed.

    Adding a held-out model to the central registry must not retroactively make
    the fixed seven-model selection manifest invalid.
    """
    return sorted(
        {
            str(model_key)
            for prompt in prompts
            for model_key in (prompt.get("baseline_assistant_answers") or {})
        }
    )


def validate_model(config: Mapping[str, Any], model_key: str) -> None:
    if model_key not in config["models"]:
        raise ValueError(
            f"unknown model {model_key!r}; choices: {', '.join(active_models(config))}"
        )
    spec = config["models"][model_key]
    if not spec.get("enabled") or not spec.get("native_multimodal"):
        raise ValueError(f"model {model_key!r} is not enabled for native multimodal inference")
    if spec.get("thinking"):
        raise ValueError(
            f"model {model_key!r} is a native-Thinking model; margin v1 is Non-thinking only"
        )


def load_prompts(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    path = _resolve(config["dataset"]["prompts"], project_root)
    if not path.is_file():
        raise FileNotFoundError(f"margin prompt dataset not found: {path}")
    return list(read_jsonl(path))


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
    add("expected_sample_count", len(prompts) == expected_n, f"found={len(prompts)}, expected={expected_n}")
    add("unique_sample_ids", len(ids) == len(set(ids)) and all(ids), f"unique={len(set(ids))}, total={len(ids)}")
    counts = dict(sorted(Counter(str(row.get("fake_cls")) for row in prompts).items()))
    expected_counts = {
        str(key): int(value)
        for key, value in config["dataset"]["expected_category_counts"].items()
    }
    add("expected_category_counts", counts == expected_counts, f"found={counts}, expected={expected_counts}")
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
        cell_counts = Counter(
            (str(row.get("fake_cls")), str(row.get("analysis_cohort")))
            for row in prompts
        )
        configured_cells = config["dataset"].get("expected_class_cohort_counts")
        if configured_cells:
            expected_cells = {
                (str(fake_cls), str(cohort)): int(count)
                for fake_cls, cohorts in configured_cells.items()
                for cohort, count in cohorts.items()
            }
            add(
                "expected_class_cohort_cells",
                dict(cell_counts) == expected_cells,
                f"found={dict(sorted(cell_counts.items()))}, expected={dict(sorted(expected_cells.items()))}",
            )
        else:
            expected_cell = int(config["dataset"].get("expected_class_cohort_count", 0))
            expected_cells = {
                (fake_cls, cohort): expected_cell
                for fake_cls in expected_counts
                for cohort in expected_cohorts
            }
            add(
                "balanced_class_cohort_cells",
                expected_cell > 0 and dict(cell_counts) == expected_cells,
                f"found={dict(sorted(cell_counts.items()))}, expected_each={expected_cell}",
            )
    mismatch = [row for row in prompts if str(row.get("fake_cls")) == "mismatch"]
    add("mismatch_excluded", not mismatch, f"violations={len(mismatch)}")
    missing = [
        f"{row.get('sample_id')}:{condition}"
        for row in prompts
        for condition in CONDITIONS
        if not str(row.get(condition) or "").strip()
        and not (
            condition == "credibility_appeal"
            and row.get("credibility_available") is False
        )
    ]
    add("all_available_persuasion_text_present", not missing, f"violations={len(missing)}")
    embedded_errors: list[str] = []
    for model_key in frozen_baseline_models(prompts):
        states, status = baseline_states(prompts, config, model_key, project_root)
        wrong = sum(not bool(state.get("is_correct")) for state in states.values())
        if len(states) != len(prompts) or wrong:
            embedded_errors.append(
                f"{model_key}:resolved={len(states)},wrong={wrong},missing={status['missing']}"
            )
    add(
        "nonthinking_baselines_resolved_and_correct",
        not embedded_errors,
        "all frozen cohort models" if not embedded_errors else "; ".join(embedded_errors),
    )
    if check_images:
        unreadable = 0
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
                unreadable += 1
        add("all_images_readable", unreadable == 0, f"violations={unreadable}")
    else:
        add("all_images_readable", True, "skipped")
    return findings


def model_result_dir(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    root = _resolve(config["reporting"]["results_root"], project_root)
    return root / str(config["models"][model_key]["output_name"])


def baseline_path(
    config: Mapping[str, Any], model_key: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return model_result_dir(config, model_key, project_root) / "baseline_margins.jsonl"


def protocol_path(
    config: Mapping[str, Any],
    model_key: str,
    protocol: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol: {protocol}")
    return model_result_dir(config, model_key, project_root) / f"{protocol}_margins.jsonl"


def _pair_payload(
    score: PairScore,
    gt: str,
    *,
    tie_tolerance: float,
    previous_label: str | None = None,
) -> dict[str, Any]:
    real = score.candidates["Real"]
    fake = score.candidates["Fake"]
    correct = score.candidates[gt]
    wrong_label = "Fake" if gt == "Real" else "Real"
    wrong = score.candidates[wrong_label]
    margin = correct.total_logprob - wrong.total_logprob
    label_gap = real.total_logprob - fake.total_logprob
    sequence_argmax = "Real" if label_gap > tie_tolerance else "Fake" if label_gap < -tie_tolerance else "Tie"
    selected = previous_label if sequence_argmax == "Tie" and previous_label in LABELS else sequence_argmax
    return {
        "candidate_scores": {
            label: asdict(score.candidates[label]) for label in LABELS
        },
        "correct_label": gt,
        "wrong_label": wrong_label,
        "correct_logprob": correct.total_logprob,
        "wrong_logprob": wrong.total_logprob,
        "margin": margin,
        "real_minus_fake_logprob": label_gap,
        "sequence_argmax_prediction": sequence_argmax,
        "selected_prediction": selected,
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


def _base_record(
    prompt: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    backend: MarginBackend,
) -> dict[str, Any]:
    return {
        "sample_id": str(prompt["sample_id"]),
        "model_key": model_key,
        "model": config["models"][model_key]["hub_id"],
        "display_name": config["models"][model_key]["display_name"],
        "gt": str(prompt["gt"]),
        "fake_cls": str(prompt["fake_cls"]),
        "fake_subcategory": prompt.get("fake_subcategory"),
        "split": prompt.get("split"),
        "analysis_cohort": prompt.get("analysis_cohort"),
        "selector_model": prompt.get("selector_model"),
        "matched_pair_id": prompt.get("matched_pair_id"),
        "selector_successful_conditions": prompt.get(
            "selector_successful_conditions"
        ),
        "selector_n_successful_conditions": prompt.get(
            "selector_n_successful_conditions"
        ),
        "score_method": "teacher_forced_complete_sequence_sum",
        "do_sample": False,
        "actual_generation": False,
        "eos_scored": False,
        "backend_metadata": backend.metadata,
        "dataset_sha256": config["_prompts_sha256"],
        "created_at_utc": _now(),
    }


def _baseline_messages(
    prompt: Mapping[str, Any], image_path: Path, baseline_template: str
) -> list[dict[str, Any]]:
    return [
        _user(
            render_prompt(baseline_template, str(prompt["original_news_text"])),
            image_path,
        )
    ]


def _followup(persuasion: str, final_probe: str) -> str:
    return f"{persuasion.strip()}\n\n{final_probe.strip()}"


def _conversation(
    prompt: Mapping[str, Any],
    image_path: Path,
    baseline_template: str,
    baseline_answer: str,
    prior_turns: Sequence[tuple[str, str]],
    current_followup: str,
) -> list[dict[str, Any]]:
    messages = _baseline_messages(prompt, image_path, baseline_template)
    messages.append(_assistant(baseline_answer))
    for question, answer in prior_turns:
        messages.extend([_user(question), _assistant(answer)])
    messages.append(_user(current_followup))
    return messages


def _read_unique(path: Path, fields: Sequence[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    if not path.is_file():
        return {}
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = tuple(str(row.get(field) or "") for field in fields)
        if key in indexed:
            raise ValueError(f"duplicate result key {key} in {path}")
        indexed[key] = row
    return indexed


def prepare_error_retries(path: str | Path) -> tuple[int, Path]:
    """Archive and remove system-error rows while preserving successful scores."""
    target = Path(path)
    archive = target.parent / f"{target.stem}.failed_attempts.jsonl"
    if not target.is_file():
        return 0, archive
    rows = list(read_jsonl(target))
    failed = [row for row in rows if str(row.get("error") or "").strip()]
    if not failed:
        return 0, archive
    for row in failed:
        append_jsonl(
            archive,
            {**row, "archived_at_utc": _now(), "archive_reason": "system_error_retry"},
        )
    write_jsonl_atomic(
        target, [row for row in rows if not str(row.get("error") or "").strip()]
    )
    return len(failed), archive


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
        "note": "Tokenization is contextual at the assistant answer position; every row also stores its actual token IDs.",
        "candidates": {
            label: {
                "text": pair.candidates[label].text,
                "token_ids": pair.candidates[label].token_ids,
                "tokens": pair.candidates[label].tokens,
                "excluded_eos_ids": pair.candidates[label].excluded_eos_ids,
            }
            for label in LABELS
        },
        "template_alignment_strategy": pair.template_alignment_strategy,
        "inserted_generation_control_token_ids": (
            pair.inserted_generation_control_token_ids
        ),
        "inserted_generation_control_tokens": (
            pair.inserted_generation_control_tokens
        ),
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


def _behavioral_results(
    config: Mapping[str, Any],
    model_key: str,
    protocol: str,
    project_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    by_protocol = config["reporting"].get("behavioral_persuasion_configs", {})
    configured = by_protocol.get(protocol) if isinstance(by_protocol, Mapping) else None
    persuasion_config_path = _resolve(
        configured
        or config["reporting"].get(
            "persuasion_config", "configs/mmfakebench_modality_persuasion.yaml"
        ),
        project_root,
    )
    if not persuasion_config_path.is_file():
        return {}
    from .mmfakebench_modality_persuasion import load_config as load_persuasion_config

    persuasion_config = load_persuasion_config(persuasion_config_path, project_root)
    path = persuasion_prediction_path(
        persuasion_config, protocol, model_key, project_root
    )
    return _read_unique(path, ("sample_id", "condition"))


def run_baselines(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    backend: MarginBackend,
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    validate_model(config, model_key)
    output = baseline_path(config, model_key, project_root)
    existing = _read_unique(output, ("sample_id",))
    baseline_template = str(config["_formal_config"]["prompts"]["binary"]["template"])
    candidates = {key: str(value) for key, value in config["scoring"]["candidates"].items()}
    tolerance = float(config["scoring"].get("tie_tolerance", 1e-8))
    configured_root = config["dataset"]["source_root"]
    log_every = max(1, int(config["inference"].get("log_every", 10)))
    written = errors = 0
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        if (sample_id,) in existing:
            continue
        image_path = resolve_formal_image(
            prompt,
            source_root=source_root,
            configured_source_root=configured_root,
            project_root=project_root,
        )
        messages = _baseline_messages(prompt, image_path, baseline_template)
        record = {
            **_base_record(prompt, config, model_key, backend),
            "run_id": f"baseline:{model_key}:{sample_id}",
            "phase": "baseline",
            "protocol": None,
            "context_sha256": _canonical_hash(messages),
        }
        fatal = False
        try:
            pair = backend.score_messages(messages, candidates)
            record.update(_pair_payload(pair, str(prompt["gt"]), tie_tolerance=tolerance))
            record["status"] = "completed"
            record["error"] = None
            _write_tokenization_manifest(config, model_key, pair, backend, project_root)
        except Exception as exc:
            record.update({"status": "system_error", "error": f"{type(exc).__name__}: {exc}"})
            fatal = _is_oom(exc)
            errors += 1
        append_jsonl(output, record)
        existing[(sample_id,)] = record
        written += 1
        if written == 1 or written % log_every == 0:
            error_detail = (
                f"; error={record['error']}" if record.get("error") else ""
            )
            print(
                f"[margin/baseline/{model_key}] written={written}: {sample_id} -> "
                f"M={record.get('margin')}{error_detail}"
            )
        if fatal:
            raise RuntimeError(
                f"CUDA OOM at baseline/{sample_id}; reduce competing GPU use and rerun with --retry-errors"
            )
    return {"existing": len(existing) - written, "written": written, "errors": errors, "total": len(existing)}


def _post_record(
    *,
    prompt: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    protocol: str,
    condition: str,
    condition_index: int,
    history: Sequence[str],
    persuasion_text: str | None,
    followup_text: str | None,
    baseline: Mapping[str, Any],
    pair: PairScore | None,
    previous_margin: float | None,
    previous_label: str,
    backend: MarginBackend,
    context: Sequence[Mapping[str, Any]] | None,
    error: str | None,
    status: str,
    behavioral: Mapping[str, Any] | None,
) -> dict[str, Any]:
    tolerance = float(config["scoring"].get("tie_tolerance", 1e-8))
    record = {
        **_base_record(prompt, config, model_key, backend),
        "run_id": f"{protocol}:{model_key}:{prompt['sample_id']}:{condition}",
        "phase": "post_persuasion",
        "protocol": protocol,
        "condition": condition,
        "condition_index": condition_index,
        "condition_label": CONDITION_LABELS[condition],
        "history_conditions": list(history),
        "persuasion_text": persuasion_text,
        "full_followup_text": followup_text,
        "baseline_margin": baseline.get("margin"),
        "previous_margin": previous_margin,
        "previous_selected_prediction": previous_label,
        "status": status,
        "error": error,
        "context_sha256": _canonical_hash(context) if context is not None else None,
        "behavioral_result_available": behavioral is not None,
        "behavioral_prediction": behavioral.get("final_prediction") if behavioral else None,
        "behavioral_flipped": behavioral.get("flipped") if behavioral else None,
        "behavioral_parseable": behavioral.get("parseable") if behavioral else None,
        "behavioral_probe_exact_match": False if behavioral else None,
    }
    if pair is None:
        return record
    record.update(
        _pair_payload(
            pair,
            str(prompt["gt"]),
            tie_tolerance=tolerance,
            previous_label=previous_label,
        )
    )
    margin = float(record["margin"])
    baseline_margin = float(baseline["margin"])
    displacement = baseline_margin - margin
    record.update({
        "margin_shift": margin - baseline_margin,
        "persuasion_displacement": displacement,
        "stage_margin_shift": (
            margin - previous_margin if previous_margin is not None else None
        ),
        "stage_displacement": (
            previous_margin - margin if previous_margin is not None else None
        ),
        "moved_toward_wrong": displacement > tolerance,
        "below_decision_boundary": margin < -tolerance,
        "boundary_crossed_from_baseline": baseline_margin > tolerance and margin < -tolerance,
        "stage_boundary_crossed": (
            previous_margin is not None
            and previous_margin > tolerance
            and margin < -tolerance
        ),
        "stage_recovered_correct_side": (
            previous_margin is not None
            and previous_margin < -tolerance
            and margin > tolerance
        ),
    })
    return record


def run_protocol(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    protocol: str,
    backend: MarginBackend,
    *,
    source_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    validate_model(config, model_key)
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol: {protocol}")
    baseline_rows = _read_unique(baseline_path(config, model_key, project_root), ("sample_id",))
    missing_baselines = [
        str(row["sample_id"])
        for row in prompts
        if (str(row["sample_id"]),) not in baseline_rows
    ]
    if missing_baselines:
        raise RuntimeError(
            f"missing {len(missing_baselines)} baseline margins; run_baselines must complete first"
        )
    output = protocol_path(config, model_key, protocol, project_root)
    existing = _read_unique(output, ("sample_id", "condition"))
    states, _ = baseline_states(prompts, config, model_key, project_root)
    behavioral = _behavioral_results(config, model_key, protocol, project_root)
    baseline_template = str(config["_formal_config"]["prompts"]["binary"]["template"])
    final_probe = str(config["scoring"]["final_probe"])
    candidates = {key: str(value) for key, value in config["scoring"]["candidates"].items()}
    configured_root = config["dataset"]["source_root"]
    log_every = max(1, int(config["inference"].get("log_every", 10)))
    written = errors = scored_count = 0
    skipped_ineligible = 0
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        baseline = baseline_rows[(sample_id,)]
        if baseline.get("error"):
            continue
        state = states.get(sample_id)
        if state is None:
            raise RuntimeError(
                f"no behavioral baseline for {sample_id}/{model_key}; "
                "run the formal binary baseline or Direct persuasion screening first"
            )
        if not state.get("is_correct"):
            for index, condition in enumerate(CONDITIONS):
                key = (sample_id, condition)
                if key in existing:
                    continue
                persuasion = prompt.get(condition)
                available = not (
                    condition == "credibility_appeal"
                    and prompt.get("credibility_available") is False
                )
                persuasion_text = str(persuasion) if available else None
                followup = (
                    _followup(persuasion_text, final_probe)
                    if persuasion_text is not None
                    else None
                )
                record = _post_record(
                    prompt=prompt,
                    config=config,
                    model_key=model_key,
                    protocol=protocol,
                    condition=condition,
                    condition_index=index,
                    history=[],
                    persuasion_text=persuasion_text,
                    followup_text=followup,
                    baseline=baseline,
                    pair=None,
                    previous_margin=float(baseline["margin"]),
                    previous_label=str(prompt["gt"]),
                    backend=backend,
                    context=None,
                    error=None,
                    status="skipped_ineligible_baseline",
                    behavioral=behavioral.get(key),
                )
                append_jsonl(output, record)
                existing[key] = record
                written += 1
                skipped_ineligible += 1
            continue
        baseline_answer = str(state["assistant_answer"])
        image_path = resolve_formal_image(
            prompt,
            source_root=source_root,
            configured_source_root=configured_root,
            project_root=project_root,
        )
        history: list[str] = []
        prior_turns: list[tuple[str, str]] = []
        previous_margin = float(baseline["margin"])
        previous_label = str(prompt["gt"])
        blocked_by_error = False
        for index, condition in enumerate(CONDITIONS):
            key = (sample_id, condition)
            if key in existing:
                saved = existing[key]
                if protocol == "cascade":
                    if saved.get("status") == "skipped_unavailable":
                        continue
                    if saved.get("error"):
                        blocked_by_error = True
                        break
                    followup = str(saved.get("full_followup_text") or "")
                    answer = str(saved.get("selected_prediction") or "")
                    if not followup or answer not in LABELS:
                        raise RuntimeError(f"invalid saved cascade state: {key}")
                    answer_text = str(config["scoring"]["candidates"][answer])
                    prior_turns.append((followup, answer_text))
                    history.append(condition)
                    previous_margin = float(saved["margin"])
                    previous_label = answer
                continue
            if protocol == "cascade" and blocked_by_error:
                break
            persuasion = prompt.get(condition)
            available = not (
                condition == "credibility_appeal"
                and prompt.get("credibility_available") is False
            )
            if not available:
                record = _post_record(
                    prompt=prompt,
                    config=config,
                    model_key=model_key,
                    protocol=protocol,
                    condition=condition,
                    condition_index=index,
                    history=history,
                    persuasion_text=None,
                    followup_text=None,
                    baseline=baseline,
                    pair=None,
                    previous_margin=previous_margin,
                    previous_label=previous_label,
                    backend=backend,
                    context=None,
                    error=None,
                    status="skipped_unavailable",
                    behavioral=behavioral.get(key),
                )
                append_jsonl(output, record)
                existing[key] = record
                written += 1
                continue
            followup = _followup(str(persuasion), final_probe)
            messages = _conversation(
                prompt,
                image_path,
                baseline_template,
                baseline_answer,
                prior_turns if protocol == "cascade" else [],
                followup,
            )
            pair: PairScore | None = None
            error: str | None = None
            fatal = False
            try:
                pair = backend.score_messages(messages, candidates)
                status = "completed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                status = "system_error"
                fatal = _is_oom(exc)
                errors += 1
            record = _post_record(
                prompt=prompt,
                config=config,
                model_key=model_key,
                protocol=protocol,
                condition=condition,
                condition_index=index,
                history=history,
                persuasion_text=str(persuasion),
                followup_text=followup,
                baseline=baseline,
                pair=pair,
                previous_margin=previous_margin,
                previous_label=previous_label,
                backend=backend,
                context=messages,
                error=error,
                status=status,
                behavioral=behavioral.get(key),
            )
            append_jsonl(output, record)
            existing[key] = record
            written += 1
            scored_count += 1
            if scored_count == 1 or scored_count % log_every == 0:
                error_detail = f"; error={error}" if error else ""
                print(
                    f"[margin/{protocol}/{model_key}] scores={scored_count}: "
                    f"{sample_id} {condition} -> M={record.get('margin')}"
                    f"{error_detail}"
                )
            if error:
                if fatal:
                    raise RuntimeError(
                        f"CUDA OOM at {protocol}/{sample_id}/{condition}; rerun with --retry-errors"
                    )
                if protocol == "cascade":
                    blocked_by_error = True
                    break
                continue
            if protocol == "cascade":
                selected = str(record["selected_prediction"])
                if selected not in LABELS:
                    selected = previous_label
                prior_turns.append((followup, str(candidates[selected])))
                history.append(condition)
                previous_margin = float(record["margin"])
                previous_label = selected
    return {
        "existing": len(existing) - written,
        "written": written,
        "scored": scored_count,
        "skipped_ineligible": skipped_ineligible,
        "errors": errors,
        "total_records": len(existing),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _numeric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("margin") is not None and not row.get("error")]
    margins = [float(row["margin"]) for row in valid]
    displacements = [
        float(row["persuasion_displacement"])
        for row in valid
        if row.get("persuasion_displacement") is not None
    ]
    stage_displacements = [
        float(row["stage_displacement"])
        for row in valid
        if row.get("stage_displacement") is not None
    ]
    behavioral = [
        row for row in valid
        if row.get("behavioral_result_available") and row.get("behavioral_parseable")
    ]
    confusion = Counter(
        (
            "behavior_flip" if row.get("behavioral_flipped") else "behavior_resist",
            "margin_wrong" if float(row["margin"]) < 0 else "margin_correct",
        )
        for row in behavioral
    )
    return {
        "n_rows": len(rows),
        "n_scored": len(valid),
        "system_errors": sum(bool(str(row.get("error") or "").strip()) for row in rows),
        "skipped_unavailable": sum(row.get("status") == "skipped_unavailable" for row in rows),
        "skipped_ineligible_baseline": sum(
            row.get("status") == "skipped_ineligible_baseline" for row in rows
        ),
        "mean_margin": statistics.fmean(margins) if margins else None,
        "median_margin": statistics.median(margins) if margins else None,
        "q25_margin": _percentile(margins, 0.25),
        "q75_margin": _percentile(margins, 0.75),
        "below_boundary_rate": sum(value < 0 for value in margins) / len(margins) if margins else None,
        "mean_displacement": statistics.fmean(displacements) if displacements else None,
        "median_displacement": statistics.median(displacements) if displacements else None,
        "moved_toward_wrong_rate": sum(value > 0 for value in displacements) / len(displacements) if displacements else None,
        "mean_stage_displacement": statistics.fmean(stage_displacements) if stage_displacements else None,
        "median_stage_displacement": statistics.median(stage_displacements) if stage_displacements else None,
        "behavioral_comparison_n": len(behavioral),
        "behavior_margin_concordance": {
            f"{left}__{right}": confusion[(left, right)]
            for left in ("behavior_flip", "behavior_resist")
            for right in ("margin_wrong", "margin_correct")
        },
    }


def _margin_bin(value: float, q25: float, q75: float) -> str:
    if value <= q25:
        return "low_initial_margin"
    if value >= q75:
        return "high_initial_margin"
    return "middle_initial_margin"


def independent_summary(
    rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_values = [float(row["margin"]) for row in baseline_rows if row.get("margin") is not None and not row.get("error")]
    q25 = float(_percentile(baseline_values, 0.25) or 0.0)
    q75 = float(_percentile(baseline_values, 0.75) or 0.0)
    by_bin: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("baseline_margin") is not None:
            by_bin[_margin_bin(float(row["baseline_margin"]), q25, q75)].append(row)
    return {
        "overall": _numeric_summary(rows),
        "by_condition": {
            condition: _numeric_summary([row for row in rows if row.get("condition") == condition])
            for condition in CONDITIONS
        },
        "by_fake_cls": {
            fake_cls: {
                condition: _numeric_summary([
                    row for row in rows
                    if row.get("fake_cls") == fake_cls and row.get("condition") == condition
                ])
                for condition in CONDITIONS
            }
            for fake_cls in sorted({str(row.get("fake_cls")) for row in rows})
        },
        "by_analysis_cohort": {
            cohort: {
                condition: _numeric_summary([
                    row for row in rows
                    if row.get("analysis_cohort") == cohort
                    and row.get("condition") == condition
                ])
                for condition in CONDITIONS
            }
            for cohort in sorted({
                str(row.get("analysis_cohort"))
                for row in rows
                if row.get("analysis_cohort") is not None
            })
        },
        "by_fake_cls_and_cohort": {
            fake_cls: {
                cohort: {
                    condition: _numeric_summary([
                        row for row in rows
                        if row.get("fake_cls") == fake_cls
                        and row.get("analysis_cohort") == cohort
                        and row.get("condition") == condition
                    ])
                    for condition in CONDITIONS
                }
                for cohort in sorted({
                    str(row.get("analysis_cohort"))
                    for row in rows
                    if row.get("analysis_cohort") is not None
                })
            }
            for fake_cls in sorted({str(row.get("fake_cls")) for row in rows})
        },
        "by_initial_margin_bin": {
            key: _numeric_summary(group) for key, group in sorted(by_bin.items())
        },
        "initial_margin_bin_thresholds": {"q25": q25, "q75": q75},
    }


def _cascade_trajectory_outcomes(
    rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id"))].append(row)
    baseline_index = {str(row["sample_id"]): row for row in baseline_rows}
    first_crossing = Counter()
    recovered = crossings = trajectories = baseline_wrong = 0
    for sample_id, sample_rows in grouped.items():
        baseline = baseline_index.get(sample_id)
        if baseline is None or baseline.get("margin") is None or baseline.get("error"):
            continue
        ordered = sorted(sample_rows, key=lambda row: int(row.get("condition_index", -1)))
        scored = [row for row in ordered if row.get("margin") is not None and not row.get("error")]
        if not scored:
            continue
        trajectories += 1
        m0 = float(baseline["margin"])
        if m0 < 0:
            baseline_wrong += 1
        first = next((row for row in scored if m0 > 0 and float(row["margin"]) < 0), None)
        if first is not None:
            first_crossing[str(first["condition"])] += 1
            later = [row for row in scored if int(row["condition_index"]) > int(first["condition_index"])]
            if any(float(row["margin"]) > 0 for row in later):
                recovered += 1
        signs = [1 if m0 > 0 else -1 if m0 < 0 else 0] + [
            1 if float(row["margin"]) > 0 else -1 if float(row["margin"]) < 0 else 0
            for row in scored
        ]
        crossings += sum(left * right == -1 for left, right in zip(signs, signs[1:]))
    return {
        "trajectory_count": trajectories,
        "baseline_already_wrong_by_margin": baseline_wrong,
        "first_correct_to_wrong_crossing_stage": {
            condition: first_crossing[condition] for condition in CONDITIONS
        },
        "ever_crossed_correct_to_wrong": sum(first_crossing.values()),
        "recovered_after_first_crossing": recovered,
        "all_sign_crossings": crossings,
    }


def cascade_summary(
    rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = {
        "overall": _numeric_summary(rows),
        "by_stage": {
            condition: _numeric_summary([row for row in rows if row.get("condition") == condition])
            for condition in CONDITIONS
        },
        **_cascade_trajectory_outcomes(rows, baseline_rows),
        "history_policy": "At each stage, write the full sequence-likelihood argmax answer into the conversation; exact ties retain the prior label.",
    }
    cohorts = sorted({
        str(row.get("analysis_cohort"))
        for row in rows
        if row.get("analysis_cohort") is not None
    })
    if cohorts:
        baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
        result["by_analysis_cohort_trajectory"] = {}
        for cohort in cohorts:
            cohort_rows = [
                row for row in rows if row.get("analysis_cohort") == cohort
            ]
            cohort_ids = {str(row["sample_id"]) for row in cohort_rows}
            cohort_baselines = [
                baseline_by_id[sample_id]
                for sample_id in cohort_ids
                if sample_id in baseline_by_id
            ]
            result["by_analysis_cohort_trajectory"][cohort] = (
                _cascade_trajectory_outcomes(cohort_rows, cohort_baselines)
            )
        result["by_analysis_cohort_stage"] = {
            cohort: {
                condition: _numeric_summary([
                    row for row in rows
                    if row.get("analysis_cohort") == cohort
                    and row.get("condition") == condition
                ])
                for condition in CONDITIONS
            }
            for cohort in cohorts
        }
        result["by_fake_cls_and_cohort_stage"] = {
            fake_cls: {
                cohort: {
                    condition: _numeric_summary([
                        row for row in rows
                        if row.get("fake_cls") == fake_cls
                        and row.get("analysis_cohort") == cohort
                        and row.get("condition") == condition
                    ])
                    for condition in CONDITIONS
                }
                for cohort in cohorts
            }
            for fake_cls in sorted({str(row.get("fake_cls")) for row in rows})
        }
    return result


def aggregate_model(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_key: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    validate_model(config, model_key)
    baselines = list(read_jsonl(baseline_path(config, model_key, project_root))) if baseline_path(config, model_key, project_root).is_file() else []
    protocols = {
        protocol: (
            list(read_jsonl(protocol_path(config, model_key, protocol, project_root)))
            if protocol_path(config, model_key, protocol, project_root).is_file()
            else []
        )
        for protocol in PROTOCOLS
    }
    expected = len(prompts) * len(CONDITIONS)
    summary = {
        "model_key": model_key,
        "display_name": config["models"][model_key]["display_name"],
        "comparison_scope": "within_model_only",
        "integrity": {
            "expected_samples": len(prompts),
            "baseline_rows": len(baselines),
            "baseline_errors": sum(bool(row.get("error")) for row in baselines),
            "expected_rows_per_protocol": expected,
            **{
                f"{protocol}_rows": len(rows) for protocol, rows in protocols.items()
            },
            **{
                f"{protocol}_errors": sum(bool(row.get("error")) for row in rows)
                for protocol, rows in protocols.items()
            },
        },
        "baseline": {
            **_numeric_summary(baselines),
            "by_analysis_cohort": {
                cohort: _numeric_summary([
                    row for row in baselines if row.get("analysis_cohort") == cohort
                ])
                for cohort in sorted({
                    str(row.get("analysis_cohort"))
                    for row in baselines
                    if row.get("analysis_cohort") is not None
                })
            },
            "by_fake_cls_and_cohort": {
                fake_cls: {
                    cohort: _numeric_summary([
                        row for row in baselines
                        if row.get("fake_cls") == fake_cls
                        and row.get("analysis_cohort") == cohort
                    ])
                    for cohort in sorted({
                        str(row.get("analysis_cohort"))
                        for row in baselines
                        if row.get("analysis_cohort") is not None
                    })
                }
                for fake_cls in sorted({str(row.get("fake_cls")) for row in baselines})
            },
        },
        "independent": independent_summary(protocols["independent"], baselines),
        "cascade": cascade_summary(protocols["cascade"], baselines),
        "interpretation": {
            "margin": "M=logP(correct sequence|x)-logP(wrong sequence|x)",
            "margin_shift": "DeltaM=M_after-M_baseline; negative means movement toward the wrong answer",
            "persuasion_displacement": "D=M_baseline-M_after; positive means movement toward the wrong answer",
            "cross_model_warning": "Raw margins are tokenizer/model dependent and should not be compared as calibrated confidence across models.",
            "behavioral_warning": "Linked old generation verdicts used a shorter follow-up instruction, so behavioral-vs-margin comparisons are descriptive rather than exact prompt-matched replications.",
        },
    }
    integrity = summary["integrity"]
    integrity["formal_complete"] = bool(
        integrity["baseline_rows"] == len(prompts)
        and integrity["baseline_errors"] == 0
        and all(integrity[f"{protocol}_rows"] == expected for protocol in PROTOCOLS)
        and all(integrity[f"{protocol}_errors"] == 0 for protocol in PROTOCOLS)
    )
    output = model_result_dir(config, model_key, project_root) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return summary


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def render_report(summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# MMFakeBench teacher-forced persuasion margins",
        "",
        "Margins use complete `Finish[REAL].` and `Finish[FAKE].` sequences. Values are for within-model comparison only.",
        "",
        "| Model | Baseline median M | Independent median D | Cascade final median D | Cascade ever crossed | Complete |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        final_stage = summary["cascade"]["by_stage"][CONDITIONS[-1]]
        lines.append(
            f"| {summary['display_name']} | {_fmt(summary['baseline']['median_margin'])} | "
            f"{_fmt(summary['independent']['overall']['median_displacement'])} | "
            f"{_fmt(final_stage['median_displacement'])} | "
            f"{summary['cascade']['ever_crossed_correct_to_wrong']} | "
            f"{summary['integrity']['formal_complete']} |"
        )
    lines.extend([
        "",
        "`M > 0` supports the correct label; `D = M0 - M_after > 0` means persuasion moved the model toward the wrong label.",
        "Cascade runs all seven stages and records first crossing plus later recovery; it does not stop at the first flip.",
        "Old generated verdicts are linked when available, but their shorter final instruction is not an exact prompt match.",
        "",
    ])
    return "\n".join(lines)


def aggregate_all(
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    model_keys: Sequence[str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    selected = list(model_keys) if model_keys is not None else active_models(config)
    summaries = [
        aggregate_model(prompts, config, model_key, project_root=project_root)
        for model_key in selected
    ]
    report = _resolve(config["reporting"]["report"], project_root)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(render_report(summaries), encoding="utf-8")
    temporary.replace(report)
    return summaries
