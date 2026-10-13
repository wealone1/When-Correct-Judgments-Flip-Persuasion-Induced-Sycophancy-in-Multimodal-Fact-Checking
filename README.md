# Multimodal Persuasion Evaluation

Anonymous submission code for evaluating whether multimodal language models change a correct misinformation verdict after user persuasion. The package contains the formal MMFakeBench runners, frozen experiment inputs, model configuration, and regression tests.

## Included experiments

1. **Formal baseline** on 7,700 MMFakeBench examples:
   - binary `Real`/`Fake` prediction;
   - error-modality localization (`ORIGINAL`, `TEXT REFUTES`, or `IMAGE REFUTES`).
2. **Seven-condition persuasion** on the frozen 3,599-example seven-model shared-correct subset:
   - simple disagreement;
   - repetition;
   - logical appeal;
   - credibility appeal;
   - emotional appeal;
   - text-grounded persuasion;
   - image-grounded persuasion.
3. **Training-free CoT variants** using the same seven persuasion conditions:
   - generic CoT;
   - structured multimodal CoT;
   - persuasion-audit CoT.
4. **Focused cascade** on the corresponding frozen 255-example diagnostic set.
5. **Teacher-forced full-label margin analysis**, in independent and cascade protocols.
6. **Leave-one-modality-out contribution analysis**.

Prompt-generation, dataset-selection, plotting, spreadsheet, and paper-result aggregation utilities are intentionally excluded. The checked-in JSONL files are the frozen inputs used by the submitted experiments.

## Repository layout

```text
submission_code/
├── configs/                         # portable experiment/model configuration
├── data/                            # frozen manifests and persuasion inputs; no images
├── src/prototype_induction/         # evaluation implementation
├── tests/                           # parser, protocol, and scoring regression tests
├── run_mmfakebench_formal.py
├── run_mmfakebench_modality_persuasion.py
├── run_mmfakebench_margin.py
└── run_mmfakebench_modality_contribution.py
```

## External assets

This package does not redistribute MMFakeBench images or model weights.

- Download MMFakeBench separately and pass its root with `--source-root`.
- Download each model separately and pass its directory with `--model-path`.
- The expected dataset layout is `MMFakeBench_{test,val}/...` under the supplied source root.

The model keys accepted by the runners are:

```text
gemma4-12b-it
gemma4-26b
internvl35-8b
qwen35-4b
qwen35-4b-thinking-4096
qwen35-9b
qwen35-9b-thinking-4096
qwen35-35b
qwen3-vl-8b
qwen3-vl-8b-thinking-4096
```

Margin and modality-contribution scoring are defined only for non-thinking models.

## Installation

Python 3.10 or newer is required. Install the PyTorch build appropriate for the server's CUDA version, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the tests without loading model weights:

```bash
pytest -q
```

## Reproducing the experiments

Set paths once for the examples below:

```bash
export MMFAKEBENCH_ROOT=/path/to/MMFakeBench
export MODEL_ROOT=/path/to/model
```

### 1. Validate frozen inputs

```bash
python run_mmfakebench_formal.py validate \
  --source-root "$MMFAKEBENCH_ROOT"

python run_mmfakebench_modality_persuasion.py validate \
  --source-root "$MMFAKEBENCH_ROOT"

python run_mmfakebench_margin.py \
  --config configs/mmfakebench_analysis_margin.yaml \
  validate --skip-images

python run_mmfakebench_modality_contribution.py \
  --config configs/mmfakebench_modality_contribution.yaml \
  validate --skip-images
```

### 2. Formal 7,700-example baseline

```bash
CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_formal.py run \
  --model qwen3-vl-8b \
  --task both \
  --model-path "$MODEL_ROOT" \
  --source-root "$MMFAKEBENCH_ROOT"
```

Use `--limit 1` for a smoke test. Remove it for a complete run.

### 3. Seven independent persuasion conditions

```bash
CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_modality_persuasion.py run \
  --model qwen3-vl-8b \
  --mode independent \
  --reasoning-mode direct \
  --model-path "$MODEL_ROOT" \
  --source-root "$MMFAKEBENCH_ROOT"
```

### 4. Three CoT conditions

Run the preceding command three times, changing `--reasoning-mode` to:

```text
generic_cot
structured_multimodal_cot
persuasion_audit_cot
```

Each run uses the same frozen examples and persuasion messages as the direct condition.

### 5. Focused seven-stage cascade

```bash
CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_modality_persuasion.py \
  --config configs/mmfakebench_analysis_persuasion.yaml \
  run \
  --model qwen3-vl-8b \
  --mode cascade \
  --reasoning-mode direct \
  --model-path "$MODEL_ROOT" \
  --source-root "$MMFAKEBENCH_ROOT"
```

### 6. Teacher-forced margins

```bash
CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_margin.py \
  --config configs/mmfakebench_analysis_margin.yaml \
  run \
  --model qwen3-vl-8b \
  --protocol both \
  --model-path "$MODEL_ROOT" \
  --source-root "$MMFAKEBENCH_ROOT"
```

### 7. Modality contribution

```bash
CUDA_VISIBLE_DEVICES=0 python run_mmfakebench_modality_contribution.py \
  --config configs/mmfakebench_modality_contribution.yaml \
  run \
  --model qwen3-vl-8b \
  --model-path "$MODEL_ROOT" \
  --source-root "$MMFAKEBENCH_ROOT"
```

All runners append completed records and skip existing keys, so interrupted runs resume safely. Use `--retry-errors` to archive and rerun only system-error records.

## Output locations

Outputs are written below `results/` according to the selected config. These generated files are ignored by version control and are not included in this package.

## Notes on reproducibility

- Decoding is greedy (`do_sample: false`) unless a model-specific thinking configuration explicitly requires otherwise.
- The 3,599-example pool requires a correct, parseable formal binary baseline from all seven reported non-thinking models, including InternVL3.5-8B.
- The 255-example diagnostic set is the earlier R7/S1+ analysis cohort restricted to this same InternVL-correct pool.
- Formal prompts, seven persuasion messages, CoT instructions, candidate labels, and margin probes are frozen in `configs/` and `data/`.
- Margin scores use teacher forcing over the complete label sequence (`Finish[REAL].` or `Finish[FAKE].`) and exclude EOS.
- Absolute paths in the original experiment records were replaced by `/path/to/MMFakeBench`; image resolution uses `--source-root` plus each row's relative source path.

