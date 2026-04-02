# Reproducing DCPS Experiments

This repository exposes the retained DCPS workflows through shell scripts in `scripts/`.

## Main experiment

- `scripts/DCPS.sh`
  - Main DCPS multi-domain task incremental learning pipeline.
  - Trains sequentially on the 11 benchmark datasets.
  - Runs evaluation with `src.general_eval`.

## Retained supplementary experiments

- `scripts/DCPS-ablationdepth.sh`
  - Prompt depth ablation.

- `scripts/DCPS-ablationlength.sh`
  - Prompt length ablation.

- `scripts/DCPS-fewshot.sh`
  - Few-shot experiment entry point.

## Common arguments

The retained public command-line interface accepts the following path-related flags:

- `--data-location`
  - Root directory containing prepared datasets.

- `--save`
  - Checkpoint output directory.

- `--output-dir`
  - Evaluation output directory.

## Before running

1. Install the dependencies from `requirements.txt` or create the Conda environment from `environment.yml`.
2. Prepare the datasets under a single local root such as `data/`.
3. Update the dataset path in the script you plan to run, or pass `--data-location` directly.
4. Make sure a CUDA-enabled PyTorch environment is available for full training.

## Recommended release checklist

Before publishing a paper release, verify:

- `python scripts/validate_repo.py`
- `python -m src.main --help`
- `python -m src.general_eval --help`
- `bash scripts/DCPS.sh` has valid local paths
- checkpoints and large weights are not committed
- `docs/release_checklist.md` is fully reviewed
