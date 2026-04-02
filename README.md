# DCPS: Dynamic Cache-Prompt Synergy

Official codebase for **Dynamic Cache-Prompt Synergy (DCPS)**, a prompt-based continual learning method for CLIP under multi-domain incremental learning.

This repository keeps the DCPS training and checkpoint-evaluation workflow, plus the retained ablation and few-shot scripts.

## Highlights

- Prompt-based continual learning on top of CLIP.
- Multi-domain task incremental learning on 11 image classification datasets.
- Cache-prompt synergy for stronger robustness under distribution shifts.
- Main experiment pipeline plus retained ablation and few-shot scripts.

## Repository Layout

```text
.
|- custom_clip/          # DCPS model implementations and CLIP wrappers
|- docs/                 # Reproduction notes and release guidance
|- notebooks/            # Optional local notebooks
|- scripts/              # DCPS training, evaluation, ablation, and few-shot scripts
|- src/                  # Training, evaluation, datasets, and core model code
|- tools/                # Small utility scripts
|- requirements.txt
`- README.md
```

## Environment

Recommended environment:

- Python 3.10
- PyTorch 2.0.1
- torchvision 0.15.2
- CUDA-capable GPU for full training

Install dependencies with either `pip` or Conda.

Using `pip`:

```bash
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate dcps
```

If you need a specific CUDA build of PyTorch, install `torch` and `torchvision` from the official PyTorch index first, then install the remaining packages from `requirements.txt`. Keep `numpy` on the pinned 1.x line from `requirements.txt` to avoid binary compatibility issues with older SciPy wheels.

## Data Preparation

Prepare your datasets under a single root directory, for example:

```text
data/
|- Aircraft/
|- Caltech101/
|- CIFAR100/
|- DTD/
|- EuroSAT/
|- Flowers/
|- Food/
|- MNIST/
|- OxfordPet/
|- StanfordCars/
`- SUN397/
```

The MTIL experiments in this repository use the following datasets:

- `Aircraft`
- `Caltech101`
- `CIFAR100`
- `DTD`
- `EuroSAT`
- `Flowers`
- `Food`
- `MNIST`
- `OxfordPet`
- `StanfordCars`
- `SUN397`

Use `--data-location` to point every script to your local dataset root.

## Quick Start

Train DCPS on a single dataset:

```bash
python -m src.main \
  --train-mode prompt \
  --trainer DCPS \
  --train-dataset Aircraft \
  --eval-datasets Aircraft \
  --data-location data \
  --save ckpt/exp_dcps_single
```

Evaluate a checkpoint set:

```bash
python -m src.general_eval \
  --eval-names exp_dcps_single \
  --data-location data \
  --output-dir results
```

Run the main multi-domain DCPS script:

```bash
bash scripts/DCPS.sh
```

## Retained Supplementary Scripts

- `scripts/DCPS-ablationdepth.sh`
- `scripts/DCPS-ablationlength.sh`
- `scripts/DCPS-fewshot.sh`

## Validation

Run the lightweight repository smoke checks before publishing changes or tagging a release:

```bash
python scripts/validate_repo.py
```

This checks Python compilation and the retained CLI help entry points without requiring datasets or checkpoints.

## Reproducing Experiments

The `scripts/` directory contains the retained public experiment entry points:

- `scripts/DCPS.sh`: main DCPS training and evaluation pipeline
- `scripts/DCPS-ablationdepth.sh`: prompt depth ablation
- `scripts/DCPS-ablationlength.sh`: prompt length ablation
- `scripts/DCPS-fewshot.sh`: few-shot setting

See `docs/reproduce.md` for a compact mapping from scripts to experiment types.
See `docs/method.md` for a concise method overview and code map.

## Outputs

By default:

- checkpoints are written under `ckpt/`
- evaluation outputs are written under `results/`

You can override output locations with:

- `--save`
- `--output-dir`

## Notes

- This repository does not require checking large model weights into git.
- Local caches, checkpoints, experimental outputs, and datasets should stay outside version control.
- See `CONTRIBUTING.md` for change hygiene and `docs/release_checklist.md` for the final public release pass.

## Troubleshooting

- If you see NumPy or SciPy binary compatibility errors, recreate the environment and keep `numpy` on the pinned 1.x version from `requirements.txt`.
- On Windows, duplicate OpenMP runtime errors usually indicate a mixed environment. Recreate a clean environment instead of mixing incompatible binary packages from different toolchains.

## Citation

If you use DCPS in your research, please cite the corresponding paper. The repository includes a `CITATION.cff` entry for the current paper title and author list. Please verify the final venue metadata before the release tag.

## License

This repository is released under the MIT License. See `LICENSE`.

## Acknowledgments

This repository includes research code built around CLIP-based continual learning experiments. Third-party licenses and attributions should be preserved in any redistribution.
