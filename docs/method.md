# DCPS Method Overview

DCPS stands for **Dynamic Cache-Prompt Synergy**.

The method targets **multi-domain task incremental learning** with CLIP. In this setting, the model sees a sequence of tasks from different datasets and must learn each new task without collapsing on earlier ones.

## Core idea

DCPS combines two ingredients:

- **Prompt learning**
  - The model learns task-adaptive prompts on top of the CLIP backbone instead of fully fine-tuning the full network.

- **Cache-based adaptation**
  - The method augments prediction with cache signals that refine the logits during evaluation.

The central design goal is to let prompt learning handle continual representation adaptation while the cache improves robustness under domain and distribution shifts.

## Main code entry points

- `src/main.py`
  - Main training and evaluation entry point.

- `src/models/prompt_tune.py`
  - Core DCPS training loop for prompt-based continual learning.

- `custom_clip/PromptCross.py`
  - Main DCPS model implementation and cache-related logic.

- `src/models/evaluation.py`
  - Evaluation logic and prompt selection used during training checkpoints.

- `src/general_eval.py`
  - Sequential checkpoint evaluation across the MTIL benchmark.

## Benchmark setting

The current public scripts target the following 11 datasets:

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

## Practical note

This repository is organized as a focused experiment codebase, not a general-purpose framework. The public-facing priority is:

1. clear method entry points
2. reproducible scripts
3. explicit dataset and output paths
4. minimal repository noise
