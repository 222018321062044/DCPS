# Main Experiment Repo Trim Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce the repository to the DCPS main experiment workflow only by removing ENGINE, sensitivity-analysis, profiling, and manuscript artifacts.

**Architecture:** Keep the single supported path as `src.main` for DCPS training plus `src.general_eval` for checkpoint evaluation. Remove non-main entrypoints, delete orphaned files, and update docs and validation so the public surface matches the reduced scope.

**Tech Stack:** Python 3.10, PyTorch, shell scripts, Markdown docs

---

### Task 1: Trim the public narrative

**Files:**
- Modify: `README.md`
- Modify: `docs/reproduce.md`
- Modify: `docs/method.md`
- Modify: `docs/release_checklist.md`

**Step 1: Remove references to ENGINE, sensitivity analysis, profiling, and paper artifacts**

**Step 2: Keep only the DCPS main experiment scripts and outputs**

**Step 3: Ensure the docs describe one primary workflow: train with `src.main`, evaluate with `src.general_eval`, run `scripts/DCPS.sh`**

### Task 2: Delete non-main-experiment files

**Files:**
- Delete: `custom_clip/ENGINE.py`
- Delete: `src/eval_engine.py`
- Delete: `src/train_engine.py`
- Delete: `src/profiling.py`
- Delete: `src/sensitivity_analysis.py`
- Delete: `src/train_and_analyze.py`
- Delete: `src/analysis_summary.py`
- Delete: `src/visualization.py`
- Delete: `src/models/sensitivity_online.py`
- Delete: `src/models/sensitivity_hook.py`
- Delete: `scripts/ENGINE.sh`
- Delete: `scripts/run_analysis.sh`
- Delete: `scripts/run_analysis.bat`
- Delete: `scripts/run_train_and_analyze.sh`
- Delete: `scripts/train_with_analysis.sh`
- Delete: `scripts/convert_sensitivity_to_csv.py`
- Delete: `paper/`

**Step 1: Remove every file that only serves ENGINE, sensitivity analysis, profiling, or manuscript packaging**

**Step 2: Keep the DCPS main experiment script and core training/evaluation modules intact**

### Task 3: Apply minimal compatibility updates

**Files:**
- Modify: `src/main.py`
- Modify: `src/args.py`
- Modify: `src/models/evaluation.py`
- Modify: `scripts/DCPS.sh`
- Modify: `scripts/validate_repo.py`

**Step 1: Remove the ENGINE training branch from the main entrypoint**

**Step 2: Remove sensitivity-analysis CLI flags and validation checks**

**Step 3: Remove only the deleted-feature hooks from evaluation and leave the remaining main experiment flow intact**

**Step 4: Update `scripts/DCPS.sh` so it runs the main DCPS training and final evaluation only**

### Task 4: Verify the reduced repository

**Files:**
- Run: `python scripts/validate_repo.py --skip-compileall`
- Run: `python -m src.main --help`
- Run: `python -m src.general_eval --help`

**Step 1: Run the lightweight validation script**

**Step 2: Confirm the retained entrypoints still expose valid CLI help**

**Step 3: Summarize the new repository scope and any remaining cleanup opportunities**
