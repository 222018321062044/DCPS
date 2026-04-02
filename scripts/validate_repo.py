#!/usr/bin/env python3
"""Run lightweight repository validation checks for the public DCPS release."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PY_COMPILE_TARGETS = [
    "src/args.py",
    "src/main.py",
    "src/general_eval.py",
    "scripts/check_checkpoint.py",
    "tools/npy_check.py",
    "tools/t.py",
]

COMPILEALL_TARGETS = ["src", "custom_clip", "scripts", "tools", "__init__.py"]

HELP_COMMANDS = [
    ("src.main --help", ["-m", "src.main", "--help"]),
    ("src.general_eval --help", ["-m", "src.general_eval", "--help"]),
]


def run_command(label: str, command: list[str], verbose: bool) -> bool:
    print(f"\n[check] {label}")
    result = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if verbose and result.stdout:
        print(result.stdout.rstrip())
    if verbose and result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)

    if result.returncode != 0:
        if not verbose and result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        print(f"[fail] {label}")
        return False

    print(f"[ok] {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DCPS repository entry points.")
    parser.add_argument(
        "--skip-compileall",
        action="store_true",
        help="Skip the recursive compileall check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print command output even for passing checks.",
    )
    args = parser.parse_args()

    checks: list[tuple[str, list[str]]] = [
        ("py_compile", ["-m", "py_compile", *PY_COMPILE_TARGETS]),
    ]
    if not args.skip_compileall:
        checks.append(("compileall", ["-m", "compileall", *COMPILEALL_TARGETS]))
    checks.extend(HELP_COMMANDS)

    failures: list[str] = []
    for label, command in checks:
        if not run_command(label, command, args.verbose):
            failures.append(label)

    print("\n" + "=" * 80)
    if failures:
        print("Validation failed for:")
        for label in failures:
            print(f"- {label}")
        return 1

    print("All validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
