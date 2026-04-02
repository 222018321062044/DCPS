# Contributing to DCPS

This repository is organized as a paper codebase. Changes should stay focused,
easy to review, and directly connected to training, evaluation, analysis, or
documentation for DCPS.

## Before opening a change

1. Create or activate a clean Python 3.10 environment.
2. Install the dependencies from `requirements.txt` or `environment.yml`.
3. Keep datasets, checkpoints, and experiment outputs outside version control.

## Development expectations

- Prefer small, self-contained changes.
- Preserve the public method name `DCPS` in user-facing docs and scripts.
- Avoid committing local data, checkpoints, figures, notebooks outputs, or logs.
- Keep internal refactors conservative unless they are required for correctness.

## Validation

Run the repository smoke checks before finalizing a change:

```bash
python scripts/validate_repo.py
```

If you modify experiment scripts or CLI flags, also re-check the related help
entry points manually.

## Documentation

Update the relevant docs when behavior changes:

- `README.md` for public usage and environment notes
- `docs/reproduce.md` for experiment entry points
- `docs/method.md` for code map or method-facing implementation notes

## Release-sensitive files

The following should be reviewed carefully before a public paper release:

- `README.md`
- `requirements.txt`
- `environment.yml`
- `.gitignore`
- `docs/release_checklist.md`

## License and citation

Do not change licensing or citation metadata casually. The repository now
includes `LICENSE` and `CITATION.cff`, but the citation metadata should still
be checked against the final venue metadata before the camera-ready release.
