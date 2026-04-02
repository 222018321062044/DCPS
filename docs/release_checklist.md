# DCPS Release Checklist

Use this checklist before turning the repository into the public paper release.

## Required repository checks

- Verify the public method name is consistently `DCPS` in user-facing files.
- Run `python scripts/validate_repo.py`.
- Confirm `README.md` matches the final command-line interface.
- Confirm `docs/reproduce.md` points to the correct experiment scripts.
- Confirm `.gitignore` excludes datasets, checkpoints, results, and local tooling files.

## Environment checks

- Test installation from `requirements.txt`.
- Test environment creation from `environment.yml`.
- Confirm the runtime uses a NumPy 1.x environment compatible with the pinned dependencies.
- If using Windows, confirm the environment does not trigger duplicate OpenMP runtime errors during normal execution.

## Experiment assets

- Remove local checkpoints and large weights from the repository tree.
- Remove local result dumps, debug figures, and notebook outputs that are not meant for release.
- Verify all paths in `scripts/*.sh` are either relative or clearly documented.

## Metadata still to finalize

- Verify that `LICENSE` matches the intended redistribution terms.
- Verify the final venue metadata in `CITATION.cff`.

## Final manual review

- Read the repository root as if you were a new user.
- Check that the first-run path is obvious from `README.md`.
- Check that every public script mentioned in the docs actually exists.
