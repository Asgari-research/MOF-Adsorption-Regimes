# Repository audit — 1 October 2026

## Scope

Audit target: the downloaded `main` snapshot supplied as `MOF-Adsorption-Regimes-main.zip`, compared byte-for-byte against the prepared repository overlay.

## Content-state result

The downloaded `main` snapshot matches the prepared repository overlay byte-for-byte for every common file. The only additional file in `main` is the pre-existing top-level `LICENSE` file.

The branch handoff therefore reached `main` at the content level. A GitHub ZIP does not contain `.git`, so merge strategy/history (merge commit, squash, or fast-forward) must be checked in a local clone with `git log`.

## Verification results

- 108 files in the supplied `main` snapshot before this cleanup update.
- 12/12 locked publication PDFs match the SHA-256 manifest.
- Figure-source data hashes pass.
- Frozen-original analysis hashes pass.
- 15 Python files parse successfully as syntax trees.
- 4/4 focused identifier-matching tests pass.
- 41 CSV files parse with consistent row widths.
- 2 JSON files parse successfully.
- 14 CIF files contain basic unit-cell and atom-coordinate fields.
- 12 PDFs have valid PDF signatures.
- No likely API tokens/private keys were detected by a lightweight static scan.
- No files exceed 50 MiB; therefore no file approaches GitHub's 100 MiB per-blob limit.
- No archive/cache/log files were present in the downloaded repository snapshot.

Five exact duplicate source-data payload groups are intentionally retained under figure-specific filenames for provenance. They are not corruption.

## Findings requiring cleanup

### R01 — license documentation contradiction

`LICENSE` is present and contains the MIT license, while the repository documentation stated that no license was established and that the package intentionally did not add a license file. The latter statement was true of the overlay package but false of the integrated repository because the pre-existing repository already had `LICENSE`.

The cleanup updates the documentation to state the actual current condition without deciding whether MIT was author-approved for public release.

### R02 — stale integration instructions

`docs/GITHUB_WSL2_RUNBOOK.md` still described the pre-integration overlay/import process and a temporary branch that is no longer the active workflow. `docs/GIT_COMMIT_PLAN.md` similarly described the already-completed initial import.

The cleanup converts both documents into post-integration maintenance guidance.

### R03 — integrity verifier was narrower than the repository claim

The original verifier checked hashes for final figures, figure data, and frozen analysis plus a few required files. It did not verify exact manifest/tree coverage, file byte counts, portable plotting hashes, Python syntax, forbidden generated files, GitHub 100 MiB limits, simple secret patterns, or license-document consistency.

The cleanup strengthens `scripts/verify_repository.py` without importing or executing the frozen analysis.

### R04 — no automatic CI integrity check

The repository had local verification scripts but no GitHub Actions workflow. The cleanup adds a standard-library-only CI workflow that verifies hashes, repository structure, focused unit tests, and forbidden generated files on pushes and pull requests. It does not regenerate figures and does not rerun models.

## Scientific/reproducibility boundary unchanged

No scientific data, CIF, final figure, or frozen-original analysis file is changed by this cleanup. Known scientific issues and missing reconstruction inputs remain documented. End-to-end model reproduction is still not claimed.
