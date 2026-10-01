# MOF Adsorption Regimes

Repository-support snapshot for the manuscript **“Systematic Underprediction Reveals a Hidden Low-Pressure CO₂ Adsorption Regime in Compact Metal–Organic Frameworks.”**

Target private repository: `https://github.com/Asgari-research/MOF-Adsorption-Regimes`

## What is authoritative

The 12 PDFs under `figures/final/` are the locked publication artwork:

- Main Figures 1–4
- Supporting Figures S1–S8

Run `python verify_final_figures.py` from the repository root before and after any Git operation that touches the figure tree. These PDFs must not be regenerated or silently replaced during repository cleanup.

## Repository contents

- `figures/final/` — immutable publication PDFs.
- `integrity/` — SHA-256 lock records for the final PDFs.
- `figure_generation/` — portable plotting/review snapshot plus the supplied source data and 14 CIFs. Its outputs go to `outputs_review/`, never to `figures/final/`.
- `analysis/frozen_original/` — byte-preserved original analysis/publication scripts from the supplied code archive. They are retained for provenance, not as the default execution path.
- `analysis/proposed_repairs/` — isolated reference fixes/tests for known static defects; these are not wired into the frozen analysis.
- `analysis/audit_evidence/` — compact evidence needed to understand unresolved scientific/reproducibility issues.
- `docs/` — WSL2/Git instructions, data provenance, reproducibility limits, and issue disposition.
- `scripts/verify_repository.py` — repository-level integrity and completeness checks.

## Reproducibility boundary

This snapshot supports:

1. verification of the locked publication figures;
2. regeneration of review copies of the current plotting figures from the supplied CSV/CIF inputs;
3. static inspection of the original analysis code and its documented feature/preprocessing behavior.

It does **not** support a verified end-to-end rerun of the trained models because the supplied materials do not include the complete original raw inputs, row-level train/calibration/test manifests, all per-row prediction/interval files, the exact numerical elite threshold for the representative 0.15-bar job, or the original training lockfile/environment.

The plotting review path is intentionally separate from the frozen analysis and from the locked publication artwork.

## Start here

For Ubuntu/WSL2, read `docs/GITHUB_WSL2_RUNBOOK.md` and then run:

```bash
python verify_final_figures.py
python scripts/verify_repository.py
cd figure_generation
python check_environment.py
python run_all_figures.py
```

Generated review assets appear only under `figure_generation/outputs_review/` and are ignored by Git.

## Important unresolved items

Before any public release or DOI deposition, resolve the author-list discrepancy, license choice, citation metadata, external-identifier matching defect, coverage-level lineage, predictor semantics, exact elite threshold, and missing raw reproduction inputs described in `docs/SCIENTIFIC_REPRODUCIBILITY_NOTES.md`.
