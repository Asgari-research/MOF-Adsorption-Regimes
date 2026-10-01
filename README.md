# MOF Adsorption Regimes

Companion repository for the manuscript **“Systematic Underprediction Reveals a Hidden Low-Pressure CO₂ Adsorption Regime in Compact Metal–Organic Frameworks.”**

## Repository status

This repository preserves the audited code, source data used for the supplied figure-level analyses, selected CIFs, and locked publication figures available in the project snapshot. The repository is suitable for provenance review and integrity checking, but it is **not** presented as a verified end-to-end rerun package for the trained models because several original raw/reconstruction inputs and the historical training environment are not available in the supplied materials.

The 12 PDFs under `figures/final/` are the authoritative publication artwork:

- Main Figures 1–4
- Supporting Figures S1–S8

They are protected by SHA-256 manifests under `integrity/` and must not be silently regenerated or replaced.

## Quick integrity check

No Conda environment and no figure regeneration are required for repository verification. From the repository root, run:

```bash
python verify_final_figures.py
python scripts/verify_repository.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s analysis/proposed_repairs/tests -v
python scripts/check_forbidden_git_files.py
```

These checks verify the locked publication figures, figure-source data, frozen original analysis snapshot, repository structure, Python syntax, and the focused reference tests for the proposed identifier-matching repair.

## Repository contents

- `figures/final/` — immutable publication PDFs.
- `integrity/` — SHA-256 lock records for the final PDFs and curated plotting source.
- `figure_generation/` — optional plotting/review snapshot plus supplied source CSVs and 14 CIFs. Review outputs go only to `outputs_review/`.
- `analysis/frozen_original/` — byte-preserved original analysis/publication scripts. These files are provenance records and are not silently modified.
- `analysis/proposed_repairs/` — isolated reference implementations/tests for documented static defects; they are not wired into the frozen analysis.
- `analysis/audit_evidence/` — compact evidence supporting the documented scientific/reproducibility limitations.
- `docs/` — provenance, feature semantics, reproducibility limits, figure-lock policy, and local Git/WSL2 guidance.
- `scripts/verify_repository.py` — repository-level integrity and consistency audit.
- `.github/workflows/repository-integrity.yml` — lightweight CI that performs integrity checks without model reruns or figure regeneration.

## Reproducibility boundary

The supplied snapshot supports:

1. verification of the locked publication figures;
2. verification of the retained figure-source CSV/CIF inputs;
3. static inspection of the original analysis code and documented preprocessing behavior;
4. optional generation of review copies of plotting figures from the retained figure-level inputs.

It does **not** establish a verified end-to-end rerun of the trained models. Missing materials include the complete original raw inputs, row-level train/calibration/test manifests, all per-row prediction/interval files, the exact numerical elite threshold for the representative 0.15-bar job, original external database tables/versions used by the matcher, and the historical training lockfile/environment. See `analysis/audit_evidence/MISSING_INPUTS.md` and `docs/SCIENTIFIC_REPRODUCIBILITY_NOTES.md`.

The frozen code also contains documented issues that must remain visible, including external-identifier matching, coverage-level summary lineage, predictor semantics, and casebook annotation mapping. The repository preserves those results rather than silently refitting or rewriting the historical analysis.

## Optional figure-review path

You do **not** need to run this section merely to verify or update GitHub. It is only for regenerating non-authoritative review copies.

```bash
cd figure_generation
python check_environment.py
python run_all_figures.py
```

Generated review assets appear only under `figure_generation/outputs_review/` and are ignored by Git. They must never overwrite `figures/final/`.

## Authorship, license, and citation status

The supplied project materials contain a six-author versus seven-author metadata discrepancy. This repository therefore does not create a `CITATION.cff` until the final author list/order and citation metadata are author-approved.

The repository currently contains an MIT `LICENSE` file inherited from the pre-existing repository state. The audit package did not create that license and cannot establish whether all relevant rights holders approved it for public release. Before making the repository public or minting a DOI, the authors should explicitly confirm the intended software/data/artwork licensing scope or replace/remove the file as appropriate. See `docs/AUTHORSHIP_LICENSE_CITATION_PENDING.md`.
