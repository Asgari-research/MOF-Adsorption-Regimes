# Git include/exclude policy

## Include

- `figures/final/main/*.pdf`
- `figures/final/si/*.pdf`
- `integrity/*`
- `figure_generation/src/*.py`
- `figure_generation/data/**`
- `figure_generation/*.py`, environment files and source-data manifests
- `analysis/frozen_original/**`
- `analysis/proposed_repairs/**`
- compact audit evidence under `analysis/audit_evidence/`
- `docs/**`
- repository verification scripts

## Exclude

- regenerated `figure_generation/outputs_review/`
- old/rejected figure variants and preview sweeps
- Blender/OVITO exploratory outputs not needed to reproduce provenance
- handoff ZIPs
- caches and bytecode
- local Conda/venv directories
- model checkpoints/predictions/results that are not deliberately curated for deposition
- debug logs and temporary files

Do not use `git add .` until `git status --short` has been reviewed.
