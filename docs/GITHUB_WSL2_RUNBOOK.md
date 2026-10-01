# Ubuntu/WSL2 repository workflow

This document describes the **current post-integration workflow** for the private Git repository. The historical overlay/import step is complete and does not need to be repeated.

## 1. Synchronize local `main`

From Ubuntu/WSL2:

```bash
cd ~/work/MOF_Adsorption_Regimes_GitHub/repository/MOF-Adsorption-Regimes
git fetch --prune
git switch main
git pull --ff-only
git status --short
git branch -vv
git log --oneline --decorate --graph -10
```

A clean `git status --short` should print nothing.

If a previously used local integration branch still exists after its remote branch was deleted, first compare it with `main`:

```bash
git diff --stat main..github-repository-complete-2026-10-01
git diff --stat github-repository-complete-2026-10-01..main
```

If both commands show no content difference, the local branch can be removed:

```bash
git branch -D github-repository-complete-2026-10-01
```

Do not force-delete a branch when those comparisons show unmerged content.

## 2. Verify repository integrity

No plotting environment or model rerun is required:

```bash
python verify_final_figures.py
python scripts/verify_repository.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s analysis/proposed_repairs/tests -v
python scripts/check_forbidden_git_files.py
```

The integrity checks are deliberately separate from the optional plotting environment.

## 3. Optional review plotting only

Only if review figures are intentionally needed:

```bash
conda env create -f figure_generation/environment.yml -n mof-regimes-figures
conda activate mof-regimes-figures
python figure_generation/check_environment.py
cd figure_generation
python run_all_figures.py
cd ..
```

Review outputs are written to `figure_generation/outputs_review/`, are ignored by Git, and are not authoritative publication figures.

## 4. Future repository changes

For future edits, start from synchronized `main` and use a short-lived branch:

```bash
git switch main
git pull --ff-only
git switch -c <descriptive-branch-name>
```

Before committing:

```bash
git status --short
git diff --stat
git diff
python verify_final_figures.py
python scripts/verify_repository.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s analysis/proposed_repairs/tests -v
python scripts/check_forbidden_git_files.py
```

Stage deliberately; do not use `git add .` without first reviewing `git status --short`.

After the change is committed and pushed, review the GitHub diff before merging to `main`.

## 5. Protected scientific artifacts

Repository maintenance must not silently alter:

- `figures/final/**`;
- files tracked by `integrity/FINAL_FIGURE_MANIFEST.csv`;
- byte-preserved files under `analysis/frozen_original/`;
- retained source-data files without an explicit scientific revision and corresponding provenance update.
