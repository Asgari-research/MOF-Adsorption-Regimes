# Recommended commit plan

For a clean history, use two commits rather than one large undifferentiated commit.

## Commit 1 — immutable provenance and final artwork

Stage:

```bash
git add figures/final integrity verify_final_figures.py analysis/frozen_original analysis/audit_evidence
```

Suggested message:

```text
Preserve locked figures and frozen analysis snapshot
```

## Commit 2 — portable review tooling and documentation

Stage:

```bash
git add README.md .gitignore figure_generation analysis/proposed_repairs docs scripts
```

Suggested message:

```text
Add portable WSL2 review tooling and reproducibility notes
```

Before each commit, run:

```bash
python verify_final_figures.py
python scripts/verify_repository.py
git diff --cached --stat
git diff --cached --name-status
```
