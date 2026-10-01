# Git history and future commit policy

The initial repository integration is complete. The historical two-part integration separated immutable provenance/final artwork from repository tooling/documentation. That import should not be replayed on `main`.

For future work:

1. synchronize `main` with `git pull --ff-only`;
2. create a short-lived descriptive branch;
3. make one scientifically coherent change per commit where practical;
4. run repository integrity checks before every commit that touches tracked scientific artifacts;
5. review the GitHub diff before merging;
6. avoid rewriting published/shared history unless all collaborators explicitly agree.

Recommended pre-commit checks:

```bash
python verify_final_figures.py
python scripts/verify_repository.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s analysis/proposed_repairs/tests -v
python scripts/check_forbidden_git_files.py
git diff --cached --stat
git diff --cached --name-status
```

Changes to `figures/final/`, integrity manifests, frozen analysis source, or retained scientific source data should be treated as explicit scientific revision events rather than routine repository cleanup.
