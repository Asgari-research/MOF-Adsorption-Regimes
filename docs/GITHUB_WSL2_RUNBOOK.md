# Ubuntu/WSL2 GitHub runbook

Target private repository:

`https://github.com/Asgari-research/MOF-Adsorption-Regimes`

The package was prepared without remote access to that private repository. The steps below are intentionally inspect-first and do not delete unknown repository files.

## Phase 1 — inspect your existing clone

From Ubuntu/WSL2:

```bash
cd /path/to/your/MOF-Adsorption-Regimes
git rev-parse --show-toplevel
git remote -v
git branch --show-current
git status --short
```

Confirm that the remote corresponds to `Asgari-research/MOF-Adsorption-Regimes` and that you understand any existing local changes before continuing.

If the working tree is clean, update without creating a merge commit:

```bash
git pull --ff-only
```

Create a dedicated branch:

```bash
git switch -c github-repository-cleanup-2026-10-01
```

## Phase 2 — preview the overlay

Extract this handoff package outside the Git clone. Then run from the handoff directory:

```bash
bash tools/preflight_existing_repo.sh /path/to/your/MOF-Adsorption-Regimes
bash tools/preview_overlay.sh /path/to/your/MOF-Adsorption-Regimes
```

Read the preview. Nothing is copied by those commands.

When satisfied:

```bash
bash tools/apply_overlay.sh /path/to/your/MOF-Adsorption-Regimes
```

The apply script uses `rsync` **without `--delete`**, so it does not remove unrelated repository files.

## Phase 3 — verify inside the repository

```bash
cd /path/to/your/MOF-Adsorption-Regimes
python verify_final_figures.py
python scripts/verify_repository.py
```

Create the plotting environment with Miniforge/Conda:

```bash
conda env create -f figure_generation/environment.yml -n mof-regimes-figures
conda activate mof-regimes-figures
python figure_generation/check_environment.py
```

If the environment already exists:

```bash
conda activate mof-regimes-figures
conda env update -f figure_generation/environment.yml --prune
```

On a standard WSL2 installation with the Windows C: drive mounted, the plotting code checks `/mnt/c/Windows/Fonts/arial.ttf` and registers Arial directly with Matplotlib. If that file does not exist, plotting falls back to DejaVu Sans and prints a warning.

Run the safe review plotting path:

```bash
cd figure_generation
python run_all_figures.py
cd ..
```

Review outputs are created under `figure_generation/outputs_review/` and are ignored by Git. They must not replace the locked PDFs.

Run the focused proposed-repair tests:

```bash
python -m unittest discover -s analysis/proposed_repairs/tests -v
```

## Phase 4 — inspect, stage, commit, push

First inspect everything:

```bash
git status --short
git diff --stat
git diff
```

Then stage deliberately:

```bash
git add README.md .gitignore \
  figures/final integrity verify_final_figures.py \
  figure_generation analysis docs scripts
```

Review the staged patch:

```bash
git diff --cached --stat
git diff --cached --name-status
```

Verify the locked figures one more time:

```bash
python verify_final_figures.py
python scripts/verify_repository.py
```

Commit:

```bash
git commit -m "Curate reproducible figure and analysis repository snapshot"
```

Push the branch to the private GitHub remote:

```bash
git push -u origin github-repository-cleanup-2026-10-01
```

Do not merge to the default branch until you have reviewed the GitHub diff and resolved any collision with files already present in the private repository.
