# Final figure lock

The authoritative publication artwork consists of 12 user-supplied PDFs:

- `figures/final/main/Figure_1.pdf` through `Figure_4.pdf`
- `figures/final/si/Figure_S1.pdf` through `Figure_S8.pdf`

Their SHA-256 values are stored under `integrity/`.

## Rules

- Do not edit, recompress, optimize, redraw, recolor, rerender, or replace these PDFs during Git cleanup.
- Do not copy `figure_generation/outputs_review/` over `figures/final/`.
- Any future scientific figure change is a new explicit revision cycle and requires a new integrity manifest.
- Run `python verify_final_figures.py` before staging and after committing figure-related changes.
