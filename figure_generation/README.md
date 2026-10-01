# Figure-generation review snapshot

This directory is a **review/regeneration path**, not the authority for the publication PDFs.

The source CSVs, casebook CSV and 14 CIFs were copied byte-for-byte from the supplied project evidence. The plotting source was adapted only for portability and safe review output:

- outputs are written to `outputs_review/`;
- Arial discovery supports Ubuntu/WSL2 via `/mnt/c/Windows/Fonts/arial.ttf`;
- SI Figure S8 falls back to the supplied CIFs when the original frozen OVITO/Blender PNG masters are unavailable.

That SI S8 fallback is a review rendering only. The supplied frozen PNG masters referenced by the historical source were **not present** in the handoff, so this directory does not claim byte-for-byte regeneration of the locked final PDFs.

## Run

```bash
python check_environment.py
python run_all_figures.py
```

Or generate one review figure:

```bash
python run_one_figure.py fig1
python run_one_figure.py si8
```

The authoritative publication PDFs remain under `../figures/final/`.

## Data-status warning

Some retained CSV columns are provenance evidence for known defects. In particular, external-match annotations and several casebook cluster-derived fields are not validated scientific annotations. See `../docs/DATA_PROVENANCE.md` and `../docs/SCIENTIFIC_REPRODUCIBILITY_NOTES.md` before reusing them.
