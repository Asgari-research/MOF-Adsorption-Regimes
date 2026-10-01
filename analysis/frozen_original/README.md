# Frozen original analysis snapshot

This directory preserves the supplied original code snapshot byte-for-byte. Do **not** edit these files in place.

The main script is:

`src/conformal_mof_anomaly_screening_pipeline_paperB_v5_highimpact_qc.py`

The two publication-packaging scripts contain environment-specific `/mnt/data/...` paths; one performs top-level deletion/ZIP recreation and must not be imported casually. The frozen main analysis also contains known static issues documented in `../../docs/SCIENTIFIC_REPRODUCIBILITY_NOTES.md`.

No model training or rerun is required to verify this repository package. Full reproduction is not claimed because required raw inputs and the original training environment are incomplete.

Files named `README_ORIGINAL.md`, `AUTHORS_ORIGINAL.md`, and `requirements_original.txt` are preserved metadata from the supplied code archive. `AUTHORS_ORIGINAL.md` lists seven authors, while the active manuscript lists six; the discrepancy is intentionally unresolved here.
