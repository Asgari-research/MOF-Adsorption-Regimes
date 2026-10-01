# Data and figure-source provenance

## Byte-preserved inputs

The canonical figure-generation data under `figure_generation/data/` match the supplied evidence archive byte-for-byte for:

- 23 main-figure source CSVs;
- 10 SI-figure source CSVs;
- the complete 14-row candidate casebook CSV;
- all 14 selected CIFs.

`figure_generation/DATA_SHA256.csv` records their hashes.

## Important status distinctions

The presence of a field in a retained CSV does not mean that the field is scientifically validated.

### External identifier fields — quarantined as unverified

The original pipeline used broad identifier-column substring matching. Saved casebook rows labelled `exact_id_match` contain boolean/numeric pseudo-identifiers among the selected keys. The current external-match count is therefore not a validated exact-match count. Retain the raw evidence for auditability, but do not cite those fields as verified provenance or experimental validation.

### Casebook cluster fields — unverified mapping

Saved `geometry_cluster`, `metal_cluster`, `functional_cluster`, and `ligand_cluster` casebook fields were populated from filename columns in the supplied run. They must not be interpreted as validated cluster assignments. This issue is separate from the model-input cluster-code semantics.

### Fields used by the current review plotting path

The portable review figures use supported columns such as `display_id`, `y_true`, `y_pred`, `residual`, decision classes, saved aggregate metrics, and the supplied CIFs. The plotting code intentionally does not use the invalid external-match status as evidence.

## Missing raw reconstruction inputs

Full train/calibration/test row IDs, complete per-job prediction and residual files, original external database tables/versions, exact representative-job elite threshold, and the original training lockfile are not in the supplied materials. See `analysis/audit_evidence/MISSING_INPUTS.md`.
