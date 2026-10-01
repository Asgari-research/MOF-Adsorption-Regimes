# Scientific and reproducibility notes

This file records code/repository limitations that must remain visible until resolved. It is not a request to rerun the models.

## A01 — external identifier matching

**Status: confirmed defect; current exact-match annotations are unverified.**

The frozen pipeline builds identifier columns using broad substring hints including `id`. Candidate match keys consequently admit fields that are not identifiers, including boolean/numeric values. All 14 saved casebook entries labelled `exact_id_match` contain suspicious keys such as `false`, `true`, or numeric values. The current automated count cannot be treated as verified structure identity.

A standalone stricter reference implementation is provided under `analysis/proposed_repairs/identifier_matching.py`. It is not wired into the frozen pipeline because a corrected scientific reconstruction also needs the original external tables/versions.

## A02 — mixed coverage-level summary keys

**Status: confirmed.**

The frozen `run_single_job` loops over 0.80, 0.90 and 0.95 coverage but writes unsuffixed summary keys repeatedly. The surviving aggregate summary keys therefore describe the final 0.95 pass. Saved figure summaries based on those keys must be labelled as 95% where supported. The current review plotting code makes the documented display correction for Figure 4A and labels widths/coverage as 95%.

Do not globally change every 90% quantity to 95%: candidate flags and several decision panels explicitly use the 90% calculations.

## A03 — empty-selection false-promotion rate

**Status: confirmed.**

The adaptive gate selects zero candidates. A 0/0 false-promotion fraction is undefined, not zero. Review Figure 3 displays this as `N/A (no selected candidates)`.

## A04 — shortlist overlap

**Status: confirmed interpretation limit.**

Jaccard values compare top sets across split realizations whose eligible held-out populations change. Interpret them as shortlist overlap across split realizations, not as pure fixed-pool ranking stability. The code prefers the adaptive 90% lower-bound column when present.

## A05 — predictor semantics

**Status: inclusion confirmed; influence not quantified.**

The supplied 59-column `geometry_topology_clusters` family includes `arc_mof_dim__Unnamed: 0`, a numerical row-index-like field, plus cluster IDs represented as numeric codes. The frozen preprocessing is dtype-driven. For random forest, numerical columns receive median imputation while categoricals are one-hot encoded; for HGB, numeric columns receive median imputation while categoricals receive ordinal encoding; ridge additionally standardizes numeric columns.

This establishes inclusion/semantics, not target leakage or effect size. Do not silently remove/recode these inputs and present the resulting model as the same analysis.

## A06 — representative job and missing elite threshold

The full 14-case casebook identifies the representative chemistry-facing job as:

`co2_0p15_uptake_mmol_g_co2_at_0_15_bar_geometry_topology_clusters_rf_topology_grouped_seed43`

The test population is 7,306 structures. Recovered one-sided/two-sided calibration thresholds are preserved in `analysis/audit_evidence/recovered_constants.json`. The exact numerical elite uptake threshold itself is not present in the supplied snapshot; only its definition as the 95th percentile of training+calibration labels is known.

The representative-file selector also uses test empirical coverage proximity as one preference, so the example is selected rather than an independent confirmatory cohort.

## A09 — post-hoc PLD filter mapping

The frozen filter search accepts any column whose slug contains `pld`, `di`, or `limiting_diameter`. The token `di` is overly broad and may select unintended columns. The saved empty PLD subset must not be interpreted as a physical absence of valid pore sizes without the original dataframe/schema.

## A13 — frozen-code execution hazards

- Original requirements are unpinned and do not reconstruct the historical training environment.
- Publication builder scripts contain `/mnt/data/...` hard-coded paths.
- `prepare_final_publication_figures_tables_full.py` performs top-level deletion/recreation of output paths and must not be imported for inspection.
- Five top-level function names occur twice in the main pipeline; later definitions override earlier ones.
- Full end-to-end reproduction is not established by syntax validity alone.

## A15 — casebook cluster annotation mapping

The saved casebook `geometry_cluster`, `metal_cluster`, `functional_cluster`, and `ligand_cluster` fields map to filenames rather than validated cluster assignments. Derived mechanistic labels that depend on those fields are therefore unverified. This does not prove that the fitted model itself used filenames.

## Release boundary

Before public release/DOI deposition, resolve or clearly qualify the above issues, the exact elite threshold, cohort row-ID lineage, final author list, license, citation metadata, and missing reproduction inputs. The private Git repository can still preserve the code, data and locked figures now as long as those boundaries remain explicit.
