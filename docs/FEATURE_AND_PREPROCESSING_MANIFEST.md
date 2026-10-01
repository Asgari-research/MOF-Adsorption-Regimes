# Feature and preprocessing manifest

## Representative feature family

The chemistry-facing representative job uses the `geometry_topology_clusters` family with **59 supplied columns**. The machine-readable inventory is retained at:

`analysis/audit_evidence/table_si_feature_family_columns.csv`

Notable semantics:

- `arc_mof_dim__Unnamed: 0` is numeric, has 191,608 non-missing values, and is unique for all non-missing rows; it behaves like a row-index field.
- `arc_mof_dim__Dimensionality` is categorical with three observed values.
- global geometry fields and prefixed duplicate/overlapping geometry fields are both present.
- `flig_clusters__cluster_id`, `func_clusters__cluster_id`, `geo_clusters__cluster_id`, and `mc_clusters__cluster_id` are numeric in the supplied merged schema.

## Frozen preprocessing behavior

The main pipeline splits inputs by pandas dtype.

- **Random forest**: numeric = median imputation; categorical = most-frequent imputation + one-hot encoding.
- **HistGradientBoostingRegressor**: numeric = median imputation; categorical = most-frequent imputation + ordinal encoding with unknown value `-1`.
- **Ridge**: numeric = median imputation + standard scaling; categorical = most-frequent imputation + one-hot encoding.

Therefore numeric cluster codes can be interpreted as ordered numerical inputs in RF/HGB pathways rather than as nominal categories. The repository records this behavior without asserting how much it affected the saved predictions.

No corrected refit is included or implied.
