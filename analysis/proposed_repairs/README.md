# Proposed repairs — not wired into frozen results

These helpers document safe future behavior without changing the supplied model outputs.

## Identifier matching (A01)

`identifier_matching.py` demonstrates three required safeguards:

1. explicit identifier-column allowlisting rather than broad substring matching;
2. rejection of boolean/numeric pseudo-identifiers;
3. preservation of matched row, source column, and raw value so ambiguity remains auditable.

Run:

```bash
python -m unittest discover -s analysis/proposed_repairs/tests -v
```

A corrected scientific external-match reconstruction still requires the original external database tables and versions. Passing these unit tests does not convert the historical 14 saved matches into verified matches.

## Coverage summaries (A02)

A future rerun should store coverage-dependent summary metrics with level-specific names (for example `test_empirical_coverage_80`, `_90`, `_95`) rather than repeatedly overwriting unsuffixed keys. No frozen result is modified here.

## Casebook cluster annotations (A15)

No automatic repair is included because the correct source schema is absent. Future mapping must name explicit cluster-assignment columns rather than relying on broad hint matching.
