import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve()
REPAIR_DIR = HERE.parents[1]
if str(REPAIR_DIR) not in sys.path:
    sys.path.insert(0, str(REPAIR_DIR))

from identifier_matching import (  # noqa: E402
    allowed_identifier_columns,
    build_identifier_index,
    lookup,
    normalize_identifier,
)


class IdentifierMatchingTests(unittest.TestCase):
    def test_rejects_boolean_and_numeric_pseudo_identifiers(self):
        for value in [True, False, "true", "FALSE", 0.0794, "0.0794", 12, "12"]:
            self.assertIsNone(normalize_identifier(value), value)

    def test_accepts_real_textual_identifiers(self):
        self.assertEqual(normalize_identifier("OSAVUA"), "osavua")
        self.assertEqual(normalize_identifier("VAGTAA01.cif"), "vagtaa01")
        self.assertEqual(normalize_identifier("DB12-VAGTAA01_clean"), "db12-vagtaa01_clean")

    def test_column_selection_is_allowlist_not_id_substring(self):
        cols = ["refcode", "mofid", "void", "valid", "candidate_score", "database_code"]
        self.assertEqual(allowed_identifier_columns(cols), ["refcode", "mofid", "database_code"])

    def test_provenance_is_preserved_and_ambiguity_not_hidden(self):
        rows = [
            {"refcode": "OSAVUA", "valid": True},
            {"refcode": "OSAVUA", "valid": False},
            {"refcode": "VAGTAA01", "valid": True},
        ]
        idx = build_identifier_index(rows, rows[0].keys())
        hits = lookup(idx, "OSAVUA")
        self.assertEqual(len(hits), 2)
        self.assertEqual({h.row_index for h in hits}, {0, 1})
        self.assertTrue(all(h.column == "refcode" for h in hits))


if __name__ == "__main__":
    unittest.main()
