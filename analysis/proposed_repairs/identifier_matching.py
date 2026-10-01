"""Reference-only stricter identifier matching helpers.

This module is intentionally NOT imported by the frozen analysis pipeline.  It
exists to document a safer replacement design for audit issue A01 while the
original external tables/versions remain unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_IDENTIFIER_COLUMNS = {
    "filename",
    "file_name",
    "name",
    "structure_name",
    "structure_id",
    "mof",
    "mofid",
    "mof_id",
    "mofkey",
    "mof_key",
    "refcode",
    "csd_refcode",
    "database_code",
    "cif",
    "cif_name",
}

_EXT_RE = re.compile(r"\.(?:cif|json|txt|csv)$", flags=re.IGNORECASE)
_NUMERIC_ONLY_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class MatchEvidence:
    row_index: int
    column: str
    raw_value: str


def canonical_column_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def allowed_identifier_columns(columns: Iterable[Any], allowlist: Iterable[str] = DEFAULT_IDENTIFIER_COLUMNS) -> list[str]:
    """Select only explicitly permitted identifier fields.

    Deliberately avoids substring rules such as ``"id" in column_name``.
    """
    allowed = {canonical_column_name(x) for x in allowlist}
    return [str(c) for c in columns if canonical_column_name(c) in allowed]


def normalize_identifier(value: Any) -> str | None:
    """Normalize a genuine textual identifier and reject pseudo-identifiers.

    Pure booleans, boolean-like strings, numeric scalars, and purely numeric
    strings are rejected.  Alphanumeric refcodes such as ``VAGTAA01`` remain
    valid.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return None

    text = str(value).strip().strip("'\"")
    if not text:
        return None
    low = text.lower()
    if low in {"nan", "none", "null", "true", "false", "yes", "no"}:
        return None
    if _NUMERIC_ONLY_RE.fullmatch(text):
        return None

    text = _EXT_RE.sub("", text).strip().lower()
    if not text:
        return None
    return text


def build_identifier_index(rows: Sequence[Mapping[str, Any]], columns: Iterable[str]) -> dict[str, list[MatchEvidence]]:
    """Build an auditable identifier index preserving row/column/raw provenance."""
    allowed = allowed_identifier_columns(columns)
    index: dict[str, list[MatchEvidence]] = {}
    for i, row in enumerate(rows):
        for column in allowed:
            if column not in row:
                continue
            raw = row[column]
            key = normalize_identifier(raw)
            if key is None:
                continue
            index.setdefault(key, []).append(MatchEvidence(i, column, str(raw)))
    return index


def lookup(index: Mapping[str, Sequence[MatchEvidence]], candidate: Any) -> list[MatchEvidence]:
    key = normalize_identifier(candidate)
    if key is None:
        return []
    return list(index.get(key, ()))
