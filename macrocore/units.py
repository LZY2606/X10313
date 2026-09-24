"""Domain quantities carry a unit or coordinate basis.

Imported data keeps its raw representation; normalization happens only in the
derived layer. Values that cannot be converted stay in place with a diagnostic
instead of being dropped to make a graph look complete.
"""
from __future__ import annotations

from typing import Optional, Tuple

CANONICAL_SPAN_BASIS = {
    "offset_unit": "utf8-codepoint",
    "line_base": 1,
    "col_base": 1,
    "coordinate_system": "source-text",
}

STEP = "step"
COUNT = "count"
TOKEN = "token"
SCOPE = "scope"
IDENTIFIER = "identifier"


def qty(value, unit: str, basis: Optional[dict] = None) -> dict:
    out = {"value": value, "unit": unit}
    if basis is not None:
        out["basis"] = basis
    return out


def normalize_span(raw: Optional[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """Normalize a raw span to the canonical basis.

    Returns (normalized, diagnostic). Exactly one of them is None. A missing
    basis or an offset unit that cannot be converted without the original text
    leaves the raw value in place with a diagnostic.
    """
    if raw is None:
        return None, None
    basis = raw.get("basis")
    if not basis:
        return None, {"diagnostic": "missing coordinate basis; raw span retained", "raw": raw}
    unit = basis.get("offset_unit")
    if unit != "utf8-codepoint":
        return None, {
            "diagnostic": (
                f"cannot convert offset_unit {unit!r} to 'utf8-codepoint' without source text; "
                "raw span retained"
            ),
            "raw": raw,
        }
    if basis.get("line_base") != 1 or basis.get("col_base") != 1:
        return None, {
            "diagnostic": "line/col bases other than 1 are not normalizable here; raw span retained",
            "raw": raw,
        }
    normalized = {
        "start": raw["start"],
        "end": raw["end"],
        "line": raw.get("line"),
        "col": raw.get("col"),
        "basis": dict(CANONICAL_SPAN_BASIS),
    }
    return normalized, None
