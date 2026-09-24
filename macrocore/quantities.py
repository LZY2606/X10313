"""Domain quantities carry units / coordinate bases.

Importers keep the raw representation; only this derived layer normalises.
Values that cannot be converted stay in place with a diagnostic instead of
being dropped.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

KNOWN_BASES = ("char-offset", "utf8-byte", "line-col")


def _utf8_offset_map(text: str) -> np.ndarray:
    """Map utf8 byte offsets -> char offsets for ``text``."""
    raw = text.encode("utf-8")
    table = np.zeros(len(raw) + 1, dtype=np.int64)
    byte_pos = 0
    for char_pos, ch in enumerate(text):
        width = len(ch.encode("utf-8"))
        table[byte_pos : byte_pos + width] = char_pos
        byte_pos += width
    table[len(raw)] = len(text)
    return table


def _line_col_map(text: str) -> np.ndarray:
    """(line, col) -> char offset lookup table, row per line."""
    lines = text.split("\n")
    starts = np.zeros(len(lines), dtype=np.int64)
    acc = 0
    for idx, ln in enumerate(lines):
        starts[idx] = acc
        acc += len(ln) + 1
    return starts


def normalize_offset(value: int, basis: str, text: str) -> Tuple[Optional[int], Optional[dict]]:
    """Normalise ``value`` measured in ``basis`` to a char offset.

    Returns (canonical_value, diagnostic).  Exactly one of the two is None.
    """
    if basis == "char-offset":
        if 0 <= value <= len(text):
            return value, None
        return None, {"kind": "range", "message": f"char offset {value} out of range 0..{len(text)}"}
    if basis == "utf8-byte":
        table = _utf8_offset_map(text)
        if 0 <= value < len(table):
            return int(table[value]), None
        return None, {"kind": "range", "message": f"utf8 byte offset {value} out of range"}
    return None, {
        "kind": "unknown-basis",
        "message": f"cannot convert basis {basis!r} to char-offset; value left in place",
        "raw": {"value": value, "basis": basis},
    }


def normalize_position(value: dict, text: str) -> Tuple[Optional[int], Optional[dict]]:
    """Normalise a position record {value, basis} (or {line, col}) to a char offset."""
    basis = value.get("basis")
    if basis == "line-col":
        starts = _line_col_map(text)
        line, col = value.get("line", 1), value.get("col", 1)
        if 1 <= line <= len(starts):
            return int(starts[line - 1]) + (col - 1), None
        return None, {"kind": "range", "message": f"line {line} out of range"}
    return normalize_offset(int(value.get("value", 0)), basis or "char-offset", text)
