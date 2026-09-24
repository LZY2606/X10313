"""Units and coordinate bases for every domain quantity.

Raw import preserves the original textual representation. Derived quantities
(normalized offsets, byte/char conversions) are attached in a separate layer.
Values that cannot be normalized stay in place with a diagnostic instead of
being dropped to make a diagram look complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Coordinate bases. Zero-based UTF-16 code units match most browser editors;
# the raw import still records the original 1-based textual coordinates if any.
CHAR_OFFSET = "char_offset:0:utf16"
LINE_COLUMN = "line:1:column:0:utf16"
BYTE_OFFSET = "byte_offset:0:utf8"

# Domain units.
EXPANSION_STEP = "macro_invocations"
SCOPE_COUNT = "scopes"
DEPTH = "frames"
SYMBOL_COUNT = "symbols"


@dataclass(frozen=True)
class Quantity:
    """A number together with its unit / coordinate basis."""

    value: Any
    unit: str
    raw: str | None = None
    basis_note: str = ""

    def to_json(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "raw": self.raw,
            "basis_note": self.basis_note,
        }

    @staticmethod
    def from_json(data: dict) -> "Quantity":
        return Quantity(
            value=data["value"],
            unit=data["unit"],
            raw=data.get("raw"),
            basis_note=data.get("basis_note", ""),
        )


@dataclass
class ConversionDiagnostic:
    """A value that could not be normalized is retained, not discarded."""

    path: str
    raw: str
    reason: str
    severity: str = "warning"

    def to_json(self) -> dict:
        return {"path": self.path, "raw": self.raw, "reason": self.reason,
                "severity": self.severity}
