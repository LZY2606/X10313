"""Domain quantities with units / coordinate bases.

Raw representations are preserved at import; only the derived layer
normalizes.  Values that cannot be converted stay in place and carry a
diagnostic - nothing is dropped for the sake of drawing a chart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ALGORITHM_VERSION = "units/1.0.0"


@dataclass
class Quantity:
    value: Any
    unit: str            # e.g. "byte", "char", "utf16-code-unit", "line", "visual-column"
    basis: str           # coordinate basis, e.g. "utf-8", "origin-0", "origin-1"

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "basis": self.basis}


@dataclass
class Diagnostic:
    code: str
    message: str
    quantity: dict

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "quantity": self.quantity}


@dataclass
class Normalized:
    quantity: Quantity
    algorithm_version: str = ALGORITHM_VERSION
    diagnostics: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity.to_dict(),
            "algorithm_version": self.algorithm_version,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# conversion table: (from_unit, to_unit) -> factor; basis must be compatible
_LINEAR = {
    ("byte", "char"): ("utf-8", None),      # needs re-decode, handled specially
    ("utf16-code-unit", "char"): ("utf-16", None),
}


def normalize(quantity: Quantity, target_unit: str,
              source_text: Optional[str] = None) -> Normalized:
    """Normalize a quantity to the target unit.

    Unconvertible values are returned in their original position with a
    diagnostic attached (never silently dropped).
    """
    if quantity.unit == target_unit:
        return Normalized(quantity=quantity)
    key = (quantity.unit, target_unit)
    if key in _LINEAR and source_text is not None:
        encoding, _ = _LINEAR[key]
        raw = source_text.encode(encoding)
        if quantity.unit == "byte":
            prefix = raw[: int(quantity.value)]
            converted = len(prefix.decode(encoding, errors="strict"))
        else:  # utf16 code units -> chars
            units = int(quantity.value)
            converted = len(raw[: units * 2].decode(encoding, errors="ignore"))
        return Normalized(quantity=Quantity(converted, target_unit, "origin-0"))
    diag = Diagnostic(
        code="unconvertible-unit",
        message=f"cannot convert {quantity.unit}/{quantity.basis} to {target_unit}; "
                f"value left in original position",
        quantity=quantity.to_dict(),
    )
    return Normalized(quantity=quantity, diagnostics=[diag])
