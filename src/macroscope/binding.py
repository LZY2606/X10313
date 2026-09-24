"""Binding resolution by *scope-set relation* (never by name renaming).

All identifiers and binders sharing one symbol are placed in a NumPy bit
matrix.  A candidate binder is eligible iff **every** scope of the binder
is present at the use site (binder scope set ⊆ use scope set).  The winner
is the most specific eligible binder (largest scope set).  Two maximal,
incomparable candidates make the result ``ambiguous``; every legal
candidate, its specificity and its rejection/selection reason is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Binding:
    key: str                  # stable binder id
    symbol: str
    kind: str                 # local | macro | core | module
    binder_scopes: frozenset[str]
    site: str                 # definition-site node id
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "symbol": self.symbol, "kind": self.kind,
                "binder_scopes": sorted(self.binder_scopes),
                "site": self.site, "detail": self.detail}


@dataclass(frozen=True)
class Candidate:
    binding: Binding
    eligible: bool
    subset: bool
    specificity: int
    missing: tuple[str, ...]
    maximal: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        d = self.binding.as_dict()
        d.update({"eligible": self.eligible, "subset": self.subset,
                  "specificity": self.specificity,
                  "missing": sorted(self.missing),
                  "maximal": self.maximal, "reason": self.reason})
        return d


@dataclass(frozen=True)
class Resolution:
    node_id: str
    symbol: str
    status: str               # resolved | free | ambiguous
    binding: Optional[Binding]
    candidates: tuple[Candidate, ...]
    scope_index: dict[str, int]
    use_scopes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id, "symbol": self.symbol,
            "status": self.status,
            "binding": self.binding.as_dict() if self.binding else None,
            "candidates": [c.as_dict() for c in self.candidates],
            "scope_index": self.scope_index,
            "use_scopes": sorted(self.use_scopes),
        }


def _bitset(scopes: frozenset[str], index: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(index), dtype=np.uint8)
    for s in scopes:
        j = index.get(s)
        if j is not None:
            v[j] = 1
    return v


def resolve(symbol: str, use_scopes: frozenset[str],
            bindings: list[Binding], node_id: str) -> Resolution:
    same = [b for b in bindings if b.symbol == symbol]
    if not same:
        return Resolution(node_id, symbol, "free", None, (), {},
                          tuple(sorted(use_scopes)))

    all_scopes = sorted(set(use_scopes) |
                        {s for b in same for s in b.binder_scopes})
    index = {s: i for i, s in enumerate(all_scopes)}
    use_vec = _bitset(use_scopes, index)

    scored: list[tuple[Binding, bool, int, frozenset[str], str]] = []
    for b in same:
        cand = _bitset(b.binder_scopes, index)
        missing_mask = (cand == 1) & (use_vec == 0)
        missing = frozenset(s for s, j in index.items() if missing_mask[j])
        subset = not missing
        specificity = int(cand.sum()) if subset else -1
        if subset:
            why = f"binder scopes {sorted(b.binder_scopes)} ⊆ use set"
        else:
            why = (f"rejected: binder scopes {sorted(missing)} absent at "
                   f"use site (capture would be unhygienic)")
        scored.append((b, subset, specificity, missing, why))

    eligible = [row for row in scored if row[1]]
    candidates: list[Candidate] = []
    status = "free"
    chosen: Optional[Binding] = None

    if eligible:
        max_sp = max(r[2] for r in eligible)
        maximals = [r for r in eligible if r[2] == max_sp]
        distinct = {r[0].key for r in maximals}
        if len(distinct) == 1:
            status = "resolved"
            chosen = maximals[0][0]
        else:
            status = "ambiguous"
        winner_keys = distinct if status == "ambiguous" else {chosen.key}
        for b, ok, sp, missing, why in scored:
            is_max = ok and sp == max_sp and b.key in winner_keys
            if status == "ambiguous" and is_max:
                why += " — tie: incomparable maximal binders, retained"
            elif is_max:
                why += " — most specific"
            candidates.append(
                Candidate(b, ok, ok, sp, tuple(sorted(missing)), is_max, why))
    else:
        for b, ok, sp, missing, why in scored:
            candidates.append(
                Candidate(b, False, False, -1, tuple(sorted(missing)),
                          False, why))

    return Resolution(node_id, symbol, status, chosen, tuple(candidates),
                      index, tuple(sorted(use_scopes)))
