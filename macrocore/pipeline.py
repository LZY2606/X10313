"""Import + derivation pipeline.

Raw submissions are validated, stored verbatim (original representation kept),
then a derived layer normalizes quantities and runs the expander. Values that
cannot be normalized stay in place with a diagnostic; nothing is dropped.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .expand import ALGO_VERSION, Expander, ExpansionError
from .lexer import LexError, tokenize
from .parser import ParseError, parse_program
from .store import Store

NORMALIZER_VERSION = "quantity-normalizer/1.0.0"
IMPORTER_VERSION = "importer/1.1.0"

DEFAULT_PARAMS = {"budget": 8, "seed": 0}


class ImportFormatError(Exception):
    def __init__(self, diagnostics: List[dict]):
        super().__init__("import format error")
        self.diagnostics = diagnostics


# ------------------------------------------------------------ normalization
_QTY_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z_/]+)?\s*$")

_UNIT_TABLE = {
    "steps": ("count", 1.0),
    "step": ("count", 1.0),
    "count": ("count", 1.0),
    "ms": ("s", 0.001),
    "s": ("s", 1.0),
}


def normalize_quantity(raw: Any) -> dict:
    """Normalize a quantity to a canonical unit; keep raw representation."""
    entry: Dict[str, Any] = {"raw": raw, "status": "ok"}
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        entry.update(value=float(raw), unit="count", basis="unitless-count")
        return entry
    if isinstance(raw, str):
        m = _QTY_RE.match(raw)
        if m:
            value = float(m.group(1))
            unit = m.group(2) or "count"
            if unit in _UNIT_TABLE:
                canon, factor = _UNIT_TABLE[unit]
                entry.update(value=value * factor, unit=canon,
                             basis=f"normalized-from:{unit}")
                return entry
            entry.update(status="unconvertible",
                         diagnostic=f"unknown unit '{unit}'; value left in original position")
            return entry
    entry.update(status="unconvertible",
                 diagnostic="cannot parse quantity; value left in original position")
    return entry


def normalize_quantities(quantities: Dict[str, Any]) -> dict:
    return {
        "algo_version": NORMALIZER_VERSION,
        "quantities": {k: normalize_quantity(v) for k, v in quantities.items()},
    }


# ------------------------------------------------------------------- import
def parse_submission(content: str) -> Tuple[str, dict, List[dict]]:
    """Return (language, quantities, diagnostics). Raises ImportFormatError."""
    diagnostics: List[dict] = []
    text = content.strip()
    if not text:
        raise ImportFormatError([{"phase": "import", "message": "empty submission"}])
    if text.startswith("{"):
        try:
            env = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ImportFormatError([{
                "phase": "import",
                "message": f"invalid JSON envelope: {exc.msg} at line {exc.lineno} col {exc.colno}",
            }])
        if not isinstance(env, dict) or "source" not in env:
            raise ImportFormatError([{
                "phase": "import",
                "message": "JSON envelope must contain a 'source' field",
            }])
        source = env["source"]
        if not isinstance(source, str):
            raise ImportFormatError([{"phase": "import",
                                      "message": "'source' must be a string"}])
        quantities = env.get("quantities", {})
        if not isinstance(quantities, dict):
            diagnostics.append({"phase": "import",
                                "message": "'quantities' must be an object; ignored"})
            quantities = {}
        return source, quantities, diagnostics
    return content, {}, diagnostics


def import_source(store: Store, content: str, kind: str = "source",
                  label: Optional[str] = None,
                  parent_version: Optional[str] = None,
                  params: Optional[dict] = None) -> dict:
    """Validate, store raw (immutable), create version, derive expansion."""
    source, quantities, import_diags = parse_submission(content)
    try:
        tokens = tokenize(source)
        prog = parse_program(tokens)
    except (LexError, ParseError) as exc:
        raise ImportFormatError(import_diags + [exc.to_data()])

    raw = store.add_raw(kind, content)
    version = store.add_version(parent_version, kind, label or f"{kind} submission",
                                raw["id"], params=params or {})
    eff_params = dict(DEFAULT_PARAMS)
    eff_params.update(params or {})

    derived_norm = None
    if quantities:
        norm_payload = normalize_quantities(quantities)
        derived_norm = store.add_derived(
            "normalized-quantities", raw["id"], version["id"],
            NORMALIZER_VERSION, {}, [raw["id"]], norm_payload)

    expansion = compute_expansion(store, version["id"], eff_params)
    return {
        "raw_id": raw["id"],
        "raw_created": raw["created"],
        "version_id": version["id"],
        "lineage": version["lineage"],
        "staled": version["staled"],
        "tokens": len(tokens),
        "import_diagnostics": import_diags,
        "normalized_quantities": derived_norm,
        "expansion": expansion,
    }


def compute_expansion(store: Store, version_id: str, params: dict) -> dict:
    ver = store.get_version(version_id)
    raw = store.get_raw(ver["raw_id"])
    source, _, _ = parse_submission(raw["content"])
    tokens = tokenize(source)
    prog = parse_program(tokens)
    eff = dict(DEFAULT_PARAMS)
    eff.update(params or {})
    expander = Expander(budget=int(eff.get("budget", 8)), seed=int(eff.get("seed", 0)))
    result = expander.expand_program(prog)
    result["tokens"] = [t.to_data() for t in tokens]
    result["source"] = source
    der = store.add_derived(
        "expansion", raw["id"], version_id, ALGO_VERSION, eff,
        [version_id, raw["id"]], result)
    return {"derived_id": der["id"], "params_fp": der["params_fp"],
            "status": result["stats"]["status"],
            "stats": result["stats"], "event": der["event"]}


def compare_expansions(store: Store, derived_a: str, derived_b: str) -> dict:
    """Compare two derived expansions (e.g. two macro versions)."""
    import numpy as np

    a = store.get_derived(derived_a)
    b = store.get_derived(derived_b)
    pa, pb = a["payload"], b["payload"]

    def edge_keys(p: dict) -> Dict[tuple, dict]:
        out = {}
        for e in p["edges"]:
            out[(e["name"], e["status"], tuple(e["use_scopes"]))] = e
        return out

    ea, eb = edge_keys(pa), edge_keys(pb)
    only_a = sorted(set(ea) - set(eb))
    only_b = sorted(set(eb) - set(ea))
    changed = sorted(k for k in set(ea) & set(eb)
                     if ea[k].get("binding") != eb[k].get("binding"))

    names = sorted({bn["name"] for bn in pa["bindings"]} |
                   {bn["name"] for bn in pb["bindings"]})
    mat = np.zeros((len(names), 2), dtype=float)
    for i, n in enumerate(names):
        mat[i, 0] = len({tuple(bn["scopes"]) for bn in pa["bindings"] if bn["name"] == n})
        mat[i, 1] = len({tuple(bn["scopes"]) for bn in pb["bindings"] if bn["name"] == n})
    return {
        "a": {"id": a["id"], "algo_version": a["algo_version"],
              "params_fp": a["params_fp"], "status": a["status"],
              "stats": pa["stats"]},
        "b": {"id": b["id"], "algo_version": b["algo_version"],
              "params_fp": b["params_fp"], "status": b["status"],
              "stats": pb["stats"]},
        "edges_only_in_a": [list(k) for k in only_a],
        "edges_only_in_b": [list(k) for k in only_b],
        "edges_changed_binding": [list(k) for k in changed],
        "binding_scope_counts": {n: {"a": int(mat[i, 0]), "b": int(mat[i, 1])}
                                 for i, n in enumerate(names)},
        "diagnostics_delta": {"a": len(pa["diagnostics"]), "b": len(pb["diagnostics"])},
    }
