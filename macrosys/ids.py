"""Stable identities, fingerprints, algorithm versions and unit bases.

Every displayed value in the system carries a stable identity derived from
content hashes, so export -> delete -> re-import preserves identity, order
and provenance relations.
"""
from __future__ import annotations

import hashlib
import json

ALGO_LEX = "lex.v1"
ALGO_PARSE = "parse.v1"
ALGO_EXPAND = "expand.v1"
ALGO_BIND = "bind.v1"
ALGO_CANDIDATE = "candidate.v1"
ALGO_NORMALIZE = "normalize.v1"

SCHEMA_VERSION = "schema.v1"

# Domain unit / coordinate bases. Raw representations are kept on import;
# normalization happens only in the derived layer.
UNIT_BASES = {
    "span.offset": "utf8-codepoint",
    "span.line": "1-based",
    "span.col": "1-based",
    "time": "logical-tick",
    "rng": "numpy-PCG64-seeded",
    "graph.layout": "svg-user-unit",
}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(canonical_json(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def sid(*parts, n: int = 16) -> str:
    """Stable short id."""
    return content_hash(*parts)[:n]


def params_fingerprint(params: dict) -> str:
    return content_hash({"algo": ALGO_EXPAND, "params": params})
