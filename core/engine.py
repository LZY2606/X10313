"""End-to-end engine pipeline: source -> tokens -> scopes -> expansion -> bindings."""
from __future__ import annotations

from . import lexer
from .build import build
from .expander import Expander
from .resolver import BindingResolver

ALGO_VERSION = "scope-mirror-1.0.0"


def parameter_fingerprint(params: dict) -> str:
    import hashlib
    import json
    body = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return "fp_" + hashlib.sha256(body.encode()).hexdigest()[:16]


def analyze(source: str, budget: int = 64, seed: str = "run") -> dict:
    tokens = lexer.tokenize(source)
    raw_forms, parse_diagnostics = lexer.parse(tokens)
    built = build(raw_forms, tokens, seed=seed)
    expander = Expander(built, budget=budget, seed=seed,
                        algo_version=ALGO_VERSION)
    expansion = expander.run()
    resolver = BindingResolver(built, expansion, seed=seed)
    binding_edges = resolver.resolve_all()
    return {
        "algo_version": ALGO_VERSION,
        "parameter_fingerprint": parameter_fingerprint({"budget": budget}),
        "tokens": [t.to_json() for t in tokens],
        "parse_diagnostics": parse_diagnostics,
        "expansion": expansion.to_json(),
        "binding_edges": binding_edges,
        "scopes": expansion.scopes,
    }
