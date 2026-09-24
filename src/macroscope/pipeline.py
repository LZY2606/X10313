"""Pure derivation pipeline: import -> read -> expand -> view payload.

Every derived artifact carries:

* ``algorithm_version`` – code version that produced it
* ``params_fingerprint`` – the exact parameters used
* ``direct_sources``     – stable ids of immediate inputs
* ``virtual_time``       – deterministic clock tick
* ``status``             – current | stale (stale results stay viewable)

Raw inputs are immutable; revisions are new versions linked by
``parent_version_id``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .expander import ExpanderConfig, expand
from .ids import VirtualClock, digest
from .reader import read_program
from .syntax import GraphBuilder


ALGORITHM_VERSION = "expander-1.0.0"
PIPELINE_VERSION = "pipeline-1.0.0"


@dataclass
class ImportResult:
    input_id: str
    tokens: list[dict[str, Any]]
    read_diagnostics: list[dict[str, Any]]
    root_id: str


def import_source(*, project_id: str, text: str, parent_version_id: Optional[str],
                  label: str, clock: VirtualClock,
                  seq_index: int) -> dict[str, Any]:
    """Tokenise + read without expanding (import layer preserves raw form)."""
    base = digest([project_id, "input", seq_index, text, label,
                   parent_version_id], "in", 20)
    version_id = f"{project_id}-{base}"
    run_id = f"{version_id}-read"
    builder = GraphBuilder(run_id)
    parsed = read_program(text, builder, file=label)
    payload = {
        "id": version_id,
        "project_id": project_id,
        "seq_index": seq_index,
        "label": label,
        "parent_version_id": parent_version_id,
        "raw_text": text,
        "raw_representation": {
            "encoding": "utf-8",
            "line_ending": "preserved",
            "coordinate_basis": "1-based line/column, 0-based character offset",
        },
        "tokens": [t.as_dict() for t in parsed["tokens"]],
        "diagnostics": [d.as_dict() for d in parsed["diagnostics"]],
        "read_root_id": parsed["root"].id,
        "read_graph": builder.graph_dict(),
        "created_vt": clock.now(),
        "immutable": True,
    }
    return payload


def run_expansion(*, project_id: str, input_version: dict[str, Any],
                  params: dict[str, Any], clock: VirtualClock,
                  attempt_seq: int) -> dict[str, Any]:
    """Replay tokenise -> read -> expand deterministically from raw text."""
    config = ExpanderConfig(
        macro_budget=int(params.get("macro_budget", 64)),
        seed=int(params.get("seed", 20260925)))
    params_fp = digest(config.fingerprint(), "pfp", 12)
    run_id = digest([input_version["id"], params_fp, attempt_seq], "run", 16)

    builder = GraphBuilder(run_id)
    parsed = read_program(input_version["raw_text"], builder,
                          file=input_version["label"])
    result = expand(parsed["root"], config, input_version["id"], builder)

    artifact_id = digest([run_id, "artifact"], "art", 16)
    graph = builder.graph_dict()
    payload = {
        "id": artifact_id,
        "run_id": run_id,
        "project_id": project_id,
        "input_version_id": input_version["id"],
        "input_seq_index": input_version["seq_index"],
        "direct_sources": [input_version["id"]],
        "algorithm_version": ALGORITHM_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "params_fingerprint": params_fp,
        "params": config.fingerprint(),
        "attempt_seq": attempt_seq,
        "created_vt": clock.now(),
        "status": "current",
        "expansion_status": result.status,
        "call_count": result.call_count,
        "macro_budget": result.budget,
        "tokens": [t.as_dict() for t in parsed["tokens"]],
        "reader_diagnostics": [d.as_dict() for d in parsed["diagnostics"]],
        "diagnostics": [d.as_dict() for d in result.diagnostics],
        "bindings": [b.as_dict() for b in result.bindings],
        "resolutions": result.resolutions,
        "generated_symbols": result.generated_symbols,
        "expanded_root_id": result.root.id if result.root else None,
        "graph": graph,
    }
    return payload
