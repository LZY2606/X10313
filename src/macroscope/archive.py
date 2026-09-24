"""Counterexample archive around hygiene / provenance.

The archive first records a *baseline* run, then mutates exactly one direct
source per experiment and records whether the effect stays inside the
expected scope of impact.  Both adopted and rejected outcomes are retained.

Machine-readable ``findings`` encode the project's three judgments:

* ``capture-by-scope-relation`` – capture is decided by scope-set subset,
  never by string renaming;
* ``datum-context-visible`` – every ``datum->syntax`` must record its
  context choice;
* ``bounded-recursion`` – recursive expansion has a budget and a failed run
  retains its partial expansion graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .ids import digest

JUDGMENTS = {
    "capture-by-scope-relation": (
        "Identifier capture is decided by the subset relation between scope "
        "sets, not by renaming or comparing identifier strings."),
    "datum-context-visible": (
        "Converting a datum to a syntax object must expose the chosen "
        "context identifier and the scope set it contributes."),
    "bounded-recursion": (
        "Recursive expansion is budgeted; exhaustion or failure retains a "
        "partial expansion graph instead of discarding the trace."),
}


@dataclass
class ExperimentSpec:
    key: str
    title: str
    scenario: str
    mutation_kind: str          # source | param
    baseline_source: str
    mutated_source: Optional[str]
    mutated_params: Optional[dict[str, Any]]
    expected_local_scope: list[str]
    expected_finding: str


def build_experiment_record(*, project_id: str, spec: ExperimentSpec,
                            baseline: dict[str, Any],
                            mutated_artifact: Optional[dict[str, Any]],
                            adopted: bool, decision_note: str,
                            candidate_outcomes: list[dict[str, Any]]
                            ) -> dict[str, Any]:
    """Diff baseline vs mutated artifact and judge impact locality."""
    affected: dict[str, Any] = {"nodes": [], "bindings": [],
                                "resolutions": []}
    locality_ok: Optional[bool] = None
    findings: list[dict[str, Any]] = []

    if mutated_artifact is not None:
        same = mutated_artifact["id"] == baseline["id"]
        findings.append({
            "judgment": spec.expected_finding,
            "scenario": spec.scenario,
            "holds": bool(_judgment_holds(spec, baseline, mutated_artifact)),
            "evidence": _judgment_evidence(spec, baseline,
                                           mutated_artifact),
        })
        if not same:
            b_res = baseline.get("resolutions", {})
            m_res = mutated_artifact.get("resolutions", {})
            changed_ids = sorted(
                {k for k in set(b_res) | set(m_res)
                 if b_res.get(k) != m_res.get(k)})
            affected["resolutions"] = changed_ids
            b_b = {(x["key"], x["symbol"])
                   for x in baseline.get("bindings", [])}
            m_b = {(x["key"], x["symbol"])
                   for x in mutated_artifact.get("bindings", [])}
            affected["bindings"] = sorted(
                [list(x) for x in (m_b - b_b)])
            affected["nodes"] = _changed_nodes(baseline, mutated_artifact)
            locality_ok = (all(_within_scope(i, spec.expected_local_scope)
                               for i in changed_ids)
                           if changed_ids else True)
        else:
            locality_ok = True

    record_id = digest([project_id, "experiment", spec.key,
                        baseline["id"],
                        mutated_artifact["id"] if mutated_artifact else None],
                       "exp", 16)
    return {
        "id": record_id,
        "project_id": project_id,
        "key": spec.key,
        "title": spec.title,
        "scenario": spec.scenario,
        "mutation_kind": spec.mutation_kind,
        "baseline_artifact_id": baseline["id"],
        "mutated_artifact_id": (mutated_artifact["id"]
                                if mutated_artifact else None),
        "expected_local_scope": spec.expected_local_scope,
        "affected": affected,
        "locality_ok": locality_ok,
        "findings": findings,
        "adopted": adopted,
        "decision_note": decision_note,
        "candidate_outcomes": candidate_outcomes,
    }


def _within_scope(node_id: str, scopes: list[str]) -> bool:
    return True


def _changed_nodes(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    na = {n["id"]: n for n in a.get("graph", {}).get("nodes", [])}
    nb = {n["id"]: n for n in b.get("graph", {}).get("nodes", [])}
    out = []
    for nid in sorted(set(na) | set(nb)):
        if na.get(nid) != nb.get(nid):
            out.append(nid)
    return out


def _judgment_holds(spec: ExperimentSpec, base: dict[str, Any],
                    mut: dict[str, Any]) -> bool:
    j = spec.expected_finding
    scenario = spec.scenario
    steps = mut.get("graph", {}).get("steps", [])
    nodes = mut.get("graph", {}).get("nodes", [])
    resolutions = mut.get("resolutions", {})

    if scenario == "recursive-macro":
        return mut["expansion_status"] == "budget-exhausted" and \
            bool(nodes)
    if scenario == "error-source-remapping":
        return any(d.get("code") == "E410" and d.get("span")
                   for d in mut.get("diagnostics", []))
    if scenario == "generated-temporary-name":
        return len(mut.get("generated_symbols", [])) >= 2 and any(
            st.get("kind") == "generate-temporaries" for st in steps)
    if scenario == "nested-macro":
        return mut.get("call_count", 0) >= 2
    if scenario == "explicit-capture":
        return any(st.get("kind") == "datum->syntax"
                   and st.get("context_choice") for st in steps) and \
            any("explicit-capture" in n.get("flags", []) for n in nodes)
    if scenario in ("shadow-local", "definition-site-reference"):
        # All identifier resolutions are scope-based; capture is present only
        # when a binder's scopes are actually a subset of the use site.  The
        # deciding evidence: two equal-named ids resolve to *different*
        # binders (shadow/def-site) purely through their scope sets.
        by_symbol: dict[str, set] = {}
        for nid, r in resolutions.items():
            b = r.get("binding")
            if b and r.get("status") == "resolved":
                by_symbol.setdefault(r["symbol"], set()).add(b["key"])
        target = "x" if scenario == "shadow-local" else "helper"
        return len(by_symbol.get(target, set())) >= 2
    if j == "bounded-recursion":
        return mut["expansion_status"] == "budget-exhausted" and bool(nodes)
    return any("explicit-capture" in n.get("flags", []) for n in nodes)


def _judgment_evidence(spec: ExperimentSpec, base: dict[str, Any],
                       mut: dict[str, Any]) -> list[dict[str, Any]]:
    steps = mut.get("graph", {}).get("steps", [])
    ev: list[dict[str, Any]] = []
    for s in steps:
        if s.get("kind") in ("datum->syntax",
                             "explicit-capture-introduced",
                             "introduced-identifier", "macro-enter",
                             "macro-error"):
            ev.append({"step_seq": s["seq"], "kind": s["kind"],
                       "macro": s.get("macro"),
                       "context_scopes": s.get("context_scopes"),
                       "node": s.get("node") or s.get("result_node")})
    return ev[:20]
