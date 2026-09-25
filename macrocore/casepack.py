"""Offline case-pack runner for the macro-scope mirror.

Executes the declarative scenario in cases/macro_scope_mirror.json against a
Store, and records every judgment basis (capture-by-scope, datum->syntax
visibility, budget partial graph, ...) as machine-readable case-check events.
The replay log lets a reviewer trace from final state back to the material,
rule version and decision that caused each difference.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .pipeline import import_source
from .store import Store

PACK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cases", "macro_scope_mirror.json")


def load_case_pack(path: Optional[str] = None) -> dict:
    with open(path or PACK_PATH, "r", encoding="utf-8") as fh:
        pack = json.load(fh)
    if pack.get("format") != "macro-scope-mirror/case/v1":
        raise ValueError("unsupported case pack format")
    return pack


class CaseRunner:
    def __init__(self, store: Store, pack: dict):
        self.store = store
        self.pack = pack
        self.fragments: Dict[str, dict] = pack["fragments"]
        self.imported: Dict[str, dict] = {}
        self.replay: List[dict] = []
        self.last_resolution_event: Optional[str] = None
        self.last_conflict_target: Optional[str] = None

    def _fragment_source(self, name: str) -> str:
        frag = self.fragments[name]
        if "envelope" in frag:
            return json.dumps(frag["envelope"], ensure_ascii=False)
        return frag["source"]

    def _expansion(self, fragment: str) -> dict:
        version_id = self.imported[fragment]["version_id"]
        for r in self.store.list_derived(kind="expansion", include_stale=True):
            if r["subject_version"] == version_id:
                return r
        raise KeyError(f"no expansion derived for {fragment}@{version_id}")

    def _record(self, check_id: str, passed: bool, evidence: dict) -> None:
        evt = self.store.add_event("case-check", {
            "check_id": check_id,
            "passed": passed,
            "evidence": evidence,
            "rule_refs": [
                "capture-decided-by-scope-relation",
                "datum->syntax-context-visible",
                "recursive-expansion-budget-partial-graph",
            ],
        })
        self.replay.append({"op": "check", "check_id": check_id, "passed": passed,
                            "event": evt["id"], "evidence": evidence})

    def run(self) -> dict:
        self.store.register_case_pack(self.pack["pack_id"], self.pack["title"])
        for step in self.pack["scenario"]:
            op = step["op"]
            if op == "import":
                self._op_import(step)
            elif op == "revise-params":
                self._op_revise_params(step)
            elif op == "revise-source":
                self._op_revise_source(step)
            elif op == "decide":
                self._op_decide(step)
            elif op == "resolve":
                self._op_resolve(step)
            elif op == "undo-resolution":
                self._op_undo_resolution(step)
            elif op == "check":
                self._op_check(step)
            else:
                raise ValueError(f"unknown scenario op {op}")
        checks = [r for r in self.replay if r["op"] == "check"]
        return {
            "pack_id": self.pack["pack_id"],
            "title": self.pack["title"],
            "steps": len(self.pack["scenario"]),
            "checks_total": len(checks),
            "checks_passed": sum(1 for c in checks if c["passed"]),
            "replay": self.replay,
            "imported": {k: {"version_id": v["version_id"], "raw_id": v["raw_id"],
                             "derived_id": v["expansion"]["derived_id"]}
                         for k, v in self.imported.items()},
        }

    # ------------------------------------------------------------------ ops
    def _op_import(self, step: dict) -> None:
        frag = self.fragments[step["fragment"]]
        params = dict(frag.get("params", {}))
        params.update(step.get("params", {}))
        result = import_source(self.store, self._fragment_source(step["fragment"]),
                               kind="case-fragment", label=step["label"], params=params)
        self.imported[step["fragment"]] = result
        self.replay.append({"op": "import", "fragment": step["fragment"],
                            "version_id": result["version_id"],
                            "raw_id": result["raw_id"],
                            "derived_id": result["expansion"]["derived_id"],
                            "status": result["expansion"]["status"]})

    def _op_revise_params(self, step: dict) -> None:
        prev = self.imported[step["fragment"]]
        result = import_source(self.store, self._fragment_source(step["fragment"]),
                               kind="case-fragment", label=step["label"],
                               parent_version=prev["version_id"],
                               params=step["params"])
        self.imported[step["fragment"]] = result
        self.replay.append({"op": "revise-params", "fragment": step["fragment"],
                            "parent": prev["version_id"],
                            "version_id": result["version_id"],
                            "staled": result["staled"],
                            "derived_id": result["expansion"]["derived_id"]})

    def _op_revise_source(self, step: dict) -> None:
        prev = self.imported[step["fragment"]]
        result = import_source(self.store, step["source"], kind="case-fragment",
                               label=step["label"],
                               parent_version=prev["version_id"])
        self.imported[step["fragment"]] = result
        self.replay.append({"op": "revise-source", "fragment": step["fragment"],
                            "parent": prev["version_id"],
                            "version_id": result["version_id"],
                            "staled": result["staled"],
                            "derived_id": result["expansion"]["derived_id"]})

    def _edge_target(self, fragment: str, edge_name: str) -> str:
        der = self._expansion(fragment)
        for e in der["payload"]["edges"]:
            if e["name"] == edge_name:
                return e["id"]
        raise KeyError(f"no edge named {edge_name} in {fragment}")

    def _op_decide(self, step: dict) -> None:
        target = self._edge_target(step["fragment"], step["edge_name"])
        der = self._expansion(step["fragment"])
        evt = self.store.add_decision(target, step["verdict"], step["rationale"],
                                      step["reviewer"], derived_id=der["id"])
        self.last_conflict_target = target
        self.replay.append({"op": "decide", "target": target,
                            "verdict": step["verdict"], "event": evt["id"]})

    def _op_resolve(self, step: dict) -> None:
        target = self._edge_target(step["fragment"], step["edge_name"])
        der = self._expansion(step["fragment"])
        evt = self.store.resolve_conflict(target, step["verdict"], step["rationale"],
                                          step["reviewer"], derived_id=der["id"])
        self.last_resolution_event = evt["id"]
        self.replay.append({"op": "resolve", "target": target,
                            "verdict": step["verdict"], "event": evt["id"]})

    def _op_undo_resolution(self, step: dict) -> None:
        if not self.last_resolution_event:
            raise ValueError("no resolution to undo")
        evt = self.store.undo_event(self.last_resolution_event, step["reviewer"],
                                    step.get("note", ""))
        self.replay.append({"op": "undo", "target_event": self.last_resolution_event,
                            "event": evt["id"]})

    def _op_check(self, step: dict) -> None:
        spec = self.pack["checks"][step["id"]]
        passed, evidence = self._run_check(spec)
        self._record(step["id"], passed, evidence)

    # --------------------------------------------------------------- checks
    def _run_check(self, spec: dict):
        kind = spec["kind"]
        handler = getattr(self, "_check_" + kind.replace("-", "_"), None)
        if handler is None:
            return False, {"error": f"unknown check kind {kind}"}
        return handler(spec)

    def _check_baseline_preserved(self, spec: dict):
        """原 token、span、scope set、def/use site、宏调用栈、生成符号、转换步骤均保留。"""
        missing = []
        for name, imp in self.imported.items():
            der = self._expansion(name)
            p = der["payload"]
            if not p.get("tokens"):
                missing.append(f"{name}:tokens")
            if not any(n.get("span") for n in p["nodes"]):
                missing.append(f"{name}:spans")
            if not p.get("transform_steps") and name not in ("ambiguous-candidates",
                                                             "quantities-probe"):
                missing.append(f"{name}:transform_steps")
            for t in p.get("transform_steps", []):
                if not t.get("intro_scope") or not t.get("use_scope"):
                    missing.append(f"{name}:scopes@{t['step']}")
                if t.get("macro_def_span") is None:
                    missing.append(f"{name}:def_site@{t['step']}")
            for sm in p.get("source_map", []):
                if "call_stack" not in sm:
                    missing.append(f"{name}:call_stack")
        return (not missing), {"missing": missing,
                               "fragments": sorted(self.imported.keys())}

    def _check_edge(self, spec: dict):
        der = self._expansion(spec["fragment"])
        edges = [e for e in der["payload"]["edges"] if e["name"] == spec["name"]]
        if not edges:
            return False, {"error": "edge not found"}
        e = edges[-1]
        ok = e["status"] == spec["expect_status"]
        binding = next((b for b in der["payload"]["bindings"]
                        if b["id"] == e.get("binding")), None)
        evidence = {"edge_id": e["id"], "status": e["status"], "binding": binding,
                    "rule": e["rule"]}
        if binding:
            ok = ok and binding["kind"] == spec["expect_binding_kind"]
            ok = ok and binding["step"] == spec["expect_binding_step"]
        if spec.get("expect_invisible_candidate"):
            invis = [c for c in e["candidates"] if not c["visible"]]
            ok = ok and len(invis) >= 1
            evidence["invisible_candidates"] = invis
        return ok, evidence

    def _check_nested_stack(self, spec: dict):
        der = self._expansion(spec["fragment"])
        steps = der["payload"]["transform_steps"]
        macros = [s["macro"] for s in steps]
        ok = macros == spec["expect_macros"]
        nodes = der["payload"]["nodes"]
        inner_if = [n for n in nodes if n["kind"] == "core:if"]
        ok = ok and inner_if and inner_if[0]["call_stack"] == spec["expect_macros"]
        return ok, {"transform_macros": macros,
                    "if_call_stack": inner_if[0]["call_stack"] if inner_if else None}

    def _check_definition_site(self, spec: dict):
        der = self._expansion(spec["fragment"])
        p = der["payload"]
        edges = [e for e in p["edges"] if e["name"] == spec["name"]]
        template_edge = next((e for e in edges
                              if str(e["use_origin"]).startswith("mac:")), None)
        arg_edge = next((e for e in edges
                         if not str(e["use_origin"]).startswith("mac:")), None)
        if not template_edge or not arg_edge:
            return False, {"error": "expected two helper edges", "edges": edges}
        tb = next(b for b in p["bindings"] if b["id"] == template_edge["binding"])
        ab = next(b for b in p["bindings"] if b["id"] == arg_edge["binding"])
        ok = tb["kind"] == "global" and ab["kind"] == "let"
        return ok, {"template_use": {"edge": template_edge["id"], "binding": tb},
                    "argument_use": {"edge": arg_edge["id"], "binding": ab}}

    def _check_explicit_capture(self, spec: dict):
        der = self._expansion(spec["fragment"])
        p = der["payload"]
        d2s_nodes = [n for n in p["nodes"] if n["kind"] == "datum->syntax"]
        if not d2s_nodes:
            return False, {"error": "no datum->syntax node recorded"}
        node = d2s_nodes[0]
        edge = next((e for e in p["edges"] if e["name"] == spec["name"]), None)
        if not edge or edge["status"] != "resolved":
            return False, {"error": "capture edge not resolved", "edge": edge}
        binding = next(b for b in p["bindings"] if b["id"] == edge["binding"])
        ok = (str(binding["origin"]).startswith("d2s:")
              and node.get("context_origin")
              and node.get("context_scopes") is not None)
        return ok, {"datum_choice_node": node, "capture_binding": binding,
                    "edge_id": edge["id"]}

    def _check_budget_partial(self, spec: dict):
        der = self._expansion(spec["fragment"])
        p = der["payload"]
        stats = p["stats"]
        exhausted = [n for n in p["nodes"] if n["status"] == "budget_exhausted"]
        macro_calls = [n for n in p["nodes"] if n["kind"] == "macro-call"]
        ok = (stats["status"] == "budget_exhausted"
              and stats["steps_used"] == spec["expect_steps"] + 1
              and len(exhausted) == 1
              and len(macro_calls) == spec["expect_steps"] + 1
              and len(p["transform_steps"]) == spec["expect_steps"])
        return ok, {"stats": stats, "exhausted_node": exhausted[0]["id"] if exhausted else None,
                    "partial_macro_call_nodes": [n["id"] for n in macro_calls]}

    def _check_gensym_distinct(self, spec: dict):
        der = self._expansion(spec["fragment"])
        p = der["payload"]
        gensym_bindings = [b for b in p["bindings"] if "~g" in b["name"]]
        names = {b["name"] for b in gensym_bindings}
        ok = (len(gensym_bindings) == spec["expect_count"]
              and len(names) == spec["expect_count"]
              and p["stats"]["generated_symbols"] == spec["expect_count"])
        tmp_edges = [e for e in p["edges"] if e["name"] == "tmp"]
        ok = ok and tmp_edges and tmp_edges[0]["status"] == "unbound"
        return ok, {"gensym_names": sorted(names),
                    "user_tmp_status": tmp_edges[0]["status"] if tmp_edges else None}

    def _check_error_remap(self, spec: dict):
        der = self._expansion(spec["fragment"])
        p = der["payload"]
        diags = [d for d in p["diagnostics"] if d["code"] == "unbound-identifier"]
        frob = next((d for d in diags if "frobnicate" in d["message"]), None)
        miss = next((d for d in diags if "missing-fn" in d["message"]), None)
        if not frob or not miss:
            return False, {"error": "expected two unbound diagnostics",
                           "diagnostics": diags}
        frob_origin = next((c for c in frob["remap_chain"]
                            if c.get("kind") == "original-token"), None)
        miss_tpl = next((c for c in miss["remap_chain"]
                         if c.get("kind") == "macro-template-token"), None)
        ok = (frob_origin is not None and frob_origin.get("span") is not None
              and miss_tpl is not None and miss_tpl.get("macro") == "broken"
              and miss_tpl.get("span") is not None
              and any("macro_call_stack" in c for c in miss["remap_chain"]))
        return ok, {"frobnicate_remap": frob_origin, "missing_fn_remap": miss_tpl,
                    "missing_fn_stack": miss["macro_call_stack"]}

    def _check_quantity_normalization(self, spec: dict):
        rows = self.store.list_derived(kind="normalized-quantities", include_stale=True)
        if not rows:
            return False, {"error": "no normalized-quantities derived"}
        q = rows[-1]["payload"]["quantities"]
        budget = q.get("expansion_budget", {})
        latency = q.get("latency_probe", {})
        bad = q.get("unconvertible_probe", {})
        ok = (budget.get("value") == 8.0 and budget.get("unit") == "count"
              and budget.get("raw") == "8 steps"
              and abs(latency.get("value", -1) - 0.25) < 1e-9
              and latency.get("unit") == "s"
              and bad.get("status") == "unconvertible"
              and bad.get("raw") == "many"
              and "diagnostic" in bad)
        return ok, {"quantities": q}

    def _check_param_revision_stales(self, spec: dict):
        frag = spec["fragment"]
        current = self.imported[frag]
        all_der = self.store.list_derived(kind="expansion", include_stale=True)
        stale = [d for d in all_der if d["status"] == "stale"]
        cur_der = self._expansion(frag)
        old_der = [d for d in stale
                   if d["subject_version"] != current["version_id"]]
        ok = (len(old_der) >= 1
              and cur_der["status"] == "current"
              and cur_der["payload"]["params"]["budget"] == 3
              and old_der[0]["payload"]["params"]["budget"] == 5)
        return ok, {"stale_derived": [d["id"] for d in stale],
                    "current_derived": cur_der["id"],
                    "current_budget": cur_der["payload"]["params"]["budget"]}

    def _check_local_change_scoped_stale(self, spec: dict):
        changed = self.imported[spec["fragment"]]
        stale = self.store.list_derived(include_stale=True)
        stale_ids = {d["id"] for d in stale if d["status"] == "stale"}
        changed_old_ok = any(d["subject_version"] != changed["version_id"]
                             and d["id"] in stale_ids for d in stale)
        unaffected_ok = True
        detail = {}
        for other in spec["unaffected"]:
            der = self._expansion(other)
            detail[other] = der["status"]
            unaffected_ok = unaffected_ok and der["status"] == "current"
        return (changed_old_ok and unaffected_ok), {
            "stale_derived": sorted(stale_ids), "unaffected_status": detail}

    def _check_conflict_state(self, spec: dict):
        decisions = self.store.effective_decisions()
        target = self.last_conflict_target
        entry = next((d for d in decisions if d["target"] == target), None)
        ok = entry is not None and entry["state"] == spec["expect"]
        return ok, {"target": target, "decision_entry": entry}

    def _check_capture_by_scope(self, spec: dict):
        der = self._expansion(spec["fragment"])
        p = der["payload"]
        edge = next((e for e in p["edges"] if e["name"] == spec["name"]), None)
        if not edge:
            return False, {"error": "edge not found"}
        invisible = [c for c in edge["candidates"] if not c["visible"]]
        same_name = [c for c in invisible if c["name"] == spec["name"]]
        ok = (edge["status"] == "unbound"
              and len(same_name) >= 1
              and all("NOT subset" in c["reason"] for c in same_name)
              and edge["rule"].startswith("binding visible iff"))
        return ok, {"edge_id": edge["id"], "status": edge["status"],
                    "hygiene_barrier_candidates": same_name, "rule": edge["rule"]}

    def _check_ambiguous_kept(self, spec: dict):
        der = self._expansion(spec["fragment"])
        edge = next((e for e in der["payload"]["edges"]
                     if e["name"] == spec["name"]), None)
        if not edge:
            return False, {"error": "edge not found"}
        kept = edge.get("kept_candidates", [])
        ok = (edge["status"] == "ambiguous"
              and len(kept) == spec["expect_count"]
              and edge.get("binding") is None)
        return ok, {"edge_id": edge["id"], "kept_candidates": kept}

    def _check_history_view_stable(self, spec: dict):
        events = self.store.events()
        first_check = next((e for e in events if e["type"] == "case-check"), None)
        if not first_check:
            return False, {"error": "no case-check events"}
        view = self.store.view_at(first_check["id"])
        later = self.store.list_derived(include_stale=True)
        ok = (len(view["derived"]) < len(later)
              and all(d["created_seq"] <= first_check["seq"] for d in view["derived"]))
        return ok, {"as_of_event": first_check["id"],
                    "derived_then": len(view["derived"]),
                    "derived_now": len(later)}
