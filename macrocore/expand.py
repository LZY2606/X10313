"""Sets-of-scopes macro expander.

Binding is decided purely by scope-set inclusion
(binder.scopes <= use.scopes, most specific binder wins), never by string
renaming. Each macro step creates an intro scope and a use scope; datum->syntax
can explicitly select a visible context. Recursion has an explicit step budget
and the partial expansion graph is retained on failure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .syntax import Atom, Ctx, Id, Lst, Span, Syn, add_scope, label, replace_scopes, to_data

ALGO_VERSION = "expander/1.2.0"
RESOLUTION_RULE = (
    "binding visible iff binder.scopes is a subset of use.scopes; "
    "among visible binders the largest scope set wins; ties are ambiguous"
)


@dataclass
class Binding:
    id: str
    name: str
    scopes: frozenset
    kind: str
    span: Optional[Span]
    origin: str
    step: int

    def to_data(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "scopes": sorted(self.scopes),
            "kind": self.kind,
            "span": self.span.to_data() if self.span else None,
            "origin": self.origin,
            "step": self.step,
        }


@dataclass
class Macro:
    name: str
    params: tuple
    template: Syn
    def_scopes: frozenset
    def_span: Optional[Span]
    def_origin: str
    binding_id: str


class ExpansionError(Exception):
    def __init__(self, message: str, diagnostics: List[dict]):
        super().__init__(message)
        self.diagnostics = diagnostics


class Expander:
    def __init__(self, budget: int = 8, seed: int = 0):
        self.budget = int(budget)
        self.seed = int(seed)
        self.steps_used = 0
        self._n = {"scope": 0, "binding": 0, "node": 0, "edge": 0, "diag": 0, "gensym": 0}
        self.macros: Dict[str, Macro] = {}
        self.bindings: List[Binding] = []
        self.nodes: List[dict] = []
        self.edges: List[dict] = []
        self.diagnostics: List[dict] = []
        self.source_map: List[dict] = []
        self.transform_steps: List[dict] = []
        self._by_id: Dict[str, dict] = {}
        self.rng = random.Random(seed)

    def _fresh(self, kind: str, tag: str) -> str:
        self._n[kind] += 1
        return f"{tag}{self._n[kind]}"

    def _span_d(self, span: Optional[Span]) -> Optional[dict]:
        return span.to_data() if span else None

    def _add_node(self, syn: Optional[Syn], parent: Optional[str], kind: str,
                  call_stack: List[str], **extra: Any) -> str:
        nid = self._fresh("node", "N")
        node: Dict[str, Any] = {
            "id": nid,
            "kind": kind,
            "label": label(syn) if syn is not None else extra.pop("node_label", ""),
            "origin": syn.ctx.origin if syn is not None else extra.pop("origin", ""),
            "span": self._span_d(syn.ctx.span) if syn is not None else extra.pop("span", None),
            "scopes": sorted(syn.ctx.scopes) if syn is not None else extra.pop("scopes", []),
            "step": syn.ctx.step if syn is not None else extra.pop("step", 0),
            "parent": parent,
            "children": [],
            "call_stack": list(call_stack),
            "status": "ok",
            "note": "",
        }
        node.update(extra)
        self.nodes.append(node)
        self._by_id[nid] = node
        if parent:
            self._by_id[parent]["children"].append(nid)
        return nid

    def _add_binding(self, name: str, scopes: frozenset, kind: str,
                     span: Optional[Span], origin: str, step: int) -> Binding:
        b = Binding(self._fresh("binding", "B"), name, scopes, kind, span, origin, step)
        self.bindings.append(b)
        return b

    def _diag(self, message: str, syn: Syn, call_stack: List[str],
              code: str, severity: str = "error") -> dict:
        did = self._fresh("diag", "D")
        d = {
            "id": did,
            "code": code,
            "severity": severity,
            "message": message,
            "origin": syn.ctx.origin,
            "span": self._span_d(syn.ctx.span),
            "scopes": sorted(syn.ctx.scopes),
            "macro_call_stack": list(call_stack),
            "remap_chain": self._remap_chain(syn.ctx.origin, call_stack),
        }
        self.diagnostics.append(d)
        return d

    def _remap_chain(self, origin: str, call_stack: List[str]) -> List[dict]:
        """Map a generated origin back through macro steps to a physical source span."""
        chain: List[dict] = []
        cur = origin
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            if cur.startswith("gen:"):
                step = int(cur.split(":")[1])
                tstep = next((t for t in self.transform_steps if t["step"] == step), None)
                chain.append({"origin": cur, "kind": "generated-symbol",
                              "macro_node": tstep["macro_node"] if tstep else None,
                              "span": tstep["macro_def_span"] if tstep else None})
                cur = tstep["macro_def_origin"] if tstep else ""
            elif cur.startswith("d2s:"):
                target = cur[4:]
                chain.append({"origin": cur, "kind": "datum->syntax", "maps_to_context": target})
                cur = target
            elif cur.startswith("mac:"):
                parts = cur.split(":", 2)
                mname, tok = parts[1], parts[2]
                macro = self.macros.get(mname)
                chain.append({"origin": cur, "kind": "macro-template-token",
                              "macro": mname, "token": tok,
                              "span": self._span_d(macro.def_span) if macro else None})
                cur = ""
            else:
                node = self._origin_node(cur)
                chain.append({"origin": cur, "kind": "original-token",
                              "span": node["span"] if node else None})
                cur = ""
        if call_stack:
            chain.append({"macro_call_stack": list(call_stack)})
        return chain

    def _origin_node(self, origin: str) -> Optional[dict]:
        return next((n for n in self.nodes if n["origin"] == origin), None)

    # ------------------------------------------------------------------ entry
    def expand_program(self, prog: Lst) -> dict:
        root = self._add_node(prog, None, "program", [], node_label="program",
                              origin="prog", span=self._span_d(prog.ctx.span))
        for form in prog.items:
            self.expand(form, [], root)
        errors = [d for d in self.diagnostics if d["severity"] == "error"]
        budget_hit = any(n["status"] == "budget_exhausted" for n in self.nodes)
        status = "budget_exhausted" if budget_hit else ("errors" if errors else "ok")
        return {
            "algo_version": ALGO_VERSION,
            "params": {"budget": self.budget, "seed": self.seed},
            "resolution_rule": RESOLUTION_RULE,
            "nodes": self.nodes,
            "edges": self.edges,
            "bindings": [b.to_data() for b in self.bindings],
            "macros": [
                {
                    "name": m.name,
                    "params": list(m.params),
                    "def_scopes": sorted(m.def_scopes),
                    "def_span": self._span_d(m.def_span),
                    "def_origin": m.def_origin,
                    "binding_id": m.binding_id,
                    "template": label(m.template),
                }
                for m in self.macros.values()
            ],
            "diagnostics": self.diagnostics,
            "source_map": self.source_map,
            "transform_steps": self.transform_steps,
            "stats": {
                "status": status,
                "steps_used": self.steps_used,
                "budget": self.budget,
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "bindings": len(self.bindings),
                "diagnostics": len(self.diagnostics),
                "generated_symbols": self._n["gensym"],
            },
        }

    # ------------------------------------------------------------- expansion
    def expand(self, syn: Syn, call_stack: List[str], parent: Optional[str]) -> Optional[str]:
        if isinstance(syn, Id):
            nid = self._add_node(syn, parent, "identifier", call_stack)
            self._resolve(syn, nid, call_stack)
            return nid
        if isinstance(syn, Atom):
            return self._add_node(syn, parent, "atom", call_stack)
        if isinstance(syn, Lst):
            if not syn.items:
                return self._add_node(syn, parent, "empty-list", call_stack)
            head = syn.items[0]
            if isinstance(head, Id) and head.name in self.macros:
                macro = self.macros[head.name]
                if macro.def_scopes <= head.ctx.scopes:
                    return self._expand_macro(syn, macro, call_stack, parent)
            if isinstance(head, Id) and head.name == "quote":
                return self._add_node(syn, parent, "quote", call_stack)
            if isinstance(head, Id) and head.name == "defmacro":
                return self._expand_defmacro(syn, call_stack, parent)
            if isinstance(head, Id) and head.name == "def":
                return self._expand_def(syn, call_stack, parent)
            if isinstance(head, Id) and head.name == "let":
                return self._expand_let(syn, call_stack, parent)
            if isinstance(head, Id) and head.name == "lambda":
                return self._expand_lambda(syn, call_stack, parent)
            if isinstance(head, Id) and head.name == "if":
                nid = self._add_node(syn, parent, "core:if", call_stack)
                for part in syn.items[1:]:
                    self.expand(part, call_stack, nid)
                return nid
            nid = self._add_node(syn, parent, "application", call_stack)
            for part in syn.items:
                self.expand(part, call_stack, nid)
            return nid
        raise TypeError(f"cannot expand {syn!r}")

    def _expand_def(self, syn: Lst, call_stack: List[str], parent: Optional[str]) -> str:
        nid = self._add_node(syn, parent, "core:def", call_stack)
        if len(syn.items) != 3 or not isinstance(syn.items[1], Id):
            self._malformed(syn, nid, call_stack, "def expects (def name value)")
            return nid
        name_id = syn.items[1]
        b = self._add_binding(name_id.name, frozenset(), "global",
                              name_id.ctx.span, name_id.ctx.origin, 0)
        self._by_id[nid]["binding"] = b.id
        self.expand(syn.items[2], call_stack, nid)
        return nid

    def _expand_defmacro(self, syn: Lst, call_stack: List[str], parent: Optional[str]) -> str:
        nid = self._add_node(syn, parent, "core:defmacro", call_stack)
        if (len(syn.items) != 4 or not isinstance(syn.items[1], Id)
                or not isinstance(syn.items[2], Lst)):
            self._malformed(syn, nid, call_stack,
                            "defmacro expects (defmacro name (params...) template)")
            return nid
        name_id = syn.items[1]
        params = tuple(p.name for p in syn.items[2].items if isinstance(p, Id))
        b = self._add_binding(name_id.name, frozenset(name_id.ctx.scopes), "macro",
                              name_id.ctx.span, name_id.ctx.origin, 0)
        self.macros[name_id.name] = Macro(
            name=name_id.name,
            params=params,
            template=syn.items[3],
            def_scopes=frozenset(name_id.ctx.scopes),
            def_span=name_id.ctx.span,
            def_origin=name_id.ctx.origin,
            binding_id=b.id,
        )
        self._by_id[nid]["binding"] = b.id
        return nid

    def _expand_let(self, syn: Lst, call_stack: List[str], parent: Optional[str]) -> str:
        nid = self._add_node(syn, parent, "core:let", call_stack)
        if len(syn.items) < 3 or not isinstance(syn.items[1], Lst):
            self._malformed(syn, nid, call_stack, "let expects (let ((name val) ...) body ...)")
            return nid
        scope = self._fresh("scope", "b")
        self._by_id[nid]["introduced_scope"] = scope
        for clause in syn.items[1].items:
            if not isinstance(clause, Lst) or len(clause.items) != 2 or not isinstance(clause.items[0], Id):
                self._malformed(syn, nid, call_stack, "let clause must be (name value)")
                continue
            name_id, val = clause.items
            self.expand(val, call_stack, nid)
            bound = add_scope(name_id, scope)
            b = self._add_binding(bound.name, bound.ctx.scopes, "let",
                                  bound.ctx.span, bound.ctx.origin, bound.ctx.step)
            self._by_id[nid].setdefault("bindings_introduced", []).append(b.id)
        for body in syn.items[2:]:
            self.expand(add_scope(body, scope), call_stack, nid)
        return nid

    def _expand_lambda(self, syn: Lst, call_stack: List[str], parent: Optional[str]) -> str:
        nid = self._add_node(syn, parent, "core:lambda", call_stack)
        if len(syn.items) < 3 or not isinstance(syn.items[1], Lst):
            self._malformed(syn, nid, call_stack, "lambda expects (lambda (params...) body ...)")
            return nid
        scope = self._fresh("scope", "b")
        self._by_id[nid]["introduced_scope"] = scope
        for p in syn.items[1].items:
            if not isinstance(p, Id):
                self._malformed(syn, nid, call_stack, "lambda parameter must be an identifier")
                continue
            bound = add_scope(p, scope)
            b = self._add_binding(bound.name, bound.ctx.scopes, "lambda",
                                  bound.ctx.span, bound.ctx.origin, bound.ctx.step)
            self._by_id[nid].setdefault("bindings_introduced", []).append(b.id)
        for body in syn.items[2:]:
            self.expand(add_scope(body, scope), call_stack, nid)
        return nid

    def _expand_macro(self, syn: Lst, macro: Macro, call_stack: List[str],
                      parent: Optional[str]) -> str:
        self.steps_used += 1
        step = self.steps_used
        intro = self._fresh("scope", "m")
        use = self._fresh("scope", "u")
        call_site = self._add_node(syn, parent, "macro-call", call_stack,
                                   macro=macro.name, step_no=step)
        if self.steps_used > self.budget:
            node = self._by_id[call_site]
            node["status"] = "budget_exhausted"
            node["note"] = (f"expansion budget {self.budget} exceeded at step {step}; "
                            "partial expansion graph retained")
            self._diag(
                f"macro '{macro.name}' expansion stopped: budget {self.budget} exceeded "
                f"(step {step}); partial graph kept",
                syn, call_stack + [macro.name], code="budget-exhausted", severity="error")
            return call_site
        args = list(syn.items[1:])
        if len(args) != len(macro.params):
            self._diag(
                f"macro '{macro.name}' expects {len(macro.params)} args, got {len(args)}",
                syn, call_stack + [macro.name], code="arity")
            self._by_id[call_site]["status"] = "error"
            return call_site
        scoped_args = [add_scope(a, use) for a in args]
        mapping = dict(zip(macro.params, scoped_args))
        for pname, arg in zip(macro.params, scoped_args):
            self.source_map.append({
                "kind": "macro-argument",
                "macro": macro.name,
                "step": step,
                "param": pname,
                "origin": arg.ctx.origin,
                "span": self._span_d(arg.ctx.span),
                "call_stack": call_stack + [macro.name],
            })
        result = self._instantiate(macro.template, mapping, intro, macro, step,
                                   call_stack + [macro.name], call_site)
        self.transform_steps.append({
            "step": step,
            "macro": macro.name,
            "macro_node": call_site,
            "intro_scope": intro,
            "use_scope": use,
            "macro_def_span": self._span_d(macro.def_span),
            "macro_def_origin": macro.def_origin,
            "call_site_span": self._span_d(syn.ctx.span),
            "result_label": label(result)[:200],
        })
        out_node = self._add_node(result, call_site, "macro-output",
                                  call_stack + [macro.name], macro=macro.name, step_no=step)
        self._by_id[call_site]["output_node"] = out_node
        self.expand(result, call_stack + [macro.name], out_node)
        return call_site

    def _instantiate(self, template: Syn, mapping: Dict[str, Syn], intro: str,
                     macro: Macro, step: int, call_stack: List[str],
                     call_site: str) -> Syn:
        if isinstance(template, Id):
            if template.name in mapping:
                return mapping[template.name]
            return Id(template.name, Ctx(
                template.ctx.span,
                macro.def_scopes | {intro},
                f"mac:{macro.name}:{template.ctx.origin}",
                step,
            ))
        if isinstance(template, Atom):
            return Atom(template.value, Ctx(
                template.ctx.span,
                template.ctx.scopes | {intro},
                f"mac:{macro.name}:{template.ctx.origin}",
                step,
            ))
        if isinstance(template, Lst):
            head = template.items[0] if template.items else None
            if isinstance(head, Id) and head.name == "quote" and len(template.items) == 2:
                inner = template.items[1]
                return replace_scopes(inner, inner.ctx.scopes | {intro},
                                      origin=f"mac:{macro.name}:{inner.ctx.origin}", step=step)
            if isinstance(head, Id) and head.name == "datum->syntax" and len(template.items) == 3:
                ctx_ref, datum_form = template.items[1], template.items[2]
                if not isinstance(ctx_ref, Id) or ctx_ref.name not in mapping:
                    raise ExpansionError(
                        "datum->syntax context must be a macro parameter",
                        [self._diag("datum->syntax context is not a macro parameter",
                                    template, call_stack, code="datum->syntax")])
                ctx_syn = mapping[ctx_ref.name]
                if not (isinstance(datum_form, Lst) and datum_form.items
                        and isinstance(datum_form.items[0], Id)
                        and datum_form.items[0].name == "quote"):
                    raise ExpansionError(
                        "datum->syntax datum must be quoted",
                        [self._diag("datum->syntax datum must be (quote ...)",
                                    template, call_stack, code="datum->syntax")])
                raw = datum_form.items[1]
                new_origin = f"d2s:{ctx_syn.ctx.origin}"
                result = replace_scopes(raw, ctx_syn.ctx.scopes, origin=new_origin, step=step)
                choice_node = self._add_node(
                    result, call_site, "datum->syntax", call_stack,
                    context_origin=ctx_syn.ctx.origin,
                    context_scopes=sorted(ctx_syn.ctx.scopes),
                    context_span=self._span_d(ctx_syn.ctx.span),
                    datum_label=label(raw),
                    macro=macro.name,
                )
                self._by_id[choice_node]["note"] = (
                    f"datum->syntax chose context of parameter '{ctx_ref.name}' "
                    f"(origin {ctx_syn.ctx.origin}); datum scopes replaced with "
                    f"{sorted(ctx_syn.ctx.scopes)}")
                self.source_map.append({
                    "kind": "datum->syntax",
                    "macro": macro.name,
                    "step": step,
                    "node": choice_node,
                    "context_origin": ctx_syn.ctx.origin,
                    "context_span": self._span_d(ctx_syn.ctx.span),
                    "datum_origin": raw.ctx.origin,
                    "result_origin": new_origin,
                    "call_stack": list(call_stack),
                })
                return result
            if isinstance(head, Id) and head.name == "gensym" and len(template.items) == 2:
                base_form = template.items[1]
                base = base_form.value if isinstance(base_form, Atom) else "g"
                gid = self._fresh("gensym", "g")
                name = f"{base}~{gid}"
                origin = f"gen:{step}:{gid}"
                self.source_map.append({
                    "kind": "gensym",
                    "macro": macro.name,
                    "step": step,
                    "symbol": name,
                    "origin": origin,
                    "intro_scope": intro,
                    "call_stack": list(call_stack),
                })
                return Id(name, Ctx(None, macro.def_scopes | {intro}, origin, step))
            items = tuple(
                self._instantiate(t, mapping, intro, macro, step, call_stack, call_site)
                for t in template.items
            )
            return Lst(items, Ctx(
                template.ctx.span,
                template.ctx.scopes | {intro},
                f"mac:{macro.name}:{template.ctx.origin}",
                step,
            ))
        raise TypeError(f"cannot instantiate {template!r}")

    # ------------------------------------------------------------- resolution
    def _resolve(self, idsyn: Id, node_id: str, call_stack: List[str]) -> None:
        visible = [b for b in self.bindings
                   if b.name == idsyn.name and b.scopes <= idsyn.ctx.scopes]
        invisible = [b for b in self.bindings
                     if b.name == idsyn.name and not (b.scopes <= idsyn.ctx.scopes)]
        edge = {
            "id": self._fresh("edge", "E"),
            "use_node": node_id,
            "name": idsyn.name,
            "use_origin": idsyn.ctx.origin,
            "use_span": self._span_d(idsyn.ctx.span),
            "use_scopes": sorted(idsyn.ctx.scopes),
            "rule": RESOLUTION_RULE,
            "call_stack": list(call_stack),
            "candidates": [
                {**b.to_data(), "visible": True,
                 "reason": f"{sorted(b.scopes)} subset of use scopes"}
                for b in sorted(visible, key=lambda x: x.id)
            ] + [
                {**b.to_data(), "visible": False,
                 "reason": f"{sorted(b.scopes)} NOT subset of use scopes "
                           f"{sorted(idsyn.ctx.scopes)} (hygiene barrier)"}
                for b in sorted(invisible, key=lambda x: x.id)
            ],
        }
        if not visible:
            edge["status"] = "unbound"
            edge["binding"] = None
            self._by_id[node_id]["status"] = "unbound"
            self._diag(f"unbound identifier '{idsyn.name}'", idsyn, call_stack,
                       code="unbound-identifier")
        else:
            max_card = max(len(b.scopes) for b in visible)
            best = [b for b in visible if len(b.scopes) == max_card]
            if len(best) > 1:
                edge["status"] = "ambiguous"
                edge["binding"] = None
                edge["kept_candidates"] = [b.id for b in best]
                self._by_id[node_id]["status"] = "ambiguous"
                self._diag(
                    f"identifier '{idsyn.name}' has {len(best)} maximal candidates; "
                    "all legal candidates retained",
                    idsyn, call_stack, code="ambiguous-binding", severity="warning")
            else:
                edge["status"] = "resolved"
                edge["binding"] = best[0].id
                self._by_id[node_id]["binding"] = best[0].id
        self.edges.append(edge)

    def _malformed(self, syn: Lst, nid: str, call_stack: List[str], message: str) -> None:
        self._by_id[nid]["status"] = "error"
        self._diag(message, syn, call_stack, code="malformed-form")
