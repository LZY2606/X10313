"""Hygienic macro expander based on binding-as-sets-of-scopes.

Design invariants:
- Capture is decided by scope-set containment, never by string renaming.
- Every datum->syntax context choice (intro / use-site / capture / gensym)
  is recorded as a first-class, machine-readable event.
- Recursive expansion has a budget; on exhaustion the partial expansion
  graph is preserved, not discarded.
- Ambiguous resolutions keep *all* legal candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .syntax import SIdent, SInt, SList, Span, Syn, syn_to_json, syn_to_str, with_scope

CORE_FORMS = ("let", "def", "defmacro", "if", "begin", "+", "-", "*")


@dataclass
class Binding:
    bid: str
    name: str
    scopes: frozenset
    kind: str  # core | def | macro | local | gensym
    span: Optional[Span]
    macro: Optional["Macro"] = None

    def to_dict(self) -> dict:
        return {
            "bid": self.bid,
            "name": self.name,
            "scopes": sorted(self.scopes),
            "kind": self.kind,
            "span": self.span.to_dict() if self.span else None,
        }


@dataclass
class Macro:
    name: str
    params: List[str]
    template: Syn
    def_span: Optional[Span]


@dataclass
class Step:
    sid: str
    macro: str
    parent: Optional[str]
    depth: int
    intro_scope: str
    use_span: Optional[Span]
    def_span: Optional[Span]
    call: str

    def to_dict(self) -> dict:
        return {
            "sid": self.sid,
            "macro": self.macro,
            "parent": self.parent,
            "depth": self.depth,
            "intro_scope": self.intro_scope,
            "use_span": self.use_span.to_dict() if self.use_span else None,
            "def_span": self.def_span.to_dict() if self.def_span else None,
            "call": self.call,
        }


class Expander:
    def __init__(self, budget: int = 64):
        self.budget = budget
        self.status = "ok"
        self._scope_n = 0
        self._step_n = 0
        self._bind_n = 0
        self.bindings: List[Binding] = []
        self.steps: Dict[str, Step] = {}
        self.resolutions: List[dict] = []
        self.context_choices: List[dict] = []
        self.diagnostics: List[dict] = []
        for name in CORE_FORMS:
            self._add_binding(name, frozenset(), "core", None)

    # -- identity helpers -------------------------------------------------
    def _fresh_scope(self, prefix: str) -> str:
        self._scope_n += 1
        return f"{prefix}{self._scope_n}"

    def _add_binding(self, name, scopes, kind, span, macro=None) -> Binding:
        self._bind_n += 1
        b = Binding(f"b{self._bind_n}", name, frozenset(scopes), kind, span, macro)
        self.bindings.append(b)
        return b

    # -- resolution: capture decided by scope relations -------------------
    def resolve(self, ident: SIdent) -> Optional[Binding]:
        same_name = [b for b in self.bindings if b.name == ident.name]
        universe = sorted({s for b in same_name for s in b.scopes} | set(ident.scopes))
        rejected, viable = [], []
        if same_name:
            ivec = np.array([s in ident.scopes for s in universe], dtype=bool)
            for b in same_name:
                bvec = np.array([s in b.scopes for s in universe], dtype=bool)
                if bool(np.all(bvec <= ivec)):
                    viable.append(b)
                else:
                    missing = sorted(set(b.scopes) - set(ident.scopes))
                    rejected.append(
                        {"bid": b.bid, "kind": b.kind, "reason": "scope-not-contained",
                         "missing_scopes": missing}
                    )
        chosen, candidates = None, []
        if viable:
            mx = max(len(b.scopes) for b in viable)
            top = [b for b in viable if len(b.scopes) == mx]
            candidates = [b.bid for b in top]
            if len(top) == 1:
                chosen = top[0]
            else:
                self.diagnostics.append({
                    "kind": "ambiguous",
                    "message": f"ambiguous reference {ident.name!r}: {len(top)} legal candidates kept",
                    "node": ident.nid,
                    "candidates": [b.bid for b in top],
                    "remapped": self.remap(ident),
                })
        self.resolutions.append({
            "node": ident.nid,
            "name": ident.name,
            "scopes": sorted(ident.scopes),
            "origin": ident.origin,
            "span": ident.span.to_dict() if ident.span else None,
            "chosen": chosen.bid if chosen else None,
            "candidates": candidates,
            "viable": [b.bid for b in viable],
            "rejected": rejected,
        })
        if chosen is None and not candidates:
            self.diagnostics.append({
                "kind": "unbound",
                "message": f"unbound identifier {ident.name!r}",
                "node": ident.nid,
                "remapped": self.remap(ident),
            })
        return chosen

    # -- source remapping --------------------------------------------------
    def call_stack(self, sid: Optional[str]) -> List[str]:
        chain = []
        while sid:
            chain.append(sid)
            sid = self.steps[sid].parent
        return chain

    def remap(self, node: Syn) -> dict:
        if node.origin == "src":
            return {"site": "source", "span": node.span.to_dict() if node.span else None}
        step = self.steps.get(node.origin)
        if step is None:
            return {"site": "unknown", "span": None}
        return {
            "site": "macro",
            "step": step.sid,
            "macro": step.macro,
            "def_span": step.def_span.to_dict() if step.def_span else None,
            "use_span": step.use_span.to_dict() if step.use_span else None,
            "call_stack": self.call_stack(step.sid),
            "span": node.span.to_dict() if node.span else None,
        }

    # -- template instantiation: datum -> syntax context choice ------------
    def instantiate(self, tmpl: Syn, mapping: Dict[str, Syn], intro: str,
                    call_scopes: frozenset, gensyms: Dict[str, str], step: Step) -> Syn:
        nid = f"{tmpl.nid}@{step.sid}"
        if isinstance(tmpl, SInt):
            return SInt(tmpl.value, tmpl.scopes | {intro}, tmpl.span, step.sid, nid)
        if isinstance(tmpl, SIdent):
            if tmpl.name in mapping:
                arg = mapping[tmpl.name]
                self.context_choices.append({
                    "step": step.sid, "datum": tmpl.name, "choice": "use-site",
                    "from_node": tmpl.nid, "to_node": arg.nid,
                    "scopes": sorted(arg.scopes),
                })
                return arg
            new = SIdent(tmpl.name, tmpl.scopes | {intro}, tmpl.span, step.sid, nid)
            self.context_choices.append({
                "step": step.sid, "datum": tmpl.name, "choice": "intro",
                "from_node": tmpl.nid, "to_node": nid,
                "scopes": sorted(new.scopes),
            })
            return new
        # SList
        if tmpl.items and isinstance(tmpl.items[0], SIdent):
            head = tmpl.items[0].name
            if head == "gensym" and len(tmpl.items) == 2 and isinstance(tmpl.items[1], SIdent):
                base = tmpl.items[1].name
                if base not in gensyms:
                    gensyms[base] = self._fresh_scope("g")
                g = gensyms[base]
                new = SIdent(base, tmpl.scopes | {intro, g}, tmpl.span, step.sid, nid)
                self.context_choices.append({
                    "step": step.sid, "datum": base, "choice": "gensym",
                    "from_node": tmpl.nid, "to_node": nid,
                    "scopes": sorted(new.scopes), "gensym_scope": g,
                })
                return new
            if head == "capture" and len(tmpl.items) == 2 and isinstance(tmpl.items[1], SIdent):
                base = tmpl.items[1].name
                new = SIdent(base, frozenset(call_scopes), tmpl.span, step.sid, nid)
                self.context_choices.append({
                    "step": step.sid, "datum": base, "choice": "capture",
                    "from_node": tmpl.nid, "to_node": nid,
                    "scopes": sorted(new.scopes),
                })
                return new
        items = tuple(self.instantiate(i, mapping, intro, call_scopes, gensyms, step)
                      for i in tmpl.items)
        return SList(items, tmpl.scopes | {intro}, tmpl.span, step.sid, nid)

    # -- expansion ----------------------------------------------------------
    def expand(self, syn: Syn, parent_step: Optional[str] = None, depth: int = 0) -> Syn:
        if isinstance(syn, SInt):
            return syn
        if isinstance(syn, SIdent):
            self.resolve(syn)
            return syn
        if not syn.items:
            return syn
        head = syn.items[0]
        if isinstance(head, SIdent):
            b = self.resolve(head)
            if b is not None and b.kind == "macro":
                return self._expand_macro(syn, b, parent_step, depth)
            if b is not None and b.name == "let":
                return self._expand_let(syn, parent_step, depth)
            if b is not None and b.name == "def":
                return self._expand_def(syn, parent_step, depth)
            if b is not None and b.name == "defmacro":
                return self._expand_defmacro(syn)
        items = tuple(self.expand(i, parent_step, depth) for i in syn.items)
        return SList(items, syn.scopes, syn.span, syn.origin, syn.nid)

    def _expand_macro(self, syn: SList, b: Binding, parent_step, depth) -> Syn:
        macro = b.macro
        if len(self.steps) >= self.budget:
            self.status = "budget_exceeded"
            self.diagnostics.append({
                "kind": "budget_exceeded",
                "message": f"expansion budget {self.budget} exhausted at macro {macro.name!r}; "
                           "partial expansion graph preserved",
                "node": syn.nid,
                "remapped": {"site": "macro", "call_stack": self.call_stack(parent_step),
                             "use_span": syn.span.to_dict() if syn.span else None},
            })
            return syn  # unexpanded residue kept in the partial graph
        args = list(syn.items[1:])
        if len(args) != len(macro.params):
            self.diagnostics.append({
                "kind": "arity",
                "message": f"macro {macro.name!r} expects {len(macro.params)} args, got {len(args)}",
                "node": syn.nid,
                "remapped": {"site": "macro",
                             "use_span": syn.span.to_dict() if syn.span else None,
                             "def_span": macro.def_span.to_dict() if macro.def_span else None},
            })
            return syn
        self._step_n += 1
        sid = f"s{self._step_n}"
        intro = self._fresh_scope("i")
        step = Step(sid, macro.name, parent_step, depth, intro, syn.span, macro.def_span,
                    syn_to_str(syn))
        self.steps[sid] = step
        mapping = dict(zip(macro.params, args))
        body = self.instantiate(macro.template, mapping, intro, syn.scopes, {}, step)
        return self.expand(body, parent_step=sid, depth=depth + 1)

    def _expand_let(self, syn: SList, parent_step, depth) -> Syn:
        # (let (x e) body)
        if len(syn.items) != 3 or not isinstance(syn.items[1], SList) \
                or len(syn.items[1].items) != 2 \
                or not isinstance(syn.items[1].items[0], SIdent):
            self.diagnostics.append({"kind": "syntax", "message": "malformed let",
                                     "node": syn.nid, "remapped": self.remap(syn)})
            return syn
        x, e = syn.items[1].items
        body = syn.items[2]
        scope = self._fresh_scope("l")
        x2 = with_scope(x, scope)
        body2 = with_scope(body, scope)
        self._add_binding(x.name, x2.scopes, "local", x.span)
        e2 = self.expand(e, parent_step, depth)
        body3 = self.expand(body2, parent_step, depth)
        binding = SList((x2, e2), syn.items[1].scopes, syn.items[1].span, syn.items[1].origin,
                        syn.items[1].nid)
        return SList((syn.items[0], binding, body3), syn.scopes, syn.span, syn.origin, syn.nid)

    def _expand_def(self, syn: SList, parent_step, depth) -> Syn:
        if len(syn.items) != 3 or not isinstance(syn.items[1], SIdent):
            self.diagnostics.append({"kind": "syntax", "message": "malformed def",
                                     "node": syn.nid, "remapped": self.remap(syn)})
            return syn
        name = syn.items[1]
        self._add_binding(name.name, name.scopes, "def", name.span)
        val = self.expand(syn.items[2], parent_step, depth)
        return SList((syn.items[0], name, val), syn.scopes, syn.span, syn.origin, syn.nid)

    def _expand_defmacro(self, syn: SList) -> Syn:
        # (defmacro name (p...) template)
        if len(syn.items) != 4 or not isinstance(syn.items[1], SIdent) \
                or not isinstance(syn.items[2], SList) \
                or not all(isinstance(p, SIdent) for p in syn.items[2].items):
            self.diagnostics.append({"kind": "syntax", "message": "malformed defmacro",
                                     "node": syn.nid, "remapped": self.remap(syn)})
            return syn
        name = syn.items[1]
        params = [p.name for p in syn.items[2].items]
        macro = Macro(name.name, params, syn.items[3], name.span)
        self._add_binding(name.name, name.scopes, "macro", name.span, macro)
        return syn

    # -- result -------------------------------------------------------------
    def result(self, output: List[Syn]) -> dict:
        return {
            "status": self.status,
            "budget": self.budget,
            "steps_used": len(self.steps),
            "output": [syn_to_json(s) for s in output],
            "output_text": "\n".join(syn_to_str(s) for s in output),
            "graph": {
                "nodes": [s.to_dict() for s in self.steps.values()],
                "edges": [{"from": s.parent, "to": s.sid}
                          for s in self.steps.values() if s.parent],
            },
            "bindings": [b.to_dict() for b in self.bindings],
            "resolutions": self.resolutions,
            "context_choices": self.context_choices,
            "diagnostics": self.diagnostics,
        }


def expand_program(forms: List[Syn], budget: int = 64) -> dict:
    ex = Expander(budget)
    out = [ex.expand(f) for f in forms]
    return ex.result(out)
