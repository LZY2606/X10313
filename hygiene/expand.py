"""Scope-set hygienic expander.

The expander consumes parsed syntax objects and produces:
  * a fully (or partially, on budget failure) expanded syntax tree;
  * a scope registry (program/use/bind/introduction scopes, ordered);
  * binding edges for every identifier occurrence (resolved or unknown,
    including rejection candidates);
  * macro call records with the full call stack, candidate clauses and
    reject reasons;
  * transformation steps with per-step scope snapshots;
  * context-choice (datum->syntax) and gensym events.

Binding resolution is purely relational: an identifier is captured by a
binding iff the binding's scope is in the identifier's scope set and no
later-created visible binding scope is also present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .syntax import Syntax, Birth
from .patterns import match_pattern, instantiate, MatchFail

CORE_FORMS = {"quote", "lambda", "let", "if", "begin", "set!", "define",
              "define-syntax"}
TEMPLATE_ONLY = {"datum->syntax", "with-temp"}


@dataclass
class ScopeInfo:
    sid: str
    kind: str            # program | bind | intro
    label: str
    order: int
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"sid": self.sid, "kind": self.kind, "label": self.label,
                "order": self.order, "detail": self.detail}


@dataclass
class BindingInfo:
    bid: str
    name: str
    scope_id: str
    site: str            # definition | use-site-lambda | use-site-let | temp | global-prim
    symbol_oid: str
    kind: str            # value | macro
    order: int

    def as_dict(self) -> dict:
        return {"bid": self.bid, "name": self.name, "scope_id": self.scope_id,
                "site": self.site, "symbol_oid": self.symbol_oid, "kind": self.kind,
                "order": self.order}


class BudgetExceeded(Exception):
    def __init__(self, budget: int):
        self.budget = budget
        super().__init__(f"macro expansion budget exhausted at {budget} calls")


class ExpandError(Exception):
    def __init__(self, msg: str, obj: Syntax | None = None):
        self.msg = msg
        self.obj = obj
        super().__init__(msg)


@dataclass
class Clause:
    index: int
    pattern: Syntax
    template: Syntax


@dataclass
class MacroInfo:
    name: str
    literals: list[str]
    clauses: list[Clause]
    binding: BindingInfo
    definition_oid: str
    frozen_scope_note: str


@dataclass
class TreeNode:
    node_id: str
    label: str
    kind: str
    oid: str
    span: dict | None
    children: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"node_id": self.node_id, "label": self.label, "kind": self.kind,
                "oid": self.oid, "span": self.span,
                "children": [c.as_dict() for c in self.children], "detail": self.detail}


def _walk_unique(obj: Syntax):
    """Yield each distinct syntax object once, pre-order (id identity)."""
    seen: set[int] = set()
    stack = [obj]
    while stack:
        s = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        yield s
        if s.is_list:
            stack.extend(reversed(s.val))


def _snapshot(obj: Syntax) -> dict:
    if obj.is_list:
        return {"oid": obj.oid, "list": [_snapshot(x) for x in obj.val]}
    return {"oid": obj.oid, "symbol": obj.symbol, "value": obj.val,
            "scopes": sorted(obj.scopes)}


class Expander:
    def __init__(self, reg, clock, *, budget: int = 64, clause_overrides=None,
                 algorithm_version: str = "hygiene-algo-v1"):
        self.reg = reg
        self.clock = clock
        self.budget = int(budget)
        self.calls_used = 0
        self.clause_overrides = clause_overrides or {}
        self.algorithm_version = algorithm_version

        self.scopes: list[ScopeInfo] = []
        self.bindings: list[BindingInfo] = []
        self.binding_edges: list[dict] = []
        self.macro_calls: list[dict] = []
        self.steps: list[dict] = []
        self.context_choices: list[dict] = []
        self.gensyms: list[dict] = []
        self.status = "ok"
        self.failure: dict | None = None
        self.macros: dict[str, MacroInfo] = {}
        self.global_env: dict[str, str] = {}

        self._scope_seq = 0
        self._bind_seq = 0
        self._call_seq = 0
        self._node_seq = 0
        self._step_seq = 0
        self.call_stack: list[str] = []

        self.program_scope = self.new_scope("program", "P0:initial-program",
                                            {"why": "stamped on every parsed token"})

    # ----- registries -------------------------------------------------
    def new_scope(self, kind: str, label: str, detail=None) -> str:
        sid = f"sc-{self._scope_seq:03d}"
        self._scope_seq += 1
        self.scopes.append(ScopeInfo(sid, kind, label, len(self.scopes), detail or {}))
        return sid

    def new_binding(self, name, scope_id, site, symbol_oid, kind="value") -> BindingInfo:
        bid = f"b-{self._bind_seq:04d}"
        self._bind_seq += 1
        bi = BindingInfo(bid, name, scope_id, site, symbol_oid, kind, len(self.bindings))
        self.bindings.append(bi)
        return bi

    def _step(self, title, *, before=None, after=None, detail=None):
        self._step_seq += 1
        self.steps.append({
            "step_id": f"st-{self._step_seq:05d}",
            "ts_ms": self.clock.tick(),
            "title": title,
            "call_stack": list(self.call_stack),
            "before": before,
            "after": after,
            "detail": detail or {},
        })

    # ----- binding resolution ----------------------------------------
    def resolve(self, ident: Syntax, bind_scope_owner=None):
        sym = ident.symbol
        candidates = []
        for b in self.bindings:
            if b.name == sym and b.scope_id in ident.scopes:
                candidates.append(b)
        visible = sorted(candidates, key=lambda b: (self._scope_order(b.scope_id), b.order),
                         reverse=True)
        rejected = [{
            "bid": b.bid, "name": b.name, "site": b.site,
            "scope_id": b.scope_id, "scope_kind": self._scope_kind(b.scope_id),
            "reason": "shadowed by a later-created visible binding scope"
                      if visible and b is not visible[0] else "visible",
        } for b in candidates]
        chosen = visible[0] if visible else None
        edge = {
            "edge_id": f"be-{len(self.binding_edges):05d}",
            "oid": ident.oid,
            "symbol": sym,
            "scopes": sorted(ident.scopes),
            "status": "resolved" if chosen else "unknown",
            "binding_id": chosen.bid if chosen else None,
            "site": chosen.site if chosen else None,
            "candidates": rejected,
            "call_stack": list(self.call_stack),
            "rule": "capture decided by scope-set membership and creation order, "
                    "not by the identifier's spelling",
            "span": ident.span.as_dict() if ident.span else None,
            "birth": ident.birth.as_dict() if ident.birth else None,
        }
        self.binding_edges.append(edge)
        return chosen, edge

    def _scope_order(self, sid: str) -> int:
        for s in self.scopes:
            if s.sid == sid:
                return s.order
        return -1

    def _scope_kind(self, sid: str) -> str:
        for s in self.scopes:
            if s.sid == sid:
                return s.kind
        return "?"

    # ----- marks ------------------------------------------------------
    def _apply_mark(self, root: Syntax, mark: str, stage: str):
        touched = []
        for s in _walk_unique(root):
            present = s.toggle_scope(mark)
            touched.append({"oid": s.oid, "symbol": s.symbol, "present": present,
                            "scopes": sorted(s.scopes)})
        self._step(f"macro mark {stage}", detail={"mark": mark, "stage": stage,
                                                  "objects": touched})

    # ----- top level --------------------------------------------------
    def expand_program(self, forms):
        tree_children = []
        expanded_forms = []
        for form in forms:
            if self.status != "ok":
                break
            try:
                out, node = self._expand(form, [], self.program_scope)
                expanded_forms.append(out)
                tree_children.append(node)
            except BudgetExceeded as exc:
                self.status = "budget_exceeded"
                self.failure = {"kind": "budget", "budget": exc.budget,
                                "calls_used": self.calls_used,
                                "msg": str(exc),
                                "partial_forms_expanded": len(expanded_forms)}
                break
            except ExpandError as exc:
                self.status = "error"
                self.failure = {"kind": "expand_error", "msg": exc.msg,
                                "oid": exc.obj.oid if exc.obj else None,
                                "span": exc.obj.span.as_dict() if exc.obj and exc.obj.span else None,
                                "partial_forms_expanded": len(expanded_forms)}
                break
        tree = TreeNode("tr-root", "program (syntax object expansion)", "root",
                        "", None, tree_children,
                        {"status": self.status, "program_scope": self.program_scope})
        return expanded_forms, tree

    def _tree_node(self, label, kind, obj, detail=None, children=None):
        self._node_seq += 1
        return TreeNode(f"tn-{self._node_seq:05d}", label, kind, obj.oid,
                        obj.span.as_dict() if obj.span else None, children or [],
                        detail or {})

    def _expand(self, obj: Syntax, local_bindings, enclosing_scope):
        if not obj.is_list:
            sym = obj.symbol
            if sym is None:
                return obj, self._tree_node(repr(obj.val), "literal", obj)
            chosen, edge = self.resolve(obj)
            kind = "use"
            detail = {"status": edge["status"], "binding_id": edge["binding_id"],
                      "scopes": sorted(obj.scopes), "birth": edge["birth"]}
            return obj, self._tree_node(f"{sym}", kind, obj, detail)

        if not obj.val:
            return obj, self._tree_node("()", "literal", obj)

        head = obj.val[0]
        head_sym = head.symbol

        if head_sym in TEMPLATE_ONLY:
            raise ExpandError(f"{head_sym} is a macro-template primitive and may not "
                              f"appear in user programs", obj)

        if head_sym == "quote":
            node = self._tree_node("(quote …)", "quote", obj,
                                   {"datum_retained": True})
            return obj, node

        if head_sym == "define-syntax":
            return self._expand_define_syntax(obj)

        if head_sym == "define":
            return self._expand_define(obj, local_bindings, enclosing_scope)

        if head_sym == "lambda":
            return self._expand_lambda(obj, local_bindings, enclosing_scope)

        if head_sym == "let":
            return self._expand_let(obj, local_bindings, enclosing_scope)

        if head_sym == "if":
            return self._expand_subforms(obj, "if", local_bindings, enclosing_scope)

        if head_sym == "begin":
            return self._expand_subforms(obj, "begin", local_bindings, enclosing_scope)

        if head_sym == "set!":
            return self._expand_set(obj, local_bindings, enclosing_scope)

        # macro invocation?
        macro = None
        if head_sym is not None:
            mb = self._lookup_macro_binding(head)
            if mb is not None:
                macro = self.macros.get(mb.name)
        if macro is not None:
            expanded = self._invoke_macro(obj, macro)
            return self._expand(expanded, local_bindings, enclosing_scope)

        return self._expand_subforms(obj, "application", local_bindings, enclosing_scope)

    def _lookup_macro_binding(self, ident: Syntax):
        chosen, _ = self.resolve(ident)
        if chosen is not None and chosen.kind == "macro":
            return chosen
        return None

    def lookup_binding(self, ident: Syntax):
        """Resolve without appending a new edge (edges are recorded at expand time)."""
        sym = ident.symbol
        visible = [b for b in self.bindings
                   if b.name == sym and b.scope_id in ident.scopes]
        visible.sort(key=lambda b: (self._scope_order(b.scope_id), b.order), reverse=True)
        return visible[0] if visible else None

    def _expand_subforms(self, obj, label, local_bindings, enclosing_scope):
        children = []
        new_elems = []
        for i, sub in enumerate(obj.val):
            out, node = self._expand(sub, local_bindings, enclosing_scope)
            new_elems.append(out)
            children.append(node)
        new_obj = Syntax(new_elems, True, obj.span, set(obj.scopes),
                         birth=obj.birth, serial=obj.serial)
        new_obj.origins = list(obj.origins)
        new_obj.mark_touches = list(obj.mark_touches)
        self.reg.record(new_obj)
        return new_obj, self._tree_node(f"({label} …)", "core", new_obj,
                                        {"head": obj.val[0].symbol}, children)

    def _expand_set(self, obj, local_bindings, enclosing_scope):
        if len(obj.val) != 3 or obj.val[1].symbol is None:
            raise ExpandError("set! requires exactly (set! identifier expr)", obj)
        ident = obj.val[1]
        chosen, edge = self.resolve(ident)
        val_out, val_node = self._expand(obj.val[2], local_bindings, enclosing_scope)
        elems = [obj.val[0], ident, val_out]
        new_obj = Syntax(elems, True, obj.span, set(obj.scopes), birth=obj.birth)
        self.reg.record(new_obj)
        node = self._tree_node("(set! …)", "core", new_obj,
                               {"target_status": edge["status"],
                                "binding_id": edge["binding_id"]}, [val_node])
        return new_obj, node

    def _expand_define(self, obj, local_bindings, enclosing_scope):
        if len(obj.val) < 3:
            raise ExpandError("define requires a name and a body", obj)
        target = obj.val[1]
        if target.is_list:
            # sugar: (define (f x) body) -> (define f (lambda (x) body))
            if not target.val or target.val[0].symbol is None:
                raise ExpandError("malformed function definition", obj)
            name = target.val[0]
            params = target.val[1:]
            body = obj.val[2:]
            lam_tmpl = self._build_lambda_syntax(params, body, obj)
            return self._expand_define_value(name, lam_tmpl, obj, local_bindings,
                                             enclosing_scope, sugar=True)
        if target.symbol is None:
            raise ExpandError("define target must be an identifier", obj)
        return self._expand_define_value(target, obj.val[2], obj, local_bindings,
                                         enclosing_scope, sugar=False)

    def _build_lambda_syntax(self, params, body, origin):
        pnode = Syntax(list(params), True, None, birth=Birth("template",
                                                             detail="define-sugar-params"))
        self.reg.record(pnode)
        lam = Syntax([self._core_symbol("lambda", origin), pnode] + list(body), True,
                     origin.span, birth=Birth("template", detail="define->lambda sugar"))
        self.reg.record(lam)
        return lam

    def _core_symbol(self, name: str, origin: Syntax) -> Syntax:
        s = Syntax(name, False, origin.val[0].span if origin.is_list else origin.span,
                   birth=Birth("template", detail=f"core:{name}"))
        self.reg.record(s)
        return s

    def _expand_define_value(self, name, val_obj, whole, local_bindings,
                             enclosing_scope, sugar):
        bid_scope = self.program_scope
        name.add_scope(bid_scope)
        bi = self.new_binding(name.symbol, bid_scope, "definition", name.oid, "value")
        self.global_env[name.symbol] = bid_scope
        val_out, val_node = self._expand(val_obj, local_bindings, enclosing_scope)
        elems = [whole.val[0], name, val_out]
        new_obj = Syntax(elems, True, whole.span, set(whole.scopes), birth=whole.birth)
        self.reg.record(new_obj)
        chosen, edge = self.resolve(name)
        node = self._tree_node(f"(define {name.symbol} …)", "define", new_obj,
                               {"binding_id": bi.bid, "scope_id": bid_scope,
                                "sugar": sugar, "definition_site": name.span.as_dict()},
                               [val_node])
        return new_obj, node

    def _expand_lambda(self, obj, local_bindings, enclosing_scope):
        if len(obj.val) < 3 or not obj.val[1].is_list:
            raise ExpandError("lambda requires a parameter list and a body", obj)
        params = obj.val[1].val
        param_scopes = []
        for p in params:
            if p.symbol is None:
                raise ExpandError("lambda parameters must be identifiers", p)
            scope = self.new_scope("bind", f"B:lambda:{p.symbol}",
                                   {"site": "use-site", "kind": "lambda-param"})
            p.add_scope(scope)
            bi = self.new_binding(p.symbol, scope, "use-site-lambda", p.oid)
            param_scopes.append(scope)
        self._add_scopes(obj.val[2:], param_scopes)
        body_children = []
        new_bodies = []
        for b in obj.val[2:]:
            out, n = self._expand(b, local_bindings, param_scopes[-1] if param_scopes
                                  else enclosing_scope)
            new_bodies.append(out)
            body_children.append(n)
        elems = [obj.val[0], obj.val[1]] + new_bodies
        new_obj = Syntax(elems, True, obj.span, set(obj.scopes), birth=obj.birth)
        self.reg.record(new_obj)
        node = self._tree_node(
            "(lambda (" + " ".join(p.symbol for p in params) + ") …)", "lambda",
            new_obj, {"param_scopes": param_scopes,
                      "definition_use_note": "use-site scope added to bodies only"},
            body_children)
        return new_obj, node

    # ----- macro definition ------------------------------------------
    def _expand_define_syntax(self, obj):
        if len(obj.val) != 3:
            raise ExpandError("define-syntax requires a name and one syntax-rules", obj)
        name = obj.val[1]
        sr = obj.val[2]
        if name.symbol is None or not sr.is_list or sr.val[0].symbol != "syntax-rules":
            raise ExpandError("expected (define-syntax name (syntax-rules (lit...) ...))",
                              obj)
        if len(sr.val) < 2 or not sr.val[1].is_list:
            raise ExpandError("syntax-rules requires a literal list", sr)
        lits = [t.symbol for t in sr.val[1].val if t.symbol is not None]
        clauses = []
        for idx, cl in enumerate(sr.val[2:]):
            if not cl.is_list or len(cl.val) != 2:
                raise ExpandError("each syntax-rules clause needs a pattern and template",
                                  cl)
            clauses.append(Clause(idx, cl.val[0], cl.val[1]))
        bi = self.new_binding(name.symbol, self.program_scope, "definition", name.oid,
                              kind="macro")
        macro = MacroInfo(name=name.symbol, literals=lits, clauses=clauses, binding=bi,
                          definition_oid=obj.oid,
                          frozen_scope_note=("clause templates are frozen with the "
                                             "definition-site scope set "
                                             f"{sorted(name.scopes)}"))
        self.macros[name.symbol] = macro
        chosen, edge = self.resolve(name)
        node = self._tree_node(f"(define-syntax {name.symbol} …)", "define-syntax", obj,
                               {"binding_id": bi.bid, "literals": lits,
                                "clause_count": len(clauses),
                                "definition_site": name.span.as_dict(),
                                "frozen_scopes": sorted(name.scopes),
                                "frozen_scope_note": macro.frozen_scope_note})
        return obj, node

    # ----- macro invocation ------------------------------------------
    def _invoke_macro(self, call: Syntax, macro: MacroInfo) -> Syntax:
        self._call_seq += 1
        call_oid = f"mc-{self._call_seq:04d}"
        self.call_stack.append(call_oid)
        self.calls_used += 1
        if self.calls_used > self.budget:
            self.macro_calls.append({
                "call_oid": call_oid, "macro": macro.name,
                "status": "budget_exceeded",
                "call_stack": list(self.call_stack),
                "definition_oid": macro.definition_oid,
                "use_site": call.span.as_dict() if call.span else None,
                "input_snapshot": _snapshot(call),
                "budget": self.budget, "calls_used": self.calls_used,
            })
            self.call_stack.pop()
            raise BudgetExceeded(self.budget)

        mark = self.new_scope("intro", f"M:{macro.name}#{self._call_seq}",
                              {"macro": macro.name, "call_oid": call_oid,
                               "why": "fresh introduction mark; kept by template "
                                      "objects, dropped by passed-through input"})

        candidates = []
        match_env = None
        for cl in macro.clauses:
            try:
                env = match_pattern(cl.pattern, call, set(macro.literals))
                candidates.append({"clause": cl.index, "status": "legal",
                                   "reason": "pattern matches"})
                if match_env is None:
                    match_env = env
            except MatchFail as fail:
                candidates.append({"clause": cl.index, "status": "rejected",
                                   "reason": f"pattern mismatch: {fail.reason}",
                                   "span": fail.obj.span.as_dict() if fail.obj and
                                   fail.obj.span else None})
        legal = [c["clause"] for c in candidates if c["status"] == "legal"]
        if not legal:
            rec = self._call_record(call_oid, macro, call, mark, candidates, None,
                                    "no_match", before=None)
            self.macro_calls.append(rec)
            self.call_stack.pop()
            raise ExpandError(f"no syntax-rules clause of {macro.name!r} matches the call",
                              call)

        chosen_clause = legal[0]
        override = self.clause_overrides.get(call_oid)
        override_note = None
        if override is not None:
            if override in legal:
                chosen_clause = override
                override_note = f"human decision selected clause {override}"
            else:
                override_note = (f"human decision asked for clause {override} but it is "
                                 f"not a legal match here; ignored")
        for c in candidates:
            if c["status"] == "legal" and c["clause"] != chosen_clause:
                c["reason"] = ("kept legal candidate; clause "
                               f"{chosen_clause} used by order tie-break/decision")

        # re-match with the chosen clause so bindings reflect it exactly
        chosen = macro.clauses[chosen_clause]
        env = match_pattern(chosen.pattern, call, set(macro.literals))
        env["_"] = call

        before = _snapshot(call)
        self._step(f"enter macro {macro.name} [{call_oid}] clause {chosen_clause}",
                   before=before,
                   detail={"macro": macro.name, "call_oid": call_oid,
                           "definition_oid": macro.definition_oid,
                           "use_site": call.span.as_dict() if call.span else None,
                           "call_stack": list(self.call_stack),
                           "legal_clauses": legal})
        # step 1: fresh mark on the whole invocation input
        self._apply_mark(call, mark, "pre: add to every input object")

        local_events: list[dict] = []
        output = instantiate(chosen.template, env, self.reg, call_oid=call_oid,
                             macro_name=macro.name, events=local_events,
                             clock=self.clock)
        for ev in local_events:
            ev["clause"] = chosen_clause
            if ev["kind"] == "context_choice":
                self.context_choices.append(ev)
            else:
                self.gensyms.append(ev)

        # step 2: same mark toggled again on output; objects from the input
        # lose it, objects introduced by the template keep it.
        self._apply_mark(output, mark, "post: toggle on result (flip)")

        ambiguous = len(legal) > 1
        rec = self._call_record(
            call_oid, macro, call, mark, candidates, chosen_clause,
            "ambiguous" if ambiguous else "ok", before=before, after=_snapshot(output),
            output_oid=output.oid, override_note=override_note,
            local_events=local_events)
        self.macro_calls.append(rec)
        self._step(f"leave macro {macro.name} [{call_oid}]", before=before,
                   after=_snapshot(output),
                   detail={"call_oid": call_oid, "chosen_clause": chosen_clause,
                           "ambiguous": ambiguous, "call_stack": list(self.call_stack),
                           "hygiene_rule": "capture is scope-set membership; strings "
                                           "are never renamed"})
        self.call_stack.pop()
        return output

    def _call_record(self, call_oid, macro, call, mark, candidates, chosen_clause,
                     status, *, before, after=None, output_oid=None,
                     override_note=None, local_events=None):
        return {
            "call_oid": call_oid,
            "macro": macro.name,
            "status": status,
            "macro_binding_id": macro.binding.bid,
            "definition_oid": macro.definition_oid,
            "definition_site": macro.binding.site,
            "use_site": call.span.as_dict() if call.span else None,
            "use_oid": call.oid,
            "mark_scope": mark,
            "call_stack": list(self.call_stack),
            "candidates": candidates,
            "legal_clauses": [c["clause"] for c in candidates if c["status"] == "legal"],
            "chosen_clause": chosen_clause,
            "ambiguous": status == "ambiguous",
            "override_note": override_note,
            "budget": self.budget,
            "calls_used": self.calls_used,
            "before": before,
            "after": after,
            "output_oid": output_oid,
            "events_during_call": local_events or [],
        }

    # ----- primitives & artifact -------------------------------------
    PRIMITIVES = ["+", "-", "*", "=", "<", ">", "cons", "car", "cdr", "list",
                  "null?", "display", "eq?"]

    def seed_primitives(self):
        for p in self.PRIMITIVES:
            ident = Syntax(p, False, None,
                           birth=Birth("parsed", detail="builtin-primitive"))
            self.reg.record(ident)
            ident.add_scope(self.program_scope)
            self.new_binding(p, self.program_scope, "global-prim", ident.oid, "value")

    def artifact(self, tokens, source_text, tree) -> dict:
        syntax_objs = sorted(self.reg.objs.values(),
                             key=lambda s: (s.serial is None, s.serial or 0))
        return {
            "algorithm_version": self.algorithm_version,
            "status": self.status,
            "failure": self.failure,
            "budget": self.budget,
            "calls_used": self.calls_used,
            "program_scope": self.program_scope,
            "tokens": [t.as_dict() for t in tokens],
            "source_text": source_text,
            "scopes": [s.as_dict() for s in self.scopes],
            "bindings": [b.as_dict() for b in self.bindings],
            "binding_edges": self.binding_edges,
            "macro_calls": self.macro_calls,
            "steps": self.steps,
            "context_choices": self.context_choices,
            "gensyms": self.gensyms,
            "syntax_objects": [s.as_dict() for s in syntax_objs],
            "tree": tree.as_dict(),
        }

    def _add_scopes(self, objs, scopes):
        stack = list(objs)
        while stack:
            s = stack.pop()
            for sc in scopes:
                s.add_scope(sc)
            if s.is_list:
                stack.extend(s.val)

    def _expand_let(self, obj, local_bindings, enclosing_scope):
        if len(obj.val) != 3 or not obj.val[1].is_list:
            raise ExpandError("let requires bindings and one body: (let ((x e)) body)",
                              obj)
        bind_scope = self.new_scope("bind", "B:let", {"site": "use-site",
                                                      "kind": "let"})
        binding_nodes = []
        names = []
        for clause in obj.val[1].val:
            if not clause.is_list or len(clause.val) != 2 or clause.val[0].symbol is None:
                raise ExpandError("malformed let binding", clause)
            name, expr = clause.val
            name.add_scope(bind_scope)
            bi = self.new_binding(name.symbol, bind_scope, "use-site-let", name.oid)
            out, n = self._expand(expr, local_bindings, enclosing_scope)
            names.append((name, out))
            binding_nodes.append(n)
        body = obj.val[2]
        body.add_scope(bind_scope)
        self._add_scopes([body], [bind_scope])
        body_out, body_node = self._expand(body, local_bindings, bind_scope)
        pairs = Syntax([Syntax([nm, ex], True, c.span, birth=c.birth)
                        for (nm, ex), c in zip(names, obj.val[1].val)], True,
                       obj.val[1].span, birth=obj.val[1].birth)
        self.reg.record(pairs)
        for pair in pairs.val:
            self.reg.record(pair)
        new_obj = Syntax([obj.val[0], pairs, body_out], True, obj.span,
                         set(obj.scopes), birth=obj.birth)
        self.reg.record(new_obj)
        node = self._tree_node(
            "(let (" + " ".join(nm.symbol for nm, _ in names) + ") …)", "let",
            new_obj, {"bind_scope": bind_scope,
                      "note": "init expressions cannot see each other; body can"},
            binding_nodes + [body_node])
        return new_obj, node
