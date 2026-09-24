"""Hygienic macro expander for the tiny language.

Hygiene model (scope sets, not string renaming):
  - each binding carries a scope set;
  - entering a binder (lambda/let) adds one fresh scope to the whole body;
  - every macro definition adds a definition-site scope to its template;
  - every macro use adds a fresh per-use scope to template-introduced nodes;
  - substituted arguments keep their original scope set;
  - an identifier resolves to a binding when the binding's scope set is a
    subset of the identifier's scope set, choosing the maximal set.
    Equal-set ties resolve to the most recent binding (shadowing) while all
    candidates are kept; incomparable maximal sets are reported ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .syntax import NodeIds, Span, Syntax, read

ALGO_VERSION = "macrocore/1.0.0"

CORE_FORMS = {"lambda", "let", "if", "define", "set!", "begin",
              "quote", "defmacro"}
BUILTINS = ["+", "-", "*", "/", "=", "<", ">", "not", "and", "or",
            "cons", "car", "cdr", "list", "print"]


@dataclass
class Binding:
    id: int
    name: str
    scopes: frozenset
    kind: str  # 'builtin' | 'var' | 'macro'
    span: Optional[Span]
    node_id: Optional[int]


@dataclass
class MacroClause:
    params: list[str]
    rest: Optional[str]
    template: Syntax


@dataclass
class MacroDef:
    name: str
    clauses: list[MacroClause]
    def_span: Optional[Span]
    def_scope: int
    version: str = "v1"


@dataclass
class Step:
    id: int
    macro: str
    input_id: int
    output_id: Optional[int]
    use_scope: int
    call_span: Optional[Span]
    def_span: Optional[Span]
    stack: list[int]
    parent: Optional[int]
    status: str  # 'ok' | 'budget-exceeded' | 'arity-error' | 'template-error'


@dataclass
class Diagnostic:
    kind: str
    message: str
    node_id: Optional[int]
    span: Optional[Span]
    macro_def_span: Optional[Span] = None
    use_site_span: Optional[Span] = None
    stack: list[int] = field(default_factory=list)


@dataclass
class Edge:
    node_id: int
    name: str
    role: str  # 'reference' | 'binder'
    status: str  # resolved | resolved-shadow | ambiguous | unbound
    binding_id: Optional[int]
    candidates: list[int]
    scopes: list[int]
    span: Optional[Span]
    prov: Optional[int]


class Expander:
    def __init__(self, budget: int = 128, source: str = "",
                 macro_version: str = "v1"):
        self.ids = NodeIds()
        self.nodes: dict[int, Syntax] = {}
        self.bindings: list[Binding] = []
        self.macros: dict[str, MacroDef] = {}
        self.steps: list[Step] = []
        self.edges: list[Edge] = []
        self.diagnostics: list[Diagnostic] = []
        self.binder_nodes: dict[int, int] = {}
        self.scope_counter = 0
        self.gensym_counter = 0
        self.budget = budget
        self.macro_steps = 0
        self.failed = False
        self.call_stack: list[int] = []
        self.source = source
        self.macro_version = macro_version
        for name in BUILTINS:
            self._bind(name, frozenset(), "builtin", None, None)

    # ---- id / scope / node allocation -------------------------------------
    def fresh_scope(self) -> int:
        self.scope_counter += 1
        return self.scope_counter

    def _new(self, datum, scopes, span=None, origin="source",
             prov=None) -> Syntax:
        node = Syntax(self.ids.take(), datum, frozenset(scopes),
                      span, origin, prov)
        self.nodes[node.id] = node
        return node

    def _register_tree(self, node: Syntax) -> Syntax:
        self.nodes[node.id] = node
        if isinstance(node.datum, list):
            for child in node.datum:
                self._register_tree(child)
        return node

    def _add_scope(self, node: Syntax, scope: int) -> Syntax:
        """Return a copy of the tree with `scope` added to every node."""
        if isinstance(node.datum, list):
            return self._new(
                [self._add_scope(c, scope) for c in node.datum],
                set(node.scopes) | {scope}, node.span, node.origin, node.prov)
        return self._new(node.datum, set(node.scopes) | {scope},
                         node.span, node.origin, node.prov)

    def _bind(self, name, scopes, kind, span, node_id) -> Binding:
        binding = Binding(len(self.bindings) + 1, name, frozenset(scopes),
                          kind, span, node_id)
        self.bindings.append(binding)
        if node_id is not None:
            self.binder_nodes[node_id] = binding.id
        return binding

    def _resolve(self, name: str, scopes: frozenset):
        candidates = [b for b in self.bindings
                      if b.name == name and b.scopes <= scopes]
        if not candidates:
            return None, []
        maximal = [b for b in candidates
                   if not any(b.scopes < c.scopes for c in candidates)]
        best = max(maximal, key=lambda b: b.id)
        return best, maximal

    # ---- macro definitions and template instantiation ---------------------
    def _parse_params(self, node: Syntax):
        params: list[str] = []
        rest = None
        items = list(node.datum)
        idx = 0
        while idx < len(items):
            part = items[idx]
            if part.is_ident and part.datum == ".":
                idx += 1
                rest = items[idx].datum
                idx += 1
                continue
            params.append(part.datum)
            idx += 1
        return params, rest

    def _make_clause(self, params_node: Syntax, template: Syntax,
                     def_scope: int) -> MacroClause:
        params, rest = self._parse_params(params_node)
        template_d = self._add_scope(template, def_scope)
        return MacroClause(params, rest, template_d)

    def _defmacro(self, stx: Syntax):
        parts = stx.datum[1:]
        name = parts[0].datum
        body = parts[1:]
        def_scope = self.fresh_scope()
        clauses: list[MacroClause] = []
        multi = (body and isinstance(body[0].datum, list)
                 and body[0].datum
                 and isinstance(body[0].datum[0].datum, list))
        if multi:
            for clause in body:
                clauses.append(self._make_clause(
                    clause.datum[0], clause.datum[1], def_scope))
        else:
            clauses.append(self._make_clause(body[0], body[1], def_scope))
        self.macros[name] = MacroDef(name, clauses, stx.span,
                                     def_scope, self.macro_version)
        return self._new(f"<<defmacro {name}>>", stx.datum[0].scopes,
                         stx.span, "source", None)

    def _instantiate(self, tmpl: Syntax, args: dict, rest_args,
                     rest_name, use_scope: int, step_id: int):
        datum = tmpl.datum
        if tmpl.is_ident and datum.startswith("$"):
            key = datum[1:]
            if key in args:
                return args[key]
            if rest_name is not None and key == rest_name:
                self.diagnostics.append(Diagnostic(
                    "template-error",
                    f"rest parameter ${rest_name} used outside splice",
                    tmpl.id if tmpl.id in self.nodes else None,
                    tmpl.span, stack=list(self.call_stack)))
                return tmpl
            self.diagnostics.append(Diagnostic(
                "template-error", f"unknown template parameter ${key}",
                tmpl.id if tmpl.id in self.nodes else None,
                tmpl.span, stack=list(self.call_stack)))
            return tmpl
        if isinstance(datum, list):
            if datum and datum[0].is_ident:
                head = datum[0].datum
                if head == "datum->syntax":
                    ctx = self._instantiate(datum[1], args, rest_args,
                                            rest_name, use_scope, step_id)
                    name = datum[2].datum
                    return self._new(name, ctx.scopes, ctx.span,
                                     "generated", step_id)
                if head == "gensym":
                    base = datum[1].datum if len(datum) > 1 else "g"
                    self.gensym_counter += 1
                    return self._new(f"{base}#g{self.gensym_counter}",
                                     frozenset({self.fresh_scope()}), None,
                                     "generated", step_id)
            children: list[Syntax] = []
            for el in datum:
                if (el.is_ident and rest_name is not None
                        and el.datum == "$" + rest_name):
                    children.extend(rest_args)
                else:
                    children.append(self._instantiate(
                        el, args, rest_args, rest_name, use_scope, step_id))
            return self._new(children, set(tmpl.scopes) | {use_scope},
                             tmpl.span, "macro", step_id)
        return self._new(datum, set(tmpl.scopes) | {use_scope},
                         tmpl.span, "macro", step_id)

    def _expand_macro(self, stx: Syntax) -> Syntax:
        macro = self.macros[stx.datum[0].datum]
        call_args = stx.datum[1:]
        clause = None
        for candidate in macro.clauses:
            nmin = len(candidate.params)
            if candidate.rest is None and len(call_args) == nmin:
                clause = candidate
                break
            if candidate.rest is not None and len(call_args) >= nmin:
                clause = candidate
                break
        step = Step(len(self.steps) + 1, macro.name, stx.id, None,
                    self.scope_counter + 1, stx.span, macro.def_span,
                    list(self.call_stack),
                    self.call_stack[-1] if self.call_stack else None,
                    "ok")
        if clause is None:
            step.status = "arity-error"
            self.steps.append(step)
            self.diagnostics.append(Diagnostic(
                "arity-error",
                f"no clause of macro {macro.name} matches {len(call_args)} args",
                stx.id, stx.span, macro.def_span, stx.span,
                list(self.call_stack)))
            return stx
        if self.macro_steps >= self.budget:
            step.status = "budget-exceeded"
            self.steps.append(step)
            self.failed = True
            self.diagnostics.append(Diagnostic(
                "budget-exceeded",
                f"expansion budget of {self.budget} steps exhausted "
                f"while expanding {macro.name}",
                stx.id, stx.span, macro.def_span, stx.span,
                list(self.call_stack)))
            return stx
        self.steps.append(step)
        self.macro_steps += 1
        use_scope = self.fresh_scope()
        step.use_scope = use_scope
        args = dict(zip(clause.params, call_args[:len(clause.params)]))
        rest_args = call_args[len(clause.params):] if clause.rest else []
        self.call_stack.append(step.id)
        body = self._instantiate(clause.template, args, rest_args,
                                 clause.rest, use_scope, step.id)
        result = self._expand(body)
        self.call_stack.pop()
        step.output_id = result.id
        return result

    # ---- core forms -------------------------------------------------------
    def _rebuild_list(self, stx: Syntax, children: list[Syntax]) -> Syntax:
        return self._new(children, stx.scopes, stx.span,
                         stx.origin, stx.prov)

    def _expand_lambda(self, stx: Syntax):
        scope = self.fresh_scope()
        params_node = stx.datum[1]
        new_params: list[Syntax] = []
        for param in params_node.datum:
            param2 = self._new(param.datum, set(param.scopes) | {scope},
                               param.span, param.origin, param.prov)
            self._bind(param2.datum, param2.scopes, "var",
                       param.span, param2.id)
            new_params.append(param2)
        body_forms = [self._expand(self._add_scope(f, scope))
                      for f in stx.datum[2:]]
        return self._rebuild_list(stx, [
            self._new("lambda", stx.datum[0].scopes,
                      stx.datum[0].span, "source", None),
            self._new(new_params, params_node.scopes, params_node.span,
                      params_node.origin, params_node.prov)] + body_forms)

    def _expand_let(self, stx: Syntax):
        scope = self.fresh_scope()
        bindings_node = stx.datum[1]
        expanded_bindings: list[Syntax] = []
        pairs: list[tuple[Syntax, Syntax]] = []
        for pair in bindings_node.datum:
            name_node = pair.datum[0]
            value_node = self._expand(pair.datum[1])
            pairs.append((name_node, value_node))
        for name_node, value_node in pairs:
            name2 = self._new(name_node.datum,
                             set(name_node.scopes) | {scope},
                             name_node.span, name_node.origin, name_node.prov)
            self._bind(name2.datum, name2.scopes, "var",
                       name_node.span, name2.id)
            expanded_bindings.append(self._new(
                [name2, value_node], pair.scopes, pair.span,
                pair.origin, pair.prov))
        body_forms = [self._expand(self._add_scope(f, scope))
                      for f in stx.datum[2:]]
        return self._rebuild_list(stx, [
            self._new("let", stx.datum[0].scopes,
                      stx.datum[0].span, "source", None),
            self._new(expanded_bindings, bindings_node.scopes,
                      bindings_node.span, bindings_node.origin,
                      bindings_node.prov)] + body_forms)

    def _expand(self, stx: Syntax) -> Syntax:
        datum = stx.datum
        if not isinstance(datum, list) or not datum:
            return stx
        head = datum[0]
        if head.is_ident:
            name = head.datum
            if name == "quote":
                return stx
            if name == "defmacro":
                return self._defmacro(stx)
            if name == "lambda":
                return self._expand_lambda(stx)
            if name == "let":
                return self._expand_let(stx)
            if name in ("if", "set!", "begin"):
                children = [self._new(name, head.scopes, head.span,
                                      "source", None)]
                children += [self._expand(child) for child in datum[1:]]
                return self._rebuild_list(stx, children)
            if name == "define":
                value = self._expand(datum[2])
                name_node = datum[1]
                self._bind(name_node.datum, name_node.scopes, "var",
                           name_node.span, name_node.id)
                return self._rebuild_list(stx, [
                    self._new("define", head.scopes, head.span,
                              "source", None), name_node, value])
            if name in self.macros:
                return self._expand_macro(stx)
        children = [self._expand(child) for child in datum]
        return self._rebuild_list(stx, children)

    # ---- final edge pass and serialization -------------------------------
    def _walk_edges(self, node: Syntax):
        datum = node.datum
        if isinstance(datum, list):
            if datum and datum[0].is_ident and datum[0].datum == "quote":
                return
            for child in datum:
                self._walk_edges(child)
            return
        if not node.is_ident:
            return
        if node.id in self.binder_nodes:
            binding_id = self.binder_nodes[node.id]
            self.edges.append(Edge(
                node.id, datum, "binder", "binding", binding_id,
                [binding_id], sorted(node.scopes), node.span, node.prov))
            return
        if datum in CORE_FORMS or datum.startswith("<<"):
            return
        best, maximal = self._resolve(datum, node.scopes)
        if best is None:
            self.edges.append(Edge(node.id, datum, "reference", "unbound",
                                   None, [], sorted(node.scopes),
                                   node.span, node.prov))
            def_span = use_span = None
            stack = []
            if node.prov is not None:
                step = {s.id: s for s in self.steps}.get(node.prov)
                if step is not None:
                    def_span, use_span = step.def_span, step.call_span
                    stack = list(step.stack)
            self.diagnostics.append(Diagnostic(
                "unbound", f"unbound identifier {datum}", node.id,
                node.span, def_span, use_span, stack))
            return
        if len(maximal) == 1:
            status = "resolved"
        elif len({b.scopes for b in maximal}) == 1:
            status = "resolved-shadow"
        else:
            status = "ambiguous"
        self.edges.append(Edge(
            node.id, datum, "reference", status, best.id,
            [b.id for b in maximal], sorted(node.scopes),
            node.span, node.prov))

    def expand_program(self, source: str):
        forms, tokens, _ = read(source, self.ids)
        for form in forms:
            self._register_tree(form)
        roots = [self._expand(form) for form in forms]
        for root in roots:
            self._walk_edges(root)
        return roots, tokens

    # ---- serialization ----------------------------------------------------
    def _span(self, span: Optional[Span]):
        return span.to_dict() if span is not None else None

    def _node_dict(self, node: Syntax) -> dict:
        if isinstance(node.datum, list):
            kind, label, children = "list", None, [c.id for c in node.datum]
        else:
            kind = "ident" if node.is_ident else "literal"
            label = repr(node.datum) if not node.is_ident else node.datum
            children = []
        return {"id": node.id, "kind": kind, "label": label,
                "children": children, "scopes": sorted(node.scopes),
                "span": self._span(node.span), "origin": node.origin,
                "prov": node.prov}

    def result(self, roots: list[Syntax], tokens) -> dict:
        return {
            "algo_version": ALGO_VERSION,
            "macro_version": self.macro_version,
            "status": "failed" if self.failed else "ok",
            "budget": {"value": self.budget, "unit": "macro-steps"},
            "macro_steps": self.macro_steps,
            "nodes": [self._node_dict(n) for n in self.nodes.values()],
            "roots": [r.id for r in roots],
            "tokens": [t.to_dict() for t in tokens],
            "bindings": [{"id": b.id, "name": b.name,
                          "scopes": sorted(b.scopes), "kind": b.kind,
                          "span": self._span(b.span),
                          "node_id": b.node_id} for b in self.bindings],
            "edges": [{"node_id": e.node_id, "name": e.name,
                       "role": e.role, "status": e.status,
                       "binding_id": e.binding_id,
                       "candidates": e.candidates, "scopes": e.scopes,
                       "span": self._span(e.span), "prov": e.prov}
                      for e in self.edges],
            "steps": [{"id": s.id, "macro": s.macro,
                       "input_id": s.input_id, "output_id": s.output_id,
                       "use_scope": s.use_scope,
                       "call_span": self._span(s.call_span),
                       "def_span": self._span(s.def_span),
                       "stack": s.stack, "parent": s.parent,
                       "status": s.status} for s in self.steps],
            "diagnostics": [{"kind": d.kind, "message": d.message,
                             "node_id": d.node_id,
                             "span": self._span(d.span),
                             "macro_def_span": self._span(d.macro_def_span),
                             "use_site_span": self._span(d.use_site_span),
                             "stack": d.stack} for d in self.diagnostics],
        }


def expand_source(source: str, budget: int = 128,
                  macro_version: str = "v1") -> tuple[dict, "Expander"]:
    """Run the full pipeline: read -> expand -> bind -> serialize."""
    expander = Expander(budget=budget, source=source,
                        macro_version=macro_version)
    roots, tokens = expander.expand_program(source)
    return expander.result(roots, tokens), expander


def _binding_descriptor(result: dict, binding_id):
    for binding in result["bindings"]:
        if binding["id"] == binding_id:
            return {
                "name": binding["name"],
                "kind": binding["kind"],
                "span_start": binding["span"]["start"]
                if binding["span"] else None,
            }
    return None


def _edge_key(result: dict, edge: dict):
    """Stable key across macro versions: role+name+span+occurrence index."""
    start = edge["span"]["start"] if edge["span"] else None
    return (edge["role"], edge["name"], start)


def compare_results(result_a: dict, result_b: dict) -> dict:
    """Compare two expansion results edge by edge.

    Edges are keyed independently of per-run node and scope ids so that
    macro-version diffs remain stable across re-expansion.
    """
    def index_edges(result):
        seen: dict[tuple, int] = {}
        indexed = {}
        for edge in result["edges"]:
            base = _edge_key(result, edge)
            seen[base] = seen.get(base, 0) + 1
            key = (base[0], base[1], base[2], seen[base])
            indexed[key] = edge
        return indexed

    edges_a = index_edges(result_a)
    edges_b = index_edges(result_b)
    changes = []
    for key in sorted(set(edges_a) | set(edges_b)):
        ea, eb = edges_a.get(key), edges_b.get(key)
        if ea is None:
            changes.append({"key": list(key), "kind": "added", "b": _view(eb, result_b)})
        elif eb is None:
            changes.append({"key": list(key), "kind": "removed", "a": _view(ea, result_a)})
        else:
            ba = _binding_descriptor(result_a, ea["binding_id"])
            bb = _binding_descriptor(result_b, eb["binding_id"])
            if ea["status"] != eb["status"] or ba != bb:
                changes.append({"key": list(key), "kind": "changed",
                                "a": _view(ea, result_a), "b": _view(eb, result_b)})
    return {
        "a": {"status": result_a["status"],
              "macro_version": result_a.get("macro_version")},
        "b": {"status": result_b["status"],
              "macro_version": result_b.get("macro_version")},
        "changes": changes,
    }


def _view(edge: dict, result: dict) -> dict:
    return {
        "node_id": edge["node_id"], "name": edge["name"],
        "role": edge["role"], "status": edge["status"],
        "binding": _binding_descriptor(result, edge["binding_id"]),
        "candidates": [_binding_descriptor(result, cid)
                       for cid in edge["candidates"]],
        "prov": edge["prov"],
    }
