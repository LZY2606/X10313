"""Hygienic macro expander with provenance and a bounded budget.

The expander instantiates macro templates, attaches macro-introduction scopes,
and records every transformation as a machine-readable event plus an edge in
the expansion graph. Capture is resolved later from scope sets; the expander
never renames identifiers by string comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .build import MODULE_SCOPE, MacroDef
from .syntax import OriginFrame, Syn, Token, stable_id

CORE_FORMS = {
    "quote", "lambda", "if", "mif", "begin", "define", "defmacro", "let",
    "datum->syntax", "join-contexts", "gen-temp", "set!",
}


@dataclass
class Event:
    event_id: str
    type: str
    detail: dict
    step: int

    def to_json(self) -> dict:
        return {"event_id": self.event_id, "type": self.type,
                "detail": self.detail, "step": self.step}


@dataclass
class GraphNode:
    node_id: str
    label: str
    kind: str
    syntax: Syn | None
    invocation_id: str | None = None
    depth: int = 0
    partial: bool = False

    def to_json(self) -> dict:
        return {
            "node_id": self.node_id, "label": self.label, "kind": self.kind,
            "sid": self.syntax.sid if self.syntax else None,
            "invocation_id": self.invocation_id, "depth": self.depth,
            "partial": self.partial,
            "span": self.syntax.first_span().to_json()
            if self.syntax and self.syntax.first_span() else None,
        }


@dataclass
class GraphEdge:
    src: str
    dst: str
    label: str

    def to_json(self) -> dict:
        return {"src": self.src, "dst": self.dst, "label": self.label}


@dataclass
class Diagnostic:
    code: str
    message: str
    span: dict | None
    origin_chain: list[dict]
    invocation_stack: list[str]
    severity: str = "error"
    candidates: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "code": self.code, "message": self.message, "span": self.span,
            "origin_chain": self.origin_chain,
            "invocation_stack": self.invocation_stack,
            "severity": self.severity, "candidates": self.candidates,
        }


@dataclass
class ExpansionResult:
    expanded: list[Syn]
    events: list[Event]
    graph_nodes: list[GraphNode]
    graph_edges: list[GraphEdge]
    diagnostics: list[Diagnostic]
    scopes: list[dict]
    invocations: list[dict]
    steps_used: int
    budget: int
    budget_exhausted: bool
    seed: str

    def to_json(self) -> dict:
        return {
            "expanded": [s.to_json() for s in self.expanded],
            "events": [e.to_json() for e in self.events],
            "graph_nodes": [n.to_json() for n in self.graph_nodes],
            "graph_edges": [e.to_json() for e in self.graph_edges],
            "diagnostics": [d.to_json() for d in self.diagnostics],
            "scopes": self.scopes, "invocations": self.invocations,
            "steps_used": self.steps_used, "budget": self.budget,
            "budget_exhausted": self.budget_exhausted, "seed": self.seed,
        }


class Expander:
    def __init__(self, build_result, budget: int = 64, seed: str = "run",
                 algo_version: str = "expander-1.0.0"):
        self.build = build_result
        self.budget = budget
        self.steps_used = 0
        self.seed = seed
        self.algo_version = algo_version
        self.events: list[Event] = []
        self.graph_nodes: list[GraphNode] = []
        self.graph_edges: list[GraphEdge] = []
        self.diagnostics: list[Diagnostic] = []
        self.invocations: list[dict] = []
        self.scopes = list(build_result.scope_registry)
        self._event_seq = 0
        self._node_seq = 0
        self._scope_seq = 0
        self._sym_seq = 0
        self.budget_exhausted = False
        self._macro_counters: dict[str, int] = {}

    # ---- deterministic identity helpers ------------------------------------
    def _eid(self, prefix: str) -> str:
        self._event_seq += 1
        return f"{prefix}-{self._event_seq:04d}"

    def _nid(self) -> str:
        self._node_seq += 1
        return f"gnode-{self._node_seq:04d}"

    def _fresh_scope(self, kind: str, note: str, invocation_id: str) -> str:
        sid = f"scp-macro-{self._scope_seq}"
        self._scope_seq += 1
        self.scopes.append({"scope_id": sid, "kind": kind, "note": note,
                            "invocation_id": invocation_id})
        return sid

    def _gensym(self, tag: str, invocation_id: str) -> str:
        self._sym_seq += 1
        # Fixed-seed pseudo-randomness: hash of seed + tag + counter.
        suffix = stable_id("gensym", self.seed, tag, self._sym_seq)[:8]
        symbol = f"__{tag}_{suffix}"
        self.record_event("generated-symbol", {
            "symbol": symbol, "tag": tag, "invocation_id": invocation_id,
            "counter": self._sym_seq, "unit": "symbols",
            "seed": self.seed,
        })
        return symbol

    def record_event(self, etype: str, detail: dict) -> Event:
        event = Event(self._eid("evt"), etype, detail, len(self.events) + 1)
        self.events.append(event)
        return event

    def add_node(self, label, kind, syntax, invocation_id=None, depth=0,
                 partial=False) -> str:
        node = GraphNode(self._nid(), label, kind, syntax, invocation_id,
                         depth, partial)
        self.graph_nodes.append(node)
        return node.node_id

    def add_edge(self, src, dst, label):
        if src and dst:
            self.graph_edges.append(GraphEdge(src, dst, label))

    # ---- public entry ------------------------------------------------------
    def run(self) -> ExpansionResult:
        root_id = self.add_node("module", "root", None)
        expanded_forms = []
        for form in self.build.module:
            src_id = self.add_node(self._label(form), "input", form)
            self.add_edge(root_id, src_id, "form")
            out = self.expand_form(form, [], root_id, 0)
            dst_id = self.add_node(self._label(out), "expanded", out)
            self.add_edge(src_id, dst_id, "expands-to")
            expanded_forms.append(out)
        return ExpansionResult(
            expanded=expanded_forms, events=self.events,
            graph_nodes=self.graph_nodes, graph_edges=self.graph_edges,
            diagnostics=self.diagnostics, scopes=self.scopes,
            invocations=self.invocations, steps_used=self.steps_used,
            budget=self.budget, budget_exhausted=self.budget_exhausted,
            seed=self.seed,
        )

    def _label(self, node: Syn) -> str:
        if node.atom() is not None:
            return str(node.datum)
        head = node.children[0].datum if node.children else "()"
        return f"({head} ...)" if node.children else "()"

    # ---- macro-time evaluation ---------------------------------------------
    def meval(self, node: Syn, env: dict[str, int]):
        atom = node.atom()
        if atom is not None:
            if isinstance(atom, (int, float)):
                return ("value", atom)
            if isinstance(atom, str):
                if atom in ("true", "#t"):
                    return ("value", 1)
                if atom in ("false", "#f"):
                    return ("value", 0)
                if atom in env:
                    return ("value", env[atom])
                return ("unknown", atom)
        head = self._head_name(node)
        if head in ("+", "-", "*", "=", "<", ">", "<=", ">="):
            values = []
            for arg in node.children[1:]:
                status, val = self.meval(arg, env)
                if status != "value":
                    return ("unknown", self._label(arg))
                values.append(val)
            try:
                if head == "+":
                    return ("value", sum(values))
                if head == "-":
                    return ("value", values[0] - sum(values[1:]))
                if head == "*":
                    prod = 1
                    for v in values:
                        prod *= v
                    return ("value", prod)
                if head == "=":
                    return ("value", int(values[0] == values[1]))
                if head == "<":
                    return ("value", int(values[0] < values[1]))
                if head == ">":
                    return ("value", int(values[0] > values[1]))
                if head == "<=":
                    return ("value", int(values[0] <= values[1]))
                if head == ">=":
                    return ("value", int(values[0] >= values[1]))
            except (TypeError, IndexError):
                return ("unknown", head)
        if head == "mif":
            test_status, test_val = self.meval(node.children[1], env)
            if test_status != "value":
                return ("unknown", f"(mif {test_val} ...)")
            branch = node.children[2] if test_val else node.children[3]
            return self.meval(branch, env)
        return ("unknown", self._label(node))

    def _head_name(self, node: Syn):
        if node.is_pair() and node.children:
            head = node.children[0].atom()
            if isinstance(head, str):
                return head
        return None

    # ---- expansion ----------------------------------------------------------
    def expand_form(self, node: Syn, stack: list[str], parent_graph: str,
                    depth: int, env: dict | None = None) -> Syn:
        env = {} if env is None else env
        atom = node.atom()
        if atom is not None or not node.is_pair():
            return node
        if node.quoted:
            return node
        head_name = self._head_name(node)

        if head_name == "quote":
            return node
        if head_name == "defmacro":
            # Macro definitions are retained as-is (the template is stored
            # separately), but do expand nothing inside it.
            return node
        if head_name in ("lambda", "let", "define"):
            return self._expand_binder_form(node, stack, parent_graph, depth, env)
        if head_name == "begin":
            return self._expand_children(node, stack, parent_graph, depth, env)
        if head_name == "if":
            return self._expand_children(node, stack, parent_graph, depth, env)
        if head_name == "mif":
            return self._expand_mif(node, stack, parent_graph, depth, env)
        if head_name == "datum->syntax":
            return self._expand_datum_to_syntax(node, stack, parent_graph, depth, env)
        if head_name == "join-contexts":
            return self._expand_join_contexts(node, stack, parent_graph, depth, env)

        macro = self._lookup_macro(node.children[0]) if head_name else None
        if macro is not None:
            return self._expand_macro_call(node, macro, stack, parent_graph, depth,
                                           env)

        # Plain application: expand operator/operand positions.
        return self._expand_children(node, stack, parent_graph, depth, env)

    def _expand_binder_form(self, node, stack, parent_graph, depth, env):
        head = node.children[0].datum
        out_children = [node.children[0]]
        if head == "lambda":
            # children[1] is the parameter list (atoms, not expanded);
            # children[2:] are bodies and must be expanded.
            out_children.append(node.children[1])
            for child in node.children[2:]:
                out_children.append(self.expand_form(
                    child, stack, parent_graph, depth, env))
        elif head == "let":
            binding_node = node.children[1]
            new_pairs = []
            for pair in binding_node.children:
                rhs = self.expand_form(pair.children[1], stack,
                                       parent_graph, depth, env)
                new_pairs.append(self._recombine(
                    pair, [pair.children[0], rhs]))
            new_bindings = self._recombine(binding_node, new_pairs)
            out_children.append(new_bindings)
            for child in node.children[2:]:
                out_children.append(self.expand_form(
                    child, stack, parent_graph, depth, env))
        else:  # define
            out_children.append(node.children[1])
            for child in node.children[2:]:
                out_children.append(self.expand_form(
                    child, stack, parent_graph, depth, env))
        return self._recombine(node, out_children)

    def _expand_children(self, node, stack, parent_graph, depth, env=None):
        env = {} if env is None else env
        out_children = [node.children[0]]
        for child in node.children[1:]:
            out_children.append(
                self.expand_form(child, stack, parent_graph, depth, env))
        return self._recombine(node, out_children)

    def _recombine(self, node: Syn, children: list[Syn]) -> Syn:
        result = Syn([c.datum for c in children], node.scopes,
                     list(node.origin), token=node.token, quoted=node.quoted)
        result.children = children
        return result

    def _expand_mif(self, node, stack, parent_graph, depth, env=None):
        env = {} if env is None else env
        # Expand the test position first, then macro-time evaluate.
        test = self.expand_form(node.children[1], stack, parent_graph, depth)
        status, value = self.meval(test, env)
        if status == "value":
            chosen = node.children[2] if value else node.children[3]
            expanded_branch = self.expand_form(chosen, stack, parent_graph,
                                               depth, env)
            frame = OriginFrame("transform", "mif-branch-taken",
                                node.first_span(),
                                detail=f"test={value!r}")
            expanded_branch.push_origin(frame)
            self.record_event("mif-decision", {
                "sid": node.sid, "test_status": "known", "test_value": value,
                "taken": self._label(chosen),
            })
            return expanded_branch
        marker = Syn("#%unknown", node.scopes, list(node.origin))
        marker.origin.append(OriginFrame(
            "error", "mif-test-unknown", node.first_span(),
            detail=f"cannot reduce {value!r} at macro time"))
        self.diagnostics.append(Diagnostic(
            code="mif-unknown-test",
            message=f"macro-time test could not be reduced: {value}",
            span=(node.first_span().to_json()
                  if node.first_span() else None),
            origin_chain=[f.to_json() for f in node.origin],
            invocation_stack=list(stack), severity="unknown"))
        self.record_event("mif-decision", {
            "sid": node.sid, "test_status": "unknown", "test_value": str(value),
            "candidates": [self._label(node.children[2]),
                           self._label(node.children[3])],
        })
        candidates = [self.expand_form(c, stack, parent_graph, depth, env)
                      for c in node.children[2:4]]
        unknown_node = self._recombine(node, [node.children[0], test, marker]
                                       + candidates)
        return unknown_node

    def _lookup_macro(self, name_node: Syn) -> MacroDef | None:
        name = name_node.atom()
        if not isinstance(name, str) or name not in self.build.macros:
            return None
        macro = self.build.macros[name]
        # Scope-based lookup: the definition-site (module) scope must be on the
        # use-site identifier. A purely local occurrence that shadows the name
        # lacks it and therefore does not resolve to the macro.
        if MODULE_SCOPE in name_node.scopes:
            return macro
        return None

    def _expand_macro_call(self, node, macro, stack, parent_graph, depth,
                            env=None):
        env = {} if env is None else env
        # Budget: one macro invocation per step. Exhaustion keeps the partial
        # expansion graph and marks this node instead of crashing the run.
        if self.steps_used >= self.budget:
            self.budget_exhausted = True
            partial_id = self.add_node(self._label(node), "budget-exceeded",
                                       node, None, depth, partial=True)
            self.add_edge(parent_graph, partial_id, "unexpanded-call")
            self.diagnostics.append(Diagnostic(
                code="budget-exceeded",
                message=(f"expansion budget of {self.budget} macro "
                         "invocations exhausted; partial expansion retained"),
                span=(node.first_span().to_json()
                      if node.first_span() else None),
                origin_chain=[f.to_json() for f in node.origin],
                invocation_stack=list(stack) + [macro.name],
                severity="error"))
            self.record_event("budget-exceeded", {
                "macro": macro.name, "budget": self.budget,
                "unit": "macro_invocations",
                "sid": node.sid, "partial_node": partial_id,
            })
            marker = Syn("#%budget-exceeded", node.scopes, list(node.origin))
            marker.children = node.children
            return marker

        self.steps_used += 1
        self._macro_counters[macro.name] = \
            self._macro_counters.get(macro.name, 0) + 1
        ordinal = self._macro_counters[macro.name]
        invocation_id = stable_id(
            "invoke", self.seed, macro.sid, self.steps_used, ordinal)
        new_stack = stack + [invocation_id]

        call_id = self.add_node(f"({macro.name} ...)", "macro-call", node,
                                invocation_id, depth)
        self.add_edge(parent_graph, call_id, "invoke")

        intro_scope = self._fresh_scope(
            "macro-intro", f"macro-introduction for {macro.name}#{ordinal}",
            invocation_id)
        frame = OriginFrame(
            "macro", macro.name, node.first_span(),
            invocation_id=invocation_id, step=self.steps_used,
            macro_name=macro.name,
            detail=f"invocation #{ordinal} (depth {depth + 1})")
        self.invocations.append({
            "invocation_id": invocation_id, "macro": macro.name,
            "ordinal": ordinal, "depth": depth + 1,
            "step": self.steps_used, "parent_stack": list(stack),
            "intro_scope": intro_scope,
            "call_span": (node.first_span().to_json()
                          if node.first_span() else None),
            "macro_sid": macro.sid,
        })
        self.record_event("macro-invocation", {
            "invocation_id": invocation_id, "macro": macro.name,
            "ordinal": ordinal, "depth": depth + 1,
            "step": self.steps_used, "unit": "macro_invocations",
            "intro_scope": intro_scope, "stack": list(new_stack),
            "call_sid": node.sid,
        })

        # Arguments are expanded first (they originate at the use site, so
        # they keep use-site scopes and are NOT given the intro scope).
        arg_nodes = node.children[1:]
        arg_map = {}
        expanded_args = []
        child_env = dict(env)
        for arg_index, (param, arg) in enumerate(zip(macro.params, arg_nodes)):
            expanded = self.expand_form(arg, new_stack, call_id, depth + 1,
                                        env)
            expanded_args.append(expanded)
            arg_map[param] = expanded
            val_status, val = self.meval(arg_nodes[arg_index], env)
            if val_status == "value":
                child_env[param] = val
            self.record_event("arg-bind", {
                "invocation_id": invocation_id, "param": param,
                "arg_sid": arg.sid, "macro_time_value":
                val if val_status == "value" else None,
                "macro_time_status": val_status,
            })
        if len(macro.params) != len(arg_nodes):
            self.diagnostics.append(Diagnostic(
                code="macro-arity",
                message=(f"macro {macro.name} expects {len(macro.params)} "
                         f"arguments but got {len(arg_nodes)}"),
                span=(node.first_span().to_json()
                      if node.first_span() else None),
                origin_chain=[f.to_json() for f in node.origin],
                invocation_stack=list(new_stack)))

        instantiated = self.instantiate(
            macro.template, macro, arg_map, macro.def_scopes, intro_scope,
            invocation_id, frame, depth + 1)

        out_id = self.add_node(self._label(instantiated), "macro-output",
                               instantiated, invocation_id, depth)
        self.add_edge(call_id, out_id, "produces")

        # Re-expand the produced syntax (nested / recursive macros).
        result = self.expand_form(instantiated, new_stack, out_id, depth,
                                  child_env)
        return result

    # ---- template instantiation (template is a Syn syntax object) ----------
    def instantiate(self, template, macro, arg_map, def_scopes, intro_scope,
                    invocation_id, frame, depth):
        if template.is_pair():
            children = [self.instantiate(
                piece, macro, arg_map, def_scopes, intro_scope,
                invocation_id, frame, depth) for piece in template.children]
            node = Syn([c.datum for c in children], frozenset(intro_scope),
                       [])
            node.children = children
            node.quoted = template.quoted
            # The intro scope marks every macro-produced node, but it must not
            # be pushed into substituted arguments (they keep use-site scopes).
            self._add_intro_except_subst(node, intro_scope)
            node.origin.append(frame)
            return node
        return self._instantiate_atom(
            template, macro, arg_map, def_scopes, intro_scope,
            invocation_id, frame)

    def _add_intro_except_subst(self, node, intro_scope):
        for child in node.children:
            substituted = getattr(child, "_from_substitution", False)
            if not substituted:
                child.scopes = child.scopes | {intro_scope}
                self._add_intro_except_subst(child, intro_scope)

    def _instantiate_atom(self, template, macro, arg_map, def_scopes,
                          intro_scope, invocation_id, frame):
        text = template.datum
        if isinstance(text, str) and text in arg_map:
            # Substitution: the use-site occurrence keeps its own scopes, so a
            # call-site lambda captures it and a template binding cannot.
            substituted = arg_map[text]
            clone = substituted._clone()
            clone._from_substitution = True
            clone.origin.append(OriginFrame(
                "transform", f"substitute param {text}",
                substituted.first_span(), invocation_id=invocation_id,
                macro_name=macro.name))
            return clone
        # Template literal: definition-site scopes (already on the template)
        # plus this invocation's intro scope.
        node = Syn(text, frozenset(template.scopes) | {intro_scope},
                   list(template.origin), token=template.token,
                   quoted=template.quoted)
        node.origin.append(OriginFrame(
            "macro", f"template literal {text}",
            template.first_span(), invocation_id=invocation_id,
            macro_name=macro.name,
            detail="definition-site reference; macro-intro scope attached"))
        if text == "gen-temp":
            symbol = self._gensym("tmp", invocation_id)
            node.datum = symbol
            node.token = None
        return node

    # ---- datum->syntax / join-contexts (explicit context selection) ---------
    def _expand_datum_to_syntax(self, node, stack, parent_graph, depth, env=None):
        env = {} if env is None else env
        if len(node.children) != 3:
            self._malformed(node, stack, "datum->syntax expects 2 arguments")
            return node
        context = node.children[1]
        datum = node.children[2]
        context_scopes = frozenset(context.scopes)
        context_id = context.sid
        self.record_event("context-select", {
            "operator": "datum->syntax", "context_sid": context_id,
            "context_spelling": context.atom(),
            "context_scopes": sorted(context_scopes),
            "result_sid": node.sid,
            "basis": "scopes copied from explicit context identifier",
        })

        def reshape(dnode: Syn) -> Syn:
            if dnode.is_pair():
                kids = [reshape(c) for c in dnode.children]
                out = Syn([k.datum for k in kids], context_scopes,
                          list(node.origin))
                out.children = kids
                out.origin.append(OriginFrame(
                    "transform", "datum->syntax reshape",
                    node.first_span(), detail="scopes from explicit context"))
                return out
            out = Syn(dnode.datum, context_scopes, [], token=dnode.token)
            out.origin.append(OriginFrame(
                "transform", "datum->syntax atom", node.first_span(),
                detail=(f"context={context_id}; datum lifted with explicit "
                        "call-site scopes (explicit capture)")))
            return out

        # datum may be a quoted literal or an identifier/symbol template var.
        if datum.is_pair() and self._head_name(datum) == "quote":
            quoted_form = datum.children[1]
            shaped = reshape(quoted_form)
        else:
            shaped = reshape(datum)
        shaped.push_origin(OriginFrame(
            "transform", "datum->syntax result", node.first_span(),
            detail="visible context choice"))
        return self.expand_form(shaped, stack, parent_graph, depth, env)

    def _expand_join_contexts(self, node, stack, parent_graph, depth, env=None):
        env = {} if env is None else env
        if len(node.children) < 3:
            self._malformed(node, stack,
                            "join-contexts expects at least 2 identifiers")
            return node
        ids = node.children[1:]
        merged = frozenset()
        member_scopes = []
        for ident in ids:
            merged |= frozenset(ident.scopes)
            member_scopes.append({
                "sid": ident.sid, "spelling": ident.atom(),
                "scopes": sorted(ident.scopes),
            })
        first = ids[0]
        out = Syn(first.datum, merged, [], token=first.token)
        for ident in ids:
            out.origin.extend(ident.origin)
        out.origin.append(OriginFrame(
            "transform", "join-contexts", node.first_span(),
            detail="union of identifier contexts; ambiguous if several "
                   "binders match"))
        self.record_event("context-select", {
            "operator": "join-contexts", "members": member_scopes,
            "merged_scopes": sorted(merged), "result_sid": out.sid,
            "basis": "union of all member identifier scope sets",
        })
        return self.expand_form(out, stack, parent_graph, depth, env)

    def _malformed(self, node, stack, message):
        self.diagnostics.append(Diagnostic(
            code="malformed-form", message=message,
            span=(node.first_span().to_json()
                  if node.first_span() else None),
            origin_chain=[f.to_json() for f in node.origin],
            invocation_stack=list(stack), severity="error"))
        self.record_event("error-remap", {
            "code": "malformed-form", "message": message,
            "sid": node.sid,
            "span": (node.first_span().to_json()
                     if node.first_span() else None),
            "origin_chain": [f.to_json() for f in node.origin],
            "invocation_stack": list(stack),
        })
