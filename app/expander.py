"""Scope-set macro expander with provenance tracking.

Hygiene model (Flatt-style "one moding" scopes, see cases README):

* binding forms add a *fresh* lexical scope to their whole subtree and install
  a binding ``(name -> scope)``;
* a use-site identifier resolves to the binding whose scope set is the largest
  strict subset of the use's scope set.  Capture is therefore a relation
  between scope sets -- the symbol spelling never participates;
* a macro copies its template syntax (definition-site scopes intact) and
  swaps in call-site arguments; identifiers that the template mentions freely
  resolve at the definition site, while argument identifiers keep call-site
  scopes;
* ``datum->syntax`` deliberately attaches a chosen context to a scope-less
  datum.  The context choice is recorded as a visible decision candidate;
* expansion has a macro-call budget.  Exhaustion fails the run but keeps the
  partial expansion graph and every step emitted so far.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import Binding, Node, Origin, Span
from .eval import Closure, Context, Evaluator, ExpandLimit, MetaError, Pair, Primitive, Value

BINDING_FORMS = {"lambda", "let"}
CORE_FORMS = {"define", "defmacro", "quote", "quasiquote", "if", "begin", "set!"}


@dataclass
class Candidate:
    kind: str                  # binding|macro|context|explicit-capture
    target_sid: str
    name: str
    options: list[dict]
    chosen: Optional[str]      # sid of chosen option
    reason: str
    status: str = "open"       # open|resolved|rejected|stale

    def to_json(self) -> dict:
        return {"kind": self.kind, "target_sid": self.target_sid, "name": self.name,
                "options": self.options, "chosen": self.chosen, "reason": self.reason,
                "status": self.status}


@dataclass
class Edge:
    src: str
    dst: str
    kind: str                  # expand|copy|arg|splice|gensym|d2s|bind|use|def|remap

    def to_json(self) -> dict:
        return {"src": self.src, "dst": self.dst, "kind": self.kind}


@dataclass
class ExpansionResult:
    forms: list[Node]
    bindings: list[Binding]
    edges: list[Edge]
    steps: list[dict]
    candidates: list[Candidate]
    diagnostics: list[dict]
    tokens: list = field(default_factory=list)
    success: bool = True
    budget_used: int = 0
    def_site: Optional[Span] = None


@dataclass
class _Frame:
    macro: str
    call_sid: str
    call_span: Optional[Span]
    stack: tuple[str, ...]


class Expander:
    def __init__(self, tokens=None, params: Optional[dict] = None,
                 decisions: Optional[dict] = None, run_id: str = "r1",
                 algorithm_version: str = "scope-set-1"):
        self.tokens = tokens or []
        self.params = params or {"budget": 64}
        self.decisions = decisions or {}
        self.run_id = run_id
        self.algorithm_version = algorithm_version
        self.bindings: list[Binding] = []
        self.edges: list[Edge] = []
        self.candidates: list[Candidate] = []
        self.steps: list[dict] = []
        self.diagnostics: list[dict] = []
        self.macros: dict[str, Any] = {}
        self.frames: list[_Frame] = []
        self.scope_next = 0
        self.node_next = 0
        self.gen_next = 0
        self.ctx_next = 0
        self.budget_used = 0
        self.partial: list[Node] = []
        self.sid_span: dict[str, Optional[Span]] = {}
        self.meta_base = self._primitives()

    # ---- ids / scopes ------------------------------------------------------
    def _scope(self) -> int:
        sid = self.scope_next
        self.scope_next += 1
        return sid

    def _nsid(self) -> str:
        sid = f"exp::{self.run_id}::n{self.node_next}"
        self.node_next += 1
        return sid

    def _step(self, kind: str, **detail: Any) -> dict:
        step = {"seq": len(self.steps), "type": kind,
                "algo": self.algorithm_version, **detail}
        self.steps.append(step)
        return step

    def _remember(self, node: Node) -> Node:
        self.sid_span.setdefault(node.sid, node.span)
        return node

    # ---- binding resolution -----------------------------------------------
    def resolve(self, node: Node) -> tuple[Optional[Binding], Optional[Candidate]]:
        if node.lit != "symbol":
            return None, None
        name = node.atom_text()
        applicable = [b for b in self.bindings
                      if b.name == name and b.scopes <= node.scopes]
        applicable.sort(key=lambda b: len(b.scopes), reverse=True)
        cand = None
        chosen: Optional[Binding] = None
        if not applicable:
            self.diagnostics.append({
                "code": "unbound", "name": name, "sid": node.sid,
                "span": node.span.to_json() if node.span else None})
            self._step("resolve", sid=node.sid, name=name, status="unbound")
            return None, None
        top = applicable[0]
        tied = [b for b in applicable if len(b.scopes) == len(top.scopes)]
        if len(tied) > 1:
            options = [{"site": b.site, "kind": b.kind,
                        "scopes": sorted(b.scopes),
                        "span": b.span.to_json() if b.span else None} for b in tied]
            cand = Candidate("binding", node.sid, name, options, None,
                             "two bindings have equal specificity; scope relation "
                             "does not determine a unique target")
            decision = self.decisions.get(node.sid)
            if decision:
                chosen = next((b for b in tied if b.site == decision), top)
                cand.chosen = chosen.site
                cand.status = "resolved"
            else:
                cand.status = "open"
                self.diagnostics.append({
                    "code": "ambiguous-binding", "name": name, "sid": node.sid,
                    "options": [b.site for b in tied],
                    "span": node.span.to_json() if node.span else None})
        else:
            chosen = top
        status = "macro" if chosen and chosen.kind == "defmacro" else "resolved"
        self._step("resolve", sid=node.sid, name=name,
                   target=chosen.site if chosen else None, status=status if chosen
                   else (cand.status if cand else "unbound"),
                   use_scopes=sorted(node.scopes),
                   binding_scopes=sorted(chosen.scopes) if chosen else None)
        if chosen:
            self.edges.append(Edge(node.sid, chosen.site, "use"))
        if cand:
            self.candidates.append(cand)
        return chosen, cand

    # ---- expansion ---------------------------------------------------------
    def expand_program(self, forms: list[Node]) -> ExpansionResult:
        out: list[Node] = []
        try:
            for form in forms:
                expanded = self.expand(form, frozenset())
                out.append(expanded)
                self.partial.append(expanded)
        except ExpandLimit as exc:
            self._step("budget-exhausted", budget=self.params.get("budget"),
                       used=self.budget_used, message=exc.message,
                       partial=[f.sid for f in out])
            self.diagnostics.append({
                "code": "budget-exhausted", "message": exc.message,
                "budget": self.params.get("budget"), "used": self.budget_used,
                "span": self._remap(exc.node).to_json() if exc.node and self._remap(exc.node) else None})
            return self._result(out, success=False)
        except MetaError as exc:
            span = self._error_span(exc)
            self._step("expand-error", message=exc.message, phase=exc.phase,
                       sid=exc.node.sid if exc.node else None,
                       remapped_span=span.to_json() if span else None,
                       call_stack=[f.macro for f in self.frames])
            self.diagnostics.append({
                "code": f"meta-error:{exc.phase}", "message": exc.message,
                "sid": exc.node.sid if exc.node else None,
                "span": (exc.node.span.to_json() if exc.node and exc.node.span else None),
                "remapped_span": span.to_json() if span else None,
                "call_stack": [f.macro for f in self.frames]})
            return self._result(out, success=False)
        return self._result(out, success=True)

    def _result(self, forms: list[Node], success: bool) -> ExpansionResult:
        return ExpansionResult(
            forms=forms, bindings=self.bindings, edges=self.edges,
            steps=self.steps, candidates=self.candidates,
            diagnostics=self.diagnostics, tokens=self.tokens, success=success,
            budget_used=self.budget_used)

    def expand(self, node: Node, ambient: frozenset[int]) -> Node:
        node = self._remember(node)
        if node.kind == "atom":
            if node.lit == "symbol":
                self.resolve(node)
            return node
        if not node.children:
            return node
        head = node.children[0]
        if head.kind == "atom" and head.atom_text() == "quote":
            return node
        if head.kind == "atom" and head.atom_text() == "defmacro":
            return self.register_macro(node)
        if head.kind == "atom" and head.atom_text() == "define":
            return self.expand_define(node, ambient)
        binding, _ = self.resolve(head) if head.kind == "atom" else (None, None)
        if binding is not None and binding.kind == "defmacro" \
                and head.atom_text() in self.macros:
            return self.invoke_macro(node, head.atom_text(), ambient)
        if head.kind == "atom" and head.atom_text() == "lambda":
            return self.expand_lambda(node, ambient)
        if head.kind == "atom" and head.atom_text() == "let":
            return self.expand_let(node, ambient)
        return self.expand_sequence(node, ambient)

    def expand_sequence(self, node: Node, ambient: frozenset[int]) -> Node:
        new_children = tuple(self.expand(child, ambient) for child in node.children)
        return self._remember(Node(node.kind, node.sid, node.scopes, node.span,
                                   new_children, node.value, node.lit, node.origin,
                                   node.fresh, node.datum, node.dot))

    # ---- binding forms -----------------------------------------------------
    def expand_define(self, node: Node, ambient: frozenset[int]) -> Node:
        target = node.children[1]
        scope = self._scope()
        if target.kind == "atom":
            self._install(target, "define", ambient | {scope})
            new_scopes = set(node.scopes) | {scope}
            self.edges.append(Edge(node.sid, target.sid, "def"))
            body = tuple(self.expand(c, ambient | {scope}) for c in node.children[2:])
            return self._finish(node, (node.children[0], target) + body, new_scopes)
        # (define (f x) ...) sugar
        sig = [target.children[0]] + list(target.children[1:])
        head_id = target.children[0]
        self._install(head_id, "define", ambient | {scope})
        self.edges.append(Edge(node.sid, head_id.sid, "def"))
        body = tuple(self.expand(c, ambient | {scope}) for c in node.children[2:])
        params = tuple(self.expand(a, ambient | {scope}) for a in target.children[1:])
        siglist = Node("list", self._nsid(), head_id.scopes | {scope}, target.span,
                       (head_id,) + params, None, "symbol",
                       Origin("macro", basis=target.sid))
        new_children = (node.children[0], siglist) + body
        return self._finish(node, new_children, set(node.scopes) | {scope})

    def expand_lambda(self, node: Node, ambient: frozenset[int]) -> Node:
        scope = self._scope()
        params = []
        for param in node.children[1].children:
            self._install(param, "lambda", ambient | {scope})
            self.edges.append(Edge(node.sid, param.sid, "bind"))
            params.append(param.with_scopes(add=(scope,)))
        param_list = Node("list", self._nsid(), node.children[1].scopes | {scope},
                          node.children[1].span, tuple(params), None, "symbol",
                          Origin("macro", basis=node.children[1].sid))
        body = tuple(self.expand(c, ambient | {scope}) for c in node.children[2:])
        return self._finish(node, (node.children[0], param_list) + body,
                            set(node.scopes) | {scope})

    def expand_let(self, node: Node, ambient: frozenset[int]) -> Node:
        scope = self._scope()
        bindings: list[Node] = []
        for bind in node.children[1].children:
            var = bind.children[0]
            val = self.expand(bind.children[1], ambient)
            self._install(var, "let", ambient | {scope})
            self.edges.append(Edge(node.sid, var.sid, "bind"))
            bindings.append(Node("list", self._nsid(), bind.scopes | {scope},
                                 bind.span,
                                 (var.with_scopes(add=(scope,)), val), None,
                                 "symbol", Origin("macro", basis=bind.sid)))
        bind_list = Node("list", self._nsid(), node.children[1].scopes | {scope},
                         node.children[1].span, tuple(bindings), None, "symbol",
                         Origin("macro", basis=node.children[1].sid))
        body = tuple(self.expand(c, ambient | {scope}) for c in node.children[2:])
        return self._finish(node, (node.children[0], bind_list) + body,
                            set(node.scopes) | {scope})

    def _install(self, atom: Node, kind: str, scopes: frozenset[int]) -> None:
        binding = Binding(atom.atom_text(), frozenset(scopes), atom.sid, kind,
                          atom.span)
        self.bindings.append(binding)
        self.edges.append(Edge(atom.sid, atom.sid, "bind"))
        self._step("bind", site=atom.sid, name=atom.atom_text(), kind=kind,
                   scopes=sorted(scopes))

    def _finish(self, node: Node, children: tuple[Node, ...], scopes: set[int]) -> Node:
        return self._remember(Node(node.kind, node.sid, frozenset(scopes), node.span,
                                   children, node.value, node.lit, node.origin,
                                   node.fresh, node.datum, node.dot))

    # ---- macros ------------------------------------------------------------
    def register_macro(self, node: Node) -> Node:
        # (defmacro (name p ...) body)
        sig = node.children[1]
        name_atom = sig.children[0]
        params = tuple(a.atom_text() for a in sig.children[1:])
        body = node.children[2]
        scope = self._scope()
        def_scopes = set(name_atom.scopes) | {scope}
        self._install(name_atom, "defmacro", frozenset(def_scopes))
        closure = Closure(params, body, {}, name=name_atom.atom_text())
        self.macros[name_atom.atom_text()] = closure
        self._step("macro-define", name=name_atom.atom_text(),
                   site=name_atom.sid, params=list(params),
                   def_scope=scope,
                   span=name_atom.span.to_json() if name_atom.span else None)
        self.edges.append(Edge(node.sid, name_atom.sid, "def"))
        return self._finish(node, node.children, set(node.scopes) | {scope})

    def invoke_macro(self, node: Node, name: str, ambient: frozenset[int]) -> Node:
        budget = int(self.params.get("budget", 64))
        if self.budget_used >= budget:
            raise ExpandLimit(
                f"macro budget {budget} exhausted at call to {name!r}", budget,
                self.partial, node)
        closure = self.macros[name]
        self.budget_used += 1
        stack = tuple(f.macro for f in self.frames)
        frame = _Frame(name, node.sid, node.span, stack)
        self.frames.append(frame)
        args = list(node.children[1:])
        env: dict[str, Any] = {}
        for pname, arg in zip(closure.params, args):
            env[pname] = arg
        # context primitives obtain the current call site
        self.ctx_next += 1
        ctx = Context(self.ctx_next, node)
        env["__context__"] = ctx
        self._step("macro-call", name=name, site=node.sid,
                   args=[a.sid for a in args],
                   def_site=next((b.site for b in self.bindings
                                  if b.name == name and b.kind == "defmacro"), None),
                   call_stack=list(stack), budget_after=self.budget_used,
                   span=node.span.to_json() if node.span else None)
        try:
            raw = Evaluator(self, dict(closure.env)).run(closure.body, env)
        finally:
            self.frames.pop()
        produced = self._to_node(raw, node, name, closure, depth=0)
        self.edges.append(Edge(node.sid, produced.sid, "expand"))
        self._step("macro-result", call=node.sid, name=name, result=produced.sid)
        return self.expand(produced, ambient)

    def _to_node(self, value: Any, call: Node, name: str, closure: Closure,
                 depth: int) -> Node:
        if isinstance(value, Node):
            return value
        if isinstance(value, Value):
            return self.datum_to_syntax(call, value.v, explicit=False)
        raise MetaError(f"macro {name!r} returned non-syntax value {value!r}", call)

    # ---- template construction (definition-site scopes preserved) ---------
    def copy_template_atom(self, form: Node) -> Node:
        if self.frames:
            frame = self.frames[-1]
            origin = Origin("macro", macro=frame.macro,
                            call_site=frame.call_sid, call_stack=frame.stack,
                            template_sid=form.sid)
        else:
            origin = form.origin
        new = Node("atom", self._nsid(), form.scopes, form.span, (), form.value,
                   form.lit, origin)
        self.edges.append(Edge(form.sid, new.sid, "copy"))
        return self._remember(new)

    def make_template_list(self, form: Node, parts: list[Any], splices: list[dict],
                           template: Node) -> Node:
        frame = self.frames[-1] if self.frames else None
        origin = Origin("macro", macro=frame.macro if frame else None,
                        call_site=frame.call_sid if frame else None,
                        call_stack=frame.stack if frame else (),
                        template_sid=form.sid)
        children: list[Node] = []
        for part in parts:
            child = part if isinstance(part, Node) else self.datum_to_syntax(
                template, part.v if isinstance(part, Value) else part,
                explicit=False)
            children.append(child)
        for splice in reversed(splices):
            at = splice["at"]
            children[at:at] = splice["items"]
        new = Node("list", self._nsid(), form.scopes, form.span, tuple(children),
                   None, "symbol", origin, dot=form.dot)
        for child in children:
            if isinstance(child, Node):
                self.edges.append(Edge(form.sid, child.sid, "arg"
                                       if child.origin.kind == "parse" else "copy"))
        self.edges.append(Edge(form.sid, new.sid, "splice" if splices else "copy"))
        return self._remember(new)

    def make_gensym(self, site: Node, base: str) -> Node:
        serial = self.gen_next
        self.gen_next += 1
        name = f"{base}__{self.run_id}__g{serial}"
        frame = self.frames[-1] if self.frames else None
        origin = Origin("gensym", macro=frame.macro if frame else None,
                        call_site=frame.call_sid if frame else None,
                        call_stack=frame.stack if frame else (),
                        gen_serial=serial, basis=site.sid)
        node = Node("atom", f"exp::{self.run_id}::gen{serial}", frozenset(), None,
                    (), name, "symbol", origin, fresh=True)
        self._step("gensym", sid=node.sid, name=name, base=base, serial=serial,
                   call_stack=list(origin.call_stack))
        self.edges.append(Edge(site.sid, node.sid, "gensym"))
        return self._remember(node)

    def datum_to_syntax(self, site: Node, datum: Any, explicit: bool = True,
                        ctx: Any = None) -> Node:
        if isinstance(datum, Node):
            return datum
        basis = ctx.node if isinstance(ctx, Context) else site
        frame = self.frames[-1] if self.frames else None
        origin = Origin("d2s", macro=frame.macro if frame else None,
                        call_site=frame.call_sid if frame else None,
                        call_stack=frame.stack if frame else (), basis=basis.sid)
        node = self._build_datum(datum, basis, origin)
        self.edges.append(Edge(site.sid, node.sid, "d2s"))
        options = [{
            "site": basis.sid,
            "label": "call-site context" if isinstance(basis.sid, str) else "?",
            "scopes": sorted(basis.scopes) if basis.kind == "atom" else [],
            "span": basis.span.to_json() if basis.span else None}]
        cand = Candidate("context", node.sid,
                         node.atom_text() if node.kind == "atom" else "(datum-list)",
                         options, basis.sid,
                         "datum has no lexical context; datum->syntax selected the "
                         "call-site syntax object", )
        if explicit:
            cand.kind = "explicit-capture"
            cand.status = "resolved"
            self._step("datum-to-syntax", sid=node.sid, basis=basis.sid,
                       scopes=sorted(node.scopes) if node.kind == "atom" else [],
                       explicit=True)
        self.candidates.append(cand)
        return node

    def _build_datum(self, datum: Any, basis: Node, origin: Origin) -> Node:
        if isinstance(datum, list):
            children = tuple(self._build_datum(item, basis, origin) for item in datum)
            return self._remember(Node("list", self._nsid(), basis.scopes, basis.span,
                                       children, None, "symbol", origin, datum=True))
        if isinstance(datum, Pair):
            children = []
            cur: Any = datum
            while isinstance(cur, Pair):
                children.append(self._build_datum(cur.car, basis, origin))
                cur = cur.cdr
            if cur is not None:
                children.append(self._build_datum(cur, basis, origin))
            return self._remember(Node("list", self._nsid(), basis.scopes, basis.span,
                                       tuple(children), None, "symbol", origin,
                                       datum=True, dot=cur is not None))
        if isinstance(datum, dict) and "__unconvertible__" in datum:
            raw = datum["__unconvertible__"]
            self.diagnostics.append({
                "code": "unconvertible", "value": raw, "basis": basis.sid,
                "message": "kept in place; normalization failed"})
            node = Node("atom", self._nsid(), basis.scopes, basis.span, (), raw,
                        "number", origin, datum=True)
            return self._remember(node)
        if isinstance(datum, bool):
            lit, text = "boolean", "#t" if datum else "#f"
        elif isinstance(datum, (int, float)):
            lit, text = "number", _num_text(datum)
        else:
            lit, text = "symbol", str(datum)
        scopes = basis.scopes if basis.kind == "atom" else frozenset()
        node = Node("atom", self._nsid(), scopes, basis.span, (), text, lit, origin,
                    datum=True)
        return self._remember(node)

    # ---- error source remapping -------------------------------------------
    def _remap(self, node: Optional[Node]) -> Optional[Span]:
        cur = node
        seen: set[str] = set()
        while cur is not None and cur.sid not in seen:
            seen.add(cur.sid)
            origin = cur.origin
            if cur.span is not None and origin.kind in ("parse", "sugar"):
                return cur.span
            if origin.call_site:
                cur = self._sid_node(origin.call_site)
                continue
            if origin.template_sid:
                cur = self._sid_node(origin.template_sid)
                continue
            if origin.basis:
                cur = self._sid_node(origin.basis)
                continue
            break
        return cur.span if cur is not None else None

    def _sid_node(self, sid: str) -> Optional[Node]:
        for node in self.partial:
            for found in node.walk():
                if found.sid == sid:
                    return found
        return None

    def _error_span(self, exc: MetaError) -> Optional[Span]:
        remapped = self._remap(exc.node)
        if exc.node is not None:
            self._step("source-remap", sid=exc.node.sid,
                       original=exc.node.span.to_json() if exc.node.span else None,
                       remapped=remapped.to_json() if remapped else None,
                       call_stack=[f.macro for f in self.frames])
        return remapped

    # ---- meta primitives ---------------------------------------------------
    def _primitives(self) -> dict[str, Primitive]:
        def call_context(site, ev):  # noqa: ANN001
            return ev.env["__context__"]

        def d2s(site, datum, ctx=None):
            value = datum.v if isinstance(datum, Value) else datum
            return self.datum_to_syntax(site, value, explicit=True, ctx=ctx)

        def gensym(site, base=Value("g")):
            name = base.v if isinstance(base, Value) else "g"
            return self.make_gensym(site, str(name))

        def list_p(site, *items):
            return Value(list(items))

        def cons(site, a, b):
            bv = b.v if isinstance(b, Value) else b
            av = a
            return Value(Pair(av, bv))

        def car(site, pair):
            v = pair.v if isinstance(pair, Value) else pair
            if isinstance(v, Pair):
                return v.car
            return v[0] if isinstance(v, list) else MetaError("car of non-pair", site)

        def cdr(site, pair):
            v = pair.v if isinstance(pair, Value) else pair
            if isinstance(v, Pair):
                return Value(v.cdr)
            return Value(v[1:])

        def eq(site, a, b):
            av = a.v if isinstance(a, Value) else a
            bv = b.v if isinstance(b, Value) else b
            return av == bv

        def arithmetic(symbol):  # noqa: ANN001
            import operator
            op = {"+": operator.add, "-": operator.sub, "*": operator.mul}[symbol]

            def fn(site, *nums):
                vals = [n.v if isinstance(n, Value) else n for n in nums]
                return Value(op(*vals))
            return fn

        return {
            "call-context": Primitive("call-context", call_context, wants_ctx=False),
            "datum->syntax": Primitive("datum->syntax", d2s),
            "gensym": Primitive("gensym", gensym),
            "fresh-id": Primitive("fresh-id", gensym),
            "list": Primitive("list", list_p),
            "cons": Primitive("cons", cons),
            "car": Primitive("car", car),
            "cdr": Primitive("cdr", cdr),
            "eq?": Primitive("eq?", eq),
            "equal?": Primitive("equal?", eq),
            "+": Primitive("+", arithmetic("+")),
            "-": Primitive("-", arithmetic("-")),
            "*": Primitive("*", arithmetic("*")),
        }


def _num_text(num) -> str:  # noqa: ANN001
    if isinstance(num, float) and num.is_integer():
        return str(int(num))
    return str(num)


def fingerprint_params(params: dict, algo: str = "scope-set-1") -> str:
    import json
    body = json.dumps({"algo": algo, "params": params}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
