"""Scope-set macro expander.

Hygiene is judged purely by scope-set membership (Flatt-style sets of scopes),
never by string renaming. Every expansion step, binding edge, scope set,
generated symbol, capture judgement and datum->syntax context choice is kept as
first-class, machine-readable provenance. Recursive macro expansion has a step
budget; on failure the partial expansion graph is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .syntax import Span, Syntax
from . import ALGO_VERSION

CORE_FORMS = {"let", "lambda", "if", "quote", "def", "defmacro"}
PRIMITIVE_CALLS = {"+", "-", "*", "=", "list"}

RESOLUTION_BASIS = "scope-set-membership:not-string-renaming"
BUDGET_UNIT = "macro-expansion-steps"


class BudgetExceeded(Exception):
    pass


@dataclass
class Binding:
    bid: str
    name: str
    scopes: frozenset
    kind: str  # def | defmacro | let | lambda | gensym
    fragment_id: Optional[str]
    span: Optional[Span]
    macro: Optional[Any] = None


@dataclass
class Macro:
    name: str
    params: List[str]
    template: Syntax
    def_scope: str
    fragment_id: str
    def_span: Optional[Span]


class Env:
    """Lexical frame chain. Scope sets decide which visible bindings apply."""

    def __init__(self, parent: Optional["Env"] = None):
        self.parent = parent
        self.table: Dict[str, List[Binding]] = {}

    def bind(self, binding: Binding) -> None:
        self.table.setdefault(binding.name, []).append(binding)

    def visible(self, name: str) -> List[Binding]:
        out: List[Binding] = []
        env: Optional[Env] = self
        while env is not None:
            out.extend(env.table.get(name, []))
            env = env.parent
        return out


class Expander:
    def __init__(self, run_seed: str, budget: int = 16,
                 context_choices: Optional[Dict[str, str]] = None):
        self.run_seed = run_seed
        self.budget = budget
        self.context_choices = context_choices or {}
        self.counter = 0
        self.module_scope = self._scope("module")
        self.steps: List[dict] = []
        self.bindings: Dict[str, Binding] = {}
        self.binding_edges: List[dict] = []
        self.diagnostics: List[dict] = []
        self.candidates: List[dict] = []
        self.syntax_table: Dict[str, Syntax] = {}
        self.source_map: List[dict] = []
        self.generated_symbols: List[dict] = []
        self.capture_checks: List[dict] = []
        self.context_choice_events: List[dict] = []
        self.used_fragments: set = set()
        self.macro_calls: List[dict] = []
        self.used = 0
        self.budget_exceeded = False
        self._gensym_n = 0
        self._global = Env()
        self._pending_nodes: Dict[str, dict] = {}

    # ------------------------------------------------------------------ ids
    def _next(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}-{self.run_seed}-{self.counter}"

    def _scope(self, kind: str) -> str:
        return self._next(f"scp-{kind}")

    def mk(self, datum, span, scopes, origin=None, step=None, explicit=False) -> Syntax:
        syn = Syntax(self._next("syn"), datum, span, frozenset(scopes), origin, step)
        self.syntax_table[syn.sid] = syn
        if origin is not None:
            self.source_map.append({
                "sid": syn.sid, "origin_sid": origin, "step": step,
                "span": syn.span.to_obj() if syn.span else None,
                "datum_text": syn.datum_text(), "explicit": explicit,
            })
        return syn

    def add_scope(self, syn: Syntax, scopes, step=None, explicit=False) -> Syntax:
        """Return a copy carrying additional scopes (recurses into lists)."""
        scopes = frozenset(scopes) | syn.scopes
        if syn.is_list:
            children = [self.add_scope(c, scopes, step, explicit) for c in syn.datum]
            return self.mk(children, syn.span, scopes, syn.sid, step, explicit)
        return self.mk(syn.datum, syn.span, scopes, syn.sid, step, explicit)

    def copy_like(self, syn: Syntax, datum, scopes, step=None, explicit=False) -> Syntax:
        return self.mk(datum, syn.span, frozenset(scopes), syn.sid, step, explicit)

    def record_step(self, step: dict) -> None:
        self.steps.append(step)

    # -------------------------------------------------------------- bindings
    def resolve(self, env: Env, name: str, scopes: frozenset
                ) -> Tuple[Optional[Binding], List[Binding]]:
        eligible = [b for b in env.visible(name) if b.scopes <= frozenset(scopes)]
        if not eligible:
            return None, []
        maximal: List[Binding] = []
        for b in eligible:
            if not any(o is not b and b.scopes < o.scopes for o in eligible):
                maximal.append(b)
        unique, seen = [], set()
        for b in maximal:
            if b.bid not in seen:
                seen.add(b.bid)
                unique.append(b)
        if len(unique) == 1:
            return unique[0], []
        return None, unique

    # --------------------------------------------------------------- program
    def load_fragment(self, fragment_id: str, form: Syntax, kind: str) -> None:
        """Process a def/defmacro fragment into the global environment."""
        step_id = self._next("step")
        form = self.add_scope(form, {self.module_scope}, step_id)
        self.record_step({
            "step_id": step_id, "kind": kind, "parent_step": None,
            "fragment_id": fragment_id, "summary": kind,
            "inputs": [form.sid], "outputs": [], "macro_stack": [],
        })
        self.used_fragments.add(fragment_id)
        if kind == "defmacro":
            name_syn, params_syn, template = form.datum[1], form.datum[2], form.datum[3]
            def_scope = self._scope("defsite")
            params = [p.datum for p in params_syn.datum]
            template_s = self.add_scope(template, {self.module_scope, def_scope}, step_id)
            macro = Macro(name_syn.datum, params, template_s, def_scope,
                          fragment_id, name_syn.span)
            bid = self._next("bnd")
            binding = Binding(bid, name_syn.datum, name_syn.scopes, "defmacro",
                              fragment_id, name_syn.span, macro=macro)
            self.bindings[bid] = binding
            self._global.bind(binding)
            self.binding_edges.append({
                "from_sid": name_syn.sid, "binding_id": bid,
                "status": "defines-macro", "name": name_syn.datum,
                "scopes": sorted(name_syn.scopes),
            })
            self.steps[-1]["outputs"].append(template_s.sid)
        elif kind == "def":
            name_syn, init_syn = form.datum[1], form.datum[2]
            bid = self._next("bnd")
            binding = Binding(bid, name_syn.datum, name_syn.scopes, "def",
                              fragment_id, name_syn.span)
            self.bindings[bid] = binding
            self._global.bind(binding)
            self.binding_edges.append({
                "from_sid": name_syn.sid, "binding_id": bid,
                "status": "defines-value", "name": name_syn.datum,
                "scopes": sorted(name_syn.scopes),
            })
            init_out = self.expand(init_syn, self._global, [])
            self.steps[-1]["outputs"].append(init_out.sid)

    def expand_fragment(self, fragment_id: str, form: Syntax) -> Syntax:
        form = self.add_scope(form, {self.module_scope})
        self.used_fragments.add(fragment_id)
        try:
            return self.expand(form, self._global, [])
        except BudgetExceeded:
            return form

    # -------------------------------------------------------------- expansion
    def expand(self, syn: Syntax, env: Env, stack: List[dict]) -> Syntax:
        if syn.is_ident:
            return self._expand_identifier(syn, env)
        if not syn.is_list or not syn.datum:
            return syn
        head = syn.datum[0]
        if head.is_ident:
            name = head.datum
            binding, _ = self.resolve(env, name, head.scopes)
            if binding is not None and binding.kind == "defmacro":
                return self._expand_macro_use(syn, binding, env, stack)
            if name == "let":
                return self._expand_let(syn, env, stack)
            if name == "lambda":
                return self._expand_lambda(syn, env, stack)
            if name in ("def", "defmacro"):
                self._diag(f"{name} is only allowed at top level", syn)
                return syn
            if name == "quote":
                return syn
        children_out = [self.expand(c, env, stack) for c in syn.datum]
        return self.copy_like(syn, children_out, syn.scopes)

    def _expand_identifier(self, syn: Syntax, env: Env) -> Syntax:
        pending = self._pending_nodes.get(syn.sid)
        if pending is not None:
            self._resolve_pending(syn, pending, env)
            return syn
        binding, ties = self.resolve(env, syn.datum, syn.scopes)
        if binding is None and ties:
            self._ambiguous(syn, ties)
            return syn
        self.binding_edges.append({
            "from_sid": syn.sid,
            "binding_id": binding.bid if binding else None,
            "status": "resolved" if binding else "unbound",
            "name": syn.datum, "scopes": sorted(syn.scopes),
        })
        if binding is None:
            self._unbound(syn)
        else:
            self._capture_check(syn, binding)
        return syn

    def _expand_let(self, syn: Syntax, env: Env, stack: List[dict]) -> Syntax:
        step_id = self._next("step")
        _, bind_syn, body = syn.datum
        name_syn, init_syn = bind_syn.datum
        init_out = self.expand(init_syn, env, stack)
        bscope = self._scope("bind")
        bound = self.add_scope(name_syn, {bscope}, step_id)
        body2 = self.add_scope(body, {bscope}, step_id)
        bid = self._next("bnd")
        binding = Binding(bid, name_syn.datum, bound.scopes, "let", None, name_syn.span)
        self.bindings[bid] = binding
        self.binding_edges.append({
            "from_sid": bound.sid, "binding_id": bid, "status": "binds",
            "name": name_syn.datum, "scopes": sorted(bound.scopes),
        })
        inner = Env(env)
        inner.bind(binding)
        body_out = self.expand(body2, inner, stack)
        out = self.copy_like(
            syn,
            [self.copy_like(syn.datum[0], "let", syn.datum[0].scopes, step_id),
             self.copy_like(bind_syn, [bound, init_out], bind_syn.scopes, step_id),
             body_out],
            syn.scopes, step_id)
        self.record_step({
            "step_id": step_id, "kind": "let", "parent_step": None,
            "fragment_id": None, "summary": f"let-bind {name_syn.datum}",
            "inputs": [syn.sid], "outputs": [out.sid],
            "macro_stack": [f["macro"] for f in stack], "binding_scope": bscope,
        })
        return out

    def _expand_lambda(self, syn: Syntax, env: Env, stack: List[dict]) -> Syntax:
        step_id = self._next("step")
        _, params_syn, body = syn.datum
        bscope = self._scope("bind")
        inner = Env(env)
        new_params = []
        for p in params_syn.datum:
            bound = self.add_scope(p, {bscope}, step_id)
            bid = self._next("bnd")
            binding = Binding(bid, p.datum, bound.scopes, "lambda", None, p.span)
            self.bindings[bid] = binding
            inner.bind(binding)
            self.binding_edges.append({
                "from_sid": bound.sid, "binding_id": bid, "status": "binds",
                "name": p.datum, "scopes": sorted(bound.scopes),
            })
            new_params.append(bound)
        body2 = self.add_scope(body, {bscope}, step_id)
        body_out = self.expand(body2, inner, stack)
        params_out = self.copy_like(params_syn, new_params, params_syn.scopes, step_id)
        out = self.copy_like(syn, [syn.datum[0], params_out, body_out],
                             syn.scopes, step_id)
        self.record_step({
            "step_id": step_id, "kind": "lambda", "parent_step": None,
            "fragment_id": None, "summary": "lambda",
            "inputs": [syn.sid], "outputs": [out.sid],
            "macro_stack": [f["macro"] for f in stack], "binding_scope": bscope,
        })
        return out

    # ------------------------------------------------------------------ macros
    def _expand_macro_use(self, syn: Syntax, binding: Binding, env: Env,
                          stack: List[dict]) -> Syntax:
        macro: Macro = binding.macro
        self.used += 1
        step_id = self._next("step")
        frame = {
            "macro": macro.name, "fragment_id": macro.fragment_id,
            "call_sid": syn.sid, "step_id": step_id,
            "call_span": syn.span.to_obj() if syn.span else None,
            "def_span": macro.def_span.to_obj() if macro.def_span else None,
            "stack": [f["macro"] for f in stack],
        }
        self.macro_calls.append(frame)
        self.used_fragments.add(macro.fragment_id)
        if self.used > self.budget:
            self.budget_exceeded = True
            self._diag(
                f"macro expansion budget ({self.budget} {BUDGET_UNIT}) exceeded "
                f"while expanding {macro.name!r}; partial expansion graph retained",
                syn, step_id=step_id, kind="budget-exceeded")
            raise BudgetExceeded(step_id)
        args = syn.datum[1:]
        if len(args) != len(macro.params):
            self._diag(
                f"macro {macro.name!r} expects {len(macro.params)} arguments, "
                f"got {len(args)}", syn, step_id=step_id, kind="arity-error")
            return syn
        use_scope = self._scope("usesite")
        subst = {p: a for p, a in zip(macro.params, args)}
        instance = self._instantiate(macro.template, subst, use_scope, macro,
                                     step_id, syn.sid, path="")
        self.record_step({
            "step_id": step_id, "kind": "macro-use", "parent_step": None,
            "fragment_id": macro.fragment_id,
            "summary": f"expand macro {macro.name}",
            "inputs": [syn.sid], "outputs": [instance.sid],
            "macro_stack": [f["macro"] for f in stack] + [macro.name],
            "use_scope": use_scope, "def_scope": macro.def_scope,
            "call_span": frame["call_span"], "def_span": frame["def_span"],
        })
        return self.expand(instance, env, stack + [frame])

    def _instantiate(self, template: Syntax, subst: Dict[str, Syntax],
                     use_scope: str, macro: Macro, step_id: str,
                     call_sid: str, path: str) -> Syntax:
        if template.is_ident:
            name = template.datum
            if name in subst:
                return subst[name]  # use-site argument keeps its own scopes
            scopes = template.scopes | {use_scope}
            return self.mk(name, template.span, scopes, template.sid, step_id)
        if not template.is_list:
            return self.mk(template.datum, template.span,
                           template.scopes | {use_scope}, template.sid, step_id)
        items = template.datum
        if items and items[0].is_ident:
            head = items[0].datum
            if head == "datum->syntax" and len(items) == 3:
                return self._datum_to_syntax(items[1], items[2], subst, use_scope,
                                             macro, step_id, call_sid, path)
            if head == "gensym" and len(items) == 2:
                return self._gensym(items[1], use_scope, macro, step_id)
            if head == "with-gensym" and len(items) == 3:
                names = [n.datum for n in items[1].datum]
                subst2 = dict(subst)
                for n in names:
                    subst2[n] = self._gensym_raw(n, use_scope, macro, step_id,
                                                 items[1].span)
                return self._instantiate(items[2], subst2, use_scope, macro,
                                         step_id, call_sid, path + "/wg")
        children = []
        for i, c in enumerate(items):
            children.append(self._instantiate(c, subst, use_scope, macro,
                                              step_id, call_sid, f"{path}/{i}"))
        return self.mk(children, template.span, template.scopes | {use_scope},
                       template.sid, step_id)

    def _datum_to_syntax(self, ctx_tpl: Syntax, datum_tpl: Syntax,
                         subst: Dict[str, Syntax], use_scope: str,
                         macro: Macro, step_id: str, call_sid: str,
                         path: str) -> Syntax:
        ctx = ctx_tpl.datum if ctx_tpl.is_ident else None
        datum = datum_tpl.datum if datum_tpl.is_ident else datum_tpl.datum
        defsite_scopes = frozenset(
            {self.module_scope, macro.def_scope, use_scope})
        options = [{"label": "defsite", "scopes": sorted(defsite_scopes)}]
        for p in macro.params:
            if p in subst and subst[p].is_ident:
                options.append({"label": f"param:{p}",
                                "scopes": sorted(subst[p].scopes)})
        node_key = f"{macro.fragment_id}:{path}@{call_sid}"
        event = {"node_key": node_key, "datum": datum, "options": options,
                 "chosen": None, "step_id": step_id,
                 "basis": "datum->syntax context selection is explicit"}
        recorded = False
        if ctx == "auto":
            chosen = self.context_choices.get(node_key)
            event["chosen"] = chosen
            self.context_choice_events.append(event)
            recorded = True
            cid = f"cand:{node_key}"
            if chosen is None:
                syn = self.mk(datum, datum_tpl.span, frozenset(),
                              datum_tpl.sid, step_id)
                self._pending_nodes[syn.sid] = {
                    "cid": cid, "node_key": node_key, "options": options}
                self.candidates.append({
                    "cid": cid, "node_key": node_key, "syntax_id": syn.sid,
                    "name": datum, "status": "pending",
                    "options": options,
                    "reason": "datum->syntax context is ambiguous; "
                              "all legal candidates retained",
                })
                return syn
            ctx = chosen
        if ctx == "defsite":
            scopes = defsite_scopes
            label = "defsite"
        elif ctx is not None and ctx.startswith("param:"):
            label = ctx
            scopes = frozenset(subst[ctx[6:]].scopes)
        elif ctx in subst:
            label = f"param:{ctx}"
            scopes = frozenset(subst[ctx].scopes)
        else:
            self._diag(f"datum->syntax: unknown context {ctx!r}", datum_tpl,
                       step_id=step_id, kind="datum-context-error")
            scopes = defsite_scopes
            label = "defsite"
        if not recorded:
            event["chosen"] = label
            self.context_choice_events.append(event)
        return self.mk(datum, datum_tpl.span, scopes, datum_tpl.sid, step_id,
                       explicit=True)

    def _gensym_raw(self, base: str, use_scope: str, macro: Macro,
                    step_id: str, span) -> Syntax:
        self._gensym_n += 1
        symbol = f"{base}.{self._gensym_n}"
        scopes = frozenset({self.module_scope, macro.def_scope, use_scope})
        syn = self.mk(symbol, span, scopes, None, step_id)
        self.generated_symbols.append({
            "symbol": symbol, "base": base, "sid": syn.sid, "step_id": step_id,
            "macro": macro.name, "scopes": sorted(scopes),
            "origin_span": span.to_obj() if span else None,
        })
        return syn

    def _gensym(self, base_tpl: Syntax, use_scope: str, macro: Macro,
                step_id: str) -> Syntax:
        base = base_tpl.datum
        if isinstance(base, str) and base.startswith('"'):
            base = base.strip('"')
        return self._gensym_raw(str(base), use_scope, macro, step_id,
                                base_tpl.span)

    # ------------------------------------------------------- pending & checks
    def _resolve_pending(self, syn: Syntax, pending: dict, env: Env) -> None:
        resolutions = []
        for opt in pending["options"]:
            binding, ties = self.resolve(env, syn.datum, frozenset(opt["scopes"]))
            resolutions.append({
                "label": opt["label"],
                "resolution": binding.bid if binding else None,
                "resolution_kind": binding.kind if binding else (
                    "ambiguous" if ties else "unbound"),
            })
        for cand in self.candidates:
            if cand["cid"] == pending["cid"]:
                cand["resolutions"] = resolutions
        self.binding_edges.append({
            "from_sid": syn.sid, "binding_id": None,
            "status": "pending-decision", "name": syn.datum,
            "scopes": sorted(syn.scopes), "candidate_id": pending["cid"],
        })

    def _ambiguous(self, syn: Syntax, ties: List[Binding]) -> None:
        cid = f"cand:tie-{syn.sid}"
        self.candidates.append({
            "cid": cid, "node_key": None, "syntax_id": syn.sid,
            "name": syn.datum, "status": "pending",
            "options": [{"label": b.bid, "scopes": sorted(b.scopes)} for b in ties],
            "reason": "equally maximal bindings; all legal candidates retained",
        })
        self.binding_edges.append({
            "from_sid": syn.sid, "binding_id": None,
            "status": "ambiguous", "name": syn.datum,
            "scopes": sorted(syn.scopes), "candidate_id": cid,
        })
        self._diag(f"ambiguous binding for {syn.datum!r}: "
                   f"{len(ties)} equally maximal candidates", syn,
                   kind="ambiguous-binding")

    def _unbound(self, syn: Syntax) -> None:
        self._diag(f"unbound identifier {syn.datum!r}", syn,
                   kind="unbound-identifier")

    def _capture_check(self, syn: Syntax, binding: Binding) -> None:
        introduced = any(s.startswith("scp-usesite") or s.startswith("scp-defsite")
                         for s in syn.scopes)
        capture = (introduced and binding.kind in ("let", "lambda")
                   and not (binding.scopes & syn.scopes))
        explicit = False
        for ev in reversed(self.context_choice_events):
            if ev["chosen"] and ev["chosen"].startswith("param:"):
                explicit = True
                break
        self.capture_checks.append({
            "syntax_id": syn.sid, "name": syn.datum,
            "binding_id": binding.bid, "binding_kind": binding.kind,
            "id_scopes": sorted(syn.scopes),
            "binding_scopes": sorted(binding.scopes),
            "introduced": introduced, "capture": capture,
            "explicit": explicit and capture,
            "basis": RESOLUTION_BASIS,
        })

    def _diag(self, message: str, syn: Syntax, step_id=None,
              kind="unbound-identifier") -> None:
        chain = []
        cur: Optional[Syntax] = syn
        seen = set()
        while cur is not None and cur.sid not in seen:
            seen.add(cur.sid)
            chain.append({
                "sid": cur.sid,
                "span": cur.span.to_obj() if cur.span else None,
                "datum_text": cur.datum_text()[:80],
            })
            cur = self.syntax_table.get(cur.origin) if cur.origin else None
        self.diagnostics.append({
            "did": self._next("diag"), "kind": kind, "message": message,
            "syntax_id": syn.sid,
            "span": syn.span.to_obj() if syn.span else None,
            "origin_chain": chain, "step_id": step_id,
            "macro_stack": [f["macro"] for f in self.macro_calls],
        })

    # ----------------------------------------------------------------- result
    def result(self) -> dict:
        scope_sets = {}
        for sid, syn in self.syntax_table.items():
            if syn.is_ident:
                scope_sets[sid] = sorted(syn.scopes)
        return {
            "algo_version": ALGO_VERSION,
            "syntax": {sid: s.to_obj() for sid, s in self.syntax_table.items()},
            "steps": self.steps,
            "bindings": {bid: {
                "bid": b.bid, "name": b.name, "scopes": sorted(b.scopes),
                "kind": b.kind, "fragment_id": b.fragment_id,
                "span": b.span.to_obj() if b.span else None,
            } for bid, b in self.bindings.items()},
            "binding_edges": self.binding_edges,
            "scope_sets": scope_sets,
            "source_map": self.source_map,
            "generated_symbols": self.generated_symbols,
            "macro_calls": self.macro_calls,
            "candidates": self.candidates,
            "diagnostics": self.diagnostics,
            "capture_checks": self.capture_checks,
            "context_choice_events": self.context_choice_events,
            "used_fragments": sorted(self.used_fragments),
            "budget": {"limit": self.budget, "used": self.used,
                       "exceeded": self.budget_exceeded, "unit": BUDGET_UNIT},
        }
