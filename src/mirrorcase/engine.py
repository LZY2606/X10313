"""Scope-set hygienic macro expander.

Algorithm (Flatt, "Binding as Sets of Scopes", 2016):

* every identifier carries a set of scope ids;
* at a macro call a fresh *use scope* is added (origin unaware) to the
  whole input form; the transformer template runs; afterwards the scope
  is flipped - removed from use-derived pieces, retained on introduced
  pieces (tracked via introduction labels);
* a binding ``(name, B)`` applies to a reference ``(name, R)`` when
  ``B <= R``; the *most specific* applicable binding wins (largest set,
  largest scope id as tie breaker). Capture is therefore decided purely
  by scope relations - no string renaming occurs;
* ``datum->syntax`` copies the scope/label context of another syntax
  object, making the context choice explicit and auditable;
* expansion runs under a step budget; on exhaustion expansion stops and
  the partial expansion graph is retained.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from . import ALGORITHM_VERSION
from .macropat import (ELLIPSIS, Instantiator, MatchFailure, match_pattern,
                       pattern_vars)
from .syntax import OriginLink, SPAN_BASIS, SObj, Span

CORE_FORMS = {
    "define", "define-syntax", "lambda", "λ", "if", "begin", "quote",
    "let", "set!", "and", "or",
}
PRIMITIVES = {"+", "-", "*", "/", "=", "<", ">", "<=", ">=", "cons",
              "car", "cdr", "list", "null?", "pair?", "not", "modulo"}
# template-time operators (valid only while a transformer runs)
TEMPLATE_OPS = {"datum->syntax", "syntax-gensym"}


@dataclass
class MacroRule:
    pattern: SObj
    template: SObj
    pattern_vars: set[str]


@dataclass
class MacroDef:
    name: str
    literals: set[str]
    rules: list[MacroRule]
    sid: int
    defining_event: str = ""


@dataclass
class BindingRecord:
    slot: str                 # stable slot identity
    symbol: str
    kind: str                 # local | top | core | primitive | macro
    scopes: frozenset[int]
    sid: int                  # binder syntax id
    def_event: str = ""       # macro call that introduced it ("" if user code)


@dataclass
class ExpansionConfig:
    budget: int = 64
    algorithm_version: str = ALGORITHM_VERSION

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.algorithm_version.encode())
        h.update(f"budget={self.budget}".encode())
        return "fp_" + h.hexdigest()[:16]


class Expansion:
    def __init__(self, config: ExpansionConfig, source_id: str):
        self.config = config
        self.source_id = source_id
        self.next_scope = 1
        self.next_step = 0
        self.sid_counter = [0]
        self.bindings: list[BindingRecord] = []
        self.steps: list[dict] = []
        self.diagnostics: list[dict] = []
        self.scope_ledger: list[dict] = []
        self.macro_stack: list[int] = []
        self.gensyms: list[dict] = []
        self.datum_events: list[dict] = []
        self.candidates: list[dict] = []
        self.status = "ok"

    # ---- bookkeeping --------------------------------------------------
    def fresh_scope(self, why: str, step: Optional[int] = None,
                    sid: int = -1) -> int:
        sc = self.next_scope
        self.next_scope += 1
        self.scope_ledger.append(
            {"scope": sc, "why": why, "step": step, "sid": sid})
        return sc

    def fresh_step(self) -> int:
        self.next_step += 1
        return self.next_step

    def diag(self, code: str, message: str, sobj: Optional[SObj],
             extra: Optional[dict] = None) -> None:
        d = {
            "code": code,
            "message": message,
            "sid": sobj.sid if sobj is not None else None,
            "span": sobj.span.to_json() if sobj is not None and sobj.span
                    else None,
            "origin": (sobj.origin.chain() if sobj is not None and sobj.origin
                       else []),
            "macro_stack": list(self.macro_stack),
        }
        if extra:
            d.update(extra)
        self.diagnostics.append(d)

    # ---- bindings -----------------------------------------------------
    def install(self, symbol: str, scopes: frozenset[int], kind: str,
                sid: int, def_event: str = "") -> BindingRecord:
        # identical redeclaration updates the slot instead of duplicating
        for b in self.bindings:
            if b.symbol == symbol and b.scopes == scopes and b.kind == kind:
                return b
        slot = f"slot_{len(self.bindings):04d}_{symbol}"
        rec = BindingRecord(slot, symbol, kind, scopes, sid, def_event)
        self.bindings.append(rec)
        return rec

    def lookup(self, symbol: str, scopes: frozenset[int],
               sid: int = -1) -> tuple[Optional[BindingRecord], list[str]]:
        """Return (binding-or-None, candidate slot ids).

        Applicable: ``B.scopes <= scopes``. Winner is the maximum under
        the ordering ``B1 < B2`` iff B2 has a distinguishing scope with
        id larger than every scope distinguishing B1 (i.e. larger scopes
        win; later/local scopes dominate). Multiple maxima => ambiguous,
        all legal candidates are recorded.
        """
        app = [b for b in self.bindings
               if b.symbol == symbol and b.scopes <= scopes]
        if not app:
            return None, []

        def dist(a: BindingRecord, b: BindingRecord) -> bool:
            only_b = b.scopes - a.scopes
            only_a = a.scopes - b.scopes
            if not only_b:
                return False
            if not only_a:
                return True  # strict superset
            return max(only_b) > max(only_a)

        maxima = [b for b in app
                  if not any(dist(b, o) for o in app if o is not b)]
        if len(maxima) == 1:
            self.candidates.append({
                "sid": sid, "symbol": symbol, "winner": maxima[0].slot,
                "considered": [b.slot for b in app],
                "rejected": [
                    {"slot": b.slot,
                     "reason": ("outscoped" if b.scopes <= maxima[0].scopes
                                else "incomparable")}
                    for b in app if b is not maxima[0]],
                "status": "resolved",
            })
            return maxima[0], [b.slot for b in app]
        slots = [b.slot for b in maxima]
        self.candidates.append({
            "sid": sid, "symbol": symbol, "winner": None,
            "considered": [b.slot for b in app],
            "rejected": [],
            "status": "ambiguous",
            "message": "no unique most-specific binding; all legal "
                       "candidates retained",
        })
        self.diag("ambiguous-binding",
                  f"identifier {symbol!r} has multiple maximal bindings",
                  None, {"candidates": slots})
        return None, [b.slot for b in app]

    # ---- template-time operators -------------------------------------
    def _prepare_template(self, t: SObj, ctx_sobj: SObj,
                          step: int, scope: int) -> SObj:
        """Pre-walk a transformer template before instantiation.

        Handles ``(datum->syntax <id> <sym-id>)`` and
        ``(syntax-gensym <base-id>)``. These can only meaningfully appear
        inside a running transformer; the context they select is recorded
        as a first-class, auditable event.
        """
        if not t.is_form:
            return t
        head = t.head_symbol()
        if head == "datum->syntax" and len(t.items) == 3:
            ctx_name = t.items[1]
            sym_name = t.items[2]
            ctx = self._find_ctx(ctx_name, ctx_sobj)
            out = SObj.leaf(
                sym_name.atom, "id",
                Span.synthetic_at("datum->syntax", sym_name.span),
                OriginLink("datum->syntax", step, sym_name.origin))
            out.scopes = ctx.scopes
            # selecting the context also adopts its introduction labels,
            # so the flip keeps whichever scope the context carries
            out.labels = ctx.labels | {step}
            out.sid = self.sid_counter[0] = self.sid_counter[0] + 1
            self.datum_events.append({
                "step": step,
                "produced": out.symbol,
                "out_sid": out.sid,
                "context_sid": ctx.sid,
                "context_symbol": ctx.atom if ctx.kind == "id" else None,
                "context_scopes": sorted(ctx.scopes),
                "context_labels": sorted(ctx.labels),
                "scopes": sorted(out.scopes),
                "rule": "scope-relations-not-renaming",
            })
            return out
        if head == "syntax-gensym" and len(t.items) == 2:
            base = t.items[1]
            if base.kind != "id":
                self.diag("bad-gensym", "syntax-gensym expects an identifier",
                          t)
                return t
            n = len(self.gensyms)
            name = f"{base.atom}#g{self.source_id[-8:]}_{step}_{n}"
            out = SObj.leaf(
                name, "id",
                Span.synthetic_at("syntax-gensym", base.span),
                OriginLink("syntax-gensym", step, base.origin))
            out.scopes = frozenset({scope})
            out.labels = frozenset({step})
            out.gen_name = name
            out.sid = self.sid_counter[0] = self.sid_counter[0] + 1
            self.gensyms.append({
                "step": step, "sid": out.sid, "base": base.atom,
                "name": name, "scopes": [scope], "labels": [step],
            })
            return out
        new_items = [self._prepare_template(c, ctx_sobj, step, scope)
                     for c in t.items]
        new_tail = (self._prepare_template(t.tail, ctx_sobj, step, scope)
                    if t.tail else None)
        t.items = new_items
        t.tail = new_tail
        return t

    def _find_ctx(self, ctx_pat: SObj, ctx_sobj: SObj) -> SObj:
        # ctx is a pattern variable: find the matching sub-syntax object
        # in the macro call input by symbol path fallback
        target = ctx_pat.atom
        found: list[SObj] = []

        def walk(o: SObj) -> None:
            if o.kind == "id" and o.atom == target:
                found.append(o)
            if o.is_form:
                for c in o.items:
                    walk(c)
                if o.tail:
                    walk(o.tail)

        walk(ctx_sobj)
        return found[0] if found else ctx_sobj

    # ---- macro invocation ---------------------------------------------
    def invoke_macro(self, form: SObj, macro: MacroDef) -> SObj:
        step = self.fresh_step()
        if step > self.config.budget:
            self.status = "budget-exceeded"
            self.steps.append({
                "step": step, "macro": macro.name,
                "status": "budget-exceeded",
                "message": ("expansion budget exhausted; partial expansion "
                            "graph retained"),
                "sid": form.sid,
                "input": self._snapshot(form),
                "macro_stack": list(self.macro_stack),
                "rule_rejections": [],
            })
            self.diag("budget-exceeded",
                      f"expansion budget {self.config.budget} exhausted", form)
            return form
        use_scope = self.fresh_scope("macro-use", step, form.sid)
        self.macro_stack.append(step)
        self.steps.append({
            "step": step, "macro": macro.name, "status": "entered",
            "sid": form.sid, "macro_sid": macro.sid,
            "macro_stack": list(self.macro_stack),
            "scopes_added": [use_scope],
        })
        form.add_scope_inplace(use_scope)

        matched = False
        rejections: list[dict] = []
        result: Optional[SObj] = None
        for ri, rule in enumerate(macro.rules):
            try:
                binds = match_pattern(rule.pattern, form, macro.literals, {})
            except MatchFailure as mf:
                rejections.append({"rule": ri, "reason": mf.reason})
                continue
            matched = True
            self._prepare_template(rule.template, form, step, use_scope)
            inst = Instantiator(binds, rule.pattern_vars, step, use_scope,
                                macro.literals)
            result = inst.run(rule.template)
            # allocate stable sids for the newly introduced subtree
            self._assign_new_sids(result)
            self.steps.append({
                "step": step, "macro": macro.name, "status": "produced",
                "rule": ri, "output": self._snapshot(result),
            })
            break

        if not matched:
            self.macro_stack.pop()
            self.diag("no-matching-rule",
                      f"macro {macro.name!r}: no rule matched", form,
                      {"rule_rejections": rejections})
            self.steps.append({
                "step": step, "macro": macro.name,
                "status": "no-matching-rule",
                "rule_rejections": rejections,
            })
            self.status = "stuck"
            return form

        # origin-aware flip: use pieces lose the scope, introduced keep it
        result.flip_inplace(use_scope, step)  # type: ignore[union-attr]
        self.steps.append({
            "step": step, "macro": macro.name, "status": "flipped",
            "rule_rejections": rejections,
        })
        self.macro_stack.pop()
        # continue expansion of the result under the shared budget
        expanded = self.expand(result)  # type: ignore[arg-type]
        self.steps.append({
            "step": step, "macro": macro.name, "status": "completed",
            "output": self._snapshot(expanded),
            "macro_stack": list(self.macro_stack),
        })
        return expanded

    def _assign_new_sids(self, o: SObj) -> None:
        stack = [o]
        while stack:
            n = stack.pop()
            if n.sid < 0:
                self.sid_counter[0] += 1
                n.sid = self.sid_counter[0]
            if n.is_form:
                if n.tail:
                    stack.append(n.tail)
                stack.extend(reversed(n.items))

    # ---- expansion ----------------------------------------------------
    def expand(self, form: SObj) -> SObj:
        if self.status == "budget-exceeded":
            return form
        if not form.is_form:
            return form
        head = form.items[0] if form.items else None
        if head is not None and head.kind == "id":
            macro = self.macro_env.get(head.atom)
            binding, _ = self.lookup(head.atom, head.scopes, head.sid)
            if macro is not None and binding is not None and binding.kind == "macro":
                return self.invoke_macro(form, macro)
        return self.expand_core(form)

    def expand_core(self, form: SObj) -> SObj:
        head = form.head_symbol()
        if head == "quote" and len(form.items) == 2:
            return form
        if head in ("lambda", "λ"):
            return self._expand_lambda(form)
        if head == "if":
            return self._expand_nary(form, 3)
        if head in ("begin", "and", "or", "set!"):
            form.items = [self.expand(c) for c in form.items]
            if form.tail:
                form.tail = self.expand(form.tail)
            return form
        if head == "let":
            return self._expand_let(form)
        if head in TEMPLATE_OPS:
            self.diag("template-op-outside-macro",
                      f"{head} is only valid inside a macro template", form)
            self.status = "stuck"
            return form
        # application
        expanded_items: list[SObj] = []
        for c in form.items:
            ec = self.expand(c)
            if ec.is_form and ec is not c:
                # macro may have expanded a subform
                pass
            expanded_items.append(ec)
        form.items = expanded_items
        if form.tail:
            form.tail = self.expand(form.tail)
        self._record_refs(form)
        return form

    def _expand_nary(self, form: SObj, arity: int) -> SObj:
        if len(form.items) - 1 != arity:
            self.diag("arity-error",
                      f"{form.head_symbol()} expects {arity} forms", form)
        form.items = [self.expand(c) for c in form.items]
        return form

    def _expand_lambda(self, form: SObj) -> SObj:
        if len(form.items) < 3 or not form.items[1].is_form:
            self.diag("bad-lambda", "malformed lambda", form)
            return form
        params = form.items[1]
        scope = self.fresh_scope("lambda", sid=params.sid)
        for p in params.items:
            if p.kind == "id":
                self.install(p.atom, p.scopes, "local", p.sid,
                             def_event=(f"step{self.macro_stack[-1]}"
                                        if self.macro_stack else ""))
        if params.tail and params.tail.kind == "id":
            self.install(params.tail.atom, params.tail.scopes, "local",
                         params.tail.sid)
        bodies = [self.expand(c) for c in form.items[2:]]
        for b in bodies:
            b.add_scope_inplace(scope)
        form.items = [form.items[0], params] + bodies
        return form

    def _expand_let(self, form: SObj) -> SObj:
        if len(form.items) != 3 or not form.items[1].is_form:
            self.diag("bad-let", "malformed let", form)
            return form
        scope = self.fresh_scope("let", sid=form.items[1].sid)
        # Flatt: the binder is installed WITHOUT the new scope; the new
        # scope is added to the body (and body only). The binder applies
        # wherever its (smaller) scope set is a subset of the ref's.
        new_clauses: list[SObj] = []
        for clause in form.items[1].items:
            if not clause.is_form or len(clause.items) != 2:
                self.diag("bad-let-binding", "malformed let binding", clause)
                new_clauses.append(clause)
                continue
            name = clause.items[0]
            rhs = self.expand(clause.items[1])
            self.install(name.atom, name.scopes, "local", name.sid,
                         def_event=(f"step{self.macro_stack[-1]}"
                                    if self.macro_stack else ""))
            clause.items[1] = rhs
            new_clauses.append(clause)
        form.items[1].items = new_clauses
        form.items[2] = self.expand(form.items[2])
        form.items[2].add_scope_inplace(scope)
        return form

    def _record_refs(self, form: SObj) -> None:
        # kept for internal use; full resolution happens after expansion
        pass

    def resolve_all_refs(self, form: SObj) -> None:
        """Walk the final expanded tree and record every identifier
        reference as a candidate/binding edge (once per sid)."""
        seen: set[int] = set()
        stack: list[SObj] = [form]
        while stack:
            n = stack.pop()
            if n.is_form:
                if n.tail:
                    stack.append(n.tail)
                stack.extend(reversed(n.items))
                continue
            if n.kind == "id" and n.sid not in seen:
                seen.add(n.sid)
                self.lookup(n.atom, n.scopes, n.sid)

    # ---- snapshots / serialization -----------------------------------
    def _snapshot(self, o: SObj, depth: int = 0) -> Any:
        """JSON-able syntax snapshot including scopes, labels, origin."""
        if depth > 40:
            return {"sid": o.sid, "truncated": True}
        base: dict[str, Any] = {
            "sid": o.sid,
            "kind": o.kind,
            "scopes": sorted(o.scopes),
            "labels": sorted(o.labels),
            "origin": o.origin.chain() if o.origin else [],
        }
        if o.span is not None:
            base["span"] = o.span.to_json()
        if o.gen_name:
            base["gen_name"] = o.gen_name
        if o.is_form:
            base["items"] = [self._snapshot(c, depth + 1) for c in o.items]
            if o.tail is not None:
                base["tail"] = self._snapshot(o.tail, depth + 1)
        else:
            base["atom"] = o.atom
        return base


# ---------------------------------------------------------------- driver

def _parse_syntax_rules(form: SObj, literals_src: SObj,
                        rules_src: list[SObj], exp: Expansion,
                        sid_counter: list[int]) -> MacroDef:
    lits: set[str] = set()
    for t in literals_src.items:
        if t.kind == "id":
            lits.add(t.atom)
    rules: list[MacroRule] = []
    for r in rules_src:
        if not r.is_form or len(r.items) != 2:
            raise ValueError("syntax-rules clause must be (pat tmpl)")
        pat, tmpl = r.items
        # definition-site scopes are already on pattern/template (module)
        rules.append(MacroRule(pat, tmpl, pattern_vars(pat, lits)))
    name = form.items[1].atom
    return MacroDef(name=name, literals=lits, rules=rules,
                    sid=form.items[1].sid)


def run_program(forms: list[SObj], tokens: list[Any],
                config: ExpansionConfig, source_id: str,
                reader_diagnostics: Optional[list[dict]] = None) -> dict:
    """Top-level driver. Input sids are assigned first (preorder), so
    reader syntax ids are stable before any macro step allocates ids."""
    exp = Expansion(config, source_id)
    exp.macro_env: dict[str, MacroDef] = {}  # type: ignore[attr-defined]

    # stable sids for the original program
    for f in forms:
        exp._assign_new_sids(f)
    module_scope = exp.fresh_scope("module", sid=0)

    # core / primitive bindings visible at module scope
    for c in CORE_FORMS:
        exp.install(c, frozenset({module_scope}), "core", -1)
    for p in PRIMITIVES:
        exp.install(p, frozenset({module_scope}), "primitive", -1)

    expanded_top: list[dict] = []
    top_events: list[dict] = []
    idx = 0
    while idx < len(forms):
        f = forms[idx]
        f.add_scope_inplace(module_scope)
        head = f.head_symbol()
        if head == "define-syntax":
            name_obj = f.items[1]
            sr = f.items[2]
            if sr.head_symbol() != "syntax-rules":
                exp.diag("bad-syntax-rules",
                         "define-syntax expects syntax-rules", f)
                idx += 1
                continue
            macro = _parse_syntax_rules(f, sr.items[1], sr.items[2:],
                                        exp, exp.sid_counter)
            exp.macro_env[macro.name] = macro  # type: ignore[attr-defined]
            exp.install(macro.name, name_obj.scopes, "macro", name_obj.sid)
            top_events.append({"kind": "define-syntax", "name": macro.name,
                               "sid": name_obj.sid, "rules": len(macro.rules)})
            idx += 1
            continue
        if head == "define" and len(f.items) >= 3 and f.items[1].kind == "id":
            name_obj = f.items[1]
            f.items = [f.items[0], name_obj] + [
                exp.expand(c) for c in f.items[2:]]
            exp.install(name_obj.atom, name_obj.scopes, "top", name_obj.sid)
            top_events.append({"kind": "define", "name": name_obj.atom,
                               "sid": name_obj.sid})
            exp.resolve_all_refs(f)
            expanded_top.append(exp._snapshot(f))
            idx += 1
            continue
        out = exp.expand(f)
        exp.resolve_all_refs(out)
        expanded_top.append(exp._snapshot(out))
        idx += 1

    binding_edges = _binding_edges(exp)
    return {
        "status": exp.status,
        "algorithm_version": config.algorithm_version,
        "parameter_fingerprint": config.fingerprint(),
        "budget": config.budget,
        "span_basis": SPAN_BASIS,
        "source_id": source_id,
        "tokens": [t.to_json() for t in tokens],
        "expanded": expanded_top,
        "top_events": top_events,
        "steps": exp.steps,
        "scope_ledger": exp.scope_ledger,
        "bindings": [
            {"slot": b.slot, "symbol": b.symbol, "kind": b.kind,
             "scopes": sorted(b.scopes), "sid": b.sid,
             "def_event": b.def_event}
            for b in exp.bindings
        ],
        "binding_edges": binding_edges,
        "gensyms": exp.gensyms,
        "datum_events": exp.datum_events,
        "candidates": exp.candidates,
        "diagnostics": (reader_diagnostics or []) + exp.diagnostics,
        "next_scope": exp.next_scope,
        "next_sid": exp.sid_counter[0],
    }


def _binding_edges(exp: Expansion) -> list[dict]:
    """Resolve every reference recorded in candidates to its binder."""
    edges: list[dict] = []
    for cand in exp.candidates:
        if cand["winner"]:
            b = next(x for x in exp.bindings if x.slot == cand["winner"])
            edges.append({
                "ref_sid": cand["sid"], "symbol": cand["symbol"],
                "binder_sid": b.sid, "binder_kind": b.kind,
                "slot": b.slot, "relation": "most-specific-subset",
            })
        else:
            edges.append({
                "ref_sid": cand["sid"], "symbol": cand["symbol"],
                "binder_sid": None, "binder_kind": "ambiguous",
                "slot": None, "relation": "ambiguous",
                "candidates": cand["considered"],
            })
    return edges
