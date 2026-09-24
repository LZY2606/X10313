"""Hygienic macro expander with full provenance.

Algorithm (sets-of-scopes style, simplified):

* Each binder (``lambda``/``let``) allocates a fresh *binding scope* which
  is added to the body; capture is decided purely by scope-set subset
  relations (see :mod:`macroscope.binding`), never by renaming strings.
* Each macro invocation allocates a fresh *mark* that is flipped onto the
  whole argument form before invocation and flipped onto the result after
  invocation.  Identifiers introduced by the transformer therefore never
  capture (and cannot be captured by) user code, unless the user explicitly
  hands them a context via ``datum->syntax``.
* ``datum->syntax`` records its *context choice* as a first-class visible
  event and propagates that context's scope set; that is the one sanctioned
  escape hatch ("explicit capture").

The expander runs under a macro-call budget.  When the budget is exhausted
expansion stops, the partial expansion graph is retained, and the result is
marked ``budget-exhausted`` rather than being discarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .binding import Binding, resolve
from .ctenv import CTError, Closure, ListVal
from .ctinterp import MetaEvaluator
from .syntax import (Diagnostic, GraphBuilder, Stx, add_scope, flip_scope)

CORE_FORMS = frozenset({
    "lambda", "λ", "let", "let*", "if", "begin", "quote", "quasiquote",
    "set!", "define", "define-syntax", "datum->syntax", "#%app", "#%core",
})


@dataclass
class MacroEntry:
    name: str
    def_site_id: str
    def_scopes: frozenset[str]
    closure: Closure
    source_version: str
    def_env_scope: str = ""
    captured_bindings: tuple = ()   # Binding objects visible at def site


@dataclass
class ExpanderConfig:
    macro_budget: int = 64
    seed: int = 20260925
    algorithm_version: str = "expander-1.0.0"

    def fingerprint(self) -> dict[str, Any]:
        return {"algorithm_version": self.algorithm_version,
                "macro_budget": self.macro_budget, "seed": self.seed}


@dataclass
class ExpansionResult:
    root: Optional[Stx]
    status: str                      # complete | budget-exhausted | failed
    bindings: list[Binding]
    resolutions: dict[str, Any]
    macro_stack: list[dict[str, Any]]
    diagnostics: list[Diagnostic]
    call_count: int
    budget: int
    generated_symbols: list[dict[str, Any]]


class ExpandContext:
    def __init__(self, builder: GraphBuilder, config: ExpanderConfig,
                 input_version_id: str):
        self.builder = builder
        self.config = config
        self.input_version_id = input_version_id
        self.macros: list[MacroEntry] = []
        self.bindings: list[Binding] = []
        self.resolutions: dict[str, Any] = {}
        self.stack: list[dict[str, Any]] = []
        self.diagnostics: list[Diagnostic] = []
        self.call_count = 0
        self.generated: list[dict[str, Any]] = []
        self.partial_root: Optional[Stx] = None

    # -- binding helpers ---------------------------------------------------
    def add_binding(self, symbol: str, scope: str, kind: str, site: str,
                    scopes: frozenset[str], detail: str = "") -> None:
        self.bindings.append(
            Binding(f"bind:{scope}", symbol, kind,
                    frozenset(scopes) | {scope}, site, detail))

    def resolve_id(self, node: Stx, *, macro_mode: bool = False
                   ) -> tuple[str, Optional[Any]]:
        """Return (status, binding-or-macro-entry).

        Macro resolution happens first; a local ``let`` shadowing a macro
        name resolves to the local because macro eligibility requires the
        definition-site scopes to be a subset of the use site.
        """
        if macro_mode:
            macro_entry = self._resolve_macro(node)
            if macro_entry is not None:
                self.resolutions[node.id] = {
                    "node_id": node.id, "symbol": node.value,
                    "status": "macro",
                    "binding": {"key": f"macro:{macro_entry.name}",
                                "symbol": node.value, "kind": "macro",
                                "binder_scopes": sorted(
                                    macro_entry.def_scopes),
                                "site": macro_entry.def_site_id,
                                "detail": "macro definition site"},
                    "candidates": [],
                    "scope_index": {},
                    "use_scopes": sorted(node.scopes)}
                return "macro", macro_entry
        effective = list(self.bindings)
        for entry in self.macros:
            if entry.def_env_scope and entry.def_env_scope in node.scopes:
                for b in entry.captured_bindings:
                    aliased = Binding(
                        b.key + ":" + entry.name, b.symbol,
                        "module@defsite",
                        frozenset({entry.def_env_scope}),
                        b.site,
                        f"visible at definition site of {entry.name}")
                    effective.append(aliased)
        r = resolve(node.value, node.scopes, effective, node.id)
        self.resolutions[node.id] = r.as_dict()
        if r.status == "ambiguous":
            self.diagnostics.append(Diagnostic(
                "E300", f"ambiguous identifier {node.value!r}: "
                        f"{len(r.candidates)} incomparable maximal binders",
                "unknown", node.span, node.id))
        return r.status, r.binding

    def _resolve_macro(self, node: Stx) -> Optional[MacroEntry]:
        candidates = [m for m in self.macros if m.name == node.value]
        eligible: list[MacroEntry] = []
        for m in candidates:
            if m.def_scopes <= node.scopes:
                eligible.append(m)
        if not eligible:
            return None
        # most specific definition-site scope set wins
        eligible.sort(key=lambda m: len(m.def_scopes & node.scopes),
                      reverse=True)
        return eligible[0]

    # -- expansion ---------------------------------------------------------
    def expand_program(self, root: Stx) -> Stx:
        top_scope = "g:module.001"
        out: list[Stx] = []
        for form in root.seq:
            self.partial_root = root
            out.append(self.expand_toplevel(form, top_scope))
        expanded = self.builder.node(
            kind="seq", lexeme="<program-expanded>", span=None,
            reason="expand-program", seq=tuple(out))
        self.partial_root = expanded
        self._top_scope = top_scope
        return expanded

    def expand_toplevel(self, stx: Stx, top_scope: str) -> Stx:
        if stx.kind != "seq" or not stx.seq:
            return self.expand_expr(stx, frozenset({top_scope}))
        head = stx.seq[0]
        if head.is_id("define-syntax"):
            return self.expand_define_syntax(stx, top_scope)
        if head.is_id("define"):
            return self.expand_define(stx, frozenset({top_scope}), top=True)
        return self.expand_expr(stx, frozenset({top_scope}))

    def expand_define_syntax(self, stx: Stx, top_scope: str = "g:module.001") -> Stx:
        name_node = self.builder.clone(
            stx.seq[1], scopes=stx.seq[1].scopes | {top_scope},
            reason="define-syntax-module-scope")
        body = stx.seq[2]
        meta = MetaEvaluator(self.builder, name_node.value)
        closure = meta.eval(body, {})
        if not isinstance(closure, Closure):
            raise CTError("define-syntax body must evaluate to a lambda",
                          body)
        def_env_scope = self.builder.fresh_binding(f"defenv:{name_node.value}")
        captured = tuple(b for b in self.bindings
                         if "g:module.001" in b.binder_scopes)
        entry = MacroEntry(name_node.value, name_node.id,
                           frozenset(name_node.scopes), closure,
                           self.input_version_id,
                           def_env_scope=def_env_scope,
                           captured_bindings=captured)
        self.macros.append(entry)
        self.builder.log_step(
            "define-syntax", name=name_node.value,
            def_site=name_node.id, def_scopes=sorted(name_node.scopes),
            params=list(closure.params))
        return self.builder.node(
            kind="seq", lexeme="<syntax-def>",
            reason="define-syntax-recorded",
            seq=(self.builder.node(kind="id", value="#%syntax-def",
                                   lexeme="#%syntax-def",
                                   scopes=frozenset(name_node.scopes),
                                   parents=(name_node.id,),
                                   reason="syntax-def-marker"),
                 self.builder.clone(name_node, reason="syntax-def-name")))

    def expand_define(self, stx: Stx, scopes: frozenset[str], *,
                      top: bool = False) -> Stx:
        name_node = stx.seq[1]
        scope = self.builder.fresh_binding(name_node.value)
        self.add_binding(name_node.value, scope,
                         "module" if top else "local",
                         name_node.id, frozenset(scopes),
                         "define binding site")
        body = self.expand_expr(stx.seq[2], scopes)
        bound = self.builder.clone(
            name_node, scopes=name_node.scopes | scopes | {scope},
            reason=f"bind-define {scope}")
        return self.builder.node(
            kind="seq", lexeme="()", span=stx.span,
            parents=(stx.id,), reason="expand-define",
            seq=(self.builder.clone(stx.seq[0], reason="core define"),
                 bound, body))

    def expand_expr(self, stx: Stx, scopes: frozenset[str]) -> Stx:
        if stx.kind != "seq" or not stx.seq:
            return self.expand_atom(stx, scopes)

        if scopes - stx.scopes:
            stx = self.builder.clone(
                stx, scopes=stx.scopes | scopes,
                reason="enter-body-scope",
                seq=tuple(self.builder.clone(
                    c, scopes=c.scopes | scopes,
                    reason="enter-body-scope-child")
                    if scopes - c.scopes else c for c in stx.seq))

        head = stx.seq[0]
        status, binding = (self.resolve_id(head, macro_mode=True)
                           if head.kind == "id" else ("free", None))
        if status == "macro":
            return self.invoke_macro(stx, binding, scopes)

        if head.kind == "id" and head.value in ("lambda", "λ"):
            return self.expand_lambda(stx, scopes)
        if head.is_id("let"):
            return self.expand_let(stx, scopes, recursive=False)
        if head.is_id("let*"):
            return self.expand_let(stx, scopes, recursive=True)
        if head.is_id("if"):
            items = [self.expand_expr(s, scopes) for s in stx.seq[1:]]
            return self.rebuild_core(stx, "expand-if", tuple(items))
        if head.is_id("begin"):
            items = [self.expand_expr(s, scopes) for s in stx.seq[1:]]
            return self.rebuild_core(stx, "expand-begin", tuple(items))
        if head.is_id("set!"):
            name = self.expand_atom(stx.seq[1], scopes)
            val = self.expand_expr(stx.seq[2], scopes)
            return self.rebuild_core(stx, "expand-set!", (name, val))
        if head.is_id("quote"):
            return stx
        if head.is_id("quasiquote"):
            return stx
        # plain application
        items = tuple(self.expand_expr(s, scopes) for s in stx.seq)
        app = self.rebuild_core(stx, "expand-app", items)
        if status == "ambiguous":
            return self.builder.clone(app, flags=app.flags + ("ambiguous-head",))
        return app

    def rebuild_core(self, stx: Stx, reason: str, items: tuple[Stx, ...]) -> Stx:
        return self.builder.node(
            kind="seq", lexeme=stx.bracket, span=stx.span,
            parents=(stx.id,), reason=reason, seq=items, bracket=stx.bracket)

    def expand_atom(self, stx: Stx, scopes: frozenset[str]) -> Stx:
        if stx.kind != "id":
            return stx
        status, _ = self.resolve_id(stx)
        flags = stx.flags
        if status == "free":
            flags = flags + ("free-identifier",)
        elif status == "ambiguous":
            flags = flags + ("ambiguous",)
        if scopes - stx.scopes:
            stx = self.builder.clone(
                stx, scopes=stx.scopes | scopes,
                reason="add-body-scopes", flags=flags)
        elif flags != stx.flags:
            stx = self.builder.clone(stx, flags=flags)
        return stx

    def expand_lambda(self, stx: Stx, scopes: frozenset[str]) -> Stx:
        params = stx.seq[1].seq
        scope = self.builder.fresh_binding("lambda")
        new_params = []
        for p in params:
            self.add_binding(p.value, scope, "local", p.id,
                             frozenset(scopes), "lambda parameter")
            bound_p = self.builder.clone(
                p, scopes=p.scopes | scopes | {scope},
                reason=f"bind-param {scope}")
            self.resolve_id(bound_p)
            new_params.append(bound_p)
        body_scopes = scopes | {scope}
        body = tuple(self.expand_expr(s, body_scopes) for s in stx.seq[2:])
        head = self.builder.clone(stx.seq[0], reason="core lambda")
        plist = self.builder.node(
            kind="seq", lexeme="()", span=stx.seq[1].span,
            parents=(stx.seq[1].id,), reason="lambda-params",
            seq=tuple(new_params))
        return self.builder.node(
            kind="seq", lexeme="()", span=stx.span, parents=(stx.id,),
            reason="expand-lambda", seq=(head, plist) + body)

    def expand_let(self, stx: Stx, scopes: frozenset[str], *,
                   recursive: bool) -> Stx:
        clauses = stx.seq[1].seq
        scope = self.builder.fresh_binding("letrec" if recursive else "let")
        rhs_scopes = scopes | ({scope} if recursive else frozenset())
        new_clauses = []
        for cl in clauses:
            name = cl.seq[0]
            rhs = self.expand_expr(cl.seq[1], rhs_scopes)
            self.add_binding(name.value, scope, "local", name.id,
                             frozenset(scopes),
                             "letrec binding" if recursive else "let binding")
            bound_name = self.builder.clone(
                name, scopes=name.scopes | scopes | {scope},
                reason=f"bind-let {scope}")
            self.resolve_id(bound_name)
            new_clauses.append(self.builder.node(
                kind="seq", lexeme="()", span=cl.span, parents=(cl.id,),
                reason="let-clause", seq=(bound_name, rhs)))
        body_scopes = scopes | {scope}
        body = tuple(self.expand_expr(s, body_scopes) for s in stx.seq[2:])
        clist = self.builder.node(
            kind="seq", lexeme="()", span=stx.seq[1].span,
            parents=(stx.seq[1].id,),
            reason="letrec-clauses" if recursive else "let-clauses",
            seq=tuple(new_clauses))
        return self.builder.node(
            kind="seq", lexeme="()", span=stx.span, parents=(stx.id,),
            reason="expand-letrec" if recursive else "expand-let",
            seq=(self.builder.clone(stx.seq[0], reason="core let"),
                 clist) + body)

    # -- macro invocation --------------------------------------------------
    def invoke_macro(self, stx: Stx, entry: "MacroEntry",
                     scopes: frozenset[str]) -> Stx:
        self.call_count += 1
        if self.call_count > self.config.macro_budget:
            self.diagnostics.append(Diagnostic(
                "E400",
                f"macro expansion budget {self.config.macro_budget} "
                "exhausted; partial expansion retained",
                "unknown", stx.span, stx.id))
            partial = self.rebuild_core(stx, "macro-budget-exhausted",
                                        tuple(stx.seq))
            return self.builder.clone(
                partial, flags=partial.flags + ("budget-exhausted",))

        name = entry.name
        mark = self.builder.fresh_mark(name)
        frame = {
            "call": self.call_count, "macro": name, "site": stx.id,
            "span": stx.span.as_dict() if stx.span else None,
            "mark": mark, "def_site": entry.def_site_id,
            "def_scopes": sorted(entry.def_scopes),
            "budget_remaining": self.config.macro_budget - self.call_count,
        }
        self.stack.append(frame)

        # Flip the mark onto the argument form (pre-invocation flip).
        marked = flip_scope(self.builder, stx, mark)
        self.builder.log_step("macro-enter", **frame,
                              input_node=stx.id, marked_node=marked.id)

        meta = self._make_macro_meta(entry)
        try:
            result_host = meta.eval_callable(entry.closure, [marked])
            result = self._host_to_stx(result_host, stx, entry, marked)
        except CTError as exc:
            frame["error"] = exc.args[0]
            self.builder.log_step("macro-error", **frame,
                                  message=exc.args[0],
                                  error_node=(exc.node.id if exc.node
                                              else stx.id))
            self.diagnostics.append(Diagnostic(
                "E410", f"macro {name!r} failed: {exc.args[0]}", "error",
                (exc.node.span if exc.node and exc.node.span else stx.span),
                exc.node.id if exc.node else stx.id))
            self.stack.pop()
            failed = self.rebuild_core(
                stx, f"macro-error:{name}",
                tuple(marked.seq) if marked.kind == "seq" else ())
            return self.builder.clone(
                failed, flags=failed.flags + ("macro-error",))

        # Post-invocation flip: introduced identifiers gain the mark, so
        # they cannot collide with user identifiers (hygiene).
        flipped = flip_scope(self.builder, result, mark)
        def_env = getattr(entry, "def_env_scope", None)
        if def_env is not None:
            flipped = self._add_definition_env(flipped, def_env, entry.name)
            self.builder.log_step("definition-environment",
                                  macro=entry.name, scope=def_env,
                                  node=flipped.id)
        # Restore the real source call site span on the top form.
        self.builder.log_step(
            "macro-exit", call=self.call_count, macro=name,
            result_node=flipped.id, stack_depth=len(self.stack))
        self.stack.pop()
        # Re-expand the macro output: nested macros must fire, core forms
        # introduced by the template bind their own parameters, and free
        # identifier references get resolved against the surrounding scopes.
        expanded = self.expand_expr(flipped, scopes)
        return self.builder.clone(expanded, parents=(stx.id,),
                                 reason=f"macro-call:{name}")

    def _add_definition_env(self, stx: Stx, scope: str,
                            macro_name: str) -> Stx:
        """Give *introduced* identifiers the macro's definition-env scope.

        Use-site syntax that flowed through an unquote does not carry the
        template introduction scope, so it is deliberately excluded.
        """
        intro_prefixes = ("i:tpl:", "i:tmp:")

        def walk(node: Stx) -> Stx:
            if node.kind == "seq":
                items = tuple(walk(c) for c in node.seq)
                return self.builder.clone(
                    node, seq=items, reason="def-env-seq")
            if node.kind == "id" and any(
                    sc.startswith(intro_prefixes) for sc in node.scopes):
                return self.builder.clone(
                    node, scopes=node.scopes | {scope},
                    reason=f"def-env-identifier:{macro_name}")
            return node

        return walk(stx)

    def _find_entry(self, name: str) -> MacroEntry:
        cands = [m for m in self.macros if m.name == name]
        cands.sort(key=lambda m: len(m.def_scopes), reverse=True)
        return cands[0]

    def _make_macro_meta(self, entry: MacroEntry) -> "MacroMeta":
        return MacroMeta(self, entry)

    def _host_to_stx(self, host: Any, call_stx: Stx, entry: MacroEntry,
                     marked: Stx) -> Stx:
        """Convert a host value (datums + Stx) into one syntax object.

        Plain symbols/strings appearing as host data are converted with
        ``datum->syntax`` semantics; the visible context-choice event is
        emitted by :class:`MacroMeta` for explicit calls.
        """
        return host_to_stx(self, host, call_stx, entry, marked)


def host_to_stx(ctx: "ExpandContext", host: Any, call_stx: Stx,
                entry: MacroEntry, marked: Stx) -> Stx:
    b = ctx.builder
    if isinstance(host, Stx):
        return host
    if isinstance(host, ListVal):
        raise CTError("macro returned a host list (ListVal); use a syntax "
                      "list or datum->syntax to produce syntax", call_stx)
    if isinstance(host, list):
        items = tuple(host_to_stx(ctx, x, call_stx, entry, marked)
                      for x in host)
        return b.node(kind="seq", lexeme="()", span=call_stx.span,
                      parents=(call_stx.id, marked.id),
                      reason=f"macro-output-list:{entry.name}", seq=items)
    if isinstance(host, dict) and "#raw" in host:
        return b.node(kind="raw", value=host["#raw"], lexeme=host["#raw"],
                      span=None, parents=(call_stx.id,),
                      reason=f"macro-output-raw:{entry.name}",
                      flags=("diagnostic",))
    if isinstance(host, bool):
        return b.node(kind="bool", value=host, lexeme=str(host).lower(),
                      span=None, parents=(call_stx.id,),
                      reason=f"macro-output-literal:{entry.name}")
    if isinstance(host, (int, float)):
        return b.node(kind="number", value=host, lexeme=str(host), span=None,
                      parents=(call_stx.id,),
                      reason=f"macro-output-literal:{entry.name}")
    if host is None:
        return b.node(kind="id", value="#<void>", lexeme="#<void>",
                      span=None, parents=(call_stx.id,),
                      reason=f"macro-output-void:{entry.name}")
    raise CTError(f"macro returned non-syntax value: {host!r}", call_stx)


class MacroMeta(MetaEvaluator):
    """Compile-time evaluator inside one macro invocation."""

    def __init__(self, ctx: ExpandContext, entry: MacroEntry):
        super().__init__(ctx.builder, entry.name)
        self.ctx = ctx
        self.entry = entry
        self.globals.update(self._capture_prims())

    def eval_callable(self, closure: Closure, args: list[Any]) -> Any:
        env = dict(zip(closure.params, args))
        result = None
        for sub in closure.body:
            result = self.eval(sub, env)
        return result

    def _capture_prims(self) -> dict[str, Any]:
        ctx, entry, b = self.ctx, self.entry, self.ctx.builder

        def datum_to_syntax(context: Any, datum: Any) -> Any:
            if not isinstance(context, Stx):
                raise CTError("datum->syntax: first argument must be a "
                              "syntax object (context identifier)")
            context_scopes = frozenset(context.scopes)
            call_frame = ctx.stack[-1] if ctx.stack else {}
            mark = call_frame.get("mark")
            final_scopes = (context_scopes | ({mark} if mark else frozenset()))
            b.log_step(
                "datum->syntax", macro=entry.name,
                context_node=context.id,
                context_symbol=context.value if context.kind == "id" else None,
                context_scopes=sorted(context_scopes),
                applied_mark=mark,
                final_scopes=sorted(final_scopes),
                datum=_datum_preview(datum),
                context_choice="explicit-context-object",
                definition_site=entry.def_site_id)

            def convert(host: Any, depth: int = 0) -> Any:
                if isinstance(host, Stx):
                    # Embedded syntax keeps its own scopes (matched input);
                    # only surrounding host data adopts the context.
                    return host
                if isinstance(host, (list, ListVal)):
                    elems = host.items if isinstance(host, ListVal) else host
                    return b.node(
                        kind="seq", lexeme="()",
                        span=context.span if depth == 0 else None,
                        parents=(context.id,),
                        reason=f"datum->syntax-list:{entry.name}",
                        seq=tuple(convert(x, depth + 1) for x in elems),
                        scopes=final_scopes if depth == 0 else frozenset())
                if isinstance(host, dict) and "#raw" in host:
                    return b.node(kind="raw", value=host["#raw"],
                                  lexeme=host["#raw"], span=None,
                                  parents=(context.id,),
                                  reason=f"datum->syntax-raw:{entry.name}",
                                  scopes=final_scopes, flags=("diagnostic",))
                if isinstance(host, bool):
                    return b.node(kind="bool", value=host,
                                  lexeme=str(host).lower(),
                                  scopes=final_scopes, parents=(context.id,),
                                  reason=f"datum->syntax:{entry.name}")
                if isinstance(host, (int, float)):
                    return b.node(kind="number", value=host, lexeme=str(host),
                                  scopes=final_scopes, parents=(context.id,),
                                  reason=f"datum->syntax:{entry.name}")
                if isinstance(host, str):
                    # A host symbol -> identifier that *explicitly* adopts the
                    # context's scopes. This is the visible capture hatch.
                    captures_local = any(
                        s.startswith(("b:",)) for s in context_scopes)
                    flags = ("introduced",)
                    if captures_local:
                        flags = ("introduced", "explicit-capture")
                    node = b.node(
                        kind="id", value=host, lexeme=host,
                        scopes=final_scopes, span=context.span,
                        parents=(context.id,),
                        reason=f"datum->syntax:{entry.name}",
                        flags=flags)
                    b.log_step(
                        "explicit-capture-introduced" if captures_local
                        else "introduced-identifier",
                        macro=entry.name, node=node.id, symbol=host,
                        inherited_scopes=sorted(context_scopes),
                        context_node=context.id)
                    return node
                raise CTError(f"datum->syntax: cannot convert {host!r}")

            result = convert(datum)
            return result

        def gen_temps(template: Any = None) -> Any:
            n = 1
            if isinstance(template, Stx) and template.kind == "seq":
                n = len(template.seq)
            elif isinstance(template, (list, ListVal)):
                n = (len(template.items) if isinstance(template, ListVal)
                     else len(template))
            out = []
            mark = ctx.stack[-1]["mark"] if ctx.stack else None
            for _ in range(n):
                label = f"$tmp_{entry.name}_{ctx.call_count}_" \
                        f"{len(ctx.generated) + len(out) + 1:02d}"
                intro = b.fresh_intro("tmp")
                node = b.node(
                    kind="id", value=label, lexeme=label,
                    scopes=frozenset({intro, mark} if mark else {intro}),
                    span=None,
                    reason=f"generate-temporaries:{entry.name}",
                    flags=("introduced", "generated-symbol"))
                rec = {"symbol": label, "node": node.id,
                       "macro": entry.name, "intro_scope": intro,
                       "call": ctx.call_count}
                ctx.generated.append(rec)
                b.log_step("generate-temporaries", **rec)
                out.append(node)
            return out

        return {"datum->syntax": datum_to_syntax,
                "generate-temporaries": gen_temps}


def _datum_preview(datum: Any) -> Any:
    if isinstance(datum, Stx):
        return datum.to_plain()
    if isinstance(datum, list):
        return [_datum_preview(x) for x in datum]
    return datum


def expand(root: Stx, config: ExpanderConfig, input_version_id: str,
           builder: GraphBuilder) -> ExpansionResult:
    ctx = ExpandContext(builder, config, input_version_id)
    status = "complete"
    expanded: Optional[Stx] = None
    try:
        expanded = ctx.expand_program(root)
        if any(d.severity == "unknown" for d in ctx.diagnostics):
            status = "complete-with-unknowns"
        if any("budget-exhausted" in n.flags for n in builder.nodes.values()):
            status = "budget-exhausted"
    except CTError as exc:
        status = "failed"
        ctx.diagnostics.append(Diagnostic(
            "E500", f"expansion failed: {exc.args[0]}", "error",
            exc.node.span if exc.node and exc.node.span else None,
            exc.node.id if exc.node else None))
        builder.log_step("expansion-failed", message=exc.args[0])
    return ExpansionResult(
        root=expanded if expanded is not None else ctx.partial_root,
        status=status, bindings=ctx.bindings,
        resolutions=ctx.resolutions,
        macro_stack=ctx.stack, diagnostics=ctx.diagnostics,
        call_count=ctx.call_count, budget=config.macro_budget,
        generated_symbols=ctx.generated)
