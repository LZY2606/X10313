"""scope-set 卫生展开器 + 绑定解释器。

卫生模型（Dybvig/Friedman/Haynes/Hieb "syntactic closures" 风格的
cancellation mark，等价于 Flatt set-of-scopes 的 toggle）：

* 进入一次宏调用，先对**整个输入** toggle 一个使用处标记 scope U；
* 宏模板里的字面标识符携带**定义处引入 scope** M；
* 展开结果再 toggle U：宏引入的 M 保留（U^U 抵消），
  用户代码上的 M 被抵消（M^M）。于是：
    - 宏引入的 `it` 只可能与宏内部/datum->syntax 的绑定冲突；
    - 用户代码的 `it` 不会与宏内部临时名冲突；
  捕获由 scope 关系判定，而不是字符串改名。

绑定解释：在所有候选 binder 中按 scope 交集大小（specificity，NumPy 向量化
统计）剪枝；无法唯一决定时保留全部并列候选（ambiguous）；无候选即 unbound。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .interp import (Bool as TBool, Closure, Env, Interp, Num as TNum,
                     TransformError, _raw)
from .model import (BASIS_SCOPE_DEPTH, UNIT_BUDGET, Bool, IdGen,
                    Num, Span, Stx, Sym)
from .reader import SCOPE_CORE, SCOPE_TOP

CORE_FORMS = {"lambda", "let", "letrec", "if", "begin", "quote",
              "set!", "define", "define-syntax"}
CORE_PRIMS = {"+", "-", "*", "=", "<", ">", "<=", ">=", "cons", "car", "cdr",
              "list", "null?", "pair?", "not", "display", "void"}


class ExpandError(Exception):
    def __init__(self, message: str, stx: Optional[Stx] = None,
                 call_chain: Optional[list[str]] = None,
                 kind: str = "expand_error"):
        super().__init__(message)
        self.message = message
        self.stx = stx
        self.call_chain = call_chain or []
        self.kind = kind


@dataclass
class Binder:
    kind: str            # core | local | top | macro
    name: Optional[str]
    scope_set: frozenset[int]
    site_sid: str        # define/lambda/let 节点 sid
    order: int = 0
    gensym_key: Optional[str] = None
    depth: int = 0

    def identity_key(self) -> tuple:
        return (self.kind, tuple(sorted(self.scope_set)), self.gensym_key,
                self.site_sid)


@dataclass
class BindingEdge:
    occ_sid: str
    binder: Binder
    status: str          # unique | ambiguous | unbound | core | macro
    specificity: int
    candidates: tuple[Binder, ...]
    rejected: tuple[tuple[str, Binder, int], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "occ_sid": self.occ_sid,
            "status": self.status,
            "binder": _binder_json(self.binder) if self.binder else None,
            "specificity": self.specificity,
            "specificity_basis": "|use_scopes ∩ binder_scope_superset|",
            "candidates": [_binder_json(b) for b in self.candidates],
            "rejected": [
                {"reason": r, "binder": _binder_json(b), "overlap": ov}
                for r, b, ov in self.rejected],
        }


def _binder_json(b: Binder) -> dict[str, Any]:
    return {"kind": b.kind, "name": b.name,
            "scope_set": sorted(b.scope_set),
            "site_sid": b.site_sid, "order": b.order,
            "gensym_key": b.gensym_key,
            "depth": b.depth, "depth_basis": BASIS_SCOPE_DEPTH}


@dataclass
class MacroCall:
    call_id: str
    macro_name: str
    macro_sid: str
    input_sid: str
    output_sid: Optional[str]
    parent_call_id: Optional[str]
    use_scope: int
    intro_scope: int
    site_sid: str
    status: str          # ok | error | budget_exceeded
    error: Optional[str]
    depth: int


@dataclass
class Step:
    step_id: str
    call_id: Optional[str]
    kind: str
    detail: dict[str, Any]
    seq: int


@dataclass
class ScopeInfo:
    label: int
    kind: str           # core | top | use_mark | intro | local | gensym
    call_id: Optional[str]
    depth: int


@dataclass
class Trace:
    calls: dict[str, MacroCall] = field(default_factory=dict)
    call_order: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    scopes: dict[int, ScopeInfo] = field(default_factory=dict)
    edges: dict[str, BindingEdge] = field(default_factory=dict)
    binders: list[Binder] = field(default_factory=list)
    macro_defs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    expansions: dict[str, Stx] = field(default_factory=dict)
    status: str = "ok"
    budget_used: int = 0
    seq_counter: int = 0

    def next_seq(self) -> int:
        self.seq_counter += 1
        return self.seq_counter

    def add_step(self, call_id: Optional[str], kind: str,
                 detail: dict[str, Any]) -> Step:
        st = Step(step_id=f"step-{self.seq_counter+1:04d}", call_id=call_id,
                  kind=kind, detail=detail, seq=self.next_seq())
        self.steps.append(st)
        return st


@dataclass
class MacroDef:
    name: str
    closure: Closure
    intro_scope: int
    def_site_sid: str


@dataclass
class ExpandResult:
    program: list[Stx]
    trace: Trace


class Expander:
    def __init__(self, idgen: IdGen, *, budget: int = 64, seed: str = "scopelens"):
        self.idgen = idgen
        self.budget = budget
        self.trace = Trace()
        self.interp = Interp(idgen, self._emit_step)
        self.call_counter = 0
        self.macro_counter = 0
        self.binder_seq = 0
        self.primitives = self._build_primitives()
        self.trace.scopes[SCOPE_CORE] = ScopeInfo(SCOPE_CORE, "core", None, 0)
        self.trace.scopes[SCOPE_TOP] = ScopeInfo(SCOPE_TOP, "top", None, 0)
        self._core_binders: dict[str, Binder] = {}
        for name in sorted(CORE_FORMS | CORE_PRIMS):
            b = Binder(kind="core", name=name,
                       scope_set=frozenset({SCOPE_CORE}),
                       site_sid=f"core:{name}", order=0, depth=0)
            self._core_binders[name] = b
            self.trace.binders.append(b)

    # ------------------------------------------------------------------
    def _emit_step(self, kind: str, detail: dict[str, Any]) -> None:
        self.trace.add_step(detail.get("_call_id"), kind,
                            {k: v for k, v in detail.items() if k != "_call_id"})

    def _fresh_scope(self, kind: str, call_id: Optional[str], depth: int) -> int:
        label = self.idgen.fresh_scope()
        self.trace.scopes[label] = ScopeInfo(label, kind, call_id, depth)
        return label

    def _add_binder(self, kind: str, name: Optional[str],
                    scope_set: frozenset[int], site_sid: str, *,
                    gensym_key: Optional[str] = None,
                    depth: int = 0) -> Binder:
        self.binder_seq += 1
        b = Binder(kind=kind, name=name, scope_set=frozenset(scope_set),
                   site_sid=site_sid, order=self.binder_seq,
                   gensym_key=gensym_key, depth=depth)
        self.trace.binders.append(b)
        return b

    # ------------------------------------------------------------------
    def expand_program(self, forms: list[Stx]) -> ExpandResult:
        top_scope = self._fresh_scope("local", None, 0)
        top_env = Env(dict(self.primitives))
        macros: dict[str, MacroDef] = {}
        # 两遍：先收集顶层 define / define-syntax（允许前向引用与递归宏）
        pre: list[tuple[str, Stx, Optional[Stx]]] = []
        for f in forms:
            h = f.head() if f.is_list else None
            if h == "define":
                pre.append(("define", f, f.items[1]))
            elif h == "define-syntax":
                pre.append(("define-syntax", f, f.items[1]))
        for kind, form, name_stx in pre:
            if not name_stx.is_sym():
                self._error("define 的名字必须是标识符", name_stx, [], "bad_syntax")
                continue
            if kind == "define":
                self._add_binder("top", name_stx.sym_name,
                                 frozenset({SCOPE_CORE, SCOPE_TOP, top_scope}),
                                 form.sid)
            else:
                self._define_macro(name_stx, form.items[2], top_env,
                                   top_scope, macros, form)
        out: list[Stx] = []
        for f in forms:
            out.append(self.expand(f, top_scope, top_env, macros, [], top_scope))
        self.trace.status = "ok" if not self.trace.errors else (
            "budget_exceeded" if any(e["kind"] == "budget_exceeded"
                                     for e in self.trace.errors) else "error")
        return ExpandResult(out, self.trace)

    def _define_macro(self, name_stx: Stx, body: Stx, env: Env,
                      owner_scope: int, macros: dict[str, MacroDef],
                      site: Stx) -> None:
        intro = self._fresh_scope("intro", None, 0)
        self.trace.add_step(None, "macro_defined", {
            "name": name_stx.sym_name,
            "def_site_sid": site.sid,
            "def_span": site.span.to_json() if site.span else None,
            "intro_scope": intro,
            "scope_basis": BASIS_SCOPE_DEPTH,
        })
        self.trace.macro_defs.append({
            "name": name_stx.sym_name, "site_sid": site.sid,
            "name_sid": name_stx.sid,
            "def_span": site.span.to_json() if site.span else None,
            "intro_scope": intro})
        try:
            closure = self.interp.eval(body, env, intro)
        except TransformError as exc:
            self._error(f"宏定义求值失败：{exc.message}", exc.stx or site, [],
                        "bad_transformer")
            return
        if not isinstance(closure, Closure):
            self._error("define-syntax 的体必须求值为 lambda", site, [],
                        "bad_transformer")
            return
        macro = MacroDef(name_stx.sym_name, closure, intro, site.sid)
        macros[name_stx.sym_name] = macro
        self._add_binder("macro", name_stx.sym_name,
                         frozenset({SCOPE_CORE, SCOPE_TOP, owner_scope}),
                         site.sid)

    # ------------------------------------------------------------------
    def expand(self, stx: Stx, local_scope: int, env: Env,
               macros: dict[str, MacroDef], call_stack: list[str],
               body_scope: int) -> Stx:
        if not stx.is_list:
            self._resolve_atom(stx, macros)
            return stx
        h = stx.head()
        if h == "define":
            return self._expand_define(stx, local_scope, env, macros,
                                       call_stack, body_scope)
        if h == "define-syntax":
            return stx  # 定义已在预处理建立；展开期保留为记录节点
        if h == "lambda":
            return self._expand_lambda(stx, env, macros, call_stack)
        if h in ("let", "letrec"):
            return self._expand_let(stx, env, macros, call_stack, recursive=(h == "letrec"))
        if h == "if":
            return self._expand_each(stx, env, macros, call_stack, body_scope,
                                     skip={0})
        if h == "begin":
            return self._expand_each(stx, env, macros, call_stack, body_scope)
        if h == "set!":
            return self._expand_set(stx, env, macros, call_stack, body_scope)
        if h == "quote":
            return stx
        if h is not None and h in macros:
            return self._expand_macro_call(stx, macros[h], env, macros,
                                           call_stack, local_scope, body_scope)
        # 应用表达式
        return self._expand_each(stx, env, macros, call_stack, body_scope)

    def _expand_each(self, stx: Stx, env: Env, macros: dict[str, MacroDef],
                     call_stack: list[str], body_scope: int,
                     skip: Optional[set[int]] = None) -> Stx:
        skip = skip or set()
        items = []
        for i, it in enumerate(stx.items):
            if i in skip:
                items.append(it)
                if it.is_list:
                    self._resolve_head(it, macros)
            else:
                items.append(self.expand(it, body_scope, env, macros,
                                         call_stack, body_scope))
        return self.idgen.clone(stx, items=tuple(items)) if False else \
            self._rebuild(stx, items)

    def _rebuild(self, stx: Stx, items: list[Stx]) -> Stx:
        return self.idgen.make(atom=None, items=tuple(items), span=stx.span,
                               scopes=stx.scopes, origin=stx.origin,
                               ctx_sid=stx.ctx_sid, gensym_key=stx.gensym_key,
                               synthesized=stx.synthesized, step_id=stx.step_id,
                               basis=stx.basis)

    def _expand_lambda(self, stx: Stx, env: Env,
                       macros: dict[str, MacroDef],
                       call_stack: list[str]) -> Stx:
        param_names = stx.items[1]
        scope = self._fresh_scope("local", None, len(call_stack))
        local_env = Env({}, env)
        bind_items: list[Stx] = []
        for p in param_names.items:
            if not p.is_sym():
                self._error("lambda 参数必须是标识符", p, call_stack, "bad_syntax")
                bind_items.append(p)
                continue
            self._add_binder(
                "local", p.sym_name,
                p.scopes | {scope} if False else frozenset({SCOPE_CORE, SCOPE_TOP, scope}),
                stx.sid, gensym_key=p.gensym_key,
                depth=len(call_stack) + 1)
            bind_items.append(self.idgen.clone(p, scopes=p.scopes | {scope}))
        bodies = [self.expand(self._annotate(b, scope), scope, local_env,
                              macros, call_stack, scope)
                  for b in stx.items[2:]]
        items = [stx.items[0], self.idgen.list_stx(bind_items, param_names.span,
                                                   param_names.scopes), *bodies]
        return self._rebuild(stx, items)

    def _expand_let(self, stx: Stx, env: Env, macros: dict[str, MacroDef],
                    call_stack: list[str], *, recursive: bool) -> Stx:
        bindings = stx.items[1]
        scope = self._fresh_scope("local", None, len(call_stack))
        local_env = Env({}, env)
        new_bindings: list[Stx] = []
        if recursive:
            for b in bindings.items:
                name = b.items[0]
                local_env.bindings[name.sym_name] = None
                self._add_binder("local", name.sym_name,
                                 frozenset({SCOPE_CORE, SCOPE_TOP, scope}),
                                 b.sid, gensym_key=name.gensym_key,
                                 depth=len(call_stack) + 1)
        for b in bindings.items:
            name = b.items[0]
            if not recursive:
                local_env.bindings[name.sym_name] = None
                self._add_binder("local", name.sym_name,
                                 frozenset({SCOPE_CORE, SCOPE_TOP, scope}),
                                 b.sid, gensym_key=name.gensym_key,
                                 depth=len(call_stack) + 1)
            val = self.expand(self._annotate(b.items[1], scope),
                              scope, local_env if recursive else env,
                              macros, call_stack, scope)
            name2 = self.idgen.clone(name, scopes=name.scopes | {scope})
            new_bindings.append(self._rebuild(b, [name2, val]))
        bodies = [self.expand(self._annotate(x, scope), scope, local_env,
                              macros, call_stack, scope)
                  for x in stx.items[2:]]
        items = [stx.items[0],
                 self.idgen.list_stx(new_bindings, bindings.span, bindings.scopes),
                 *bodies]
        return self._rebuild(stx, items)

    def _expand_define(self, stx: Stx, local_scope: int, env: Env,
                       macros: dict[str, MacroDef], call_stack: list[str],
                       body_scope: int) -> Stx:
        name = stx.items[1]
        val = self.expand(stx.items[2], body_scope, env, macros, call_stack,
                          body_scope)
        return self._rebuild(stx, [stx.items[0], name, val])

    def _expand_set(self, stx: Stx, env: Env, macros: dict[str, MacroDef],
                    call_stack: list[str], body_scope: int) -> Stx:
        name = stx.items[1]
        self._resolve_atom(name, macros)
        val = self.expand(stx.items[2], body_scope, env, macros, call_stack,
                          body_scope)
        return self._rebuild(stx, [stx.items[0], name, val])

    # ------------------------------------------------------------------
    def _expand_macro_call(self, call: Stx, macro: MacroDef, env: Env,
                           macros: dict[str, MacroDef], call_stack: list[str],
                           local_scope: int, body_scope: int) -> Stx:
        self.call_counter += 1
        call_id = f"call-{self.call_counter:04d}"
        if self.trace.budget_used >= self.budget:
            self._error(
                f"递归展开预算耗尽（预算 {self.budget} {UNIT_BUDGET}s）",
                call, call_stack, "budget_exceeded")
            err = self.idgen.list_stx(
                [self.idgen.atom_stx(Sym("%%error%%"), call.span, call.scopes),
                 call], call.span, call.scopes, synthesized=True)
            return err
        self.trace.budget_used += 1
        use_scope = self._fresh_scope("use_mark", call_id, len(call_stack))
        depth = len(call_stack) + 1
        mc = MacroCall(call_id=call_id, macro_name=macro.name,
                       macro_sid=macro.def_site_sid, input_sid=call.sid,
                       output_sid=None,
                       parent_call_id=call_stack[-1] if call_stack else None,
                       use_scope=use_scope, intro_scope=macro.intro_scope,
                       site_sid=call.sid, status="ok", error=None, depth=depth)
        self.trace.calls[call_id] = mc
        self.trace.call_order.append(call_id)
        self.trace.add_step(call_id, "macro_invoked", {
            "macro_name": macro.name, "def_site_sid": macro.def_site_sid,
            "use_site_sid": call.sid,
            "use_span": call.span.to_json() if call.span else None,
            "call_stack": call_stack + [call_id],
            "budget_used": self.trace.budget_used,
            "budget_unit": UNIT_BUDGET,
        })
        marked = self._toggle_scope(call, use_scope, call_id, "mark_input")
        introducer = self.interp.make_introducer(macro.intro_scope)

        def expand_loop(sub: Stx) -> Stx:
            return self.expand(sub, body_scope, env, macros,
                               call_stack + [call_id], body_scope)

        try:
            raw_out = self.interp.apply(
                macro.closure,
                [marked.items if False else marked, introducer],
                call, macro.intro_scope, expand_loop)
        except TransformError as exc:
            mc.status = "error"
            mc.error = exc.message
            self._error(f"宏 {macro.name} 变换失败：{exc.message}",
                        exc.stx or call, call_stack + [call_id],
                        "transform_failed")
            err = self.idgen.list_stx(
                [self.idgen.atom_stx(Sym("%%error%%"), call.span, call.scopes),
                 call], call.span, call.scopes, synthesized=True,
                step_id=self.trace.steps[-1].step_id)
            mc.output_sid = err.sid
            return err
        if not isinstance(raw_out, Stx):
            mc.status = "error"
            mc.error = "宏必须返回 syntax object（用 (syntax ...) 或 datum->syntax）"
            self._error(mc.error, call, call_stack + [call_id],
                        "bad_transformer")
            err = self.idgen.list_stx(
                [self.idgen.atom_stx(Sym("%%error%%"), call.span, call.scopes)],
                call.span, call.scopes, synthesized=True)
            mc.output_sid = err.sid
            return err
        unmarked = self._toggle_scope(raw_out, use_scope, call_id,
                                      "unmark_output")
        mc.output_sid = unmarked.sid
        self.trace.add_step(call_id, "macro_returned", {
            "input_sid": call.sid, "marked_sid": marked.sid,
            "raw_output_sid": raw_out.sid, "output_sid": unmarked.sid,
            "call_stack": call_stack + [call_id],
        })
        # 结果重新进入展开（嵌套宏、递归宏在此发生）
        reexpanded = self.expand(unmarked, body_scope, env, macros,
                                 call_stack + [call_id], body_scope)
        self.trace.expansions[call_id] = reexpanded
        return reexpanded

    def _toggle_scope(self, stx: Stx, label: int, call_id: str,
                      phase: str) -> Stx:
        def walk(s: Stx) -> Stx:
            sc = s.scopes ^ {label}
            if s.is_list:
                return self.idgen.list_stx(
                    [walk(i) for i in s.items], s.span, sc,
                    synthesized=s.synthesized, origin=s.sid,
                    ctx_sid=s.ctx_sid, step_id=None)
            return self.idgen.clone(s, scopes=sc, origin=s.sid)

        out = walk(stx)
        self.trace.add_step(call_id, "scope_toggle", {
            "phase": phase, "scope_label": label,
            "scope_kind": self.trace.scopes[label].kind,
            "input_sid": stx.sid, "output_sid": out.sid,
            "scope_basis": BASIS_SCOPE_DEPTH,
        })
        return out

    # ------------------------------------------------------------------
    def _annotate(self, stx: Stx, label: int) -> Stx:
        """给进入新局部作用域的整段代码加注 scope（binder 与引用成对加注）。"""
        if stx.is_list:
            return self.idgen.list_stx(
                [self._annotate(i, label) for i in stx.items], stx.span,
                stx.scopes | {label}, synthesized=stx.synthesized,
                origin=stx.origin, ctx_sid=stx.ctx_sid, step_id=stx.step_id)
        return self.idgen.clone(stx, scopes=stx.scopes | {label})

    def _resolve_head(self, stx: Stx, macros: dict[str, MacroDef]) -> None:
        if stx.is_list and stx.items:
            self._resolve_atom(stx.items[0], macros)

    def _resolve_atom(self, occ: Stx, macros: dict[str, MacroDef]) -> None:
        if not occ.is_sym():
            return
        use = occ.scopes
        scored: list[tuple[int, int, int, Binder]] = []
        rejected: list[tuple[str, Binder, int]] = []
        for b in self.trace.binders:
            if b.gensym_key != occ.gensym_key:
                rejected.append(("gensym_key 不匹配（生成符号按身份而非名字）", b, 0))
                continue
            if b.name != occ.sym_name:
                rejected.append(("名字不同（捕获由 scope 关系判定，名字仅用于索引）",
                                 b, 0))
                continue
            overlap = len(use & b.scope_set)
            if not b.scope_set <= use:
                rejected.append(("binder scope 集合不是使用处 scope 的子集", b,
                                 overlap))
                continue
            scored.append((overlap, b.depth, b.order, b))
        if not scored:
            edge = BindingEdge(occ.sid, None, "unbound", 0, (), tuple(rejected))
            self.trace.edges[occ.sid] = edge
            self._error(f"未绑定的标识符 {occ.sym_name!r}", occ, [],
                        "unbound", record=False)
            edge_err = self.trace.errors[-1]
            edge_err["occ_sid"] = occ.sid
            return
        spec = max(x[0] for x in scored)
        best = [x for x in scored if x[0] == spec]
        # NumPy 向量化 specificity：交集计数矩阵（可审计、确定性）
        matrix = self._specificity_matrix(use)
        if len(best) == 1:
            _, depth, order, b = best[0]
            status = "core" if b.kind == "core" else (
                "macro" if b.kind == "macro" else "unique")
            rej = tuple(rejected) + tuple(
                ("specificity 较小", x[3], x[0]) for x in scored if x[0] < spec)
            self.trace.edges[occ.sid] = BindingEdge(
                occ.sid, b, status, spec, (b,), rej)
        else:
            cands = tuple(x[3] for x in best)
            # 同 specificity 且无法以词法深度区分 => 全部合法候选保留
            depths = {x[1] for x in best}
            if len(depths) == 1:
                rej = tuple(rejected) + tuple(
                    ("specificity 较小", x[3], x[0])
                    for x in scored if x[0] < spec)
                self.trace.edges[occ.sid] = BindingEdge(
                    occ.sid, None, "ambiguous", spec, cands, rej)
                self._error(
                    f"标识符 {occ.sym_name!r} 有 {len(cands)} 个并列合法绑定，"
                    "无法唯一决定", occ, [], "ambiguous", record=False)
            else:
                deepest = max(depths)
                win = [x for x in best if x[1] == deepest]
                if len(win) == 1:
                    _, _, _, b = win[0]
                    self.trace.edges[occ.sid] = BindingEdge(
                        occ.sid, b,
                        "core" if b.kind == "core" else
                        ("macro" if b.kind == "macro" else "unique"),
                        spec, (b,),
                        tuple(rejected) + tuple(
                            ("specificity 相同但词法深度较浅", x[3], spec)
                            for x in best if x is not win[0]))
                else:
                    cands2 = tuple(x[3] for x in win)
                    self.trace.edges[occ.sid] = BindingEdge(
                        occ.sid, None, "ambiguous", spec, cands2,
                        tuple(rejected))
                    self._error(
                        f"标识符 {occ.sym_name!r} 有 {len(cands2)} 个并列合法绑定",
                        occ, [], "ambiguous", record=False)

    def _specificity_matrix(self, use: frozenset[int]) -> dict[str, Any]:
        labels = sorted(self.trace.scopes)
        if not labels:
            return {"labels": [], "matrix": []}
        arr = np.zeros((len(self.trace.binders), len(labels)), dtype=np.int8)
        for i, b in enumerate(self.trace.binders):
            for j, lab in enumerate(labels):
                if lab in b.scope_set and lab in use:
                    arr[i, j] = 1
        return {"labels": labels,
                "overlap": [int(x) for x in arr.sum(axis=1)],
                "algo": "numpy.int8 binder×scope intersection counts"}

    def _error(self, message: str, stx: Optional[Stx],
               call_chain: list[str], kind: str, *, record: bool = True) -> None:
        rec = {
            "kind": kind, "message": message,
            "span": stx.span.to_json() if stx and stx.span else None,
            "stx_sid": stx.sid if stx else None,
            "call_stack": list(call_chain),
        }
        self.trace.errors.append(rec)
        self.trace.add_step(call_chain[-1] if call_chain else None,
                            "error_recorded", rec)

    # ------------------------------------------------------------------
    # 变换语言原语
    # ------------------------------------------------------------------
    def _build_primitives(self) -> dict[str, Any]:
        idgen = self.idgen

        def stx_is_list(v, *_a):
            return isinstance(v, Stx) and v.is_list

        def stx_is_sym(v, *_a):
            return isinstance(v, Stx) and v.is_sym()

        def stx_is_num(v, *_a):
            return isinstance(v, Stx) and isinstance(v.atom, Num)

        def stx_e(v, *_a):
            s = v[0]
            if not isinstance(s, Stx):
                raise TransformError("syntax-e 需要 syntax 对象")
            return list(s.items) if s.is_list else s.atom

        def stx_car(v, *_a):
            s = v[0]
            if not (isinstance(s, Stx) and s.is_list and s.items):
                raise TransformError("stx-car 需要非空 list syntax", s)
            return s.items[0]

        def stx_cdr(v, *_a):
            s = v[0]
            if not (isinstance(s, Stx) and s.is_list):
                raise TransformError("stx-cdr 需要 list syntax", s)
            return idgen.list_stx(list(s.items[1:]), s.span, s.scopes,
                                  synthesized=True, origin=s.sid)

        def stx_null(v, *_a):
            s = v[0]
            return isinstance(s, Stx) and s.is_list and not s.items

        def stx_cons(v, call, interp, macro_scope, loop):
            head, tail = v
            if not isinstance(tail, Stx) or not tail.is_list:
                raise TransformError("stx-cons 的第二个参数必须是 list syntax",
                                     call)
            return idgen.list_stx([head, *tail.items], tail.span, tail.scopes,
                                  synthesized=True, origin=tail.sid)

        def stx_list(v, call, *_a):
            if not all(isinstance(x, Stx) for x in v):
                raise TransformError("stx-list 的参数必须全部是 syntax", call)
            scopes: frozenset[int] = frozenset()
            for x in v:
                scopes |= x.scopes
            return idgen.list_stx(list(v), None, scopes, synthesized=True)

        def datum_to_syntax(v, call, interp, macro_scope, loop):
            """唯一允许把裸 datum 提升为 syntax 的入口：必须显式选择上下文。"""
            datum, ctx = v
            if not isinstance(ctx, Stx):
                raise TransformError("datum->syntax 第二个参数必须是提供上下文的"
                                     " syntax 对象", call)
            out = _lift_datum(idgen, datum, ctx, call)
            self.trace.add_step(None, "datum_to_syntax", {
                "ctx_sid": ctx.sid, "ctx_scopes": sorted(ctx.scopes),
                "output_sid": out.sid,
                "rule": "复制上下文标识符的 scope 集合（可见的上下文选择）",
            })
            return out

        def gensym_prim(v, call, *_a):
            base = v[0]
            name = base.name if isinstance(base, Sym) else str(_raw(base))
            key = idgen.gensym(name)
            return idgen.make(atom=Sym(name + "$" + key.split("#")[-1]),
                              span=None, scopes=frozenset(), gensym_key=key,
                              synthesized=True)

        def stx_eq(v, *_a):
            a, b = v
            if isinstance(a, Stx) and isinstance(b, Stx):
                ka = a.gensym_key or a.sid
                kb = b.gensym_key or b.sid
                return ka == kb
            return a is b

        def sym_eq(v, *_a):
            a, b = v
            na = a.name if isinstance(a, Sym) else (
                a.sym_name if isinstance(a, Stx) and a.is_sym() else None)
            nb = b.name if isinstance(b, Sym) else (
                b.sym_name if isinstance(b, Stx) and b.is_sym() else None)
            return na == nb

        def stx_nat(v, *_a):
            s = v[0]
            if isinstance(s, Stx) and isinstance(s.atom, Num):
                return s.atom
            if isinstance(s, TNum):
                return s
            raise TransformError("stx-nat 需要数字 syntax")

        def num_stx(v, call, *_a):
            n = v[0]
            if isinstance(n, TNum):
                return idgen.make(atom=n, span=None, scopes=frozenset(),
                                  synthesized=True)
            return idgen.make(atom=Num(int(n)), span=None,
                              scopes=frozenset(), synthesized=True)

        def mk_arith(fn):
            def p(v, call, *_a):
                nums = [x.value if isinstance(x, TNum) else
                        x.atom.value if isinstance(x, Stx) and isinstance(x.atom, Num)
                        else None for x in v]
                if any(x is None for x in nums):
                    raise TransformError("算术原语需要数字", call)
                return TNum(fn(nums))
            return p

        def mk_cmp(fn):
            def p(v, call, *_a):
                nums = [x.value if isinstance(x, TNum) else x.atom.value
                        for x in v]
                return fn(nums[0], nums[1])
            return p

        return {
            "stx-list?": stx_is_list, "stx-symbol?": stx_is_sym,
            "stx-nat?": stx_is_num, "syntax-e": stx_e, "stx-car": stx_car,
            "stx-cdr": stx_cdr, "stx-null?": stx_null, "stx-cons": stx_cons,
            "stx-list": stx_list, "datum->syntax": datum_to_syntax,
            "gensym": gensym_prim, "stx-eq?": stx_eq, "sym-eq?": sym_eq,
            "stx-nat": stx_nat, "num->stx": num_stx,
            "+": mk_arith(sum), "-": mk_arith(lambda xs: xs[0] - xs[1]),
            "*": mk_arith(lambda xs: xs[0] * xs[1]),
            "=": mk_cmp(lambda a, b: a == b), "<": mk_cmp(lambda a, b: a < b),
            ">": mk_cmp(lambda a, b: a > b), "sub1": mk_arith(lambda xs: xs[0] - 1),
            "zero?": lambda v, *_a: (v[0].value if isinstance(v[0], TNum)
                                     else v[0].atom.value) == 0,
        }


def _lift_datum(idgen: IdGen, datum: Any, ctx: Stx, site: Stx) -> Stx:
    sc = ctx.scopes
    if isinstance(datum, Stx):
        return idgen.clone(datum, scopes=sc, ctx_sid=ctx.sid, synthesized=True)
    if isinstance(datum, list):
        return idgen.list_stx([_lift_datum(idgen, d, ctx, site) for d in datum],
                              site.span, sc, synthesized=True, ctx_sid=ctx.sid)
    if isinstance(datum, (Sym, Num, Bool)):
        return idgen.make(atom=datum, span=None, scopes=sc, synthesized=True,
                          ctx_sid=ctx.sid)
    if isinstance(datum, (TNum, TBool)):
        return idgen.make(atom=Num(datum.value) if isinstance(datum, TNum)
                          else Bool(datum.value), span=None, scopes=sc,
                          synthesized=True, ctx_sid=ctx.sid)
    raise TransformError(f"datum->syntax 无法提升的值：{datum!r}", site)
