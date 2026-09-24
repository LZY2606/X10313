"""宏变换函数的小型解释器（explicit-renaming 风格）。

变换函数是用本语言写出的闭包，调用时收到：
  - use form（syntax object）
  - introducer：toggle 一个“宏引入 scope”的函数（卫生标记）
  - expand-loop：递归展开回调（供 transformer 主动触发嵌套展开时使用）

上下文选择原语全部可见：
  datum->syntax  : 选择目标 syntax 的 scope 上下文（显式捕获的唯一入口）
  syntax-e       : syntax -> datum（剥去 wrapper）
  gensym         : 生成带稳定 key 的临时标识符
特殊形式 (syntax <tmpl>)：模板标识符自动携带当前“宏引入 scope”；
模板中与局部模式变量同名的标识符引用该变量（模式插值）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .model import IdGen, Num, Span, Stx, Sym


class TransformError(Exception):
    def __init__(self, message: str, stx: Optional[Stx] = None):
        super().__init__(message)
        self.message = message
        self.stx = stx


@dataclass
class Closure:
    params: tuple[str, ...]
    body: list[Stx]
    env: "Env"
    name: str = "<lambda>"


@dataclass
class Env:
    bindings: dict[str, Any]
    parent: Optional["Env"] = None

    def lookup(self, name: str) -> Any:
        env: Optional[Env] = self
        while env is not None:
            if name in env.bindings:
                return env.bindings[name]
            env = env.parent
        raise TransformError(f"变换函数引用了未绑定的名字 {name!r}")

    def has(self, name: str) -> bool:
        env: Optional[Env] = self
        while env is not None:
            if name in env.bindings:
                return True
            env = env.parent
        return False


def _is_truthy(v: Any) -> bool:
    return v is not False and v is not None


def _syntax_lit(stx: Stx, scope_label: int, idgen: IdGen) -> Stx:
    if stx.is_list:
        new_items = [_syntax_lit(i, scope_label, idgen) for i in stx.items]
        return idgen.list_stx(new_items, stx.span, stx.scopes ^ {scope_label},
                              synthesized=True, origin=stx.sid)
    return idgen.clone(stx, scopes=stx.scopes ^ {scope_label}, synthesized=True,
                       origin=stx.sid)


class Interp:
    def __init__(self, idgen: IdGen, emit_step: Callable[..., None]):
        self.idgen = idgen
        self.emit_step = emit_step

    def make_introducer(self, label: int) -> Callable[[Stx], Stx]:
        def introduce(stx: Stx) -> Stx:
            out = _syntax_lit(stx, label, self.idgen)
            self.emit_step(kind="macro_introduction_toggle", detail={
                "scope_label": label,
                "basis": "scope-introduction-depth",
                "input_sid": stx.sid, "output_sid": out.sid,
                "rule": "symmetric-difference (cancellation mark)",
            })
            return out
        return introduce

    # ------------------------------------------------------------------
    def eval_tmpl(self, tmpl: Stx, env: Env, intro_scope: int) -> Stx:
        """(syntax ...) 模板编译：局部变量插值，字面量加宏引入 scope。"""
        if tmpl.is_sym():
            name = tmpl.sym_name
            if env.has(name):  # 模式变量插值
                val = env.lookup(name)
                if not isinstance(val, Stx):
                    raise TransformError(f"模板变量 {name!r} 的值不是 syntax", tmpl)
                return val
            return _syntax_lit(tmpl, intro_scope, self.idgen)
        if tmpl.is_list:
            items = [self.eval_tmpl(i, env, intro_scope) for i in tmpl.items]
            return self.idgen.list_stx(items, tmpl.span,
                                       tmpl.scopes ^ {intro_scope},
                                       synthesized=True, origin=tmpl.sid)
        return tmpl

    def eval(self, e: Stx, env: Env, macro_scope: int,
             expand_loop: Optional[Callable[[Stx], Stx]] = None) -> Any:
        atom = e.atom
        if isinstance(atom, (Num, Bool)):
            return atom
        if e.is_sym():
            return env.lookup(e.sym_name)
        if not e.is_list or not e.items:
            raise TransformError("非法的变换表达式", e)

        head = e.items[0]
        args = e.items[1:]
        h = head.sym_name if head.is_sym() else None

        if h == "quote":
            return args[0]  # 裸 datum/syntax 原样返回
        if h == "syntax":
            return self.eval_tmpl(args[0], env, macro_scope)
        if h == "if":
            test = self.eval(args[0], env, macro_scope, expand_loop)
            branch = args[1] if _is_truthy(_raw(test)) else args[2]
            return self.eval(branch, env, macro_scope, expand_loop)
        if h == "begin":
            result: Any = False
            for a in args:
                result = self.eval(a, env, macro_scope, expand_loop)
            return result
        if h == "lambda":
            params = tuple(_param_name(p) for p in args[0].items)
            return Closure(params, list(args[1:]), env)
        if h == "let":
            return self._eval_let(args, env, macro_scope, expand_loop, recursive=False)
        if h == "letrec":
            return self._eval_let(args, env, macro_scope, expand_loop, recursive=True)
        if h == "and":
            for a in args:
                v = _raw(self.eval(a, env, macro_scope, expand_loop))
                if not _is_truthy(v):
                    return False
            return True
        if h == "or":
            for a in args:
                v = _raw(self.eval(a, env, macro_scope, expand_loop))
                if _is_truthy(v):
                    return v
            return False

        proc = self.eval(head, env, macro_scope, expand_loop)
        vals = [self.eval(a, env, macro_scope, expand_loop) for a in args]
        return self.apply(proc, vals, e, macro_scope, expand_loop)

    def _eval_let(self, args: list[Stx], env: Env, macro_scope: int,
                  expand_loop: Optional[Callable[[Stx], Stx]], *,
                  recursive: bool) -> Any:
        bindings = args[0].items
        body = args[1:]
        new_env = Env({}, env)
        if recursive:
            for b in bindings:
                name = _param_name(b.items[0])
                new_env.bindings[name] = None
        for b in bindings:
            name = _param_name(b.items[0])
            val = self.eval(b.items[1], new_env if recursive else env,
                            macro_scope, expand_loop)
            new_env.bindings[name] = val
        result: Any = False
        for a in body:
            result = self.eval(a, new_env, macro_scope, expand_loop)
        return result

    def apply(self, proc: Any, vals: list[Any], call_stx: Stx,
              macro_scope: int, expand_loop: Optional[Callable[[Stx], Stx]]) -> Any:
        if isinstance(proc, Closure):
            if len(vals) != len(proc.params):
                raise TransformError(
                    f"参数个数不符：{proc.name} 需要 {len(proc.params)} 个，"
                    f"实际 {len(vals)} 个", call_stx)
            local = dict(zip(proc.params, vals))
            result: Any = False
            for a in proc.body:
                result = self.eval(a, Env(local, proc.env), macro_scope, expand_loop)
            return result
        if callable(proc):
            return proc(vals, call_stx, self, macro_scope, expand_loop)
        raise TransformError(f"不可应用的值：{proc!r}", call_stx)


def _param_name(stx: Stx) -> str:
    if not stx.is_sym():
        raise TransformError("绑定位置必须是标识符", stx)
    return stx.sym_name  # type: ignore[return-value]


def _raw(v: Any) -> Any:
    """Num/Bool 在变换语言里即真值本体；Stx 包数字仅用于输出构造。"""
    if isinstance(v, (Num, Bool)):
        return v.value
    return v
