"""展开结果的小型求值器：用于验收“绑定被解释正确”（运行时佐证）。

未绑定/歧义/错误节点以运行时异常显式呈现，绝不静默。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Bool, Num, Stx, Sym
from .expander import BindingEdge


class EvalError(Exception):
    pass


@dataclass
class Cell:
    val: Any


def _truthy(v: Any) -> bool:
    return v is not False and v is not None


def evaluate(program: list[Stx], edges: dict[str, BindingEdge]) -> Any:
    top: dict[str, Cell] = {}

    def binding_of(stx: Stx) -> BindingEdge:
        e = edges.get(stx.sid)
        if e is None:
            raise EvalError(f"标识符 {stx.sym_name!r} 没有绑定解释记录")
        return e

    def eval_expr(e: Stx, env: dict[str, Cell]) -> Any:
        if isinstance(e.atom, Num):
            return e.atom.value
        if isinstance(e.atom, Bool):
            return e.atom.value
        if e.is_sym():
            edge = binding_of(e)
            if edge.status in ("unique", "top", "local"):
                cell = env.get(edge.binder.identity_key())
                if cell is None:
                    cell = top.get(edge.binder.identity_key())
                if cell is None:
                    raise EvalError(f"绑定未初始化：{e.sym_name}")
                return cell.val
            if edge.status == "unbound":
                raise EvalError(f"运行到未绑定标识符：{e.sym_name}")
            if edge.status == "ambiguous":
                raise EvalError(f"运行到歧义标识符：{e.sym_name}")
            raise EvalError(f"核心/宏名字不能当值：{e.sym_name}")
        if not e.is_list or not e.items:
            raise EvalError("非法表达式")
        h = e.head()
        if h == "quote":
            return e.items[1]
        if h == "lambda":
            params = tuple(p.sym_name for p in e.items[1].items)
            keys = tuple(binding_of(p).binder.identity_key()
                         for p in e.items[1].items)
            body = e.items[2:]

            def fn(args: list[Any]) -> Any:
                local = dict(env)
                for k, a in zip(keys, args):
                    local[k] = Cell(a)
                res: Any = None
                for b in body:
                    res = eval_expr(b, local)
                return res
            return fn
        if h in ("let", "letrec"):
            rec = h == "letrec"
            local = dict(env)
            cells = []
            for b in e.items[1].items:
                k = binding_of(b.items[0]).binder.identity_key()
                c = Cell(None)
                local[k] = c
                cells.append((c, b.items[1], local if rec else env))
            if not rec:
                for c, val, ve in cells:
                    c.val = eval_expr(val, ve)
            else:
                for c, val, ve in cells:
                    c.val = eval_expr(val, ve)
            res: Any = None
            for b in e.items[2:]:
                res = eval_expr(b, local)
            return res
        if h == "if":
            return eval_expr(e.items[2] if _truthy(eval_expr(e.items[1], env))
                             else e.items[3], env)
        if h == "begin":
            res = None
            for b in e.items[1:]:
                res = eval_expr(b, env)
            return res
        if h == "set!":
            edge = binding_of(e.items[1])
            k = edge.binder.identity_key()
            c = env.get(k) or top.get(k)
            if c is None:
                raise EvalError(f"set! 目标未绑定：{e.items[1].sym_name}")
            c.val = eval_expr(e.items[2], env)
            return None
        # 核心原语应用
        proc = eval_expr(e.items[0], env)
        args = [eval_expr(a, env) for a in e.items[1:]]
        if callable(proc):
            return proc(args)
        raise EvalError(f"不可应用：{proc!r}")

    def install_prim(key: str, fn: Any) -> None:
        top[("core", (0,), None, f"core:{key}")] = Cell(fn)

    def arith(fn):
        return lambda xs: fn(xs)

    install_prim("+", arith(sum))
    install_prim("-", arith(lambda xs: xs[0] - xs[1]))
    install_prim("*", arith(lambda xs: xs[0] * xs[1]))
    install_prim("=", arith(lambda xs: xs[0] == xs[1]))
    install_prim("<", arith(lambda xs: xs[0] < xs[1]))
    install_prim(">", arith(lambda xs: xs[0] > xs[1]))
    install_prim("not", lambda xs: not _truthy(xs[0]))
    install_prim("sub1", lambda xs: xs[0] - 1)
    install_prim("zero?", lambda xs: xs[0] == 0)
    install_prim("cons", lambda xs: (xs[0], xs[1]))
    install_prim("car", lambda xs: xs[0][0])
    install_prim("cdr", lambda xs: xs[0][1])
    install_prim("list", lambda xs: list(xs))
    install_prim("null?", lambda xs: isinstance(xs[0], list) and not xs[0])
    install_prim("pair?", lambda xs: isinstance(xs[0], tuple))
    install_prim("display", lambda xs: xs[0])
    install_prim("void", lambda xs: None)

    result: Any = None
    for f in program:
        if f.head() == "define":
            name = f.items[1]
            edge = binding_of(name)
            k = edge.binder.identity_key()
            top.setdefault(k, Cell(None)).val = eval_expr(f.items[2], top)
        elif f.head() == "define-syntax":
            continue
        else:
            result = eval_expr(f, top)
    return result
