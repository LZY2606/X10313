"""Compile-time evaluator for the meta-language used by transformer bodies.

A transformer is a host-side callable produced by evaluating the RHS of a
``define-syntax`` in this tiny interpreter.  The meta-language is itself a
subset of the same surface language:

  (lambda (x ...) body ...)
  (let ((x e) ...) body ...)
  (if c t e) (begin e ...) (quote datum)
  (quasiquote template) with ,e and ,@e
  (#%lambda x e) (#%call f a...) (#%quote d)  (internal AST)

Compile-time primitives (visible to user code):
  syntax-e  syntax->datum  datum->syntax  generate-temporaries
  bound-identifier=?  free-identifier=?  syntax-span  list  cons
  car cdr vector?  + - * = < >  and or not  error
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .syntax import GraphBuilder, Stx


class CTError(Exception):
    def __init__(self, message: str, node: Optional[Stx] = None):
        super().__init__(message)
        self.node = node


@dataclass(frozen=True)
class ListVal:
    """A genuine host list returned by the compile-time ``list`` primitive.

    Bare Python lists represent syntax lists (output of a transformer);
    ``ListVal`` is opaque meta-level data (``syntax-e`` results etc.).
    """
    items: tuple[Any, ...]


@dataclass
class Closure:
    params: tuple[str, ...]
    body: tuple[Any, ...]
    env: dict[str, Any]
    name: str = "lambda"
    eval_fn: Optional[Callable[..., Any]] = None


# Host values produced by compile-time evaluation are plain Python data:
#   Stx                 - a syntax object
#   list                - a host list
#   str/int/float/bool/None
#   Closure             - first-class lambda
#   dict(kind="raw")    - unconvertible data carried through untouched


def make_prims(builder: GraphBuilder, macro_name: str) -> dict[str, Any]:
    def as_items(x: Any) -> tuple[Any, ...]:
        if isinstance(x, ListVal):
            return x.items
        if isinstance(x, Stx) and x.kind == "seq":
            return tuple(x.seq)
        if isinstance(x, list):
            return tuple(x)
        raise CTError(f"expected a list, got {type(x).__name__}")

    def syntax_e(s: Any) -> Any:
        if isinstance(s, Stx) and s.kind == "seq":
            return ListVal(tuple(s.seq))
        if isinstance(s, Stx):
            raise CTError(f"syntax-e: expected a syntax list, got {s.lexeme}", s)
        raise CTError(f"syntax-e: expected syntax object, got {type(s).__name__}")

    def to_datum(s: Any) -> Any:
        if isinstance(s, Stx):
            return s.to_plain()
        if isinstance(s, ListVal):
            return [to_datum(x) for x in s.items]
        if isinstance(s, list):
            return [to_datum(x) for x in s]
        return s

    def bound_eq(a: Any, b: Any) -> bool:
        return (isinstance(a, Stx) and isinstance(b, Stx)
                and a.value == b.value and a.scopes == b.scopes)

    def free_eq(a: Any, b: Any) -> bool:
        return (isinstance(a, Stx) and isinstance(b, Stx)
                and a.value == b.value)

    def syntax_span(s: Any) -> Any:
        if isinstance(s, Stx) and s.span is not None:
            sp = s.span
            return {"file": sp.file, "offset": sp.offset, "line": sp.line,
                    "col": sp.col, "length": sp.length, "unit": sp.unit}
        return None

    counter = {"n": 0}

    def gen_temps(template: Any = None) -> Any:
        n = 1
        items = as_items(template) if template is not None else ()
        n = len(items)
        out: list[Stx] = []
        for _ in range(n):
            counter["n"] += 1
            label = f"$tmp_{macro_name}_{counter['n']:02d}"
            intro = builder.fresh_intro("tmp")
            node = builder.node(
                kind="id", value=label, lexeme=label,
                scopes=frozenset({intro}), span=None,
                reason=f"generate-temporaries:{macro_name}",
                flags=("introduced", "generated-symbol"))
            builder.log_step("generate-temporaries", macro=macro_name,
                             symbol=label, node=node.id, intro_scope=intro)
            out.append(node)
        return out

    def err(msg: Any = "compile-time error", node: Any = None) -> Any:
        text = msg.value if isinstance(msg, Stx) else str(msg)
        raise CTError(text, node if isinstance(node, Stx) else None)

    def num(x: Any) -> Any:
        if isinstance(x, Stx):
            if x.kind in ("number", "bool"):
                return x.value
            raise CTError(f"expected a numeric literal, got {x.lexeme}", x)
        return x

    def truth(x: Any) -> Any:
        return x.value if isinstance(x, Stx) and x.kind == "bool" else x

    return {
        "syntax-e": syntax_e,
        "syntax->datum": to_datum,
        "bound-identifier=?": bound_eq,
        "free-identifier=?": free_eq,
        "syntax-span": syntax_span,
        "generate-temporaries": gen_temps,
        "error": err,
        "list": lambda *xs: ListVal(tuple(xs)),
        "cons": lambda a, b: ListVal((a,) + as_items(b)),
        "car": lambda xs: as_items(xs)[0],
        "cdr": lambda xs: ListVal(as_items(xs)[1:]),
        "cadr": lambda xs: as_items(xs)[1],
        "caddr": lambda xs: as_items(xs)[2],
        "cadddr": lambda xs: as_items(xs)[3],
        "length": lambda xs: len(as_items(xs)),
        "append": lambda *ls: ListVal(tuple(
            x for l in ls for x in as_items(l))),
        "vector?": lambda xs: isinstance(xs, Stx) and xs.bracket == "{}",
        "pair?": lambda xs: len(as_items(xs)) > 0,
        "null?": lambda xs: len(as_items(xs)) == 0,
        "identifier?": lambda xs: isinstance(xs, Stx) and xs.kind == "id",
        "symbol->string": lambda s: s.value if isinstance(s, Stx) else str(s),
        "+": lambda *xs: sum(num(x) for x in xs),
        "-": lambda a, b=None: -num(a) if b is None else num(a) - num(b),
        "*": lambda *xs: __import__("math").prod(num(x) for x in xs),
        "=": lambda a, b: num(a) == num(b),
        "<": lambda a, b: num(a) < num(b),
        ">": lambda a, b: num(a) > num(b),
        "<=": lambda a, b: num(a) <= num(b),
        ">=": lambda a, b: num(a) >= num(b),
        "and": lambda *xs: all(truth(x) for x in xs),
        "or": lambda *xs: any(truth(x) for x in xs),
        "not": lambda x: not truth(x),
        "display": lambda *xs: None,
        "void": lambda *xs: None,
    }
