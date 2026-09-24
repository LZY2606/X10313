"""Interpreter for compile-time meta-language expressions."""
from __future__ import annotations

from typing import Any, Optional

from .ctenv import CTError, Closure, ListVal, make_prims
from .syntax import GraphBuilder, Stx


class MetaEvaluator:
    def __init__(self, builder: GraphBuilder, macro_name: str = "<toplevel>"):
        self.builder = builder
        self.macro_name = macro_name
        self._template_intros: dict[str, str] = {}
        self.prims = make_prims(builder, macro_name)
        self.globals: dict[str, Any] = dict(self.prims)

    # -- quoted datums -----------------------------------------------------
    def _quote_value(self, stx: Stx) -> Any:
        if stx.kind != "seq":
            if stx.kind == "raw":
                return {"#raw": stx.lexeme}
            return stx.value
        tag = stx.seq[0].value if stx.seq else ""
        if tag == "quote":
            return self._quote_value(stx.seq[1])
        if tag == "quasiquote":
            return self._quasi_seq(stx.seq[1], 1, env)
        return [self._quote_value(c) for c in stx.seq]

    def _template_leaf(self, stx: Stx) -> Any:
        """A quoted template symbol becomes a *fresh* introduced identifier.

        It carries a template introduction scope plus the current macro mark
        (if any), so hygiene is preserved; user identifiers only arrive via
        unquote and keep their own scope sets.
        """
        intro = self._template_intros.get(self.macro_name)
        if intro is None:
            intro = self.builder.fresh_intro(f"tpl:{self.macro_name}")
            self._template_intros[self.macro_name] = intro
        if stx.kind == "id":
            return self.builder.node(
                kind="id", value=stx.value, lexeme=stx.lexeme,
                scopes=frozenset({intro}), span=stx.span,
                parents=(stx.id,),
                reason=f"template-identifier:{self.macro_name}",
                flags=("introduced", "template"))
        return self._quote_value(stx)

    def _quasi_seq(self, stx: Stx, depth: int, env: dict[str, Any]) -> Any:
        if stx.kind != "seq":
            return self._template_leaf(stx)
        tag = stx.seq[0].value if stx.seq else ""
        if tag == "unquote" and depth == 1:
            return self.eval(stx.seq[1], env)
        if tag == "unquote-splicing":
            raise CTError(",@ outside splicing position", stx)
        if tag == "quasiquote":
            return [stx.seq[0].value,
                    self._quasi_seq(stx.seq[1], depth + 1, env)]
        if tag == "unquote":
            return [stx.seq[0].value,
                    self._quasi_seq(stx.seq[1], depth - 1, env)]
        out: list[Any] = []
        for c in stx.seq:
            if (c.kind == "seq" and c.seq
                    and c.seq[0].value == "unquote-splicing"
                    and depth == 1):
                piece = self.eval(c.seq[1], env)
                if isinstance(piece, ListVal):
                    piece = list(piece.items)
                if not isinstance(piece, list):
                    raise CTError(",@ value must be a list", c)
                out.extend(piece)
            else:
                out.append(self._quasi_seq(c, depth, env))
        return out

    # -- evaluation --------------------------------------------------------
    def eval(self, stx: Stx, env: dict[str, Any]) -> Any:
        if stx.kind != "seq":
            if stx.kind == "id":
                if stx.value in env:
                    return env[stx.value]
                if stx.value in self.globals:
                    return self.globals[stx.value]
                raise CTError(f"compile-time unbound identifier: {stx.value}",
                              stx)
            if stx.kind == "raw":
                return {"#raw": stx.lexeme}
            return stx.value

        head = stx.seq[0]
        tag = head.value if head.kind == "id" else None

        if tag == "quote":
            return self._quote_value(stx.seq[1])
        if tag == "quasiquote":
            return self._quasi_seq(stx.seq[1], 1, env)
        if tag == "lambda":
            params = tuple(p.value for p in stx.seq[1].seq)
            return Closure(params, tuple(stx.seq[2:]), dict(env),
                           eval_fn=self.eval)
        if tag == "if":
            cond = self.eval(stx.seq[1], env)
            return self.eval(stx.seq[2 if cond else 3], env)
        if tag == "begin":
            result: Any = None
            for sub in stx.seq[1:]:
                result = self.eval(sub, env)
            return result
        if tag == "let":
            local = dict(env)
            for clause in stx.seq[1].seq:
                local[clause.seq[0].value] = self.eval(clause.seq[1], env)
            result = None
            for sub in stx.seq[2:]:
                result = self.eval(sub, local)
            return result
        if tag == "define":
            val = self.eval(stx.seq[2], env)
            env[stx.seq[1].value] = val
            return None
        # application
        fn = self.eval(head, env)
        args = [self.eval(a, env) for a in stx.seq[1:]]
        if isinstance(fn, Closure):
            call_env = dict(fn.env)
            call_env.update(dict(zip(fn.params, args)))
            result = None
            for sub in fn.body:
                result = self.eval(sub, call_env)
            return result
        if callable(fn):
            return fn(*args)
        raise CTError(f"not a compile-time function: {tag!r}", head)
