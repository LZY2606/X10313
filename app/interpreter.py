"""Tiny interpreter over fully expanded syntax objects.

Numeric conversion happens here (the *derived* layer): an atom whose raw text
looks numeric but cannot be normalized is kept at its original site with a
diagnostic instead of silently being dropped or turned into 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .models import Node


class InterpError(Exception):
    def __init__(self, message: str, node: Optional[Node] = None):
        super().__init__(message)
        self.message = message
        self.node = node


@dataclass
class Lam:
    params: tuple[str, ...]
    body: tuple[Node, ...]
    env: dict


def _atom_value(node: Node) -> Any:
    if node.lit == "boolean":
        return node.atom_text() == "#t"
    if node.lit == "string":
        return _decode(node.atom_text())
    if node.lit == "number":
        return _number(node)
    raise InterpError("not a literal", node)


def _number(node: Node):
    text = node.atom_text()
    try:
        if any(c in text for c in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        raise InterpError(f"unconvertible number literal: {text!r}", node)


def _decode(body: str) -> str:
    out, i = [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            out.append({"n": "\n", "t": "\t"}.get(body[i + 1], body[i + 1]))
            i += 2
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


class Env:
    def __init__(self, parent: Optional["Env"] = None):
        self.parent = parent
        self.vars: dict[str, Any] = {}

    def find(self, name: str) -> "Env":
        if name in self.vars:
            return self
        if self.parent is not None:
            return self.parent.find(name)
        raise KeyError(name)

    def get(self, name: str) -> Any:
        return self.find(name).vars[name]

    def set(self, name: str, value: Any) -> None:
        self.find(name).vars[name] = value

    def define(self, name: str, value: Any) -> None:
        self.vars[name] = value


def interpret(forms: list[Node]) -> tuple[list[Any], list[dict]]:
    env = Env()
    diagnostics: list[dict] = []
    values: list[Any] = []
    for form in forms:
        try:
            values.append(eval_node(form, env))
        except InterpError as exc:
            diagnostics.append({
                "code": "interpreter", "message": exc.message,
                "sid": exc.node.sid if exc.node else None,
                "span": exc.node.span.to_json() if exc.node and exc.node.span else None})
    return values, diagnostics


def eval_node(node: Node, env: Env) -> Any:
    if node.kind == "atom":
        if node.lit != "symbol":
            return _atom_value(node)
        return env.get(node.atom_text())
    head = node.children[0].atom_text() if node.children else ""
    if head == "quote":
        return _quoted(node.children[1])
    if head == "lambda":
        params = tuple(a.atom_text() for a in node.children[1].children)
        return Lam(params, tuple(node.children[2:]), env)
    if head == "if":
        cond = eval_node(node.children[1], env)
        return eval_node(node.children[2] if cond else node.children[3], env)
    if head == "begin":
        result = None
        for form in node.children[1:]:
            result = eval_node(form, env)
        return result
    if head == "let":
        local = Env(env)
        for bind in node.children[1].children:
            local.define(bind.children[0].atom_text(),
                         eval_node(bind.children[1], env))
        result = None
        for form in node.children[2:]:
            result = eval_node(form, local)
        return result
    if head == "set!":
        env.set(node.children[1].atom_text(), eval_node(node.children[2], env))
        return None
    if head == "define":
        target = node.children[1]
        if target.kind == "list":
            name = target.children[0].atom_text()
            params = tuple(a.atom_text() for a in target.children[1:])
            fn = Lam(params, tuple(node.children[2:]), env)
            env.define(name, fn)
            return None
        env.define(target.atom_text(), eval_node(node.children[2], env))
        return None
    fn = eval_node(node.children[0], env)
    args = [eval_node(a, env) for a in node.children[1:]]
    return apply_fn(fn, args, node)


def apply_fn(fn: Any, args: list[Any], site: Node) -> Any:
    if isinstance(fn, Lam):
        local = Env(fn.env)
        for name, value in zip(fn.params, args):
            local.define(name, value)
        result = None
        for form in fn.body:
            result = eval_node(form, local)
        return result
    if callable(fn):
        return fn(*args)
    raise InterpError("not applicable", site)


def _quoted(node: Node) -> Any:
    if node.kind == "atom":
        if node.lit == "boolean":
            return node.atom_text() == "#t"
        if node.lit == "number":
            return _number(node)
        return node.atom_text()
    return [_quoted(c) for c in node.children]
