"""Tiny meta-level evaluator used to expand macro templates.

Runtime values are tagged ``Node`` syntax objects or plain Python values
(wrapped in :class:`Value`) produced by ``quote``.  ``quasiquote`` builds
syntax objects that carry the *definition-site* scope set; unquoted holes splice
the caller's syntax (with its own scopes and spans), which is exactly the
mechanism hygiene is built from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import Node, Origin, Span, Binding


class MetaError(Exception):
    def __init__(self, message: str, node: Optional[Node] = None,
                 phase: str = "meta"):
        super().__init__(message)
        self.message = message
        self.node = node
        self.phase = phase


class ExpandLimit(Exception):
    def __init__(self, message: str, budget: int, partial: list[Node],
                 node: Optional[Node] = None):
        super().__init__(message)
        self.message = message
        self.budget = budget
        self.partial = partial
        self.node = node


@dataclass(frozen=True)
class Value:
    """Plain (syntax-less) datum as produced by ``quote``."""
    v: Any


@dataclass
class Closure:
    params: tuple[str, ...]
    body: Node
    env: dict
    name: str = "lambda"


@dataclass
class Primitive:
    name: str
    fn: Callable
    wants_ctx: bool = False


@dataclass
class Pair:
    car: Any
    cdr: Any


@dataclass
class Context:
    """Handle given to a macro at its call site (``(call-context)``)."""
    id: int
    node: Node


class Evaluator:
    def __init__(self, expander: "Expander", env: Optional[dict] = None):  # type: ignore[name-defined]
        self.ex = expander
        self.env = env or dict(expander.meta_base)

    def run(self, node: Node, env: Optional[dict] = None) -> Any:
        e = self.env if env is None else env
        if node.kind == "atom":
            return self.eval_atom(node, e)
        head = node.children[0].atom_text() if node.children else ""
        method = getattr(self, f"form_{_safe(head)}", None)
        if method is not None:
            return method(node, e)
        fn = self.eval(node.children[0], e)
        args = [self.eval(a, e) for a in node.children[1:]]
        return self.apply(fn, args, node)

    def eval_atom(self, node: Node, env: dict) -> Any:
        text = node.atom_text()
        if node.lit == "boolean":
            return text == "#t"
        if node.lit == "number":
            num = _to_number(text)
            if num is None:
                raise MetaError(f"unconvertible number literal: {text!r}", node,
                                phase="normalize")
            return num
        if node.lit == "string":
            return _decode_string(text)
        if text in env:
            return env[text]
        raise MetaError(f"unbound meta identifier: {text}", node)

    def apply(self, fn: Any, args: list, site: Node) -> Any:
        if isinstance(fn, Closure):
            if len(args) != len(fn.params):
                raise MetaError(
                    f"{fn.name}: expected {len(fn.params)} args, got {len(args)}", site)
            local = dict(fn.env)
            local.update(zip(fn.params, args))
            return Evaluator(self.ex).run(fn.body, local)
        if isinstance(fn, Primitive):
            return fn.fn(site, self, *args) if fn.wants_ctx else fn.fn(site, *args)
        raise MetaError(f"not applicable: {_show(fn)}", site)

    # ---- special forms -----------------------------------------------------
    def form_lambda(self, node: Node, env: dict) -> Closure:
        params = tuple(a.atom_text() for a in node.children[1].children)
        return Closure(params, node.children[2], env)

    def form_let(self, node: Node, env: dict) -> Any:
        local = dict(env)
        for binding in node.children[1].children:
            name = binding.children[0].atom_text()
            local[name] = self.run(binding.children[1], local)
        result: Any = None
        for form in node.children[2:]:
            result = self.run(form, local)
        return result

    def form_if(self, node: Node, env: dict) -> Any:
        test = self.run(node.children[1], env)
        branch = node.children[2] if _truthy(test) else node.children[3]
        return self.run(branch, env)

    def form_begin(self, node: Node, env: dict) -> Any:
        result: Any = None
        for form in node.children[1:]:
            result = self.run(form, env)
        return result

    def form_set(self, node: Node, env: dict) -> Any:  # set!
        name = node.children[1].atom_text()
        if name not in env:
            raise MetaError(f"set! on unbound identifier: {name}", node)
        env[name] = self.run(node.children[2], env)
        return None

    def form_quote(self, node: Node, env: dict) -> Value:
        return Value(_datum(node.children[1]))

    def form_quasiquote(self, node: Node, env: dict) -> Any:
        return self._qq(node.children[1], env, node)

    def _qq(self, form: Node, env: dict, template: Node) -> Any:
        if form.kind == "list" and form.children:
            head = form.children[0]
            if head.kind == "atom" and head.atom_text() == "unquote":
                return self.run(form.children[1], env)
            parts: list[Any] = []
            spliced: list[dict] = []
            for idx, child in enumerate(form.children):
                if child.kind == "list" and child.children and \
                        child.children[0].kind == "atom" and \
                        child.children[0].atom_text() == "unquote-splicing":
                    seq = self.run(child.children[1], env)
                    items = _seq_items(seq)
                    spliced.append({"at": len(parts), "items": items})
                    parts.extend(items)
                else:
                    parts.append(self._qq(child, env, template))
            return self.ex.make_template_list(form, parts, spliced, template)
        # atoms in a template are copied with definition-site scopes
        return self.ex.copy_template_atom(form)

    # ---- identifier/context primitives ------------------------------------
    def form_identifier_eq(self, node: Node, env: dict) -> bool:
        a, b = self.run(node.children[1], env), self.run(node.children[2], env)
        return isinstance(a, Node) and isinstance(b, Node) and a.kind == "atom" \
            and b.kind == "atom" and a.atom_text() == b.atom_text() \
            and a.scopes == b.scopes


def _safe(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)


def _truthy(value: Any) -> bool:
    return value is not False and value is not None


def _to_number(text: str) -> Optional[float]:
    try:
        if any(ch in text for ch in ".eE") or text[:1] in "+-":
            return float(text)
        return float(int(text))
    except ValueError:
        return None


def _decode_string(body: str) -> str:
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _datum(node: Node) -> Any:
    if node.kind == "atom":
        if node.lit == "boolean":
            return node.atom_text() == "#t"
        if node.lit == "number":
            num = _to_number(node.atom_text())
            if num is None:
                return {"__unconvertible__": node.atom_text()}
            return num
        if node.lit == "string":
            return _decode_string(node.atom_text())
        return node.atom_text()
    items = [_datum(c) for c in node.children]
    if node.dot and len(items) >= 2:
        tail = items.pop()
        pair = Pair(tail, None)
        for item in reversed(items):
            pair = Pair(item, pair)
        return pair
    return items


def _seq_items(value: Any) -> list[Any]:
    if isinstance(value, Value):
        value = value.v
    if isinstance(value, list):
        return value
    if isinstance(value, Pair):
        out = []
        cur: Any = value
        while isinstance(cur, Pair):
            out.append(cur.car)
            cur = cur.cdr
        return out
    raise MetaError("splice target is not a sequence")


def _show(value: Any) -> str:
    if isinstance(value, Node):
        return value.atom_text() or "(...)"
    if isinstance(value, Value):
        return repr(value.v)
    return repr(value)
