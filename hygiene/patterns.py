"""syntax-rules pattern matching and template instantiation.

Patterns (syntax datum):
  (p ...)            list pattern
  identifier         literal to be matched by name (literals given separately)
  _                  wildcard
  other atoms        self-evaluating literal match
A pattern element followed by the ``...`` identifier repeats zero or more.

Templates add two transformer-internal primitives (they are syntax errors in
user programs, enforced by the parser):
  (datum->syntax <ctx> <sym>)  manufacture an identifier using <ctx>'s scope
                               set — the *context choice* is recorded
  (with-temp (x ...) body)     bind template vars to generated temporary
                               names (hygienic gensym) then splice body
"""
from __future__ import annotations

from .syntax import Syntax, Birth


class MatchFail(Exception):
    def __init__(self, reason: str, obj: Syntax | None = None):
        self.reason = reason
        self.obj = obj
        super().__init__(reason)


def match_pattern(pat: Syntax, obj: Syntax, literals: set[str]) -> dict:
    """Return pattern-variable bindings or raise MatchFail with a reason.

    Repeated occurrences of the same pattern variable must be structurally
    equal; the mismatch reason is retained by the caller as a reject cause.
    """
    env: dict = {}
    _match(pat, obj, literals, env)
    return env


def _match(pat: Syntax, obj: Syntax, literals: set[str], env: dict) -> None:
    if pat.is_list:
        if not obj.is_list:
            raise MatchFail("pattern expects a list", obj)
        pv = pat.val
        i = 0
        oi = 0
        while i < len(pv):
            if i + 1 < len(pv) and _is_ellipsis(pv[i + 1]):
                rep_pat = pv[i]
                tail_pats = pv[i + 2:]
                tail_len = len(tail_pats)
                rep_count = len(obj.val) - oi - tail_len
                if rep_count < 0:
                    raise MatchFail("too few list elements for repeated pattern", obj)
                frames: list[dict] = []
                for k in range(rep_count):
                    frame: dict = {}
                    _match(rep_pat, obj.val[oi + k], literals, frame)
                    frames.append(frame)
                var = _pattern_var(rep_pat)
                env[var] = frames
                oi += rep_count
                i += 2
                for tp in tail_pats:
                    if oi >= len(obj.val):
                        raise MatchFail("missing tail element", obj)
                    _match(tp, obj.val[oi], literals, env)
                    oi += 1
                    i += 1
                break
            if oi >= len(obj.val):
                raise MatchFail("list shorter than pattern", obj)
            _match(pv[i], obj.val[oi], literals, env)
            i += 1
            oi += 1
        if oi != len(obj.val):
            raise MatchFail("list longer than pattern", obj)
        return
    sym = pat.symbol
    if sym == "_":
        return
    if sym is not None:
        if sym in literals:
            if obj.symbol != sym:
                raise MatchFail(f"expected literal {sym!r}, got {obj.symbol!r}", obj)
            return
        var = sym
        if var in env and not _same_datum(env[var], obj):
            raise MatchFail(f"repeated pattern variable {var!r} not structurally equal", obj)
        env[var] = obj
        return
    if obj.is_list or obj.val != pat.val:
        raise MatchFail(f"expected literal datum {pat.val!r}", obj)


def _pattern_var(p: Syntax) -> str:
    sym = p.symbol
    return sym if sym is not None else "_"


def _is_ellipsis(s: Syntax) -> bool:
    return s.symbol == "..."


def _same_datum(a, b: Syntax) -> bool:
    if isinstance(a, Syntax):
        if a.is_list != b.is_list:
            return False
        if a.is_list:
            return len(a.val) == len(b.val) and all(
                _same_datum(x, y) for x, y in zip(a.val, b.val))
        return a.val == b.val
    return a == b.val


def instantiate(template: Syntax, env: dict, reg, *, call_oid: str, macro_name: str,
                frame_stack: list[dict] | None = None, events: list | None = None,
                clock=None) -> Syntax:
    """Build output syntax from a frozen definition-site template.

    Template objects are cloned (so a definition is reusable across calls) and
    inherit the template's frozen scope set: this is exactly what makes a
    definition-site reference resolve at the macro's definition scope.
    """
    frame_stack = frame_stack or []
    return _instantiate(template, env, reg, call_oid=call_oid, macro_name=macro_name,
                        frame_stack=frame_stack, events=events, clock=clock,
                        ellipsis_depth=0)


def _instantiate(t: Syntax, env: dict, reg, *, call_oid, macro_name, frame_stack,
                 events, clock, ellipsis_depth: int) -> Syntax:
    if t.is_list:
        tv = t.val
        # (datum->syntax ctx sym)
        if len(tv) == 3 and tv[0].symbol == "datum->syntax":
            return _do_datum_to_syntax(tv, env, reg, call_oid, macro_name,
                                       frame_stack, events)
        # (with-temp (x ...) body)
        if len(tv) == 3 and tv[0].symbol == "with-temp" and tv[1].is_list:
            return _do_with_temp(tv, env, reg, call_oid, macro_name, frame_stack,
                                 events, clock, ellipsis_depth)
        out_elems: list[Syntax] = []
        i = 0
        while i < len(tv):
            if i + 1 < len(tv) and _is_ellipsis(tv[i + 1]):
                sub = tv[i]
                var = _single_var(sub)
                frames = env.get(var)
                if var not in env or not isinstance(frames, list):
                    raise MatchFail(f"template variable {var!r} has no repetition")
                for frame in frames:
                    out_elems.append(_instantiate(sub, env, reg, call_oid=call_oid,
                                                  macro_name=macro_name,
                                                  frame_stack=frame_stack + [frame],
                                                  events=events, clock=clock,
                                                  ellipsis_depth=ellipsis_depth + 1))
                i += 2
                continue
            out_elems.append(_instantiate(tv[i], env, reg, call_oid=call_oid,
                                          macro_name=macro_name, frame_stack=frame_stack,
                                          events=events, clock=clock,
                                          ellipsis_depth=ellipsis_depth))
            i += 1
        node = Syntax(out_elems, True, t.span,
                      birth=Birth("template", detail=f"list:{macro_name}", call_oid=call_oid,
                                  macro_name=macro_name, token_id=_tok(t)))
        reg.record(node)
        return node
    sym = t.symbol
    if sym is not None and sym != "...":
        val = _lookup_var(sym, env, frame_stack)
        if val is not None:
            return val  # input-derived object: keeps its own scopes/birth
    node = t.clone(birth_kind="template", detail=f"atom:{sym if sym is not None else t.val}",
                   call_oid=call_oid, macro_name=macro_name)
    reg.record(node)
    return node


def _lookup_var(name: str, env: dict, frame_stack: list[dict]):
    for frame in reversed(frame_stack):
        if name in frame:
            return frame[name]
    return env.get(name)


def _single_var(t: Syntax) -> str:
    if t.symbol is not None:
        return t.symbol
    if t.is_list:
        for e in t.val:
            if e.symbol not in (None, "..."):
                return e.symbol
    raise MatchFail("ellipsis template must center on a pattern variable")


def _do_datum_to_syntax(tv, env, reg, call_oid, macro_name, frame_stack, events):
    ctx_expr, sym_expr = tv[1], tv[2]
    donor = _instantiate(ctx_expr, env, reg, call_oid=call_oid, macro_name=macro_name,
                         frame_stack=frame_stack, events=events, clock=None,
                         ellipsis_depth=0)
    raw = sym_expr.val if sym_expr.symbol is not None else None
    recipient = Syntax(raw, False, sym_expr.span, set(donor.scopes),
                       birth=Birth("datum_to_syntax", detail=f"explicit-capture:{raw}",
                                   call_oid=call_oid, macro_name=macro_name,
                                   token_id=_tok(sym_expr), context_donor=donor.oid))
    reg.record(recipient)
    if events is not None:
        events.append({
            "kind": "context_choice",
            "call_oid": call_oid,
            "macro": macro_name,
            "recipient_oid": recipient.oid,
            "donor_oid": donor.oid,
            "donor_symbol": donor.symbol,
            "scopes": sorted(donor.scopes),
            "rule": "datum->syntax adopts the donor's scope set explicitly; "
                    "no scope means no capture",
        })
    return recipient


def _do_with_temp(tv, env, reg, call_oid, macro_name, frame_stack, events, clock,
                  ellipsis_depth):
    names_node, body = tv[1], tv[2]
    local = {}
    for name_tok in names_node.val:
        nm = name_tok.symbol
        suffix = clock.rand16() if clock is not None else "0000"
        raw = f"#{nm}~{suffix}"
        gen = Syntax(raw, False, name_tok.span,
                     birth=Birth("gensym", detail=f"temp:{nm}", call_oid=call_oid,
                                 macro_name=macro_name, token_id=_tok(name_tok)))
        reg.record(gen)
        local[nm] = gen
        if events is not None:
            events.append({
                "kind": "gensym",
                "call_oid": call_oid,
                "macro": macro_name,
                "name": nm,
                "generated_oid": gen.oid,
                "generated_symbol": raw,
                "rule": "with-temp creates a fresh symbol; its scopes are empty, so "
                        "user identifiers cannot capture it by name",
            })
    merged = dict(env)
    merged.update(local)
    return _instantiate(body, merged, reg, call_oid=call_oid, macro_name=macro_name,
                        frame_stack=frame_stack, events=events, clock=clock,
                        ellipsis_depth=ellipsis_depth)


def _tok(t: Syntax):
    return t.birth.token_id if t.birth else None
