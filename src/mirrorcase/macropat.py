"""syntax-rules pattern matching and template instantiation.

Patterns are syntax objects (id leaves / forms / dotted tails).
``...`` is the ellipsis marker. Pattern matching is hygiene blind: it
compares symbols for literals and binds everything else - capture is
decided later by scope relations, never by renaming strings.
"""

from __future__ import annotations

from typing import Any, Optional

from .syntax import OriginLink, SObj

ELLIPSIS = "..."


class MatchFailure(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _atom_equal(pat: SObj, val: SObj) -> bool:
    if pat.kind != val.kind:
        return False
    if pat.kind == "id":
        return pat.atom == val.atom  # literals compare by symbol
    return pat.atom == val.atom


def _collect_repeat_vars(pat: SObj, literals: set[str],
                         acc: set[str]) -> None:
    if not pat.is_form:
        return
    items = pat.items
    for idx, sub in enumerate(items):
        if (sub.kind == "id" and sub.atom == ELLIPSIS and idx > 0
                and idx == len(items) - 1):
            _vars_under(items[idx - 1], literals, acc)
        else:
            _collect_repeat_vars(sub, literals, acc)
    if pat.tail is not None:
        _collect_repeat_vars(pat.tail, literals, acc)


def _vars_under(pat: SObj, literals: set[str], acc: set[str]) -> None:
    if pat.kind == "id":
        if pat.atom not in literals and pat.atom != "_":
            acc.add(pat.atom)
        return
    if pat.is_form:
        for c in pat.items:
            _vars_under(c, literals, acc)
        if pat.tail is not None:
            _vars_under(pat.tail, literals, acc)


def pattern_vars(pat: SObj, literals: set[str]) -> set[str]:
    acc: set[str] = set()
    _vars_under(pat, literals, acc)
    return acc


def match_pattern(pat: SObj, val: SObj, literals: set[str],
                  binds: dict[str, Any]) -> dict[str, Any]:
    """Returns binds; raises MatchFailure.

    Binds map name -> either SObj (single) or list[SObj] (repeated).
    """
    if pat.kind == "id":
        name = pat.atom
        if name == "_":
            return binds
        if name in literals:
            if val.kind != "id" or val.atom != name:
                raise MatchFailure(
                    f"literal mismatch: expected {name!r}")
            return binds
        if name in binds:
            prev = binds[name]
            if isinstance(prev, SObj) and prev.atom == val.atom:
                return binds
            raise MatchFailure(f"pattern variable {name!r} bound twice")
        binds[name] = val
        return binds
    if not pat.is_form:
        if not val.is_form and _atom_equal(pat, val):
            return binds
        raise MatchFailure("atom mismatch")
    if not val.is_form:
        raise MatchFailure("expected a list")
    return _match_form(pat, val, literals, binds)


def _match_form(pat: SObj, val: SObj, literals: set[str],
                binds: dict[str, Any]) -> dict[str, Any]:
    pit: list[SObj] = pat.items
    vit: list[SObj] = val.items
    pi = vi = 0
    while pi < len(pit):
        sub = pit[pi]
        is_rep = (pi + 1 < len(pit)
                  and pit[pi + 1].kind == "id"
                  and pit[pi + 1].atom == ELLIPSIS)
        if is_rep:
            rep_vars: set[str] = set()
            _vars_under(sub, literals, rep_vars)
            # consume everything up to the remaining fixed tail
            fixed_after = len(pit) - (pi + 2)
            if pat.tail is not None:
                fixed_after = 0
            end = len(vit) - fixed_after
            if end < vi:
                raise MatchFailure("not enough values for ellipsis")
            rows: list[dict[str, SObj]] = []
            for piece in vit[vi:end]:
                row: dict[str, SObj] = {}
                match_pattern(sub, piece, literals, row)
                for k, v in row.items():
                    if k in binds:
                        raise MatchFailure(
                            f"repeat var {k!r} also bound outside")
                rows.append(row)
            for rv in rep_vars:
                if not rows:
                    binds[rv] = []
                else:
                    if not all(rv in r for r in rows):
                        raise MatchFailure(
                            f"repeat var {rv!r} missing in some rows")
                    binds[rv] = [r[rv] for r in rows]
            vi = end
            pi += 2
            continue
        if vi >= len(vit):
            raise MatchFailure("list too short")
        match_pattern(sub, vit[vi], literals, binds)
        pi += 1
        vi += 1
    if pat.tail is not None:
        if vi < len(vit):
            remainder = SObj.form(
                vit[vi:], pat.span, origin=pat.tail.origin,
                tail=val.tail)
            match_pattern(pat.tail, remainder, literals, binds)
        elif val.tail is not None:
            match_pattern(pat.tail, val.tail, literals, binds)
        else:
            # bind the tail to the empty list
            match_pattern(
                pat.tail,
                SObj.form([], pat.span, origin=pat.tail.origin),
                literals, binds)
    else:
        if vi != len(vit):
            raise MatchFailure("list too long")
        if val.tail is not None:
            raise MatchFailure("unexpected dotted tail")
    return binds


# ---------------------------------------------------------------- templates

def _referenced_ids(t: SObj, acc: set[str]) -> None:
    if t.kind == "id":
        acc.add(t.atom)
        return
    if t.is_form:
        for c in t.items:
            _referenced_ids(c, acc)
        if t.tail is not None:
            _referenced_ids(t.tail, acc)


class Instantiator:
    """Instantiates a template under a match.

    ``introduced(step, scope, origin)`` marks template-literal pieces;
    substituted pieces keep their own identity/scopes/labels.
    """

    def __init__(self, binds: dict[str, Any], repeat_vars: set[str],
                 step: int, scope: int, literals: set[str]):
        self.binds = binds
        self.repeat_vars = repeat_vars
        self.step = step
        self.scope = scope
        self.literals = literals

    def run(self, t: SObj) -> SObj:
        return self._inst(t, origin_link=True)

    def _inst(self, t: SObj, origin_link: bool = True) -> SObj:
        if t.kind == "id" and t.atom in self.binds and t.atom not in self.literals:
            v = self.binds[t.atom]
            if isinstance(v, list):
                raise MatchFailure(
                    f"repeat variable {t.atom!r} used outside ellipsis")
            # substituted use-site piece keeps its own scopes/labels
            return v.clone(OriginLink("macro-subst", self.step, t.origin))
        if t.is_form:
            return self._inst_form(t)
        # template literal atom (introduced: keeps the use scope)
        out = t.clone(OriginLink("macro-template", self.step, t.origin))
        out.mark_self_inplace(self.step, self.scope)
        return out

    def _inst_form(self, t: SObj) -> SObj:
        items = t.items
        out_items: list[SObj] = []
        i = 0
        while i < len(items):
            sub = items[i]
            is_rep = (i + 1 < len(items)
                      and items[i + 1].kind == "id"
                      and items[i + 1].atom == ELLIPSIS)
            if is_rep:
                refs: set[str] = set()
                _referenced_ids(sub, refs)
                reps = [r for r in refs if r in self.binds
                        and isinstance(self.binds[r], list)]
                if not reps:
                    raise MatchFailure("ellipsis without repeat variable")
                rows = len(self.binds[reps[0]])
                for r in reps[1:]:
                    if len(self.binds[r]) != rows:
                        raise MatchFailure("inconsistent ellipsis lengths")
                for row in range(rows):
                    saved = {k: self.binds[k] for k in self.binds}
                    for r in reps:
                        self.binds[r] = self.binds[r][row]
                    try:
                        piece = self._inst(sub)
                    finally:
                        self.binds.clear()
                        self.binds.update(saved)
                    out_items.append(piece)
                i += 2
                continue
            out_items.append(self._inst(sub))
            i += 1
        tail = self._inst(t.tail) if t.tail is not None else None
        if tail is not None and tail.is_form and tail.tail is None:
            out_items.extend(tail.items)
            tail = None
        out = SObj.form(
            out_items, t.span,
            origin=OriginLink("macro-template", self.step),
            tail=tail)
        out.mark_self_inplace(self.step, self.scope)
        return out
