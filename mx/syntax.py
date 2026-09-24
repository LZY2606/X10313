"""Syntax objects: datum + source span + scope set + provenance chain.

A Syn is the central carrier. Scope sets are *per identifier occurrence*;
hygiene is decided by scope-set relations, never by the spelling/name.

sid is stable under replay (deterministic from construction order within a
source/run and an origin discriminator), so exported/re-imported projects keep
identical identities.
"""
from itertools import count

from .errors import FormatError


class IdIssuer:
    def __init__(self, prefix, start=1):
        self.prefix = prefix
        self._c = count(start)
        self._start = start

    def issue(self):
        return f"{self.prefix}{next(self._c)}"

    def fork(self, sub):
        return IdIssuer(f"{self.prefix}{sub}-", self._start)


class Syn:
    __slots__ = ("sid", "kind", "value", "children", "span", "scopes",
                 "origin", "flags", "macro_mark")

    def __init__(self, kind, value=None, children=None, span=None, scopes=None,
                 sid=None, origin=None, flags=None, macro_mark=None):
        self.kind = kind            # id | num | str | bool | list | nil
        self.value = value
        self.children = children or []
        self.span = span
        self.scopes = frozenset(scopes or ())
        self.sid = sid
        self.origin = list(origin or ())   # list of origin records
        self.flags = set(flags or ())      # e.g. introduced, gensym, copied-ctx
        self.macro_mark = macro_mark

    # ---- structural helpers -------------------------------------------------
    def atom(self):
        return self.kind in ("id", "num", "str", "bool", "nil")

    def datum(self):
        if self.kind == "list":
            return [c.datum() for c in self.children]
        if self.kind == "nil":
            return None
        return self.value

    def name(self):
        return self.value if self.kind == "id" else None

    def fresh_copy(self, sid, scopes=None, add_origin=None, flags=()):
        """Shallow re-stamp: same datum, new scopes, extended provenance."""
        s = Syn(self.kind, self.value, self.children, self.span,
                scopes if scopes is not None else self.scopes, sid=sid,
                origin=self.origin + ([add_origin] if add_origin else []),
                flags=set(self.flags) | set(flags), macro_mark=self.macro_mark)
        return s

    def to_dict(self):
        return {
            "sid": self.sid, "kind": self.kind, "value": self.value,
            "children": [c.to_dict() for c in self.children],
            "span": self.span, "scopes": sorted(self.scopes),
            "origin": self.origin, "flags": sorted(self.flags),
            "macro_mark": self.macro_mark,
        }


# ---------------------------------------------------------------------------
# Construction primitives (used both by parser and macro transformers)
# ---------------------------------------------------------------------------
class SynFactory:
    """Deterministic issuer + constructors.

    Original syntax gets sids `s1..`; introduced syntax `i1..`; gensyms g....
    Order is fixed by the replay sequence, not wall-clock time.
    """

    def __init__(self):
        self._orig = IdIssuer("s")
        self._intro = IdIssuer("i")
        self._gensym = IdIssuer("g")
        self._scopes = IdIssuer("sc")
        self.macro_seq = 0

    def new_scope(self):
        return self._scopes.issue()

    def new_macro_mark(self):
        self.macro_seq += 1
        return f"m{self.macro_seq}"

    def mk_original(self, kind, value, children, span):
        return Syn(kind, value=value, children=children, span=span,
                   sid=self._orig.issue())

    def mk_introduced(self, kind, value=None, children=None, ctx=None,
                      origin=None, flags=(), scopes=None):
        sid = self._intro.issue()
        if scopes is None:
            scopes = ctx.scopes if ctx is not None else frozenset()
        s = Syn(kind, value=value, children=children,
                span=ctx.span if ctx is not None else None,
                scopes=scopes, sid=sid, origin=list(origin or ()),
                flags=set(flags) | {"introduced"})
        return s

    def gensym(self, base, ctx=None, origin=None):
        sid = self._gensym.issue()
        # Deterministic generated symbol: readable spelling carries stable
        # suffix, but binding is keyed by scope set + sid, never the spelling.
        name = f"{base}#{sid}"
        s = Syn("id", value=name, span=ctx.span if ctx else None,
                scopes=ctx.scopes if ctx else frozenset(), sid=sid,
                origin=list(origin or ()),
                flags={"introduced", "gensym"}, macro_mark=ctx.macro_mark if ctx else None)
        return s

    def datum_to_syn(self, datum, ctx, origin, mode="copy"):
        """Visible datum->syntax context choice.

        mode=copy: identifiers take ctx's scope set (explicit capture capable).
        mode=intro: identifiers are macro-introduced (hygienic).
        mode=empty: identifiers carry no scopes (free/top-level only).
        """
        if isinstance(datum, Syn):
            scopes = {"copy": ctx.scopes, "empty": frozenset(),
                      "intro": datum.scopes}[mode]
            sid = self._intro.issue()
            return datum.fresh_copy(sid, scopes=scopes,
                                    add_origin={**origin, "mode": mode,
                                                "ctx_sid": ctx.sid},
                                    flags=("datum->syn",))
        if isinstance(datum, list):
            kids = [self.datum_to_syn(x, ctx, origin, mode) for x in datum]
            return self.mk_introduced("list", children=kids, ctx=ctx,
                                      origin=[{**origin, "mode": mode}],
                                      flags=("datum->syn",))
        if datum is None:
            return self.mk_introduced("nil", ctx=ctx, origin=[origin],
                                      flags=("datum->syn",))
        if isinstance(datum, bool):
            return self.mk_introduced("bool", value=datum, ctx=ctx,
                                      origin=[origin], flags=("datum->syn",))
        if isinstance(datum, (int, float)):
            return self.mk_introduced("num", value=datum, ctx=ctx,
                                      origin=[origin], flags=("datum->syn",))
        if isinstance(datum, str):
            # strings in datum position are identifiers
            scopes = {"copy": ctx.scopes, "empty": frozenset(),
                      "intro": frozenset()}[mode]
            return self.mk_introduced("id", value=datum, ctx=ctx,
                                      scopes=scopes, origin=[{**origin,
                                                              "mode": mode}],
                                      flags=("datum->syn",))
        raise FormatError(f"cannot convert datum value {datum!r} to syntax; "
                          f"value left in place with diagnostic")
