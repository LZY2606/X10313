"""Core language engine: lexer, parser, sets-of-scopes macro expansion, binding.

Scope discipline follows Flatt's "sets of scopes" model:
  * every syntax object carries a set of scope labels;
  * macro calls add a fresh scope to macro-introduced syntax (intro) and to the
    original call term (use) -- the two symmetric marks;
  * identifiers bind when their scope sets are related by subset, with the most
    recent scope breaking ties;
  * capture is therefore decided purely by scope-set relations, never by
    renaming strings.

All identifiers are stable and deterministic for a given program and parameter
fingerprint so artifacts can be replayed byte-for-byte.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import ALGORITHM_VERSION

# ---------------------------------------------------------------------------
# Scope sets backed by NumPy bit sets.
# ---------------------------------------------------------------------------

class ScopeSet:
    """An ordered set of scope labels stored as a NumPy uint64 bitset."""

    __slots__ = ("bits", "registry")

    def __init__(self, bits: Optional[np.ndarray] = None, registry: "ScopeRegistry|None" = None):
        self.registry = registry
        self.bits = bits if bits is not None else np.zeros(4, dtype=np.uint64)

    def _grow(self, idx: int) -> None:
        need = idx // 64 + 1
        if self.bits.shape[0] < need:
            padded = np.zeros(need, dtype=np.uint64)
            padded[: self.bits.shape[0]] = self.bits
            self.bits = padded

    def add(self, label: str) -> None:
        idx = self.registry.index(label)
        self._grow(idx)
        self.bits[idx // 64] |= np.uint64(1 << (idx % 64))

    def copy(self) -> "ScopeSet":
        return ScopeSet(self.bits.copy(), self.registry)

    def labels(self) -> list[str]:
        out: list[str] = []
        for word_idx, word in enumerate(self.bits):
            w = int(word)
            while w:
                b = w & -w
                bit = w.bit_length() - 1
                out.append(self.registry.label(word_idx * 64 + bit))
                w ^= b
        out.sort(key=lambda s: self.registry.order(s))
        return out

    def has(self, label: str) -> bool:
        if not self.registry.knows(label):
            return False
        idx = self.registry.index(label)
        if idx // 64 >= self.bits.shape[0]:
            return False
        return bool(self.bits[idx // 64] & np.uint64(1 << (idx % 64)))

    def subset_of(self, other: "ScopeSet") -> bool:
        n = max(self.bits.shape[0], other.bits.shape[0])
        a = np.zeros(n, dtype=np.uint64); a[: self.bits.shape[0]] = self.bits
        b = np.zeros(n, dtype=np.uint64); b[: other.bits.shape[0]] = other.bits
        return bool(np.all((a & ~b) == 0))

    def union(self, other: "ScopeSet") -> "ScopeSet":
        n = max(self.bits.shape[0], other.bits.shape[0])
        a = np.zeros(n, dtype=np.uint64); a[: self.bits.shape[0]] = self.bits
        b = np.zeros(n, dtype=np.uint64); b[: other.bits.shape[0]] = other.bits
        return ScopeSet(a | b, self.registry)

    def difference(self, other: "ScopeSet") -> "ScopeSet":
        n = max(self.bits.shape[0], other.bits.shape[0])
        a = np.zeros(n, dtype=np.uint64); a[: self.bits.shape[0]] = self.bits
        b = np.zeros(n, dtype=np.uint64); b[: other.bits.shape[0]] = other.bits
        return ScopeSet(a & ~b, self.registry)

    def signature(self) -> str:
        return "+".join(self.labels()) or "(empty)"


class ScopeRegistry:
    """Stable assignment of scope labels to bit indices in creation order."""

    def __init__(self) -> None:
        self._label_to_idx: dict[str, int] = {}
        self._idx_to_label: list[str] = []

    def index(self, label: str) -> int:
        if label not in self._label_to_idx:
            self._label_to_idx[label] = len(self._idx_to_label)
            self._idx_to_label.append(label)
        return self._label_to_idx[label]

    def knows(self, label: str) -> bool:
        return label in self._label_to_idx

    def order(self, label: str) -> int:
        return self._label_to_idx[label]

    def label(self, idx: int) -> str:
        return self._idx_to_label[idx]

    def empty(self) -> ScopeSet:
        return ScopeSet(registry=self)


# ---------------------------------------------------------------------------
# Tokens, spans and syntax objects.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    source_id: str
    start: int
    end: int
    line: int
    col: int

    def to_dict(self) -> dict:
        return {"source_id": self.source_id, "start": self.start, "end": self.end,
                "line": self.line, "col": self.col}


@dataclass(frozen=True)
class Token:
    kind: str          # id | number | boolean | lparen | rparen | quote | comma | comma-at
    text: str
    span: Span
    original: str      # exact original spelling (never normalised away)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "span": self.span.to_dict(),
                "original": self.original}


@dataclass(frozen=True)
class OriginStep:
    """One hop in a provenance chain."""
    kind: str            # parse | template | splice | datum->syntax | gensym | expansion
    call_id: Optional[str]
    detail: str
    span: Optional[dict] = None
    parent_sid: Optional[str] = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "call_id": self.call_id, "detail": self.detail,
                "span": self.span, "parent_sid": self.parent_sid}


class Syntax:
    """A syntax object: atom (value) or list, with scopes + provenance."""

    def __init__(self, value, span: Optional[Span], sid: str,
                 scopes: ScopeSet, origin: list[OriginStep] | None = None,
                 *, is_list: bool = False, intro_tags: tuple[str, ...] = (),
                 external: tuple[str, ...] = ()):
        self.value = value
        self.span = span
        self.sid = sid
        self.scopes = scopes
        self.origin = origin or []
        self.is_list = is_list
        # macro-call ids whose intro mark this node carries from template build
        self.intro_tags = intro_tags
        # call ids for which this node was classified external (datum->syntax /
        # splice / gensym), driving symmetric mark removal visibility
        self.external = external

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        d: dict[str, Any] = {"sid": self.sid, "kind": "list" if self.is_list else "atom",
                             "scopes": self.scopes.labels(),
                             "scope_bits": scope_bits_sig(self.scopes),
                             "origin": [o.to_dict() for o in self.origin],
                             "intro_tags": list(self.intro_tags),
                             "external": list(self.external)}
        if self.span is not None:
            d["span"] = self.span.to_dict()
        if self.is_list:
            d["items"] = [c.to_dict() for c in self.value]
        else:
            d["atom"] = self.value
            if self.span is not None:
                d["original"] = self.span_text_original()
        return d

    def span_text_original(self) -> str | None:
        return self.span and getattr(self, "_original", None)

    # -- helpers -----------------------------------------------------------
    def atom_symbol(self) -> Optional[str]:
        return None if self.is_list else self.value

    def deep_copy(self, sid_fn, origin: OriginStep | None = None,
                  keep_external: bool = True) -> "Syntax":
        new_origin = list(self.origin)
        if origin is not None:
            new_origin.append(origin)
        if self.is_list:
            return Syntax([c.deep_copy(sid_fn, origin, keep_external) for c in self.value],
                          self.span, sid_fn(), self.scopes.copy(), new_origin,
                          is_list=True, intro_tags=self.intro_tags,
                          external=self.external if keep_external else ())
        s = Syntax(self.value, self.span, sid_fn(), self.scopes.copy(), new_origin,
                   is_list=False, intro_tags=self.intro_tags,
                   external=self.external if keep_external else ())
        if self.span is not None:
            s._original = getattr(self, "_original", None)
        return s


def scope_bits_sig(scopes: ScopeSet) -> str:
    words = [hex(int(w))[2:] for w in scopes.bits]
    return "|".join(words)


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class LexError(Exception):
    def __init__(self, message: str, source_id: str, start: int, line: int, col: int):
        super().__init__(message)
        self.message = message
        self.source_id = source_id
        self.start = start
        self.line = line
        self.col = col

    def to_dict(self) -> dict:
        return {"code": "E_LEX", "message": self.message,
                "span": Span(self.source_id, self.start, self.start, self.line, self.col).to_dict()}


_SIMPLE = {"(": "lparen", ")": "rparen"}


def _token_kind(text: str) -> str:
    if text == "#t" or text == "#f":
        return "boolean"
    try:
        float(text)
        return "number"
    except ValueError:
        return "id"


def lex(text: str, source_id: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, line_start = 0, 1, 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r":
            i += 1
            continue
        if ch == "\n":
            line += 1
            i += 1
            line_start = i
            continue
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        col = i - line_start + 1
        start = i
        if ch in "()":
            tokens.append(Token(_SIMPLE[ch], ch,
                                Span(source_id, i, i + 1, line, col), ch))
            i += 1
            continue
        if ch == "'":
            tokens.append(Token("quote", "'", Span(source_id, i, i + 1, line, col), "'"))
            i += 1
            continue
        if ch == ",":
            if i + 1 < n and text[i + 1] == "@":
                tokens.append(Token("comma-at", ",@",
                                    Span(source_id, i, i + 2, line, col), ",@"))
                i += 2
            else:
                tokens.append(Token("comma", ",",
                                    Span(source_id, i, i + 1, line, col), ","))
                i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                else:
                    if text[j] == "\n":
                        line += 1
                        line_start = j + 1
                    j += 1
            if j >= n:
                raise LexError("unterminated string literal", source_id, start, line, col)
            raw = text[i:j + 1]
            tokens.append(Token("string", raw, Span(source_id, i, j + 1, line, col), raw))
            i = j + 1
            continue
        if ch in "[]{}":
            raise LexError(f"unexpected character {ch!r}; s-expressions use () only",
                           source_id, i, line, col)
        # generic atom
        j = i
        while j < n and text[j] not in " \t\r\n();',\"":
            j += 1
        raw = text[i:j]
        tokens.append(Token(_token_kind(raw), raw, Span(source_id, i, j, line, col), raw))
        i = j
    return tokens


# ---------------------------------------------------------------------------
# Parser (tokens -> syntax objects; base scopes applied later by the expander)
# ---------------------------------------------------------------------------

@dataclass
class ParseDiagnostic:
    code: str
    message: str
    span: Optional[Span]
    expected: Optional[str] = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "span": self.span.to_dict() if self.span else None,
                "expected": self.expected}


def parse_sources(sources: dict[str, str], registry: ScopeRegistry,
                  prefix: str) -> tuple[dict[str, list[Syntax]], dict[str, list[Token]],
                                        list[ParseDiagnostic]]:
    """Parse each source independently; return per-source top-level forms."""
    forms_by_source: dict[str, list[Syntax]] = {}
    tokens_by_source: dict[str, list[Token]] = {}
    diags: list[ParseDiagnostic] = []
    counter = [0]

    def fresh_sid() -> str:
        counter[0] += 1
        return f"{prefix}p{counter[0]}"

    for source_id, text in sources.items():
        toks = lex(text, source_id)
        tokens_by_source[source_id] = toks
        pos = [0]

        def tok_at(k: int) -> Token | None:
            return toks[k] if k < len(toks) else None

        def parse_form(depth: int) -> Syntax | None:
            t = tok_at(pos[0])
            if t is None:
                return None
            if t.kind == "rparen":
                diags.append(ParseDiagnostic("E_PARSE", "unexpected ')'", t.span))
                pos[0] += 1
                return None
            if t.kind == "lparen":
                return parse_list(depth)
            if t.kind in ("comma", "comma-at"):
                diags.append(ParseDiagnostic(
                    "E_PARSE", f"{t.text!r} may only appear inside a macro template", t.span))
                pos[0] += 1
                return None
            return parse_atom()

        def parse_atom() -> Syntax:
            t = toks[pos[0]]
            pos[0] += 1
            val: Any
            if t.kind == "boolean":
                val = t.text == "#t"
            elif t.kind == "number":
                val = float(t.text) if "." in t.text else int(t.text)
            elif t.kind == "string":
                val = _decode_string(t.text)
            else:
                val = t.text
            s = Syntax(val, t.span, fresh_sid(), registry.empty(),
                       [OriginStep("parse", None, f"parsed atom {t.text!r}", t.span.to_dict())])
            s._original = t.original
            return s

        def parse_list(depth: int) -> Syntax:
            open_tok = toks[pos[0]]
            pos[0] += 1
            items: list[Syntax] = []
            while True:
                t = tok_at(pos[0])
                if t is None:
                    diags.append(ParseDiagnostic(
                        "E_PARSE", "missing ')' (unexpected end of input)",
                        open_tok.span, expected=")"))
                    break
                if t.kind == "rparen":
                    pos[0] += 1
                    break
                if t.kind == "quote":
                    items.append(parse_quote())
                elif t.kind in ("comma", "comma-at"):
                    pos[0] += 1
                    cs = Syntax(t.text, t.span, fresh_sid(), registry.empty(),
                                [OriginStep("parse", None,
                                            f"parsed template marker {t.text!r}",
                                            t.span.to_dict())])
                    cs._original = t.original
                    items.append(cs)
                else:
                    f = parse_form(depth + 1)
                    if f is not None:
                        items.append(f)
            span = Span(open_tok.span.source_id, open_tok.span.start,
                        toks[pos[0] - 1].span.end if items or pos[0] > 1 else open_tok.span.end,
                        open_tok.span.line, open_tok.span.col)
            return Syntax(items, span, fresh_sid(), registry.empty(),
                          [OriginStep("parse", None, "parsed list", span.to_dict())],
                          is_list=True)

        def parse_quote() -> Syntax:
            qt = toks[pos[0]]
            pos[0] += 1
            quoted = parse_form(depth + 1)
            if quoted is None:
                diags.append(ParseDiagnostic("E_PARSE", "quote missing an expression",
                                             qt.span, expected="expression"))
                quoted = Syntax("__quote_error__", qt.span, fresh_sid(),
                                registry.empty(), is_list=False)
            q = Syntax("quote", qt.span, fresh_sid(), registry.empty(),
                       [OriginStep("parse", None, "parsed ' as quote", qt.span.to_dict())])
            q._original = "'"
            span = Span(qt.span.source_id, qt.span.start,
                        quoted.span.end if quoted.span else qt.span.end,
                        qt.span.line, qt.span.col)
            return Syntax([q, quoted], span, fresh_sid(), registry.empty(),
                          [OriginStep("parse", None, "parsed quoted datum", span.to_dict())],
                          is_list=True)

        forms: list[Syntax] = []
        while pos[0] < len(toks):
            f = parse_form(0)
            if f is not None:
                forms.append(f)
        forms_by_source[source_id] = forms
    return forms_by_source, tokens_by_source, diags


def _decode_string(raw: str) -> str:
    body = raw[1:-1]
    out: list[str] = []
    k = 0
    while k < len(body):
        c = body[k]
        if c == "\\" and k + 1 < len(body):
            esc = body[k + 1]
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(esc, esc))
            k += 2
        else:
            out.append(c)
            k += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Macro expander (sets-of-scopes, with provenance + partial expansion graph)
# ---------------------------------------------------------------------------

CORE_FORMS = {"lambda", "let", "define", "defmacro", "if", "begin", "set!", "quote"}
PRIMITIVES = {"+", "-", "*", "=", "<", ">", "not", "cons", "car", "cdr", "list"}
META_HEAD = {"datum->syntax", "gensym", "quote-syntax", "bound-identifier=?",
             "free-identifier=?", "%let", "%if"}


@dataclass
class MacroDef:
    name: str
    params: list[str]
    template: Syntax
    def_scopes: ScopeSet
    context_syntax: Syntax
    sid: str
    source_id: str
    meta_defs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Binder:
    symbol: str
    sid: str
    kind: str            # lambda | let | define | primitive | macro
    scopes: ScopeSet
    span: Optional[Span]
    origin_sid: str

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "sid": self.sid, "kind": self.kind,
                "scopes": self.scopes.labels(), "scope_bits": scope_bits_sig(self.scopes),
                "span": self.span.to_dict() if self.span else None,
                "origin_sid": self.origin_sid, "scope_order": max_scope_order(self.scopes)}


@dataclass
class BindingEdge:
    use_sid: str
    binder_sid: str | None
    symbol: str
    status: str          # bound | unbound | core | shadowed-candidate
    candidates: list[dict]
    reason: str
    span: Optional[dict]
    origin_chain: list[dict]

    def to_dict(self) -> dict:
        return {"use_sid": self.use_sid, "binder_sid": self.binder_sid,
                "symbol": self.symbol, "status": self.status,
                "candidates": self.candidates, "reason": self.reason,
                "span": self.span, "origin_chain": self.origin_chain}


@dataclass
class Diagnostic:
    code: str
    message: str
    severity: str
    span: Optional[Span]
    sid: str | None
    origin_chain: list[dict]
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "severity": self.severity,
                "span": self.span.to_dict() if self.span else None, "sid": self.sid,
                "origin_chain": self.origin_chain, "detail": self.detail}


def max_scope_order(scopes: ScopeSet) -> int:
    orders = [scopes.registry.order(l) for l in scopes.labels()]
    return max(orders) if orders else -1


class ExpandError(Exception):
    def __init__(self, code: str, message: str, node: Optional[Syntax],
                 detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.node = node
        self.detail = detail or {}


class _Rng:
    """Deterministic LCG seeded per run (fixed seed in tests)."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state


class Expander:
    def __init__(self, forms_by_source, tokens_by_source, parse_diags,
                 registry: ScopeRegistry, params: dict, source_order: list[str]):
        self.forms_by_source = forms_by_source
        self.tokens_by_source = tokens_by_source
        self.registry = registry
        self.params = params
        self.source_order = source_order
        self.budget = int(params.get("recursion_budget", 64))
        self.budget_initial = self.budget
        self.rng = _Rng(int(params.get("gensym_seed", 20260924)))
        self.gensym_counter = 0
        self.id_counter = 0
        self.call_counter = 0
        self.steps: list[dict] = []
        self.calls: list[dict] = []
        self.binders: list[Binder] = []
        self.macros: dict[str, MacroDef] = {}
        self.binder_by_sid: dict[str, Binder] = {}
        self.diagnostics: list[Diagnostic] = [
            Diagnostic(d.code, d.message, "error", d.span, None, [],
                       {"expected": d.expected} if d.expected else {})
            for d in parse_diags
        ]
        self.expanded_roots: list[dict] = []
        self.scope_notes: list[dict] = []
        self.budget_exceeded = False
        self._call_stack: list[str] = []

    # -- identity ----------------------------------------------------------
    def fresh_sid(self) -> str:
        self.id_counter += 1
        return f"g{self.id_counter}"

    def add_note(self, scope: str, role: str, sid: str, call_id: str | None, detail: str) -> None:
        self.scope_notes.append({"scope": scope, "role": role, "sid": sid,
                                 "call_id": call_id, "detail": detail})

    # -- macro resolution --------------------------------------------------
    def resolve_macro(self, sym: str, scopes: ScopeSet):
        macro = self.macros.get(sym)
        if macro is None:
            return None
        if scopes.has(f"def:{macro.sid}") or macro.def_scopes.subset_of(scopes):
            return macro
        return macro  # macros are project-global once defined; subset recorded for display

    # -- template meta-evaluation -----------------------------------------
    def eval_meta(self, node: Syntax, env: dict[str, Syntax], macro: MacroDef, call_id: str):
        if node.is_list:
            head_sym = node.value[0].atom_symbol() if node.value else None
            args = node.value[1:]
            if head_sym == "%if":
                cond = self.meta_truthy(self.eval_meta(args[0], env, macro, call_id))
                branch = args[1] if cond else args[2]
                return self.eval_meta(branch, env, macro, call_id)
            if head_sym == "%let":
                new_env = dict(env)
                bindings = args[0].value
                for pair in bindings:
                    name = pair.value[0].value
                    val = self.eval_meta(pair.value[1], env, macro, call_id)
                    new_env[name] = val
                result = None
                for body in args[1:]:
                    result = self.eval_meta(body, new_env, macro, call_id)
                return result
            if head_sym == "quote":
                return args[0]
            raise ExpandError("E_META", f"unknown meta form {head_sym!r}", node)
        sym = node.atom_symbol()
        if isinstance(node.value, str):
            if sym in env:
                return env[sym]
            if sym in macro.meta_defs:
                return macro.meta_defs[sym]
            if sym in PRIMITIVES or sym in CORE_FORMS:
                return node
            raise ExpandError("E_META_UNBOUND",
                              f"template variable {sym!r} is not a macro parameter", node)
        return node

    @staticmethod
    def meta_truthy(v) -> bool:
        if isinstance(v, Syntax):
            if v.is_list:
                return bool(v.value)
            return v.value is not False and v.value is not None
        return bool(v)

    def gen_symbol(self, base: str, call_id: str, macro: MacroDef) -> Syntax:
        self.gensym_counter += 1
        r = self.rng.next_u32()
        name = f"{base}__g{self.gensym_counter:03d}_{r:08x}"
        scopes = macro.def_scopes.copy()
        s = Syntax(name, None, self.fresh_sid(), scopes,
                   [OriginStep("gensym", call_id,
                               f"generated temporary {name!r} from base {base!r}",
                               None)],
                   external=(call_id,))
        self.add_note(f"gensym:{self.gensym_counter}", "generated-symbol", s.sid,
                      call_id, f"{base} -> {name}")
        return s

    def build_template(self, tmpl: Syntax, env: dict[str, Syntax], macro: MacroDef,
                       call_id: str, context_syntax: Syntax | None) -> Syntax:
        if tmpl.is_list:
            items = tmpl.value
            head_sym = items[0].atom_symbol() if items else None
            if head_sym == "datum->syntax":
                rest = items[1:]
                if len(rest) == 2:
                    ctx_expr, datum_expr = rest
                elif len(rest) == 1:
                    ctx_expr = None
                    datum_expr = rest[0]
                else:
                    raise ExpandError("E_TEMPLATE",
                                      "datum->syntax expects 1 or 2 arguments", node)
                if ctx_expr is None or (not ctx_expr.is_list
                                        and ctx_expr.atom_symbol() == "macro-use-context"):
                    target_scopes = self._use_context(env, macro, call_id)
                    ctx_desc = "macro-use-context (use-site)"
                else:
                    ctx = self.eval_meta(ctx_expr, env, macro, call_id)
                    target_scopes = ctx.scopes.copy()
                    ctx_desc = f"context identifier {ctx.value!r}"
                raw = self.eval_meta(datum_expr, env, macro, call_id)
                built = self._strip_and_reattach(raw, target_scopes, call_id, ctx_desc)
                return built
            if head_sym == "gensym":
                base_node = self.eval_meta(items[1], env, macro, call_id)
                return self.gen_symbol(str(base_node.value), call_id, macro)
            if head_sym == "quote-syntax":
                raw = items[1]
                return self._strip_and_reattach(raw, macro.def_scopes.copy(), call_id,
                                                "quote-syntax (definition-site)")
            if head_sym in ("bound-identifier=?", "free-identifier=?"):
                a = self.eval_meta(items[1], env, macro, call_id)
                b = self.eval_meta(items[2], env, macro, call_id)
                same = self.scope_equal(a.scopes, b.scopes)
                if head_sym == "free-identifier=?":
                    ba = self.resolve_binder(a.value, a.scopes) if isinstance(a.value, str) else None
                    bb = self.resolve_binder(b.value, b.scopes) if isinstance(b.value, str) else None
                    same = bool(ba and bb and ba.sid == bb.sid)
                return Syntax(bool(same), None, self.fresh_sid(),
                              self.registry.empty(),
                              [OriginStep("template", call_id,
                                          f"{head_sym} -> {same}", None)])
            if head_sym in ("%if", "%let"):
                return self.eval_meta(tmpl, env, macro, call_id)
            out: list[Syntax] = []
            k = 0
            while k < len(items):
                child = items[k]
                if not child.is_list and child.value == ",":
                    if k + 1 >= len(items):
                        raise ExpandError("E_TEMPLATE", "comma without expression", child)
                    val = self.eval_meta(items[k + 1], env, macro, call_id)
                    out.append(self._splice_copy(val, call_id))
                    k += 2
                    continue
                if child.is_list and child.value and child.value[0].value == ",@":
                    val = self.eval_meta(child.value[1], env, macro, call_id)
                    self._splice(val, out, macro, call_id, child)
                else:
                    out.append(self.build_template(child, env, macro, call_id, context_syntax))
                k += 1
            span = tmpl.span
            return Syntax(out, span, self.fresh_sid(), tmpl.scopes.copy(),
                          list(tmpl.origin) + [OriginStep(
                              "template", call_id, "copied template list",
                              tmpl.span.to_dict() if tmpl.span else None)],
                          is_list=True)
        # atom: literal from definition site, or unquote handled by caller
        return self._clone_def_atom(tmpl, macro, call_id)

    def _splice_copy(self, val, call_id: str) -> Syntax:
        if not isinstance(val, Syntax):
            raise ExpandError("E_TEMPLATE", "unquote did not produce syntax", None)
        return val.deep_copy(
            self.fresh_sid,
            OriginStep("splice", call_id, "substituted macro argument",
                       val.span.to_dict() if val.span else None))

    def _clone_def_atom(self, tmpl: Syntax, macro: MacroDef, call_id: str) -> Syntax:
        s = Syntax(tmpl.value, tmpl.span, self.fresh_sid(), macro.def_scopes.copy(),
                   list(tmpl.origin) + [OriginStep("template", call_id,
                                                   f"definition-site literal {tmpl.value!r}",
                                                   tmpl.span.to_dict() if tmpl.span else None)])
        if tmpl.span is not None:
            s._original = getattr(tmpl, "_original", None)
        return s

    @staticmethod
    def scope_equal(a: ScopeSet, b: ScopeSet) -> bool:
        n = max(a.bits.shape[0], b.bits.shape[0])
        x = np.zeros(n, dtype=np.uint64); x[: a.bits.shape[0]] = a.bits
        y = np.zeros(n, dtype=np.uint64); y[: b.bits.shape[0]] = b.bits
        return bool(np.array_equal(x, y))

    def _use_context(self, env, macro: MacroDef, call_id: str) -> ScopeSet:
        # use-site context = lexical scopes of the call term WITHOUT this call's
        # transient use/intro marks, so a captured identifier lands in exactly
        # the caller's lexical environment
        def clean(sc: ScopeSet) -> ScopeSet:
            out = self.registry.empty()
            for label in sc.labels():
                if label.startswith(("use:", "intro:")):
                    continue
                out.add(label)
            return out
        for v in env.values():
            node = v[0] if isinstance(v, list) and v else v
            if isinstance(node, Syntax) and node.span is not None:
                return clean(node.scopes)
        return clean(macro.context_syntax.scopes)

    def _splice(self, val, out, macro: MacroDef, call_id: str, marker: Syntax) -> None:
        if isinstance(val, Syntax) and val.is_list:
            items = val.value
        elif isinstance(val, list):
            items = val
        else:
            raise ExpandError("E_SPLICE", ",@ requires a list", marker)
        for elt in items:
            copy = elt.deep_copy(self.fresh_sid,
                                 OriginStep("splice", call_id,
                                            "spliced macro argument",
                                            elt.span.to_dict() if elt.span else None))
            copy.external = tuple(dict.fromkeys(copy.external + (call_id,)))
            out.append(copy)

    def _strip_and_reattach(self, raw: Syntax, target_scopes: ScopeSet, call_id: str,
                            ctx_desc: str) -> Syntax:
        def walk(n: Syntax) -> Syntax:
            s = Syntax(n.value, n.span, self.fresh_sid(), target_scopes.copy(),
                       list(n.origin) + [OriginStep(
                           "datum->syntax", call_id,
                           f"context chosen via {ctx_desc}",
                           n.span.to_dict() if n.span else None)],
                       is_list=n.is_list,
                       external=tuple(dict.fromkeys(n.external + (call_id,))))
            if n.is_list:
                s.value = [walk(c) for c in n.value]
            elif n.span is not None:
                s._original = getattr(n, "_original", None)
            return s
        return walk(raw)

    # -- marks -------------------------------------------------------------
    def mark_tree(self, node: Syntax, scope: str, call_id: str, role: str) -> None:
        node.scopes.add(scope)
        if role == "intro" and scope not in node.intro_tags:
            node.intro_tags = node.intro_tags + (scope,)
        self.add_note(scope, role, node.sid, call_id,
                      f"{role} mark on {'list' if node.is_list else node.value!r}")
        if node.is_list:
            for c in node.value:
                self.mark_tree(c, scope, call_id, role)

    def prune_intro(self, node: Syntax, call_id: str, intro_scope: str,
                    use_scope: str) -> None:
        """Remove this call's intro mark from external nodes; the use mark is
        dropped from macro-introduced nodes so hygiene is symmetric."""
        if call_id in node.external:
            node.scopes = node.scopes.difference(self._singleton(intro_scope))
        else:
            node.scopes = node.scopes.difference(self._singleton(use_scope))
        if node.is_list:
            for c in node.value:
                self.prune_intro(c, call_id, intro_scope, use_scope)

    def _singleton(self, label: str) -> ScopeSet:
        s = self.registry.empty()
        s.add(label)
        return s

    # -- expansion driver --------------------------------------------------
    def run(self) -> dict:
        module_scope = "module"
        self.registry.index(module_scope)
        for sid_source in self.source_order:
            for form in self.forms_by_source[sid_source]:
                self._add_scope_tree(form, module_scope)
        for name in sorted(PRIMITIVES) + sorted(CORE_FORMS):
            kind = "primitive" if name in PRIMITIVES else "core"
            sc = self.registry.empty()
            sc.add(module_scope)
            b = Binder(name, f"prim::{name}", kind, sc, None, f"prim::{name}")
            self.binders.append(b)
            self.binder_by_sid[b.sid] = b

        for source_id in self.source_order:
            for form in self.forms_by_source[source_id]:
                try:
                    expanded = self.expand(form, self.registry.empty(), [])
                    self.expanded_roots.append(expanded.to_dict())
                except ExpandError as exc:
                    self.diagnostics.append(self._diag(exc))
                    self.expanded_roots.append(self._partial(form, exc))
                except RecursionError:
                    exc = ExpandError("E_RECURSION", "recursion limit reached", form)
                    self.diagnostics.append(self._diag(exc))
                    self.expanded_roots.append(self._partial(form, exc))

        self.resolve_uses()
        return self.payload()

    def _add_scope_tree(self, node: Syntax, scope: str) -> None:
        node.scopes.add(scope)
        if node.is_list:
            for c in node.value:
                self._add_scope_tree(c, scope)

    def _diag(self, exc: ExpandError) -> Diagnostic:
        node = exc.node
        return Diagnostic(exc.code, exc.message, "error",
                          node.span if node else None,
                          node.sid if node else None,
                          [o.to_dict() for o in node.origin] if node else [],
                          exc.detail)

    def _partial(self, node: Syntax, exc: ExpandError) -> dict:
        d = node.to_dict()
        d["partial_error"] = {"code": exc.code, "message": exc.message}
        return d

    def new_scope(self, kind: str, sid: str) -> str:
        label = f"{kind}:{sid}"
        self.registry.index(label)
        return label

    def expand(self, node: Syntax, local_scopes: ScopeSet, stack: list[str]) -> Syntax:
        if not node.is_list:
            return node
        items = node.value
        if not items:
            return node
        head = items[0]
        sym = head.atom_symbol()

        if sym == "defmacro":
            return self.expand_defmacro(items, node)
        if sym == "define":
            return self.expand_define(items, node)
        if sym == "lambda":
            return self.expand_lambda(items, node, stack)
        if sym == "let":
            return self.expand_let(items, node, stack)
        if sym in ("if", "set!", "begin", "quote"):
            return self.expand_ordinary(items, node, stack)

        macro = self.macros.get(sym) if isinstance(sym, str) else None
        if macro is not None:
            return self.expand_macro_call(node, macro, stack)
        # application or unknown form: expand children first, re-check macros
        expanded_items = [self.expand(c, local_scopes, stack) for c in items]
        expanded = Syntax(expanded_items, node.span, node.sid, node.scopes.copy(),
                          list(node.origin), is_list=True,
                          intro_tags=node.intro_tags, external=node.external)
        h = expanded_items[0]
        hsym = h.atom_symbol()
        if isinstance(hsym, str) and self.macros.get(hsym) is not None:
            return self.expand_macro_call(expanded, self.macros[hsym], stack)
        if isinstance(hsym, str) and hsym not in CORE_FORMS and hsym not in PRIMITIVES:
            binder = self.resolve_binder(hsym, h.scopes)
            if binder is None:
                # recorded as an unbound *function* later by resolve_uses
                pass
        if expanded_items and expanded_items[0].is_list:
            callee = expanded_items[0]
            if callee.value and callee.value[0].atom_symbol() == "lambda":
                self.check_lambda_arity(callee, expanded_items[1:], node)
        return expanded

    def expand_defmacro(self, items, node) -> Syntax:
        name = items[1].value
        params = [p.value for p in items[2].value]
        template = items[3]
        sid = node.sid
        def_scope = self.new_scope("def", sid)
        self._add_scope_tree(template, def_scope)
        # template also carries module scope so core forms resolve
        def_scopes = template.scopes.copy()
        ctx = items[1]
        macro = MacroDef(name, params, template, def_scopes, ctx, sid,
                         node.span.source_id if node.span else "")
        self.macros[name] = macro
        sc = def_scopes.copy()
        binder = Binder(name, f"macro::{sid}", "macro", sc,
                        items[1].span, sid)
        self.binders.append(binder)
        self.binder_by_sid[binder.sid] = binder
        self.steps.append({"kind": "defmacro", "sid": sid, "name": name,
                           "params": params, "scope": def_scope,
                           "def_scopes": def_scopes.labels(),
                           "span": items[1].span.to_dict()})
        return Syntax("__defmacro__", node.span, node.sid, node.scopes.copy(),
                      list(node.origin) + [OriginStep(
                          "expansion", None, f"registered macro {name!r}",
                          node.span.to_dict() if node.span else None)])

    def expand_define(self, items, node) -> Syntax:
        target = items[1]
        value = self.expand(items[2], self.registry.empty(), [])
        sid = target.sid
        visible = target.scopes.copy()
        binder = Binder(target.value, f"def::{sid}", "define", visible, target.span, sid)
        self.binders.append(binder)
        self.binder_by_sid[binder.sid] = binder
        self.steps.append({"kind": "define", "sid": sid, "name": target.value,
                           "scope": visible.signature(), "span": target.span.to_dict()})
        return Syntax(["__define__", target, value], node.span, node.sid,
                      node.scopes.copy(), list(node.origin) + [OriginStep(
                          "expansion", None,
                          f"top-level binding {target.value!r} introduced",
                          node.span.to_dict() if node.span else None)], is_list=True)

    def expand_lambda(self, items, node, stack) -> Syntax:
        params_node = items[1]
        body = items[2:]
        param_scope = self.new_scope("lambda", params_node.sid)
        new_params: list[Syntax] = []
        for p in params_node.value:
            self._add_scope_tree(p, param_scope)
            b = Binder(p.value, f"lam::{p.sid}", "lambda", p.scopes.copy(), p.span, p.sid)
            self.binders.append(b)
            self.binder_by_sid[b.sid] = b
            new_params.append(p)
        expanded_body = [self.expand(b, self.registry.empty(), stack) for b in body]
        # push binder scope over the fully expanded body so generated and
        # use-site body identifiers both see the parameters, without changing
        # which nodes carry macro intro tags
        for eb in expanded_body:
            self._add_scope_tree(eb, param_scope)
        wrapped = expanded_body[0] if len(expanded_body) == 1 else Syntax(
            expanded_body, node.span, self.fresh_sid(), node.scopes.copy(),
            [OriginStep("expansion", None, "wrapped multi-expression body in begin",
                        node.span.to_dict() if node.span else None)], is_list=True)
        self.steps.append({"kind": "lambda", "sid": params_node.sid,
                           "params": [p.value for p in new_params], "scope": param_scope})
        return Syntax(["lambda", Syntax(new_params, params_node.span, params_node.sid,
                                        params_node.scopes.copy(),
                                        list(params_node.origin), is_list=True),
                       wrapped], node.span, node.sid, node.scopes.copy(),
                      list(node.origin), is_list=True)

    def expand_let(self, items, node, stack) -> Syntax:
        bindings = items[1]
        body = items[2:]
        scope = self.new_scope("let", bindings.sid)
        new_pairs: list[Syntax] = []
        for pair in bindings.value:
            bind_node = pair.value[0]
            val = self.expand(pair.value[1], self.registry.empty(), stack)
            self._add_scope_tree(bind_node, scope)
            b = Binder(bind_node.value, f"let::{bind_node.sid}", "let",
                       bind_node.scopes.copy(), bind_node.span, bind_node.sid)
            self.binders.append(b)
            self.binder_by_sid[b.sid] = b
            new_pairs.append(Syntax([bind_node, val], pair.span, pair.sid,
                                    pair.scopes.copy(), list(pair.origin), is_list=True))
        expanded_body = [self.expand(b, self.registry.empty(), stack) for b in body]
        for eb in expanded_body:
            self._add_scope_tree(eb, scope)
        wrapped = expanded_body[0] if len(expanded_body) == 1 else Syntax(
            expanded_body, node.span, self.fresh_sid(), node.scopes.copy(),
            [OriginStep("expansion", None, "wrapped multi-expression body in begin",
                        node.span.to_dict() if node.span else None)], is_list=True)
        self.steps.append({"kind": "let", "sid": bindings.sid,
                          "binders": [p.value[0].value for p in new_pairs], "scope": scope})
        return Syntax(["let", Syntax(new_pairs, bindings.span, bindings.sid,
                                     bindings.scopes.copy(), list(bindings.origin),
                                     is_list=True), wrapped],
                      node.span, node.sid, node.scopes.copy(), list(node.origin),
                      is_list=True)

    def expand_ordinary(self, items, node, stack) -> Syntax:
        new_items = [items[0]] + [self.expand(c, self.registry.empty(), stack)
                                  for c in items[1:]]
        return Syntax(new_items, node.span, node.sid, node.scopes.copy(),
                      list(node.origin), is_list=True)

    def check_lambda_arity(self, callee: Syntax, args: list[Syntax], call_node: Syntax) -> None:
        params = callee.value[1]
        arg_count = len(args)
        param_count = len(params.value)
        if arg_count != param_count:
            self.diagnostics.append(Diagnostic(
                "E_ARITY",
                f"lambda expects {param_count} argument(s) but received {arg_count}",
                "error", call_node.span, call_node.sid,
                [o.to_dict() for o in call_node.origin],
                {"expected": param_count, "received": arg_count}))

    def expand_macro_call(self, node: Syntax, macro: MacroDef, stack: list[str]) -> Syntax:
        if self.budget <= 0:
            self.budget_exceeded = True
            raise ExpandError("E_BUDGET",
                              f"recursion budget {self.budget_initial} exhausted while expanding "
                              f"{macro.name!r}; partial expansion graph retained", node,
                              {"budget": self.budget_initial, "macro": macro.name,
                               "call_stack": list(self._call_stack)})
        self.budget -= 1
        self.call_counter += 1
        call_id = f"call{self.call_counter:03d}"
        self._call_stack.append(call_id)
        intro_scope = f"intro:{call_id}"
        use_scope = f"use:{call_id}"
        self.registry.index(intro_scope)
        self.registry.index(use_scope)

        args = node.value[1:]
        if len(args) != len(macro.params):
            raise ExpandError("E_MACRO_ARITY",
                              f"macro {macro.name!r} expects {len(macro.params)} argument(s) "
                              f"but received {len(args)}", node,
                              {"expected": len(macro.params), "received": len(args)})

        # symmetric marks: original call term receives the use mark
        self.mark_tree(node, use_scope, call_id, "use")

        # evaluate (but do not expand) arguments in the template environment
        env: dict[str, Syntax] = {}
        for pname, arg in zip(macro.params, args):
            env[pname] = arg

        self.calls.append({
            "call_id": call_id, "macro": macro.name, "stack": list(stack),
            "site_span": node.span.to_dict() if node.span else None,
            "site_sid": node.sid,
            "definition_sid": macro.sid,
            "definition_scopes": macro.def_scopes.labels(),
            "intro_scope": intro_scope, "use_scope": use_scope,
            "budget_remaining": self.budget,
        })

        built = self.build_template(macro.template, env, macro, call_id,
                                    macro.context_syntax)
        # intro mark on everything the macro introduced
        self.mark_tree(built, intro_scope, call_id, "intro")
        # symmetric pruning per node origin classification
        self.prune_intro(built, call_id, intro_scope, use_scope)
        self.steps.append({"kind": "macro-call", "call_id": call_id,
                           "macro": macro.name, "sid": node.sid,
                           "output_sid": built.sid,
                           "intro_scope": intro_scope, "use_scope": use_scope,
                           "capture_rule": "scope-subset",
                           "context_choices": self._context_choices_snapshot(call_id)})
        origin = OriginStep("expansion", call_id,
                            f"macro {macro.name!r} expanded (intro={intro_scope}, "
                            f"use={use_scope})",
                            node.span.to_dict() if node.span else None, node.sid)
        built.origin.append(origin)

        result = self.expand(built, self.registry.empty(), stack + [call_id])
        self._call_stack.pop()
        return result

    def _context_choices_snapshot(self, call_id: str) -> list[dict]:
        return [n for n in self.scope_notes if n.get("call_id") == call_id
                and n["role"] == "generated-symbol"]

    # -- binding resolution ------------------------------------------------
    def resolve_binder(self, sym: str, scopes: ScopeSet):
        candidates = [b for b in self.binders if b.symbol == sym]
        applicable = [b for b in candidates if b.scopes.subset_of(scopes)]
        if not applicable:
            return None
        applicable.sort(key=lambda b: (max_scope_order(b.scopes), self.binders.index(b)),
                        reverse=True)
        return applicable[0]

    def resolve_uses(self) -> None:
        edges: list[BindingEdge] = []
        seen_uses: set[str] = set()
        MARKERS = {"__define__", "__defmacro__"}

        def record_atom(d: dict, in_head: bool) -> None:
            sid = d["sid"]
            sym = d.get("atom")
            if not isinstance(sym, str) or sid in seen_uses:
                return
            seen_uses.add(sid)
            scopes = self.registry.empty()
            for label in d.get("scopes", []):
                scopes.add(label)
            if sym in MARKERS:
                edges.append(BindingEdge(sid, None, sym, "core", [],
                                         "internal expansion marker",
                                         d.get("span"), d.get("origin", [])))
                return
            if sym in CORE_FORMS and in_head:
                edges.append(BindingEdge(sid, f"prim::{sym}", sym, "core", [],
                                         "core form recognised by symbol at head position",
                                         d.get("span"), d.get("origin", [])))
                return
            candidates = [b for b in self.binders if b.symbol == sym]
            applicable = [b for b in candidates if b.scopes.subset_of(scopes)]
            applicable.sort(key=lambda b: (max_scope_order(b.scopes),
                                           self.binders.index(b)), reverse=True)
            rejected = []
            chosen = applicable[0] if applicable else None
            for b in (applicable[1:] if chosen else []):
                rejected.append({"binder_sid": b.sid, "kind": b.kind,
                                 "reason": "shadowed: newer scope wins",
                                 "newer_scope_order": max_scope_order(chosen.scopes),
                                 "scope_order": max_scope_order(b.scopes)})
            for b in candidates:
                if b in applicable:
                    continue
                rejected.append({"binder_sid": b.sid, "kind": b.kind,
                                 "reason": "scope-set not a subset (hygiene barrier)",
                                 "missing_scopes": b.scopes.difference(scopes).labels()})
            if chosen is None:
                self.diagnostics.append(Diagnostic(
                    "E_UNBOUND", f"identifier {sym!r} is unbound", "error",
                    self._span_from_dict(d), sid, d.get("origin", []),
                    {"candidates_rejected": rejected}))
                edges.append(BindingEdge(sid, None, sym, "unbound", rejected,
                                         "no binder whose scope set is a subset of this use",
                                         d.get("span"), d.get("origin", [])))
            else:
                edges.append(BindingEdge(sid, chosen.sid, sym, "bound", rejected,
                                         "resolved by scope-set subset, newest scope as tiebreak",
                                         d.get("span"), d.get("origin", [])))

        def walk(d: dict, in_head: bool, skip_head: bool) -> None:
            if d.get("kind") != "list":
                record_atom(d, in_head)
                return
            items = d.get("items", [])
            for idx, c in enumerate(items):
                head_sym = items[0].get("atom") if items else None
                if idx == 0 and head_sym in ("quote", "__define__", "__defmacro__"):
                    # skip entirely: quoted datum, registered name
                    continue
                if head_sym == "__define__" and idx == 1:
                    walk(c, False, True)
                    continue
                walk(c, idx == 0, False)

        for root in self.expanded_roots:
            walk(root, False, False)
        self.binding_edges = [e.to_dict() for e in edges]

    @staticmethod
    def _span_from_dict(d: dict):
        sp = d.get("span")
        if not sp:
            return None
        return Span(sp["source_id"], sp["start"], sp["end"], sp["line"], sp["col"])

    # -- output ------------------------------------------------------------
    def payload(self) -> dict:
        tokens = {sid: [t.to_dict() for t in toks]
                  for sid, toks in self.tokens_by_source.items()}
        return {
            "steps": self.steps,
            "macro_calls": self.calls,
            "scope_notes": self.scope_notes,
            "binders": [b.to_dict() for b in self.binders],
            "binding_edges": getattr(self, "binding_edges", []),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "expanded_roots": self.expanded_roots,
            "tokens": tokens,
            "budget": {"initial": self.budget_initial, "remaining": self.budget,
                       "exceeded": self.budget_exceeded},
            "generated_symbols": [n for n in self.scope_notes
                                   if n["role"] == "generated-symbol"],
        }


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {"recursion_budget": 64, "gensym_seed": 20260924}


def params_fingerprint(params: dict) -> str:
    """Stable fingerprint of the expansion parameters (algorithm input, not UI)."""
    merged = dict(DEFAULT_PARAMS)
    merged.update(params or {})
    keys = sorted(merged)
    body = ",".join(f"{k}={merged[k]!r}" for k in keys)
    return "fp_" + hashlib.sha256(
        f"{ALGORITHM_VERSION}|{body}".encode()).hexdigest()[:16]


def analyze(sources: dict[str, str], source_order: list[str] | None = None,
            params: dict | None = None, run_id: str = "r1") -> dict:
    """Lex, parse, expand and explain bindings for one revision of sources."""
    merged = dict(DEFAULT_PARAMS)
    merged.update(params or {})
    registry = ScopeRegistry()
    order = source_order or sorted(sources)
    try:
        forms, tokens, diags = parse_sources(sources, registry, f"{run_id}:")
    except LexError as exc:
        return {
            "fatal": exc.to_dict(),
            "steps": [], "macro_calls": [], "scope_notes": [], "binders": [],
            "binding_edges": [], "expanded_roots": [], "tokens": {},
            "budget": {"initial": merged["recursion_budget"], "remaining": merged["recursion_budget"],
                       "exceeded": False}, "generated_symbols": [],
            "diagnostics": [exc.to_dict()],
        }
    ex = Expander(forms, tokens, diags, registry, merged, order)
    payload = ex.run()
    payload["algorithm_version"] = ALGORITHM_VERSION
    payload["params"] = merged
    payload["params_fingerprint"] = params_fingerprint(merged)
    payload["source_order"] = order
    payload["run_id"] = run_id
    return payload
