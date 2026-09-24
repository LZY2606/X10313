"""Turn parsed token trees into syntax objects and assign lexical scopes.

Scope assignment follows the symmetric sets-of-scopes discipline: the same
fresh scope is added to both the binding identifier and the region it governs.
Because a macro-introduced identifier never receives the call-site lambda
scope, scope-set containment (not the spelling) decides whether it is captured.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .syntax import OriginFrame, Syn, Token, stable_id

MODULE_SCOPE = "scp-module-0"


@dataclass
class Binder:
    name: str
    kind: str                 # module | define | lambda | let
    scope_set: frozenset[str]
    region_set: frozenset[str]
    sid: str
    binding_sid: str


@dataclass
class MacroDef:
    name: str
    params: list[str]
    template: object           # Syn template (definition-site scopes attached)
    def_scopes: frozenset[str]
    span_token: Token
    sid: str


@dataclass
class BuildResult:
    module: list[Syn]
    macros: dict[str, MacroDef]
    binders: list[Binder]
    scope_registry: list[dict]
    tokens: list[Token]


def _atom_syn(token: Token, scopes, origin):
    return Syn(token.text, frozenset(scopes), list(origin), token=token)


def _build_node(form, scopes, origin, quote=False):
    if isinstance(form, Token):
        node = _atom_syn(form, scopes, origin)
        node.quoted = quote
        return node
    children = []
    for item in form:
        children.append(_build_node(item, scopes, origin, quote=quote))
    node = Syn([c.datum for c in children], frozenset(scopes), list(origin))
    node.children = children
    node.quoted = quote
    return node


def _head(node: Syn) -> str | None:
    if node.is_pair() and node.children and node.children[0].atom() is not None:
        return node.children[0].datum
    return None


def _name_of(node: Syn) -> str | None:
    atom = node.atom()
    return atom if isinstance(atom, str) else None


def build(forms, tokens: list[Token], seed: str = "src") -> BuildResult:
    scope_registry: list[dict] = []
    binders: list[Binder] = []
    counter = 0

    def fresh(kind: str, note: str) -> str:
        nonlocal counter
        sid = f"scp-{kind}-{counter}"
        counter += 1
        scope_registry.append({
            "scope_id": sid, "kind": kind, "note": note,
        })
        return sid

    # Module scope is present on every original token (the def-site context).
    scope_registry.append({"scope_id": MODULE_SCOPE, "kind": "module",
                           "note": "top-level / definition-site context"})

    # Pass 1: placeholders; real Syn templates are linked after build.
    macros: dict[str, MacroDef] = {}

    # Convert each top-level form to a Syn tree with only the module scope.
    nodes: list[Syn] = []
    for form in forms:
        origin = [OriginFrame("token", "raw import", _form_span(form))]
        nodes.append(_build_node(form, {MODULE_SCOPE}, origin))

    # Link macro definitions to their built syntax template.
    for node in nodes:
        if _head(node) == "defmacro" and len(node.children) >= 4:
            name_node = node.children[1]
            param_list = node.children[2]
            param_names = ([c.datum for c in param_list.children
                            if isinstance(c.atom(), str)]
                           if param_list.is_pair() else [])
            name_tok = name_node.token
            macros[name_node.datum] = MacroDef(
                name=name_node.datum, params=param_names,
                template=node.children[3],
                def_scopes=frozenset({MODULE_SCOPE}),
                span_token=name_tok,
                sid=stable_id("macro", name_node.datum,
                              name_tok.span.offset if name_tok else 0),
            )

    # Pass 2: assign temporal scopes in program order.
    module_binders: list[Binder] = []
    for index, node in enumerate(nodes):
        head = _head(node)
        if head == "define" and len(node.children) >= 3:
            name_node = node.children[1]
            name = _name_of(name_node)
            grp = fresh("define", f"define group {index}"
                        + (f": {name}" if name else ""))
            # define is visible on its own RHS and on all following original
            # forms (supports recursion / later references).
            node.add_scope(grp)
            for later in nodes[index + 1:]:
                later.add_scope(grp)
            if name is not None:
                binders.append(Binder(
                    name=name, kind="define",
                    scope_set=frozenset(name_node.scopes),
                    region_set=name_node.scopes,
                    sid=stable_id("binder", name, sorted(name_node.scopes)),
                    binding_sid=stable_id("binding", name,
                                          sorted(name_node.scopes)),
                ))
        elif head == "defmacro":
            # A macro definition does not introduce a use-site lexical scope.
            pass
        else:
            expr_scope = fresh("expr", f"top-level expression {index}")
            node.add_scope(expr_scope)

    # Pass 3: lambda / let / quote scopes inside each form.
    def annotate(node: Syn, quote=False):
        head = _head(node)
        if node.is_pair() and head == "quote":
            for child in node.children[1:]:
                _mark_quoted(child)
            return
        if not node.is_pair():
            return
        if head == "defmacro":
            # Macro templates are instantiated per call; allocating lambda/let
            # scopes here would make separate invocations share a scope and
            # break hygiene. Scopes are attached after expansion instead.
            return
        if head == "lambda" and len(node.children) >= 3:
            lam_scope = fresh("lambda", "lambda parameter region")
            param_list = node.children[1]
            bodies = node.children[2:]
            if param_list.is_pair():
                for p in param_list.children:
                    p.add_scope(lam_scope)
            for body in bodies:
                body.add_scope(lam_scope)
            if param_list.is_pair():
                region = None
                for p in param_list.children:
                    nm = _name_of(p)
                    if nm is None:
                        continue
                    region = frozenset(p.scopes)
                    binders.append(Binder(
                        name=nm, kind="lambda",
                        scope_set=frozenset(p.scopes),
                        region_set=frozenset(p.scopes),
                        sid=stable_id("binder", nm, sorted(p.scopes)),
                        binding_sid=stable_id("binding", nm, sorted(p.scopes)),
                    ))
            for body in bodies:
                annotate(body)
            return
        if head == "let" and len(node.children) >= 3:
            let_scope = fresh("let", "let body region")
            binding_node = node.children[1]
            bodies = node.children[2:]
            names: list[str] = []
            if binding_node.is_pair():
                for pair in binding_node.children:
                    if pair.is_pair() and len(pair.children) >= 2:
                        name_node = pair.children[0]
                        name_node.add_scope(let_scope)
                        nm = _name_of(name_node)
                        if nm is not None:
                            names.append(nm)
                        annotate(pair.children[1])
            for body in bodies:
                body.add_scope(let_scope)
            for nm in names:
                binders.append(Binder(
                    name=nm, kind="let",
                    scope_set=frozenset({let_scope, MODULE_SCOPE}),
                    region_set=frozenset({let_scope, MODULE_SCOPE}),
                    sid=stable_id("binder", nm, let_scope),
                    binding_sid=stable_id("binding", nm, let_scope),
                ))
            for body in bodies:
                annotate(body)
            return
        for child in node.children:
            annotate(child)

    def _mark_quoted(node: Syn):
        node.quoted = True
        for child in node.children:
            _mark_quoted(child)

    for node in nodes:
        annotate(node)

    return BuildResult(
        module=nodes, macros=macros, binders=binders,
        scope_registry=scope_registry, tokens=tokens,
    )


def _form_span(form):
    if isinstance(form, Token):
        return form.span
    if isinstance(form, list) and form:
        return _form_span(form[0])
    return None
