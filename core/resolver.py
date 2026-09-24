"""Explain every identifier binding using scope-set containment.

Decisions are made from scope relations, never from string renaming. For each
identifier occurrence we enumerate every binder whose scope set is a subset of
the occurrence's scope set, reject the others with a reason, and either pick
the most specific binder or retain *all* tied candidates as an ambiguity.
"""
from __future__ import annotations

import numpy as np

from .build import MODULE_SCOPE, Binder
from .syntax import OriginFrame, Syn, stable_id

CORE_HEADS = {
    "quote", "lambda", "if", "mif", "begin", "define", "defmacro", "let",
    "datum->syntax", "join-contexts", "gen-temp", "set!",
}
PRIMITIVES = {"+", "-", "*", "=", "<", ">", "<=", ">="}


class BindingResolver:
    def __init__(self, build_result, expansion, seed="run"):
        self.build = build_result
        self.exp = expansion
        self.seed = seed
        self.binders: list[Binder] = list(build_result.binders)
        self.scope_index = {s["scope_id"]: i
                            for i, s in enumerate(expansion.scopes)}
        self._new_scope_seq = sum(
            1 for s in expansion.scopes if s["kind"] in ("lambda", "let"))
        self.edges: list[dict] = []

    def _allocate_scope(self, kind, note):
        # Continue numbering that does not collide with parser/macro scopes.
        self._new_scope_seq += 1
        sid = f"scp-{kind}-post{self._new_scope_seq}"
        self.exp.scopes.append({
            "scope_id": sid, "kind": kind, "note": note,
            "introduced_by": "post-expansion allocation",
        })
        self.scope_index[sid] = len(self.exp.scopes) - 1
        return sid

    def _parser_param_sets(self):
        return {frozenset(b.scope_set) for b in self.build.binders}

    def _add_binder(self, name, node, kind, region_nodes=()):
        binder = Binder(
            name=name, kind=kind, scope_set=frozenset(node.scopes),
            region_set=frozenset(node.scopes),
            sid=stable_id("binder", name, sorted(node.scopes)),
            binding_sid=stable_id("binding", name, sorted(node.scopes)),
        )
        self.binders.append(binder)
        return binder

    def allocate_generated(self):
        """Assign fresh symmetric scopes to binding forms produced by macros."""

        parser_sets = self._parser_param_sets()

        def already_allocated(name_node):
            return frozenset(name_node.scopes) in parser_sets

        def walk(node: Syn):
            if not node.is_pair() or node.quoted:
                for child in getattr(node, "children", []):
                    walk(child)
                return
            head = node.children[0].atom() if node.children else None
            if head == "lambda" and len(node.children) >= 3 \
                    and node.children[1].is_pair() \
                    and node.children[1].children \
                    and not already_allocated(node.children[1].children[0]):
                scope = self._allocate_scope(
                    "lambda", "post-expansion lambda parameter region")
                params = node.children[1]
                for pnode in params.children:
                    pnode.add_scope(scope)
                    if isinstance(pnode.atom(), str):
                        self._add_binder(pnode.datum, pnode, "lambda")
                for body in node.children[2:]:
                    body.add_scope(scope)
                    walk(body)
                return
            if head == "let" and len(node.children) >= 3 \
                    and not self._let_allocated(node, parser_sets):
                scope = self._allocate_scope(
                    "let", "post-expansion let body region")
                binding_node = node.children[1]
                names = []
                if binding_node.is_pair():
                    for pair in binding_node.children:
                        if pair.is_pair() and len(pair.children) >= 2:
                            name_node = pair.children[0]
                            name_node.add_scope(scope)
                            if isinstance(name_node.atom(), str):
                                names.append(name_node)
                            walk(pair.children[1])
                for body in node.children[2:]:
                    body.add_scope(scope)
                    walk(body)
                for name_node in names:
                    self._add_binder(name_node.datum, name_node, "let")
                return
            for child in node.children:
                walk(child)

        for form in self.exp.expanded:
            walk(form)

    def _let_allocated(self, node, parser_sets):
        binding_node = node.children[1]
        if not binding_node.is_pair():
            return True
        for pair in binding_node.children:
            if pair.is_pair() and len(pair.children) >= 2:
                name_node = pair.children[0]
                # Parser-allocated let binders carry exactly {module, let}.
                if any(frozenset(b.scope_set) == frozenset(name_node.scopes)
                       for b in self.build.binders if b.name == name_node.datum):
                    return True
        return False

    # ---- vectorized scope containment --------------------------------------
    def _bitmask(self, scope_set) -> int:
        mask = 0
        for s in scope_set:
            idx = self.scope_index.get(s)
            if idx is not None:
                mask |= 1 << idx
        return mask

    def resolve_all(self):
        self.allocate_generated()
        for form in self.exp.expanded:
            self._walk_refs(form, None, in_head=False, quoted=False)
        return self.edges

    def _candidate_binders(self, name, ref: Syn):
        rows = []
        for b in self._all_for(name):
            mask_b = self._bitmask(b.scope_set)
            mask_r = self._bitmask(ref.scopes)
            contained = (mask_b & mask_r) == mask_b
            rows.append((b, contained, len(b.scope_set)))
        return rows

    def _all_for(self, name):
        result = list(self._binder_index.get(name, []))
        macro = self.build.macros.get(name)
        if macro is not None:
            result.append(Binder(
                name=name, kind="macro",
                scope_set=frozenset({MODULE_SCOPE}),
                region_set=frozenset({MODULE_SCOPE}),
                sid=stable_id("binder-macro", name),
                binding_sid=stable_id("binding-macro", name)))
        return result

    def _walk_refs(self, node: Syn, binder_rows, in_head, quoted):
        if node.atom() is not None:
            if not quoted and isinstance(node.datum, str):
                self._explain(node, in_head)
            return
        if not node.is_pair():
            return
        if node.children and self._head(node) == "quote":
            for child in node.children[1:]:
                self._mark_quoted(child)
            return
        for index, child in enumerate(node.children):
            self._walk_refs(child, binder_rows, index == 0, quoted)

    def _mark_quoted(self, node):
        if node.atom() is not None:
            self.edges.append(self._edge(node, "quoted", []))
            return
        for child in node.children:
            self._mark_quoted(child)

    def _head(self, node):
        if node.is_pair() and node.children:
            return node.children[0].atom()
        return None

    def _edge(self, ref, status, candidates, chosen=None):
        return {
            "ref_sid": ref.sid, "spelling": ref.datum,
            "scopes": sorted(ref.scopes), "status": status,
            "chosen_binding_sid": chosen, "candidates": candidates,
            "span": (ref.first_span().to_json()
                     if ref.first_span() else None),
            "origin_chain": [f.to_json() for f in ref.origin],
        }

    def _explain(self, ref: Syn, in_head: bool):
        self._binder_index = {b.name: [] for b in self.binders}
        for b in self.binders:
            self._binder_index.setdefault(b.name, []).append(b)

        name = ref.datum
        if name.startswith("#%"):
            self.edges.append(self._edge(ref, "marker", []))
            return
        if in_head and name in CORE_HEADS:
            self.edges.append(self._edge(ref, "core-form", []))
            return
        if name in PRIMITIVES:
            self.edges.append(self._edge(ref, "primitive", []))
            return

        rows = self._candidate_binders(name, ref)
        candidates = []
        matching = []
        for binder, contained, specificity in rows:
            extra = sorted(set(ref.scopes) - binder.scope_set)
            missing = sorted(binder.scope_set - ref.scopes)
            if contained:
                reject = None
                matching.append((binder, specificity))
            else:
                reject = ("scope-set not a subset of occurrence scopes; "
                          f"missing scopes {missing}")
            candidates.append({
                "binding_sid": binder.binding_sid, "binder_sid": binder.sid,
                "name": binder.name, "kind": binder.kind,
                "scope_set": sorted(binder.scope_set),
                "specificity": specificity,
                "contained": bool(contained),
                "rejected_reason": reject,
                "extra_occurrence_scopes": extra,
                "missing_scopes": missing,
            })

        if not matching:
            self.edges.append(self._edge(ref, "unbound", candidates))
            return
        max_spec = max(spec for _, spec in matching)
        tied = [b for b, spec in matching if spec == max_spec]
        if len(tied) > 1:
            tied_sids = {b.binding_sid for b in tied}
            for cand in candidates:
                if cand["binding_sid"] in tied_sids:
                    cand["rejected_reason"] = None
                    cand["tied"] = True
            self.edges.append(self._edge(ref, "ambiguous", candidates))
            return
        chosen = tied[0]
        # Mark non-chosen matching candidates as shadowed.
        for cand in candidates:
            if (cand["contained"] and cand["rejected_reason"] is None
                    and cand["binding_sid"] != chosen.binding_sid):
                cand["rejected_reason"] = (
                    f"shadowed by more specific binding {chosen.binding_sid} "
                    f"({chosen.kind})")
        self.edges.append(self._edge(ref, "bound", candidates,
                                     chosen.binding_sid))
