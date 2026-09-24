"""Syntax-object 模型：稳定身份、source span、scope set、来源元数据。

所有领域量都携带基准标签（坐标基准/单位）；syntax 对象是 hygienic 展开的载体：
每个标识符与每个 list 节点都携带一个 scope 标签集合（set-of-scopes 模型）。
身份是内容寻址的（canonical JSON + SHA-256），因此在固定 seed 下重放、
导出/删除/再导入后身份、顺序与来源关系保持不变。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Optional

# 固定坐标基准与单位标记（领域量必须携带单位或坐标基准）
BASIS_CODEPOINT = "utf8-codepoint-offset"
BASIS_SCOPE_DEPTH = "scope-introduction-depth"
UNIT_EXPANSION_STEP = "macro-expansion-step"
UNIT_BUDGET = "macro-call"
UNIT_TOKEN = "token"


@dataclass(frozen=True)
class Span:
    start: int  # codepoint offset，闭区间起始
    end: int    # codepoint offset，开区间终点
    line: int
    col: int

    def to_json(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "line": self.line,
                "col": self.col, "basis": BASIS_CODEPOINT}


@dataclass(frozen=True)
class Sym:
    name: str

    def key(self) -> str:
        return "s:" + self.name


@dataclass(frozen=True)
class Num:
    value: int

    def key(self) -> str:
        return f"n:{self.value}"


@dataclass(frozen=True)
class Bool:
    value: bool

    def key(self) -> str:
        return "b:" + ("t" if self.value else "f")


Atom = Sym | Num | Bool


@dataclass(frozen=True)
class Stx:
    sid: str
    atom: Optional[Atom]
    items: tuple["Stx", ...]
    span: Optional[Span]
    scopes: frozenset[int]
    origin: Optional[str]          # 直接来源 stx 的 sid（转换链）
    ctx_sid: Optional[str]         # datum->syntax 的上下文标识符 sid（可见的上下文选择）
    gensym_key: Optional[str]      # 生成符号的稳定唯一 key
    synthesized: bool
    step_id: Optional[str]         # 由哪个转换步骤生成
    basis: str = BASIS_CODEPOINT

    # ---- 便捷谓词 ----
    @property
    def is_list(self) -> bool:
        return self.atom is None

    def is_sym(self, name: Optional[str] = None) -> bool:
        return isinstance(self.atom, Sym) and (name is None or self.atom.name == name)

    @property
    def sym_name(self) -> Optional[str]:
        return self.atom.name if isinstance(self.atom, Sym) else None

    def head(self) -> Optional[str]:
        if self.is_list and self.items and isinstance(self.items[0].atom, Sym):
            return self.items[0].atom.name
        return None

    def datum_key(self) -> Any:
        if self.is_list:
            return ["list", [i.datum_key() for i in self.items]]
        return self.atom.key()  # type: ignore[union-attr]

    def to_json(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "atom": self.atom.key() if self.atom is not None else None,
            "items": [i.sid for i in self.items],
            "span": self.span.to_json() if self.span else None,
            "scopes": sorted(self.scopes),
            "origin": self.origin,
            "ctx_sid": self.ctx_sid,
            "gensym_key": self.gensym_key,
            "synthesized": self.synthesized,
            "step_id": self.step_id,
            "basis": self.basis,
        }


@dataclass
class IdGen:
    """内容寻址 ID + 确定性 gensym 计数器（固定 seed 重放 => 同一身份）。"""

    seed: str = "scopelens"
    _scope_counter: int = 2  # 0=core, 1=top
    _gensym_counters: dict[str, int] = field(default_factory=dict)
    _memo: dict[str, Stx] = field(default_factory=dict)

    def fresh_scope(self) -> int:
        self._scope_counter += 1
        return self._scope_counter

    def gensym(self, base: str) -> str:
        n = self._gensym_counters.get(base, 0) + 1
        self._gensym_counters[base] = n
        # 生成符号的稳定 key：含 seed 与计数，确定性且全局唯一
        return f"gensym:{self.seed}:{base}#{n}"

    @staticmethod
    def _digest(payload: Any) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def make(self, *, atom: Optional[Atom], items: tuple[Stx, ...] = (),
             span: Optional[Span] = None, scopes: frozenset[int] = frozenset(),
             origin: Optional[str] = None, ctx_sid: Optional[str] = None,
             gensym_key: Optional[str] = None, synthesized: bool = True,
             step_id: Optional[str] = None, basis: str = BASIS_CODEPOINT) -> Stx:
        payload = [
            "v2",
            atom.key() if atom is not None else None,
            [i.sid for i in items],
            sorted(scopes),
            origin,
            ctx_sid,
            gensym_key,
            span.to_json() if span is not None else None,
            synthesized,
            step_id,
            basis,
            self.seed,
        ]
        sid = self._digest(payload)
        stx = self._memo.get(sid)
        if stx is None:
            stx = Stx(sid=sid, atom=atom, items=tuple(items), span=span,
                      scopes=frozenset(scopes), origin=origin, ctx_sid=ctx_sid,
                      gensym_key=gensym_key, synthesized=synthesized,
                      step_id=step_id, basis=basis)
            self._memo[sid] = stx
        return stx

    def atom_stx(self, atom: Atom, span: Span,
                 scopes: frozenset[int] | None = None) -> Stx:
        return self.make(atom=atom, span=span, scopes=scopes or frozenset(),
                         synthesized=False)

    def list_stx(self, items: list[Stx], span: Optional[Span],
                 scopes: frozenset[int] = frozenset(), *,
                 synthesized: bool = False, origin: Optional[str] = None,
                 ctx_sid: Optional[str] = None, step_id: Optional[str] = None) -> Stx:
        return self.make(atom=None, items=tuple(items), span=span, scopes=scopes,
                         synthesized=synthesized, origin=origin, ctx_sid=ctx_sid,
                         step_id=step_id)

    def clone(self, stx: Stx, *, scopes: Optional[frozenset[int]] = None,
              origin: Optional[str] = None, ctx_sid: Optional[str] = None,
              gensym_key: Optional[str] = None, step_id: Optional[str] = None,
              synthesized: Optional[bool] = None) -> Stx:
        return self.make(
            atom=stx.atom, items=stx.items, span=stx.span,
            scopes=stx.scopes if scopes is None else scopes,
            origin=stx.origin if origin is None else origin,
            ctx_sid=stx.ctx_sid if ctx_sid is None else ctx_sid,
            gensym_key=stx.gensym_key if gensym_key is None else gensym_key,
            synthesized=stx.synthesized if synthesized is None else synthesized,
            step_id=stx.step_id if step_id is None else step_id,
            basis=stx.basis,
        )

    def all_stx(self) -> dict[str, Stx]:
        return dict(self._memo)
