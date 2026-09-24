"""Offline case pack: 宏作用域镜 (macro scope mirror).

The pack is domain material (a tiny-language program exercising hygienic
macro expansion), not a generic table. `stage_case` actually replays it
through the expander and the store instead of returning canned results.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from macrocore import ALGO_VERSION, expand_source
from store import Store

CASE_PATH = os.path.join(os.path.dirname(__file__), "cases",
                         "macro_scope_mirror.case.json")


def load_case(path: Optional[str] = None) -> dict:
    with open(path or CASE_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def normalize_raw_tokens(raw_tokens: list[dict]) -> dict:
    """Derived-layer normalization.

    Raw representations are preserved verbatim. Values that cannot be
    converted stay in place with a diagnostic instead of being dropped.
    """
    normalized, diagnostics = [], []
    for idx, item in enumerate(raw_tokens):
        raw = item["raw_span"]
        kept = {"index": idx, "text": item["text"], "raw": raw}
        if raw.get("unit") == "line:col":
            line_s, col_s = str(raw["value"]).split(":")
            kept["normalized"] = {"line": int(line_s), "col": int(col_s),
                                  "basis": "1-based-line:col",
                                  "offset": None}
        elif raw.get("unit") == "line" and isinstance(raw["value"], int):
            kept["normalized"] = {"line": raw["value"], "col": 1,
                                  "basis": "1-based-line", "offset": None}
            kept["diagnostic"] = {
                "kind": "partial-conversion",
                "message": "line-only span normalized to line:col; "
                           "offset requires the source table"}
        else:
            kept["normalized"] = None
            kept["diagnostic"] = {
                "kind": "unconvertible-unit",
                "message": f"cannot convert {raw.get('unit')!r} span "
                           f"{raw.get('value')!r}; raw value retained"}
            diagnostics.append(kept["diagnostic"])
        normalized.append(kept)
    return {"items": normalized, "diagnostics": diagnostics}


def _expansion_event(store: Store, version_id: str, source: str,
                     budget: int, macro_version: str, label: str,
                     stage: str) -> str:
    result, _ = expand_source(source, budget=budget,
                              macro_version=macro_version)
    return store.record_event(
        "expansion",
        {"stage": stage, "label": label, "result": result},
        algo_version=ALGO_VERSION,
        params={"budget": budget, "macro_version": macro_version},
        depends_on=[version_id])


def stage_case(store: Store, case: Optional[dict] = None) -> dict:
    """Replay the full narrative. Returns stable ids of every stage."""
    case = case or load_case()
    ids: dict[str, str] = {}
    sources = case["sources"]
    budget = case["params"]["initial_budget"]
    revised_budget = case["params"]["revised_budget"]

    # stage 1: immutable import
    ids["input"] = store.add_input(sources["v1"])
    v1 = store.add_version(None, "source",
                           "初始导入：宏作用域镜 v1",
                           {"input_id": ids["input"],
                            "source": sources["v1"]})
    ids["v1"] = v1

    # raw tokens preserved verbatim as their own event
    store.record_event(
        "import-artifact",
        {"label": "原始 raw span 原样保存（含 prose-note）",
         "raw_tokens": case["raw_tokens"]},
        depends_on=[v1])

    # stage 2: derived normalization (unconvertible values stay + diag)
    normalized = normalize_raw_tokens(case["raw_tokens"])
    store.record_event(
        "normalization",
        {"label": "raw span 规范化", **normalized},
        algo_version=ALGO_VERSION, params={"rule": "span-normalize/1"},
        depends_on=[v1])

    # stage 3: expand v1
    ids["expand_v1"] = _expansion_event(
        store, v1, sources["v1"], budget, "v1",
        "展开 v1：七个边界情形全部经过", "expand")

    # stage 4: parameter revision (budget 128 -> 5)
    v2 = store.add_version(
        v1, "params", "参数修订：展开预算 128 -> 5",
        {"budget": revised_budget, "unit": "macro-steps"})
    ids["v_params"] = v2

    # stage 5: budget probe -> partial expansion graph
    ids["expand_budget"] = _expansion_event(
        store, v2, sources["budget_probe"], revised_budget, "v1",
        "小预算重放：递归宏预算耗尽，保留部分展开图",
        "expand-budget")

    # stage 6: macro version v2 (aif gensym) for comparison
    ids["expand_macro_v2"] = _expansion_event(
        store, v1, sources["macro_v2_aif_gensym"], budget, "v2",
        "宏版本 v2：aif 改用 gensym，不再捕获 it",
        "expand-macro-v2")

    # stage 7: local source change tmp -> tmp2 (only local deps stale)
    ids["input_v2"] = store.add_input(sources["v2_local_change"])
    v3 = store.add_version(
        v1, "source", "局部源数据改动：tmp -> tmp2",
        {"input_id": ids["input_v2"],
         "source": sources["v2_local_change"],
         "affected_region": "first let block only"})
    ids["v_source2"] = v3
    stale = store.supersede(v1)
    ids["stale_after_source"] = ",".join(stale)
    ids["expand_v2_source"] = _expansion_event(
        store, v3, sources["v2_local_change"], budget, "v1",
        "局部源数据改动后的当前展开", "expand-revised-source")

    # stage 8/9: conflicting human decisions on the same candidate
    target = f"{ids['expand_v1']}:edge:explicit-capture:it"
    ids["decide_accept"] = store.decide(
        target, "accept",
        "接受：it 经 datum->syntax 显式捕获到 aif 绑定")
    ids["decide_reject"] = store.decide(
        target, "reject",
        "冲突：审阅者认为不应允许显式捕获 use-site 的 it")

    # stage 10: resolve conflict by undoing the rejection
    ids["undo_reject"] = store.undo(ids["decide_reject"])
    return ids
