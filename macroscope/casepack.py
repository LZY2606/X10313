"""Offline domain case pack: "宏作用域镜 — 语法对象、卫生重命名与来源展开图".

Narrative: a compiler team imports tokens, lexical scopes and macro
definitions of a small language, expands syntax objects and explains every
identifier binding. The pack walks through, in order:

1. initial import  — original tokens, source spans, scope sets,
   definition/use sites, macro call stack, generated symbols and transform
   steps are all preserved (baseline archive);
2. parameter revision — the recursion budget is revised (new event);
3. local source change — one token changes in a revision of the first
   input; only artifacts depending on that input family go stale;
4. conflicting human decisions — two reviewers disagree on one candidate;
   both verdicts stay in history.

The replay actually passes through: same-name local variables, nested
macros, definition-site references, explicit capture, recursive macros,
generated temp names and error source remapping.
"""
from __future__ import annotations

from typing import Callable, Dict

INITIAL_SOURCE = """\
; 宏作用域镜 · baseline: 编译器工具组导入 token / 词法作用域 / 宏定义
(def helper 99)
(defmacro get-helper () helper)
(defmacro with-y (e) (let (y 10) e))
(defmacro with-y2 (e) (let (y 10) (+ y e)))
(defmacro sneaky () (capture y))
(defmacro inc (x) (+ 1 x))
(defmacro inc2 (x) (inc (inc x)))
(defmacro twice (e) (let ((gensym tmp) e) (+ (gensym tmp) (gensym tmp))))
(defmacro bad () missing-var)
(let (y 1) (let (y 2) (with-y y)))
(let (helper 1) (get-helper))
(let (y 7) (with-y2 (sneaky)))
(inc2 0)
(let (tmp 5) (twice 2))
(bad)
"""

# 局部源数据改动: 只改一个直接来源 (内层 y 的初值 2 -> 3)
REVISED_SOURCE = INITIAL_SOURCE.replace("(let (y 2)", "(let (y 3)")

RECURSIVE_SOURCE = """\
; 第二分支: 递归宏 — 展开预算有限, 失败保留部分展开图
(defmacro inf (x) (inf x))
(inf 1)
"""

SAMPLES: Dict[str, str] = {
    "same-name-local": "(defmacro with-y (e) (let (y 10) e))\n(let (y 1) (let (y 2) (with-y y)))\n",
    "nested-macros": "(defmacro inc (x) (+ 1 x))\n(defmacro inc2 (x) (inc (inc x)))\n(inc2 0)\n",
    "definition-site": "(def helper 99)\n(defmacro get-helper () helper)\n(let (helper 1) (get-helper))\n",
    "explicit-capture": "(defmacro with-y2 (e) (let (y 10) (+ y e)))\n(defmacro sneaky () (capture y))\n(let (y 7) (with-y2 (sneaky)))\n",
    "recursive-macro": "(defmacro inf (x) (inf x))\n(inf 1)\n",
    "gensym-temp": "(defmacro twice (e) (let ((gensym tmp) e) (+ (gensym tmp) (gensym tmp))))\n(let (tmp 5) (twice 2))\n",
    "error-remap": "(defmacro bad () missing-var)\n(bad)\n",
}

CASEPACK_ID = "macroscope-mirror/v1"


def load_casepack(store, expand_and_record: Callable[[str], dict]) -> dict:
    """Seed the store with the full case-pack narrative. `expand_and_record`
    is the app's expand endpoint logic: input id -> artifact dict."""
    summary = {"casepack": CASEPACK_ID, "steps": []}

    base = store.add_input(INITIAL_SOURCE)
    summary["steps"].append({"step": "initial-import", "input": base["id"]})
    art1 = expand_and_record(base["id"])
    summary["steps"].append({"step": "baseline-expansion", "artifact": art1["id"],
                             "status": art1["payload"]["status"]})

    rec = store.add_input(RECURSIVE_SOURCE)
    store.set_params({"budget": 8})
    summary["steps"].append({"step": "param-revision", "params": {"budget": 8}})
    art2 = expand_and_record(rec["id"])
    summary["steps"].append({"step": "recursive-budget-exceeded",
                             "artifact": art2["id"],
                             "status": art2["payload"]["status"]})

    rev = store.add_input(REVISED_SOURCE, parent=base["id"])
    summary["steps"].append({"step": "local-source-change", "input": rev["id"],
                             "parent": base["id"]})

    # conflicting human decisions on the same candidate resolution
    target = None
    for r in art1["payload"]["resolutions"]:
        if r["name"] == "y" and r["chosen"]:
            target = f"{art1['id']}:{r['node']}"
            break
    if target is None:
        target = f"{art1['id']}:{art1['payload']['resolutions'][0]['node']}"
    store.add_event("decision", {"target": target, "verdict": "accept",
                                 "rationale": "reviewer-A: scope-set resolution is correct"},
                    sources=[art1["id"]])
    store.add_event("decision", {"target": target, "verdict": "reject",
                                 "rationale": "reviewer-B: wants the other candidate kept"},
                    sources=[art1["id"]])
    summary["steps"].append({"step": "conflicting-decisions", "target": target})

    art3 = expand_and_record(rev["id"])
    summary["steps"].append({"step": "revised-expansion", "artifact": art3["id"],
                             "status": art3["payload"]["status"]})
    return summary
