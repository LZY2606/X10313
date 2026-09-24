"""Application services: seed the offline case pack and run counterexamples."""
from __future__ import annotations

from typing import Any, Optional

from .archive import JUDGMENTS, ExperimentSpec, build_experiment_record
from .cases import CASE_PACK
from .ids import digest
from .store import Store


def seed_case_project(store: Store, *, name: str = "宏作用域镜案例"
                      ) -> dict[str, Any]:
    """Create the shipped domain project with all seven scenarios.

    The initial import keeps the full baseline.  The recursive-macro case
    additionally receives a parameter revision (budget tightening), a
    local-only source edit, and a pair of conflicting human decisions.
    """
    proj = store.create_project(name)
    pid = proj["id"]
    seeded: dict[str, Any] = {"project_id": pid, "inputs": [],
                              "artifacts": [], "experiments": []}

    for case in CASE_PACK["cases"]:
        label = f"{case['key']}.scm"
        iv = store.submit_input(pid, case["source"], label)
        art = store.compute_artifact(pid, iv["id"],
                                     {"macro_budget": 64,
                                      "seed": store.clock.seed})
        seeded["inputs"].append(iv["id"])
        seeded["artifacts"].append(art["id"])
        _record_case_experiment(store, pid, case, iv, art)

    # parameter revision + local source change + conflicting decision,
    # expressed as new immutable versions with parent links.
    rec = next(c for c in CASE_PACK["cases"]
               if c["key"] == "recursive-macro")
    parent = store.latest_input_id(pid)
    iv_param = store.submit_input(
        pid, rec["source"], "recursive-macro-param-v2.scm",
        parent_version_id=parent)
    art_tight = store.compute_artifact(pid, iv_param["id"],
                                       {"macro_budget": 2,
                                        "seed": store.clock.seed})
    store.add_checkpoint(pid, "初始版本+参数修订")

    local_edit = rec["source"].replace("(countdown 3)", "(countdown 5)")
    iv_local = store.submit_input(
        pid, local_edit, "recursive-macro-local-edit.scm",
        parent_version_id=iv_param["id"])
    art_local = store.compute_artifact(pid, iv_local["id"],
                                       {"macro_budget": 64,
                                        "seed": store.clock.seed})
    store.record_decision(
        pid, "artifact", art_tight["id"], "accept",
        "接受预算 2 的失败结果作为有界递归证据",
        {"candidate_status": "budget-exhausted",
         "rejected_alternative": "无限展开"})
    store.record_decision(
        pid, "artifact", art_tight["id"], "reject",
        "冲突决定：拒绝把预算耗尽当作终止（人工分歧）",
        {"conflicts_with": "accepted bounded-recursion"})
    store.add_checkpoint(pid, "局部改动+冲突裁定")
    seeded.update({"param_artifact": art_tight["id"],
                   "local_artifact": art_local["id"]})
    return seeded


def _record_case_experiment(store: Store, pid: str, case: dict[str, Any],
                            iv: dict[str, Any], art: dict[str, Any]) -> None:
    spec = ExperimentSpec(
        key=case["key"], title=case["title"], scenario=case["scenario"],
        mutation_kind="source", baseline_source=case["source"],
        mutated_source=case.get("source"), mutated_params=None,
        expected_local_scope=[], expected_finding=case["judgment"])

    mutated_art: Optional[dict[str, Any]] = None
    adopted = True
    note = case["expect"]
    candidates: list[dict[str, Any]] = []

    if case["key"] == "recursive-macro":
        # second macro version: tighter budget -> budget-exhausted
        tight = store.compute_artifact(
            pid, iv["id"], {"macro_budget": 2, "seed": store.clock.seed},
            attempt_seq=1)
        # keep first current artifact; re-create baseline attempt order
        mutated_art = tight
        candidates = [
            {"outcome": "terminates", "status": "complete",
             "params": {"macro_budget": 64}, "adopted": True},
            {"outcome": "budget-exhausted with partial graph",
             "status": "budget-exhausted", "params": {"macro_budget": 2},
             "adopted": False},
        ]
        note = "两种结果都保留：正常终止被采纳，预算耗尽的部分展开图" \
               "作为反例证据保留但不冒充当前结果"
    elif case["key"] == "explicit-capture":
        candidates = [
            {"outcome": "hygienic: introduced it stays unbound to caller",
             "adopted": False,
             "rejected_reason": "未通过 datum->syntax 传递上下文，无法实现 anaphor"},
            {"outcome": "explicit capture via visible context",
             "adopted": True,
             "rejected_reason": ""},
        ]
    else:
        candidates = [{"outcome": case["expect"], "adopted": True,
                       "rejected_reason": ""}]

    record = build_experiment_record(
        project_id=pid, spec=spec, baseline=art,
        mutated_artifact=mutated_art if mutated_art is not None else art,
        adopted=adopted,
        decision_note=note, candidate_outcomes=candidates)
    store.save_experiment(record)


def replay(store: Store, project_id: str) -> dict[str, Any]:
    """Deterministically rebuild every current artifact from raw inputs.

    Replayed artifacts must have identical stable ids to the stored ones;
    the comparison is returned for verification.
    """
    report: list[dict[str, Any]] = []
    for iv in store.list_inputs(project_id):
        existing = store.current_artifact(project_id, iv["id"])
        params = (existing["params"] if existing
                  else {"macro_budget": 64, "seed": store.clock.seed})
        replay_art = store.compute_artifact(
            project_id, iv["id"], params,
            attempt_seq=existing["attempt_seq"] if existing else 0)
        report.append({
            "input_version_id": iv["id"],
            "stored_artifact_id": existing["id"] if existing else None,
            "replayed_artifact_id": replay_art["id"],
            "identical": (existing is not None
                          and existing["id"] == replay_art["id"]
                          and existing["graph"] == replay_art["graph"]),
        })
    return {"project_id": project_id, "checks": report,
            "all_identical": all(x["identical"] for x in report)}
