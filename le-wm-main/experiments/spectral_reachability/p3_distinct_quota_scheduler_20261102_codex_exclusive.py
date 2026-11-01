"""Fixed distinct P3 quota traversal over the existing model boundary.

No device, model or physical environment is created here. This is a scheduling
seam; callers retain responsibility for host-CUDA admission and complete
instrumentation. A successful result is not scientific or physical admission.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from . import p3_c3_dynamic_budget_20261102_exclusive as budget
from . import p3_c3_model_boundary_20261102_exclusive as boundary

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "experiments/spectral_reachability/p3_c3_dynamic_budget_20261102_exclusive.py":
        "1df244c745027c092880ea72459253dfeef6d28d823ea3ee13e0aea911205778",
    "experiments/spectral_reachability/p3_c3_model_boundary_20261102_exclusive.py":
        "33659afbec04e583827dbaa954bc68ba299de4b2cf5441cd2695d5f125d7c76a",
    "experiments/spectral_reachability/p3_c3_live_executor_bridge_20261102_exclusive.py":
        "0261c3e84525fb849eb8c9d6a84f47d54ae600b0f6cda4b70bc0ca84db826c51",
    "results/p3_effective_equal_budget_saved_20261101_parallel.json":
        "de279df2bb5696b48f316b0ba0745f64101319ec9e2af9d523746a4b92cc02a7",
    "results/p3_effective_equal_budget_host_result_20261101_parallel.json":
        "f95c07943dad455c67424d4006dff14538733fa455e513a1b39be0d97b0d68b0",
}
ARMS = ("signal", "reverse")
AXES = boundary.AXES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol():
    for name, expected in SOURCES.items():
        path = ROOT / name
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"source identity: {name}")
    frozen = budget.protocol()
    require(frozen["points"] == [
        {"budget_point": 20, "signal": [15, 5], "reverse": [5, 15],
         "candidate_total_equal": True, "strategy_difference_precommitted": True},
        {"budget_point": 30, "signal": [22, 8], "reverse": [8, 22],
         "candidate_total_equal": True, "strategy_difference_precommitted": True},
    ], "fixed allocations")
    return {"schema": "p3_distinct_quota_scheduler_20261102_codex_exclusive.v1",
            "source_sha256": dict(SOURCES), "allocations": frozen["points"],
            "candidate_order": "root 0 then root 1; candidate indices ascending from zero",
            "winner": "minimum finite score per root; stable smallest candidate index",
            "failure": "retain attempted and returned; fail arm; no padding, retry or replacement",
            "comparison": "complete candidates, attempted=returned, equal recorded axes",
            "host_cuda_enforced_here": False, "complete_ledger_certified": False,
            "equal_total_compute": False, "flops": "unknown", "energy": "unknown",
            "P3_admission": False, "C3_admission": False, "gate_modified": False}


def run_arm(adapter, evaluate):
    """Execute one arm through the adapter, preserving partial failure evidence.

    The callback receives (meter, root, index). It must use the supplied meter
    for its model calls and return a finite scalar. No physical operation is
    authorized by this function.
    """
    protocol()
    require(isinstance(adapter, boundary.BoundaryAdapter) and callable(evaluate), "adapter/callback")
    require(adapter.started == [0, 0] and adapter.completed == [0, 0]
            and not adapter.failed and all(v == 0 for v in adapter.attempted.values()), "fresh adapter")
    candidates, error = [], None
    try:
        for root, quota in enumerate(adapter.quotas):
            for index in range(quota):
                before = dict(adapter.attempted)
                value = adapter.candidate(root, index, lambda meter: evaluate(meter, root, index))
                require(type(value) in (int, float) and math.isfinite(value), "finite scalar score")
                require(all(adapter.attempted[k] > before[k] for k in AXES),
                        "candidate must dispatch each measured model axis")
                candidates.append({"root": root, "index": index, "score": float(value)})
    except Exception as exc:
        adapter.failed = True
        error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    snapshot = adapter.snapshot()
    complete = (error is None and snapshot["candidate_panel_complete"]
                and snapshot["attempted"] == snapshot["returned"])
    winners = []
    if complete:
        for root in (0, 1):
            subset = [r for r in candidates if r["root"] == root]
            require(len(subset) == adapter.quotas[root] and subset, "nonempty complete root")
            winners.append(min(subset, key=lambda r: (r["score"], r["index"])))
    return {"point": adapter.point, "arm": adapter.arm, "status": "complete" if complete else "failed",
            "allocation": list(adapter.quotas), "candidates": candidates, "winners": winners,
            "error": error, "ledger": snapshot, "P3_admission": False, "C3_admission": False}


def compare(signal, reverse):
    protocol()
    require(signal["arm"] == "signal" and reverse["arm"] == "reverse"
            and signal["point"] == reverse["point"], "arm identity")
    point = signal["point"]
    frozen = {row["budget_point"]: row for row in budget.protocol()["points"]}
    require(point in frozen and signal["allocation"] == frozen[point]["signal"]
            and reverse["allocation"] == frozen[point]["reverse"]
            and signal["allocation"] != reverse["allocation"], "fixed distinguishable allocations")
    complete = signal["status"] == reverse["status"] == "complete"
    a, b = signal["ledger"], reverse["ledger"]
    equal = (complete and a["candidate_panel_complete"] and b["candidate_panel_complete"]
             and a["caps"] == b["caps"]
             and a["attempted"] == a["returned"] == b["attempted"] == b["returned"])
    return {"point": point, "both_complete": complete, "equal_measured_model_axes": equal,
            "candidate_total_equal": sum(signal["allocation"]) == sum(reverse["allocation"]),
            "policy_distinct": True, "complete_ledger_certified": False,
            "evidence_scope": "adapter snapshots, not independently authenticated execution traces",
            "raw_model_bypass_detected": "unknown", "host_cuda_identity": "not_checked_here",
            "equal_total_compute": False, "P3_admission": False, "C3_admission": False}
