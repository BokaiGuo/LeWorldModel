"""Seal temporal-vs-endpoint retention before reading physical pool labels."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_retention_counterfactual_pool_20261107_codex as pool
from experiments.pc_input_permissions import file_hash

ROOT = pool.run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_TEMPORAL_RETENTION_FIXED_POOL_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_temporal_retention_fixed_pool_plan_20261107_codex.json"
OUT = ROOT / "results/pc_tworoom_temporal_retention_fixed_pool_result_20261107_codex.json"


def keep(scores):
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("expected three prefixes by three finite probes")
    return np.argsort(matrix.min(axis=1), kind="stable")[:2].tolist()


def coverage(entry, indices):
    if (len(indices) != 2 or len(set(indices)) != 2 or any(i not in range(3) for i in indices)
            or len(entry["physical"]) != 6):
        raise ValueError("invalid retained pool")
    prefix = entry["prefix"]
    ever, tail = [], []
    for trace in entry["physical"]:
        if len(trace) > entry["remaining"] or any(x["success"] != (x["distance"] < 16) for x in trace):
            raise ValueError("invalid physical trace")
        ever.append(any(x["success"] for x in trace))
        tail.append(not entry["initial_success"] and not any(x["success"] for x in trace[:prefix])
                    and any(x["success"] for x in trace[prefix:]))
    selected = [2 * i + j for i in indices for j in range(2)]
    return {"ever": any(ever[i] for i in selected), "tail": any(tail[i] for i in selected)}


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    original = json.loads(pool.run.OUTPUT.read_text())
    if (original["runner_sha256"] != file_hash(pool.run.__file__)
            or original["protocol_sha256"] != file_hash(pool.run.PROTOCOL)
            or original["input_sha256"] != file_hash(pool.run.INPUT)):
        raise ValueError("source drift")
    decisions = []
    for row in original["rows"]:
        for old, effect, _, _ in pool.matched_decisions(row):
            endpoint = np.asarray(old["probe_cost"])
            temporal = np.asarray(old["temporal_scores"][1]).reshape(3, 3)
            np.testing.assert_allclose(endpoint, effect["probe_cost"], rtol=0, atol=1e-3)
            np.testing.assert_allclose(temporal, np.asarray(effect["temporal_scores"][1]).reshape(3, 3),
                                       rtol=0, atol=1e-3)
            original_kept = keep(endpoint)
            if original_kept != old["retained"]:
                raise ValueError("baseline retention mismatch")
            decisions.append({"episode": row["episode"], "iteration": old["iteration"],
                              "episode_initial_success": row["initial_success"],
                              "endpoint": endpoint.tolist(), "temporal": temporal.tolist(),
                              "baseline": original_kept, "temporal_kept": keep(temporal)})
    if len(decisions) != 119 or len({x["episode"] for x in decisions}) != 40:
        raise ValueError("unexpected fixed panel")
    with PLAN.open("x") as stream:
        json.dump({"source_sha256": file_hash(pool.run.OUTPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "decisions": decisions}, stream, allow_nan=False)
    print(f"sealed {len(decisions)} model-only decisions", flush=True)


def evaluate():
    if OUT.exists():
        raise FileExistsError(OUT)
    decisions, physical = json.loads(PLAN.read_text()), json.loads(pool.OUTPUT.read_text())
    if (decisions["source_sha256"] != file_hash(pool.run.OUTPUT)
            or decisions["protocol_sha256"] != file_hash(PROTOCOL)
            or decisions["runner_sha256"] != file_hash(Path(__file__))
            or physical["source_sha256"] != file_hash(pool.run.OUTPUT)
            or physical["input_sha256"] != file_hash(pool.run.INPUT)
            or physical["protocol_sha256"] != file_hash(pool.PROTOCOL)
            or physical["runner_sha256"] != file_hash(pool.__file__)):
        raise ValueError("artifact drift")
    index = {(x["episode"], x["iteration"]): x for x in physical["entries"]}
    if len(index) != len(physical["entries"]) or len(index) != len(decisions["decisions"]):
        raise ValueError("physical pool identity mismatch")
    rows = []
    for decision in decisions["decisions"]:
        entry = index[(decision["episode"], decision["iteration"])]
        if decision["baseline"] != entry["retained"]["lookahead"]:
            raise ValueError("mismatched retention")
        rows.append({"episode": decision["episode"], "iteration": decision["iteration"],
                     "episode_initial_success": decision["episode_initial_success"],
                     "changed": decision["baseline"] != decision["temporal_kept"],
                     "baseline": coverage(entry, decision["baseline"]),
                     "temporal": coverage(entry, decision["temporal_kept"])})
    episodes = {}
    for row in rows:
        ep = episodes.setdefault(row["episode"], {"initial_success": row["episode_initial_success"],
                                                   "baseline": False, "temporal": False})
        if ep["initial_success"] != row["episode_initial_success"]:
            raise ValueError("inconsistent episode")
        for arm in ("baseline", "temporal"):
            ep[arm] |= row[arm]["tail"]
    subset = [x for x in episodes.values() if not x["initial_success"]]
    temporal_only = sum(x["temporal"] and not x["baseline"] for x in subset)
    baseline_only = sum(x["baseline"] and not x["temporal"] for x in subset)
    summary = {"decisions": len(rows), "changed": sum(x["changed"] for x in rows),
               "decision_ever": {arm: sum(r[arm]["ever"] for r in rows) for arm in ("baseline", "temporal")},
               "decision_tail": {arm: sum(r[arm]["tail"] for r in rows) for arm in ("baseline", "temporal")},
               "episode_tail": {arm: sum(e[arm] for e in episodes.values()) for arm in ("baseline", "temporal")},
               "non_initial_n": len(subset),
               "non_initial_tail": {arm: sum(e[arm] for e in subset) for arm in ("baseline", "temporal")},
               "temporal_only": temporal_only, "baseline_only": baseline_only,
               "development_exit": temporal_only >= 3 and baseline_only <= 1
                                   and sum(x["temporal"] for x in subset) >= sum(x["baseline"] for x in subset)}
    with OUT.open("x") as stream:
        json.dump({"plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(pool.OUTPUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "summary": summary, "rows": rows}, stream, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "evaluate"):
        raise SystemExit("expected plan|evaluate")
    {"plan": plan, "evaluate": evaluate}[sys.argv[1]]()
