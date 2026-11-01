"""Reconstruct saved H2 calibration traces; this is not Full D2P."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "physical": ("results/p2_composition_paired_physical_20261027_v2_main01/physical_report.json", "e696af5758c88172dafccb30cf62901fe91f9640ebf4df7ad57d4f2f19da8a4d"),
    "planning": ("results/p2_composition_paired_20261027_v2_prepared01/report.json", "0c2a92842fa993cdc3dc566bab86bf124db15a6a0d5b054cbfef5e2d75f86148"),
    "prior_outcomes": ("results/p2_composition_paired_outcomes_20261027_v2_report01/report.json", "2d53a10c6aca92be5e83b9883d7d2fcd8b693d380bad89fae5738a8ef9b69f09"),
}
OUT = ROOT / "results/codex_sra_composition_timestep_20261030_LOCAL01.json"
ARMS = ("joint_h2_no_composition", "conditional_suffix_composition")
ROOTS = (
    "9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333",
    "23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050",
    "4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca",
    "cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2",
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load():
    data = {}
    for name, (path, sha) in FILES.items():
        raw = (ROOT / path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == sha, "source hash: " + name)
        data[name] = json.loads(raw)
    return data


def vector(x):
    need(isinstance(x, list) and len(x) == 7 and all(
        type(v) in (int, float) and math.isfinite(v) for v in x), "bad state/goal")
    return x


def metric(state, goal):
    d = [a - b for a, b in zip(vector(state), vector(goal))]
    return {"agent_xy": math.hypot(*d[:2]), "object_xy": math.hypot(*d[2:4]),
            "joint_xy": math.sqrt(sum(x*x for x in d[:4])),
            "wrapped_angle": abs((d[4] + math.pi) % (2*math.pi) - math.pi)}


def timeline(row):
    need(row["corrected_search_horizon"] == 2 and row["planned_commands"] == 10, "wrong horizon")
    n = row["completed_commands"]
    need(type(n) is int and 0 <= n <= 10 and row["physical_env_step_returns"] == n, "bad return count")
    need(row["status"] in ("completed", "failed", "guard_stopped", "terminated", "truncated"), "unknown status")
    if row["status"] == "completed":
        need(n == 10 and row["full_horizon_completed"] is True and not row["failure"], "bad completed status")
    steps = {0: vector(row["context_identity"]["state"])}
    for item in row["trace"]:
        t = item["primitive_step"]
        need(type(t) is int and 1 <= t <= n and t not in steps, "bad or duplicate step")
        need(item["kind"] == "observation", "not an observation")
        steps[t] = vector(item["state"])
    if n == 10 and 10 in steps:
        need(steps[10] == vector(row["final_state"]), "final state mismatch")
    return steps


def audit(data):
    physical, planning, prior = data["physical"], data["planning"], data["prior_outcomes"]
    need(physical["schema"] == "p2_composition_paired_physical_20261027_v2.physical_before_encoding", "schema")
    indexed = {}
    for row in physical["rows"]:
        key = (row["root_sha256"], row["method"])
        need(key[0] in ROOTS and key[1] in ARMS and key not in indexed, "unexpected/duplicate row")
        need(row["root_index"] == ROOTS.index(key[0]), "root index mismatch")
        indexed[key] = row
    need(len(indexed) == 8, "missing rows")
    plans = {r["root_sha256"]: r for r in planning["rows"]}
    need(len(planning["rows"]) == 4 and set(plans) == set(ROOTS), "planning roots")
    old = {r["root_sha256"]: r for r in prior["paired"]}
    need(len(prior["paired"]) == 4 and set(old) == set(ROOTS), "prior roots")
    pairs, budgets = [], []
    for root in ROOTS:
        a, b = (indexed[root, arm] for arm in ARMS)
        need(a["context_identity"] == b["context_identity"], "context or goal mismatch")
        goal = vector(a["context_identity"]["goal"])
        sa, sb = timeline(a), timeline(b)
        plan = plans[root]
        need(plan["root_index"] == ROOTS.index(root) and plan["predict_calls"] == 252
             and plan["action_encoder_calls"] == 252, "resource mismatch")
        ops = plan["completed_operations"]
        need(ops == {"jacobian_calls": 41, "scalar_output_vjp_directions": 7872,
                     "svd_calls": 42, "bounded_solve_calls": 41}, "operation mismatch")
        receipt = plan["resource_receipt"]
        need(receipt["shared"]["predict_calls"] == 4
             and receipt["unique"]["predict_calls"] == 252
             and [receipt["standalone_attributed"][arm]["predict_calls"] for arm in ARMS] == [88, 168]
             and [receipt["exclusive"][arm]["predict_calls"] for arm in ARMS] == [84, 164],
             "resource attribution mismatch")
        arms = {r["arm"]: r for r in plan["results"]}
        need(len(plan["results"]) == 2 and set(arms) == set(ARMS), "planning arms")
        for arm in ARMS:
            r = arms[arm]
            need(r["complete"] and r["selection_evaluations"] == 40
                 and sum(x["predict_calls"] for x in r["ledger"]) == 84
                 and all(x["completed"] for x in r["ledger"]), "incomplete planner ledger")
        budgets.append({"root_sha256": root, "standalone_attribution": {
            "sra": {"predict": 88, "jacobians": 1, "vjp_directions": 192, "svd": 2, "bounded_solves": 1},
            "conditional": {"predict": 168, "jacobians": 41, "vjp_directions": 7872, "svd": 42, "bounded_solves": 41}},
            "paired_unique_predict": plan["predict_calls"], "shared_predict": 4,
            "predict_difference_conditional_minus_sra": 80, "flops": None, "common_walltime_cap": None})
        for t in range(11):
            x = metric(sa[t], goal) if t in sa else None
            y = metric(sb[t], goal) if t in sb else None
            delta = {k: y[k]-x[k] for k in x} if x is not None and y is not None else None
            pairs.append({"root_sha256": root, "step": t, "sra": x, "conditional": y,
                          "conditional_minus_sra": delta,
                          "missing_arms": [arm for arm, v in zip(ARMS, (x, y)) if v is None]})
            if t == 10 and delta is not None:
                prev = old[root]
                need(prev["both_full_H2"] and math.isclose(delta["agent_xy"], prev["composition_minus_no_composition"]["agent_position_error"], abs_tol=1e-9)
                     and math.isclose(delta["object_xy"], prev["composition_minus_no_composition"]["object_position_error"], abs_tol=1e-9), "prior endpoint mismatch")
    return {"schema": "codex_sra_composition_timestep_20261030_v1",
            "source_sha256": {k: v[1] for k, v in FILES.items()},
            "fixed_denominator": 44, "observed_pairs": sum(p["conditional_minus_sra"] is not None for p in pairs),
            "pairs": pairs, "budget_by_root": budgets,
            "full_d2p_comparison": None,
            "full_d2p_reason": "No matched Full D2P physical arm established by these pinned inputs; conditional H2 is not Full D2P.",
            "limits": ["four known calibration roots, not independent confirmation", "H2 fixed open-loop conditional suffix, not full RGB feedback",
                       "unequal actual compute and proposal support; no fairness or causal attribution",
                       "VJP directions are not backward API calls; FLOPs and common wall-time cap unknown",
                       "root-time rows are repeated measurements, not independent roots",
                       "prior endpoint report and provenance validation inherited, not rerun",
                       "missing physical observations remain null; no imputation or outcome filtering"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    raw = (json.dumps(audit(load()), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if args.verify:
        need(OUT.read_bytes() == raw, "output mismatch")
        print("verified", OUT)
    else:
        with OUT.open("xb") as f:
            f.write(raw)
        print("created", OUT)


if __name__ == "__main__":
    main()
