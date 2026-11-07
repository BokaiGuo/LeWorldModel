"""One-factor generator-support transfer; exact-state development only."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from experiments import pc_waypoint_geometry_20261108_codex as geometry
from experiments.pc_lineage_cross import core, run as lineage

base = lineage.old
ROOT = base.ROOT
OUT = ROOT / "results/pc_plane_lineage_cross_20261108"
PROTOCOL = ROOT.parent / "docs/PC_PLANE_LINEAGE_CROSS_PROTOCOL_20261108_CODEX.md"
BASELINES = ("uniform_lookahead", "random_pairs", "whole_plane", "nominal_cem")
ARMS = core.MATCHED + BASELINES
BUDGET = 128


def sources():
    local = (Path(__file__), Path(__file__).with_name("audit.py"), PROTOCOL,
             geometry.OUT / "manifest.json", geometry.OUT / "summary.json",
             geometry.OUT / "independent_audit.json")
    return {**lineage.sources(), **geometry.sources(), **{str(p): base.sha(p) for p in local}}


def manifest():
    value = {
        "sources": sources(), "tasks": base.previous.tasks(), "repeats": 2,
        "arms": list(ARMS), "budget": 128, "steps": 15, "shape": [16, 17],
        "track": "exact_state_generator_transfer_development",
        "independent_confirmation": False, "gpu": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(),
                    "threads": [os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("OMP_NUM_THREADS")]},
    }
    path = OUT / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("source/config drift")
    else:
        base.write_once(path, value)
    return value


def save(name, value):
    base.write_once(OUT / name, {"manifest_sha256": base.sha(OUT / "manifest.json"), **value})


def load(name):
    value = json.loads((OUT / name).read_text())
    if value["manifest_sha256"] != base.sha(OUT / "manifest.json"):
        raise ValueError("record binding changed")
    return value


def source_chains(task, seed, count=16):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 128:
        raise ValueError("integer count in 1..128 required")
    lateral = np.random.default_rng(seed).uniform(-2, 2, count)
    longitudinal = np.random.default_rng(seed + 70000).uniform(-1, 1, count)
    programs = [geometry.generate(task, a, b) for a, b in zip(lateral, longitudinal, strict=True)]
    return np.stack([p[0] for p in programs]), np.column_stack((lateral, longitudinal))


def plan_one(task, root, repeat):
    seed = 2831000 + root * 100 + repeat
    started = time.perf_counter()
    nominals, parameters = source_chains(task, seed)
    bank = core.PairedBank(nominals)
    generation_seconds = time.perf_counter() - started
    model = base.evaluator(task)
    common = core.Ledger(model, BUDGET)
    started = time.perf_counter()
    observed, anchors = core.acquire(bank, common)
    shared_seconds = time.perf_counter() - started
    if not 113 <= common.count <= 116:
        raise ValueError("cross cost outside frozen range")
    mask = core.support(bank.shape, anchors)
    remaining = BUDGET - common.count
    outcomes = {}
    for arm in core.MATCHED:
        started = time.perf_counter()
        if arm == "random_missing":
            missing = np.argwhere(~mask)
            pairs = missing[np.random.default_rng(seed + 30000).choice(len(missing), remaining, replace=False)]
            predicted, diagnostic = None, {}
        else:
            predicted, diagnostic = core.prediction(arm, observed, anchors, seed + 20000)
            pairs = core.choose(predicted, mask, remaining)
        allocation_seconds = time.perf_counter() - started
        ledger = core.Ledger(model, BUDGET, list(common.trace), common.count)
        started = time.perf_counter()
        ledger.query(bank.compose(pairs), "proposals", pairs)
        choice = ledger.best()
        outcomes[arm] = {
            "uses_shared_cross": True, "trace": ledger.trace[len(common.trace):],
            "prediction": None if predicted is None else predicted.tolist(),
            "diagnostic": diagnostic, "proposal_pairs": pairs.tolist(), "choice": choice,
            "queries": ledger.count, "allocation_seconds": allocation_seconds,
            "planning_seconds": generation_seconds + shared_seconds + allocation_seconds
                                + time.perf_counter() - started,
        }
    for arm in BASELINES:
        ledger = core.Ledger(model, BUDGET, [common.trace[0]], 16)
        started = time.perf_counter()
        if arm == "whole_plane":
            chains, _ = source_chains(task, seed, 128)
            np.testing.assert_array_equal(chains[:16], nominals)
            ledger.query(chains[16:], arm)
            updates = []
        else:
            updates = lineage.baseline_fill(arm, bank, ledger, task, seed)
        choice = ledger.best()
        outcomes[arm] = {
            "uses_shared_cross": False, "trace": ledger.trace[1:], "updates": updates,
            "choice": choice, "queries": ledger.count,
            "planning_seconds": generation_seconds + common.trace[0]["query_wall_seconds"]
                                + time.perf_counter() - started,
        }
    for result in outcomes.values():
        if result["queries"] != BUDGET or result["choice"]["score"] > min(common.trace[0]["costs"]):
            raise ValueError("budget or incumbent violation")
    return {
        "root": root, "repeat": repeat, "task": task, "seed": seed,
        "nominals": nominals.tolist(), "parameters": parameters.tolist(),
        "nominal_sha256": hashlib.sha256(nominals.tobytes()).hexdigest(),
        "anchors": anchors.tolist(), "shared_queries": common.count, "shared_trace": common.trace,
        "observed": [[float(x) if np.isfinite(x) else None for x in row] for row in observed],
        "generation_seconds": generation_seconds, "shared_seconds": shared_seconds,
        "outcomes": outcomes, "actual_runner_queries": common.count + len(core.MATCHED) * remaining + 4 * 112,
    }


def plan():
    binding = manifest()
    for root, task in enumerate(binding["tasks"]):
        for repeat in range(2):
            name = base.record_name("plan", root, repeat)
            if (OUT / name).exists():
                load(name)
                continue
            value = plan_one(task, root, repeat)
            if sources() != binding["sources"]:
                raise ValueError("source changed during planning")
            save(name, value)
            print(f"Sealed plane-lineage r{root}/s{repeat}", flush=True)


def batches(row, arm):
    result = row["outcomes"][arm]
    common = row["shared_trace"] if result["uses_shared_cross"] else row["shared_trace"][:1]
    return common + result["trace"]


def evaluate():
    manifest()
    audited = load("audit.json")
    for root, task in enumerate(base.previous.tasks()):
        for repeat in range(2):
            pname, ename = (base.record_name(stage, root, repeat) for stage in ("plan", "evaluate"))
            if audited["plans"][pname] != base.sha(OUT / pname):
                raise ValueError("plan changed after audit")
            if (OUT / ename).exists():
                if load(ename)["plan_sha256"] != base.sha(OUT / pname):
                    raise ValueError("evaluation binding changed")
                continue
            row = load(pname)
            bank = core.PairedBank(np.asarray(row["nominals"]))
            pairs = np.asarray([(i, j) for i in range(16) for j in range(17)])
            paths, events = base.trajectory(task["start"], bank.compose(pairs))
            distances = np.linalg.norm(paths[:, :, :2] - task["goal"], axis=2)
            truth = distances[:, -1].reshape(bank.shape)
            mask = core.support(bank.shape, row["anchors"])
            nominal_best = float(truth[np.arange(16), np.arange(1, 17)].min())
            common_best, pool_best = float(truth[mask].min()), float(truth.min())
            outcomes = {}
            for arm, result in row["outcomes"].items():
                selected, blocked = base.trajectory(task["start"], np.asarray(result["choice"]["chain"])[None])
                cost = float(np.linalg.norm(selected[0, -1, :2] - task["goal"]))
                np.testing.assert_allclose(cost, result["choice"]["score"], atol=1e-12, rtol=0)
                item = {
                    "success": cost < .05, "distance": cost, "trajectory": selected[0].tolist(),
                    "blocked": blocked[0].tolist(), "planning_seconds": result["planning_seconds"],
                    "new_success_over_nominal": cost < .05 <= nominal_best,
                }
                if result["uses_shared_cross"]:
                    proposed = np.asarray(result["proposal_pairs"])
                    values = truth[proposed[:, 0], proposed[:, 1]]
                    np.testing.assert_allclose(values, result["trace"][0]["costs"], atol=1e-12, rtol=0)
                    queried_success = min(common_best, float(values.min())) < .05
                    item.update({
                        "new_success_over_common": cost < .05 <= common_best, "pool_regret": cost-pool_best,
                        "failure_type": ("success" if cost < .05 else "pool_coverage" if pool_best >= .05
                                         else "selection_loss" if queried_success else "proposal_miss"),
                    })
                    if result["prediction"] is not None:
                        prediction = np.asarray(result["prediction"])
                        error = prediction[~mask] - truth[~mask]
                        ranked = core.choose(prediction, mask, int((~mask).sum()))
                        ranking_costs = truth[ranked[:, 0], ranked[:, 1]]
                        item.update({
                            "unqueried_mse": float(np.mean(error ** 2)),
                            "unqueried_mae": float(np.mean(np.abs(error))),
                            "successful_missing_ranks": (np.flatnonzero(ranking_costs < .05)+1).tolist(),
                        })
                outcomes[arm] = item
            save(ename, {
                "root": root, "repeat": repeat, "plan_sha256": base.sha(OUT / pname),
                "truth": truth.tolist(), "full_pool_trajectories": paths.tolist(), "full_pool_events": events.tolist(),
                "nominal_success": nominal_best < .05, "common_success": common_best < .05,
                "pool_success": pool_best < .05, "outcomes": outcomes,
                "shared_queries": row["shared_queries"], "actual_runner_queries": row["actual_runner_queries"],
            })
            print(f"Evaluated plane-lineage r{root}/s{repeat}", flush=True)


def summarize():
    manifest()
    paths = [base.record_name("evaluate", i, j) for i in range(24) for j in range(2)]
    hashes = {p: base.sha(OUT / p) for p in paths}
    if (OUT / "summary.json").exists():
        previous = load("summary.json")
        if previous["evaluations"] != hashes:
            raise ValueError("summary source drift")
        print(json.dumps(previous["decision"], indent=2))
        return
    rows = [load(p) for p in paths]
    arm_summary = {}
    for arm in ARMS:
        results = [row["outcomes"][arm] for row in rows]
        item = {
            "successes": sum(r["success"] for r in results),
            "mean_distance": float(np.mean([r["distance"] for r in results])),
            "median_planning_seconds": float(np.median([r["planning_seconds"] for r in results])),
            "new_successes_over_nominal": sum(r["new_success_over_nominal"] for r in results),
        }
        if "unqueried_mse" in results[0]:
            item["mean_unqueried_mse"] = float(np.mean([r["unqueried_mse"] for r in results]))
        arm_summary[arm] = item
    bootstrap = np.random.default_rng(2839000).integers(0, 24, (10000, 24))
    comparisons = {}
    for arm in ARMS[1:]:
        success = np.asarray([int(row["outcomes"]["cross_interaction"]["success"]) -
                              int(row["outcomes"][arm]["success"]) for row in rows]).reshape(24, 2)
        distance = np.asarray([row["outcomes"]["cross_interaction"]["distance"] -
                               row["outcomes"][arm]["distance"] for row in rows]).reshape(24, 2)
        comparisons[arm] = {
            "wins": int((success > 0).sum()), "losses": int((success < 0).sum()),
            "net_successes": int(success.sum()), "mean_distance_difference": float(distance.mean()),
            "root_success_ci": np.quantile(success.mean(1)[bootstrap].mean(1), [.025, .975]).tolist(),
            "root_distance_ci": np.quantile(distance.mean(1)[bootstrap].mean(1), [.025, .975]).tolist(),
        }
    information = {}
    for arm in ("additive", "shuffled"):
        delta = np.asarray([row["outcomes"]["cross_interaction"]["unqueried_mse"] -
                            row["outcomes"][arm]["unqueried_mse"] for row in rows]).reshape(24, 2).mean(1)
        information[arm] = {"better_roots": int((delta < 0).sum()), "root_mse_differences": delta.tolist()}
    info_pass = all(v["better_roots"] >= 16 for v in information.values())
    component_pass = all(comparisons[a]["net_successes"] >= 4 and comparisons[a]["mean_distance_difference"] < 0
                         for a in ("additive", "shuffled", "uniform_lookahead"))
    summary = {
        "evaluations": hashes, "arms": arm_summary, "comparisons": comparisons, "information": information,
        "coverage": {key: sum(row[key] for row in rows) for key in ("nominal_success", "common_success", "pool_success")},
        "cross_failures": {cat: sum(row["outcomes"]["cross_interaction"]["failure_type"] == cat for row in rows)
                           for cat in ("success", "pool_coverage", "proposal_miss", "selection_loss")},
        "cost": {"per_arm_queries": 128, "per_arm_physics_steps": 1920,
                 "actual_runner_queries": sum(row["actual_runner_queries"] for row in rows),
                 "pool_evaluation_queries": 48*272, "selected_replays": 48*len(ARMS),
                 "gpu_or_visual_system_cost": False},
        "decision": {"information_exit": info_pass, "component_utilization_exit": component_pass,
                     "stop_this_configuration": not (info_pass and component_pass),
                     "full_planner_admission": False, "rgb_admission": False,
                     "gate_promotion": False, "full_research_complete": False},
    }
    save("summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("arms", "coverage", "cross_failures", "decision")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "audit", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Test the unchanged cross descriptor with 2D nominal support; exact-state development only.", flush=True)
        if args.stage in ("plan", "all"):
            plan()
        if args.stage in ("audit", "all"):
            from experiments.pc_plane_lineage_cross.audit import audit
            audit()
        if args.stage in ("evaluate", "all"):
            evaluate()
        if args.stage in ("summarize", "all"):
            summarize()


if __name__ == "__main__":
    main()
