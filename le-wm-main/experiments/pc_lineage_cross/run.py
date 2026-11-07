"""Paid complete-program preservation, with isolated post-selection evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from experiments.pc_cross_query import exact_run as old
from experiments.pc_lineage_cross import core

ROOT = old.ROOT
OUT = ROOT / "results/pc_lineage_preserving_cross_20261107"
PROTOCOL = ROOT.parent / "docs/PC_LINEAGE_PRESERVING_CROSS_PROTOCOL_20261107_CODEX.md"
BASELINES = ("uniform_lookahead", "random_pairs", "waypoint_random", "nominal_cem")
ARMS = core.MATCHED + BASELINES
BUDGET = 128


def sources():
    paths = [PROTOCOL, *Path(__file__).parent.glob("*.py")]
    return {**old.dependencies(), **{str(p): old.sha(p) for p in paths}}


def manifest():
    data = {
        "sources": sources(), "tasks": old.previous.tasks(), "repeats": 2,
        "arms": list(ARMS), "budget": BUDGET, "bank_shape": [16, 17],
        "input_track": "exact_state", "independent_confirmation": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(),
                    "threads": [os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("OMP_NUM_THREADS")],
                    "gpu": False},
    }
    path = OUT / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError("frozen source/config drift")
    else:
        old.write_once(path, data)
    return data


def save(name, value):
    old.write_once(OUT / name, {"manifest_sha256": old.sha(OUT / "manifest.json"), **value})


def load(name):
    value = json.loads((OUT / name).read_text())
    if value["manifest_sha256"] != old.sha(OUT / "manifest.json"):
        raise ValueError("record source drift")
    return value


def source_chains(task, seed, count=16):
    offsets = np.random.default_rng(seed).uniform(-2, 2, count)
    chains = np.stack([old.previous.geom.make_chain(task["start"], task["goal"], x)[0] for x in offsets])
    return chains, offsets


def baseline_fill(arm, bank, ledger, task, seed):
    known = np.zeros(bank.shape, bool)
    known[bank.lineage[:, 0], bank.lineage[:, 1]] = True
    updates = []
    if arm == "uniform_lookahead":
        pairs = np.asarray([(i, j) for j in range(bank.shape[1])
                            for i in range(bank.shape[0]) if not known[i, j]])[:112]
        ledger.query(bank.compose(pairs), arm, pairs)
    elif arm == "random_pairs":
        missing = np.argwhere(~known)
        pairs = missing[np.random.default_rng(seed + 40000).choice(len(missing), 112, replace=False)]
        ledger.query(bank.compose(pairs), arm, pairs)
    elif arm == "waypoint_random":
        chains, _ = source_chains(task, seed, 128)
        np.testing.assert_array_equal(chains[:16], bank.nominals)
        ledger.query(chains[16:], arm)
    elif arm == "nominal_cem":
        rng = np.random.default_rng(seed + 60000)
        loc = np.asarray(ledger.best()["chain"])
        std = np.full_like(loc, 1 / 3)
        for iteration in range(7):
            samples = np.clip(loc + rng.normal(size=(16, 15, 2)) * std, -1, 1)
            samples[0] = loc
            costs = ledger.query(samples, f"warm_cem_{iteration}")
            elite = np.argsort(costs, kind="stable")[:4]
            updates.append({"mean": loc.tolist(), "std": std.tolist(), "elite": elite.tolist()})
            loc = samples[elite].mean(0)
            std = np.maximum(samples[elite].std(0), 1e-5)
    else:
        raise ValueError("unknown baseline")
    return updates


def plan_one(task, root, repeat):
    seed = 2831000 + root * 100 + repeat
    started = time.perf_counter()
    nominals, offsets = source_chains(task, seed)
    bank = core.PairedBank(nominals)
    generation_seconds = time.perf_counter() - started
    model = old.evaluator(task)
    common = core.Ledger(model, BUDGET)
    started = time.perf_counter()
    observed, anchors = core.acquire(bank, common)
    shared_seconds = time.perf_counter() - started
    if not 113 <= common.count <= 116:
        raise ValueError("cross support cost outside predeclared range")
    mask = core.support(bank.shape, anchors)
    remaining = BUDGET - common.count
    nominal_best = min(common.trace[0]["costs"])
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
        private_seconds = time.perf_counter() - started
        outcomes[arm] = {
            "uses_shared_cross": True, "trace": ledger.trace[3:],
            "prediction": None if predicted is None else predicted.tolist(),
            "diagnostic": diagnostic, "proposal_pairs": pairs.tolist(),
            "choice": choice, "queries": ledger.count,
            "planning_seconds": generation_seconds + shared_seconds + allocation_seconds + private_seconds,
            "allocation_seconds": allocation_seconds,
        }
    for arm in BASELINES:
        ledger = core.Ledger(model, BUDGET, [common.trace[0]], 16)
        started = time.perf_counter()
        updates = baseline_fill(arm, bank, ledger, task, seed)
        choice = ledger.best()
        outcomes[arm] = {
            "uses_shared_cross": False, "trace": ledger.trace[1:],
            "updates": updates, "choice": choice, "queries": ledger.count,
            "planning_seconds": generation_seconds + common.trace[0]["query_wall_seconds"]
                                + time.perf_counter() - started,
        }
    for arm, result in outcomes.items():
        if result["queries"] != 128 or result["choice"]["score"] > nominal_best + 1e-12:
            raise ValueError(f"{arm}: budget or nominal incumbent loss")
    return {
        "root": root, "repeat": repeat, "seed": seed, "task": task,
        "nominals": nominals.tolist(), "offsets": offsets.tolist(),
        "nominal_bytes_sha256": hashlib.sha256(nominals.tobytes()).hexdigest(),
        "anchors": anchors.tolist(),
        "observed": [[float(x) if np.isfinite(x) else None for x in row] for row in observed],
        "shared_trace": common.trace, "shared_queries": common.count,
        "generation_seconds": generation_seconds, "shared_seconds": shared_seconds,
        "outcomes": outcomes, "actual_runner_queries": common.count + 7 * remaining + 4 * 112,
    }


def plan():
    binding = manifest()
    for root, task in enumerate(binding["tasks"]):
        for repeat in range(2):
            name = old.record_name("plan", root, repeat)
            if (OUT / name).exists():
                load(name)
                continue
            row = plan_one(task, root, repeat)
            if binding["sources"] != sources():
                raise ValueError("source changed during run")
            save(name, row)
            print(f"sealed r{root}/s{repeat}: common_queries={row['shared_queries']} "
                  f"cross={row['outcomes']['cross_interaction']['choice']['score']:.5f} "
                  f"warm_CEM={row['outcomes']['nominal_cem']['choice']['score']:.5f}", flush=True)


def batches(row, arm):
    result = row["outcomes"][arm]
    return (row["shared_trace"] if result["uses_shared_cross"] else row["shared_trace"][:1]) + result["trace"]


def evaluate():
    manifest()
    audited = load("audit.json")
    for root, task in enumerate(old.previous.tasks()):
        for repeat in range(2):
            pname = old.record_name("plan", root, repeat)
            if audited["plans"][pname] != old.sha(OUT / pname):
                raise ValueError("plan changed after auditing")
            name = old.record_name("evaluate", root, repeat)
            if (OUT / name).exists():
                if load(name)["plan_sha256"] != old.sha(OUT / pname):
                    raise ValueError("evaluation source drift")
                continue
            row = load(pname)
            bank = core.PairedBank(np.asarray(row["nominals"]))
            pairs = np.asarray([(i, j) for i in range(16) for j in range(17)])
            trajectory, events = old.trajectory(task["start"], bank.compose(pairs))
            distances = np.linalg.norm(trajectory[:, :, :2] - task["goal"], axis=2)
            truth = distances[:, -1].reshape(bank.shape)
            nominal_costs = truth[np.arange(16), np.arange(1, 17)]
            nominal_best = float(nominal_costs.min())
            mask = core.support(bank.shape, row["anchors"])
            common_best = float(truth[mask].min())
            outcomes = {}
            for arm, result in row["outcomes"].items():
                path, blocked = old.trajectory(task["start"], np.asarray(result["choice"]["chain"])[None])
                distance = float(np.linalg.norm(path[0, -1, :2] - task["goal"]))
                if abs(distance - result["choice"]["score"]) > 1e-12:
                    raise ValueError("selected chain score mismatch")
                item = {
                    "distance": distance, "success": distance < .05,
                    "trajectory": path[0].tolist(), "blocked": blocked[0].tolist(),
                    "planning_seconds": result["planning_seconds"],
                    "new_success_over_nominal": distance < .05 <= nominal_best,
                    "incumbent_loss": distance > nominal_best + 1e-12,
                }
                if result["uses_shared_cross"]:
                    p = np.asarray(result["proposal_pairs"])
                    candidate_scores = truth[p[:, 0], p[:, 1]]
                    np.testing.assert_allclose(candidate_scores, result["trace"][0]["costs"], atol=1e-12, rtol=0)
                    queried_success = min(common_best, float(candidate_scores.min())) < .05
                    item.update({
                        "pool_regret": distance - float(truth.min()),
                        "new_success_over_common": distance < .05 <= common_best,
                        "failure_type": ("success" if distance < .05 else "pool_coverage"
                                         if truth.min() >= .05 else "selection_loss" if queried_success else "proposal_miss"),
                    })
                    if result["prediction"] is not None:
                        prediction = np.asarray(result["prediction"])
                        error = prediction[~mask] - truth[~mask]
                        sorted_missing = core.choose(prediction, mask, int((~mask).sum()))
                        ordered = truth[sorted_missing[:, 0], sorted_missing[:, 1]]
                        item.update({"unqueried_mse": float(np.mean(error ** 2)),
                                     "unqueried_mae": float(np.mean(np.abs(error))),
                                     "successful_missing_ranks": (np.flatnonzero(ordered < .05) + 1).tolist()})
                outcomes[arm] = item
            save(name, {
                "root": root, "repeat": repeat, "plan_sha256": old.sha(OUT / pname),
                "truth": truth.tolist(), "all_distances": distances.tolist(),
                "full_pool_trajectories": trajectory.tolist(), "full_pool_events": events.tolist(),
                "nominal_success": nominal_best < .05, "common_success": common_best < .05,
                "pool_success": bool(truth.min() < .05),
                "shared_queries": row["shared_queries"], "actual_runner_queries": row["actual_runner_queries"],
                "outcomes": outcomes,
            })
            print(f"evaluated sealed r{root}/s{repeat}", flush=True)


def summarize():
    manifest()
    paths = [old.record_name("evaluate", root, repeat) for root in range(24) for repeat in range(2)]
    source_hashes = {name: old.sha(OUT / name) for name in paths}
    if (OUT / "summary.json").exists():
        previous = load("summary.json")
        if previous["evaluations"] != source_hashes:
            raise ValueError("evaluation sources changed")
        print(json.dumps(previous["decision"], indent=2))
        return
    rows = [load(name) for name in paths]
    arm_summary = {}
    for arm in ARMS:
        data = [row["outcomes"][arm] for row in rows]
        arm_summary[arm] = {
            "successes": sum(x["success"] for x in data),
            "mean_distance": float(np.mean([x["distance"] for x in data])),
            "median_planning_seconds": float(np.median([x["planning_seconds"] for x in data])),
            "new_successes_over_nominal": sum(x["new_success_over_nominal"] for x in data),
            "incumbent_loss_runs": sum(x["incumbent_loss"] for x in data),
        }
        if "unqueried_mse" in data[0]:
            arm_summary[arm]["mean_unqueried_mse"] = float(np.mean([x["unqueried_mse"] for x in data]))
    boot = np.random.default_rng(2839000).integers(0, 24, (10000, 24))
    comparisons = {}
    for arm in ARMS:
        if arm == "cross_interaction":
            continue
        success = np.array([int(row["outcomes"]["cross_interaction"]["success"])
                            - int(row["outcomes"][arm]["success"]) for row in rows]).reshape(24, 2)
        distance = np.array([row["outcomes"]["cross_interaction"]["distance"]
                             - row["outcomes"][arm]["distance"] for row in rows]).reshape(24, 2)
        comparisons[arm] = {
            "wins": int((success > 0).sum()), "losses": int((success < 0).sum()),
            "net_successes": int(success.sum()), "mean_distance_difference": float(distance.mean()),
            "root_success_ci": np.quantile(success.mean(1)[boot].mean(1), [.025, .975]).tolist(),
            "root_distance_ci": np.quantile(distance.mean(1)[boot].mean(1), [.025, .975]).tolist(),
        }
    information = {}
    for other in ("additive", "shuffled"):
        delta = np.array([row["outcomes"]["cross_interaction"]["unqueried_mse"]
                          - row["outcomes"][other]["unqueried_mse"] for row in rows]).reshape(24, 2).mean(1)
        information[other] = {"better_roots": int((delta < 0).sum()), "root_mse_differences": delta.tolist()}
    info_pass = all(x["better_roots"] >= 16 for x in information.values())
    deploy_pass = all(x["net_successes"] >= 4 and x["mean_distance_difference"] < 0
                      for x in comparisons.values())
    summary = {
        "evaluations": source_hashes, "arms": arm_summary, "comparisons": comparisons, "information": information,
        "coverage": {key: sum(row[key] for row in rows) for key in ("nominal_success", "common_success", "pool_success")},
        "cross_failures": {category: sum(row["outcomes"]["cross_interaction"]["failure_type"] == category for row in rows)
                           for category in ("success", "pool_coverage", "proposal_miss", "selection_loss")},
        "cost": {
            "per_arm_queries": 128, "per_arm_physics_steps": 1920,
            "shared_counts": [row["shared_queries"] for row in rows],
            "actual_runner_queries": sum(row["actual_runner_queries"] for row in rows),
            "evaluation_pool_queries": 48 * 272, "selected_replay_queries": 48 * len(ARMS),
            "gpu_or_full_visual_cost": False,
        },
        "decision": {
            "lineage_engineering_invariant_pass": all(x["incumbent_loss_runs"] == 0 for x in arm_summary.values()),
            "information_exit": info_pass, "deployment_exit": deploy_pass,
            "admit_mechanism_and_rgb": info_pass and deploy_pass,
            "stop_this_configuration": not (info_pass and deploy_pass),
            "gate_promotion": False, "full_research_complete": False,
        },
    }
    save("summary.json", summary)
    print(json.dumps({"arms": arm_summary, "coverage": summary["coverage"],
                      "decision": summary["decision"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "audit", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ("plan", "all"):
            print("Test paid full-program retention and descriptor increments at 128 queries; exact-state only.", flush=True)
            plan()
        if args.stage in ("audit", "all"):
            from experiments.pc_lineage_cross.audit import audit
            audit()
        if args.stage in ("evaluate", "all"):
            evaluate()
        if args.stage in ("summarize", "all"):
            summarize()


if __name__ == "__main__":
    main()
