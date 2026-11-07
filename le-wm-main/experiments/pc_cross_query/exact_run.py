"""Sealed, resumable crossed-query development; evaluation is a later stage."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_descriptor_combination_proposal_20261107_codex as previous
from experiments.pc_cross_query import core
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/pc_cross_query_interaction_20261107"
PROTOCOL = ROOT.parent / "docs/PC_CROSS_QUERY_INTERACTION_PROTOCOL_20261107_CODEX.md"
BASELINES = ("uniform_lookahead", "random_pairs", "waypoint_random", "step_cem", "block_cem")
ARMS = core.MATCHED_ARMS + BASELINES
REPEATS = 2
BUDGET = 128


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    modules = (previous, previous.base, previous.base.exact,
               previous.base.exact.wall, previous.base.exact.wall.physics,
               previous.geom, previous.geom.baseline)
    paths = [PROTOCOL, ROOT / "experiments/research_runtime/artifacts.py"]
    paths += list(Path(__file__).parent.glob("*.py"))
    paths += [Path(module.__file__) for module in modules]
    return {str(path): sha(path) for path in sorted(set(paths))}


def write_once(path, value):
    if path.exists():
        raise FileExistsError(path)
    # Reject NaN in persistent evidence. Unknown observations are explicit nulls.
    json.dumps(value, allow_nan=False)
    atomic_write_json(path, value)


def manifest():
    path = OUT / "manifest.json"
    payload = {
        "dependencies": dependencies(), "tasks": previous.tasks(),
        "repeats": REPEATS, "arms": list(ARMS), "budget": BUDGET,
        "input_track": "exact_state", "independent_confirmation": False,
        "planning_model": "analytic_wall_unicycle_15_steps",
        "no_rgb_or_frozen_visual_gate_promotion": True,
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform(),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "gpu_used": False,
        },
    }
    if not path.exists():
        write_once(path, payload)
    elif json.loads(path.read_text()) != payload:
        raise ValueError("manifest/source/config drift")
    return payload


def record_name(stage, root, repeat):
    return f"{stage}_r{root:02d}_s{repeat}.json"


def load_record(name):
    record = json.loads((OUT / name).read_text())
    if record["manifest_sha256"] != sha(OUT / "manifest.json"):
        raise ValueError("record manifest mismatch")
    return record


def save_record(name, value):
    write_once(OUT / name, {"manifest_sha256": sha(OUT / "manifest.json"), **value})


def trajectory(start, actions):
    """Exactly H transitions; no historical padding transition is charged away."""
    actions = np.asarray(actions, float)
    if actions.ndim != 3 or actions.shape[-1] != 2 or not np.isfinite(actions).all():
        raise ValueError("finite batch of actions required")
    if np.max(np.abs(actions)) > 1:
        raise ValueError("actions out of bounds")
    state = np.broadcast_to(start, (len(actions), 3)).copy()
    trace = [state.copy()]
    blocked = []
    for a in np.swapaxes(actions * previous.base.exact.wall.HIGH, 0, 1):
        displacement = a[:, :1] * np.stack((np.cos(state[:, 2]), np.sin(state[:, 2])), axis=1)
        state[:, :2], event = previous.base.exact.wall.physics.step_batch(state[:, :2], displacement)
        state[:, 2] += a[:, 1]
        trace.append(state.copy())
        blocked.append(event)
    return np.stack(trace, axis=1), np.stack(blocked, axis=1)


def evaluator(task):
    def score(actions):
        started = time.perf_counter()
        traces, _ = trajectory(task["start"], actions)
        endpoints = traces[:, -1, :2]
        costs = np.linalg.norm(endpoints - task["goal"], axis=1)
        return costs, endpoints, time.perf_counter() - started
    return score


def bank(task, seed):
    offsets = np.random.default_rng(seed).uniform(-2, 2, 39)
    chains = np.stack([previous.geom.make_chain(task["start"], task["goal"], x)[0] for x in offsets])
    prefixes = chains[:8, :5]
    tails = np.concatenate((np.zeros((1, 10, 2)), chains[8:, 5:]), axis=0)
    core.validate_bank(prefixes, tails)
    return prefixes, tails, offsets


def run_cem(ledger, seed, blockwise):
    rng = np.random.default_rng(seed)
    shape = (3, 2) if blockwise else (15, 2)
    loc, scale = np.zeros(shape), np.full(shape, 1 / 3)
    updates = []
    for iteration in range(8):
        samples = np.clip(loc + rng.normal(size=(16, *shape)) * scale, -1, 1)
        samples[0] = loc
        actions = np.repeat(samples, 5, axis=1) if blockwise else samples
        costs = ledger.query(actions, f"cem_{iteration}")
        indices = np.argsort(costs, kind="stable")[:4]
        elite = samples[indices]
        updates.append({"iteration": iteration, "mean": loc.tolist(), "std": scale.tolist(),
                        "elite_indices": indices.tolist()})
        loc, scale = elite.mean(0), np.maximum(elite.std(0), 1e-5)
    return updates


def plan_one(task, root, repeat):
    seed = 2831000 + 100 * root + repeat
    started = time.perf_counter()
    prefixes, tails, offsets = bank(task, seed)
    generation_seconds = time.perf_counter() - started
    score = evaluator(task)
    common = core.Ledger(score, 116)
    started = time.perf_counter()
    observed, anchors = core.acquire_cross(prefixes, tails, common)
    common_seconds = time.perf_counter() - started
    mask = core.cross_mask(anchors)
    outcomes = {}
    for arm in core.MATCHED_ARMS:
        started = time.perf_counter()
        diagnostics = {}
        if arm == "cross_interaction":
            prediction, diagnostics = core.interaction_prediction(observed, anchors)
        elif arm == "additive":
            prediction = core.additive_prediction(observed, anchors)
        elif arm == "row_mean":
            prediction = np.broadcast_to(observed[:, :4].mean(1, keepdims=True), (8, 32)).copy()
        elif arm == "probe_min":
            prediction = np.broadcast_to(observed[:, :4].min(1, keepdims=True), (8, 32)).copy()
        elif arm == "shuffled":
            changed = core.shuffle_identity(observed, anchors, seed + 20000)
            prediction, diagnostics = core.interaction_prediction(changed, anchors)
        else:
            prediction = None
        if prediction is None:
            missing = np.argwhere(~mask)
            pairs = missing[np.random.default_rng(seed + 30000).choice(len(missing), 12, replace=False)]
        else:
            prediction[mask] = observed[mask]
            pairs = core.proposals(prediction, mask)
        allocation_seconds = time.perf_counter() - started
        ledger = core.Ledger(score, BUDGET, list(common.trace), common.count)
        started = time.perf_counter()
        ledger.query(core.compose(prefixes, tails, pairs), "proposals", pairs)
        choice = ledger.best()
        private_seconds = time.perf_counter() - started
        if ledger.count != BUDGET or choice["score"] > common.best()["score"] + 1e-12:
            raise ValueError("budget or incumbent violation")
        outcomes[arm] = {
            "prediction": None if prediction is None else prediction.tolist(),
            "diagnostics": diagnostics, "proposal_pairs": pairs.tolist(), "uses_shared_cross": True,
            "trace": ledger.trace[2:], "choice": choice,
            "query_count": ledger.count, "dynamics_steps": ledger.count * 15,
            "planning_seconds": generation_seconds + common_seconds + allocation_seconds + private_seconds,
            "allocation_seconds": allocation_seconds,
        }
    for arm in BASELINES:
        ledger = core.Ledger(score, BUDGET)
        started = time.perf_counter()
        updates = []
        if arm == "uniform_lookahead":
            pairs = np.asarray([(i, j) for i in range(8) for j in range(16)])
            ledger.query(core.compose(prefixes, tails, pairs), arm, pairs)
        elif arm == "random_pairs":
            ids = np.random.default_rng(seed + 40000).choice(256, 128, replace=False)
            pairs = np.column_stack(np.unravel_index(ids, (8, 32)))
            ledger.query(core.compose(prefixes, tails, pairs), arm, pairs)
        elif arm == "waypoint_random":
            extra_offsets = np.random.default_rng(seed + 50000).uniform(-2, 2, 128)
            actions = np.stack([previous.geom.make_chain(task["start"], task["goal"], x)[0]
                                for x in extra_offsets])
            ledger.query(actions, arm)
        else:
            updates = run_cem(ledger, seed + 60000, arm == "block_cem")
        choice = ledger.best()
        elapsed = time.perf_counter() - started
        outcomes[arm] = {
            "trace": ledger.trace, "choice": choice, "uses_shared_cross": False,
            "updates": updates, "query_count": ledger.count, "dynamics_steps": ledger.count * 15,
            "planning_seconds": elapsed + (generation_seconds if arm in BASELINES[:2] else 0.),
        }
    return {
        "root": root, "repeat": repeat, "seed": seed, "task": task,
        "prefixes": prefixes.tolist(), "tails": tails.tolist(), "offsets": offsets.tolist(),
        "bank_sha256": hashlib.sha256(np.concatenate((prefixes.ravel(), tails.ravel())).tobytes()).hexdigest(),
        "anchors": anchors.tolist(), "a_columns": core.A_COLUMNS.tolist(),
        "observed": [[float(x) if np.isfinite(x) else None for x in row] for row in observed],
        "shared_trace": common.trace, "shared_choice": common.best(),
        "outcomes": outcomes, "generation_seconds": generation_seconds,
        "shared_seconds": common_seconds,
        "actual_runner_queries": 116 + 6 * 12 + 5 * 128,
    }


def plan():
    payload = manifest()
    for root, task in enumerate(payload["tasks"]):
        for repeat in range(REPEATS):
            name = record_name("plan", root, repeat)
            if (OUT / name).exists():
                load_record(name)
                continue
            row = plan_one(task, root, repeat)
            if payload["dependencies"] != dependencies():
                raise ValueError("source changed during planning")
            save_record(name, row)
            print(f"sealed plan root={root} repeat={repeat}: "
                  f"cross={row['outcomes']['cross_interaction']['choice']['score']:.5f}, "
                  f"additive={row['outcomes']['additive']['choice']['score']:.5f}", flush=True)


def evaluate():
    manifest()
    audit = load_record("audit.json")
    for root, task in enumerate(previous.tasks()):
        for repeat in range(REPEATS):
            plan_name = record_name("plan", root, repeat)
            if audit["plans"][plan_name] != sha(OUT / plan_name):
                raise ValueError("plan changed after audit")
            name = record_name("evaluate", root, repeat)
            if (OUT / name).exists():
                old = load_record(name)
                if old["plan_sha256"] != sha(OUT / plan_name):
                    raise ValueError("evaluation source mismatch")
                continue
            row = load_record(plan_name)
            pairs = np.asarray([(i, j) for i in range(8) for j in range(32)])
            actions = core.compose(row["prefixes"], row["tails"], pairs)
            traces, blocked = trajectory(task["start"], actions)
            distances = np.linalg.norm(traces[:, :, :2] - task["goal"], axis=2)
            truth = distances[:, -1].reshape(8, 32)
            mask = core.cross_mask(row["anchors"])
            common_score = row["shared_choice"]["score"]
            outcomes = {}
            for arm, outcome in row["outcomes"].items():
                chosen = np.asarray(outcome["choice"]["chain"])
                selected_trace, events = trajectory(task["start"], chosen[None])
                terminal = float(np.linalg.norm(selected_trace[0, -1, :2] - task["goal"]))
                if abs(terminal - outcome["choice"]["score"]) > 1e-12:
                    raise ValueError("selected-chain replay mismatch")
                item = {
                    "terminal_distance": terminal, "success": terminal < .05,
                    "selected_trajectory": selected_trace[0].tolist(), "blocked": events[0].tolist(),
                    "planning_seconds": outcome["planning_seconds"],
                    "new_success_beyond_common": terminal < .05 <= common_score,
                }
                if outcome["uses_shared_cross"]:
                    proposals = np.asarray(outcome["proposal_pairs"])
                    pcost = truth[proposals[:, 0], proposals[:, 1]]
                    np.testing.assert_allclose(pcost, outcome["trace"][0]["costs"], atol=1e-12, rtol=0)
                    item.update({
                        "candidate_pool_regret": terminal - float(truth.min()),
                        "proposal_min": float(pcost.min()),
                        "proposal_improves_common": bool(pcost.min() < common_score - 1e-12),
                        "proposal_wins_final": outcome["choice"]["query_index"] >= 116,
                    })
                    if outcome["prediction"] is not None:
                        predicted = np.asarray(outcome["prediction"])
                        error = predicted[~mask] - truth[~mask]
                        top = core.proposals(predicted, mask)
                        item.update({
                            "unqueried_mse": float(np.mean(error ** 2)),
                            "unqueried_mae": float(np.mean(np.abs(error))),
                            "proposed_missing_regret": float(truth[top[:, 0], top[:, 1]].min() - truth[~mask].min()),
                        })
                outcomes[arm] = item
            save_record(name, {
                "root": root, "repeat": repeat, "plan_sha256": sha(OUT / plan_name),
                "truth": truth.tolist(), "all_distances": distances.tolist(),
                "full_pool_trajectories": traces.tolist(), "full_pool_blocked": blocked.tolist(),
                "outcomes": outcomes, "common_success": common_score < .05,
                "pool_success": bool(truth.min() < .05),
                "missing_can_add_success": bool(truth[~mask].min() < .05 <= common_score),
                "evaluation_only_full_pool_queries": 256,
                "selected_replay_queries": len(ARMS),
            })
            print(f"evaluated sealed choices root={root} repeat={repeat}", flush=True)


def summarize():
    manifest()
    rows = [load_record(record_name("evaluate", root, repeat))
            for root in range(24) for repeat in range(REPEATS)]
    sources = {record_name("evaluate", root, repeat): sha(OUT / record_name("evaluate", root, repeat))
               for root in range(24) for repeat in range(REPEATS)}
    if (OUT / "summary.json").exists():
        old = load_record("summary.json")
        if old["evaluations"] != sources:
            raise ValueError("summary source mismatch")
        print(json.dumps(old["decision"], indent=2))
        return
    arms, comparisons = {}, {}
    for arm in ARMS:
        items = [row["outcomes"][arm] for row in rows]
        arms[arm] = {
            "successes": sum(x["success"] for x in items), "runs": len(items),
            "mean_distance": float(np.mean([x["terminal_distance"] for x in items])),
            "median_planning_seconds": float(np.median([x["planning_seconds"] for x in items])),
            "new_successes_beyond_common": sum(x["new_success_beyond_common"] for x in items),
        }
        if arm in core.MATCHED_ARMS:
            arms[arm].update({
                "proposal_improved_common": sum(x["proposal_improves_common"] for x in items),
                "mean_pool_regret": float(np.mean([x["candidate_pool_regret"] for x in items])),
            })
        if arm not in ("random_missing", *BASELINES):
            arms[arm]["mean_unqueried_mse"] = float(np.mean([x["unqueried_mse"] for x in items]))
    rng = np.random.default_rng(2839000)
    boot = rng.integers(0, 24, (10000, 24))
    for arm in ARMS:
        if arm == "cross_interaction":
            continue
        gains = np.array([int(r["outcomes"]["cross_interaction"]["success"])
                          - int(r["outcomes"][arm]["success"]) for r in rows]).reshape(24, 2)
        differences = np.array([r["outcomes"]["cross_interaction"]["terminal_distance"]
                                - r["outcomes"][arm]["terminal_distance"] for r in rows]).reshape(24, 2)
        comparisons[arm] = {
            "wins": int((gains > 0).sum()), "losses": int((gains < 0).sum()),
            "net_successes": int(gains.sum()),
            "success_rate_difference": float(gains.mean()),
            "root_bootstrap_success_ci": np.quantile(gains.mean(1)[boot].mean(1), [.025, .975]).tolist(),
            "mean_distance_difference": float(differences.mean()),
            "root_bootstrap_distance_ci": np.quantile(differences.mean(1)[boot].mean(1), [.025, .975]).tolist(),
        }
    information = {}
    for comparator in ("additive", "shuffled"):
        delta = np.array([r["outcomes"]["cross_interaction"]["unqueried_mse"]
                          - r["outcomes"][comparator]["unqueried_mse"] for r in rows]).reshape(24, 2).mean(1)
        information[comparator] = {"better_roots": int((delta < 0).sum()), "root_mean_mse_differences": delta.tolist()}
    information_pass = all(x["better_roots"] >= 16 for x in information.values())
    deployment_pass = all(x["net_successes"] >= 4 and x["mean_distance_difference"] < 0
                          for x in comparisons.values())
    summary = {
        "evaluations": sources, "arms": arms, "paired_comparisons": comparisons,
        "information": information,
        "coverage": {
            "common_success_runs": sum(r["common_success"] for r in rows),
            "full_pool_success_runs": sum(r["pool_success"] for r in rows),
            "missing_can_add_success_runs": sum(r["missing_can_add_success"] for r in rows),
        },
        "cost": {
            "deployment_queries_per_arm": 128, "deployment_dynamics_steps_per_arm": 1920,
            "actual_planning_runner_queries": 48 * (116 + 6 * 12 + 5 * 128),
            "evaluation_full_pool_queries": 48 * 256, "selected_chain_replay_queries": 48 * len(ARMS),
            "audit_recomputations_excluded_from_deployment": True,
            "visual_gpu_or_complete_closed_loop_cost": False,
        },
        "decision": {
            "information_exit": information_pass, "deployment_exit": deployment_pass,
            "admit_same_version_mechanism_and_rgb": information_pass and deployment_pass,
            "stop_expansion_of_this_configuration": not (information_pass and deployment_pass),
            "full_research_complete": False, "gate_promotion": False,
        },
    }
    save_record("summary.json", summary)
    print(json.dumps({"arms": arms, "decision": summary["decision"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "audit", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ("plan", "all"):
            print("Test whether current-task consequence interactions improve actually queried plans; "
                  "exact-state development only.", flush=True)
            plan()
        if args.stage in ("audit", "all"):
            from experiments.pc_cross_query.audit import audit
            audit()
        if args.stage in ("evaluate", "all"):
            evaluate()
        if args.stage in ("summarize", "all"):
            summarize()


if __name__ == "__main__":
    main()
