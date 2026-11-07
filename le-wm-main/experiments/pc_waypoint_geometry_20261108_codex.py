"""Generator-support audit, not an action-descriptor or RGB planning claim."""
import argparse
import fcntl
import json
import platform
import time
from pathlib import Path

import numpy as np

from experiments.pc_cross_query import exact_run as base
from experiments.pc_cross_query.audit import scalar_trajectory

ROOT = base.ROOT
OUT = ROOT / "results/pc_waypoint_geometry_20261108"
PROTOCOL = ROOT.parent / "docs/PC_WAYPOINT_GEOMETRY_PROTOCOL_20261108_CODEX.md"
ARMS = ("line_1d", "plane_2d", "lineage_mixture")
PRIOR = ROOT / "results/pc_lineage_preserving_cross_20261107"


def compile_waypoint(start, goal, waypoint):
    start, goal, waypoint = map(lambda x: np.asarray(x, float), (start, goal, waypoint))
    if start.shape != (3,) or goal.shape != (2,) or waypoint.shape != (2,):
        raise ValueError("pose, goal, waypoint shapes")
    if not np.isfinite(np.r_[start, goal, waypoint]).all():
        raise ValueError("nonfinite geometric input")
    position, heading = start[:2].copy(), float(start[2])
    actions, boundaries = [], []
    high = base.previous.base.exact.wall.HIGH
    for point in (waypoint, goal):
        displacement = point - position
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-12:
            boundaries.append(len(actions))
            continue
        angle = float(np.arctan2(displacement[1], displacement[0]))
        turn = float((angle - heading + np.pi) % (2 * np.pi) - np.pi)
        turns = int(np.ceil(abs(turn) / high[1]))
        if turns:
            actions.extend([[0., turn / turns / high[1]]] * turns)
        moves = int(np.ceil(distance / high[0]))
        actions.extend([[distance / moves / high[0], 0.]] * moves)
        position, heading = point, angle
        boundaries.append(len(actions))
    return np.asarray((actions + [[0., 0.]] * 15)[:15]), len(actions), boundaries


def generate(task, lateral, longitudinal):
    start, goal = np.asarray(task["start"]), np.asarray(task["goal"])
    delta = goal - start[:2]
    if np.linalg.norm(delta) <= 1e-12:
        raise ValueError("distinct initial and goal positions required")
    waypoint = (start[:2] + goal) / 2 + lateral * np.array([-delta[1], delta[0]]) + longitudinal * delta
    chain, required, boundaries = compile_waypoint(start, goal, waypoint)
    return chain, waypoint, required, boundaries


def free_trajectory(start, actions):
    """Independent no-wall integration: exposes compiler and time truncation errors."""
    state = np.broadcast_to(start, (len(actions), 3)).copy()
    trace = [state.copy()]
    for commands in np.swapaxes(np.asarray(actions) * [.3, np.pi / 2], 0, 1):
        state[:, 0] += commands[:, 0] * np.cos(state[:, 2])
        state[:, 1] += commands[:, 0] * np.sin(state[:, 2])
        state[:, 2] += commands[:, 1]
        trace.append(state.copy())
    return np.stack(trace, axis=1)


def sources():
    return {**base.dependencies(), str(Path(__file__)): base.sha(__file__),
            str(PROTOCOL): base.sha(PROTOCOL),
            str(PRIOR / "manifest.json"): base.sha(PRIOR / "manifest.json")}


def manifest():
    value = {"sources": sources(), "tasks": base.previous.tasks(), "arms": list(ARMS),
             "repeats": 2, "queries": 128, "physics_steps": 15,
             "track": "exact_state_generator_development", "numpy": np.__version__,
             "python": platform.python_version(), "independent_confirmation": False}
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
        raise ValueError("record binding mismatch")
    return value


def plan():
    fixed = manifest()
    for root, task in enumerate(fixed["tasks"]):
        for repeat in range(2):
            name = base.record_name("plan", root, repeat)
            if (OUT / name).exists():
                load(name)
                continue
            seed = 2831000 + 100 * root + repeat
            lateral = np.random.default_rng(seed).uniform(-2, 2, 128)
            longitudinal = np.random.default_rng(seed + 70000).uniform(-1, 1, 128)
            arms = {}
            for arm in ARMS:
                started = time.perf_counter()
                longitudinal_used = longitudinal.copy()
                if arm == "line_1d":
                    longitudinal_used[:] = 0
                elif arm == "lineage_mixture":
                    longitudinal_used[:16] = 0
                programs = [generate(task, a, b) for a, b in zip(lateral, longitudinal_used, strict=True)]
                actions = np.stack([x[0] for x in programs])
                if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                    raise ValueError("illegal compiled action")
                generation_seconds = time.perf_counter() - started
                score, endpoints, model_seconds = base.evaluator(task)(actions)
                started = time.perf_counter()
                best = int(np.argmin(score))
                selection_seconds = time.perf_counter() - started
                if arm == "line_1d":
                    legacy = np.stack([base.previous.geom.make_chain(task["start"], task["goal"], a)[0]
                                       for a in lateral])
                    np.testing.assert_array_equal(actions, legacy)
                if arm == "lineage_mixture":
                    np.testing.assert_array_equal(actions[:16], np.asarray(arms["line_1d"]["actions"])[:16])
                    if score[best] > np.min(score[:16]):
                        raise ValueError("paid incumbent lost")
                arms[arm] = {
                    "actions": actions.tolist(), "waypoints": [x[1].tolist() for x in programs],
                    "required_steps": [x[2] for x in programs], "segment_ends": [x[3] for x in programs],
                    "longitudinal": longitudinal_used.tolist(), "costs": score.tolist(),
                    "endpoints": endpoints.tolist(), "choice": best, "queries": 128,
                    "generation_seconds": generation_seconds, "model_seconds": model_seconds,
                    "selection_seconds": selection_seconds,
                    "planning_seconds": generation_seconds + model_seconds + selection_seconds,
                }
            if fixed["sources"] != sources():
                raise ValueError("source changed during planning")
            save(name, {"root": root, "repeat": repeat, "task": task, "seed": seed,
                        "lateral": lateral.tolist(), "arms": arms})
            print(f"sealed r{root}/s{repeat}: " + " ".join(
                f"{a}={arms[a]['costs'][arms[a]['choice']]:.4f}" for a in ARMS), flush=True)


def evaluate():
    manifest()
    names = [base.record_name("plan", root, repeat) for root in range(24) for repeat in range(2)]
    if not all((OUT / name).is_file() for name in names):
        raise ValueError("all plans must be sealed before evaluation")
    for root in range(24):
        for repeat in range(2):
            name = base.record_name("evaluate", root, repeat)
            pname = base.record_name("plan", root, repeat)
            if (OUT / name).exists():
                if load(name)["plan_sha256"] != base.sha(OUT / pname):
                    raise ValueError("evaluation drift")
                continue
            plan_record = load(pname)
            task, outcomes = plan_record["task"], {}
            prior_path = PRIOR / pname
            prior = json.loads(prior_path.read_text())
            if prior["manifest_sha256"] != base.sha(PRIOR / "manifest.json"):
                raise ValueError("prior reference not bound")
            for arm in ARMS:
                result = plan_record["arms"][arm]
                actions = np.asarray(result["actions"])
                if actions.shape != (128, 15, 2) or result["queries"] != 128:
                    raise ValueError("query or horizon mismatch")
                path, blocked = base.trajectory(task["start"], actions)
                distance = np.linalg.norm(path[:, -1, :2] - task["goal"], axis=1)
                np.testing.assert_allclose(distance, result["costs"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(path[:, -1, :2], result["endpoints"], atol=1e-12, rtol=0)
                if result["choice"] != int(np.argmin(distance)):
                    raise ValueError("best queried chain lost")
                free_path = free_trajectory(task["start"], actions)
                free_cost = np.linalg.norm(free_path[:, -1, :2] - task["goal"], axis=1)
                required = np.asarray(result["required_steps"])
                within = required <= 15
                if np.any(free_cost[within] > 1e-10):
                    raise ValueError("compiler failure in obstacle-free dynamics")
                selected = result["choice"]
                scalar = scalar_trajectory(task["start"], actions[selected])
                np.testing.assert_allclose(scalar, path[selected], atol=1e-12, rtol=0)
                if arm == "line_1d":
                    old_batches = [prior["shared_trace"][0], *prior["outcomes"]["waypoint_random"]["trace"]]
                    np.testing.assert_array_equal(actions, np.concatenate([x["actions"] for x in old_batches]))
                    np.testing.assert_allclose(distance, np.concatenate([x["costs"] for x in old_batches]), atol=1e-12, rtol=0)
                failed, collision, free_failed = distance >= .05, blocked.any(1), free_cost >= .05
                categories = {
                    "collision_and_free_failure": int(np.sum(failed & collision & free_failed)),
                    "collision_only": int(np.sum(failed & collision & ~free_failed)),
                    "free_failure_only": int(np.sum(failed & ~collision & free_failed)),
                    "other_failure": int(np.sum(failed & ~collision & ~free_failed)),
                    "success": int(np.sum(~failed)),
                }
                if sum(categories.values()) != 128:
                    raise ValueError("failure denominator mismatch")
                outcomes[arm] = {
                    "success": bool(distance[selected] < .05), "distance": float(distance[selected]),
                    "planning_seconds": result["planning_seconds"], "candidate_categories": categories,
                    "truncated_candidates": int(np.sum(required > 15)),
                    "all_costs": distance.tolist(), "free_costs": free_cost.tolist(),
                    "full_trajectories": path.tolist(), "blocked": blocked.tolist(),
                    "selected_scalar_trace": scalar.tolist(),
                    "max_free_compiler_error_complete": float(free_cost[within].max(initial=0)),
                }
            witnesses = []
            if repeat == 0:
                for sign in (-1, 1):
                    chain, required, _ = compile_waypoint(task["start"], task["goal"], [0., sign * 1.05])
                    path, events = base.trajectory(task["start"], chain[None])
                    score = float(np.linalg.norm(path[0, -1, :2] - task["goal"]))
                    witnesses.append({"waypoint": [0., sign * 1.05], "actions": chain.tolist(),
                                      "required_steps": required, "distance": score, "trajectory": path[0].tolist(),
                                      "blocked": events[0].tolist(), "success": score < .05,
                                      "privileged_evaluator_only": True})
            save(name, {"root": root, "repeat": repeat, "plan_sha256": base.sha(OUT / pname),
                        "prior_plan_sha256": base.sha(prior_path), "outcomes": outcomes,
                        "privileged_witnesses": witnesses})
            print(f"audited and replayed r{root}/s{repeat}", flush=True)


def summarize():
    manifest()
    names = [base.record_name("evaluate", root, repeat) for root in range(24) for repeat in range(2)]
    hashes = {name: base.sha(OUT / name) for name in names}
    if (OUT / "summary.json").exists():
        old = load("summary.json")
        if old["evaluations"] != hashes:
            raise ValueError("summary drift")
        print(json.dumps(old["decision"], indent=2))
        return
    rows = [load(name) for name in names]
    summaries = {}
    for arm in ARMS:
        results = [x["outcomes"][arm] for x in rows]
        summaries[arm] = {
            "successes": sum(x["success"] for x in results), "mean_distance": float(np.mean([x["distance"] for x in results])),
            "median_planning_seconds": float(np.median([x["planning_seconds"] for x in results])),
            "truncated_candidates": sum(x["truncated_candidates"] for x in results),
            "candidate_categories": {key: sum(x["candidate_categories"][key] for x in results)
                                     for key in results[0]["candidate_categories"]},
            "max_free_compiler_error_complete": max(x["max_free_compiler_error_complete"] for x in results),
        }
    boot = np.random.default_rng(2839000).integers(0, 24, (10000, 24))
    comparisons = {}
    for arm in ARMS[1:]:
        dsuccess = np.array([int(x["outcomes"][arm]["success"]) - int(x["outcomes"]["line_1d"]["success"])
                             for x in rows]).reshape(24, 2)
        ddistance = np.array([x["outcomes"][arm]["distance"] - x["outcomes"]["line_1d"]["distance"]
                              for x in rows]).reshape(24, 2)
        comparisons[arm] = {
            "wins": int((dsuccess > 0).sum()), "losses": int((dsuccess < 0).sum()),
            "net_successes": int(dsuccess.sum()), "mean_distance_difference": float(ddistance.mean()),
            "root_bootstrap_success_ci": np.quantile(dsuccess.mean(1)[boot].mean(1), [.025, .975]).tolist(),
            "root_bootstrap_distance_ci": np.quantile(ddistance.mean(1)[boot].mean(1), [.025, .975]).tolist(),
        }
    primary = comparisons["lineage_mixture"]
    passed = primary["net_successes"] >= 4 and primary["mean_distance_difference"] < 0
    result = {
        "evaluations": hashes, "arms": summaries, "comparisons": comparisons,
        "witness_tasks_solved": sum(any(w["success"] and w["required_steps"] <= 15 for w in x["privileged_witnesses"])
                                    for x in rows if x["repeat"] == 0),
        "cost": {"per_arm_queries": 128, "per_arm_dynamics_steps": 1920, "actual_planning_queries": 48 * 3 * 128,
                 "evaluation_full_replays": 48 * 3 * 128, "privileged_witness_queries": 48},
        "decision": {"generator_development_exit": passed, "descriptor_increment_tested": False,
                     "rgb_or_full_planner_admission": False, "gate_promotion": False, "full_research_complete": False},
    }
    save("summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Test waypoint support, compiler parity and equal-budget coverage; exact-state only.", flush=True)
        if args.stage in ("plan", "all"):
            plan()
        if args.stage in ("evaluate", "all"):
            evaluate()
        if args.stage in ("summarize", "all"):
            summarize()


if __name__ == "__main__":
    main()
