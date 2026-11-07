"""Evaluator-only exhaustive cut support; never exports a deployment selector."""
import argparse
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.pc_cross_query.audit import scalar_trajectory
from experiments.pc_plane_lineage_cross import run as prior

ROOT = prior.ROOT
OUT = ROOT / "results/pc_all_split_support_20261108"
PROTOCOL = ROOT.parent / "docs/PC_ALL_SPLIT_SUPPORT_PROTOCOL_20261108_CODEX.md"
CUTS = tuple(range(16))
PAIRS = np.asarray([(i, j) for i in range(16) for j in range(17)])


def compose(nominals, cut, pairs):
    bank = prior.core.PairedBank(nominals)
    if isinstance(cut, (bool, np.bool_)) or not isinstance(cut, (int, np.integer)) or not 0 <= cut <= 15:
        raise ValueError("integer cut in 0..15 required")
    pairs = prior.core.integer_pairs(pairs, bank.shape)
    tails = np.concatenate((np.zeros((1, 15-cut, 2)), bank.nominals[:, cut:]))
    result = np.concatenate((bank.nominals[pairs[:, 0], :cut], tails[pairs[:, 1]]), axis=1)
    if result.shape != (len(pairs), 15, 2):
        raise ValueError("physical horizon changed")
    return result


def sources():
    return {**prior.sources(), **{str(path): prior.base.sha(path) for path in (
        Path(__file__), PROTOCOL, prior.OUT / "summary.json", prior.OUT / "pool_audit.json",
    )}}


def manifest():
    prior.manifest()
    prior_summary = prior.load("summary.json")
    records = {}
    for ename, expected in prior_summary["evaluations"].items():
        if prior.base.sha(prior.OUT / ename) != expected:
            raise ValueError("prior evaluation changed")
        pname = ename.replace("evaluate_", "plan_")
        records[pname] = prior.base.sha(prior.OUT / pname)
        if prior.load(ename)["plan_sha256"] != records[pname]:
            raise ValueError("prior source-plan binding changed")
        records[ename] = expected
    value = {
        "sources": sources(), "records": records, "cuts": list(CUTS),
        "tasks": prior.base.previous.tasks(), "repeats": 2, "queries_per_cut": 272,
        "steps": 15, "success_threshold": .05,
        "track": "exact_state_evaluator_only", "new_deployment_queries": 0,
        "not_independent_confirmation": True,
    }
    path = OUT / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("all-split source/config drift")
    else:
        prior.base.write_once(path, value)
    return value


def save(name, value):
    prior.base.write_once(OUT / name, {"manifest_sha256": prior.base.sha(OUT / "manifest.json"), **value})


def load(name):
    value = json.loads((OUT / name).read_text())
    if value["manifest_sha256"] != prior.base.sha(OUT / "manifest.json"):
        raise ValueError("record manifest changed")
    return value


def save_arrays(path, **arrays):
    """Atomically publish generated arrays without replacing any existing file."""
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise ValueError("partial or conflicting existing array keys")
            for key, expected in arrays.items():
                np.testing.assert_array_equal(existing[key], expected)
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive atomic publication; existing evidence is never overwritten.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def validate_record(name):
    row = load(name)
    binding = json.loads((OUT / "manifest.json").read_text())
    expected_name = prior.base.record_name("evaluate", row["root"], row["repeat"])
    pname = prior.base.record_name("plan", row["root"], row["repeat"])
    if name != expected_name or row["task"] != binding["tasks"][row["root"]]:
        raise ValueError("row task identity changed")
    if row["source_plan_sha256"] != binding["records"][pname]:
        raise ValueError("row source-plan binding changed")
    if row["source_evaluation_sha256"] != binding["records"][name]:
        raise ValueError("row source-evaluation binding changed")
    array_path = OUT / row["array_file"]
    if prior.base.sha(array_path) != row["array_sha256"]:
        raise ValueError("array hash changed")
    with np.load(array_path, allow_pickle=False) as data:
        if data["trajectories"].shape != (16, 272, 16, 3):
            raise ValueError("incomplete all-cut trajectory")
        if data["actions"].shape != (16, 272, 15, 2) or data["costs"].shape != (16, 16, 17):
            raise ValueError("incomplete all-cut actions or costs")
        if str(data["plan_sha256"]) != row["source_plan_sha256"]:
            raise ValueError("array source identity changed")
        costs = np.linalg.norm(data["trajectories"][:, :, -1, :2] - row["task"]["goal"], axis=2).reshape(16, 16, 17)
        np.testing.assert_allclose(costs, data["costs"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal((costs < .05).any(axis=(1, 2)), row["success_by_cut"])
        np.testing.assert_allclose(costs.reshape(16, -1).min(1), row["min_cost_by_cut"], atol=1e-12, rtol=0)
    return row


def run():
    binding = manifest()
    for root, task in enumerate(binding["tasks"]):
        for repeat in range(2):
            name = prior.base.record_name("evaluate", root, repeat)
            if (OUT / name).exists():
                validate_record(name)
                continue
            pname = prior.base.record_name("plan", root, repeat)
            original = prior.load(pname)
            if original["task"] != task or (original["root"], original["repeat"]) != (root, repeat):
                raise ValueError("task identity mismatch")
            nominals = np.asarray(original["nominals"])
            bank = prior.core.PairedBank(nominals)
            all_actions, all_paths, all_events, all_costs, timings = [], [], [], [], []
            maximum_error = 0.
            for cut in CUTS:
                started = time.perf_counter()
                actions = compose(nominals, cut, PAIRS)
                np.testing.assert_array_equal(compose(nominals, cut, bank.lineage), nominals)
                paths, events = prior.base.trajectory(task["start"], actions)
                model_seconds = time.perf_counter() - started
                started = time.perf_counter()
                independent = np.stack([scalar_trajectory(task["start"], chain) for chain in actions])
                np.testing.assert_allclose(paths, independent, atol=1e-12, rtol=0)
                maximum_error = max(maximum_error, float(np.max(np.abs(paths-independent))))
                audit_seconds = time.perf_counter() - started
                costs = np.linalg.norm(paths[:, -1, :2] - task["goal"], axis=1).reshape(16, 17)
                if cut == 5:
                    previous = prior.load(name)
                    np.testing.assert_array_equal(actions, bank.compose(PAIRS))
                    np.testing.assert_allclose(paths, previous["full_pool_trajectories"], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(costs, previous["truth"], atol=1e-12, rtol=0)
                all_actions.append(actions)
                all_paths.append(paths)
                all_events.append(events)
                all_costs.append(costs)
                timings.append({"cut": cut, "generation_and_model_seconds": model_seconds,
                                "scalar_audit_seconds": audit_seconds})
            costs = np.stack(all_costs)
            nominal_costs = costs[5, np.arange(16), np.arange(1, 17)]
            nominal_success = bool(np.min(nominal_costs) < .05)
            success = (costs < .05).any(axis=(1, 2))
            if success[0] != nominal_success or success[15] != nominal_success:
                raise ValueError("identity cut introduced success")
            if np.any(nominal_success & ~success):
                raise ValueError("nominal lineage lost at a cut")
            npz_name = name.replace(".json", ".npz")
            save_arrays(OUT / npz_name, actions=np.stack(all_actions), trajectories=np.stack(all_paths),
                        collisions=np.stack(all_events), costs=costs, cuts=np.asarray(CUTS), pairs=PAIRS,
                        plan_sha256=np.asarray(binding["records"][pname]))
            if sources() != binding["sources"]:
                raise ValueError("source changed during evaluator run")
            best = np.argmin(costs.reshape(16, -1), axis=1)
            save(name, {
                "root": root, "repeat": repeat, "task": task,
                "source_plan_sha256": binding["records"][pname],
                "source_evaluation_sha256": binding["records"][name],
                "array_file": npz_name, "array_sha256": prior.base.sha(OUT / npz_name),
                "nominal_success": nominal_success, "success_by_cut": success.tolist(),
                "min_cost_by_cut": costs.reshape(16, -1).min(1).tolist(),
                "best_pairs_by_cut_evaluator_only": PAIRS[best].tolist(),
                "successful_cells_by_cut": (costs < .05).sum(axis=(1, 2)).tolist(),
                "union_success": bool(success.any()), "maximum_scalar_error": maximum_error,
                "timings": timings, "all_model_and_scalar_queries_are_evaluation_only": True,
            })
            print(f"All-cut audit r{root}/s{repeat}: k5={bool(success[5])}, union={bool(success.any())}", flush=True)


def summarize():
    manifest()
    names = [prior.base.record_name("evaluate", i, j) for i in range(24) for j in range(2)]
    rows = [validate_record(name) for name in names]
    hashes = {name: prior.base.sha(OUT / name) for name in names}
    if (OUT / "summary.json").exists():
        old = load("summary.json")
        if old["records"] != hashes:
            raise ValueError("summary sources changed")
        print(json.dumps(old["decision"], indent=2))
        return
    success = np.asarray([row["success_by_cut"] for row in rows])
    minima = np.asarray([row["min_cost_by_cut"] for row in rows])
    nominal = np.asarray([row["nominal_success"] for row in rows])
    counts = success.sum(0)
    gains = np.sum(success & ~success[:, 5, None], axis=0)
    union_gain = int(np.sum(success.any(1) & ~success[:, 5]))
    # Deliberately no winner selector is exported for deployment.
    result = {
        "records": hashes, "nominal_success_runs": int(nominal.sum()),
        "success_runs_by_cut": counts.tolist(), "new_success_runs_vs_k5": gains.tolist(),
        "mean_pool_min_by_cut": minima.mean(0).tolist(),
        "all_cut_union_success_runs": int(success.any(1).sum()),
        "all_cut_union_new_success_runs": union_gain,
        "all_cut_union_failure_runs": int(np.sum(~success.any(1))),
        "maximum_scalar_error": max(row["maximum_scalar_error"] for row in rows),
        "evaluation_model_queries": 48*16*272, "evaluation_scalar_queries": 48*16*272,
        "evaluation_model_physics_steps": 48*16*272*15, "new_planning_queries": 0,
        "decision": {
            "material_cut_support_signal": bool(gains.max() >= 4 or union_gain >= 4),
            "scope": "finite-pool evaluator ceiling only; no paid planner tested",
            "posthoc_best_cut_is_not_a_planner": True,
            "gate_promotion": False, "full_research_complete": False,
        },
    }
    save("summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "summarize", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Evaluator-only exhaustive cuts; no oracle cut/chain enters planning.", flush=True)
        if args.stage in ("run", "all"):
            run()
        if args.stage in ("summarize", "all"):
            summarize()


if __name__ == "__main__":
    main()
