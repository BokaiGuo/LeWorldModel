"""Predeclared exposed-root temporal response search and same-chain intervention."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_block_program_wall_prospective_dev_20261103_codex as tasks
from experiments import pc_successor_response_composition_e0_dev_20261103_codex as exact
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as method

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_SEARCH_E0_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_search_e0_dev_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
MECHANISM = PREFIX.with_name(PREFIX.name + "_mechanism.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    files = (Path(__file__), PROTOCOL, Path(method.__file__), Path(tasks.__file__),
             Path(exact.__file__), Path(exact.wall.__file__), Path(exact.wall.physics.__file__),
             Path(Meter.__module__.replace(".", "/") + ".py"),
             ROOT / "experiments/pc_budget_allocation_20261103_codex.py")
    return {str(Path(p).resolve()): sha(p) for p in files}


def write(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)


def planner(task, root, arm):
    meter = Meter(method.CAP)
    trace = []
    goal = np.asarray(task["goal"])

    def predict(actions):
        requested = np.asarray(actions, dtype=float)
        raw = requested.astype(np.float32)

        def evaluate(batch):
            positions = exact.endpoints(task, batch)
            return positions, np.linalg.norm(positions - goal, axis=1)

        positions, scores = meter.predict(raw, evaluate)
        trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                      "positions": positions.tolist(), "scores": scores.tolist(),
                      "tokens": len(raw) * raw.shape[1] // 5})
        return positions, scores

    started = time.monotonic()
    result = method.decide(arm, root, goal, predict, meter)
    return {"root": root, "task": task, "arm": arm, "result": result,
            "counts": meter.as_dict(), "wall_seconds": time.monotonic() - started, "trace": trace}


def plan():
    if any(path.exists() for path in (MANIFEST, PLANS, AUDIT, MECHANISM)):
        raise FileExistsError("partial or complete panel exists")
    hashes = dependencies()
    write(MANIFEST, {"dependencies": hashes, "tasks": tasks.tasks(), "arms": method.ARMS,
                     "budget": method.CAP, "scope": "exposed E0 development, not confirmation"})
    print("Plan four fixed arms on 24 exposed exact tasks; no witness injection.", flush=True)
    rows = []
    for root, task in enumerate(tasks.tasks()):
        for arm in method.ARMS:
            rows.append(planner(task, root, arm))
        print(f"root={root} four arms complete", flush=True)
    if hashes != dependencies():
        raise ValueError("dependency drift")
    write(PLANS, {"manifest_sha256": sha(MANIFEST), "dependencies": hashes, "rows": rows})
    print("Planning panel sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies() or data["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("source binding mismatch")
    rows = []
    for row in data["rows"]:
        selected = np.asarray(row["result"]["selected"]["chain"], dtype=float)
        selected_score = row["result"]["selected"]["terminal_score"]
        goal = np.asarray(row["task"]["goal"])
        tokens, found, all_scores = 0, False, []
        for batch in row["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
            if not np.array_equal(requested.astype(np.float32), raw):
                raise ValueError("float32 conversion mismatch")
            position = exact.endpoints(row["task"], raw)
            scores = np.linalg.norm(position - goal, axis=1)
            np.testing.assert_allclose(position, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
            cost = len(raw) * raw.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("batch token mismatch")
            tokens += cost
            if raw.shape[1] == 15:
                all_scores.extend(scores.tolist())
                matches = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(scores[i] - selected_score) < 1e-12 for i in matches)
        if (not found or abs(min(all_scores) - selected_score) > 1e-12
                or tokens != row["counts"]["prediction_token_samples"] or tokens > method.CAP
                or len(row["trace"]) != row["counts"]["prediction_calls"]):
            raise ValueError("selection or budget mismatch")
        rows.append({"root": row["root"], "arm": row["arm"], "tokens": tokens,
                     "score": selected_score, "global_scored_min": min(all_scores)})
    if len(rows) != 24 * len(method.ARMS):
        raise ValueError("panel size mismatch")
    write(AUDIT, {"plans_sha256": sha(PLANS), "rows": rows})
    print(f"Audited {len(rows)} plans and every prediction batch.", flush=True)


def intervene():
    if MECHANISM.exists():
        raise FileExistsError(MECHANISM)
    data, audit_data = json.loads(PLANS.read_text()), json.loads(AUDIT.read_text())
    if data["dependencies"] != dependencies() or audit_data["plans_sha256"] != sha(PLANS):
        raise ValueError("planning seal mismatch")
    by_root = {(row["root"], row["arm"]): row for row in data["rows"]}
    rows = []
    for root in range(24):
        row = by_root[root, "response"]
        task, result = row["task"], row["result"]
        chain = np.asarray(result["selected"]["chain"])
        goal = np.asarray(task["goal"])
        greedy = int(np.argmin(result["prefix_scores"]))
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(result["first_bank"][greedy]["prefix"]),
                    "random_bridge": np.asarray(by_root[root, "matched_random"]["result"]["selected"]["chain"])[:5],
                    "lookahead": np.asarray(by_root[root, "smooth_cem"]["result"]["selected"]["chain"])[:5]}
        suffixes = [np.tile(value, (10, 1)) for value in
                    ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        controls = {}
        for name, prefix in prefixes.items():
            predictions = [0]

            def predict(actions):
                raw = np.asarray(actions, dtype=np.float32)
                position = exact.endpoints(task, raw)
                predictions[0] += len(raw)
                return position, np.linalg.norm(position - goal, axis=1)

            initial = float(np.linalg.norm(exact.endpoints(task, prefix[None].astype(np.float32))[0] - goal))
            joined = np.stack([np.concatenate((prefix, suffix)) for suffix in suffixes])
            _, common = predict(joined)
            base = np.zeros((10, 2))
            best, score, generations = method.cem(base, prefix, 119, np.random.default_rng(2027111400 + root),
                                                  predict, float(common[0]), 4)
            # The common all-zero suffix is the first of the 120 allowed searches.
            axis = np.eye(10).reshape(10, 5, 2)
            tails = np.zeros((20, 10, 2))
            tails[0::2, :5] = .05 * axis
            tails[1::2, :5] = -.05 * axis
            probes = np.concatenate((np.broadcast_to(prefix, (20, 5, 2)), tails), axis=1)
            pos, _ = predict(probes)
            matrix = ((pos[0::2] - pos[1::2]) / .1).T
            controls[name] = {"prefix": prefix.tolist(), "prefix_distance": initial,
                              "common_distances": common.tolist(), "active_changes": (common-common[0]).tolist(),
                              "own_chain": best.tolist(), "own_distance": score, "generations": generations,
                              "successor_response": matrix.tolist(), "diagnostic_rollouts": predictions[0] + 1}
        initial_distance = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))
        chosen = controls["selected"]
        nonprogress = (chosen["prefix_distance"] >= initial_distance - 1e-10
                       and np.any(np.asarray(chosen["prefix"], dtype=np.float32) != 0))
        active = {name: chosen["active_changes"][-1] < controls[name]["active_changes"][-1] - 1e-10
                  for name in prefixes if name != "selected"}
        own = {name: chosen["own_distance"] < controls[name]["own_distance"] - 1e-10
               for name in prefixes if name != "selected"}
        response_gap = {name: float(np.linalg.norm(np.asarray(chosen["successor_response"])
                                      - np.asarray(controls[name]["successor_response"])))
                        for name in prefixes if name != "selected"}
        rows.append({"root": root, "task": task, "chain": chain.tolist(), "controls": controls,
                     "initial_distance": initial_distance, "nonprogress": bool(nonprogress),
                     "active_better": active, "own_better": own, "response_gap": response_gap,
                     "base_intersection": bool(nonprogress and all(active[x] and own[x] for x in ("zero", "greedy"))),
                     "all_intersection": bool(nonprogress and all(active[x] and own[x] for x in active))})
        print(f"mechanism root={root}", flush=True)
    write(MECHANISM, {"plans_sha256": sha(PLANS), "audit_sha256": sha(AUDIT), "rows": rows})
    print("Mechanism interventions sealed.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit", "intervene"):
        raise SystemExit("choose plan, audit or intervene")
    {"plan": plan, "audit": audit, "intervene": intervene}[sys.argv[1]]()
