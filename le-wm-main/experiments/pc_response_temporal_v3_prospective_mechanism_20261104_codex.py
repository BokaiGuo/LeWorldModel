"""Prospective v3 open-loop same-chain mechanism development panel."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_search_e0_dev_20261104_codex as base
from experiments import pc_response_temporal_v3_fixed_terminal_e0_20261104_codex as e0
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_feedback_v3 as method

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_PROSPECTIVE_MECHANISM_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_v3_prospective_mechanism_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")
REBUILD = PREFIX.with_name(PREFIX.name + "_intervention_rebuild.json")
ARMS = ("response", "matched_random", "smooth_cem")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.40, -.10, .20, .40), (-.9, .15, .9), (-.30, .30))]


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(method.__file__), e0.PLANS, e0.AUDIT)
    return {**base.dependencies(), **{str(p.resolve()): base.sha(p) for p in paths}}


def plan():
    if any(path.exists() for path in (MANIFEST, PLANS, AUDIT, INTERVENTION, REBUILD)):
        raise FileExistsError("existing or partial prospective experiment")
    old = base.tasks.tasks()
    if any(task in old for task in tasks()) or len(tasks()) != 24:
        raise ValueError("predeclared task grid overlaps cited development panel")
    if json.loads(e0.AUDIT.read_text())["plans_sha256"] != base.sha(e0.PLANS):
        raise ValueError("v3 exact panel seal mismatch")
    hashes = dependencies()
    base.write(MANIFEST, {"dependencies": hashes, "tasks": tasks(), "arms": ARMS, "method": method.METHOD,
                          "budget": 1920, "scope": "prospective grid; wider provenance unknown"})
    print("Prospective same-chain mechanism development: seal all plans before intervention.", flush=True)
    rows = []
    for root, task in enumerate(tasks()):
        for arm in ARMS:
            meter, trace = Meter(1920), []
            goal = np.asarray(task["goal"])

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)

                def evaluate(batch):
                    positions = base.exact.endpoints(task, batch)
                    return positions, np.linalg.norm(positions - goal, axis=1)

                positions, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                              "positions": positions.tolist(), "scores": scores.tolist(),
                              "tokens": len(raw) * raw.shape[1] // 5})
                return positions, scores

            result = method.decide(arm, root, 0, goal, predict, meter)
            rows.append({"root": root, "arm": arm, "task": task, "result": result,
                         "counts": meter.as_dict(), "trace": trace})
        print(f"root={root} three arms planned", flush=True)
    if hashes != dependencies():
        raise ValueError("dependency drift")
    base.write(PLANS, {"manifest_sha256": base.sha(MANIFEST), "dependencies": hashes, "rows": rows})
    print("All 72 plans sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies() or data["manifest_sha256"] != base.sha(MANIFEST):
        raise ValueError("seal mismatch")
    seen = set()
    for row in data["rows"]:
        key = row["root"], row["arm"]
        if key in seen or row["task"] != tasks()[row["root"]] or row["arm"] not in ARMS:
            raise ValueError("task or arm mismatch")
        seen.add(key)
        chain = np.asarray(row["result"]["selected"]["chain"])
        selected = row["result"]["selected"]["terminal_score"]
        count, full, found = 0, [], False
        for batch in row["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
            if not np.array_equal(requested.astype(np.float32), raw):
                raise ValueError("float conversion mismatch")
            positions = base.exact.endpoints(row["task"], raw)
            scores = np.linalg.norm(positions - np.asarray(row["task"]["goal"]), axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
            cost = len(raw) * raw.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("token mismatch")
            count += cost
            if raw.shape[1] == 15:
                full.extend(scores.tolist())
                found |= any(scores[i] == selected for i in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
        if (not found or min(full) != selected or count > 1920
                or count != row["counts"]["prediction_token_samples"]
                or len(row["trace"]) != row["counts"]["prediction_calls"]):
            raise ValueError("selection or accounting mismatch")
    if seen != {(root, arm) for root in range(24) for arm in ARMS}:
        raise ValueError("incomplete panel")
    base.write(AUDIT, {"plans_sha256": base.sha(PLANS), "verified_decisions": len(seen)})
    print("All 72 plans independently recalculated.", flush=True)


def calculate():
    plans = json.loads(PLANS.read_text())
    if (plans["dependencies"] != dependencies()
            or json.loads(AUDIT.read_text())["plans_sha256"] != base.sha(PLANS)):
        raise ValueError("plans not sealed")
    source = {(row["root"], row["arm"]): row for row in plans["rows"]}
    rows = []
    for root, task in enumerate(tasks()):
        row = source[root, "response"]
        chain = np.asarray(row["result"]["selected"]["chain"], dtype=np.float32)
        greedy = int(np.argmin(row["result"]["prefix_scores"]))
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(row["result"]["first_bank"][greedy]["prefix"]),
                    "random_bridge": np.asarray(source[root, "matched_random"]["result"]["selected"]["chain"][:5]),
                    "lookahead": np.asarray(source[root, "smooth_cem"]["result"]["selected"]["chain"][:5])}
        suffixes = [np.tile(value, (10, 1)) for value in
                    ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        goal = np.asarray(task["goal"])
        controls = {}
        for name, prefix in prefixes.items():
            def predict(actions):
                pos = base.exact.endpoints(task, np.asarray(actions, dtype=np.float32))
                return pos, np.linalg.norm(pos - goal, axis=1)

            prefix_distance = float(predict(np.asarray(prefix)[None])[1][0])
            joined = np.stack([np.concatenate((prefix, suffix)) for suffix in suffixes])
            _, common = predict(joined)
            best, score, generations = base.method.cem(np.zeros((10, 2)), prefix, 119,
                np.random.default_rng(2027111400 + root), predict, float(common[0]), 4)
            axes = np.eye(10).reshape(10, 5, 2)
            tails = np.zeros((20, 10, 2))
            tails[0::2, :5], tails[1::2, :5] = .05 * axes, -.05 * axes
            positions, _ = predict(np.concatenate((np.broadcast_to(prefix, (20, 5, 2)), tails), axis=1))
            response = ((positions[0::2] - positions[1::2]) / .1).T
            controls[name] = {"prefix": np.asarray(prefix).tolist(), "prefix_distance": prefix_distance,
                              "common_distances": common.tolist(), "active_changes": (common-common[0]).tolist(),
                              "own_chain": best.tolist(), "own_distance": score, "generations": generations,
                              "successor_response": response.tolist(), "diagnostic_rollouts": 146}
        initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))
        chosen, others = controls["selected"], [name for name in prefixes if name != "selected"]
        nonprogress = chosen["prefix_distance"] >= initial - 1e-10 and np.any(chain[:5] != 0)
        active = {name: bool(chosen["active_changes"][-1] < controls[name]["active_changes"][-1] - 1e-10)
                  for name in others}
        own = {name: bool(chosen["own_distance"] < controls[name]["own_distance"] - 1e-10) for name in others}
        gaps = {name: float(np.linalg.norm(np.asarray(chosen["successor_response"])
                                           - np.asarray(controls[name]["successor_response"]))) for name in others}
        def passes(names):
            return bool(nonprogress and all(active[name] and own[name] and gaps[name] > 1e-6 for name in names))
        rows.append({"root": root, "task": task, "selected_chain_float32": chain.tolist(),
                     "initial_distance": initial, "controls": controls, "nonprogress": bool(nonprogress),
                     "active_better": active, "own_better": own, "response_gap": gaps,
                     "base_intersection": passes(("zero", "greedy")), "all_intersection": passes(others)})
    return rows


def intervene(rebuild=False):
    output = REBUILD if rebuild else INTERVENTION
    if output.exists() or (not rebuild and REBUILD.exists()):
        raise FileExistsError(output)
    hashes = dependencies()
    rows = calculate()
    if hashes != dependencies():
        raise ValueError("dependency drift")
    if rebuild:
        original = json.loads(INTERVENTION.read_text())
        if original["dependencies"] != hashes or original["rows"] != rows:
            raise ValueError("intervention reconstruction mismatch")
        base.write(output, {"intervention_sha256": base.sha(INTERVENTION), "verified_roots": len(rows),
                            "scope": "same-code deterministic reconstruction"})
    else:
        base.write(output, {"dependencies": hashes, "plans_sha256": base.sha(PLANS), "rows": rows})
    print(f"{'Rebuilt' if rebuild else 'Sealed'} interventions for {len(rows)} roots.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit, "intervene": intervene, "rebuild": lambda: intervene(True)}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan, audit, intervene or rebuild")
    commands[sys.argv[1]]()
