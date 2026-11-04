"""Retrospective intervention on actual three-packet feedback execution."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_fixed_terminal_e0_20261104_codex as feedback
from experiments.spectral_reachability import pc_response_temporal_search_v1 as method

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_EXECUTED_CHAIN_MECHANISM_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_executed_chain_mechanism_20261104_codex"
OUTPUT = PREFIX.with_name(PREFIX.name + "_intervention.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    paths = (Path(__file__), PROTOCOL, feedback.PLANS, feedback.AUDIT, feedback.MANIFEST)
    return {**feedback.dependencies(), **{str(p.resolve()): sha(p) for p in paths}}


def calculate():
    plans, audit = json.loads(feedback.PLANS.read_text()), json.loads(feedback.AUDIT.read_text())
    if (plans["dependencies"] != feedback.dependencies() or audit["plans_sha256"] != sha(feedback.PLANS)):
        raise ValueError("sealed feedback mismatch")
    source = {(row["root"], row["arm"]): row for row in plans["rows"]}
    result_rows = []
    for root in range(24):
        row = source[root, "response"]
        task, goal = row["task"], np.asarray(row["task"]["goal"])
        chain = np.concatenate([np.asarray(d["executed_float32"]) for d in row["decisions"]])
        terminal = feedback.previous.exact.endpoints(task, chain[None].astype(np.float32))[0]
        np.testing.assert_allclose(terminal, row["final_state"][:2], rtol=0, atol=1e-12)
        first = row["decisions"][0]["decision"]["result"]
        greedy = int(np.argmin(first["prefix_scores"]))
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(first["first_bank"][greedy]["prefix"]),
                    "random_bridge": np.asarray(source[root, "matched_random"]["decisions"][0]["executed_float32"]),
                    "lookahead": np.asarray(source[root, "smooth_cem"]["decisions"][0]["executed_float32"])}
        suffixes = [np.tile(value, (10, 1)) for value in
                    ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        controls = {}
        for name, prefix in prefixes.items():
            def predict(actions):
                position = feedback.previous.exact.endpoints(task, np.asarray(actions, dtype=np.float32))
                return position, np.linalg.norm(position - goal, axis=1)

            prefix_distance = float(predict(prefix[None])[1][0])
            joined = np.stack([np.concatenate((prefix, suffix)) for suffix in suffixes])
            _, common = predict(joined)
            best, score, generations = method.cem(np.zeros((10, 2)), prefix, 119,
                np.random.default_rng(2027111400 + root), predict, float(common[0]), 4)
            axes = np.eye(10).reshape(10, 5, 2)
            tails = np.zeros((20, 10, 2))
            tails[0::2, :5], tails[1::2, :5] = .05 * axes, -.05 * axes
            positions, _ = predict(np.concatenate((np.broadcast_to(prefix, (20, 5, 2)), tails), axis=1))
            matrix = ((positions[0::2] - positions[1::2]) / .1).T
            controls[name] = {"prefix": prefix.tolist(), "prefix_distance": prefix_distance,
                              "common_distances": common.tolist(), "active_changes": (common-common[0]).tolist(),
                              "own_chain": best.tolist(), "own_distance": score, "generations": generations,
                              "successor_response": matrix.tolist(), "diagnostic_rollouts": 146}
        initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))
        selected = controls["selected"]
        nonprogress = selected["prefix_distance"] >= initial - 1e-10 and np.any(chain[:5] != 0)
        others = [name for name in prefixes if name != "selected"]
        active = {name: selected["active_changes"][-1] < controls[name]["active_changes"][-1] - 1e-10
                  for name in others}
        own = {name: selected["own_distance"] < controls[name]["own_distance"] - 1e-10 for name in others}
        gaps = {name: float(np.linalg.norm(np.asarray(selected["successor_response"])
                         - np.asarray(controls[name]["successor_response"]))) for name in others}
        result_rows.append({"root": root, "task": task, "executed_chain_float32": chain.tolist(),
                            "terminal_distance": row["final_distance"], "initial_distance": initial,
                            "controls": controls, "nonprogress": bool(nonprogress),
                            "active_better": active, "own_better": own, "response_gap": gaps,
                            "base_intersection": bool(nonprogress and all(active[x] and own[x] for x in ("zero", "greedy"))),
                            "all_intersection": bool(nonprogress and all(active[x] and own[x] for x in others))})
    return result_rows


def main(stage):
    target = OUTPUT if stage == "run" else AUDIT
    if target.exists() or (stage == "run" and AUDIT.exists()):
        raise FileExistsError(target)
    hashes = dependencies()
    if stage == "audit":
        previous = json.loads(OUTPUT.read_text())
        if previous["dependencies"] != hashes:
            raise ValueError("dependency mismatch")
    print(f"{stage}: actual feedback chain interventions on all 24 exposed roots", flush=True)
    rows = calculate()
    if hashes != dependencies():
        raise ValueError("dependency drift")
    if stage == "run":
        value = {"dependencies": hashes, "rows": rows}
    else:
        if rows != previous["rows"]:
            raise ValueError("deterministic reconstruction mismatch")
        value = {"intervention_sha256": sha(OUTPUT), "verified_roots": len(rows),
                 "scope": "same-code deterministic reconstruction, not independent replication"}
    feedback.write(target, value)
    print(f"{stage}: sealed {len(rows)} roots", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    main(sys.argv[1])
