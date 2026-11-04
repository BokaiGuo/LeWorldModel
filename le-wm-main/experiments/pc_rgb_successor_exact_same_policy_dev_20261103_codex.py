"""Run the unchanged RGB successor policy on exact dynamics, then intervene."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_block_program_wall_prospective_dev_20261103_codex as panel
from experiments import pc_successor_response_composition_e0_dev_20261103_codex as exact
from experiments import pc_successor_rgb_execution_dev_20261103_codex as policy
from experiments.pc_budget_allocation_20261103_codex import optimize

ROOT = policy.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_SAME_POLICY_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_rgb_successor_exact_same_policy_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    files = [PROTOCOL, Path(__file__), Path(policy.__file__), Path(policy.policy.__file__),
             Path(policy.policy.generator.__file__), Path(policy.fallback.__file__),
             Path(exact.__file__), Path(exact.wall.__file__), Path(exact.wall.physics.__file__),
             Path(panel.__file__), Path(optimize.__code__.co_filename)]
    return {str(path): sha(path) for path in files}


class Budget:
    def __init__(self):
        self.token_samples = 0

    def predict(self, task, actions):
        actions = np.asarray(actions, dtype=np.float32)
        count = len(actions) * actions.shape[1] // 5
        if self.token_samples + count > 480:
            raise ValueError("prediction cap exceeded")
        positions = exact.endpoints(task, actions)
        scores = np.linalg.norm(positions - task["goal"], axis=1)
        self.token_samples += count
        return positions, scores


def plan():
    if any(path.exists() for path in (PLANS, AUDIT, INTERVENTION)):
        raise FileExistsError("existing sealed output")
    source, rows = dependencies(), []
    for root, task in enumerate(panel.tasks()):
        meter = Budget()
        result = policy.decide("successor", root, 0, np.asarray(task["goal"]),
                               lambda actions: meter.predict(task, actions), meter)
        rows.append({"root": root, "task": task, "tokens": meter.token_samples, "result": result})
    if source != dependencies():
        raise ValueError("source changed")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": source, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} complete exact-dynamics decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    rows = []
    for row in data["rows"]:
        selected = np.asarray(row["result"]["selected"]["chain"])
        found, tokens = False, 0
        for batch in row["result"]["prediction_trace"]:
            requested = np.asarray(batch["requested_actions"])
            actions = np.asarray(batch["model_actions"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), actions)
            positions = exact.endpoints(row["task"], actions)
            scores = np.linalg.norm(positions - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            if requested.shape[1:] == selected.shape:
                indexes = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(scores[i] - row["result"]["selected"]["terminal_score"]) < 1e-12
                             for i in indexes)
            count = len(actions) * actions.shape[1] // 5
            if count != batch["tokens"]:
                raise ValueError("token mismatch")
            tokens += count
        if not found or tokens != row["tokens"] or tokens > 480:
            raise ValueError("selection or budget mismatch")
        rows.append({"root": row["root"], "tokens": tokens, "batches": len(row["result"]["prediction_trace"])})
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": rows}, stream, indent=2)
    print(f"Audited {len(rows)} decision traces.", flush=True)


def intervene():
    if INTERVENTION.exists():
        raise FileExistsError(INTERVENTION)
    data, audit_data = json.loads(PLANS.read_text()), json.loads(AUDIT.read_text())
    if data["dependencies"] != dependencies() or audit_data["plans_sha256"] != sha(PLANS):
        raise ValueError("seal mismatch")
    rows = []
    for row in data["rows"]:
        task, result = row["task"], row["result"]
        goal = np.asarray(task["goal"])
        chain = np.asarray(result["selected"]["chain"])
        first = [np.asarray(item["prefix"]) for item in result["first_bank"]]
        distances = np.linalg.norm(exact.endpoints(task, np.asarray(first)) - goal, axis=1)
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": first[int(np.argmin(distances))]}
        tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        tails.append(chain[5:])
        controls = {}
        for name, prefix in prefixes.items():
            pscore = float(np.linalg.norm(exact.endpoints(task, prefix[None])[0] - goal))
            actions = np.asarray([np.concatenate((prefix, tail)) for tail in tails])
            values = np.linalg.norm(exact.endpoints(task, actions) - goal, axis=1)
            best, score, scores = optimize(
                prefix, 15, 120, np.random.default_rng(2026121000 + row["root"]),
                lambda actions: np.linalg.norm(exact.endpoints(task, actions) - goal, axis=1))
            controls[name] = {"prefix": prefix.tolist(), "prefix_distance": pscore,
                              "common_scores": values.tolist(), "active_changes": (values - values[0]).tolist(),
                              "own_chain": best.tolist(), "own_score": score, "own_scores": scores}
        initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))
        sel = controls["selected"]
        nonprogress = sel["prefix_distance"] >= initial - 1e-10
        active = all(sel["active_changes"][-1] < controls[k]["active_changes"][-1] - 1e-10
                     for k in ("zero", "greedy"))
        own = all(sel["own_score"] < controls[k]["own_score"] - 1e-10 for k in ("zero", "greedy"))
        rows.append({"root": row["root"], "chain": chain.tolist(), "initial_distance": initial,
                     "first_bank_distances": distances.tolist(), "controls": controls,
                     "nonprogress": bool(nonprogress), "active_response": bool(active),
                     "own_advantage": bool(own), "limited_witness": bool(nonprogress and active and own)})
    with INTERVENTION.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "audit_sha256": sha(AUDIT), "rows": rows},
                  stream, indent=2, allow_nan=False)
    print(f"Intervened on {len(rows)} selected chains.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit, "intervene": intervene}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan, audit, intervene")
    commands[sys.argv[1]]()
