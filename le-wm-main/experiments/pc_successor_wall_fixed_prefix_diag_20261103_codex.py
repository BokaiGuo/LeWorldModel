"""Post-hoc equal-opportunity suffix diagnostics for sealed wall prefixes."""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_successor_response_composition_e0_dev_20261103_codex as prior
from experiments.pc_traced_adaptive_rgb_dev_20261103_codex import SuffixState

PROTOCOL = prior.policy.ROOT.parent / "docs/PC_SUCCESSOR_WALL_FIXED_PREFIX_DIAG_PROTOCOL_20261103_CODEX.md"
PREFIX = prior.policy.ROOT / "results/pc_successor_wall_fixed_prefix_diag_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_runs.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
GROUPS = ("successor_response", "successor_nominal", "successor_random", "no_composition")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    paths = (PROTOCOL, Path(__file__), Path(inspect.getfile(SuffixState)), prior.PLANS, prior.AUDIT)
    return {str(path): sha(path) for path in paths}


def candidates_for_root(rows, root):
    result = []
    subset = [row for row in rows if row["root"] == root and row["budget"] == 960]
    for group in GROUPS:
        source = next(row for row in subset if row["arm"] == group)
        for entry in source["bank"]:
            result.append({"group": group, "name": entry["name"], "prefix": entry["prefix"],
                           "source": "sealed_generated_bank"})
    task = subset[0]["task"]
    witnesses = [item for item in prior.wall.feasibility(task)
                 if item["within_horizon"] and item["steps_needed"] <= 15]
    witness = min(witnesses, key=lambda x: x["steps_needed"])
    actions = np.asarray(witness["actions"])[:15] / prior.wall.HIGH
    if np.linalg.norm(prior.endpoints(task, actions[None])[0] - task["goal"]) > 1e-10:
        raise ValueError("oracle diagnostic witness does not replay")
    for length in (5, 10):
        result.append({"group": "oracle_diagnostic", "name": f"witness_prefix_{length}",
                       "prefix": actions[:length].tolist(), "source": "privileged_feasibility"})
    return task, result


def run_candidate(task, root, candidate):
    goal = np.asarray(task["goal"])
    prefix = np.asarray(candidate["prefix"])
    state = SuffixState(prefix, 15, 2026120500 + root)
    batches, checkpoints = [], {}

    def predict(actions):
        positions = prior.endpoints(task, actions)
        scores = np.linalg.norm(positions - goal, axis=1)
        batches.append({"actions": actions.tolist(), "scores": scores.tolist()})
        return positions, scores

    for iteration in range(32):
        state.step(12, predict)
        if state.count in (192, 384):
            checkpoints[str(state.count)] = {"score": state.score, "chain": state.best.tolist()}
    return {**candidate, "checkpoints": checkpoints, "batches": batches}


def run():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("diagnostic output already exists")
    source = json.loads(prior.PLANS.read_text())
    audit = json.loads(prior.AUDIT.read_text())
    if source["dependencies"] != prior.hashes() or audit["plans_sha256"] != sha(prior.PLANS):
        raise ValueError("source plan or audit hash mismatch")
    hashes = dependencies()
    rows = []
    for root in range(8):
        task, candidates = candidates_for_root(source["rows"], root)
        if len(candidates) != 32:
            raise ValueError("unexpected candidate count")
        runs = [run_candidate(task, root, candidate) for candidate in candidates]
        rows.append({"root": root, "task": task, "runs": runs})
        print(f"root {root}: 30 sealed candidates and 2 separate oracle diagnostics", flush=True)
    if hashes != dependencies():
        raise ValueError("diagnostic dependencies changed")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows, "PC2_pass": False, "PC3_pass": False},
                  stream, separators=(",", ":"), allow_nan=False)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("diagnostic dependency mismatch")
    counts = []
    for row in data["rows"]:
        goal = np.asarray(row["task"]["goal"])
        for candidate in row["runs"]:
            prefix = np.asarray(candidate["prefix"])
            all_scores, best_seen = [], float("inf")
            for batch in candidate["batches"]:
                actions = np.asarray(batch["actions"])
                if actions.shape != (12, 15, 2):
                    raise ValueError("unexpected batch shape")
                if not np.array_equal(actions[:, :len(prefix)], np.broadcast_to(prefix, (12, *prefix.shape))):
                    raise ValueError("fixed prefix changed")
                scores = np.linalg.norm(prior.endpoints(row["task"], actions) - goal, axis=1)
                np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
                all_scores.extend(scores.tolist())
                best_seen = min(best_seen, float(np.min(scores)))
                if len(all_scores) in (192, 384):
                    checkpoint = candidate["checkpoints"][str(len(all_scores))]
                    np.testing.assert_allclose(checkpoint["score"], best_seen, rtol=0, atol=1e-12)
                    best = np.asarray(checkpoint["chain"])
                    np.testing.assert_array_equal(best[:len(prefix)], prefix)
                    actual = np.linalg.norm(prior.endpoints(row["task"], best[None])[0] - goal)
                    np.testing.assert_allclose(actual, best_seen, rtol=0, atol=1e-12)
            if len(all_scores) != 384:
                raise ValueError("invalid diagnostic budget")
            counts.append({"root": row["root"], "group": candidate["group"], "name": candidate["name"],
                           "samples": len(all_scores)})
    with AUDIT.open("x") as stream:
        json.dump({"runs_sha256": sha(PLANS), "candidates": counts}, stream, indent=2, allow_nan=False)
    print(f"audited {len(counts)} fixed-prefix diagnostics", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    (run if sys.argv[1] == "run" else audit)()
