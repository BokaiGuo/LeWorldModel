"""Fixed recorded and privileged-prefix suffix coverage diagnostic."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_witness_injection_diag_20261103_codex as witness
from experiments.pc_budget_allocation_20261103_codex import optimize

prior = witness.exp
PROTOCOL = prior.ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_FIXED_PREFIX_DIAG_PROTOCOL_20261103_CODEX.md"
PREFIX = prior.PREFIX.with_name(prior.PREFIX.name + "_fixed_prefix_diag")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(witness.__file__), prior.PLANS, prior.AUDIT,
             Path(prior.__file__), Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
             Path(prior.exact.wall.physics.__file__), Path(optimize.__code__.co_filename))
    return {str(path): prior.sha(path) for path in paths}


def run_one(task, root, name, prefix):
    trace = []

    def predict(actions):
        pos = prior.exact.endpoints(task, actions)
        scores = np.linalg.norm(pos - task["goal"], axis=1)
        trace.append({"actions": actions.tolist(), "scores": scores.tolist()})
        return scores

    best, score, values = optimize(prefix, 15, 384, np.random.default_rng(2026121500 + root), predict)
    flat_actions = np.concatenate([np.asarray(batch["actions"]) for batch in trace])
    first_index = int(np.argmin(values[:192]))
    return {"name": name, "prefix": np.asarray(prefix).tolist(), "trace": trace,
            "checkpoints": {"192": {"score": float(values[first_index]), "chain": flat_actions[first_index].tolist()},
                            "384": {"score": score, "chain": best.tolist()}}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("diagnostic already exists")
    source = dependencies()
    data = json.loads(prior.PLANS.read_text())
    if data["dependencies"] != prior.dependencies() or json.loads(prior.AUDIT.read_text())["plans_sha256"] != prior.sha(prior.PLANS):
        raise ValueError("source seal mismatch")
    rows = []
    for row in data["rows"]:
        actions, item = witness.witness(row["task"])
        candidates = [(item["name"], np.asarray(item["prefix"])) for item in row["result"]["bank"]]
        candidates += [("witness_first5", actions[:5]), ("witness_first10", actions[:10])]
        results = [run_one(row["task"], row["root"], name, prefix) for name, prefix in candidates]
        rows.append({"root": row["root"], "task": row["task"], "witness": actions.tolist(),
                     "witness_steps": item["steps_needed"], "runs": results})
    if source != dependencies():
        raise ValueError("source drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": source, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows) * 6} fixed-prefix searches.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    rows = []
    for row in data["rows"]:
        witness_score = float(np.linalg.norm(prior.exact.endpoints(
            row["task"], np.asarray(row["witness"])[None])[0] - row["task"]["goal"]))
        if witness_score >= .05:
            raise ValueError("witness failed to replay")
        for run in row["runs"]:
            count, actions_all, scores_all = 0, [], []
            for batch in run["trace"]:
                actions, scores = np.asarray(batch["actions"]), np.asarray(batch["scores"])
                prefix = np.asarray(run["prefix"])
                if (not np.array_equal(actions[:, :len(prefix)], np.broadcast_to(prefix, actions[:, :len(prefix)].shape))
                        or np.any(np.abs(actions) > 1 + 1e-12)):
                    raise ValueError("prefix or bounds mismatch")
                expected = np.linalg.norm(prior.exact.endpoints(row["task"], actions) - row["task"]["goal"], axis=1)
                np.testing.assert_allclose(expected, scores, atol=1e-12, rtol=0)
                count += len(actions)
                actions_all.extend(actions.tolist())
                scores_all.extend(scores.tolist())
            if count != 384:
                raise ValueError("sample count mismatch")
            for n in (192, 384):
                index = int(np.argmin(scores_all[:n]))
                np.testing.assert_allclose(run["checkpoints"][str(n)]["score"], scores_all[index], atol=1e-12)
                np.testing.assert_array_equal(run["checkpoints"][str(n)]["chain"], actions_all[index])
            rows.append({"root": row["root"], "name": run["name"], "count": count,
                         "witness_score": witness_score, "batches": len(run["trace"])})
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": prior.sha(PLANS), "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"Recomputed {len(rows)} fixed-prefix traces.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
