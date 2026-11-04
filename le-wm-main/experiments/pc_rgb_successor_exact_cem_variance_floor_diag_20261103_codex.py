"""One prespecified CEM scale-floor intervention on sealed fixed prefixes."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_fixed_prefix_diag_20261103_codex as source
from experiments import pc_rgb_successor_exact_cem_trace_reconstruction_20261103_codex as reconstruction

PROTOCOL = source.prior.ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_CEM_VARIANCE_FLOOR_DIAG_PROTOCOL_20261103_CODEX.md"
PREFIX = source.PREFIX.with_name(source.PREFIX.name + "_variance_floor_005")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, source.PLANS, source.AUDIT,
             reconstruction.OUTPUT, Path(reconstruction.__file__))
    return {str(p): source.prior.sha(p) for p in paths}


def run_one(task, root, prefix):
    prefix = np.asarray(prefix)
    remaining = 15 - len(prefix)
    rng = np.random.default_rng(2026121500 + root)
    loc = np.zeros((remaining, 2))
    scale = np.full((remaining, 2), 1 / 3)
    trace = []
    best_score, best_chain = float("inf"), None
    checkpoints = {}
    for index in range(32):
        samples = np.clip(loc + rng.normal(size=(12, remaining, 2)) * scale, -1, 1)
        samples[0] = loc
        chains = np.concatenate((np.broadcast_to(prefix, (12, len(prefix), 2)), samples), axis=1)
        scores = np.linalg.norm(source.prior.exact.endpoints(task, chains) - task["goal"], axis=1)
        for chain, score in zip(chains, scores, strict=True):
            if score < best_score:
                best_score, best_chain = float(score), chain.copy()
        trace.append({"actions": chains.tolist(), "scores": scores.tolist()})
        if index in (15, 31):
            checkpoints[str((index + 1) * 12)] = {"score": best_score, "chain": best_chain.tolist()}
        elite = samples[np.argsort(scores, kind="stable")[:4]]
        loc = elite.mean(0)
        scale = np.maximum(elite.std(0, ddof=0), .05)
    return {"trace": trace, "checkpoints": checkpoints}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("intervention output exists")
    hashes = dependencies()
    old = json.loads(source.PLANS.read_text())
    if (old["dependencies"] != source.dependencies() or
            json.loads(source.AUDIT.read_text())["plans_sha256"] != source.prior.sha(source.PLANS)):
        raise ValueError("sealed source mismatch")
    receipt = json.loads(reconstruction.OUTPUT.read_text())
    if any(source.prior.sha(p) != digest for p, digest in receipt["sources"].items()):
        raise ValueError("reconstruction seal mismatch")
    print("Variance-floor intervention on recorded fixed prefixes; privileged witnesses remain diagnostic.", flush=True)
    rows = []
    for row in old["rows"]:
        runs = []
        for old_run in row["runs"]:
            result = run_one(row["task"], row["root"], old_run["prefix"])
            runs.append({"name": old_run["name"], "prefix": old_run["prefix"], **result})
        rows.append({"root": row["root"], "task": row["task"], "runs": runs})
    if hashes != dependencies():
        raise ValueError("dependencies changed during run")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows) * 6} intervention searches.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    old = json.loads(source.PLANS.read_text())
    if data["dependencies"] != dependencies() or len(data["rows"]) != 24:
        raise ValueError("source drift")
    count = 0
    for row, old_row in zip(data["rows"], old["rows"], strict=True):
        if row["root"] != old_row["root"] or row["task"] != old_row["task"] or len(row["runs"]) != 6:
            raise ValueError("task mismatch")
        for run, old_run in zip(row["runs"], old_row["runs"], strict=True):
            if run["name"] != old_run["name"] or run["prefix"] != old_run["prefix"]:
                raise ValueError("prefix mismatch")
            actual = run_one(row["task"], row["root"], run["prefix"])
            if actual != {"trace": run["trace"], "checkpoints": run["checkpoints"]}:
                raise ValueError("random stream or exact dynamics mismatch")
            count += 1
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": source.prior.sha(PLANS), "replayed_searches": count}, stream, indent=2)
    print(f"Replayed {count} intervention searches.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
