"""Sealed exact-dynamics comparison of the scale-floor compiler and controls."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_controls_dev_20261103_codex as controls
from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as prior
from experiments.spectral_reachability import pc_cem_scale_floor_v1 as cem
from experiments.spectral_reachability import pc_successor_floor_compiler_v1 as compiler

PROTOCOL = prior.ROOT.parent / "docs/PC_SUCCESSOR_FLOOR_CEM_EXACT_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = prior.ROOT / "results/pc_successor_floor_cem_exact_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(cem.__file__), Path(compiler.__file__),
             prior.PLANS, prior.AUDIT, controls.PLANS, controls.AUDIT)
    return {**prior.dependencies(), **controls.dependencies(), **{str(p): prior.sha(p) for p in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    hashes = dependencies()
    source = json.loads(prior.PLANS.read_text())
    old_controls = json.loads(controls.PLANS.read_text())
    if (source["dependencies"] != prior.dependencies()
            or json.loads(prior.AUDIT.read_text())["plans_sha256"] != prior.sha(prior.PLANS)
            or old_controls["dependencies"] != controls.dependencies()
            or json.loads(controls.AUDIT.read_text())["plans_sha256"] != prior.sha(controls.PLANS)):
        raise ValueError("sealed prior mismatch")
    print("Exact development comparison; no witness inputs or independent claims.", flush=True)
    rows = []
    for task_row in source["rows"]:
        task, root = task_row["task"], task_row["root"]
        for arm in compiler.ARMS:
            meter, trace = prior.Budget(), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                pos, scores = meter.predict(task, requested)
                trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                              "positions": pos.tolist(), "scores": scores.tolist(),
                              "tokens": len(requested) * requested.shape[1] // 5})
                return pos, scores

            start = time.monotonic()
            result = compiler.decide(arm, root, np.asarray(task["goal"]), predict, meter)
            rows.append({"root": root, "task": task, "arm": arm, "result": result,
                         "tokens": meter.token_samples, "wall_seconds": time.monotonic() - start, "trace": trace})
    if hashes != dependencies():
        raise ValueError("dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    source = json.loads(prior.PLANS.read_text())
    if data["dependencies"] != dependencies() or len(data["rows"]) != 24 * len(compiler.ARMS):
        raise ValueError("dependency or row mismatch")
    summaries = []
    for row in data["rows"]:
        old = source["rows"][row["root"]]
        if row["task"] != old["task"] or row["arm"] not in compiler.ARMS:
            raise ValueError("task mismatch")
        if row["arm"] != "full_cem" and row["result"]["first_bank"] != old["result"]["first_bank"]:
            raise ValueError("first bank mismatch")
        selected = np.asarray(row["result"]["selected"]["chain"])
        found, tokens = False, 0
        for batch in row["trace"]:
            requested = np.asarray(batch["requested"])
            actions = np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), actions)
            if np.any(np.abs(actions) > 1 + 1e-7):
                raise ValueError("action bounds exceeded")
            pos = prior.exact.endpoints(row["task"], actions)
            scores = np.linalg.norm(pos - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(pos, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            if requested.shape[1:] == selected.shape:
                idx = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(scores[i] - row["result"]["selected"]["terminal_score"]) < 1e-12 for i in idx)
            cost = len(actions) * actions.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("invalid token count")
            tokens += cost
        if not found or tokens != row["tokens"] or tokens > 480:
            raise ValueError("unscored selection or budget exceeded")
        summaries.append({"root": row["root"], "arm": row["arm"], "tokens": tokens, "batches": len(row["trace"])})
    if len({(r["root"], r["arm"]) for r in summaries}) != 24 * len(compiler.ARMS):
        raise ValueError("duplicate decisions")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": prior.sha(PLANS), "rows": summaries}, stream, indent=2)
    print(f"Audited {len(summaries)} complete prediction traces.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
