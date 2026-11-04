"""Sealed exact-dynamics development comparison for multi-mode compiler."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_successor_floor_cem_exact_dev_20261103_codex as previous
from experiments.spectral_reachability import pc_response_diverse_compiler_v1 as compiler

prior = previous.prior
PROTOCOL = prior.ROOT.parent / "docs/PC_RESPONSE_DIVERSE_BEAM_EXACT_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = prior.ROOT / "results/pc_response_diverse_beam_exact_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(compiler.__file__), previous.PLANS, previous.AUDIT)
    return {**previous.dependencies(), **{str(path): prior.sha(path) for path in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    hashes = dependencies()
    previous_data = json.loads(previous.PLANS.read_text())
    if (previous_data["dependencies"] != previous.dependencies()
            or json.loads(previous.AUDIT.read_text())["plans_sha256"] != prior.sha(previous.PLANS)):
        raise ValueError("previous source seal mismatch")
    print("Multi-mode response versus greedy, random and full CEM; exposed development roots.", flush=True)
    rows = []
    for old in previous_data["rows"]:
        if old["arm"] != "successor":
            continue
        task, root = old["task"], old["root"]
        for arm in compiler.ARMS:
            meter, trace = prior.Budget(), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                pos, scores = meter.predict(task, requested)
                trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                              "positions": pos.tolist(), "scores": scores.tolist(),
                              "tokens": len(requested) * requested.shape[1] // 5})
                return pos, scores

            started = time.monotonic()
            result = compiler.decide(arm, root, np.asarray(task["goal"]), predict, meter)
            rows.append({"root": root, "task": task, "arm": arm, "result": result,
                         "tokens": meter.token_samples, "wall_seconds": time.monotonic() - started, "trace": trace})
    if len(rows) != 24 * len(compiler.ARMS) or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    rows = []
    for row in data["rows"]:
        result, task = row["result"], row["task"]
        selected = np.asarray(result["selected"]["chain"])
        total, found = 0, False
        for batch in row["trace"]:
            requested, actions = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), actions)
            if np.any(np.abs(actions) > 1 + 1e-7):
                raise ValueError("action bounds")
            positions = prior.exact.endpoints(task, actions)
            scores = np.linalg.norm(positions - task["goal"], axis=1)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            if requested.shape[1:] == selected.shape:
                indices = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(scores[i] - result["selected"]["terminal_score"]) < 1e-12 for i in indices)
            cost = len(actions) * actions.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("token mismatch")
            total += cost
        if not found or total != row["tokens"] or total > 480:
            raise ValueError("unscored chain or budget mismatch")
        rows.append({"root": row["root"], "arm": row["arm"], "tokens": total, "batches": len(row["trace"])})
    if len(rows) != 96 or len({(r["root"], r["arm"]) for r in rows}) != 96:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": prior.sha(PLANS), "rows": rows}, stream, indent=2)
    print(f"Audited {len(rows)} exact decision traces.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
