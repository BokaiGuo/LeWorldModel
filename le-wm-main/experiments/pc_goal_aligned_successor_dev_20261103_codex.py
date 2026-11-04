"""Sealed paired development evaluation for goal-aligned successor proposals."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_multidirection_successor_exact_dev_20261103_codex as source
from experiments.spectral_reachability import pc_goal_aligned_successor_compiler_v1 as compiler

PROTOCOL = source.PROTOCOL.parent / "PC_GOAL_ALIGNED_SUCCESSOR_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = source.PREFIX.with_name("pc_goal_aligned_successor_dev_20261103_codex")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
sha = source.sha


def dependencies():
    paths = (Path(__file__), Path(compiler.__file__), PROTOCOL, source.PLANS, source.AUDIT)
    return {**source.dependencies(), **{str(path): sha(path) for path in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("sealed output exists")
    hashes = dependencies()
    original = json.loads(source.PLANS.read_text())
    if original["dependencies"] != source.dependencies() or json.loads(source.AUDIT.read_text())["plans_sha256"] != sha(source.PLANS):
        raise ValueError("source seal mismatch")
    print("Comparing goal-aligned, SVD, random and full CEM on exposed exact tasks.", flush=True)
    rows = []
    for old in original["rows"]:
        root, task = old["root"], old["task"]
        for arm in compiler.ARMS:
            meter, trace = source.previous.prior.Budget(), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                positions, scores = meter.predict(task, requested)
                trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                              "positions": positions.tolist(), "scores": scores.tolist(),
                              "tokens": len(requested) * requested.shape[1] // 5})
                return positions, scores

            started = time.monotonic()
            result = compiler.decide(arm, root, task["goal"], predict, meter)
            rows.append({"root": root, "task": task, "arm": arm, "result": result,
                         "tokens": meter.token_samples, "wall_seconds": time.monotonic() - started, "trace": trace})
    if len(rows) != 96 or hashes != dependencies():
        raise ValueError("panel or source drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print("Sealed 96 paired development decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, original = json.loads(PLANS.read_text()), json.loads(source.PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    tasks = {r["root"]: r["task"] for r in original["rows"]}
    summary = []
    for row in data["rows"]:
        if row["task"] != tasks[row["root"]]:
            raise ValueError("task mismatch")
        selected = np.asarray(row["result"]["selected"]["chain"])
        total, full_scores, found = 0, [], False
        for batch in row["trace"]:
            requested, model = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), model)
            if not np.isfinite(model).all() or np.any(np.abs(model) > 1):
                raise ValueError("invalid action")
            positions = source.previous.prior.exact.endpoints(row["task"], model)
            scores = np.linalg.norm(positions - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            cost = len(model) * model.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("token mismatch")
            total += cost
            if model.shape[1] == 15:
                full_scores.extend(scores)
                matches = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(scores[i] - row["result"]["selected"]["terminal_score"]) < 1e-12 for i in matches)
        if (not found or total != row["tokens"] or total > compiler.CAP
                or abs(row["result"]["selected"]["terminal_score"] - min(full_scores)) > 1e-12):
            raise ValueError("selection or budget mismatch")
        if row["arm"] != "full_cem":
            result = row["result"]
            if result["abstained"] != (result["comparison_score"] <= result["precomparison"]["terminal_score"]):
                raise ValueError("abstention mismatch")
        summary.append({"root": row["root"], "arm": row["arm"], "tokens": total,
                        "full_samples": len(full_scores), "selected_score": row["result"]["selected"]["terminal_score"]})
    if len(summary) != 96 or len({(r["root"], r["arm"]) for r in summary}) != 96:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": summary}, stream, indent=2, allow_nan=False)
    print("Recomputed 96 paired prediction traces.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
