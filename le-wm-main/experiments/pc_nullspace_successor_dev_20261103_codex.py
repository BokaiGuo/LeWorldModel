"""Exact-model development of response-nullspace candidate generation."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_goal_aligned_successor_dev_20261103_codex as source
from experiments.spectral_reachability import pc_nullspace_successor_compiler_v1 as compiler

PROTOCOL = source.PROTOCOL.parent / "PC_NULLSPACE_SUCCESSOR_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = source.PREFIX.with_name("pc_nullspace_successor_dev_20261103_codex")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    files = (Path(__file__), Path(compiler.__file__), PROTOCOL, source.PLANS, source.AUDIT)
    return {**source.dependencies(), **{str(p): source.sha(p) for p in files}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("sealed output exists")
    hashes = dependencies()
    data = json.loads(source.PLANS.read_text())
    if data["dependencies"] != source.dependencies() or json.loads(source.AUDIT.read_text())["plans_sha256"] != source.sha(source.PLANS):
        raise ValueError("source seal mismatch")
    tasks = [r for r in data["rows"] if r["arm"] == "goal"]
    if len(tasks) != 24 or len({r["root"] for r in tasks}) != 24:
        raise ValueError("invalid source panel")
    print("Comparing nullspace exploration, matched random extra and full CEM.", flush=True)
    rows = []
    for row in tasks:
        task, root = row["task"], row["root"]
        for arm in compiler.ARMS:
            meter, trace = source.source.previous.prior.Budget(), []

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
        print(f"root {root}: three arms", flush=True)
    if len(rows) != 72 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print("Sealed 72 development plans.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, old = (json.loads(p.read_text()) for p in (PLANS, source.PLANS))
    if data["dependencies"] != dependencies():
        raise ValueError("dependency drift")
    tasks = {r["root"]: r["task"] for r in old["rows"] if r["arm"] == "goal"}
    summary = []
    for row in data["rows"]:
        if row["task"] != tasks[row["root"]]:
            raise ValueError("task mismatch")
        result, task = row["result"], row["task"]
        selected = np.asarray(result["selected"]["chain"])
        found, total, best = False, 0, float("inf")
        for batch in row["trace"]:
            requested, model = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), model)
            if not np.isfinite(model).all() or np.any(np.abs(model) > 1):
                raise ValueError("invalid action")
            positions = source.source.previous.prior.exact.endpoints(task, model)
            scores = np.linalg.norm(positions - task["goal"], axis=1)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            cost = len(model) * model.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("token mismatch")
            total += cost
            if model.shape[1] == 15:
                best = min(best, float(np.min(scores)))
                matches = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(scores[i] - result["selected"]["terminal_score"]) < 1e-12 for i in matches)
        if not found or total != row["tokens"] or total > compiler.CAP or abs(best - result["selected"]["terminal_score"]) > 1e-12:
            raise ValueError("selection or budget mismatch")
        if row["arm"] == "null":
            responses = [result["response"]] + [x["response"] for x in result["successor_response"]]
            for response in responses:
                matrix, extra = np.asarray(response["jacobian"]), np.asarray(response["extra_directions"])
                if extra.size and np.max(np.abs(matrix @ extra.T)) > 1e-8:
                    raise ValueError("null directions not in response nullspace")
        summary.append({"root": row["root"], "arm": row["arm"], "tokens": total,
                        "score": result["selected"]["terminal_score"]})
    if len(summary) != 72 or len({(r["root"], r["arm"]) for r in summary}) != 72:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": source.sha(PLANS), "rows": summary}, stream, indent=2, allow_nan=False)
    print("Recomputed 72 plans and their selections.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
