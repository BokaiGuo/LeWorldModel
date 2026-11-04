"""Sealed 24-root evaluation of the multidirection successor compiler."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_response_diverse_beam_exact_dev_20261103_codex as previous
from experiments.spectral_reachability import pc_multidirection_successor_compiler_v1 as compiler

PROTOCOL = previous.prior.ROOT.parent / "docs/PC_MULTIDIRECTION_SUCCESSOR_COMPILER_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = previous.prior.ROOT / "results/pc_multidirection_successor_exact_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
sha = previous.prior.sha


def dependencies():
    paths = (Path(__file__), Path(compiler.__file__), PROTOCOL, previous.PLANS, previous.AUDIT)
    return {**previous.dependencies(), **{str(p): sha(p) for p in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    hashes = dependencies()
    data = json.loads(previous.PLANS.read_text())
    if data["dependencies"] != previous.dependencies() or json.loads(previous.AUDIT.read_text())["plans_sha256"] != sha(previous.PLANS):
        raise ValueError("source seal mismatch")
    print("Multi-direction successor compiler on exposed exact-dynamics roots.", flush=True)
    rows = []
    for old in data["rows"]:
        if old["arm"] != "full_cem":
            continue
        root, task = old["root"], old["task"]
        meter, trace = previous.prior.Budget(), []

        def predict(actions):
            requested = np.asarray(actions, dtype=float)
            pos, scores = meter.predict(task, requested)
            trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                          "positions": pos.tolist(), "scores": scores.tolist(),
                          "tokens": len(requested) * requested.shape[1] // 5})
            return pos, scores

        started = time.monotonic()
        result = compiler.decide(root, task["goal"], predict, meter)
        rows.append({"root": root, "task": task, "result": result, "tokens": meter.token_samples,
                     "wall_seconds": time.monotonic() - started, "trace": trace})
    if len(rows) != 24 or len({r["root"] for r in rows}) != 24 or hashes != dependencies():
        raise ValueError("panel or source drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} exact decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, old = json.loads(PLANS.read_text()), json.loads(previous.PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    baseline = {r["root"]: r for r in old["rows"] if r["arm"] == "full_cem"}
    summary = []
    for row in data["rows"]:
        root, task, result = row["root"], row["task"], row["result"]
        if task != baseline[root]["task"]:
            raise ValueError("baseline task mismatch")
        chain = np.asarray(result["selected"]["chain"])
        found, total = False, 0
        for batch in row["trace"]:
            requested = np.asarray(batch["requested"])
            model = np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), model)
            if not np.isfinite(requested).all() or np.any(np.abs(requested) > 1):
                raise ValueError("invalid action")
            positions = previous.prior.exact.endpoints(task, model)
            scores = np.linalg.norm(positions - task["goal"], axis=1)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            if requested.shape[1:] == chain.shape:
                matches = np.flatnonzero(np.all(requested == chain, axis=(1, 2)))
                found |= any(abs(scores[i] - result["selected"]["terminal_score"]) < 1e-12 for i in matches)
            cost = len(requested) * requested.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("bad token count")
            total += cost
        if total != row["tokens"] or total > 480 or not found:
            raise ValueError("budget or selection mismatch")
        if result["structure_counts"]["svd"] != 1 + len(result["retained"]):
            raise ValueError("missing successor SVD")
        summary.append({"root": root, "tokens": total, "svd": result["structure_counts"]["svd"]})
    if len(summary) != 24 or len({r["root"] for r in summary}) != 24:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": summary}, stream, indent=2)
    print(f"Replayed {len(summary)} exact decisions.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
