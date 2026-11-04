"""Development budget comparison with sealed traces and deterministic audit."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import pc_multidirection_successor_exact_dev_20261103_codex as source
from experiments.spectral_reachability import pc_multidirection_multistart_compiler_v2 as compiler

PROTOCOL = source.PROTOCOL.parent / "PC_MULTIDIRECTION_MULTISTART_BUDGET_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = source.PREFIX.with_name("pc_multidirection_multistart_budget_dev_20261103_codex")
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
    print("Comparing matched token budgets on exposed tasks; no independent performance claim.", flush=True)
    rows = []
    for original_row in original["rows"]:
        task, root = original_row["task"], original_row["root"]
        for cap in (480, 1920):
            for arm in ("multistart", "full_cem"):
                meter, trace = SimpleNamespace(token_samples=0), []

                def predict(actions):
                    requested = np.asarray(actions, dtype=float)
                    model = requested.astype(np.float32)
                    tokens = len(model) * model.shape[1] // 5
                    if (model.ndim != 3 or model.shape[1] not in (5, 10, 15) or model.shape[2] != 2
                            or not np.isfinite(model).all() or np.any(np.abs(model) > 1)
                            or meter.token_samples + tokens > cap):
                        raise ValueError("invalid actions or budget exceeded")
                    position = source.previous.prior.exact.endpoints(task, model)
                    score = np.linalg.norm(position - task["goal"], axis=1)
                    meter.token_samples += tokens
                    trace.append({"requested": requested.tolist(), "model": model.tolist(),
                                  "positions": position.tolist(), "scores": score.tolist(), "tokens": tokens})
                    return position, score

                started = time.monotonic()
                result = compiler.decide(arm, root, task["goal"], predict, meter, cap)
                rows.append({"root": root, "task": task, "arm": arm, "cap": cap,
                             "tokens": meter.token_samples, "wall_seconds": time.monotonic() - started,
                             "result": result, "trace": trace})
    if len(rows) != 96 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print("Sealed 96 paired development decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("dependency drift")
    summary = []
    for row in data["rows"]:
        total, full_scores, found = 0, [], False
        selected = np.asarray(row["result"]["selected"]["chain"])
        for batch in row["trace"]:
            requested = np.asarray(batch["requested"])
            model = np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), model)
            position = source.previous.prior.exact.endpoints(row["task"], model)
            score = np.linalg.norm(position - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(position, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(score, batch["scores"], atol=1e-12, rtol=0)
            tokens = len(model) * model.shape[1] // 5
            if tokens != batch["tokens"] or not np.isfinite(model).all() or np.any(np.abs(model) > 1):
                raise ValueError("trace violation")
            total += tokens
            if model.shape[1] == 15:
                full_scores.extend(score)
                indices = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(score[i] - row["result"]["selected"]["terminal_score"]) < 1e-12 for i in indices)
        if (not found or total != row["tokens"] or total > row["cap"]
                or abs(row["result"]["selected"]["terminal_score"] - min(full_scores)) > 1e-12):
            raise ValueError("selection or budget mismatch")
        if row["arm"] == "multistart":
            result = row["result"]
            expected = result["comparison_score"] <= result["precomparison"]["terminal_score"]
            if result["abstained"] != expected:
                raise ValueError("abstention mismatch")
        summary.append({"root": row["root"], "arm": row["arm"], "cap": row["cap"], "tokens": total,
                        "selected_score": row["result"]["selected"]["terminal_score"], "full_samples": len(full_scores)})
    if len(summary) != 96 or len({(r["root"], r["arm"], r["cap"]) for r in summary}) != 96:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": summary}, stream, indent=2, allow_nan=False)
    print("Recomputed all 96 sealed prediction traces.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
