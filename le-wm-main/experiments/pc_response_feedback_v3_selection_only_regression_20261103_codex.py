"""Replay V3 at fixed contexts and verify all pre-final calls match V1."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_response_feedback_v2_sealed_context_regression_20261103_codex as v2_exp
from experiments.spectral_reachability import pc_response_diverse_feedback_v3 as v3

prior = v2_exp.prior
base = prior.prior.prior
PROTOCOL = base.ROOT.parent / "docs/PC_RESPONSE_FEEDBACK_V3_SELECTION_ONLY_PROTOCOL_20261103_CODEX.md"
PREFIX = base.ROOT / "results/pc_response_feedback_v3_selection_only_regression_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(v3.__file__), prior.PLANS, prior.AUDIT, v2_exp.PLANS, v2_exp.AUDIT)
    return {**prior.dependencies(), **{str(p): base.sha(p) for p in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    source, v2_data = json.loads(prior.PLANS.read_text()), json.loads(v2_exp.PLANS.read_text())
    if (source["dependencies"] != prior.dependencies()
            or json.loads(prior.AUDIT.read_text())["plans_sha256"] != base.sha(prior.PLANS)
            or v2_data["dependencies"] != v2_exp.dependencies()
            or json.loads(v2_exp.AUDIT.read_text())["plans_sha256"] != base.sha(v2_exp.PLANS)):
        raise ValueError("prior seal mismatch")
    hashes, rows = dependencies(), []
    print("Selection-only repair at sealed contexts; pre-comparison calls must equal V1.", flush=True)
    for row in source["rows"]:
        if row["arm"] != "response_diverse":
            continue
        stage, root = row["stages"][2], row["root"]
        task = {"start": stage["start"], "goal": row["task"]["goal"]}
        meter, trace = base.Budget(), []

        def predict(actions):
            requested = np.asarray(actions, dtype=float)
            positions, scores = meter.predict(task, requested)
            trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                          "positions": positions.tolist(), "scores": scores.tolist(),
                          "tokens": len(requested) * requested.shape[1] // 5})
            return positions, scores

        started = time.monotonic()
        result = v3.decide("response_diverse", root, 2, task["goal"], predict, meter, stage["result"]["carry"])
        if trace[:-1] != stage["trace"][:-1]:
            raise ValueError("pre-comparison search differs from V1")
        end, blocked = prior.advance(task["start"], result["selected"]["chain"])
        rows.append({"root": root, "task": task, "carry": stage["result"]["carry"], "result": result,
                     "old_selected": stage["result"]["selected"], "tokens": meter.token_samples,
                     "trace": trace, "end_state": end, "blocked": blocked,
                     "actual_distance": float(np.linalg.norm(np.asarray(end[:2]) - task["goal"])),
                     "wall_seconds": time.monotonic() - started})
    if len(rows) != 24 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} V3 context decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, old = json.loads(PLANS.read_text()), json.loads(prior.PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    sources = {r["root"]: r for r in old["rows"] if r["arm"] == "response_diverse"}
    receipts = []
    for row in data["rows"]:
        old_row = sources[row["root"]]
        stage = old_row["stages"][2]
        if (row["task"] != {"start": stage["start"], "goal": old_row["task"]["goal"]}
                or row["carry"] != stage["result"]["carry"] or row["old_selected"] != stage["result"]["selected"]
                or row["trace"][:-1] != stage["trace"][:-1]):
            raise ValueError("context or V1 search differs")
        chain = np.asarray(row["result"]["selected"]["chain"])
        scores_all, total, found = [], 0, False
        for batch in row["trace"]:
            requested, model = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), model)
            pos = base.exact.endpoints(row["task"], model)
            scores = np.linalg.norm(pos - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(pos, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            matching = np.flatnonzero(np.all(requested == chain, axis=(1, 2)))
            found |= any(abs(scores[i] - row["result"]["selected"]["terminal_score"]) < 1e-12 for i in matching)
            scores_all.extend(scores.tolist())
            if batch["tokens"] != len(model):
                raise ValueError("token mismatch")
            total += len(model)
        if not found or total != row["tokens"] or total > 480:
            raise ValueError("unscored chain or budget mismatch")
        np.testing.assert_allclose(min(scores_all), row["result"]["selected"]["terminal_score"], atol=1e-12, rtol=0)
        end, blocked = prior.advance(row["task"]["start"], chain)
        np.testing.assert_allclose(end, row["end_state"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["actual_distance"], np.linalg.norm(np.asarray(end[:2]) - row["task"]["goal"]), atol=1e-12)
        if blocked != row["blocked"]:
            raise ValueError("collision mismatch")
        receipts.append({"root": row["root"], "tokens": total, "precomparison_batches_equal": len(row["trace"]) - 1})
    if len(receipts) != 24 or len({r["root"] for r in receipts}) != 24:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": base.sha(PLANS), "rows": receipts}, stream, indent=2)
    print(f"Audited {len(receipts)} selection-only decisions.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
