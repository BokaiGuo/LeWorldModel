"""Run V2 only at sealed exact final-step contexts; audit all predictions."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_response_diverse_fixed_terminal_exact_dev_20261103_codex as prior
from experiments.spectral_reachability import pc_response_diverse_feedback_v2 as v2

PROTOCOL = prior.prior.prior.ROOT.parent / "docs/PC_RESPONSE_FEEDBACK_V2_SEALED_CONTEXT_REGRESSION_PROTOCOL_20261103_CODEX.md"
PREFIX = prior.prior.prior.ROOT / "results/pc_response_feedback_v2_sealed_context_regression_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(v2.__file__), prior.PLANS, prior.AUDIT)
    return {**prior.dependencies(), **{str(p): prior.prior.prior.sha(p) for p in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("regression output exists")
    source = json.loads(prior.PLANS.read_text())
    if (source["dependencies"] != prior.dependencies()
            or json.loads(prior.AUDIT.read_text())["plans_sha256"] != prior.prior.prior.sha(prior.PLANS)):
        raise ValueError("source seal mismatch")
    hashes, rows = dependencies(), []
    print("V2 final decision at sealed exact contexts; no new roots or first-stage search.", flush=True)
    for row in source["rows"]:
        if row["arm"] != "response_diverse":
            continue
        stage, root = row["stages"][2], row["root"]
        local = {"start": stage["start"], "goal": row["task"]["goal"]}
        carry = np.asarray(stage["result"]["carry"])
        meter, trace = prior.prior.prior.Budget(), []

        def predict(actions):
            requested = np.asarray(actions, dtype=float)
            pos, scores = meter.predict(local, requested)
            trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                          "positions": pos.tolist(), "scores": scores.tolist(),
                          "tokens": len(requested) * requested.shape[1] // 5})
            return pos, scores

        started = time.monotonic()
        result = v2.decide("response_diverse", root, 2, local["goal"], predict, meter, carry)
        selected = np.asarray(result["selected"]["chain"])
        state, blocked = prior.advance(local["start"], selected)
        rows.append({"root": root, "task": local, "carry": carry.tolist(), "result": result,
                     "tokens": meter.token_samples, "trace": trace, "end_state": state, "blocked": blocked,
                     "actual_distance": float(np.linalg.norm(np.asarray(state[:2]) - local["goal"])),
                     "old_selected": stage["result"]["selected"], "old_end_state": stage["end_state"],
                     "wall_seconds": time.monotonic() - started})
    if len(rows) != 24 or hashes != dependencies():
        raise ValueError("panel or source changed")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} V2 context decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, old = json.loads(PLANS.read_text()), json.loads(prior.PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    old_rows = {r["root"]: r for r in old["rows"] if r["arm"] == "response_diverse"}
    receipts = []
    for row in data["rows"]:
        old_row = old_rows[row["root"]]
        stage = old_row["stages"][2]
        if (row["task"] != {"start": stage["start"], "goal": old_row["task"]["goal"]}
                or row["carry"] != stage["result"]["carry"]
                or row["old_selected"] != stage["result"]["selected"]):
            raise ValueError("context mismatch")
        chain = np.asarray(row["result"]["selected"]["chain"])
        best, total, found = float("inf"), 0, False
        for batch in row["trace"]:
            requested, model = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), model)
            positions = prior.prior.prior.exact.endpoints(row["task"], model)
            scores = np.linalg.norm(positions - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            index = np.flatnonzero(np.all(requested == chain, axis=(1, 2)))
            found |= any(abs(scores[i] - row["result"]["selected"]["terminal_score"]) < 1e-12 for i in index)
            best = min(best, float(np.min(scores)))
            if batch["tokens"] != len(model):
                raise ValueError("token mismatch")
            total += len(model)
        if not found or total != row["tokens"] or total > 480:
            raise ValueError("unscored chain or budget mismatch")
        np.testing.assert_allclose(best, row["result"]["selected"]["terminal_score"], atol=1e-12, rtol=0)
        end, blocked = prior.advance(row["task"]["start"], chain)
        np.testing.assert_allclose(end, row["end_state"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["actual_distance"], np.linalg.norm(np.asarray(end[:2]) - row["task"]["goal"]), atol=1e-12)
        if blocked != row["blocked"]:
            raise ValueError("collision mismatch")
        receipts.append({"root": row["root"], "tokens": total, "best_score": best, "batches": len(row["trace"])})
    if len(receipts) != 24 or len({r["root"] for r in receipts}) != 24:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": prior.prior.prior.sha(PLANS), "rows": receipts}, stream, indent=2)
    print(f"Audited {len(receipts)} V2 decisions.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
