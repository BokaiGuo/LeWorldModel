"""Exact feedback experiment for the sealed multidirection compiler."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_multidirection_successor_exact_dev_20261103_codex as prior
from experiments import pc_response_diverse_fixed_terminal_exact_dev_20261103_codex as baseline
from experiments.spectral_reachability import pc_multidirection_feedback_v1 as feedback

ROOT = prior.previous.prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_MULTIDIRECTION_FIXED_TERMINAL_FEEDBACK_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_multidirection_fixed_terminal_feedback_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
sha = prior.sha


def dependencies():
    paths = (Path(__file__), Path(feedback.__file__), PROTOCOL, prior.PLANS, prior.AUDIT, baseline.PLANS, baseline.AUDIT)
    return {**prior.dependencies(), **baseline.dependencies(), **{str(p): sha(p) for p in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("sealed output exists")
    hashes = dependencies()
    data, controls = json.loads(prior.PLANS.read_text()), json.loads(baseline.PLANS.read_text())
    if data["dependencies"] != prior.dependencies() or json.loads(prior.AUDIT.read_text())["plans_sha256"] != sha(prior.PLANS):
        raise ValueError("compiler seal mismatch")
    if controls["dependencies"] != baseline.dependencies() or json.loads(baseline.AUDIT.read_text())["plans_sha256"] != sha(baseline.PLANS):
        raise ValueError("control seal mismatch")
    control = {r["root"]: r for r in controls["rows"] if r["arm"] == "full_cem"}
    print("Fixed-terminal multidirection feedback; sealed first decisions, exposed exact roots.", flush=True)
    rows = []
    for old in data["rows"]:
        root, task = old["root"], old["task"]
        if control[root]["task"] != task:
            raise ValueError("control task mismatch")
        initial = old["result"]["selected"]
        first = np.asarray(initial["chain"])
        state, blocked = baseline.advance(task["start"], first[:5])
        stages = [{"step": 0, "start": task["start"], "horizon": 15, "source": "sealed_prior",
                   "selected": initial, "tokens": old["tokens"], "dispatched": first[:5].tolist(),
                   "end_state": state, "blocked": blocked}]
        carry = first[5:]
        for step in (1, 2):
            local = {"start": state, "goal": task["goal"]}
            meter, trace = prior.previous.prior.Budget(), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                positions, scores = meter.predict(local, requested)
                trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                              "positions": positions.tolist(), "scores": scores.tolist(),
                              "tokens": len(requested) * requested.shape[1] // 5})
                return positions, scores

            started = time.monotonic()
            result = feedback.decide(root, step, task["goal"], predict, meter, carry)
            chain = np.asarray(result["selected"]["chain"])
            end, blocked = baseline.advance(state, chain[:5])
            stages.append({"step": step, "start": state, "horizon": 15 - 5 * step,
                           "result": result, "tokens": meter.token_samples, "trace": trace,
                           "wall_seconds": time.monotonic() - started, "dispatched": chain[:5].tolist(),
                           "end_state": end, "blocked": blocked})
            state, carry = end, chain[5:]
        rows.append({"root": root, "task": task, "stages": stages,
                     "final_distance": float(np.linalg.norm(np.asarray(state[:2]) - task["goal"])),
                     "control_final_distance": control[root]["final_distance"]})
    if len(rows) != 24 or len({r["root"] for r in rows}) != 24 or hashes != dependencies():
        raise ValueError("panel or source drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} three-decision trajectories.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    source = {r["root"]: r for r in json.loads(prior.PLANS.read_text())["rows"]}
    controls = {r["root"]: r for r in json.loads(baseline.PLANS.read_text())["rows"] if r["arm"] == "full_cem"}
    summary = []
    for row in data["rows"]:
        root, task = row["root"], row["task"]
        if task != source[root]["task"] or task != controls[root]["task"]:
            raise ValueError("task mismatch")
        if row["stages"][0]["selected"] != source[root]["result"]["selected"] or row["control_final_distance"] != controls[root]["final_distance"]:
            raise ValueError("sealed first decision or control mismatch")
        state, previous_chain = task["start"], None
        for step, stage in enumerate(row["stages"]):
            if stage["step"] != step or stage["start"] != state or stage["horizon"] != 15 - 5 * step:
                raise ValueError("clock or state mismatch")
            result = stage.get("result")
            selected = stage["selected"] if step == 0 else result["selected"]
            chain = np.asarray(selected["chain"])
            np.testing.assert_array_equal(chain[:5], stage["dispatched"])
            if len(chain) != stage["horizon"]:
                raise ValueError("horizon mismatch")
            if step:
                np.testing.assert_array_equal(result["carry"], previous_chain[5:])
                total, found, best = 0, False, float("inf")
                for batch in stage["trace"]:
                    requested, model = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
                    np.testing.assert_array_equal(requested.astype(np.float32), model)
                    positions = prior.previous.prior.exact.endpoints({"start": state, "goal": task["goal"]}, model)
                    scores = np.linalg.norm(positions - task["goal"], axis=1)
                    np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
                    if requested.shape[1:] == chain.shape:
                        best = min(best, float(np.min(scores)))
                        matches = np.flatnonzero(np.all(requested == chain, axis=(1, 2)))
                        found |= any(abs(scores[i] - selected["terminal_score"]) < 1e-12 for i in matches)
                    cost = len(requested) * requested.shape[1] // 5
                    if cost != batch["tokens"]:
                        raise ValueError("token mismatch")
                    total += cost
                if total != stage["tokens"] or total > 480 or not found:
                    raise ValueError("budget or selection mismatch")
                np.testing.assert_allclose(selected["terminal_score"], best, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(stage["trace"][0]["requested"][0], result["carry"])
                np.testing.assert_allclose(stage["trace"][0]["scores"][0], result["carry_score"], atol=1e-12, rtol=0)
            state, blocked = baseline.advance(state, stage["dispatched"])
            np.testing.assert_allclose(state, stage["end_state"], atol=1e-12, rtol=0)
            if blocked != stage["blocked"]:
                raise ValueError("collision mismatch")
            previous_chain = chain
        whole = np.concatenate([np.asarray(s["dispatched"]) for s in row["stages"]])[None].astype(np.float32)
        end = prior.previous.prior.exact.endpoints(task, whole)[0]
        np.testing.assert_allclose(end, state[:2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["final_distance"], np.linalg.norm(end - task["goal"]), atol=1e-12, rtol=0)
        summary.append({"root": root, "tokens": [s["tokens"] for s in row["stages"]]})
    if len(summary) != 24 or len({r["root"] for r in summary}) != 24:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": summary}, stream, indent=2)
    print(f"Replayed {len(summary)} fixed-terminal feedback trajectories.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
