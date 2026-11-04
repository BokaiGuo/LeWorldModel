"""Fixed-endpoint exact feedback, reusing the previously sealed first decision."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_response_diverse_beam_exact_dev_20261103_codex as prior
from experiments.spectral_reachability import pc_response_diverse_feedback_v1 as feedback

PROTOCOL = prior.prior.ROOT.parent / "docs/PC_RESPONSE_DIVERSE_FIXED_TERMINAL_EXACT_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = prior.prior.ROOT / "results/pc_response_diverse_fixed_terminal_exact_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(feedback.__file__), prior.PLANS, prior.AUDIT)
    return {**prior.dependencies(), **{str(p): prior.prior.sha(p) for p in paths}}


def advance(start, actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (5, 2) or np.any(np.abs(actions) > 1):
        raise ValueError("invalid dispatch")
    physical = np.pad(actions.astype(float), ((0, 11), (0, 0))) * prior.prior.exact.wall.HIGH
    trajectory, blocked = prior.prior.exact.wall.rollout(start, physical[None])
    return trajectory[0, 5].tolist(), blocked[0, :5].tolist()


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    hashes = dependencies()
    source = json.loads(prior.PLANS.read_text())
    if source["dependencies"] != prior.dependencies() or json.loads(prior.AUDIT.read_text())["plans_sha256"] != prior.prior.sha(prior.PLANS):
        raise ValueError("source seal mismatch")
    print("Fixed endpoint exact feedback: reusing sealed first decisions; no RGB claim.", flush=True)
    rows = []
    for old in source["rows"]:
        if old["arm"] not in feedback.ARMS:
            continue
        root, task, arm = old["root"], old["task"], old["arm"]
        selected = old["result"]["selected"]
        initial_chain = np.asarray(selected["chain"])
        first_state, blocked = advance(task["start"], initial_chain[:5])
        stages = [{"step": 0, "start": task["start"], "horizon": 15, "source": "sealed_prior",
                   "selected": selected, "tokens": old["tokens"], "dispatched": initial_chain[:5].tolist(),
                   "end_state": first_state, "blocked": blocked}]
        state, carry = first_state, initial_chain[5:]
        for step in (1, 2):
            local_task = {"start": state, "goal": task["goal"]}
            meter, trace = prior.prior.Budget(), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                positions, scores = meter.predict(local_task, requested)
                trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                              "positions": positions.tolist(), "scores": scores.tolist(),
                              "tokens": len(requested) * requested.shape[1] // 5})
                return positions, scores

            started = time.monotonic()
            result = feedback.decide(arm, root, step, task["goal"], predict, meter, carry)
            chain = np.asarray(result["selected"]["chain"])
            end, blocked = advance(state, chain[:5])
            stages.append({"step": step, "start": state, "horizon": 15 - 5 * step,
                           "result": result, "tokens": meter.token_samples, "trace": trace,
                           "wall_seconds": time.monotonic() - started, "dispatched": chain[:5].tolist(),
                           "end_state": end, "blocked": blocked})
            state, carry = end, chain[5:]
        rows.append({"root": root, "task": task, "arm": arm, "stages": stages,
                     "final_distance": float(np.linalg.norm(np.asarray(state[:2]) - task["goal"]))})
    if len(rows) != 48 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} fixed-terminal feedback trajectories.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, old = json.loads(PLANS.read_text()), json.loads(prior.PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    source = {(r["root"], r["arm"]): r for r in old["rows"]}
    summaries = []
    for row in data["rows"]:
        initial = source[(row["root"], row["arm"])]
        if row["task"] != initial["task"] or row["stages"][0]["selected"] != initial["result"]["selected"]:
            raise ValueError("first decision mismatch")
        state, previous_chain = row["task"]["start"], None
        for step, stage in enumerate(row["stages"]):
            if stage["step"] != step or stage["start"] != state or stage["horizon"] != 15 - 5 * step:
                raise ValueError("state or horizon mismatch")
            result = stage.get("result")
            selected = stage["selected"] if step == 0 else result["selected"]
            chain = np.asarray(selected["chain"])
            if len(chain) != stage["horizon"] or not np.array_equal(chain[:5], stage["dispatched"]):
                raise ValueError("dispatch mismatch")
            if step:
                np.testing.assert_array_equal(result["carry"], previous_chain[5:])
                local = {"start": state, "goal": row["task"]["goal"]}
                total, found = 0, False
                for batch in stage["trace"]:
                    requested = np.asarray(batch["requested"])
                    actions = np.asarray(batch["model"], dtype=np.float32)
                    np.testing.assert_array_equal(requested.astype(np.float32), actions)
                    pos = prior.prior.exact.endpoints(local, actions)
                    scores = np.linalg.norm(pos - local["goal"], axis=1)
                    np.testing.assert_allclose(pos, batch["positions"], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
                    if requested.shape[1:] == chain.shape:
                        indices = np.flatnonzero(np.all(requested == chain, axis=(1, 2)))
                        found |= any(abs(scores[i] - selected["terminal_score"]) < 1e-12 for i in indices)
                    cost = len(actions) * actions.shape[1] // 5
                    if cost != batch["tokens"]:
                        raise ValueError("token mismatch")
                    total += cost
                if total != stage["tokens"] or total > 480 or not found:
                    raise ValueError("unscored selection or budget exceeded")
                np.testing.assert_array_equal(stage["trace"][0]["requested"][0], result["carry"])
                np.testing.assert_allclose(stage["trace"][0]["scores"][0], result["carry_score"], atol=1e-12)
            actual_state, blocked = advance(state, stage["dispatched"])
            np.testing.assert_allclose(actual_state, stage["end_state"], atol=1e-12, rtol=0)
            if blocked != stage["blocked"]:
                raise ValueError("collision mismatch")
            state, previous_chain = actual_state, chain
        np.testing.assert_allclose(row["final_distance"], np.linalg.norm(np.asarray(state[:2]) - row["task"]["goal"]), atol=1e-12)
        summaries.append({"root": row["root"], "arm": row["arm"], "tokens": [x["tokens"] for x in row["stages"]]})
    if len(summaries) != 48 or len({(r["root"], r["arm"]) for r in summaries}) != 48:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": prior.prior.sha(PLANS), "rows": summaries}, stream, indent=2)
    print(f"Audited {len(summaries)} exact feedback trajectories.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
