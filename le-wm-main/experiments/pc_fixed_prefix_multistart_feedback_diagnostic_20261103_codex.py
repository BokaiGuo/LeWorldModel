"""Equal-budget multistart suffix planning at previously sealed prefixes."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_fixed_prefix_equal_feedback_optimizer_20261103_codex as base

ROOT = base.ROOT
PROTOCOL = ROOT.parent / "docs/PC_FIXED_PREFIX_MULTISTART_FEEDBACK_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_fixed_prefix_multistart_feedback_diagnostic_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), Path(base.__file__), PROTOCOL, base.PLANS, base.AUDIT)
    return {**base.dependencies(), **{str(p): base.sha(p) for p in paths}}


def initials(root, stage, horizon):
    rng = np.random.default_rng(2027030100 + 10 * root + stage)
    return np.concatenate((np.zeros((1, horizon, 2)), rng.uniform(-.6, .6, (3, horizon, 2))))


def optimize(root, stage, horizon, counts, predict):
    starts = initials(root, stage, horizon)
    all_best, all_score = None, float("inf")
    for restart, count in enumerate(counts):
        loc = starts[restart].copy()
        scale = np.full((horizon, 2), 1 / 3)
        rng = np.random.default_rng(2027040100 + 100 * root + 10 * stage + restart)
        for begin in range(0, count, 12):
            size = min(12, count - begin)
            samples = np.clip(loc + rng.normal(size=(size, horizon, 2)) * scale, -1, 1)
            samples[0] = loc
            values = np.asarray(predict(samples, restart), dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError("invalid score")
            for chain, value in zip(samples, values, strict=True):
                if value < all_score:
                    all_best, all_score = chain.copy(), float(value)
            if size > 1:
                elite = samples[np.argsort(values, kind="stable")[:min(4, size)]]
                loc, scale = elite.mean(0), np.maximum(elite.std(0), .05)
    return all_best, all_score, starts


def run_one(row, name):
    root, task = row["root"], row["task"]
    prefix = np.asarray(row["controls"][name]["prefix"])
    state1, blocked1 = base.source.feedback.advance(task["start"], prefix)
    local1 = {"start": state1, "goal": task["goal"]}
    traces = [[], []]

    def predict(stage, local, actions, restart):
        requested = np.asarray(actions, dtype=float)
        scores = base.score(local, requested)
        traces[stage].append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                              "scores": scores.tolist(), "restart": restart,
                              "tokens": len(requested) * requested.shape[1] // 5})
        return scores

    ten, ten_score, starts1 = optimize(root, 0, 10, (60, 60, 60, 60),
                                        lambda actions, restart: predict(0, local1, actions, restart))
    state2, blocked2 = base.source.feedback.advance(state1, ten[:5])
    local2 = {"start": state2, "goal": task["goal"]}
    carry = ten[5:]
    carry_score = float(predict(1, local2, carry[None], -1)[0])
    five, five_score, starts2 = optimize(root, 1, 5, (120, 120, 120, 119),
                                          lambda actions, restart: predict(1, local2, actions, restart))
    chosen = five if five_score < carry_score else carry
    end, blocked3 = base.source.feedback.advance(state2, chosen)
    chain = np.concatenate((prefix, ten[:5], chosen))
    distance = float(np.linalg.norm(np.asarray(end[:2]) - task["goal"]))
    np.testing.assert_allclose(distance, min(carry_score, five_score), atol=1e-12, rtol=0)
    return {"root": root, "name": name, "task": task, "prefix": prefix.tolist(),
            "nonprogress": row["nonprogress"], "first_state": state1, "second_state": state2,
            "end_state": end, "blocked": [blocked1, blocked2, blocked3], "starts": [starts1.tolist(), starts2.tolist()],
            "ten_chain": ten.tolist(), "ten_score": ten_score, "carry_score": carry_score,
            "five_chain": five.tolist(), "five_score": five_score, "selected_five": chosen.tolist(),
            "chain": chain.tolist(), "final_distance": distance, "traces": traces}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    hashes = dependencies()
    data = json.loads(base.source.OUTPUT.read_text())
    if data["dependencies"] != base.source.dependencies():
        raise ValueError("source drift")
    previous = json.loads(base.PLANS.read_text())
    if previous["dependencies"] != base.dependencies() or json.loads(base.AUDIT.read_text())["plans_sha256"] != base.sha(base.PLANS):
        raise ValueError("single-start seal mismatch")
    print("Four paired CEM initializations at each fixed prefix; no historical action injection.", flush=True)
    rows = [run_one(row, name) for row in data["rows"] for name in row["controls"]]
    if len(rows) != 120 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} multistart feedback runs.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    # The base auditor checks source prefixes, each physical prediction, both
    # budgets, selected incumbents, state transitions and whole-chain replay.
    original = base.PLANS, base.AUDIT, base.dependencies
    try:
        base.PLANS, base.AUDIT, base.dependencies = PLANS, AUDIT, dependencies
        base.audit()
    finally:
        base.PLANS, base.AUDIT, base.dependencies = original


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
