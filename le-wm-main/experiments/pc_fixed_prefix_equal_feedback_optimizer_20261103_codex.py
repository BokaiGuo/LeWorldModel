"""Fixed-prefix equal-budget exact-dynamics feedback optimizer diagnostic."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_feedback_executed_tail_intervention_20261103_codex as source
from experiments.spectral_reachability import pc_cem_scale_floor_v1 as cem

ROOT = source.feedback.prior.prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_FIXED_PREFIX_EQUAL_FEEDBACK_OPTIMIZER_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_fixed_prefix_equal_feedback_optimizer_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
sha = source.feedback.prior.prior.sha
exact = source.diag.exp.prior.exact


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(cem.__file__), source.OUTPUT)
    return {**source.dependencies(), **{str(p): sha(p) for p in paths}}


def score(task, actions):
    model = np.asarray(actions, dtype=np.float32)
    if model.ndim != 3 or model.shape[-1] != 2 or not np.isfinite(model).all() or np.any(np.abs(model) > 1):
        raise ValueError("invalid actions")
    pos = exact.endpoints(task, model)
    return np.linalg.norm(pos - task["goal"], axis=1)


def run_one(row, name):
    task, root = row["task"], row["root"]
    prefix = np.asarray(row["controls"][name]["prefix"])
    first_state, first_blocked = source.feedback.advance(task["start"], prefix)
    local = {"start": first_state, "goal": task["goal"]}
    traces = [[], []]

    def predict(stage, local_task, actions):
        actions = np.asarray(actions, dtype=float)
        values = score(local_task, actions)
        traces[stage].append({"requested": actions.tolist(), "model": actions.astype(np.float32).tolist(),
                              "scores": values.tolist(), "tokens": len(actions) * actions.shape[1] // 5})
        return values

    ten, ten_score, _ = cem.optimize([], 10, 240, np.random.default_rng(2027010100 + root),
                                      lambda actions: predict(0, local, actions))
    second_state, second_blocked = source.feedback.advance(first_state, ten[:5])
    remaining = {"start": second_state, "goal": task["goal"]}
    carry = ten[5:]
    carry_score = float(predict(1, remaining, carry[None])[0])
    five, five_score, _ = cem.optimize([], 5, 479, np.random.default_rng(2027011100 + root),
                                       lambda actions: predict(1, remaining, actions))
    chosen = five if five_score < carry_score else carry
    chosen_score = min(five_score, carry_score)
    end, third_blocked = source.feedback.advance(second_state, chosen)
    chain = np.concatenate((prefix, ten[:5], chosen))
    distance = float(np.linalg.norm(np.asarray(end[:2]) - task["goal"]))
    np.testing.assert_allclose(distance, chosen_score, atol=1e-12, rtol=0)
    return {"root": root, "name": name, "task": task, "prefix": prefix.tolist(),
            "nonprogress": row["nonprogress"], "first_state": first_state, "second_state": second_state,
            "end_state": end, "blocked": [first_blocked, second_blocked, third_blocked],
            "ten_chain": ten.tolist(), "ten_score": float(ten_score), "carry_score": carry_score,
            "five_chain": five.tolist(), "five_score": float(five_score), "selected_five": chosen.tolist(),
            "chain": chain.tolist(), "final_distance": distance, "traces": traces}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    hashes = dependencies()
    data = json.loads(source.OUTPUT.read_text())
    if data["dependencies"] != source.dependencies():
        raise ValueError("source drift")
    print("Equal per-prefix feedback optimization on exposed roots; no posterior tail injection.", flush=True)
    rows = [run_one(row, name) for row in data["rows"] for name in row["controls"]]
    if len(rows) != 120 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} equal-budget feedback runs.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, original = json.loads(PLANS.read_text()), json.loads(source.OUTPUT.read_text())
    if data["dependencies"] != dependencies() or original["dependencies"] != source.dependencies():
        raise ValueError("source drift")
    original = {r["root"]: r for r in original["rows"]}
    seen, summaries = set(), []
    for row in data["rows"]:
        root, name = row["root"], row["name"]
        key = (root, name)
        if key in seen:
            raise ValueError("duplicate")
        seen.add(key)
        old = original[root]
        if row["task"] != old["task"] or row["nonprogress"] != old["nonprogress"]:
            raise ValueError("task mismatch")
        np.testing.assert_array_equal(row["prefix"], old["controls"][name]["prefix"])
        state1, blocked1 = source.feedback.advance(row["task"]["start"], row["prefix"])
        state2, blocked2 = source.feedback.advance(state1, np.asarray(row["ten_chain"])[:5])
        state3, blocked3 = source.feedback.advance(state2, row["selected_five"])
        for actual, expected in ((state1, row["first_state"]), (state2, row["second_state"]), (state3, row["end_state"])):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        if [blocked1, blocked2, blocked3] != row["blocked"]:
            raise ValueError("blocked mismatch")
        for stage, state, horizon, expected_count in ((0, state1, 10, 240), (1, state2, 5, 480)):
            task = {"start": state, "goal": row["task"]["goal"]}
            actions, scores, tokens = [], [], 0
            for batch in row["traces"][stage]:
                requested = np.asarray(batch["requested"])
                model = np.asarray(batch["model"], dtype=np.float32)
                np.testing.assert_array_equal(requested.astype(np.float32), model)
                if requested.shape[1:] != (horizon, 2):
                    raise ValueError("horizon mismatch")
                values = score(task, model)
                np.testing.assert_allclose(values, batch["scores"], atol=1e-12, rtol=0)
                cost = len(requested) * horizon // 5
                if cost != batch["tokens"]:
                    raise ValueError("token mismatch")
                tokens += cost
                actions.extend(requested)
                scores.extend(values)
            actions, scores = np.asarray(actions), np.asarray(scores)
            if len(actions) != expected_count or tokens != 480:
                raise ValueError("budget mismatch")
            if stage == 0:
                best = int(np.argmin(scores))
                np.testing.assert_array_equal(actions[best], row["ten_chain"])
                np.testing.assert_allclose(scores[best], row["ten_score"], atol=1e-12, rtol=0)
            else:
                np.testing.assert_array_equal(actions[0], np.asarray(row["ten_chain"])[5:])
                np.testing.assert_allclose(scores[0], row["carry_score"], atol=1e-12, rtol=0)
                best = int(np.argmin(scores[1:])) + 1
                np.testing.assert_array_equal(actions[best], row["five_chain"])
                np.testing.assert_allclose(scores[best], row["five_score"], atol=1e-12, rtol=0)
                selected = best if scores[best] < scores[0] else 0
                np.testing.assert_array_equal(actions[selected], row["selected_five"])
        chain = np.concatenate((row["prefix"], np.asarray(row["ten_chain"])[:5], row["selected_five"]))
        np.testing.assert_array_equal(chain, row["chain"])
        np.testing.assert_allclose(score(row["task"], chain[None])[0], row["final_distance"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(state3[:2]) - row["task"]["goal"]),
                                   row["final_distance"], atol=1e-12, rtol=0)
        summaries.append({"root": root, "name": name, "tokens": [480, 480]})
    if len(seen) != 120:
        raise ValueError("panel mismatch")
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": summaries}, stream, indent=2)
    print(f"Replayed {len(summaries)} fixed-prefix feedback runs.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
