"""Predeclared exact fixed-terminal feedback panel for temporal response search."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_search_e0_dev_20261104_codex as previous
from experiments.spectral_reachability import pc_response_temporal_feedback_v1 as feedback
from experiments.pc_equal_resource_dev_20261103_codex import Meter

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_FIXED_TERMINAL_E0_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_fixed_terminal_e0_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)


def dependencies():
    files = (Path(__file__), PROTOCOL, Path(feedback.__file__), previous.PLANS, previous.AUDIT,
             previous.MANIFEST, previous.ROOT / "results/pc_response_temporal_search_e0_dev_20261104_codex_planning_rebuild.json")
    return {**previous.dependencies(), **{str(p.resolve()): sha(p) for p in files}}


def step_physics(state, actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (5, 2):
        raise ValueError("expected one action packet")
    padded = np.pad(actions[None], ((0, 0), (0, 11), (0, 0))) * previous.exact.wall.HIGH
    trajectory, blocked = previous.exact.wall.rollout(state, padded)
    return trajectory[0, 5].tolist(), blocked[0, :5].tolist()


def make_decision(task, root, arm, step, carry):
    meter, trace = Meter(previous.method.CAP), []
    goal = np.asarray(task["goal"])

    def predict(actions):
        requested = np.asarray(actions, dtype=float)
        raw = requested.astype(np.float32)

        def evaluate(batch):
            pos = previous.exact.endpoints(task, batch)
            return pos, np.linalg.norm(pos - goal, axis=1)

        positions, scores = meter.predict(raw, evaluate)
        trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                      "positions": positions.tolist(), "scores": scores.tolist(),
                      "tokens": len(raw) * raw.shape[1] // 5})
        return positions, scores

    started = time.monotonic()
    result = feedback.decide(arm, root, step, goal, predict, meter, carry)
    return {"result": result, "counts": meter.as_dict(), "trace": trace,
            "planning_seconds": time.monotonic() - started}


def plan():
    if any(p.exists() for p in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("prior or partial panel exists")
    original = json.loads(previous.PLANS.read_text())
    if (original["dependencies"] != previous.dependencies()
            or json.loads(previous.AUDIT.read_text())["plans_sha256"] != sha(previous.PLANS)):
        raise ValueError("prior planning seal mismatch")
    hashes = dependencies()
    tasks = previous.tasks.tasks()
    write(MANIFEST, {"dependencies": hashes, "tasks": tasks, "arms": previous.method.ARMS,
                     "method": feedback.METHOD, "budget_per_decision": previous.method.CAP})
    print("Fixed-terminal paired feedback; each arm executes its own trajectory.", flush=True)
    rows = []
    for root, task in enumerate(tasks):
        for arm in previous.method.ARMS:
            current, carry, decisions = task["start"], None, []
            for step in range(3):
                context = {"start": current, "goal": task["goal"]}
                decision = make_decision(context, root, arm, step, carry)
                selected = decision["result"]["selected"]["chain"]
                executed = np.asarray(selected[:5], dtype=np.float32)
                next_state, blocked = step_physics(current, executed)
                decisions.append({"step": step, "context": context, "carry_in": carry,
                                  "decision": decision, "executed_float32": executed.tolist(),
                                  "next_state": next_state, "blocked": blocked})
                current, carry = next_state, selected[5:] if step < 2 else None
            distance = float(np.linalg.norm(np.asarray(current[:2]) - np.asarray(task["goal"])))
            rows.append({"root": root, "arm": arm, "task": task, "decisions": decisions,
                         "final_state": current, "final_distance": distance})
        print(f"root={root} all four feedback arms complete", flush=True)
    if hashes != dependencies():
        raise ValueError("dependency drift")
    write(PLANS, {"manifest_sha256": sha(MANIFEST), "dependencies": hashes, "rows": rows})
    print("Feedback panel sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    source, original = json.loads(PLANS.read_text()), json.loads(previous.PLANS.read_text())
    if source["dependencies"] != dependencies() or source["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("dependencies changed")
    old = {(r["root"], r["arm"]): r for r in original["rows"]}
    findings = []
    for row in source["rows"]:
        root, arm, task = row["root"], row["arm"], row["task"]
        current, carry = task["start"], None
        for step, item in enumerate(row["decisions"]):
            context, decision = item["context"], item["decision"]
            if item["step"] != step or context != {"start": current, "goal": task["goal"]} or item["carry_in"] != carry:
                raise ValueError("feedback context or carry mismatch")
            trace, scores, count = decision["trace"], [], 0
            selected = decision["result"]["selected"]
            chain, selected_score = np.asarray(selected["chain"]), selected["terminal_score"]
            found = False
            for batch in trace:
                requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
                if not np.array_equal(requested.astype(np.float32), raw):
                    raise ValueError("float conversion mismatch")
                position = previous.exact.endpoints(context, raw)
                values = np.linalg.norm(position - np.asarray(task["goal"]), axis=1)
                np.testing.assert_allclose(position, batch["positions"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(values, batch["scores"], rtol=0, atol=1e-12)
                tokens = len(raw) * raw.shape[1] // 5
                if tokens != batch["tokens"]:
                    raise ValueError("batch accounting mismatch")
                count += tokens
                if raw.shape[1] == len(chain):
                    scores.extend(values.tolist())
                    found |= any(abs(values[i] - selected_score) < 1e-12
                                 for i in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
            if (not found or abs(min(scores) - selected_score) > 1e-12 or count > previous.method.CAP
                    or count != decision["counts"]["prediction_token_samples"]
                    or len(trace) != decision["counts"]["prediction_calls"]):
                raise ValueError("selected plan or meter mismatch")
            if step == 0 and selected != old[root, arm]["result"]["selected"]:
                raise ValueError("first decision differs from sealed open-loop source")
            if step and (decision["result"]["carry"] != carry
                         or trace[0]["requested"] != [carry]
                         or decision["result"]["carry_score"] != trace[0]["scores"][0]):
                raise ValueError("carry rescoring mismatch")
            executed = np.asarray(chain[:5], dtype=np.float32)
            state, blocked = step_physics(current, executed)
            if item["executed_float32"] != executed.tolist() or item["next_state"] != state or item["blocked"] != blocked:
                raise ValueError("physical dispatch mismatch")
            current, carry = state, chain[5:].tolist() if step < 2 else None
        distance = float(np.linalg.norm(np.asarray(current[:2]) - np.asarray(task["goal"])))
        if current != row["final_state"] or distance != row["final_distance"]:
            raise ValueError("final state mismatch")
        findings.append({"root": root, "arm": arm, "first_chain_match": True, "final_distance": distance})
    if len(findings) != 96:
        raise ValueError("panel mismatch")
    write(AUDIT, {"plans_sha256": sha(PLANS), "rows": findings})
    print("Audited all feedback batches, carry and dispatch; first chains match sealed plans.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
