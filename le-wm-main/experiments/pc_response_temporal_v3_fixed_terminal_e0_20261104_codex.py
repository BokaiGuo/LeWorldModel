"""v3 exact fixed-terminal feedback development panel and replay audit."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_fixed_terminal_e0_20261104_codex as prior
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_feedback_v3 as method

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_FIXED_TERMINAL_E0_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_v3_fixed_terminal_e0_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(method.__file__), prior.PLANS, prior.AUDIT)
    return {**prior.dependencies(), **{str(p.resolve()): prior.sha(p) for p in paths}}


def plan():
    if any(path.exists() for path in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("panel or partial panel already exists")
    if json.loads(prior.AUDIT.read_text())["plans_sha256"] != prior.sha(prior.PLANS):
        raise ValueError("prior panel seal mismatch")
    hashes = dependencies()
    tasks = prior.previous.tasks.tasks()
    prior.write(MANIFEST, {"dependencies": hashes, "tasks": tasks, "arms": prior.previous.method.ARMS,
                           "method": method.METHOD, "budget_per_decision": 1920})
    print("v3 exact feedback on exposed roots: development comparison, not independent confirmation.", flush=True)
    rows = []
    for root, task in enumerate(tasks):
        for arm in prior.previous.method.ARMS:
            current, carry, decisions = task["start"], None, []
            for step in range(3):
                context = {"start": current, "goal": task["goal"]}
                meter, trace = Meter(1920), []

                def predict(actions):
                    requested = np.asarray(actions, dtype=float)
                    raw = requested.astype(np.float32)

                    def evaluate(batch):
                        pos = prior.previous.exact.endpoints(context, batch)
                        return pos, np.linalg.norm(pos - np.asarray(context["goal"]), axis=1)

                    positions, scores = meter.predict(raw, evaluate)
                    trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                                  "positions": positions.tolist(), "scores": scores.tolist(),
                                  "tokens": len(raw) * raw.shape[1] // 5})
                    return positions, scores

                started = time.monotonic()
                result = method.decide(arm, root, step, task["goal"], predict, meter, carry)
                decision = {"result": result, "counts": meter.as_dict(), "trace": trace,
                            "planning_seconds": time.monotonic() - started}
                selected = result["selected"]["chain"]
                executed = np.asarray(selected[:5], dtype=np.float32)
                next_state, blocked = prior.step_physics(current, executed)
                decisions.append({"step": step, "context": context, "carry_in": carry, "decision": decision,
                                  "executed_float32": executed.tolist(), "next_state": next_state,
                                  "blocked": blocked})
                current, carry = next_state, selected[5:] if step < 2 else None
            rows.append({"root": root, "arm": arm, "task": task, "decisions": decisions,
                         "final_state": current,
                         "final_distance": float(np.linalg.norm(np.asarray(current[:2]) - np.asarray(task["goal"])))})
        print(f"root={root} four arms complete", flush=True)
    if len(rows) != 96 or dependencies() != hashes:
        raise ValueError("panel size or dependency drift")
    prior.write(PLANS, {"manifest_sha256": prior.sha(MANIFEST), "dependencies": hashes, "rows": rows})
    print("Panel sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data, manifest = json.loads(PLANS.read_text()), json.loads(MANIFEST.read_text())
    if (data["manifest_sha256"] != prior.sha(MANIFEST) or data["dependencies"] != dependencies()
            or manifest["dependencies"] != data["dependencies"]):
        raise ValueError("dependency or manifest seal mismatch")
    findings, seen = [], set()
    for row in data["rows"]:
        root, arm, task = row["root"], row["arm"], row["task"]
        key = (root, arm)
        if key in seen or task != manifest["tasks"][root] or arm not in manifest["arms"]:
            raise ValueError("task or arm mismatch")
        seen.add(key)
        current, carry = task["start"], None
        for step, item in enumerate(row["decisions"]):
            if (item["step"] != step or item["context"] != {"start": current, "goal": task["goal"]}
                    or item["carry_in"] != carry):
                raise ValueError("context mismatch")
            context, decision = item["context"], item["decision"]
            result, trace = decision["result"], decision["trace"]
            chain = np.asarray(result["selected"]["chain"])
            score = result["selected"]["terminal_score"]
            full, tokens, found = [], 0, False
            for batch in trace:
                requested = np.asarray(batch["requested"])
                raw = np.asarray(batch["raw"], dtype=np.float32)
                if not np.array_equal(requested.astype(np.float32), raw):
                    raise ValueError("float32 mismatch")
                positions = prior.previous.exact.endpoints(context, raw)
                values = np.linalg.norm(positions - np.asarray(task["goal"]), axis=1)
                np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(values, batch["scores"], rtol=0, atol=1e-12)
                batch_tokens = len(raw) * raw.shape[1] // 5
                if batch_tokens != batch["tokens"]:
                    raise ValueError("token mismatch")
                tokens += batch_tokens
                if raw.shape[1] == len(chain):
                    full.extend((requested[i].tolist(), float(value)) for i, value in enumerate(values))
                    found |= any(values[i] == score for i in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
            if (not found or min(value for _, value in full) != score or tokens > 1920
                    or tokens != decision["counts"]["prediction_token_samples"]
                    or len(trace) != decision["counts"]["prediction_calls"]):
                raise ValueError("selection or ledger mismatch")
            if step and (trace[0]["requested"] != [carry] or result["carry_score"] != trace[0]["scores"][0]):
                raise ValueError("carry mismatch")
            if arm in ("response", "matched_random"):
                start = result["refinement_start"]
                before = full[:start["full_samples_seen"]]
                if (len(before) != start["full_samples_seen"] or not before
                        or min(value for _, value in before) != start["terminal_score"]
                        or (start["chain"], start["terminal_score"]) not in before):
                    raise ValueError("joint refinement start mismatch")
                pre = result["precomparison"]
                comparison = np.asarray(pre["chain"]).copy()
                comparison[:5] = 0
                if (result["comparison_chain"] != comparison.tolist()
                        or trace[-1]["requested"] != [comparison.tolist()]
                        or trace[-1]["scores"][0] != result["comparison_score"]
                        or result["abstained"] != (result["comparison_score"] <= pre["terminal_score"])):
                    raise ValueError("abstention mismatch")
            executed = np.asarray(chain[:5], dtype=np.float32)
            state, blocked = prior.step_physics(current, executed)
            if item["executed_float32"] != executed.tolist() or item["next_state"] != state or item["blocked"] != blocked:
                raise ValueError("physical dispatch mismatch")
            current, carry = state, chain[5:].tolist() if step < 2 else None
        distance = float(np.linalg.norm(np.asarray(current[:2]) - np.asarray(task["goal"])))
        if current != row["final_state"] or distance != row["final_distance"]:
            raise ValueError("final state mismatch")
        findings.append({"root": root, "arm": arm, "final_distance": distance})
    if seen != {(root, arm) for root in range(24) for arm in manifest["arms"]}:
        raise ValueError("incomplete panel")
    prior.write(AUDIT, {"plans_sha256": prior.sha(PLANS), "rows": findings})
    print("Audited all predictions, refinement starts, abstentions, carry and dispatch.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
