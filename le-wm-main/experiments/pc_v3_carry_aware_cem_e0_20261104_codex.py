"""Carry-aware smooth CEM on the sealed v3 fixed-terminal E0 development tasks."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_fixed_terminal_e0_20261104_codex as old
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_warm_cem_v1 as warm
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_V3_CARRY_AWARE_CEM_E0_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v3_carry_aware_cem_e0_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def source():
    prior = old.prior
    data = json.loads(old.PLANS.read_text())
    audit = json.loads(old.AUDIT.read_text())
    if audit["plans_sha256"] != prior.sha(old.PLANS) or data["dependencies"] != old.dependencies():
        raise ValueError("old source audit or dependencies changed")
    rows = {(r["root"], r["arm"]): r for r in data["rows"]}
    tasks = prior.previous.tasks.tasks()
    if any((i, arm) not in rows or rows[i, arm]["task"] != task
           for i, task in enumerate(tasks) for arm in ("response", "smooth_cem")):
        raise ValueError("source task mismatch")
    return tasks, rows


def plan():
    if any(p.exists() for p in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("existing or partial panel")
    tasks, old_rows = source()
    extra = [Path(__file__), PROTOCOL, old.PLANS, old.AUDIT, old.MANIFEST, Path(seal.__file__)]
    extra.extend(Path(p) for p in old.dependencies())
    hashes = seal.capture(extra)
    old.prior.write(MANIFEST, {"seal": hashes, "tasks": tasks, "cap": 1920,
                               "scope": "exposed roots, baseline development"})
    print("Carry-aware CEM: 24 exposed Wall tasks, full fixed-terminal feedback.", flush=True)
    rows = []
    for root, task in enumerate(tasks):
        current, carry, decisions = task["start"], None, []
        for step in range(3):
            context = {"start": current, "goal": task["goal"]}
            meter, trace = Meter(1920), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)

                def evaluate(batch):
                    positions = old.prior.previous.exact.endpoints(context, batch)
                    return positions, np.linalg.norm(positions - np.asarray(task["goal"]), axis=1)

                positions, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                              "positions": positions.tolist(), "scores": scores.tolist(),
                              "tokens": len(raw) * raw.shape[1] // 5})
                return positions, scores

            started = time.monotonic()
            result = warm.decide("smooth_cem", root, step, task["goal"], predict, meter, carry)
            selected = np.asarray(result["selected"]["chain"])
            executed = selected[:5].astype(np.float32)
            state, blocked = old.prior.step_physics(current, executed)
            decisions.append({"step": step, "context": context, "carry_in": carry, "result": result,
                              "counts": meter.as_dict(), "trace": trace,
                              "planning_seconds": time.monotonic() - started,
                              "executed": executed.tolist(), "next_state": state, "blocked": blocked})
            current, carry = state, selected[5:].tolist() if step < 2 else None
        rows.append({"root": root, "task": task, "decisions": decisions, "final_state": current,
                     "final_distance": float(np.linalg.norm(np.asarray(current[:2]) - task["goal"])),
                     "response_distance": old_rows[root, "response"]["final_distance"],
                     "smooth_distance": old_rows[root, "smooth_cem"]["final_distance"]})
        print(f"root={root} complete", flush=True)
    seal.verify(hashes)
    old.prior.write(PLANS, {"manifest_sha256": old.prior.sha(MANIFEST), "rows": rows})
    print("Panel sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    tasks, old_rows = source()
    manifest, data = json.loads(MANIFEST.read_text()), json.loads(PLANS.read_text())
    seal.verify(manifest["seal"])
    if data["manifest_sha256"] != old.prior.sha(MANIFEST) or manifest["tasks"] != tasks:
        raise ValueError("manifest mismatch")
    for row in data["rows"]:
        root, task = row["root"], row["task"]
        if task != tasks[root] or row["response_distance"] != old_rows[root, "response"]["final_distance"] \
                or row["smooth_distance"] != old_rows[root, "smooth_cem"]["final_distance"]:
            raise ValueError("source mismatch")
        current, carry = task["start"], None
        for step, item in enumerate(row["decisions"]):
            if item["step"] != step or item["carry_in"] != carry or item["context"] != {
                    "start": current, "goal": task["goal"]}:
                raise ValueError("feedback mismatch")
            result, trace = item["result"], item["trace"]
            chain, score = np.asarray(result["selected"]["chain"]), result["selected"]["terminal_score"]
            tokens, full, found = 0, [], False
            for batch in trace:
                requested = np.asarray(batch["requested"])
                raw = np.asarray(batch["raw"], dtype=np.float32)
                if not np.array_equal(requested.astype(np.float32), raw):
                    raise ValueError("raw action mismatch")
                positions = old.prior.previous.exact.endpoints(item["context"], raw)
                scores = np.linalg.norm(positions - np.asarray(task["goal"]), axis=1)
                np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
                cost = len(raw) * raw.shape[1] // 5
                if cost != batch["tokens"]:
                    raise ValueError("token mismatch")
                tokens += cost
                full.extend(scores.tolist())
                found |= any(scores[i] == score for i in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
            if not found or min(full) != score or tokens > 1920 \
                    or tokens != item["counts"]["prediction_token_samples"]:
                raise ValueError("selection or budget mismatch")
            if step and (trace[0]["requested"] != [np.zeros_like(carry).tolist(), carry]
                         or result["initialization"] != ("zero" if trace[0]["scores"][0] <= trace[0]["scores"][1] else "carry")):
                raise ValueError("initialization mismatch")
            executed = chain[:5].astype(np.float32)
            state, blocked = old.prior.step_physics(current, executed)
            if item["executed"] != executed.tolist() or item["next_state"] != state or item["blocked"] != blocked:
                raise ValueError("dispatch mismatch")
            current, carry = state, chain[5:].tolist() if step < 2 else None
        if current != row["final_state"] or row["final_distance"] != float(
                np.linalg.norm(np.asarray(current[:2]) - task["goal"])):
            raise ValueError("final mismatch")
    if len(data["rows"]) != len(tasks) or {r["root"] for r in data["rows"]} != set(range(len(tasks))):
        raise ValueError("incomplete panel")
    seal.verify(manifest["seal"])
    old.prior.write(AUDIT, {"plans_sha256": old.prior.sha(PLANS), "verified_roots": len(tasks),
                            "scope": "same exact dynamics implementation, independently recalculated"})
    print(f"Audited {len(tasks)} feedback trajectories.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
