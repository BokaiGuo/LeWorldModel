"""Paired E0 evaluation and trace audit for canonical candidate RNG streams."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_tail_local_mechanism_20261104_codex as source
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_target_reach_canonical_v5 as method
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_CANONICAL_RETENTION_E0_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_canonical_retention_e0_20261104_codex"
MANIFEST, PLANS, AUDIT = [PREFIX.with_name(PREFIX.name + suffix + ".json")
                          for suffix in ("_manifest", "_plans", "_audit")]
sha, write, exact = source.prior.base.sha, source.prior.base.write, source.prior.base.exact


def plan():
    if any(path.exists() for path in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("existing or partial experiment")
    tasks = source.tasks()
    dependencies = seal.capture([Path(__file__), PROTOCOL, Path(method.__file__), Path(v4.__file__),
        Path(source.__file__), Path(exact.__file__), Path(exact.wall.__file__), source.PLANS, source.AUDIT])
    if json.loads(source.AUDIT.read_text())["plans_sha256"] != sha(source.PLANS):
        raise ValueError("source task panel audit mismatch")
    write(MANIFEST, {"seal": dependencies, "tasks": tasks, "strategies": method.STRATEGIES,
        "budget": 1920, "scope": "exposed E0 development; no independent confirmation"})
    print("V5 paired retention strategies on 24 exposed E0 tasks.", flush=True)
    rows = []
    for root, task in enumerate(tasks):
        goal = np.asarray(task["goal"])
        for strategy in method.STRATEGIES:
            meter, trace = Meter(1920), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)

                def evaluate(batch):
                    positions = exact.endpoints(task, batch)
                    return positions, np.linalg.norm(positions-goal, axis=1)

                positions, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                    "positions": positions.tolist(), "scores": scores.tolist(),
                    "tokens": len(raw) * raw.shape[1] // 5})
                return positions, scores

            result = method.decide(strategy, "response", root, 0, goal, predict, meter)
            rows.append({"root": root, "strategy": strategy, "task": task,
                         "result": result, "counts": meter.as_dict(), "trace": trace})
        print(f"planned root {root}", flush=True)
    seal.verify(dependencies)
    write(PLANS, {"manifest_sha256": sha(MANIFEST), "rows": rows})
    print("Saved 48 paired plans.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest, plans = json.loads(MANIFEST.read_text()), json.loads(PLANS.read_text())
    seal.verify(manifest["seal"])
    if plans["manifest_sha256"] != sha(MANIFEST) or manifest["tasks"] != source.tasks():
        raise ValueError("manifest binding mismatch")
    data = {(r["root"], r["strategy"]): r for r in plans["rows"]}
    if len(data) != 48 or len(plans["rows"]) != 48:
        raise ValueError("incomplete or duplicate rows")
    total, common_count, common_budget_differences = 0, 0, []
    for root, task in enumerate(source.tasks()):
        pair = [data[(root, s)] for s in method.STRATEGIES]
        if (pair[0]["result"]["first_bank"] != pair[1]["result"]["first_bank"]
                or pair[0]["result"]["prefix_scores"] != pair[1]["result"]["prefix_scores"]
                or pair[0]["result"]["successor_responses"] != pair[1]["result"]["successor_responses"]):
            raise ValueError("paired candidate bank mismatch")
        programs = [{p["parent_index"]: p for p in row["result"]["programs"]} for row in pair]
        for index in programs[0].keys() & programs[1].keys():
            left, right = programs[0][index], programs[1][index]
            if left["rng_seed"] != right["rng_seed"] or left["rng_seed"] != 2027110400 + root*100 + 100000 + index:
                raise ValueError("common candidate RNG mismatch")
            common_count += 1
            common_budget_differences.append(left["samples"] - right["samples"])
        for row in pair:
            if row["task"] != task or row["result"]["retained"] != sorted(row["result"]["retained"]):
                raise ValueError("task or candidate order mismatch")
            chain = np.asarray(row["result"]["selected"]["chain"])
            score = row["result"]["selected"]["terminal_score"]
            found, full, tokens, samples = False, [], 0, 0
            for batch in row["trace"]:
                requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
                np.testing.assert_array_equal(requested.astype(np.float32), raw)
                positions = exact.endpoints(task, raw)
                values = np.linalg.norm(positions-np.asarray(task["goal"]), axis=1)
                np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(values, batch["scores"], atol=1e-12, rtol=0)
                cost = len(raw)*raw.shape[1]//5
                if cost != batch["tokens"]:
                    raise ValueError("batch token mismatch")
                tokens, samples = tokens + cost, samples + len(raw)
                if raw.shape[1] == 15:
                    full.extend(values.tolist())
                    found |= any(values[j] == score for j in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
            if (not found or min(full) != score or tokens > 1920
                    or tokens != row["counts"]["prediction_token_samples"]
                    or samples != row["counts"]["trajectory_samples"]
                    or len(row["trace"]) != row["counts"]["prediction_calls"]):
                raise ValueError("selection or cost mismatch")
            total += samples
    seal.verify(manifest["seal"])
    write(AUDIT, {"plans_sha256": sha(PLANS), "verified_plans": len(data), "trajectory_samples": total,
        "common_candidate_pairs": common_count, "common_sample_count_differences": common_budget_differences,
        "scope": "same backend trace recomputation and paired candidate seed audit"})
    print(f"Verified {len(data)} plans, {total} trajectories and {common_count} shared candidates.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
