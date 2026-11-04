"""Sealed V5 plans on a declared nonoverlapping E0 development grid."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v5_canonical_retention_e0_20261104_codex as old
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_target_reach_canonical_v5 as method
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = old.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_E0_PROSPECTIVE_MECHANISM_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_e0_prospective_mechanism_dev_20261104_codex"
MANIFEST, PLANS, AUDIT = [PREFIX.with_name(PREFIX.name + part + ".json")
                          for part in ("_manifest", "_plans", "_audit")]
sha, write, exact = old.sha, old.write, old.exact


def tasks():
    return [{"start": [-0.5, y, theta], "goal": [0.5, gy]}
            for y, theta, gy in itertools.product((-0.29, 0.08, 0.39), (-0.8, 0.25, 0.9), (-0.15, 0.15))]


def plan():
    if any(p.exists() for p in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("existing or partial experiment")
    previous = old.source.tasks()
    if (len(tasks()) != 18 or len({json.dumps(t, sort_keys=True) for t in tasks()}) != 18
            or any(task in previous for task in tasks())):
        raise ValueError("declared grid overlaps cited development panel or has duplicates")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(method.__file__), Path(v4.__file__),
        Path(old.__file__), Path(old.source.__file__), Path(exact.__file__), Path(exact.wall.__file__),
        ROOT / "experiments/pc_v5_e0_prospective_interventions_dev_20261104_codex.py",
        ROOT / "experiments/pc_target_reach_v4_local_response_diag_20261104_codex.py",
        ROOT / "experiments/pc_v5_suffix_budget_curve_diag_20261104_codex.py",
        old.PLANS, old.AUDIT])
    if json.loads(old.AUDIT.read_text())["plans_sha256"] != sha(old.PLANS):
        raise ValueError("prior v5 audit mismatch")
    write(MANIFEST, {"seal": deps, "tasks": tasks(), "method": method.METHOD, "strategy": "response",
        "budget": 1920, "scope": "new to cited panel only; broader historical provenance unknown"})
    print("Sealing V5 plans on 18 prospectively specified development tasks.", flush=True)
    rows = []
    for root, task in enumerate(tasks()):
        meter, trace, goal = Meter(1920), [], np.asarray(task["goal"])

        def predict(actions):
            requested = np.asarray(actions, dtype=float)
            raw = requested.astype(np.float32)

            def evaluate(batch):
                positions = exact.endpoints(task, batch)
                return positions, np.linalg.norm(positions-goal, axis=1)

            positions, scores = meter.predict(raw, evaluate)
            trace.append({"requested": requested.tolist(), "raw": raw.tolist(), "positions": positions.tolist(),
                          "scores": scores.tolist(), "tokens": len(raw)*raw.shape[1]//5})
            return positions, scores

        result = method.decide("response", "response", root, 0, goal, predict, meter)
        rows.append({"root": root, "task": task, "result": result, "counts": meter.as_dict(), "trace": trace})
        print(f"root={root} planned", flush=True)
    seal.verify(deps)
    write(PLANS, {"manifest_sha256": sha(MANIFEST), "rows": rows})


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest, data = json.loads(MANIFEST.read_text()), json.loads(PLANS.read_text())
    seal.verify(manifest["seal"])
    if manifest["tasks"] != tasks() or data["manifest_sha256"] != sha(MANIFEST) or len(data["rows"]) != 18:
        raise ValueError("manifest or panel mismatch")
    samples = 0
    for root, (task, row) in enumerate(zip(tasks(), data["rows"], strict=True)):
        if row["root"] != root or row["task"] != task:
            raise ValueError("root identity mismatch")
        chain = np.asarray(row["result"]["selected"]["chain"])
        selected = row["result"]["selected"]["terminal_score"]
        found, full, tokens, count = False, [], 0, 0
        for batch in row["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), raw)
            positions = exact.endpoints(task, raw)
            scores = np.linalg.norm(positions-np.asarray(task["goal"]), axis=1)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            cost = len(raw)*raw.shape[1]//5
            if batch["tokens"] != cost:
                raise ValueError("token mismatch")
            tokens, count = tokens + cost, count + len(raw)
            if raw.shape[1] == 15:
                full.extend(scores.tolist())
                found |= any(scores[j] == selected for j in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
        if (not found or min(full) != selected or tokens > 1920
                or tokens != row["counts"]["prediction_token_samples"] or count != row["counts"]["trajectory_samples"]
                or len(row["trace"]) != row["counts"]["prediction_calls"]):
            raise ValueError("selected chain, budget, or trace mismatch")
        samples += count
    seal.verify(manifest["seal"])
    write(AUDIT, {"plans_sha256": sha(PLANS), "verified_roots": 18, "trajectory_samples": samples,
                  "scope": "same exact backend plan trace recomputation"})
    print(f"Audited 18 selected chains and {samples} trajectories.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
