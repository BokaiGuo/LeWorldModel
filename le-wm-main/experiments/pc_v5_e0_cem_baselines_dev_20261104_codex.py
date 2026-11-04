"""Post-hoc equal-prediction-token CEM baselines for the sealed V5 E0 grid."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as source
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_V5_E0_CEM_BASELINES_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = source.ROOT / "results/pc_v5_e0_cem_baselines_dev_20261104_codex"
MANIFEST, RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + x + ".json")
                           for x in ("_manifest", "_result", "_audit")]
ARMS = ("smooth", "action")


def plan():
    if any(p.exists() for p in (MANIFEST, RESULT, AUDIT)):
        raise FileExistsError("prior or partial output")
    if json.loads(source.AUDIT.read_text())["plans_sha256"] != source.sha(source.PLANS):
        raise ValueError("V5 plan audit mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, source.PLANS, source.AUDIT,
                         Path(core.__file__), Path(source.exact.__file__),
                         Path(source.exact.wall.__file__), Path(source.exact.wall.physics.__file__)])
    source.write(MANIFEST, {"dependencies": deps, "tasks": source.tasks(), "arms": ARMS,
                            "scope": "exposed panel, post-hoc baselines"})
    print("Comparing three-restart CEMs on the sealed V5 E0 panel.", flush=True)
    rows = []
    for root, task in enumerate(source.tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            meter, trace = Meter(1920), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)

                def evaluate(batch):
                    pos = source.exact.endpoints(task, batch)
                    return pos, np.linalg.norm(pos-goal, axis=1)

                positions, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                              "positions": positions.tolist(), "scores": scores.tolist(),
                              "tokens": len(raw)*3})
                return positions, scores

            zero = np.zeros((15, 2))
            initial = float(predict(zero[None])[1][0])
            best, best_score, restarts = zero.copy(), initial, []
            for repeat in range(3):
                seed = 2027112600 + root * 100 + repeat
                chain, score, generations = core.cem(zero, [], 213, np.random.default_rng(seed),
                                                     predict, initial, 4 if arm == "smooth" else 15)
                restarts.append({"seed": seed, "score": score, "chain": chain.tolist(),
                                 "generations": generations})
                if score < best_score:
                    best, best_score = chain.copy(), score
            rows.append({"root": root, "arm": arm, "task": task, "selected": best.tolist(),
                         "score": best_score, "restarts": restarts, "trace": trace, "counts": meter.as_dict()})
        print(f"root={root} completed", flush=True)
    seal.verify(deps)
    source.write(RESULT, {"manifest_sha256": source.sha(MANIFEST), "rows": rows})


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest, results = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
    seal.verify(manifest["dependencies"])
    if results["manifest_sha256"] != source.sha(MANIFEST) or len(results["rows"]) != 36:
        raise ValueError("manifest or row count")
    totals = 0
    for index, row in enumerate(results["rows"]):
        root, arm = divmod(index, len(ARMS))
        task = source.tasks()[root]
        if row["root"] != root or row["arm"] != ARMS[arm] or row["task"] != task:
            raise ValueError("task or arm mismatch")
        scores, found, n, tokens = [], False, 0, 0
        selected = np.asarray(row["selected"], dtype=float)
        for batch in row["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), raw)
            positions = source.exact.endpoints(task, raw)
            values = np.linalg.norm(positions-np.asarray(task["goal"]), axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(values, batch["scores"], rtol=0, atol=1e-12)
            found |= any(values[j] == row["score"] for j in np.flatnonzero(
                np.all(requested == selected, axis=(1, 2))))
            scores.extend(values.tolist())
            n += len(raw)
            tokens += len(raw)*3
            if batch["tokens"] != len(raw)*3:
                raise ValueError("batch token mismatch")
        if not found or min(scores) != row["score"] or n != 640 or tokens != 1920:
            raise ValueError("selection or resource mismatch")
        totals += n
    seal.verify(manifest["dependencies"])
    source.write(AUDIT, {"result_sha256": source.sha(RESULT), "verified_plans": len(results["rows"]),
                         "trajectory_samples": totals, "scope": "same exact backend action replay"})
    print(f"Audited {len(results['rows'])} plans and {totals} trajectories.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
