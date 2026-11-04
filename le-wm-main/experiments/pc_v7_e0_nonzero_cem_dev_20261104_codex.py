"""Equal-token zero plus random-start full-action CEM on the exposed v7 panel."""

import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_e0_exposed_panel_dev_20261104_codex as panel
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = panel.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_E0_NONZERO_CEM_DEV_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v7_e0_nonzero_cem_dev_20261104_codex.json"
AUDIT = OUT.with_name(OUT.stem + "_audit.json")


def main():
    if OUT.exists() or AUDIT.exists():
        raise FileExistsError("existing output")
    plan_audit = json.loads(panel.AUDIT.read_text())
    if plan_audit["plans_sha256"] != panel.source.sha(panel.OUT):
        raise ValueError("v7 plan audit mismatch")
    plans = json.loads(panel.OUT.read_text())
    deps = seal.capture([Path(__file__), PROTOCOL, panel.OUT, panel.AUDIT, Path(core.__file__),
                         Path(panel.source.exact.__file__), Path(panel.source.exact.wall.__file__)])
    print("Running zero and nonzero-initialized CEM baselines on exposed v7 roots.", flush=True)
    rows = []
    for plan in plans["rows"]:
        root, task = plan["root"], plan["task"]
        goal = np.asarray(task["goal"])
        for arm, knots in (("smooth", 4), ("action", 15)):
            meter, trace = Meter(1920), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)

                def evaluate(batch):
                    positions = panel.source.exact.endpoints(task, batch)
                    return positions, np.linalg.norm(positions-goal, axis=1)

                pos, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                              "positions": pos.tolist(), "scores": scores.tolist(),
                              "tokens": len(raw)*3})
                return pos, scores

            bases = [np.zeros((15, 2))]
            for restart in (1, 2):
                rng = np.random.default_rng(2027112900 + root*100 + restart)
                bases.append(np.clip(core.interpolate(rng.normal(0, .6, size=(1, knots, 2)), 15)[0], -1, 1))
            _, initial = predict(np.asarray(bases))
            best_idx = int(np.argmin(initial))
            best, score, restarts = bases[best_idx].copy(), float(initial[best_idx]), []
            for restart, count in enumerate((213, 212, 212)):
                seed = 2027112800 + root*100 + restart
                chain, value, generations = core.cem(bases[restart], [], count, np.random.default_rng(seed),
                                                    predict, float(initial[restart]), knots)
                restarts.append({"seed": seed, "base": bases[restart].tolist(), "initial_score": float(initial[restart]),
                                 "samples": count, "score": value, "chain": chain.tolist(), "generations": generations})
                if value < score:
                    best, score = chain.copy(), float(value)
            rows.append({"root": root, "task": task, "arm": arm, "selected": best.tolist(), "score": score,
                         "restarts": restarts, "counts": meter.as_dict(), "trace": trace})
        print(f"root={root} complete", flush=True)
    seal.verify(deps)
    panel.source.write(OUT, {"dependencies": deps, "v7_plans_sha256": panel.source.sha(panel.OUT), "rows": rows})
    samples = 0
    for row in rows:
        goal, all_scores, count, tokens = np.asarray(row["task"]["goal"]), [], 0, 0
        found = False
        for batch in row["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), raw)
            pos = panel.source.exact.endpoints(row["task"], raw)
            scores = np.linalg.norm(pos-goal, axis=1)
            np.testing.assert_allclose(pos, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            found |= any(scores[j] == row["score"] for j in np.flatnonzero(
                np.all(requested == np.asarray(row["selected"]), axis=(1, 2))))
            count += len(raw)
            tokens += len(raw)*3
            all_scores.extend(scores.tolist())
        if not found or min(all_scores) != row["score"] or count != 640 or tokens != 1920:
            raise ValueError(f"baseline trace mismatch: {row['root']} {row['arm']}")
        samples += count
    seal.verify(deps)
    panel.source.write(AUDIT, {"result_sha256": panel.source.sha(OUT), "verified_plans": len(rows),
                               "trajectory_samples": samples, "scope": "same exact backend replay"})
    print(f"Audited {len(rows)} plans and {samples} trajectory samples.", flush=True)


if __name__ == "__main__":
    main()
