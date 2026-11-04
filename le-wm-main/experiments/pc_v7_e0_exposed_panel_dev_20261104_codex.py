"""v7 complete search and trace replay on the exposed v5 exact panel."""

import json
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as source
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_V7_E0_EXPOSED_PANEL_DEV_PROTOCOL_20261104_CODEX.md"
OUT = source.ROOT / "results/pc_v7_e0_exposed_panel_dev_20261104_codex.json"
AUDIT = OUT.with_name(OUT.stem + "_audit.json")


def main():
    if OUT.exists() or AUDIT.exists():
        raise FileExistsError("refusing to overwrite results")
    prior = json.loads(source.PLANS.read_text())
    if json.loads(source.AUDIT.read_text())["plans_sha256"] != source.sha(source.PLANS):
        raise ValueError("v5 audit mismatch")
    dependencies = seal.capture([Path(__file__), PROTOCOL, source.PLANS, source.AUDIT,
                                 Path(v6.__file__), Path(v7.__file__), Path(core.__file__),
                                 Path(source.exact.__file__), Path(source.exact.wall.__file__)])
    rows, total_samples = [], 0
    print("Running v7 on the 18 already-exposed exact-dynamics development roots.", flush=True)
    for original in prior["rows"]:
        task, root = original["task"], original["root"]
        goal, meter, trace = np.asarray(task["goal"]), Meter(1920), []

        def predict(actions):
            requested = np.asarray(actions, dtype=float)
            raw = requested.astype(np.float32)

            def evaluate(batch):
                position = source.exact.endpoints(task, batch)
                return position, np.linalg.norm(position-goal, axis=1)

            position, score = meter.predict(raw, evaluate)
            trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                          "positions": position.tolist(), "scores": score.tolist(),
                          "tokens": len(raw)*raw.shape[1]//5})
            return position, score

        result = v7.decide("response", "response", root, 0, goal, predict, meter, core.NOMINALS)
        rows.append({"root": root, "task": task, "result": result, "counts": meter.as_dict(), "trace": trace})
    seal.verify(dependencies)
    source.write(OUT, {"dependencies": dependencies, "v5_plans_sha256": source.sha(source.PLANS),
                       "rows": rows, "scope": "exposed exact-dynamics development panel"})
    for row in rows:
        task, result = row["task"], row["result"]
        tokens, count = 0, 0
        for batch in row["trace"]:
            requested = np.asarray(batch["requested"])
            raw = np.asarray(batch["raw"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), raw)
            positions = source.exact.endpoints(task, raw)
            scores = np.linalg.norm(positions-np.asarray(task["goal"]), axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
            tokens += len(raw)*raw.shape[1]//5
            count += len(raw)
        final = row["trace"][-1]
        selected_idx = 1 if result["abstained"] else 0
        np.testing.assert_array_equal(np.asarray(result["selected"]["chain"], dtype=np.float32),
                                      np.asarray(final["raw"][selected_idx], dtype=np.float32))
        if (result["selected"]["terminal_score"] != final["scores"][selected_idx]
                or tokens != row["counts"]["prediction_token_samples"] or tokens > 1920
                or count != row["counts"]["trajectory_samples"] or len(row["trace"]) != row["counts"]["prediction_calls"]
                or (result["paired_comparison"]["same_actual_float32_actions"] and not result["abstained"])):
            raise ValueError(f"trace or final selection mismatch root {row['root']}")
        total_samples += count
    seal.verify(dependencies)
    source.write(AUDIT, {"plans_sha256": source.sha(OUT), "verified_roots": len(rows),
                         "trajectory_samples": total_samples,
                         "scope": "same exact backend trace replay, no independent physical claim"})
    print(f"Audited {len(rows)} roots and {total_samples} trajectories.", flush=True)


if __name__ == "__main__":
    main()
