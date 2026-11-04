"""Reproduce the sealed V5 E0 panel with two-dimensional V6."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as source
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_V6_ACTION_DIM_GENERALIZATION_DEV_PROTOCOL_20261104_CODEX.md"
OUT = source.ROOT / "results/pc_v6_action_dim_v5_e0_equivalence_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads(source.PLANS.read_text())
    if json.loads(source.AUDIT.read_text())["plans_sha256"] != source.sha(source.PLANS):
        raise ValueError("prior audit mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, source.PLANS, source.AUDIT, Path(v6.__file__),
                         Path(core.__file__), Path(source.exact.__file__), Path(source.exact.wall.__file__)])
    rows, sample_total = [], 0
    print("Checking V6 2D predictions and selections against sealed V5 E0 traces.", flush=True)
    for original in prior["rows"]:
        task, root = original["task"], original["root"]
        goal, meter, cursor, samples = np.asarray(task["goal"]), Meter(1920), 0, 0

        def predict(actions):
            nonlocal cursor, samples
            requested = np.asarray(actions, dtype=float)
            raw = requested.astype(np.float32)
            old = original["trace"][cursor]
            np.testing.assert_array_equal(requested, np.asarray(old["requested"]))
            np.testing.assert_array_equal(raw, np.asarray(old["raw"], np.float32))

            def evaluate(batch):
                positions = source.exact.endpoints(task, batch)
                return positions, np.linalg.norm(positions-goal, axis=1)

            positions, scores = meter.predict(raw, evaluate)
            np.testing.assert_allclose(positions, old["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, old["scores"], rtol=0, atol=1e-12)
            samples += len(raw)
            cursor += 1
            return positions, scores

        result = v6.decide("response", "response", root, 0, goal, predict, meter, core.NOMINALS)
        saved = original["result"]
        for key in saved:
            if key != "structure" and result[key] != saved[key]:
                raise ValueError(f"result mismatch root={root}, key={key}")
        for key in ("svd", "fd_action_samples"):
            if result["structure"][key] != saved["structure"][key]:
                raise ValueError("structure mismatch")
        if cursor != len(original["trace"]) or meter.token_samples != original["counts"]["prediction_token_samples"]:
            raise ValueError("trace or budget mismatch")
        sample_total += samples
        rows.append({"root": root, "batches": cursor, "samples": samples,
                     "tokens": meter.token_samples, "selected_score": result["selected"]["terminal_score"]})
    seal.verify(deps)
    source.write(OUT, {"dependencies": deps, "v5_plans_sha256": source.sha(source.PLANS),
                       "rows": rows, "trajectory_samples": sample_total,
                       "scope": "same exact backend and previously exposed 2D tasks; not Cube evidence"})
    print(f"Reproduced {len(rows)} plans and {sample_total} trajectories.", flush=True)


if __name__ == "__main__":
    main()
