"""Post-hoc decomposition of saved probe ranking, without new model calls."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_pointmaze_actual_fd_physical_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.PREFIX.with_name(source.PREFIX.name + "_score_attribution.json")


def ordering(a, b):
    i, j = np.triu_indices(len(a), 1)
    x, y = a[i] - a[j], b[i] - b[j]
    valid = (x != 0) & (y != 0)
    return {"pairs": int(valid.sum()), "agree": int(np.sum((np.sign(x) == np.sign(y)) & valid)),
            "fraction": float(np.mean(np.sign(x[valid]) == np.sign(y[valid]))) if valid.any() else None}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = seal.digest
    manifest = json.loads(source.MANIFEST.read_text())
    result = json.loads(source.RESULT.read_text())
    audit_path = source.PREFIX.with_name(source.PREFIX.name + "_audit.json")
    audit = json.loads(audit_path.read_text())
    seal.verify(manifest["seal"])
    if (audit["result_sha256"] != sha(source.RESULT) or audit["arrays_sha256"] != sha(source.ARRAYS)
            or result["manifest_sha256"] != sha(source.MANIFEST)):
        raise ValueError("source not audited or changed")
    decisions, _, _ = source.run.paths("response")
    decision = json.loads(decisions[2].read_text())
    model_goal = np.asarray(decision["binding"]["goal_position"], dtype=np.float64)
    physical_goal = np.asarray([1.8, 2.8], dtype=np.float64)
    rows = []
    with np.load(source.ARRAYS, allow_pickle=False) as archive:
        packets, states = archive["actions"], archive["native_states"]
        for index, packet in enumerate(packets):
            matches = [b for b in decision["trace"]
                       if np.array_equal(np.asarray(b["raw"], np.float32), packet)]
            if len(matches) != 1:
                raise ValueError("probe batch not uniquely bound")
            batch = matches[0]
            predicted = np.asarray(batch["positions"], np.float64)
            true = states[index, :, -1, :2].astype(np.float64)
            score = np.asarray(batch["scores"], np.float64)
            native_distance = np.linalg.norm(true - physical_goal, axis=1)
            predicted_goal_distance = np.linalg.norm(predicted - model_goal, axis=1)
            predicted_native_goal_distance = np.linalg.norm(predicted - physical_goal, axis=1)
            anchored = predicted - predicted[0] + true[0]
            anchored_distance = np.linalg.norm(anchored - physical_goal, axis=1)
            rows.append({"nominal": index, "model_goal": model_goal.tolist(),
                         "physical_goal": physical_goal.tolist(),
                         "model_nominal": predicted[0].tolist(), "native_nominal": true[0].tolist(),
                         "nominal_offset": (predicted[0] - true[0]).tolist(),
                         "goal_offset": (model_goal - physical_goal).tolist(),
                         "score_vs_model_goal_max_abs": float(np.max(np.abs(score-predicted_goal_distance))),
                         "relative_displacement_rms": float(np.sqrt(np.mean((anchored-true)**2))),
                         "ordering": {"score": ordering(score, native_distance),
                                      "predicted_position_with_physical_goal":
                                      ordering(predicted_native_goal_distance, native_distance),
                                      "oracle_anchored_diagnostic": ordering(anchored_distance, native_distance)}})
    seal.verify(manifest["seal"])
    source.run.old.write_json(OUT, {"rows": rows, "source_result_sha256": sha(source.RESULT),
                                   "source_audit_sha256": sha(audit_path),
                                   "script_sha256": sha(Path(__file__)),
                                   "scope": "post-hoc privileged attribution only; not a planner or gate"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
