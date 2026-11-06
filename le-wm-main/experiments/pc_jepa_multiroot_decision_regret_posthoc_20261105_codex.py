"""Post-hoc decision diagnostics on the sealed four-root PushT action bank."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_jepa_multiroot_action_effect_dev_20261105_codex as source

OUT = source.ROOT / "results/pc_jepa_multiroot_decision_regret_posthoc_20261105_codex"
WRAPPER = source.ROOT / "external/jepa-wms/evals/simu_env_planning/envs/pusht_gym_wrap.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(states, goal):
    """Mirror wrapper and also report robust 2π wrap without replacing its rule."""
    states, goal = np.asarray(states), np.asarray(goal)
    assert states.ndim == 2 and states.shape[1] >= 5 and goal.shape == states.shape[1:]
    pos = np.linalg.norm(states[:, :4] - goal[:4], axis=1)
    raw = np.abs(states[:, 4] - goal[4])
    native_angle = np.minimum(raw, 2 * np.pi - raw)
    wrapped_angle = np.abs((states[:, 4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
    native_success = (pos < 20) & (native_angle < np.pi / 9)
    wrapped_success = (pos < 20) & (wrapped_angle < np.pi / 9)
    # A descriptive joint normalized violation, not the official task metric.
    violation = np.maximum(pos / 20, wrapped_angle / (np.pi / 9))
    return pos, native_angle, wrapped_angle, native_success, wrapped_success, violation


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    receipt = source.verify()
    assert len(receipt["roots"]) == 4
    prior_audit = json.loads((source.OUT / "audit.json").read_text())
    assert prior_audit["status"] == "passed"
    rows = []
    inputs = [source.OUT / "physical.json", source.OUT / "audit.json", WRAPPER, Path(__file__)]
    for i in range(4):
        directory = source.OUT / f"root_{i}"
        latent_path = directory / "output/latents.npz"
        output_receipt = json.loads((directory / "output/receipt.json").read_text())
        assert output_receipt["status"] == "completed"
        assert sha(latent_path) == output_receipt["latents_sha256"]
        inputs.extend((directory / "physical.npz", latent_path, directory / "output/receipt.json"))
        with np.load(latent_path, allow_pickle=False) as data:
            pred, actual, goal = (data[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        with np.load(directory / "physical.npz", allow_pickle=False) as data:
            states, target = data["states"], data["goal_state"]
        assert len(pred) == len(actual) == len(states) == 64
        pos, native_angle, angle, native_ok, wrap_ok, violation = metrics(states, target)
        score_pred = ((pred - goal) ** 2).mean(axis=(1, 2))
        score_actual = ((actual - goal) ** 2).mean(axis=(1, 2))
        indices = {"predicted_score": int(np.argmin(score_pred)),
                   "actual_encoded_score": int(np.argmin(score_actual)),
                   "physical_joint_oracle": int(np.argmin(violation)),
                   "zero_action": 0}
        selections = {}
        for label, ix in indices.items():
            selections[label] = {"index": ix, "position_4d": float(pos[ix]),
                                 "wrapped_angle": float(angle[ix]),
                                 "joint_violation": float(violation[ix]),
                                 "native_success": bool(native_ok[ix]),
                                 "wrapped_success": bool(wrap_ok[ix]),
                                 "predicted_score": float(score_pred[ix]),
                                 "actual_encoded_score": float(score_actual[ix])}
        rows.append({"root": i, "selections": selections,
                     "bank_native_success_count": int(native_ok.sum()),
                     "bank_wrapped_success_count": int(wrap_ok.sum()),
                     "native_wrapped_disagreements": int(np.sum(native_ok != wrap_ok)),
                     "native_negative_angles": int(np.sum(native_angle < 0)),
                     "prediction_choice_vs_encoded_choice_same": indices["predicted_score"] == indices["actual_encoded_score"]})
    output = {"status": "completed", "analysis_type": "posthoc_on_queried_development_bank",
              "inputs_sha256": {str(p): sha(p) for p in inputs}, "rows": rows,
              "limitations": ["not an independent task sample", "not a planner evaluation",
                              "joint violation is diagnostic, not an official task score",
                              "physical oracle is evaluation-only", "all actions queried by model"]}
    OUT.mkdir()
    with (OUT / "report.json").open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
