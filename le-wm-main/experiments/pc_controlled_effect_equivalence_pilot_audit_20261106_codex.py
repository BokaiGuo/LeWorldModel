"""Audit saved controlled-effect records and calculate descriptive pilot metrics."""
import hashlib
import json

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments import pc_controlled_effect_equivalence_pilot_20261106_codex as run

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.OUT.read_text())
    if (data["protocol_sha256"] != hashlib.sha256(run.PROTOCOL.read_bytes()).hexdigest()
            or data["images_sha256"] != hashlib.sha256(run.IMAGES.read_bytes()).hexdigest()
            or len(data["rows"]) != 4):
        raise ValueError("artifact mismatch")
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as file:
        actions = np.asarray(file["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    image_data = np.load(run.IMAGES, allow_pickle=False)
    agreement = {"model_vs_encoding": 0, "model_vs_physical": 0, "encoding_vs_physical": 0}
    contacts, zeros, appearance_diff = 0, 0, []
    for row in data["rows"]:
        task = row["task"]
        if len(row["conditions"]) != 4:
            raise ValueError("missing condition")
        for color in run.COLORS:
            conditions = [c for c in row["conditions"] if c["appearance"] == color]
            for c in conditions:
                prefix = f"t{task}_{color}_{c['side']}"
                for name, result in c["actions"].items():
                    if len(result["trace"]) != 5:
                        raise ValueError("time mismatch")
                    initial, final = image_data[prefix + f"_{name}_initial"], image_data[prefix + f"_{name}_final"]
                    if run.sha_bytes(initial) != result["initial_hash"] or run.sha_bytes(final) != result["final_hash"]:
                        raise ValueError("image identity mismatch")
                    packet = np.asarray(result["packet"]).reshape(5, 2)
                    control = np.asarray(result["physical_action"])
                    np.testing.assert_allclose(scaler.inverse_transform(packet), control, atol=1e-6, rtol=0)
                    if np.abs(control).max() > 1 or not np.all(control == np.asarray(run.CONTROLS[name])):
                        raise ValueError("illegal control")
                    physical = np.linalg.norm(np.asarray(result["trace"][-1]["state"]) - row["goal"])
                    np.testing.assert_allclose(physical, result["physical_state_distance"], atol=1e-12, rtol=0)
                for metric in ("predicted_cost", "actual_encoded_cost", "physical_state_distance"):
                    effect = run.effects({name: result[metric] for name, result in c["actions"].items()})
                    np.testing.assert_allclose(effect, c[metric + "_effect"], atol=1e-12, rtol=0)
                if color == "white":
                    for name, result in c["actions"].items():
                        has_contact = any(step["contacts"] > 0 for step in result["trace"])
                        contacts += has_contact
                        zeros += has_contact and name == "zero"
                    p, e, v = (c[key + "_effect"] for key in
                               ("predicted_cost", "actual_encoded_cost", "physical_state_distance"))
                    ordering = lambda x: np.sign(x[0] - x[1])
                    agreement["model_vs_encoding"] += ordering(p) == ordering(e)
                    agreement["model_vs_physical"] += ordering(p) == ordering(v)
                    agreement["encoding_vs_physical"] += ordering(e) == ordering(v)
        for side in (-1, 1):
            w = next(c for c in row["conditions"] if c["side"] == side and c["appearance"] == "white")
            g = next(c for c in row["conditions"] if c["side"] == side and c["appearance"] == "gray")
            for name in run.CONTROLS:
                if w["actions"][name]["trace"] != g["actions"][name]["trace"]:
                    raise ValueError("appearance physics mismatch")
                if w["actions"][name]["initial_hash"] == g["actions"][name]["initial_hash"]:
                    raise ValueError("appearance did not change pixels")
            appearance_diff.extend(np.abs(np.asarray(w["predicted_cost_effect"]) -
                                          np.asarray(g["predicted_cost_effect"])).tolist())
    report = {"source_sha256": hashlib.sha256(run.OUT.read_bytes()).hexdigest(),
              "tasks": 4, "white_state_conditions": 8, "white_action_trajectories": 24,
              "white_contact_trajectories": int(contacts), "white_zero_control_contacts": int(zeros),
              "plus_minus_order_agreement_counts": {k: int(v) for k, v in agreement.items()},
              "background_effect_abs_mean": float(np.mean(appearance_diff)),
              "background_effect_abs_max": float(np.max(appearance_diff)),
              "caveat": "descriptive only; 8 conditions share 4 tasks; no camera intervention"}
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
