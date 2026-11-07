"""Replay physical records and summarize frozen effect features by root."""
import json

import numpy as np

from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as run
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    physical, model = json.loads(run.PHYSICAL.read_text()), json.loads(run.MODEL.read_text())
    if (model["physical_sha256"] != run.sha(run.PHYSICAL)
            or model["features_sha256"] != run.sha(run.FEATURES)
            or physical["images_sha256"] != run.sha(run.IMAGES)
            or physical["dependencies"] != {p: run.sha(p) for p in physical["dependencies"]}):
        raise ValueError("sealed artifact drift")
    images, features = np.load(run.IMAGES, allow_pickle=False), np.load(run.FEATURES, allow_pickle=False)
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    rows, old = [], []
    try:
        previous = run.ROOT / "results/pc_controlled_effect_equivalence_pilot_20261106_codex.json"
        pilot = json.loads(previous.read_text())
        for row in pilot["rows"]:
            for c in row["conditions"]:
                if c["appearance"] != "white":
                    continue
                env.reset(seed=771001 + row["task"],
                          options={"state": np.asarray(c["requested"]), "goal_state": np.asarray(row["goal"])})
                old.append({"task": row["task"], "side": c["side"], "geometry": run.geometry(env.unwrapped)})
        for p, m in zip(physical["rows"], model["rows"], strict=True):
            if p["root"] != m["root"]:
                raise ValueError("root mismatch")
            root, conditions = p["root"], []
            for c, mc in zip(p["conditions"], m["conditions"], strict=True):
                if (c["side"], c["appearance"]) != (mc["side"], mc["appearance"]):
                    raise ValueError("condition mismatch")
                color, side = c["appearance"], c["side"]
                costs = {"model": [], "encoded": [], "physical": []}
                contact = []
                for name in ("zero", "plus", "minus"):
                    record, mr = c["actions"][name], mc["actions"][name]
                    env.reset(seed=881001 + root, options={"state": np.asarray(c["requested_state"]),
                                                           "goal_state": np.asarray(p["goal"])})
                    actor = env.unwrapped
                    actor.variation_space["background"]["color"].set_value(np.asarray(run.COLORS[color], np.uint8))
                    np.testing.assert_allclose(snapshot(actor), record["initial_snapshot"], atol=1e-9, rtol=0)
                    if run.geometry(actor)["overlaps"]:
                        raise ValueError("penetrating corrected state")
                    key = f"r{root}_{color}_{side}_{name}"
                    np.testing.assert_array_equal(actor.render(), images[key + "_initial"])
                    for action, step in zip(record["physical_action"], record["trace"], strict=True):
                        obs, _, success, truncated, _ = actor.step(np.asarray(action, np.float32))
                        np.testing.assert_allclose(snapshot(actor), step["snapshot"], atol=1e-9, rtol=0)
                        if bool(success) != step["success"] or truncated:
                            raise ValueError("termination mismatch")
                    np.testing.assert_array_equal(actor.render(), images[key + "_final"])
                    pose = run.pose_cost(obs["state"], p["goal"])
                    np.testing.assert_allclose(pose, record["object_pose_cost"], atol=1e-10, rtol=0)
                    initial, pred, real, goal = features[key]
                    pc, rc = np.square(pred - goal).sum(), np.square(real - goal).sum()
                    np.testing.assert_allclose(pc, mr["predicted_cost"], atol=1e-4, rtol=1e-5)
                    np.testing.assert_allclose(rc, mr["real_encoded_cost"], atol=1e-4, rtol=1e-5)
                    costs["model"].append(float(pc))
                    costs["encoded"].append(float(rc))
                    costs["physical"].append(pose)
                    contact.append(any(step["agent_block_contacts"] > 0 for step in record["trace"]))
                vectors = {k: (np.asarray(v)[1:] - v[0]).tolist() for k, v in costs.items()}
                selected = {k: int(np.argmin(v)) for k, v in costs.items()}
                conditions.append({"appearance": color, "side": side, "effects": vectors, "selected": selected,
                                   "contacts": contact, "model_physical_regret":
                                   float(costs["physical"][selected["model"]] - min(costs["physical"])),
                                   "encoded_physical_regret":
                                   float(costs["physical"][selected["encoded"]] - min(costs["physical"])),
                                   "zero_physical_regret": float(costs["physical"][0] - min(costs["physical"]))})
            summary = {"root": root, "conditions": conditions, "background_effect_mae": {},
                       "state_effect_difference": {}}
            for metric in ("model", "encoded", "physical"):
                by_side = [[c for c in conditions if c["side"] == side] for side in (-1, 1)]
                summary["background_effect_mae"][metric] = float(np.mean([
                    np.abs(np.asarray(w["effects"][metric]) - g["effects"][metric])
                    for w, g in by_side]))
                left, right = [c for c in conditions if c["appearance"] == "white"]
                summary["state_effect_difference"][metric] = (
                    np.asarray(right["effects"][metric]) - left["effects"][metric]).tolist()
            rows.append(summary)
    finally:
        env.close()
    white = [c for row in rows for c in row["conditions"] if c["appearance"] == "white"]
    report = {"physical_sha256": run.sha(run.PHYSICAL), "model_sha256": run.sha(run.MODEL),
              "valid_roots": sum(row["valid"] for row in physical["rows"]), "roots": len(rows),
              "old_pilot_geometry": old, "old_penetrating_conditions": sum(bool(x["geometry"]["overlaps"]) for x in old),
              "white_state_conditions": len(white),
              "white_contact_trajectories": sum(sum(c["contacts"]) for c in white),
              "white_zero_contact_trajectories": sum(c["contacts"][0] for c in white),
              "white_selected_physical_best": sum(c["selected"]["model"] == c["selected"]["physical"] for c in white),
              "mean_model_physical_regret": float(np.mean([c["model_physical_regret"] for c in white])),
              "mean_encoded_physical_regret": float(np.mean([c["encoded_physical_regret"] for c in white])),
              "mean_zero_physical_regret": float(np.mean([c["zero_physical_regret"] for c in white])),
              "background_effect_mae": {metric: float(np.mean([r["background_effect_mae"][metric] for r in rows]))
                                        for metric in ("model", "encoded", "physical")},
              "rows": rows}
    run.save(OUT, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "old_pilot_geometry")}, indent=2))


if __name__ == "__main__":
    main()
