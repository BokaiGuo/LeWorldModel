"""Post-hoc verify native success semantics against object-only diagnostics."""
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/pc_osfal_pusht_heldout_geometry_20261107_codex"
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
MODEL = PREFIX.with_name(PREFIX.name + "_model.json")
REPLAY = PREFIX.with_name(PREFIX.name + "_replay.json")
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_NATIVE_SUCCESS_METRIC_AUDIT_POSTHOC_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_pusht_native_success_metric_audit_20261107_codex.json"
ACTIONS = ("zero", "plus", "minus")


def metrics(goal, state):
    goal, state = np.asarray(goal, np.float64), np.asarray(state, np.float64)
    if goal.shape != (7,) or state.shape != (7,) or not np.isfinite([goal, state]).all():
        raise ValueError("invalid state")
    position = float(np.linalg.norm(goal[:4] - state[:4]))
    raw_angle = abs(float(goal[4] - state[4]))
    native_angle = min(raw_angle, 2 * np.pi - raw_angle)
    wrapped_angle = abs(float(np.arctan2(np.sin(goal[4] - state[4]), np.cos(goal[4] - state[4]))))
    object_position = float(np.linalg.norm(goal[2:4] - state[2:4]))
    return {"native_success": bool(position < 20 and native_angle < np.pi / 9),
            "native_state_distance": float(np.linalg.norm(goal - state)),
            "joint_position_error": position, "wrapped_angle_error": wrapped_angle,
            "agent_position_error": float(np.linalg.norm(goal[:2] - state[:2])),
            "agent_velocity_error": float(np.linalg.norm(goal[5:] - state[5:])),
            "object_pose_cost": (object_position / 20) ** 2 + (wrapped_angle / (np.pi / 9)) ** 2,
            "success_component_proxy": (position / 20) ** 2 + (wrapped_angle / (np.pi / 9)) ** 2}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    physical, model, replay = [json.loads(path.read_text()) for path in (PHYSICAL, MODEL, REPLAY)]
    if model["physical_sha256"] != file_hash(PHYSICAL) or replay["physical_sha256"] != file_hash(PHYSICAL):
        raise ValueError("physical hash mismatch")
    if replay["model_sha256"] != file_hash(MODEL) or replay["valid_roots"] != 16:
        raise ValueError("model or physical replay mismatch")
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    env_source = Path(inspect.getfile(type(actor)))
    rows = []
    try:
        for truth, pred in zip(physical["rows"], model["rows"], strict=True):
            if truth["root"] != pred["root"] or not truth["valid"] or not pred["valid"]:
                raise ValueError("invalid roots")
            conditions = []
            for tc, mc in zip(truth["conditions"], pred["conditions"], strict=True):
                if (tc["appearance"], tc["side"]) != (mc["appearance"], mc["side"]):
                    raise ValueError("condition mismatch")
                outcomes = []
                for action in ACTIONS:
                    trace = tc["actions"][action]["trace"][-1]
                    state, goal = np.asarray(trace["state"]), np.asarray(truth["goal"])
                    item = metrics(goal, state)
                    expected_success, expected_distance = actor.eval_state(goal, state)
                    if item["native_success"] != bool(expected_success) or item["native_success"] != trace["success"]:
                        raise ValueError("native success mismatch")
                    np.testing.assert_allclose(item["native_state_distance"], expected_distance, atol=1e-10, rtol=0)
                    np.testing.assert_allclose(item["object_pose_cost"],
                                               tc["actions"][action]["object_pose_cost"], atol=1e-9, rtol=0)
                    item.update(action=action, predicted_cost=mc["actions"][action]["predicted_cost"],
                                encoded_cost=mc["actions"][action]["real_encoded_cost"])
                    outcomes.append(item)
                picks = {"model": int(np.argmin([x["predicted_cost"] for x in outcomes])),
                         "encoded": int(np.argmin([x["encoded_cost"] for x in outcomes])), "zero": 0}
                regrets = {}
                for metric in ("object_pose_cost", "native_state_distance", "success_component_proxy"):
                    values = np.array([x[metric] for x in outcomes])
                    regrets[metric] = {name: float(values[index] - values.min()) for name, index in picks.items()}
                conditions.append({"appearance": tc["appearance"], "side": tc["side"],
                                   "outcomes": outcomes, "selected": {k: ACTIONS[v] for k, v in picks.items()},
                                   "regrets": regrets})
            rows.append({"root": truth["root"], "conditions": conditions})
    finally:
        env.close()
    all_conditions = [condition for row in rows for condition in row["conditions"]]
    outcome_count = sum(len(condition["outcomes"]) for condition in all_conditions)
    summary = {"roots": len(rows), "conditions": len(all_conditions), "outcomes": outcome_count,
               "native_successes": sum(x["native_success"] for c in all_conditions for x in c["outcomes"]),
               "mean_regrets": {metric: {name: float(np.mean([c["regrets"][metric][name] for c in all_conditions]))
                                         for name in ("model", "encoded", "zero")}
                                for metric in ("object_pose_cost", "native_state_distance", "success_component_proxy")},
               "mean_agent_position_error": float(np.mean([x["agent_position_error"] for c in all_conditions
                                                            for x in c["outcomes"]]))}
    record = {"scope": "post-hoc development metric semantics, not planner inputs or gate evidence",
              "summary": summary, "rows": rows,
              "sources": {str(p): file_hash(p) for p in (PHYSICAL, MODEL, REPLAY, PROTOCOL, env_source, Path(__file__))}}
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
