"""Verify generated witness and baseline physical traces from sealed actions."""
import json

import numpy as np

from experiments import pc_tworoom_generated_witness_dev_20261107_codex as source
from experiments.pc_input_permissions import file_hash

OUT = source.ROOT / "results/pc_tworoom_generated_witness_audit_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    reference, result = json.loads(source.REFERENCE.read_text()), json.loads(source.RESULT.read_text())
    for name, path in (("public_sha256", source.PUBLIC), ("truth_sha256", source.TRUTH),
                       ("protocol_sha256", source.PROTOCOL), ("runner_sha256", source.__file__)):
        if reference[name] != file_hash(path):
            raise ValueError(f"reference drift: {name}")
    if result["reference_sha256"] != file_hash(source.REFERENCE) or result["public_sha256"] != file_hash(source.PUBLIC):
        raise ValueError("baseline provenance drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(source.PUBLIC, allow_pickle=False) as public, np.load(source.TRUTH, allow_pickle=False) as truth:
            for i, (old, actual) in enumerate(zip(reference["rows"], result["rows"], strict=True)):
                if old["seed"] != actual["seed"] or int(public["seeds"][i]) != old["seed"]:
                    raise ValueError("task order drift")
                np.testing.assert_array_equal(truth["witness"][i], source.witness_actions(i))
                arm_data = (("witness", truth["witness"][i], old["witness_trace"]),
                            ("cem", [step["action"] for step in actual["trace"]], actual["trace"]))
                for label, actions, expected in arm_data:
                    env.reset(seed=old["seed"])
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.agent_position.cpu().numpy(), truth["initial"][i])
                    actor._set_goal_state(truth["target"][i])
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    np.testing.assert_array_equal(actor._target_img.cpu().numpy().transpose(1, 2, 0), public["goal"][i])
                    for action, original in zip(actions, expected, strict=True):
                        action = np.asarray(action, np.float32)
                        if not np.isfinite(action).all() or np.abs(action).max() > 1:
                            raise ValueError("illegal action")
                        _, _, term, trunc, info = actor.step(action.copy())
                        if (abs(float(info["distance_to_target"]) - original["distance"]) > 1e-5
                                or bool(term) != original["terminated"] or bool(trunc) != original["truncated"]):
                            raise ValueError("physical replay mismatch")
                    if label == "witness":
                        np.testing.assert_array_equal(actor.agent_position.cpu().numpy(), truth["target"][i])
                    else:
                        dispatched = [a for d in actual["decisions"] for a in d["dispatched"]]
                        np.testing.assert_allclose(dispatched, actions, atol=1e-6, rtol=0)
                        if actual["success"] != any(t["terminated"] for t in expected):
                            raise ValueError("success mismatch")
                rows.append({"seed": old["seed"], "witness_steps": len(old["witness_trace"]),
                             "cem_steps": len(actual["trace"])})
    finally:
        env.close()
    with OUT.open("x") as stream:
        json.dump({"rows": rows, "reference_sha256": file_hash(source.REFERENCE),
                   "result_sha256": file_hash(source.RESULT), "runner_sha256": file_hash(__file__),
                   "scope": "same environment physical replay; model scores not recomputed"},
                  stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} reference and CEM physical traces", flush=True)


if __name__ == "__main__":
    main()
