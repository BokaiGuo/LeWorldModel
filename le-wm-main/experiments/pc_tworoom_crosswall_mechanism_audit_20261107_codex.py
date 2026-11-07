"""Replay cross-wall witnesses and executed plans; add privileged direct-goal diagnostic."""
import json

import numpy as np

from experiments import pc_tworoom_crosswall_mechanism_dev_20261107_codex as source
from experiments.pc_input_permissions import file_hash

OUT = source.ROOT / "results/pc_tworoom_crosswall_mechanism_audit_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    reference, result = json.loads(source.REFERENCE.read_text()), json.loads(source.RESULT.read_text())
    for key, path in (("public_sha256", source.PUBLIC), ("truth_sha256", source.TRUTH),
                      ("protocol_sha256", source.PROTOCOL), ("runner_sha256", source.__file__)):
        if reference[key] != file_hash(path):
            raise ValueError("reference source drift")
    if result["reference_sha256"] != file_hash(source.REFERENCE) or result["public_sha256"] != file_hash(source.PUBLIC):
        raise ValueError("result source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(source.PUBLIC, allow_pickle=False) as public, np.load(source.TRUTH, allow_pickle=False) as truth:
            for i, (ref, row) in enumerate(zip(reference["rows"], result["rows"], strict=True)):
                start, goal = source.task(i)
                np.testing.assert_array_equal(start, truth["initial"][i])
                np.testing.assert_array_equal(goal, truth["target"][i])
                if ref["seed"] != row["seed"] or ref["seed"] != int(public["seeds"][i]):
                    raise ValueError("seed mismatch")
                for label in ("witness", "16", "300"):
                    env.reset(seed=ref["seed"], options={"state": start.copy(), "target_state": goal.copy()})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    np.testing.assert_array_equal(actor._target_img.cpu().numpy().transpose(1, 2, 0), public["goal"][i])
                    if label == "witness":
                        actions, expected = truth["witness"][i], ref["witness_trace"]
                    else:
                        arm = row["arms"][label]
                        actions, expected = [s["action"] for s in arm["trace"]], arm["trace"]
                        dispatched = [action for decision in arm["decisions"] for action in decision["dispatched"]]
                        np.testing.assert_allclose(dispatched, actions, atol=1e-6, rtol=0)
                        if any(d["queries"] != int(label) for d in arm["decisions"]):
                            raise ValueError("budget mismatch")
                    for action, saved in zip(actions, expected, strict=True):
                        action = np.asarray(action, np.float32)
                        if not np.isfinite(action).all() or np.abs(action).max() > 1:
                            raise ValueError("invalid action")
                        _, _, term, trunc, info = actor.step(action.copy())
                        if (abs(float(info["distance_to_target"]) - saved["distance"]) > 1e-5
                                or bool(term) != saved["terminated"] or bool(trunc) != saved["truncated"]):
                            raise ValueError("physical replay mismatch")
                    if label != "witness" and row["arms"][label]["success"] != any(x["terminated"] for x in expected):
                        raise ValueError("success mismatch")
                env.reset(seed=ref["seed"], options={"state": start.copy(), "target_state": goal.copy()})
                actor = env.unwrapped
                speed = float(actor.variation_space["agent"]["speed"].value.item())
                direct = []
                for _ in range(25):
                    action = np.clip((goal - actor.agent_position.cpu().numpy()) / speed, -1, 1).astype(np.float32)
                    _, _, term, trunc, info = actor.step(action)
                    direct.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                   "terminated": bool(term), "truncated": bool(trunc)})
                rows.append({"seed": ref["seed"], "direct_trace": direct,
                             "direct_success": any(x["terminated"] for x in direct)})
    finally:
        env.close()
    with OUT.open("x") as stream:
        json.dump({"rows": rows, "reference_sha256": file_hash(source.REFERENCE),
                   "result_sha256": file_hash(source.RESULT), "runner_sha256": file_hash(__file__),
                   "scope": "same simulator replay and posthoc privileged direct-goal diagnostic"},
                  stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} tasks; direct_success={sum(x['direct_success'] for x in rows)}", flush=True)


if __name__ == "__main__":
    main()
