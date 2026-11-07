"""Independent physical and candidate reconstruction of the V2 development trace."""
import json

import numpy as np

from experiments import pc_tworoom_rgb_geometric_candidates_v2_20261107_codex as v2
from experiments.pc_input_permissions import file_hash

OUTPUT = v2.v1.ROOT / "results/pc_tworoom_rgb_geometric_v2_replay_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(v2.MANIFEST.read_text())
    for key, path in (("wrapper_sha256", v2.__file__), ("shared_runner_sha256", v2.v1.__file__),
                      ("protocol_sha256", v2.PROTOCOL), ("result_sha256", v2.OUTPUT)):
        if manifest[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    result = json.loads(v2.OUTPUT.read_text())
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    decisions = steps = 0
    try:
        with np.load(v2.v1.panel.PUBLIC, allow_pickle=False) as public, \
             np.load(v2.v1.panel.TRUTH, allow_pickle=False) as truth:
            if len(result["rows"]) != len(public["seeds"]):
                raise ValueError("task count mismatch")
            for i, row in enumerate(result["rows"]):
                if row["seed"] != int(public["seeds"][i]):
                    raise ValueError("seed mismatch")
                for arm in ("model", "geometry"):
                    env.reset(seed=row["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor, tail, trace_index = env.unwrapped, None, 0
                    for decision in row["arms"][arm]["decisions"]:
                        remaining = 25 - trace_index
                        if decision["remaining"] != remaining or decision["queries"] != 16:
                            raise ValueError("budget or clock mismatch")
                        pool, parsed = v2.candidate_pool(actor.render().copy(), public["goal"][i], tail, remaining)
                        np.testing.assert_allclose(pool, decision["pool"], rtol=0, atol=1e-6)
                        if (parsed is None) != (decision["parsed"] is None):
                            raise ValueError("parse mismatch")
                        scores = np.asarray(decision["scores"])
                        expected = int(np.argmin(scores)) if arm == "model" else 1
                        if decision["chosen"] != expected or decision["packet_steps"] != 16 * (remaining // 5):
                            raise ValueError("selection or packet mismatch")
                        actions = pool[expected]
                        for j, saved_action in enumerate(decision["dispatched"]):
                            np.testing.assert_allclose(actions[j], saved_action, rtol=0, atol=1e-6)
                            _, _, terminated, truncated, info = actor.step(actions[j].copy())
                            saved = row["arms"][arm]["trace"][trace_index]
                            np.testing.assert_allclose(info["distance_to_target"], saved["distance"], rtol=0, atol=1e-5)
                            if bool(terminated) != saved["terminated"] or bool(truncated) != saved["truncated"]:
                                raise ValueError("termination mismatch")
                            trace_index += 1
                            steps += 1
                        tail = actions[len(decision["dispatched"]):].copy()
                        decisions += 1
                    if trace_index != len(row["arms"][arm]["trace"]):
                        raise ValueError("trace length mismatch")
                    if row["arms"][arm]["success"] != any(t["terminated"] for t in row["arms"][arm]["trace"]):
                        raise ValueError("success mismatch")
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"result_sha256": file_hash(v2.OUTPUT), "manifest_sha256": file_hash(v2.MANIFEST),
                   "replay_runner_sha256": file_hash(__file__), "tasks": len(result["rows"]),
                   "arms": 2, "decisions": decisions, "physical_steps": steps,
                   "scope": "same-simulator candidate and physical replay; model scores not recomputed"},
                  stream, indent=2)
    print(f"replayed {decisions} candidate pools and {steps} physical steps", flush=True)


if __name__ == "__main__":
    main()
