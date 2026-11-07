"""Reconstruct the repaired pools and replay the complete physical candidate traces."""
import json

import numpy as np

from experiments import pc_tworoom_geometric_stage_repair_20261107_codex as study
from experiments.pc_input_permissions import file_hash

OUTPUT = study.ROOT / "results/pc_tworoom_geometric_stage_repair_replay_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(study.OUTPUT.read_text())
    for key, path in (("runner_sha256", study.__file__), ("protocol_sha256", study.PROTOCOL),
                      ("source_sha256", study.SOURCE)):
        if result[key] != file_hash(path):
            raise ValueError(f"result source drift: {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    count = steps = 0
    try:
        with np.load(study.prior.v1.panel.PUBLIC, allow_pickle=False) as public, \
             np.load(study.prior.v1.panel.TRUTH, allow_pickle=False) as truth:
            if len(result["rows"]) != len(public["seeds"]):
                raise ValueError("task count")
            for i, row in enumerate(result["rows"]):
                if row["seed"] != int(public["seeds"][i]):
                    raise ValueError("seed mismatch")
                pool, stages = study.candidate_pool(public["current"][i], public["goal"][i])
                np.testing.assert_allclose(pool, row["pool"], rtol=0, atol=1e-7)
                if stages != row["stages"] or len({x.tobytes() for x in pool}) != row["unique_actions"]:
                    raise ValueError("candidate metadata mismatch")
                if row["chosen"] != int(np.argmin(row["scores"])) or row["queries"] != 16 or row["packet_steps"] != 80:
                    raise ValueError("selection or budget mismatch")
                for j, actions in enumerate(pool):
                    env.reset(seed=row["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    for action, expected in zip(actions, row["chains"][j], strict=True):
                        _, _, term, trunc, info = actor.step(action.copy())
                        np.testing.assert_allclose(info["distance_to_target"], expected["distance"], rtol=0, atol=1e-5)
                        if bool(term) != expected["success"] or bool(trunc) != expected["truncated"]:
                            raise ValueError("physical outcome mismatch")
                        steps += 1
                    count += 1
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"result_sha256": file_hash(study.OUTPUT), "runner_sha256": file_hash(__file__),
                   "tasks": len(result["rows"]), "chains": count, "steps": steps,
                   "scope": "same simulator candidate reconstruction; model scores not recomputed"},
                  stream, indent=2)
    print(f"replayed {count} chains, {steps} physical steps", flush=True)


if __name__ == "__main__":
    main()
