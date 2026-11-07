"""Independent CPU replay of sealed TwoRoom directional diagnostic actions."""
import inspect
import json
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_tworoom_rgb_directional_alignment_20261107_codex as source

OUTPUT = source.source.ROOT / "results/pc_tworoom_directional_replay_contract_20261107_codex.json"


def validate_window(episodes, steps):
    episodes, steps = np.asarray(episodes), np.asarray(steps)
    if episodes.ndim != 1 or len(episodes) < 2 or not np.all(episodes == episodes[0]):
        raise ValueError("episode discontinuity")
    if steps.shape != episodes.shape or not np.array_equal(steps, np.arange(steps[0], steps[0] + len(steps))):
        raise ValueError("clock discontinuity")


def compare_trace(saved, replay):
    if not saved or len(saved) != len(replay):
        raise ValueError("trace length mismatch")
    error = []
    for original, observed in zip(saved, replay, strict=True):
        for record in (original, observed):
            distance = float(record["distance"])
            if not np.isfinite(distance) or bool(record["terminated"]) != (distance < 16.0):
                raise ValueError("success threshold mismatch")
            if record["truncated"]:
                raise ValueError("unexpected truncation")
        if original["terminated"] != observed["terminated"]:
            raise ValueError("replayed success mismatch")
        error.append(abs(float(original["distance"]) - float(observed["distance"])))
    ever = any(item["terminated"] for item in replay)
    final = replay[-1]["terminated"]
    return {"ever_success": ever, "final_success": final, "lost_success": ever and not final,
            "max_distance_error": max(error)}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    physical = json.loads(source.PHYSICAL.read_text())
    if (physical["plan_sha256"] != source.file_hash(source.PLAN)
            or plan["input_sha256"] != source.file_hash(source.source.inputs.INPUT)
            or plan["protocol_sha256"] != source.file_hash(source.PROTOCOL)
            or plan["runner_sha256"] != source.file_hash(source.__file__)):
        raise ValueError("sealed artifact drift")
    actions = np.asarray(plan["raw_actions"], dtype=np.float32)
    expected = np.stack([source.directions()[d] * h * sign for d, h, sign in plan["configs"]])
    np.testing.assert_array_equal(actions, expected)
    if actions.shape != (24, 25, 2) or np.abs(actions).max() > 1:
        raise ValueError("illegal actions")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    results = []
    try:
        actor = env.unwrapped
        env_path = Path(inspect.getfile(type(actor)))
        with h5py.File(source.source.common.DATA, "r") as dataset, \
                np.load(source.source.inputs.INPUT, allow_pickle=False) as inputs:
            for predicted, stored in zip(plan["rows"], physical["rows"], strict=True):
                row = int(predicted["row"])
                if (stored["row"], stored["root"]) != (row, predicted["root"]):
                    raise ValueError("task identity mismatch")
                validate_window(dataset["ep_idx"][row:row + 26], dataset["step_idx"][row:row + 26])
                selected = np.flatnonzero(inputs["rows"] == row)
                if len(selected) != 1:
                    raise ValueError("input identity mismatch")
                idx = selected.item()
                np.testing.assert_array_equal(inputs["current"][idx], dataset["pixels"][row])
                np.testing.assert_array_equal(inputs["goal"][idx], dataset["pixels"][row + 25])
                indices = [i for i, (_, scale, _) in enumerate(plan["configs"]) if scale in (.02, .2)]
                if stored["indices"] != indices or len(stored["traces"]) != len(indices):
                    raise ValueError("candidate coverage mismatch")
                candidates = []
                for index, old_trace in zip(indices, stored["traces"], strict=True):
                    env.reset(seed=42)
                    actor._set_state(dataset["proprio"][row])
                    actor._set_goal_state(dataset["proprio"][row + 25])
                    replay = []
                    for action in actions[index]:
                        _, _, terminated, truncated, info = actor.step(action.copy())
                        replay.append({"distance": float(info["distance_to_target"]),
                                       "terminated": bool(terminated), "truncated": bool(truncated)})
                    candidates.append({"index": index, **compare_trace(old_trace, replay)})
                results.append({"root": predicted["root"], "row": row, "candidates": candidates})
                print(f"root={predicted['root']} replayed {len(candidates)} chains", flush=True)
    finally:
        env.close()
    stat = source.source.common.DATA.stat()
    result = {"rows": results, "plan_sha256": source.file_hash(source.PLAN),
              "physical_sha256": source.file_hash(source.PHYSICAL), "environment_sha256": source.file_hash(env_path),
              "dataset_size": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns,
              "runner_sha256": source.file_hash(Path(__file__)),
              "scope": "same local implementation replay, exposed tasks; not independent physics or model validation"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    candidates = [c for row in results for c in row["candidates"]]
    print(json.dumps({"chains": len(candidates), "ever": sum(c["ever_success"] for c in candidates),
                      "final": sum(c["final_success"] for c in candidates),
                      "lost": sum(c["lost_success"] for c in candidates),
                      "max_distance_error": max(c["max_distance_error"] for c in candidates)}))


if __name__ == "__main__":
    main()
