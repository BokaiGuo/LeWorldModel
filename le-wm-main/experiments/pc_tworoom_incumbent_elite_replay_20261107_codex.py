"""Reconstruct actions and replay both elite-update arms in the same simulator."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_tworoom_incumbent_elite_fix_20261107_codex as run
from experiments.pc_input_permissions import file_hash

OUTPUT = run.ROOT / "results/pc_tworoom_incumbent_elite_replay_20261107_codex.json"
PROTOCOL = run.ROOT.parent / "docs/PC_TWOROOM_INCUMBENT_ELITE_REPLAY_PROTOCOL_20261107_CODEX.md"


def validate_arm(arm):
    trace = arm["trace"]
    decisions = arm["decisions"]
    if not 1 <= len(trace) <= 25 or not 1 <= len(decisions) <= 5:
        raise ValueError("invalid trace or decision count")
    cursor = 0
    for index, decision in enumerate(decisions):
        actions = np.asarray(decision["actions"], dtype=np.float32)
        scores = np.asarray(decision["all_scores"], dtype=np.float64)
        dispatched = decision["dispatched"]
        if (decision["iteration"] != index or decision["remaining"] != 25 - cursor
                or actions.shape != (25 - cursor, 2) or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > 1) or dispatched < 1 or dispatched > 5
                or scores.shape != (680,) or not np.isfinite(scores).all()
                or decision["queries"] != 680 or sum(decision["query_batches"]) != 680
                or decision["winner"] != int(np.argmin(scores))
                or decision["score"] != float(scores[decision["winner"]])):
            raise ValueError("invalid action, timing, query accounting or final selection")
        if hashlib.sha256(actions.tobytes()).hexdigest() != decision["actions_sha256"]:
            raise ValueError("action fingerprint mismatch")
        actual = np.asarray([s["action"] for s in trace[cursor:cursor + dispatched]], dtype=np.float32)
        if not np.array_equal(actual, actions[:dispatched]):
            raise ValueError("dispatched action mismatch")
        if bool(decision["old_tail_included"]) != (index > 0):
            raise ValueError("old tail flag mismatch")
        if index:
            previous = decisions[index - 1]
            tail = np.asarray(previous["actions"], dtype=np.float32)[previous["dispatched"]:]
            if tail.shape != actions.shape or previous["dispatched"] != 5:
                raise ValueError("old tail clock mismatch")
            rank = int(np.flatnonzero(np.argsort(scores[:100], kind="stable") == 0)[0]) + 1
            if rank != decision["old_tail_rank"]:
                raise ValueError("old tail ranking mismatch")
            if decision["winner"] == 0 and not np.array_equal(actions, tail):
                raise ValueError("selected old tail identity mismatch")
        cursor += dispatched
    if cursor != len(trace) or arm["queries"] != 680 * len(decisions):
        raise ValueError("trace or total query count mismatch")
    for index, step in enumerate(trace):
        if bool(step["terminated"]) != (float(step["distance"]) < 16.0):
            raise ValueError("recorded success criterion mismatch")
        if step["truncated"] or (step["terminated"] and index != len(trace) - 1):
            raise ValueError("invalid stopping")
    if arm["ever_success"] != any(x["terminated"] for x in trace):
        raise ValueError("success summary mismatch")


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    for name, expected in result["source_sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"source fingerprint drift: {name}")
    if result["track"] != run.RGB or [r["root"] for r in result["rows"]] != list(range(10, 50)):
        raise ValueError("track or panel mismatch")
    source = run.prior.prior.prior.source
    input_path = source.block.prior.inputs.INPUT
    data_path = source.block.prior.common.DATA
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    env_source = Path(__import__("inspect").getfile(type(env.unwrapped)))
    state_hash = hashlib.sha256()
    rows = []
    max_distance_difference = 0.0
    try:
        with np.load(input_path, allow_pickle=False) as inputs, h5py.File(data_path, "r") as data:
            for row in result["rows"]:
                root = row["root"]
                offset = int(inputs["rows"][root])
                if offset != row["row"]:
                    raise ValueError("root row mismatch")
                initial = np.asarray(data["proprio"][offset])
                goal = np.asarray(data["proprio"][offset + 25])
                state_hash.update(np.ascontiguousarray(initial).tobytes())
                state_hash.update(np.ascontiguousarray(goal).tobytes())
                initial_distance = float(np.linalg.norm(initial - goal))
                details = {}
                for name in ("original", "corrected"):
                    arm = row["arms"][name]
                    validate_arm(arm)
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(initial)
                    actor._set_goal_state(goal)
                    image_hash = hashlib.sha256(actor.render().tobytes()).hexdigest()
                    if image_hash != arm["initial_rgb_sha256"]:
                        raise ValueError("initial render mismatch")
                    for record in arm["trace"]:
                        _, _, term, trunc, info = actor.step(np.asarray(record["action"], dtype=np.float32))
                        delta = abs(float(info["distance_to_target"]) - float(record["distance"]))
                        max_distance_difference = max(max_distance_difference, delta)
                        if delta > 1e-5 or bool(term) != record["terminated"] or bool(trunc) != record["truncated"]:
                            raise ValueError("physical replay mismatch")
                    details[name] = {"steps": len(arm["trace"]), "success": arm["ever_success"],
                                     "final_distance": arm["trace"][-1]["distance"]}
                rows.append({"root": root, "row": offset, "initial_distance": initial_distance,
                             "initially_successful": initial_distance < 16.0, "arms": details})
    finally:
        env.close()
    summary = {name: {"success": sum(r["arms"][name]["success"] for r in rows),
                      "steps": sum(r["arms"][name]["steps"] for r in rows),
                      "success_initially_unsuccessful": sum(r["arms"][name]["success"]
                                                            for r in rows if not r["initially_successful"])}
               for name in ("original", "corrected")}
    payload = {"scope": "same-engine replay on previously exposed roots, not independent confirmation",
               "input_sha256": {str(p): file_hash(p) for p in [run.OUTPUT, input_path, env_source,
                                                                PROTOCOL, Path(__file__)]},
               "dataset_path": str(data_path), "dataset_whole_file_hashed": False,
               "initial_and_goal_state_sha256": state_hash.hexdigest(),
               "max_distance_difference": max_distance_difference, "summary": summary, "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({"summary": summary, "max_distance_difference": max_distance_difference}))


if __name__ == "__main__":
    main()
