"""Replay live-feedback actions and verify old-tail and visual input receipts."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_feedback_smoke_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, tensor_receipt

OUT = run.ROOT / "results/pc_shared_rgb_feedback_smoke_audit_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from eval import img_transform

    if OUT.exists():
        raise FileExistsError(OUT)
    record = json.loads(run.OUT.read_text())
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("shared_sha256", run.shared.__file__), ("adapter_sha256", run.adapter.__file__),
                      ("source_plan_sha256", run.source.PLAN)):
        if record[key] != file_hash(path):
            raise ValueError("source binding mismatch")
    if record["track"] != RGB:
        raise ValueError("wrong track")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, total = [], 0
    try:
        with (np.load(run.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.source.block.prior.common.DATA, "r") as data):
            for row in record["rows"]:
                root, task = row["root"], row["row"]
                if task != int(inputs["rows"][root]):
                    raise ValueError("task mismatch")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                goal = transform(inputs["goal"][root])[None, None, None]
                old_tail, index, max_err = None, 0, 0.
                for iteration, item in enumerate(row["rounds"]):
                    if item["iteration"] != iteration or item["logical_queries"] != 11 or item["forward_calls"] != 3:
                        raise ValueError("clock or query count mismatch")
                    parents, probes, tails = run.build_candidates(27121000 + root * 10 + iteration, old_tail)
                    if run.action_hash(parents[0]) != item["parent0_sha256"]:
                        raise ValueError("parent0 reconstruction mismatch")
                    expected_tail_hash = None if old_tail is None else run.action_hash(old_tail)
                    if item["carried_tail_sha256"] != expected_tail_hash:
                        raise ValueError("old tail mismatch")
                    image = transform(actor.render().copy())[None, None, None]
                    for receipt in item["receipts"]:
                        if (receipt["track_id"] != RGB or receipt["inputs"]["pixels"] != tensor_receipt(image)
                                or receipt["inputs"]["goal"] != tensor_receipt(goal)
                                or set(receipt["inputs"]) != {"pixels", "goal", "action", "candidate_actions"}):
                            raise ValueError("observation or permission receipt mismatch")
                    chosen = np.asarray(item["selected_actions"], dtype=np.float32)
                    if run.action_hash(chosen) != item["selected_sha256"]:
                        raise ValueError("selected action hash mismatch")
                    choice = item["choice"]
                    if choice["source"] == "parent":
                        np.testing.assert_array_equal(chosen, parents[choice["index"]])
                    elif choice["source"] == "probe":
                        p, q = divmod(choice["index"], len(probes))
                        np.testing.assert_array_equal(chosen, np.concatenate((parents[p, :10], probes[q])))
                    elif choice["source"] == "tail":
                        # The saved v1 record lacks the retained prefix identity; verify
                        # membership without claiming reconstruction of the search decision.
                        q = choice["index"] % len(tails)
                        if not any(np.array_equal(chosen, np.concatenate((p[:10], tails[q]))) for p in parents):
                            raise ValueError("selected tail chain not in eligible candidate pool")
                    else:
                        raise ValueError("unknown choice provenance")
                    dispatched = np.asarray(item["dispatched"], dtype=np.float32)
                    np.testing.assert_array_equal(dispatched, chosen[:len(dispatched)])
                    for action in dispatched:
                        _, _, term, trunc, info = actor.step(action.copy())
                        expected = row["trace"][index]
                        np.testing.assert_array_equal(action, np.asarray(expected["action"], dtype=np.float32))
                        if bool(term) != expected["terminated"] or bool(trunc) != expected["truncated"]:
                            raise ValueError("physical termination mismatch")
                        max_err = max(max_err, abs(float(info["distance_to_target"]) - expected["distance"]))
                        index += 1
                    old_tail = chosen[len(dispatched):].copy()
                    total += item["logical_queries"]
                if index != len(row["trace"]) or max_err > 1e-6:
                    raise ValueError("physical trace mismatch")
                rows.append({"root": root, "rounds": len(row["rounds"]), "steps": index,
                             "max_distance_error": max_err,
                             "ever_success": any(x["terminated"] for x in row["trace"])})
    finally:
        env.close()
    if total != record["total_logical_queries"]:
        raise ValueError("total query count mismatch")
    audit = {"source_sha256": file_hash(run.OUT), "runner_sha256": file_hash(__file__),
             "rows": rows, "total_queries": total,
             "scope": "same environment and rendering backend; retained prefix choice not independently re-scored"}
    with OUT.open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
