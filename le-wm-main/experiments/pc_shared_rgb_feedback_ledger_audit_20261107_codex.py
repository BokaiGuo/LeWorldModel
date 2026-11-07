"""Reconstruct each saved selection, then replay actual actions and RGB receipts."""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from experiments import pc_shared_rgb_feedback_ledger_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, tensor_receipt

OUT = run.prior.ROOT / "results/pc_shared_rgb_feedback_ledger_audit_20261107_codex.json"


def reconstruct(item, old_tail):
    parents, probes, tails = run.prior.build_candidates(item["candidate_seed"], old_tail)
    p = np.asarray(item["parent_cost"])
    q = np.asarray(item["probe_cost"])
    t = np.asarray(item["tail_cost"])
    if p.shape != (3,) or q.shape != (3, 2) or t.shape != (2,) or not all(np.isfinite(v).all() for v in (p, q, t)):
        raise ValueError("invalid scored candidate matrix")
    kept = int(np.argmin(q.min(axis=1)))
    if item["retained"] != [kept] or item["queries"] != 11 or item["forward_calls"] != 3:
        raise ValueError("retention or query accounting mismatch")
    scores = np.concatenate([p, q.ravel(), t])
    selected = int(np.argmin(scores))
    if selected < 3:
        source, index, action = "parent", selected, parents[selected]
    elif selected < 9:
        source, index = "probe", selected - 3
        parent, probe = divmod(index, 2)
        action = np.concatenate([parents[parent, :10], probes[probe]])
    else:
        source, index = "tail", selected - 9
        action = np.concatenate([parents[kept, :10], tails[index]])
    saved = item["choice"]
    if saved["source"] != source or saved["index"] != index or saved["cost"] != float(scores[selected]):
        raise ValueError("selected score or identity mismatch")
    np.testing.assert_array_equal(action, np.asarray(item["actions"], dtype=np.float32))
    if run.prior.action_hash(action) != item["actions_sha256"]:
        raise ValueError("selected action hash mismatch")
    return action


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from eval import img_transform

    if OUT.exists():
        raise FileExistsError(OUT)
    payload = json.loads(run.OUT.read_text())
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("prior_sha256", run.prior.OUT), ("shared_sha256", run.prior.shared.__file__),
                      ("adapter_sha256", run.prior.adapter.__file__)):
        if payload[key] != file_hash(path):
            raise ValueError("provenance mismatch")
    previous = json.loads(run.prior.OUT.read_text())
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows = []
    try:
        with (np.load(run.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.prior.source.block.prior.common.DATA, "r") as data):
            for row in payload["rows"]:
                root, task = row["root"], row["row"]
                if task != int(inputs["rows"][root]):
                    raise ValueError("task identity mismatch")
                old = next(x for x in previous["rows"] if x["root"] == root)
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                goal = transform(inputs["goal"][root])[None, None, None]
                old_tail, step, max_error, changed = None, 0, 0., 0
                for iteration, item in enumerate(row["rounds"]):
                    if item["iteration"] != iteration or item["candidate_seed"] != 27121000 + root * 10 + iteration:
                        raise ValueError("random stream mismatch")
                    if item["old_tail_sha256"] != (None if old_tail is None else run.prior.action_hash(old_tail)):
                        raise ValueError("old tail mismatch")
                    observed = transform(actor.render().copy())[None, None, None]
                    for receipt in item["receipts"]:
                        if (receipt["track_id"] != RGB or receipt["inputs"]["pixels"] != tensor_receipt(observed)
                                or receipt["inputs"]["goal"] != tensor_receipt(goal)
                                or set(receipt["inputs"]) != {"pixels", "goal", "action", "candidate_actions"}):
                            raise ValueError("input receipt mismatch")
                    action = reconstruct(item, old_tail)
                    if iteration >= len(old["rounds"]) or not np.array_equal(
                            action, np.asarray(old["rounds"][iteration]["selected_actions"], dtype=np.float32)):
                        changed += 1
                    dispatched = np.asarray(item["dispatched"], dtype=np.float32)
                    np.testing.assert_array_equal(dispatched, action[:len(dispatched)])
                    for value in dispatched:
                        _, _, term, trunc, info = actor.step(value.copy())
                        expected = row["trace"][step]
                        np.testing.assert_array_equal(value, np.asarray(expected["action"], dtype=np.float32))
                        if (bool(term), bool(trunc)) != (expected["terminated"], expected["truncated"]):
                            raise ValueError("physical termination mismatch")
                        max_error = max(max_error, abs(float(info["distance_to_target"]) - expected["distance"]))
                        step += 1
                    old_tail = action[len(dispatched):].copy()
                if step != len(row["trace"]) or max_error > 1e-6:
                    raise ValueError("physical trace mismatch")
                rows.append({"root": root, "rounds": len(row["rounds"]), "steps": step,
                             "changed_choices_vs_prior": changed, "max_distance_error": max_error,
                             "success_matches_prior": row["ever_success"] == old["ever_success"]})
    finally:
        env.close()
    result = {"source_sha256": file_hash(run.OUT), "runner_sha256": file_hash(__file__), "rows": rows,
              "scope": "saved-score reconstruction and same-environment physical replay; no GPU rescore"}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
