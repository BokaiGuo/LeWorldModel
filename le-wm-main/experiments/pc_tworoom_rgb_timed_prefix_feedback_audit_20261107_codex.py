"""Reconstruct saved decisions and replay RGB/physical traces."""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from experiments import pc_tworoom_rgb_timed_prefix_feedback_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, tensor_receipt

OUT = run.ROOT / "results/pc_tworoom_rgb_timed_prefix_feedback_audit_20261107_codex.json"


def reconstruct(item, old_tail, strategy):
    remaining = item["remaining_steps"]
    if remaining == 5:
        parents, _, _ = run.timed.build(item["candidate_seed"], old_tail, 5, count=13)
        if old_tail is None:
            parents[0] = 0
        costs = np.asarray(item["parent_cost"])
        if costs.shape != (13,) or not np.isfinite(costs).all() or item["mode"] != "direct_fallback":
            raise ValueError("invalid fallback")
        index = int(np.argmin(costs))
        if (item["choice"] != {"source": "fallback", "index": index, "cost": float(costs[index])}
                or item["retained"] or item["prefix_steps"] is not None):
            raise ValueError("fallback choice mismatch")
        action = parents[index]
        np.testing.assert_array_equal(action, np.asarray(item["actions"], dtype=np.float32))
        if run.prior.action_hash(action) != item["actions_sha256"]:
            raise ValueError("fallback hash mismatch")
        return action
    prefix = min(10, remaining-5)
    if item["mode"] != "program" or item["prefix_steps"] != prefix:
        raise ValueError("invalid prefix clock")
    parents, probes, tails = run.timed.build(item["candidate_seed"], old_tail, prefix)
    scores = [np.asarray(item["parent_cost"]), np.asarray(item["probe_cost"]).reshape(-1),
              np.asarray(item["tail_cost"])]
    if [x.shape for x in scores] != [(3,), (6,), (4,)] or not all(np.isfinite(x).all() for x in scores):
        raise ValueError("invalid saved scores")
    cursor = iter(scores)

    def score(batch):
        value = next(cursor)
        if len(batch) != len(value):
            raise ValueError("query length mismatch")
        return value

    result = run.prior.shared.search(score, parents, probes, tails, prefix_steps=prefix,
                                     retain_count=2, strategy=strategy, budget=13, seed=item["strategy_seed"])
    choice = item["choice"]
    if (result["retained_prefixes"] != item["retained"] or result["model_queries"] != item["queries"]
            or result["choice"].source != choice["source"] or result["choice"].index != choice["index"]
            or result["choice"].cost != choice["cost"]):
        raise ValueError("stored decision mismatch")
    np.testing.assert_array_equal(result["actions"], np.asarray(item["actions"], dtype=np.float32))
    if run.prior.action_hash(result["actions"]) != item["actions_sha256"]:
        raise ValueError("action hash mismatch")
    return result["actions"]


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from eval import img_transform

    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(run.OUT.read_text())
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("shared_sha256", run.prior.shared.__file__), ("adapter_sha256", run.timed.__file__),
                      ("source_plan_sha256", run.prior.source.PLAN)):
        if data[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows = []
    try:
        with (np.load(run.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.prior.source.block.prior.common.DATA, "r") as db):
            for root_row in data["rows"]:
                root, task = root_row["root"], root_row["row"]
                if task != int(inputs["rows"][root]):
                    raise ValueError("task mismatch")
                for arm in run.ARMS:
                    row = root_row["arms"][arm]
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(db["proprio"][task])
                    actor._set_goal_state(db["proprio"][task + 25])
                    goal = transform(inputs["goal"][root])[None, None, None]
                    old_tail, step, max_error = None, 0, 0.
                    for i, item in enumerate(row["decisions"]):
                        if (item["iteration"] != i or item["candidate_seed"] != 27121000 + root * 10 + i
                                or item["strategy_seed"] != 27122000 + root * 10 + i
                                or item["remaining_steps"] != 25 - step
                                or item["model_packet_steps"] != 13 * (25 - step) // 5
                                or item["old_tail_sha256"] != (None if old_tail is None else run.prior.action_hash(old_tail))):
                            raise ValueError("iteration or old tail mismatch")
                        observed = transform(actor.render().copy())[None, None, None]
                        for receipt in item["receipts"]:
                            if (receipt["track_id"] != RGB or receipt["inputs"]["pixels"] != tensor_receipt(observed)
                                    or receipt["inputs"]["goal"] != tensor_receipt(goal)
                                    or receipt["inputs"]["candidate_actions"]["shape"][2] != (25-step)//5
                                    or set(receipt["inputs"]) != {"pixels", "goal", "action", "candidate_actions"}):
                                raise ValueError("RGB input receipt mismatch")
                        chosen = reconstruct(item, old_tail, arm)
                        dispatched = np.asarray(item["dispatched"], dtype=np.float32)
                        np.testing.assert_array_equal(dispatched, chosen[:len(dispatched)])
                        for action in dispatched:
                            _, _, term, trunc, info = actor.step(action.copy())
                            expected = row["trace"][step]
                            np.testing.assert_array_equal(action, np.asarray(expected["action"], dtype=np.float32))
                            if (bool(term), bool(trunc)) != (expected["terminated"], expected["truncated"]):
                                raise ValueError("termination mismatch")
                            max_error = max(max_error, abs(float(info["distance_to_target"]) - expected["distance"]))
                            step += 1
                        old_tail = chosen[len(dispatched):].copy()
                    if step != len(row["trace"]) or max_error > 1e-6:
                        raise ValueError("trajectory mismatch")
                    rows.append({"root": root, "arm": arm, "decisions": len(row["decisions"]),
                                 "steps": step, "max_distance_error": max_error})
                print(f"audited root={root}", flush=True)
    finally:
        env.close()
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(run.OUT), "runner_sha256": file_hash(__file__), "rows": rows,
                   "scope": "saved-score reconstruction and same-backend replay; no GPU rescoring"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
