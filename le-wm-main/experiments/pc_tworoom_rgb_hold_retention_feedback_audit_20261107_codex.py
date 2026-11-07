"""Reconstruct recorded decisions and replay dispatch in the same backend."""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from experiments import pc_tworoom_rgb_hold_retention_feedback_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, tensor_receipt

OUT = run.ROOT / "results/pc_tworoom_rgb_hold_retention_feedback_audit_20261107_codex.json"


def reconstruct(d, old_tail, arm):
    remaining = d["remaining_steps"]
    if d["queries"] != 16 or d["model_packet_steps"] != 16 * (remaining // 5):
        raise ValueError("query accounting mismatch")
    if remaining == 5:
        parents, _, _ = run.search.timed.build(d["candidate_seed"], old_tail, 5, count=16)
        if old_tail is None:
            parents[0] = 0
        costs = np.asarray(d["parent_cost"])
        index = int(np.argmin(costs))
        if d["mode"] != "direct_fallback" or d["choice"] != {
                "source": "fallback", "index": index, "cost": float(costs[index])}:
            raise ValueError("fallback mismatch")
        return parents[index, :5]
    prefix = min(10, remaining - 5)
    parents, probes, tails = run.search.timed.build(d["candidate_seed"], old_tail, prefix)
    values = iter([np.asarray(d["parent_cost"], dtype=np.float32),
                   np.asarray(d["probe_cost"], dtype=np.float32).ravel(),
                   np.asarray(d["tail_cost"], dtype=np.float32)])
    def score(chains):
        x = next(values)
        if len(x) != len(chains):
            raise ValueError("score batch mismatch")
        return x
    actual = run.search.search(score, parents[:, :remaining], probes[:, :remaining-prefix],
                               tails[:, :remaining-prefix], prefix=prefix, remaining=remaining,
                               strategy=arm, seed=d["strategy_seed"])
    c = actual["choice"]
    if (d["mode"] != "program" or d["prefix_steps"] != prefix
            or d["retained"] != actual["retained_prefixes"]
            or d["choice"] != {"source": c.source, "index": c.index, "cost": c.cost}
            or not np.array_equal(d["effects"], actual["effects"])):
        raise ValueError("decision mismatch")
    return actual["actions"]


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from eval import img_transform

    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(run.OUT.read_text())
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("search_sha256", run.search.__file__), ("source_plan_sha256", run.prior.source.PLAN),
                      ("input_sha256", run.prior.source.block.prior.inputs.INPUT)):
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
                    raise ValueError("root mismatch")
                for arm in run.search.ARMS:
                    row = root_row["arms"][arm]
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(db["proprio"][task])
                    actor._set_goal_state(db["proprio"][task + 25])
                    goal = transform(inputs["goal"][root])[None, None, None]
                    old_tail, step, max_error = None, 0, 0.
                    for i, d in enumerate(row["decisions"]):
                        if (d["iteration"] != i or d["candidate_seed"] != 27121000 + root * 10 + i
                                or d["strategy_seed"] != 27122000 + root * 10 + i
                                or d["remaining_steps"] != 25 - step
                                or d["old_tail_sha256"] != (None if old_tail is None else run.prior.action_hash(old_tail))):
                            raise ValueError("clock/seed/history mismatch")
                        image = transform(actor.render().copy())[None, None, None]
                        for receipt in d["receipts"]:
                            if (receipt["track_id"] != RGB or receipt["inputs"]["pixels"] != tensor_receipt(image)
                                    or receipt["inputs"]["goal"] != tensor_receipt(goal)
                                    or set(receipt["inputs"]) != {"pixels", "goal", "action", "candidate_actions"}):
                                raise ValueError("RGB receipt mismatch")
                        chosen = reconstruct(d, old_tail, arm)
                        np.testing.assert_array_equal(chosen, np.asarray(d["actions"], np.float32))
                        if run.prior.action_hash(chosen) != d["actions_sha256"]:
                            raise ValueError("action identity mismatch")
                        dispatched = np.asarray(d["dispatched"], np.float32)
                        np.testing.assert_array_equal(dispatched, chosen[:len(dispatched)])
                        for action in dispatched:
                            _, _, term, trunc, info = actor.step(action.copy())
                            expected = row["trace"][step]
                            np.testing.assert_array_equal(action, np.asarray(expected["action"], np.float32))
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
        json.dump({"source_sha256": file_hash(run.OUT), "runner_sha256": file_hash(__file__),
                   "rows": rows, "scope": "same-backend physical replay and saved-score reconstruction; no GPU rescore"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
