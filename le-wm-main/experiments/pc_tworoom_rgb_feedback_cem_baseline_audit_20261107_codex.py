"""Audit stored CEM choices and same-backend RGB physical replay."""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from experiments import pc_tworoom_rgb_feedback_cem_baseline_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, tensor_receipt

OUT = run.ROOT / "results/pc_tworoom_rgb_feedback_cem_baseline_audit_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from eval import img_transform

    if OUT.exists():
        raise FileExistsError(OUT)
    record = json.loads(run.OUT.read_text())
    for key, path in (("protocol", run.PROTOCOL), ("runner", run.__file__), ("source", run.prior.OUT), ("pool", run.POOL)):
        if record[key] != file_hash(path):
            raise ValueError(f"drift: {key}")
    _, scaler, _, normalizer = run.prior.prior.source.block.prior.common.model_setup()
    old_plan = json.loads(run.prior.prior.source.PLAN.read_text())
    if normalizer != old_plan["normalizer_sha256"]:
        raise ValueError("normalizer drift")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(run.prior.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              np.load(run.POOL, allow_pickle=False) as pool,
              h5py.File(run.prior.prior.source.block.prior.common.DATA, "r") as data):
            if [r["root"] for r in record["rows"]] != list(range(10, 50)):
                raise ValueError("root mismatch")
            for item in record["rows"]:
                root, task = item["root"], item["row"]
                if task != int(inputs["rows"][root]):
                    raise ValueError("task mismatch")
                for arm in run.ARMS:
                    row = item["arms"][arm]
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task+25])
                    goal = transform(inputs["goal"][root])[None, None, None]
                    old_tail, step, max_error = None, 0, 0.
                    for i, dec in enumerate(row["decisions"]):
                        expected_budget = 13 if arm == "cem13" else 300
                        if (dec["iteration"] != i or dec["queries"] != expected_budget
                                or dec["seed"] != 8140000 + root*10+i
                                or dec["old_tail_sha256"] != (None if old_tail is None else run.prior.prior.action_hash(old_tail))):
                            raise ValueError("clock, seed, or old tail mismatch")
                        key = dec["pool_key"]
                        normal, costs = pool[key+"_normal"], pool[key+"_costs"]
                        if normal.shape != (expected_budget, 25, 2) or costs.shape != (expected_budget,) or not np.isfinite(costs).all():
                            raise ValueError("candidate shape or score mismatch")
                        np.testing.assert_allclose(normal[0], scaler.transform(run.old_or_hold(old_tail)),
                                                   atol=1e-6, rtol=0)
                        winner = int(np.argmin(costs))
                        if winner != dec["chosen"] or float(costs[winner]) != dec["chosen_cost"]:
                            raise ValueError("winner mismatch")
                        chosen = np.clip(scaler.inverse_transform(normal[winner]).astype(np.float32), -1, 1)
                        np.testing.assert_array_equal(chosen, np.asarray(dec["actions"], np.float32))
                        observed = transform(actor.render().copy())[None, None, None]
                        for receipt in dec["receipts"]:
                            if (receipt["track_id"] != RGB or receipt["inputs"]["pixels"] != tensor_receipt(observed)
                                    or receipt["inputs"]["goal"] != tensor_receipt(goal)
                                    or set(receipt["inputs"]) != {"pixels", "goal", "action", "candidate_actions"}):
                                raise ValueError("observation receipt mismatch")
                        dispatched = np.asarray(dec["dispatched"], dtype=np.float32)
                        np.testing.assert_array_equal(dispatched, chosen[:len(dispatched)])
                        for action in dispatched:
                            _, _, term, trunc, info = actor.step(action.copy())
                            expected = row["trace"][step]
                            np.testing.assert_array_equal(action, np.asarray(expected["action"], np.float32))
                            if (bool(term), bool(trunc)) != (expected["terminated"], expected["truncated"]):
                                raise ValueError("termination mismatch")
                            max_error = max(max_error, abs(float(info["distance_to_target"])-expected["distance"]))
                            step += 1
                        old_tail = chosen[len(dispatched):].copy()
                    if step != len(row["trace"]) or max_error > 1e-6:
                        raise ValueError("trajectory mismatch")
                    rows.append({"root": root, "arm": arm, "steps": step, "decisions": len(row["decisions"]),
                                 "max_distance_error": max_error})
                print(f"audited root={root}", flush=True)
    finally:
        env.close()
    with OUT.open("x") as stream:
        json.dump({"source": file_hash(run.OUT), "pool": file_hash(run.POOL), "runner": file_hash(__file__),
                   "rows": rows, "scope": "saved-score selection and same-backend replay; no GPU rescore"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
