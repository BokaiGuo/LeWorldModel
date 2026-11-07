"""Replay every primitive candidate and reconstruct CEM random draws."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_primitive_cem_control_20261107_codex as run


def independent_decode(start, goal, params):
    start, goal = np.asarray(start), np.asarray(goal)
    d = goal - start[:2]
    length, bearing = np.linalg.norm(d), np.arctan2(d[1], d[0])
    heading = start[2]
    controls = []
    for j in (0, 2):
        new_heading = bearing + params[j] * np.pi / 2
        turn = (new_heading - heading + np.pi) % (2 * np.pi) - np.pi
        nt = int(np.ceil(abs(turn) / (np.pi / 2)))
        if nt:
            controls.extend([(0., turn / nt / (np.pi / 2))] * nt)
        dist = (params[j + 1] + 1) * length
        nd = int(np.ceil(dist / .3))
        if nd:
            controls.extend([(dist / nd / .3, 0.)] * nd)
        heading = new_heading
    return np.array((controls + [(0., 0.)] * 15)[:15]), len(controls)


def main():
    if run.AUDIT.exists():
        raise FileExistsError(run.AUDIT)
    data = json.loads(run.OUTPUT.read_text())
    if data["dependencies"] != {p: run.prior.sha(p) for p in data["dependencies"]}:
        raise ValueError("source dependency drift")
    old = json.loads(run.bridge.OUTPUT.read_text())
    assert len(data["rows"]) == len(old["rows"]) == 24
    records = []
    for row, oldrow in zip(data["rows"], old["rows"], strict=True):
        assert row["root"] == oldrow["root"] and row["task"] == oldrow["task"]
        params, actions, costs = map(np.asarray, (row["parameters"], row["actions"], row["costs"]))
        assert params.shape == (384, 4) and actions.shape == (384, 15, 2)
        assert row["samples"] == 384 and np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1 + 1e-12
        rng = np.random.default_rng(row["seed"])
        begin = 0
        mean = scale = None
        for count in (128, 64, 64, 64, 64):
            expected = (rng.uniform(-1., 1., (count, 4)) if mean is None else
                        np.clip(mean + rng.normal(size=(count, 4)) * scale, -1, 1))
            np.testing.assert_allclose(params[begin:begin + count], expected, atol=0, rtol=0)
            idx = np.argsort(costs[begin:begin + count], kind="stable")[:16]
            elite = expected[idx]
            mean, scale = elite.mean(0), np.maximum(elite.std(0), .08)
            begin += count
        for i, p in enumerate(params):
            action, required = independent_decode(row["task"]["start"], row["task"]["goal"], p)
            np.testing.assert_allclose(actions[i], action, rtol=0, atol=1e-12)
            assert required == row["required_steps"][i]
        padded = np.pad(actions, ((0, 0), (0, 1), (0, 0))) * run.prior.exact.wall.HIGH
        traj, blocked = run.prior.exact.wall.rollout(row["task"]["start"], padded)
        replay = np.linalg.norm(traj[:, 15, :2] - row["task"]["goal"], axis=1)
        np.testing.assert_allclose(costs, replay, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(blocked[:, :15], row["blocked"])
        assert row["best"] == int(np.argmin(costs))
        records.append({"root": row["root"], "primitive_distance": float(costs[row["best"]]),
                        "bridge_distance": float(min(oldrow["costs"])),
                        "primitive_crossed_candidates": int(np.any(traj[:, 1:16, 0] > 0, axis=1).sum()),
                        "primitive_truncated": int((np.asarray(row["required_steps"]) > 15).sum())})
    with run.AUDIT.open("x") as f:
        json.dump({"source_sha256": run.prior.sha(run.OUTPUT), "audit_sha256": run.prior.sha(Path(__file__)),
                   "rows": records}, f, indent=2, allow_nan=False)
    print(f"audited {len(records)} roots and {len(records) * 384} full candidate chains")


if __name__ == "__main__":
    main()
