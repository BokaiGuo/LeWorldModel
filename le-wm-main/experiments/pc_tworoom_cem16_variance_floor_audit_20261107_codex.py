"""Reconstruct sampled actions, updates, and dispatch from sealed CEM16 records."""
import json

import numpy as np

from experiments import pc_tworoom_cem16_variance_floor_20261107_codex as run
from experiments.pc_input_permissions import file_hash


def main():
    result = json.loads(run.OUTPUT.read_text())
    if (result["protocol_sha256"] != file_hash(run.PROTOCOL)
            or result["runner_sha256"] != file_hash(run.__file__)
            or result["baseline_sha256"] != file_hash(run.base.OUTPUT)
            or result["input_sha256"] != file_hash(run.base.INPUT)):
        raise ValueError("artifact drift")
    # The normalizer is needed only for deterministic candidate reconstruction.
    _, scaler, _, normalizer = run.base.common.model_setup()
    if normalizer != result["normalizer_sha256"]:
        raise ValueError("normalizer drift")
    totals = {}
    for label, floor in (("baseline", 1e-5), ("floor", 0.2)):
        total_decisions = total_steps = 0
        for row in result["rows"]:
            arm = row["arms"][label]
            tail = None
            traces = []
            for iteration, decision in enumerate(arm["decisions"]):
                remaining = decision["remaining"]
                if remaining != 25 - len(traces) or decision["queries"] != 16 or decision["packet_steps"] != 16 * remaining // 5:
                    raise ValueError("clock or budget mismatch")
                rng = np.random.default_rng(8140000 + row["episode"] * 10 + iteration)
                zero = scaler.transform(np.zeros((1, 2)))[0]
                mean, std = np.broadcast_to(zero, (5, 2)).copy(), np.ones((5, 2))
                low, high = (-1 - scaler.mean_) / scaler.scale_, (1 - scaler.mean_) / scaler.scale_
                arrays = [np.asarray(scaler.transform(run.base.cem.old_or_hold(tail)), dtype=np.float32)]
                score_arrays = []
                for step in decision["rounds"]:
                    np.testing.assert_allclose(step["mean"], mean, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(step["std"], std, atol=1e-12, rtol=0)
                    sample = run.base.cem.candidates(rng, mean, std, low, high, 5)
                    arrays.extend(np.repeat(sample, 5, axis=1).astype(np.float32))
                    cost = np.asarray(step["costs"])
                    if cost.shape != (5,) or not np.isfinite(cost).all():
                        raise ValueError("invalid scores")
                    mean, std = run.update_elites(sample, cost, floor)
                    score_arrays.extend(cost)
                index = decision["chosen"]
                if not 0 <= index < 16:
                    raise ValueError("invalid selected index")
                actions = np.clip(scaler.inverse_transform(arrays[index]).astype(np.float32)[:remaining], -1, 1)
                np.testing.assert_allclose(actions, decision["actions"], atol=1e-6, rtol=0)
                if index:
                    np.testing.assert_allclose(decision["chosen_cost"], score_arrays[index - 1], atol=1e-7, rtol=0)
                    if any(x < decision["chosen_cost"] for x in score_arrays):
                        raise ValueError("not minimum of sampled scores")
                # Incumbent score is not separately stored; baseline parity and selected cost are checked by runner.
                count = len(decision["dispatched"])
                np.testing.assert_allclose(actions[:count], decision["dispatched"], atol=1e-6, rtol=0)
                traces.extend(decision["dispatched"])
                tail = actions[count:]
                total_decisions += 1
            if len(traces) != len(arm["trace"]):
                raise ValueError("dispatch trace length mismatch")
            for action, point in zip(traces, arm["trace"], strict=True):
                np.testing.assert_allclose(action, point["action"], atol=1e-6, rtol=0)
                if point["terminated"] != (point["distance"] < 16):
                    raise ValueError("success contract mismatch")
            total_steps += len(traces)
        totals[label] = {"decisions": total_decisions, "physical_steps": total_steps}
    if len(result["rows"]) != 40:
        raise ValueError("task count mismatch")
    for scope, rows in (("all", result["rows"]), ("non_initial", [r for r in result["rows"] if not r["initial_success"]])):
        b = np.asarray([r["arms"]["baseline"]["ever_success"] for r in rows])
        f = np.asarray([r["arms"]["floor"]["ever_success"] for r in rows])
        print(scope, "n", len(rows), "baseline", b.sum(), "floor", f.sum(),
              "floor_only", (f & ~b).sum(), "baseline_only", (b & ~f).sum())
        for label in ("baseline", "floor"):
            print(label, "queries", sum(sum(d["queries"] for d in r["arms"][label]["decisions"]) for r in rows),
                  "planning_seconds", sum(sum(d["planning_seconds"] for d in r["arms"][label]["decisions"]) for r in rows),
                  "wall_seconds", sum(r["arms"][label]["wall_seconds"] for r in rows))
    print("reconstructed", totals)


if __name__ == "__main__":
    main()
