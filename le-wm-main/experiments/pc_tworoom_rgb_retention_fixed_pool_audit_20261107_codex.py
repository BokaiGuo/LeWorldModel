"""Audit frozen selection arithmetic, candidate provenance, model scores and physical traces."""
import json

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_exact_action_identity_retention_dev_20261106_codex import select
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_tworoom_rgb_retention_fixed_pool_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, physical = json.loads(run.PLAN.read_text()), json.loads(run.PHYSICAL.read_text())
    if (physical["plan_sha256"] != file_hash(run.PLAN) or plan["runner_sha256"] != file_hash(run.__file__)
            or plan["protocol_sha256"] != file_hash(run.PROTOCOL)):
        raise ValueError("sealed source drift")
    model, scaler, transform, action_hash = run.common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or action_hash != plan["normalizer_action_sha256"]:
        raise ValueError("model or normalizer mismatch")
    audit_record = freeze_audit(RGB, "lewm", [run.__file__, run.PROTOCOL, run.inputs.INPUT,
                         run.common.CHECKPOINT / "weights.pt", run.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler", clock="5 packets of 5 steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    max_score_err = max_distance_err = 0.
    try:
        with np.load(run.inputs.INPUT, allow_pickle=False) as inputs, \
                h5py.File(run.common.DATA, "r") as data, torch.inference_mode():
            for index, (row, replay) in enumerate(zip(plan["rows"], physical["rows"], strict=True)):
                if row["row"] != replay["row"] or row["row"] != int(inputs["rows"][index]):
                    raise ValueError("root mismatch")
                pool = np.asarray(row["pool"])
                matrix = np.asarray(row["a_cost"])
                expect = select(matrix, np.asarray(row["parent_cost"]), np.asarray(row["parent_cost"]),
                                2718300 + index)
                if any(row["selected"][arm] != expect[arm] for arm in run.ARMS):
                    raise ValueError("selection mismatch")
                if row["physical_scores"] != 656 or row["counterfactual_deployment_scores"] != 440:
                    raise ValueError("budget mismatch")
                generation = np.asarray(row["generation_actions"])
                cross = np.concatenate((np.broadcast_to(pool[:, None, :10], (8, 8, 10, 2)),
                                        np.broadcast_to(np.asarray(row["a_tails"])[None], (8, 8, 15, 2))), axis=2)
                choices = np.concatenate((generation, pool[4:], cross.reshape(64, 25, 2)))
                values = np.r_[row["generation_costs"], row["parent_cost"][4:], matrix.ravel()]
                j = int(np.argmin(values))
                if j != row["incumbent_index"] or values[j] != row["incumbent_cost"]:
                    raise ValueError("incumbent mismatch")
                for arm in run.ARMS:
                    available = [(values[j], choices[j])]
                    for p in row["selected"][arm]:
                        refs = row["refinements"][p]
                        samples = np.asarray(refs["actions"])
                        if samples.shape != (36, 25, 2) or not np.array_equal(
                                samples[:, :10], np.broadcast_to(pool[p, :10], (36, 10, 2))):
                            raise ValueError("prefix not fixed")
                        k = int(np.argmin(refs["costs"]))
                        available.append((refs["costs"][k], samples[k]))
                    expected_cost, expected_chain = min(available, key=lambda item: item[0])
                    selected = row["outcomes"][arm]
                    if expected_cost != selected["cost"]:
                        raise ValueError("selected score mismatch")
                    np.testing.assert_allclose(run.to_physical(expected_chain, scaler), selected["actions"],
                                               atol=1e-7, rtol=0)
                wrapper = RGBLeWM(model, audit_record)
                pixels = transform(inputs["current"][index])[None, None, None].cuda()
                goal = transform(inputs["goal"][index])[None, None, None].cuda()
                action = np.asarray([row["outcomes"][arm]["actions"] for arm in run.ARMS])
                normalized = scaler.transform(action.reshape(-1, 2)).reshape(1, 4, 5, 10)
                scores, _ = wrapper.score({"pixels": pixels, "goal": goal,
                                            "action": torch.empty((1, 1, 0, 10), device="cuda")},
                                           torch.tensor(normalized, dtype=torch.float32, device="cuda"))
                for arm, value in zip(run.ARMS, scores[0].cpu().numpy(), strict=True):
                    max_score_err = max(max_score_err, abs(float(value) - row["outcomes"][arm]["cost"]))
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    for act, trace in zip(row["outcomes"][arm]["actions"], replay["outcomes"][arm]["trace"],
                                          strict=True):
                        _, _, term, trunc, info = actor.step(np.asarray(act, dtype=np.float32))
                        max_distance_err = max(max_distance_err, abs(float(info["distance_to_target"]) - trace["distance"]))
                        if (bool(term), bool(trunc)) != (trace["terminated"], trace["truncated"]):
                            raise ValueError("physical flag mismatch")
    finally:
        env.close()
    if max_score_err > 1e-3 or max_distance_err > 1e-6:
        raise ValueError("replay mismatch")
    with OUTPUT.open("x") as stream:
        json.dump({"rows": len(plan["rows"]), "max_model_score_error": max_score_err,
                   "max_distance_error": max_distance_err, "plan_sha256": file_hash(run.PLAN),
                   "physical_sha256": file_hash(run.PHYSICAL), "audit_sha256": file_hash(__file__),
                   "scope": "same model and simulator, not independent implementation"}, stream, indent=2)
    print(f"audited {len(plan['rows'])} tasks; model max error {max_score_err}, physical {max_distance_err}")


if __name__ == "__main__":
    audit()
