"""Recompute the sealed continuation decisions, action gradients, and replays."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_cem_gradient_continuation_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

OUT = run.ROOT / "results/pc_tworoom_cem_gradient_continuation_audit_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    plan, physical = json.loads(run.PLAN.read_text()), json.loads(run.PHYSICAL.read_text())
    if (physical["plan_sha256"] != file_hash(run.PLAN)
            or plan["source_plan_sha256"] != file_hash(run.source.PLAN)
            or plan["runner_sha256"] != file_hash(run.__file__)
            or plan["protocol_sha256"] != file_hash(run.PROTOCOL)):
        raise ValueError("sealed output drift")
    old = json.loads(run.source.PLAN.read_text())
    model, scaler, transform, action_hash = run.source.common.model_setup()
    if (model_digest(model) != plan["model_digest"] or action_hash != plan["normalizer_action_sha256"]
            or model_fingerprint(model) != old["model_sha256"]):
        raise ValueError("model or scaler drift")
    audit = freeze_audit(RGB, "lewm", [__file__, run.PLAN, run.source.inputs.INPUT,
                         run.source.common.CHECKPOINT / "weights.pt",
                         run.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; action scaler", clock="five five-step packets")
    guard = SingleFrameRGB(model, audit, expected_model_digest=plan["model_digest"],
                           action_mean=scaler.mean_, action_scale=scaler.scale_, packet_steps=5,
                           physical_low=-1, physical_high=1)
    low = np.broadcast_to((-1 - scaler.mean_) / scaler.scale_, (25, 2))
    high = np.broadcast_to((1 - scaler.mean_) / scaler.scale_, (25, 2))
    low_t = torch.tensor(low.copy().reshape(1, 1, 5, 10), dtype=torch.float32, device="cuda")
    high_t = torch.tensor(high.copy().reshape(1, 1, 5, 10), dtype=torch.float32, device="cuda")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, max_score_error, max_gradient_error, max_action_error, max_trace_error = [], 0., 0., 0., 0.
    try:
        with np.load(run.source.inputs.INPUT, allow_pickle=False) as inputs, \
                h5py.File(run.source.common.DATA, "r") as data:
            for row, oldrow, actual in zip(plan["rows"], old["rows"], physical["rows"], strict=True):
                if row["root"] != oldrow["root"] or row["row"] != actual["row"]:
                    raise ValueError("root mismatch")
                idx = int(np.flatnonzero(inputs["rows"] == row["row"])[0])
                context = {"pixels": transform(inputs["current"][idx])[None, None, None].cuda(),
                           "goal": transform(inputs["goal"][idx])[None, None, None].cuda(),
                           "action": torch.empty(1, 1, 0, 10, device="cuda")}
                initial = np.asarray(oldrow["generation_actions"])
                initial_costs = np.asarray(row["initial_costs"])
                np.testing.assert_allclose(initial_costs, oldrow["generation_costs"], atol=1e-3, rtol=0)
                incumbent, best_cost, best_i = run.select_scored(initial, initial_costs)
                assert best_i == row["incumbent_index"] and abs(best_cost-row["incumbent_cost"]) < 1e-9
                extra = run.cem_candidates(initial, initial_costs, low, high, 2720000 + row["root"])
                np.testing.assert_allclose(extra, row["cem"]["extra_actions"], atol=0, rtol=0)
                with torch.inference_mode():
                    vals, _ = guard.score(context, torch.tensor(extra.astype(np.float32).reshape(1, 20, 5, 10),
                                                                 device="cuda"))
                max_score_error = max(max_score_error, float(np.max(np.abs(vals[0].cpu().numpy() -
                                                                           row["cem"]["extra_costs"]))))
                x = torch.tensor(incumbent.astype(np.float32).reshape(1, 1, 5, 10), device="cuda")
                for j in range(20):
                    x = x.detach().requires_grad_(True)
                    max_action_error = max(max_action_error, float(np.max(np.abs(
                        x.detach().cpu().numpy().reshape(25, 2)-row["gradient"]["extra_actions"][j]))))
                    value, _ = guard.score(context, x)
                    grad, = torch.autograd.grad(value.sum(), x)
                    max_score_error = max(max_score_error, abs(float(value.item())-row["gradient"]["extra_costs"][j]))
                    max_gradient_error = max(max_gradient_error, abs(float(grad.abs().max())-
                                                                    row["gradient"]["gradient_max_abs"][j]))
                    x = run.gradient_step(x, grad, low_t, high_t)
                for arm in ("cem", "gradient"):
                    obj = row[arm]
                    assert obj["forward_candidates"] == 320
                    assert obj["backward_calls"] == (20 if arm == "gradient" else 0)
                    chosen, score, choice = run.select_scored(
                        np.concatenate((incumbent[None], np.asarray(obj["extra_actions"]))),
                        np.r_[best_cost, obj["extra_costs"]])
                    assert choice == obj["selected_index"] and abs(score-obj["selected_cost"]) < 1e-9
                    raw = run.source.to_physical(chosen, scaler).astype(np.float32)
                    np.testing.assert_allclose(raw, obj["selected_raw_actions"], rtol=0, atol=1e-7)
                    np.testing.assert_array_equal(raw, np.asarray(actual["arms"][arm]["actions"], np.float32))
                    env.reset(seed=42)
                    env.unwrapped._set_state(data["proprio"][row["row"]])
                    env.unwrapped._set_goal_state(data["proprio"][row["row"] + 25])
                    trace = []
                    for action, recorded in zip(raw, actual["arms"][arm]["trace"], strict=True):
                        _, _, term, trunc, info = env.unwrapped.step(action.copy())
                        max_trace_error = max(max_trace_error, abs(float(info["distance_to_target"])-recorded["distance"]))
                        assert bool(term) == recorded["terminated"] and bool(trunc) == recorded["truncated"]
                        trace.append(bool(term))
                    assert any(trace) == actual["arms"][arm]["ever_success"]
                rows.append({"root": row["root"], "row": row["row"]})
    finally:
        env.close()
    if max_score_error > 1e-3 or max_gradient_error > 1e-3 or max_action_error > 1e-5 or max_trace_error > 1e-6:
        raise ValueError("replay mismatch exceeds tolerance")
    guard.verify_unchanged()
    result = {"rows": rows, "plan_sha256": file_hash(run.PLAN),
              "physical_sha256": file_hash(run.PHYSICAL), "runner_sha256": file_hash(Path(__file__)),
              "max_score_error": max_score_error, "max_gradient_norm_error": max_gradient_error,
              "max_gradient_action_error": max_action_error, "max_physical_distance_error": max_trace_error,
              "scope": "same model and environment replay; no independent scientific replication"}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k.startswith("max_")}, indent=2))


if __name__ == "__main__":
    main()
