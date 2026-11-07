"""Equal-forward-opportunity RGB CEM and gradient continuation development."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_CEM_VS_GRADIENT_CONTINUATION_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_cem_gradient_continuation_plan_20261107_codex.json"
PHYSICAL = ROOT / "results/pc_tworoom_cem_gradient_continuation_physical_20261107_codex.json"


def cem_candidates(actions, costs, low, high, seed):
    actions, costs = np.asarray(actions), np.asarray(costs)
    if actions.shape[0] != 300 or costs.shape != (300,):
        raise ValueError("expected sealed 300-candidate generation")
    elite = actions[200:][np.argsort(costs[200:], kind="stable")[:10]]
    mean, std = elite.mean(0), np.maximum(elite.std(0), 1e-5)
    return np.clip(mean + np.random.default_rng(seed).normal(size=(20, *mean.shape)) * std, low, high)


def gradient_step(x, gradient, low, high):
    if x.shape != gradient.shape or not torch.isfinite(gradient).all():
        raise ValueError("invalid gradient")
    with torch.no_grad():
        return torch.maximum(torch.minimum(x - .05 * gradient / gradient.abs().max().clamp_min(1e-12),
                                           high), low).detach()


def select_scored(actions, costs):
    actions, costs = np.asarray(actions), np.asarray(costs)
    if len(actions) != len(costs) or not len(costs) or not np.isfinite(costs).all():
        raise ValueError("invalid scored pool")
    i = int(np.argmin(costs))
    return actions[i].copy(), float(costs[i]), i


def plan():
    if PLAN.exists() or PHYSICAL.exists():
        raise FileExistsError("output exists")
    old = json.loads(source.PLAN.read_text())
    if (old["input_sha256"] != file_hash(source.inputs.INPUT)
            or old["protocol_sha256"] != file_hash(source.PROTOCOL)
            or old["runner_sha256"] != file_hash(source.__file__)):
        raise ValueError("sealed input drift")
    model, scaler, transform, action_hash = source.common.model_setup()
    if model_fingerprint(model) != old["model_sha256"] or action_hash != old["normalizer_action_sha256"]:
        raise ValueError("model or normalizer drift")
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.PLAN, source.inputs.INPUT,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="five five-step packets")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    low = np.broadcast_to((-1 - scaler.mean_) / scaler.scale_, (25, 2))
    high = np.broadcast_to((1 - scaler.mean_) / scaler.scale_, (25, 2))
    low_t = torch.tensor(low.copy().reshape(1, 1, 5, 10), dtype=torch.float32, device="cuda")
    high_t = torch.tensor(high.copy().reshape(1, 1, 5, 10), dtype=torch.float32, device="cuda")
    rows = []
    with np.load(source.inputs.INPUT, allow_pickle=False) as data:
        for row in old["rows"]:
            root = row["root"]
            index = int(np.flatnonzero(data["rows"] == row["row"])[0])
            context = {"pixels": transform(data["current"][index])[None, None, None].cuda(),
                       "goal": transform(data["goal"][index])[None, None, None].cuda(),
                       "action": torch.empty(1, 1, 0, 10, device="cuda")}
            actions = np.asarray(row["generation_actions"])
            saved_costs = np.asarray(row["generation_costs"])
            recost = []
            with torch.inference_mode():
                for offset in range(0, 300, 32):
                    candidate = torch.as_tensor(actions[offset:offset + 32].astype(np.float32).reshape(1, -1, 5, 10),
                                                device="cuda")
                    values, _ = guard.score(context, candidate)
                    recost.extend(values[0].cpu().tolist())
            recost = np.asarray(recost)
            if np.max(np.abs(recost - saved_costs)) > 1e-3:
                raise ValueError("sealed CEM cost mismatch")
            incumbent, best_cost, best_index = select_scored(actions, recost)
            extras = cem_candidates(actions, recost, low, high, 2720000 + root)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                candidate = torch.as_tensor(extras.astype(np.float32).reshape(1, 20, 5, 10), device="cuda")
                cem_values, _ = guard.score(context, candidate)
            torch.cuda.synchronize()
            cem_seconds = time.perf_counter() - started
            cem_costs = np.r_[best_cost, cem_values[0].cpu().numpy()]
            cem_selected, cem_value, cem_index = select_scored(np.concatenate((incumbent[None], extras)), cem_costs)
            torch.cuda.synchronize()
            started = time.perf_counter()
            x = torch.tensor(incumbent.astype(np.float32).reshape(1, 1, 5, 10), device="cuda")
            grad_actions, grad_values, gradient_norms = [], [], []
            for _ in range(20):
                x = x.detach().requires_grad_(True)
                value, _ = guard.score(context, x)
                grad, = torch.autograd.grad(value.sum(), x)
                grad_actions.append(x.detach().cpu().numpy().reshape(25, 2).tolist())
                grad_values.append(float(value.item()))
                gradient_norms.append(float(grad.detach().abs().max()))
                x = gradient_step(x, grad, low_t, high_t)
            torch.cuda.synchronize()
            grad_seconds = time.perf_counter() - started
            grad_costs = np.r_[best_cost, grad_values]
            grad_selected, grad_value, grad_index = select_scored(
                np.concatenate((incumbent[None], np.asarray(grad_actions))), grad_costs)
            rows.append({"root": root, "row": row["row"], "initial_costs": recost.tolist(),
                         "original_cost_max_abs": float(np.max(np.abs(recost-saved_costs))),
                         "incumbent_index": best_index, "incumbent_cost": best_cost,
                         "cem": {"extra_actions": extras.tolist(), "extra_costs": cem_values[0].cpu().tolist(),
                                 "selected_index": cem_index, "selected_cost": cem_value,
                                 "selected_raw_actions": source.to_physical(cem_selected, scaler).tolist(),
                                 "extension_seconds": cem_seconds, "forward_candidates": 320, "backward_calls": 0},
                         "gradient": {"extra_actions": grad_actions, "extra_costs": grad_values,
                                      "gradient_max_abs": gradient_norms, "selected_index": grad_index,
                                      "selected_cost": grad_value,
                                      "selected_raw_actions": source.to_physical(grad_selected, scaler).tolist(),
                                      "extension_seconds": grad_seconds, "forward_candidates": 320,
                                      "backward_calls": 20}})
            print(f"root {root}: CEM={cem_value:.4f}, gradient={grad_value:.4f}", flush=True)
    guard.verify_unchanged()
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "source_plan_sha256": file_hash(source.PLAN),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "contract_sha256": file_hash(Path(__file__).with_name("pc_rgb_single_frame_contract_20261107_codex.py")),
                   "model_digest": digest, "normalizer_action_sha256": action_hash,
                   "scope": "RGB single-frame exposed dev; equal forward opportunities, not equal cost"},
                  f, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import h5py
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    planned = json.loads(PLAN.read_text())
    if (planned["source_plan_sha256"] != file_hash(source.PLAN)
            or planned["protocol_sha256"] != file_hash(PROTOCOL)
            or planned["runner_sha256"] != file_hash(__file__)):
        raise ValueError("plan dependencies drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(source.common.DATA, "r") as data:
            for row in planned["rows"]:
                results = {}
                for arm in ("cem", "gradient"):
                    env.reset(seed=42)
                    env.unwrapped._set_state(data["proprio"][row["row"]])
                    env.unwrapped._set_goal_state(data["proprio"][row["row"] + 25])
                    actions = np.asarray(row[arm]["selected_raw_actions"], dtype=np.float32)
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                        raise ValueError("invalid physical actions")
                    trace = []
                    for action in actions:
                        _, _, terminated, truncated, info = env.unwrapped.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(terminated), "truncated": bool(truncated)})
                    results[arm] = {"actions": actions.tolist(), "trace": trace,
                                    "ever_success": any(s["terminated"] for s in trace),
                                    "final_success": trace[-1]["terminated"]}
                rows.append({"root": row["root"], "row": row["row"], "arms": results})
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "rows": rows}, f, indent=2, allow_nan=False)
    for arm in ("cem", "gradient"):
        print(f"{arm}: {sum(r['arms'][arm]['ever_success'] for r in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
