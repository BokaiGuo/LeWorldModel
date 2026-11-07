"""State-assisted live feedback on sealed 160-query PointMaze CEM decisions."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from experiments import pc_pointmaze_goal25_packet_cem_20261107_codex as source
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_pointmaze_state_feedback_bridge_20261107_codex import NativeRGBBridge
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PACKET_CEM_FEEDBACK_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_pointmaze_packet_cem_feedback_20261107_codex.json"


def candidates(rng, mean, std, old_tail, first):
    mean, std, old_tail = map(np.asarray, (mean, std, old_tail))
    if mean.ndim != 2 or mean.shape[1] != 2 or std.shape != mean.shape or old_tail.shape != mean.shape:
        raise ValueError("packet shape mismatch")
    if not all(np.isfinite(x).all() for x in (mean, std, old_tail)) or (std < 0).any() or np.abs(old_tail).max() > 1:
        raise ValueError("invalid distribution or old tail")
    pool = np.clip(rng.normal(size=(16, *mean.shape)).astype(np.float32) * std + mean, -1, 1)
    pool[0] = old_tail if first else np.clip(mean, -1, 1)
    return pool


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = source.check_plan()
    physical = json.loads(source.PHYSICAL.read_text())
    if physical["plan"] != file_hash(source.PLAN) or physical["witness"] != file_hash(source.prior.PRIVATE):
        raise ValueError("sealed source mismatch")
    model, dataset, audit = load_model()
    audit = configure_context(model, audit)
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model drift")
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(source.prior.retention.corrected.prior.prior.old.inputs.CONFIG.read_text())
    mean_a, std_a = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    overall_start = time.perf_counter()
    with np.load(source.prior.PUBLIC, allow_pickle=False) as public, np.load(source.prior.PRIVATE, allow_pickle=False) as private, torch.inference_mode():
        for i, record in enumerate(plan["rows"]):
            root = 1884 + i
            if record["root"] != root or physical["rows"][i]["root"] != root:
                raise ValueError("root mismatch")
            begin = time.perf_counter()
            torch.cuda.synchronize()
            encode_start = time.perf_counter()
            target = model.encode({"visual": rgb_tensor(public["goal"][i]),
                                   "proprio": torch.as_tensor(public["goal_proprio"][i], dtype=torch.float32, device="cuda")[None, None]})
            torch.cuda.synchronize()
            goal_seconds = time.perf_counter() - encode_start
            objective = ReprTargetDistMPCObjective(cfg, target, alpha=.1)
            initial_plan = np.asarray(record["selected"]["160"]["actions"], dtype=np.float32)
            if initial_plan.shape != (25, 2):
                raise ValueError("invalid initial plan")
            executed, decisions = [], []
            with NativeRGBBridge(private["initial"][i]) as env:
                if not np.array_equal(env.initial_rgb, public["current"][i]):
                    raise ValueError("initial image mismatch")
                np.testing.assert_allclose(env.initial_proprio, public["proprio"][i], atol=1e-6, rtol=0)
                current_rgb, current_state = env.step(initial_plan[:5])
                executed.extend(initial_plan[:5].tolist())
                np.testing.assert_allclose(current_state, np.asarray(physical["rows"][i]["trace"])[1, 5, :4],
                                           atol=1e-6, rtol=0)
                tail = initial_plan[5:].reshape(4, 5, 2)[:, 0].copy()
                for k in range(4):
                    n = 4 - k
                    if tail.shape != (n, 2):
                        raise ValueError("remaining horizon mismatch")
                    observation = {"visual": rgb_tensor(current_rgb),
                                   "proprio": torch.as_tensor(current_state, dtype=torch.float32, device="cuda")[None, None]}
                    boundary = StateAssistedPointMaze(model, audit)
                    rng = np.random.default_rng(2026110700 + root * 10 + k)
                    mean, std = np.zeros((n, 2), np.float32), np.full((n, 2), .5, np.float32)
                    rounds, all_pool, all_cost = [], [], []
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for iteration in range(3):
                        pool = candidates(rng, mean, std, tail, iteration == 0)
                        costs = []
                        for j in range(0, 16, 4):
                            raw = torch.as_tensor(np.repeat(pool[j:j+4], 5, axis=1), device="cuda")
                            packed = ((raw - mean_a) / std_a).reshape(4, n, 10).transpose(0, 1).contiguous()
                            expanded = {key: value.expand(4, *value.shape[1:]).contiguous()
                                        for key, value in observation.items()}
                            costs.extend(objective(boundary.predict(expanded, packed), packed).cpu().tolist())
                        costs = np.asarray(costs)
                        if costs.shape != (16,) or not np.isfinite(costs).all():
                            raise ValueError("invalid scores")
                        elite = pool[np.argsort(costs, kind="stable")[:4]]
                        rounds.append({"mean": mean.tolist(), "std": std.tolist(), "candidates": pool.tolist(),
                                       "costs": costs.tolist()})
                        all_pool.append(pool)
                        all_cost.append(costs)
                        mean, std = elite.mean(0), elite.std(0)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    pools, costs = np.concatenate(all_pool), np.concatenate(all_cost)
                    chosen = int(np.argmin(costs))
                    selected = pools[chosen].copy()
                    action = np.repeat(selected[:1], 5, axis=0)
                    observe_start = time.perf_counter()
                    current_rgb, current_state = env.step(action)
                    observe_seconds = time.perf_counter() - observe_start
                    executed.extend(action.tolist())
                    decisions.append({"step": 5 * (k + 1), "horizon_packets": n, "old_tail": tail.tolist(),
                                      "chosen": chosen, "selected": selected.tolist(), "selected_cost": float(costs[chosen]),
                                      "query_count": 48, "model_packet_steps": 48 * n, "rounds": rounds,
                                      "planning_seconds": elapsed, "environment_seconds": observe_seconds,
                                      "observation_proprio_before": observation["proprio"][0, 0].cpu().tolist()})
                    tail = selected[1:]
                if env.step_count != 25 or len(executed) != 25:
                    raise ValueError("execution clock mismatch")
                final = current_state.copy()
            goal_state = private["goal"][i]
            distance = float(np.linalg.norm(final[:2] - goal_state[:2]))
            old_distance = float(np.asarray(physical["rows"][i]["distance"])[1, -1])
            rows.append({"root": root, "initial_success": physical["rows"][i]["initial_success"],
                         "feedback_final_distance": distance, "feedback_success": distance < .5,
                         "open_final_distance": old_distance, "open_success": old_distance < .5,
                         "initial_query_count": 160, "feedback_queries": 192,
                         "feedback_model_packet_steps": sum(x["model_packet_steps"] for x in decisions),
                         "goal_encode_seconds": goal_seconds, "wall_seconds": time.perf_counter() - begin,
                         "final_proprio": final.tolist(), "executed_actions": executed, "decisions": decisions})
            print(f"root={root}: feedback={distance:.4f} open={old_distance:.4f}", flush=True)
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model changed")
    output = {"track": STATE, "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
              "bridge": file_hash(Path(__file__).with_name("pc_pointmaze_state_feedback_bridge_20261107_codex.py")),
              "source_plan": file_hash(source.PLAN), "source_physical": file_hash(source.PHYSICAL),
              "model": plan["model"], "rows": rows, "wall_seconds": time.perf_counter() - overall_start,
              "scope": "exposed state-assisted roots; extra feedback compute; not RGB-only or independent"}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
