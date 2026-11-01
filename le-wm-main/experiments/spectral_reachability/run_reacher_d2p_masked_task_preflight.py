"""Fresh all-eight-root exact-actuator task-premise and CUDA chart preflight.

All neural predictions and all three mask contracts are sealed before any
future physical result. Full coordinate probes are reused only by the exact
subspaces that actually contain those actions; no compound outcome is invented.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots

HORIZONS, SCALES = (1, 2), (.1, .25)


def probe_spec(horizon):
    if horizon not in HORIZONS:
        raise ValueError("both declared horizons only")
    return [(scale, coordinate, sign) for scale in SCALES for coordinate in range(10 * horizon) for sign in (-1, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202649000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {k: args.output.with_suffix(v) for k, v in (("protocol", ".protocol.json"), ("predictions", ".predictions.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (args.output, *paths.values())):
        raise FileExistsError("new immutable masked-task outputs required")
    seeds = list(range(args.seed, args.seed + 8))
    excluded, bindings = bind_exclusions([args.disjoint_reference], seeds)
    prior = json.loads(args.disjoint_reference.read_text())["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if digest(args.policy / "weights.pt") != prior["weights_sha256"] or digest(args.adapter_artifact) != prior["adapter_sha256"] or source_sha != prior["source_sha256"]:
        raise ValueError("unchanged official checkpoint/source receipt required")
    roots = planned_source_roots(dataset, seeds, excluded)
    modules = {"auditor": Path(__file__), "chart": Path(chart.__file__), "p0": Path(p0.__file__), "context": Path(execution.__file__),
        "goal_contract": Path(goals.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename)}
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "weights_sha256": digest(args.policy / "weights.pt"), "adapter_sha256": digest(args.adapter_artifact), "source_sha256_receipt": source_sha,
        "tasks": {k: list(v) for k, v in chart.TASKS.items()}, "horizons": list(HORIZONS), "probe_scales": list(SCALES), "fd_epsilon": .001,
        "fd_directions_per_task": 4, "fd_median_relative_error_max": .1, "target_delta": [.65, -.65], "goal_contract": "joint-feasible-v1",
        "inactive_raw_torques_literal_zero": True, "no_post_training": True, "all_predictions_sealed_before_any_future_execution": True,
        "past_history": "Actual full-actuation source history, unchanged; exact task masks apply only to future actions.",
        "task_probe_reuse": "A full-drive single-coordinate probe belongs to a masked task iff that coordinate is active; center belongs to all. No mixed-action future inferred.",
        "damping": .01, "implementation_sha256": {k: digest(v) for k, v in modules.items()},
        "scope": "Fresh episode-disjoint exact-actuation task-premise pilot and model/physical response diagnosis. No root/task selection, planner efficacy, calibration, P1 or theory rejection.",
        "source_verification": "Reuse prior full-source SHA receipt, check actual HDF5 roots; no new complete-file SHA claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    for k in ("roots", "implementation"):
        paths[k].mkdir()
    for k, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (k + ".py"))
    print("HOST CUDA: fresh all8 roots, full drive / shoulder only / wrist only. Seal all predictions before measuring true RGB/qpos; exact zero inactive torque, no retraining or P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean, std)
    processor = shared._Preprocessor(normalizer)
    mean_t, std_t = (torch.tensor(v, device="cuda") for v in (normalizer.block_mean, normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameters, buffers = shared._parameter_hash(model), {k: v.clone() for k, v in model.named_buffers()}
    original_predict, samples = model.predict, 0

    def counted(emb, act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return original_predict(emb, act)

    model.predict = counted
    env = goals.goal_wrapper(protocol["goal_contract"])(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))

    def prepare(ctx):
        return p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"], np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))

    pending, predictions, rows = [], [], []
    try:
        for root in roots:
            seed = root["seed"]
            ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(ctx[k] != root[k] for k in ("source_episode", "source_start_row")):
                raise RuntimeError("presealed source root changed")
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            pixels, goal, history = prepare(ctx)
            emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            horizon_rows = []
            for horizon in HORIZONS:
                def terminal(flat, task="full_drive"):
                    return chart.terminal(model, emb, history, flat.reshape(1, horizon, len(chart.active_columns(task))), task=task, mean=mean_t, std=std_t)[0]
                center = torch.zeros(10 * horizon, device="cuda")
                with torch.no_grad():
                    base = terminal(center)
                full_j = torch.autograd.functional.jacobian(terminal, center.requires_grad_(True), vectorize=True).detach()
                task_rows = {}
                for task in chart.TASKS:
                    columns = chart.active_columns(task, horizon)
                    small = torch.zeros(len(columns), device="cuda")
                    compact_j = full_j if task == "full_drive" else torch.autograd.functional.jacobian(lambda value: terminal(value, task), small.requires_grad_(True), vectorize=True).detach()
                    error = float((compact_j - full_j[:, columns]).abs().max())
                    if error > 2e-6 or not torch.isfinite(compact_j).all():
                        raise RuntimeError("reduced raw chart Jacobian differs from exact full response columns")
                    generator = torch.Generator(device="cuda").manual_seed(seed + horizon * 1000 + len(columns))
                    fd_rows = []
                    for _ in range(4):
                        direction = torch.randn(len(columns), generator=generator, device="cuda")
                        direction /= direction.abs().max()
                        with torch.no_grad():
                            positive = terminal(.001 * direction, task)
                            negative = terminal(-.001 * direction, task)
                        fd, analytic = (positive - negative) / .002, compact_j @ direction
                        fd_rows.append({"direction": direction.tolist(), "positive_vector": positive.tolist(), "negative_vector": negative.tolist(),
                            "relative_error": float((fd - analytic).norm() / analytic.norm().clamp_min(1e-12))})
                    task_rows[task] = {"active_columns": columns, "model_jacobian": compact_j.tolist(), "full_column_max_error": error,
                        "fd_directions": fd_rows, "fd_median_relative_error": float(np.median([v["relative_error"] for v in fd_rows]))}
                probes = []
                for scale, coordinate, sign in probe_spec(horizon):
                    flat = torch.zeros(10 * horizon, device="cuda")
                    flat[coordinate] = sign * scale
                    raw = flat.reshape(horizon, 10).cpu().numpy()
                    expected_normalized = normalizer.transform(raw)
                    actual_normalized = chart.normalize_raw(flat.reshape(1, horizon, 10), mean_t, std_t)[0].detach().cpu().numpy()
                    if not np.allclose(expected_normalized, actual_normalized, rtol=0, atol=1e-6):
                        raise RuntimeError("future normalization is not the recorded-history normalization")
                    with torch.no_grad():
                        predicted = terminal(flat)
                    probes.append({"scale": scale, "coordinate": coordinate, "sign": sign, "raw_actions": raw.reshape(5 * horizon, 2).tolist(), "predicted_vector": predicted.tolist()})
                horizon_rows.append({"horizon": horizon, "center_predicted_vector": base.tolist(), "tasks": task_rows, "probes": probes})
            record = {**root, "root_rgb_sha256": ctx["root_rgb_sha256"], "goal_rgb_sha256": ctx["goal_rgb_sha256"], "goal_vector": target.tolist(),
                "goal_admission": goals.capture_goal_admission(env, protocol["goal_contract"]), "initial_snapshot": ctx["execution_snapshot"], "horizons": horizon_rows}
            predictions.append(record)
            pending.append((ctx, record))
            print(json.dumps({"seed": seed, "phase": "predictions", "all_tasks_retained": True}), flush=True)
        write_new(paths["predictions"], {"protocol_sha256": digest(paths["protocol"]), "rows": predictions})
        prediction_sha = digest(paths["predictions"])
        print(json.dumps({"all_predictions_presealed": True, "predictions_sha256": prediction_sha}), flush=True)
        for ctx, record in pending:
            seed = record["seed"]

            def observe(raw):
                frame, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                p0._verify_execution_identity(frame, goal, ctx["root_frame"], ctx["goal"])
                current, boundaries = ctx, []
                for j in range(len(raw) // 5):
                    current = execution.advance_executed_context(env, context=current, raw_action_block=raw[j * 5:(j + 1) * 5], action_block=5, verified_root_frame=current["root_frame"])
                    boundaries.append({"root_rgb_sha256": current["root_rgb_sha256"], "action_sha256": current["advanced_action_sha256"]})
                pixels, _, history = prepare(current)
                with torch.no_grad():
                    value = p0._latent_context(model, pixels, history)[0, -1]
                return {"observed_vector": value.tolist(), "qpos": np.asarray(env.unwrapped.env.physics.data.qpos, np.float64)[:2].tolist(),
                    "boundaries": boundaries, "raw_actions": raw.tolist()}

            measured = []
            for forecast in record["horizons"]:
                horizon = forecast["horizon"]
                center = observe(np.zeros((5 * horizon, 2), np.float32))
                endpoints = [{"scale": p["scale"], "coordinate": p["coordinate"], "sign": p["sign"],
                    **observe(np.asarray(p["raw_actions"], np.float32))} for p in forecast["probes"]]
                measured.append({"horizon": horizon, "center": center, "endpoints": endpoints})
            row = {**{k: record[k] for k in ("seed", "source_episode", "source_start_row")}, "horizons": measured}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "phase": "physical", "future_branches": 122}), flush=True)
    finally:
        model.predict = original_predict
        env.close()
    gates = {"cuda_only": True, "all8_roots": [r["seed"] for r in rows] == seeds,
        "all_sources_disjoint": len({r["source_episode"] for r in rows}) == 8 and not ({r["source_episode"] for r in rows} & excluded),
        "all976_future_branches": len(rows) == 8 and all([h["horizon"] for h in r["horizons"]] == [1, 2] and sum(1 + len(h["endpoints"]) for h in r["horizons"]) == 122 for r in rows),
        "all_masks_and_gradient_checks": all(c["full_column_max_error"] <= 2e-6 and c["fd_median_relative_error"] <= .1 for r in predictions for h in r["horizons"] for c in h["tasks"].values()),
        "predictions_unchanged": digest(paths["predictions"]) == prediction_sha, "parameters_unchanged": shared._parameter_hash(model) == parameters,
        "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()), "all_modules_eval": not any(m.training for m in model.modules()),
        "predict_sample_accounting": samples == 8 * 284, "weights_file_unchanged": digest(args.policy / "weights.pt") == protocol["weights_sha256"],
        "implementation_unchanged": all(digest(v) == protocol["implementation_sha256"][k] for k, v in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_masked_task_preflight.v1", "status": "completed" if all(gates.values()) else "fail", "cuda_host": host,
        "sealed_protocol": protocol, "protocol_sha256": digest(paths["protocol"]), "predictions_sha256": prediction_sha, "gates": gates,
        "actual_model_predict_samples": samples, "physical_futures": 976, "rows": rows})
    print(json.dumps({"gates": gates, "actual_model_predict_samples": samples}), flush=True)


if __name__ == "__main__":
    main()
