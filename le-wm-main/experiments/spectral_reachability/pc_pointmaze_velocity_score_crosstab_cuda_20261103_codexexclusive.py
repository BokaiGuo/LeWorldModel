"""SHA-bound saved-action velocity x score CUDA diagnostic; no planner/physics."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import pc_ap_durable_context_adapter_20261102_codexexclusive as durable
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT.parent / "docs/PC_POINTMAZE_VELOCITY_SCORE_CROSSTAB_CUDA_20261103_CODEXEXCLUSIVE.md"
RES = ROOT / "results"
VELOCITY_SHA = "1a31884a8ab092a04f9d6437a043c54815d1cc0821f212d47bb96a14e8a0569a"
SOURCES = {
    "velocity": ("a0_pointmaze_cp5_velocity_same_action_20261103_DEV01.json", VELOCITY_SHA),
    "bank": ("a0_pointmaze_rgb_initial_candidate_oracle_20261103_DEV01.json",
             "44f3b133ca6332b9fcb831c6accd50052c239e38a9ef1c5e7ce4657debda2e76"),
    "actual": ("a0_pointmaze_rgb_initial_real_successor_20261103_DEV01.json",
               "3331a9c74efb93f163df4799828066f0237f3a767222f3aa454b80c68e63e2fa"),
    "prior": ("a0_pointmaze_fixed_bank_cp5_refresh_20261103_DEV01.json",
              "36c814bbe1123a64dcccc77c6b0a0e818d96df87f08044b6d8b345589508ef0c"),
    "plan": ("a1_pointmaze_rgb_state_full_planning_20261103_DEV01.json",
             "4288b73e675e45526db46698a7abdcd68b619fc235e0a73bc20f91bead6d9b99"),
    "noise": ("a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json",
              "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"),
    "camera": ("a0_pointmaze_camera_metadata_20261103_DEV01.json",
               "1b7e13ee87608cbc2fd5e98f3c42b098d0416c9425ec879e2cdd0bc963fc5ecf"),
}
MODEL_HASH = "13281cfa9c0385a1b97c0217130013094eb2ba5d707d0ee34b712276d6f8a0bf"
MODES = ("zero", "causal", "privileged")
AUTH = "PC_VELOCITY_SCORE_FIXED384_DIAGNOSTIC"
LOCK = RES / "e3_p2_host_cuda_cooperative.lock"
UUID = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_new(path, payload):
    with Path(path).open("x") as f:
        json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def source_plan():
    from .run_sra_p0_pointmaze_cuda import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET

    paths, docs = {}, {}
    for name, (filename, expected) in SOURCES.items():
        path = RES / filename
        if sha(path) != expected:
            raise ValueError(f"sealed report drift: {name}")
        paths[str(path)] = expected
        docs[name] = json.loads(path.read_text())
    for name in ("velocity", "bank", "actual", "prior", "noise"):
        path = (RES / SOURCES[name][0]).with_suffix(".npz")
        actual_hash = sha(path)
        if actual_hash != docs[name]["arrays_sha256"]:
            raise ValueError(f"sealed array drift: {name}")
        paths[str(path)] = actual_hash
    for path, key in ((DEFAULT_CHECKPOINT, "checkpoint_sha256"),
                      (DEFAULT_DATASET / "states.pth", "states_sha256"),
                      (DEFAULT_DATASET / "actions.pth", "actions_sha256")):
        if sha(path) != docs["noise"][key]:
            raise ValueError(f"training asset drift: {path}")
        paths[str(path)] = docs["noise"][key]
    for path in (DEFAULT_DATASET / "seq_lengths.pth", DEFAULT_CONFIG, DOC, Path(__file__)):
        paths[str(path)] = sha(path)
    for name in (
        "a0_pointmaze_cp5_velocity_same_action_20261103.py",
        "a0_pointmaze_fixed_bank_cp5_refresh_20261103.py",
        "a0_pointmaze_predicted_proprio_goal_score_20261103.py",
        "a0_pointmaze_proprio_action_response_20261103.py",
        "a0_pointmaze_rgb_geometry_state_20261103.py",
        "a0_pointmaze_rgb_initial_candidate_oracle_20261103.py",
        "a1_pointmaze_rgb_task_spectral_dev_20261102.py",
    ):
        p = ROOT / "experiments" / name
        paths[str(p)] = sha(p)
    for path in (ROOT / "experiments/spectral_reachability/run_sra_p0_pointmaze_cuda.py",
                 Path(durable.__file__), Path(lock.__file__),
                 ROOT / "external/jepa-wms/app/plan_common/datasets/point_maze_dset.py",
                 ROOT / "external/jepa-wms/app/plan_common/datasets/preprocessor.py"):
        paths[str(path)] = sha(path)
    groups = [(r["root"], r["method"]) for r in docs["velocity"]["rows"]]
    expected_groups = [(i, m) for i in (1884, 1885, 1886, 1887)
                       for m in ("random_cem", "spectral_cem")]
    if (groups != expected_groups or
            [(r["root"], r["method"]) for r in docs["bank"]["rows"]] != groups
            or docs["velocity"]["model_before"] != MODEL_HASH
            or docs["velocity"]["model_after"] != MODEL_HASH
            or docs["velocity"]["source_sha256"] != SOURCES["bank"][1]
            or docs["velocity"]["real_sha256"] != SOURCES["actual"][1]
            or docs["velocity"]["previous_sha256"] != SOURCES["prior"][1]):
        raise ValueError("roster, model or lineage mismatch")
    return {"schema": "pc_velocity_score_fixed384_plan_v1", "files": paths,
            "groups": [list(x) for x in groups], "candidates_per_group": 48,
            "modes": list(MODES), "model_parameter_hash": MODEL_HASH, "authority": AUTH,
            "lock": str(LOCK), "cuda_uuid": UUID, "visual_rtol": 1e-4, "visual_atol": 1e-6,
            "source_closure_complete": False, "P1_admission": False,
            "qvel_scope": "offline diagnostic only; never planner",
            "independent_outcome": False, "equal_total_compute": False}


def paired_rank(left, right):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) != 48 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("48 finite paired scores required")
    i, j = np.triu_indices(len(a), 1)
    x, y = a[i] - a[j], b[i] - b[j]
    comparable = (x != 0) & (y != 0)
    return {"pairs": len(i), "comparable": int(comparable.sum()),
            "reversed": int(((x * y < 0) & comparable).sum()),
            "score_ties": int((x == 0).sum()), "reference_ties": int((y == 0).sum())}


def require_close(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"invalid {name} arrays")
    diff = np.abs(a - b)
    if not np.allclose(a, b, rtol=1e-4, atol=1e-6):
        raise ValueError(f"{name}: max_abs={float(diff.max())}")
    return float(diff.max(initial=0))


def run(plan_path, expected, output, authority):
    if authority != AUTH or sha(plan_path) != expected:
        raise ValueError("authority or plan SHA mismatch")
    plan = json.loads(Path(plan_path).read_text())
    if plan != source_plan():
        raise ValueError("plan or source drift")
    output = Path(output)
    if not output.is_absolute() or output.parent != RES or not output.name.startswith(
            "pc_pointmaze_velocity_score_crosstab_RUN_") or os.path.lexists(output):
        raise ValueError("new exclusive results directory required")
    # Reserve unique output before trying the nonblocking cooperative lock.
    output.mkdir(mode=0o700)
    write_new(output / "plan.json", plan)
    journal = durable.Journal(output / "events.jsonl")
    counts = {"goal_encode_attempted": 0, "goal_encodes": 0,
              "actual_rgb_encode_attempted": 0, "actual_rgb_encodes": 0,
              "cp5_encode_attempted": 0, "cp5_encodes": 0,
              "prediction_attempted": 0, "prediction_samples": 0, "physical_steps": 0}
    started = time.monotonic()
    try:
        journal.emit({"kind": "run_started", "plan_sha256": expected})
        with lock.gpu_lock(LOCK):
            import torch
            from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
            from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
            from experiments.a0_pointmaze_fixed_bank_cp5_refresh_20261103 import metrics
            from .run_sra_p0_pointmaze_cuda import (
                DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET,
                load_model_and_dataset, parameter_hash, require_cuda)

            device = require_cuda()
            import uuid
            actual_uuid = str(torch.cuda.get_device_properties(device).uuid)
            if str(uuid.UUID(actual_uuid.removeprefix("GPU-"))) != str(uuid.UUID(UUID.removeprefix("GPU-"))):
                raise ValueError("GPU UUID mismatch")
            args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                                      config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
            model, dataset = load_model_and_dataset(args, device)
            before = parameter_hash(model)
            if before != MODEL_HASH:
                raise ValueError("frozen model hash mismatch")
            mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
                raise ValueError("normalizer")
            camera = json.loads((RES / SOURCES["camera"][0]).read_text())["metadata"]
            offset = json.loads((RES / SOURCES["plan"][0]).read_text())["body_offset_from_static_scene"]
            # npz opened read-only; outcome distances are not accessed until after all score checks.
            with np.load((RES / SOURCES["velocity"][0]).with_suffix(".npz"), allow_pickle=False) as velocity, \
                 np.load((RES / SOURCES["bank"][0]).with_suffix(".npz"), allow_pickle=False) as bank, \
                 np.load((RES / SOURCES["actual"][0]).with_suffix(".npz"), allow_pickle=False) as actual, \
                 np.load((RES / SOURCES["prior"][0]).with_suffix(".npz"), allow_pickle=False) as prior, \
                 np.load((RES / SOURCES["noise"][0]).with_suffix(".npz"), allow_pickle=False) as inputs:
                rgb5, rgb10 = prior["rgb5"], actual["rgb10"]
                if rgb5.shape != (384, 224, 224, 3) or rgb10.shape != rgb5.shape:
                    raise ValueError("fixed RGB dimensions")
                for group, (root, method) in enumerate(plan["groups"]):
                    key = f"r{root}_{method}"
                    journal.emit({"kind": "group_started", "group": key, "counts": dict(counts)})
                    mask = torch.as_tensor(inputs[f"r{root}_initial_mask"], dtype=torch.bool, device=device)
                    goal_rgb = inputs[f"r{root}_goal_rgb"]
                    goal_state, goal_meta = estimate_state(goal_rgb, camera, offset)
                    if np.any(goal_state[2:] != 0):
                        raise ValueError("goal velocity must be unobserved zero")
                    counts["goal_encode_attempted"] += 1
                    goal = encode_with_state(model, goal_rgb, goal_state, device)
                    counts["goal_encodes"] += 1
                    plans = bank[f"{key}_plans"]
                    if plans.shape != (48, 2, 2) or not np.isfinite(plans).all() or np.any(np.abs(plans) > 1):
                        raise ValueError("candidate actions")
                    images = rgb10[group * 48:(group + 1) * 48]
                    with torch.no_grad():
                        visual = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float()[:, None].to(device)
                        counts["actual_rgb_encode_attempted"] += 48
                        observed = model.encode({"visual": visual, "proprio": torch.zeros(48, 1, 4, device=device)})["visual"][:, 0]
                        counts["actual_rgb_encodes"] += 48
                        real_cost, _ = metrics(observed, observed, goal["visual"][0, 0], mask)
                    real_cost = real_cost.cpu().numpy()
                    real_error = require_close(real_cost, actual[f"{key}_real_cost"], key + " real RGB")
                    arrays = {"actions": plans, "goal_state_rgb_public": goal_state,
                              "goal_proprio_token": goal["proprio"].cpu().numpy(),
                              "actual_visual_encoding": observed.cpu().numpy(),
                              "actual_rgb_cost": real_cost,
                              "actual_rgb_cost_delta": real_cost - actual[f"{key}_real_cost"]}
                    errors = {}
                    for mode in MODES:
                        state = velocity[f"{key}_{mode}_input"]
                        if state.shape != (48, 4) or not np.isfinite(state).all():
                            raise ValueError("cp5 state shape")
                        scores, prop_scores, mses, prop_vectors = [], [], [], []
                        with torch.no_grad():
                            for j in range(48):
                                counts["cp5_encode_attempted"] += 1
                                context = encode_with_state(model, rgb5[group*48+j], state[j], device)
                                counts["cp5_encodes"] += 1
                                suffix = ((torch.as_tensor(plans[j:j+1, 1:], device=device) - mean) / std
                                          ).repeat(1, 1, 5).permute(1, 0, 2)
                                counts["prediction_attempted"] += 1
                                result = model.unroll(context, act_suffix=suffix)
                                counts["prediction_samples"] += 1
                                pred = result["visual"][-1]
                                prop = result["proprio"][-1]
                                cost, mse = metrics(pred, observed[j:j+1], goal["visual"][0, 0], mask)
                                pcost = ((prop - goal["proprio"][0, 0]) ** 2).reshape(1, -1).mean(dim=1)
                                scores.append(float(cost.item()))
                                mses.append(float(mse.item()))
                                prop_scores.append(float(pcost.item()))
                                prop_vectors.append(prop.cpu().numpy()[0])
                        arrays[f"{mode}_visual"] = np.asarray(scores)
                        arrays[f"{mode}_proprio"] = np.asarray(prop_scores)
                        arrays[f"{mode}_mse"] = np.asarray(mses)
                        arrays[f"{mode}_predicted_proprio"] = np.asarray(prop_vectors)
                        arrays[f"{mode}_visual_delta"] = arrays[f"{mode}_visual"] - velocity[f"{key}_{mode}_cost"]
                        arrays[f"{mode}_mse_delta"] = arrays[f"{mode}_mse"] - velocity[f"{key}_{mode}_mse"]
                        errors[mode] = {
                            "visual_score_max_abs": require_close(scores, velocity[f"{key}_{mode}_cost"], key + " visual"),
                            "visual_mse_max_abs": require_close(mses, velocity[f"{key}_{mode}_mse"], key + " mse")}
                        if not np.isfinite(arrays[f"{mode}_proprio"]).all():
                            raise ValueError("nonfinite proprio score")
                    group_path = output / f"{key}.npz"
                    with group_path.open("xb") as f:
                        np.savez_compressed(f, **arrays)
                        f.flush()
                        os.fsync(f.fileno())
                    journal.emit({"kind": "group_returned", "group": key, "arrays_sha256": sha(group_path),
                                  "visual_errors": errors, "real_rgb_cost_max_abs": real_error,
                                  "goal_state_rgb_public": goal_state.tolist(), "goal_projection": goal_meta,
                                  "counts": dict(counts)})
                    print(f"completed {key}: fixed 48 candidates x 3 velocities", flush=True)
                after = parameter_hash(model)
                if before != after or source_plan() != plan:
                    raise ValueError("model or source changed")
                journal.emit({"kind": "all_scores_verified", "model_before": before, "model_after": after,
                              "counts": dict(counts)})
                # Outcome is evaluator-only and cannot change any input, prediction or selection.
                rows, evaluation = [], {}
                for root, method in plan["groups"]:
                    key = f"r{root}_{method}"
                    distance = np.asarray(bank[f"{key}_distance"])
                    evaluation[f"{key}_physical_distance"] = distance
                    with np.load(output / f"{key}.npz", allow_pickle=False) as scores:
                        real = scores["actual_rgb_cost"]
                        row = {"root": root, "method": method, "n": 48,
                               "actual_rgb_vs_physical": paired_rank(real, distance), "conditions": {}}
                        for mode in MODES:
                            row["conditions"][mode] = {}
                            for score in ("visual", "proprio"):
                                predicted = scores[f"{mode}_{score}"]
                                choice = int(np.argmin(predicted))
                                row["conditions"][mode][score] = {
                                    "vs_actual_rgb": paired_rank(predicted, real),
                                    "vs_physical_distance": paired_rank(predicted, distance),
                                    "posthoc_score_argmin": choice,
                                    "posthoc_distance_at_score_argmin": float(distance[choice])}
                        rows.append(row)
                with (output / "evaluation.npz").open("xb") as f:
                    np.savez_compressed(f, **evaluation)
                    f.flush()
                    os.fsync(f.fileno())
                write_new(output / "report.json", {"schema": "pc_velocity_score_fixed384_report_v1",
                          "plan_sha256": expected, "model_before": before, "model_after": after,
                          "counts": counts, "rows": rows, "independent_outcome": False,
                          "P1_admission": False, "privileged_qvel_planner_usage": False,
                          "cost_scope": "re-encoding and 1152 fixed predictions, no new physics",
                          "elapsed_seconds": time.monotonic() - started,
                          "evaluation_sha256": sha(output / "evaluation.npz"),
                          "group_sha256": {f"r{r}_{m}": sha(output / f"r{r}_{m}.npz")
                                           for r, m in plan["groups"]}})
                journal.emit({"kind": "run_finished", "report_sha256": sha(output / "report.json"),
                              "counts": dict(counts)})
    except BaseException as exc:
        journal.emit({"kind": "run_failed", "error_type": type(exc).__name__, "error": str(exc)[:2000],
                      "counts": dict(counts)})
        raise
    finally:
        journal.close()
    print(json.dumps({"output": str(output), "report_sha256": sha(output / "report.json"),
                      "journal_sha256": sha(output / "events.jsonl"), "counts": counts}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    run_parser = sub.add_parser("run")
    for arg in ("plan", "sha256", "output", "authority"):
        run_parser.add_argument("--" + arg, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        path = Path(args.output).resolve()
        if path.parent != RES or not path.name.startswith("pc_pointmaze_velocity_score_crosstab_PLAN_"):
            raise ValueError("exclusive plan output")
        write_new(path, source_plan())
        print(json.dumps({"plan": str(path), "sha256": sha(path)}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
