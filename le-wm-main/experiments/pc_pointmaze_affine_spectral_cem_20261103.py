"""Exposed-root spectral-injection ablation with real RGB feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA, OUTPUT as SOURCE
from experiments.a1_pointmaze_native_spectral_20261102 import packet
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.pc_pointmaze_affine_feedback_cem_20261103 import (
    OUTPUT as REFERENCE, ORIGINAL, ORIGINAL_SHA, SOURCE_SHA, projected_cost,
)
from experiments.pc_pointmaze_affine_inverse_20261103 import MEAN, STD, get_encoder
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

REFERENCE_SHA = "d1472f08afcf995168be28d62abded6b3e1083a634991efd5db61f3a20b84a4d"
LENGTHS_SHA = "66333840ffe5223486fcf8e1fdc7317c236fe1d89a17f7ce9ad8942fe6345e20"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AFFINE_SPECTRAL_CEM_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_affine_spectral_cem_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def spectral_cem(noise, scoring, injected):
    noise, injected = np.asarray(noise, np.float32), np.asarray(injected, np.float32)
    if (noise.shape != (4, 16, 2, 2) or injected.shape != (4, 2, 2)
            or not np.isfinite(noise).all() or not np.isfinite(injected).all()
            or np.any(np.abs(injected) > 1)):
        raise ValueError("invalid innovations or injected proposals")
    mean, std = np.zeros((2, 2), np.float32), np.full((2, 2), .5, np.float32)
    best_cost, best_plan, trace = float("inf"), None, []
    for iteration, batch in enumerate(noise):
        proposals = np.clip(mean + std * batch, -1, 1)
        if iteration == 0:
            proposals[:4] = injected
        values = np.asarray(scoring(proposals), np.float64)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("invalid scores")
        order = np.argsort(values, kind="stable")
        if values[order[0]] < best_cost:
            best_cost, best_plan = float(values[order[0]]), proposals[order[0]].copy()
        elite = proposals[order[:4]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .1)
        trace.append({"proposals": proposals.tolist(), "scores": values.tolist(),
                      "elite_indices": order[:4].tolist(), "mean": mean.tolist(), "std": std.tolist()})
    return best_plan, best_cost, trace


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (REFERENCE, REFERENCE_SHA), (ORIGINAL, ORIGINAL_SHA),
                           (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    source, reference, original, camera, old = (
        json.loads(p.read_text()) for p in (SOURCE, REFERENCE, ORIGINAL, CAMERA, OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (REFERENCE.with_suffix(".npz"), reference["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    if not np.isclose(mujoco.MjModel.from_xml_string(xml).opt.timestep * 5, .05, rtol=0, atol=1e-12):
        raise RuntimeError("incorrect RGB observation interval")
    weight, _, _ = get_encoder()
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences, terminals = [], {}, [], [], []
    started = time.monotonic()
    print("Compare affine-score spectral injection against saved random CEM on exposed roots.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"] or before != reference["model_before"]:
            raise RuntimeError("model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, atol=1e-7, rtol=0)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, atol=1e-7, rtol=0)):
            raise RuntimeError("proprio normalizer mismatch")
        action_mean, action_std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(action_mean).all()) or not bool(torch.isfinite(action_std).all()) or bool((action_std <= 0).any()):
            raise ValueError("invalid action normalizer")
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        if [r["root"] for r in source["rows"]] != list(range(1888, 1896)):
            raise RuntimeError("root roster mismatch")
        for prior in source["rows"]:
            root = prior["root"]
            if sha256_file(DEFAULT_DATASET / "obses" / f"episode_{root:03d}.pth") != prior["archive_sha256"]:
                raise RuntimeError("archive changed")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            rgb0, goal_rgb = saved[f"r{root}_rgb0"], saved[f"r{root}_goal_rgb"]
            goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            goal = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
            state0, _ = estimate_state(rgb0, camera["metadata"], old["body_offset_from_static_scene"])
            counts = {"initial_samples": 0, "tail_samples": 0, "initial_batches": 0, "tail_batches": 0}
            projection_seconds = {"initial": 0., "tail": 0.}
            def scorer(context, phase):
                def score(proposals):
                    counts[f"{phase}_samples"] += len(proposals)
                    counts[f"{phase}_batches"] += 1
                    plans = torch.as_tensor(proposals, dtype=torch.float32, device=device)
                    suffix = ((plans - action_mean) / action_std).repeat(1, 1, 5).permute(1, 0, 2)
                    with torch.no_grad():
                        pred = model.unroll(context, act_suffix=suffix)["proprio"][-1]
                        torch.cuda.synchronize()
                        score_started = time.monotonic()
                        result = projected_cost(pred.reshape(len(plans), 256, 16), goal, pinv, pos_std)
                        torch.cuda.synchronize()
                        projection_seconds[phase] += time.monotonic() - score_started
                    return result.cpu().numpy()
                return score
            with NativeRGBBridge(initial) as env:
                if not np.array_equal(env.initial_rgb, rgb0):
                    raise RuntimeError("reset RGB mismatch")
                context = encode_with_state(model, rgb0, state0, device)
                def response(action):
                    predicted = model.unroll(context, act_suffix=packet(action, action_mean, action_std))
                    token = predicted["proprio"][-1].reshape(256, 16).mean(dim=0)
                    return (token @ pinv.T)[:2] * pos_std
                zero = torch.zeros(2, device=device)
                basis = torch.eye(2, device=device)
                torch.cuda.synchronize()
                t0 = time.monotonic()
                jac = torch.stack([torch.autograd.functional.jvp(response, zero, d)[1].detach()
                                   for d in basis], dim=1)
                _, singular, vh = torch.linalg.svd(jac, full_matrices=False)
                torch.cuda.synchronize()
                structure_seconds = time.monotonic() - t0
                if not bool(torch.isfinite(jac).all()) or float(singular[0]) <= 1e-8:
                    raise ValueError("degenerate task Jacobian")
                direction = vh[0].cpu().numpy()
                injected = np.zeros((4, 2, 2), np.float32)
                injected[:, 0, :] = np.array([.3, -.3, .6, -.6], np.float32)[:, None] * direction
                torch.cuda.synchronize()
                t0 = time.monotonic()
                plan, cost0, trace0 = spectral_cem(saved[f"r{root}_initial_noise"],
                                                   scorer(context, "initial"), injected)
                torch.cuda.synchronize()
                initial_seconds = time.monotonic() - t0
                history = [env.initial_rgb]
                for action in np.repeat(plan[0][None], 5, axis=0):
                    history.append(env.step(action[None]))
                torch.cuda.synchronize()
                t0 = time.monotonic()
                state5 = causal_state(history[-2], history[-1], camera["metadata"],
                                      old["body_offset_from_static_scene"], .05)
                feedback = encode_with_state(model, history[-1], state5, device)
                tail, cost5, trace5 = cem(saved[f"r{root}_tail_noise"], scorer(feedback, "tail"))
                torch.cuda.synchronize()
                tail_seconds = time.monotonic() - t0
                rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                if env.step_count != 10:
                    raise RuntimeError("action clock mismatch")
                # Validation-only FD is after selection and is not fed into planning.
                with torch.no_grad():
                    finite = {str(h): torch.stack([(response(h*d) - response(-h*d)) / (2*h)
                                                    for d in basis], dim=1).cpu().tolist()
                              for h in (.01, .001)}
            if counts != {"initial_samples": 64, "tail_samples": 48, "initial_batches": 4, "tail_batches": 3}:
                raise RuntimeError("prediction count mismatch")
            sequence = np.concatenate((np.repeat(plan[0][None], 5, axis=0),
                                       np.repeat(tail[0][None], 5, axis=0)))
            starts.append(initial)
            sequences.append(sequence)
            terminals.append(rgb10)
            arrays[f"r{root}_history"], arrays[f"r{root}_rgb10"] = np.stack(history), rgb10
            rows.append({"root": root, "initial_plan": plan.tolist(), "tail": tail.tolist(),
                         "initial_cost": cost0, "tail_cost": cost5, "initial_trace": trace0,
                         "tail_trace": trace5, "cp5_input": state5.tolist(),
                         "jacobian": jac.cpu().tolist(), "singular_values": singular.cpu().tolist(),
                         "spectral_direction": direction.tolist(), "injected": injected.tolist(),
                         "validation_fd": finite, "attributable_jvp_calls": 2, "attributable_svd_calls": 1,
                         "validation_fd_prediction_calls": 8, "structure_seconds": structure_seconds,
                         "projection_seconds": projection_seconds,
                         "prediction_counts": counts, "current_rgb_encodes": 2, "physical_steps": 10,
                         "initial_plan_seconds": initial_seconds, "feedback_encode_and_plan_seconds": tail_seconds})
            print(f"root {root}: 10 physical steps", flush=True)
        after = parameter_hash(model)
        if after != before:
            raise RuntimeError("model changed")
    replay_rgb, endpoints = replay_legacy(starts, sequences, xml)
    endpoints2 = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(replay_rgb, np.stack(terminals)) or not np.array_equal(endpoints[:, :2], endpoints2[:, :2]):
        raise RuntimeError("physical replay mismatch")
    for index, row in enumerate(rows):
        target = dataset.states[row["root"], 10, :2].cpu().numpy().astype(np.float64)
        previous = next(r for r in reference["rows"] if r["root"] == row["root"])
        row.update({"endpoint_eval_only": endpoints[index].tolist(), "goal_position_eval_only": target.tolist(),
                    "distance_eval_only": float(np.linalg.norm(endpoints[index, :2] - target)),
                    "random_distance_eval_only": previous["distance_eval_only"]})
    arrays["sequences"], arrays["endpoints_eval_only"] = np.asarray(sequences), endpoints
    arrays["legacy_replay_rgb10"], arrays["legacy_replay_endpoints"] = replay_rgb, endpoints
    arrays["position_replay_endpoints"] = endpoints2
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_affine_spectral_cem.v1", "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "source_sha256": SOURCE_SHA,
              "reference_sha256": REFERENCE_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "seq_lengths_sha256": LENGTHS_SHA, "action_mean": action_mean.cpu().tolist(),
              "action_std": action_std.cpu().tolist(), "legacy_worker_sha256": sha256_file(
                  ROOT / "experiments/pointmaze_candidate_rgb_replay_worker_20261103.py"),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "physical_executions": len(rows), "physical_steps": 10 * len(rows),
              "wall_seconds": time.monotonic() - started, "rows": rows,
              "scope": "exposed roots; sample matched but not equal total compute; not admission"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
