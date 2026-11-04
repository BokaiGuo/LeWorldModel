"""Replay all fixed CEM scoring batches under the current bound inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA, OUTPUT as SOURCE
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.pc_pointmaze_affine_feedback_cem_20261103 import (
    ORIGINAL, ORIGINAL_SHA, OUTPUT as RANDOM, SOURCE_SHA, projected_cost,
)
from experiments.pc_pointmaze_affine_inverse_20261103 import MEAN, STD, get_encoder
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import (
    LENGTHS_SHA, OUTPUT as SPECTRAL, REFERENCE_SHA,
)
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SPECTRAL_SHA = "2606f1b1e6cbd70a263ed8593881df8355f2de20f15483ed0caa8db4f3e50694"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AFFINE_SCORE_REPLAY_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_affine_score_replay_20261103.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    expected = ((SOURCE, SOURCE_SHA), (RANDOM, REFERENCE_SHA), (SPECTRAL, SPECTRAL_SHA),
                (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
                (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA))
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"source hash changed: {path}")
    source, random, spectral, original, camera, old = (
        json.loads(p.read_text()) for p in (SOURCE, RANDOM, SPECTRAL, ORIGINAL, CAMERA, OLD))
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (RANDOM.with_suffix(".npz"), random["arrays_sha256"]),
                         (SPECTRAL.with_suffix(".npz"), spectral["arrays_sha256"]),
                         (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
                         (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                         (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"input hash changed: {path}")
    weight, _, _ = get_encoder()
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows = {}, []
    print("Replaying all saved random and spectral CEM score batches on host CUDA.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != random["model_before"] or before != spectral["model_before"]:
            raise ValueError("model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        action_mean, action_std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(action_mean).all()) or not bool(torch.isfinite(action_std).all()) or bool((action_std <= 0).any()):
            raise ValueError("invalid action normalizer")
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        for method, path, report in (("random", RANDOM, random), ("spectral", SPECTRAL, spectral)):
            if [r["root"] for r in report["rows"]] != list(range(1888, 1896)):
                raise ValueError("root roster mismatch")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as traces:
                for row in report["rows"]:
                    root = row["root"]
                    rgb0, goal_rgb = saved[f"r{root}_rgb0"], saved[f"r{root}_goal_rgb"]
                    history = traces[f"r{root}_history"]
                    goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
                    state0, _ = estimate_state(rgb0, camera["metadata"], old["body_offset_from_static_scene"])
                    state5 = causal_state(history[-2], history[-1], camera["metadata"],
                                          old["body_offset_from_static_scene"], .05)
                    goal = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
                    contexts = {"initial": encode_with_state(model, rgb0, state0, device),
                                "tail": encode_with_state(model, history[-1], state5, device)}
                    batch_rows = []
                    for phase in ("initial", "tail"):
                        for iteration, entry in enumerate(row[phase + "_trace"]):
                            plans = torch.as_tensor(entry["proposals"], dtype=torch.float32, device=device)
                            suffix = ((plans - action_mean) / action_std).repeat(1, 1, 5).permute(1, 0, 2)
                            with torch.no_grad():
                                pred = model.unroll(contexts[phase], act_suffix=suffix)["proprio"][-1]
                                actual = projected_cost(pred.reshape(len(plans), 256, 16), goal, pinv, pos_std).cpu().numpy()
                            old_scores = np.asarray(entry["scores"])
                            difference = np.abs(actual - old_scores)
                            order = np.argsort(actual, kind="stable")
                            original_order = np.argsort(old_scores, kind="stable")
                            key = f"r{root}_{method}_{phase}_{iteration}"
                            arrays[key] = actual
                            batch_rows.append({"phase": phase, "iteration": iteration,
                                               "max_abs_error": float(difference.max()),
                                               "max_relative_error": float((difference / np.maximum(np.abs(old_scores), 1e-30)).max()),
                                               "scores_close": bool(np.allclose(actual, old_scores, rtol=1e-4, atol=1e-6)),
                                               "full_order_same": bool(np.array_equal(order, original_order)),
                                               "elite_order_same": bool(np.array_equal(order[:4], entry["elite_indices"]))})
                    rows.append({"method": method, "root": root,
                                 "cp5_state_max_abs_error": float(np.max(np.abs(state5 - np.asarray(row["cp5_input"])))),
                                 "batches": batch_rows})
                    print(f"{method} root {root}: 7 score batches replayed", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_affine_score_replay.v1", "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "input_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p, _ in expected if p.is_relative_to(ROOT)},
              "model_before": before, "model_after": after, "prediction_batches": 112, "prediction_samples": 1792,
              "action_mean": action_mean.cpu().tolist(), "action_std": action_std.cpu().tolist(),
              "all_scores_close": all(b["scores_close"] for r in rows for b in r["batches"]),
              "all_orders_same": all(b["full_order_same"] for r in rows for b in r["batches"]),
              "all_elites_same": all(b["elite_order_same"] for r in rows for b in r["batches"]),
              "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT}: close={report['all_scores_close']} order={report['all_orders_same']}", flush=True)


if __name__ == "__main__":
    main()
