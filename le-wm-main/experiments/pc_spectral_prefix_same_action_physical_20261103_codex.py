"""Durable execution and paired evaluation of the sealed prefix bank."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments import pc_pointmaze_shadow_closed_loop_durable_20261103_codex as old
from experiments import pc_shadow_spectral_closed_loop_dev_20261103_codex as executor
from experiments.pc_pointmaze_durable_dispatch_20261103_codex import DurableJournal, save_snapshot, _sync_dir
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge

ROOT, h = prior.ROOT, prior.h
SOURCE = ROOT / "results/pc_spectral_prefix_common_suffix_dev_20261103_CODEX.json"
PROTOCOL = ROOT.parent / "docs/PC_SPECTRAL_PREFIX_SAME_ACTION_PHYSICAL_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_spectral_prefix_same_action_physical_20261103_CODEX"


def reversals(x, y):
    reverse = comparable = ties = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        a, b = float(x[i]) - float(x[j]), float(y[i]) - float(y[j])
        if a == 0 or b == 0:
            ties += 1
        else:
            comparable += 1
            reverse += int((a < 0) != (b < 0))
    return {"reversed": reverse, "comparable": comparable, "ties": ties}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sealed = json.loads(SOURCE.read_text())
    if sealed["physical_execution"] or len(sealed["rows"]) != 2:
        raise ValueError("unexpected source")
    model_source, archive, names = prior.files()
    prior.load_sealed(prior.CONTEXT)
    for path, digest in old.EXPECTED.items():
        if prior.sha(path) != digest:
            raise ValueError(f"locked input changed: {path}")
    for index, name in old.SELECTED:
        if names[index] != name:
            raise ValueError("source roster changed")
    OUTPUT.mkdir()
    _sync_dir(OUTPUT.parent)
    rows, starts, chains = [], [], []
    with DurableJournal(OUTPUT / "events.jsonl") as journal:
        try:
            journal.record("preflight", source_sha256=prior.sha(SOURCE), protocol_sha256=prior.sha(PROTOCOL),
                           runner_sha256=prior.sha(Path(__file__)), planned_branches=16)
            with np.load(prior.SOURCE.with_suffix(".npz"), allow_pickle=False) as src, \
                 np.load(prior.CONTEXT.with_suffix(".npz"), allow_pickle=False) as oldctx, \
                 np.load(old.V4, allow_pickle=False) as v4, \
                 np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
                 h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
                device = h.require_cuda()
                args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                          config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
                model, dataset = h.load_model_and_dataset(args, device)
                model.eval()
                before = h.parameter_hash(model)
                if before != sealed["model_hash_before"] or before != model_source["model_before"]:
                    raise ValueError("model hash mismatch")
                mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
                if (mean.cpu().tolist() != archive["provenance"]["action_mean"] or
                    std.cpu().tolist() != archive["provenance"]["action_std"] or
                    not np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7) or
                    not np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
                    raise ValueError("normalization mismatch")
                weights, _, _ = h.get_encoder()
                pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
                pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
                camera, old_doc = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
                metadata, offset = camera["metadata"], old_doc["body_offset_from_static_scene"]
                goal_rgb = obstacle["goal_rgb"]
                goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
                with torch.no_grad():
                    target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
                for (source_index, name), planned in zip(old.SELECTED, sealed["rows"], strict=True):
                    if name != planned["root"] or len(planned["candidates"]) != 8:
                        raise ValueError("candidate roster mismatch")
                    history = src[name + "_actions"][:12].copy()
                    if not np.array_equal(history, oldctx[name + "_actions"]):
                        raise ValueError("history mismatch")
                    start = src["starts"][source_index].copy()
                    branch_rows = []
                    for idx, candidate in enumerate(planned["candidates"]):
                        prefix = np.asarray(candidate["prefix_raw"], dtype=np.float32).reshape(5, 2)
                        suffix = np.asarray(candidate["suffix_raw"], dtype=np.float32).reshape(5, 2)
                        sequence = np.concatenate((history, prefix, suffix)).astype(np.float64)
                        if not np.isfinite(sequence).all() or np.any(np.abs(sequence) > 1 + 1e-6):
                            raise ValueError("out of bounds")
                        sequence = np.clip(sequence, -1, 1)
                        folder = OUTPUT / f"{name}_candidate{idx}"
                        folder.mkdir()
                        _sync_dir(OUTPUT)
                        def check(tick, frame):
                            if tick <= 12 and old.image_hash(frame) != v4["shadow_hashes"][2*source_index, tick]:
                                raise ValueError("sealed history frame mismatch")
                        (_, _), actual_actions, frames = executor.run_arm(
                            "baseline", lambda: ShadowRGBBridge(start), sequence, None, folder,
                            journal, iterations=0, check_frame=check)
                        np.testing.assert_array_equal(actual_actions, sequence)
                        real_scores = []
                        for tick in (12, 17, 22):
                            state = h.causal_state(frames[tick-1], frames[tick], metadata, offset, .05)
                            with torch.no_grad():
                                encoded = h.encode_with_state(model, frames[tick], state, device)
                                score = h.projected_cost(encoded["proprio"].reshape(1, 256, 16),
                                                         target, pinv, pos_std)
                            real_scores.append(float(score.item()))
                        path = folder / "trajectory.npz"
                        digest = save_snapshot(path, actions=actual_actions, frames=frames)
                        branch = {"index": idx, "name": candidate["name"],
                                  "model_prefix_score": candidate["prefix_score"],
                                  "model_terminal_score": candidate["terminal_score"],
                                  "actual_rgb_scores": real_scores, "trajectory_path": str(path),
                                  "trajectory_sha256": digest, "physical_distances": {}}
                        journal.record("branch_completed", root=name, candidate=idx,
                                       trajectory_path=str(path), trajectory_sha256=digest)
                        branch_rows.append(branch)
                        starts.append(start)
                        chains.append(sequence)
                        print(f"{name} candidate {idx} complete", flush=True)
                    rows.append({"root": name, "selected_index": planned["decision"]["selected_index"],
                                 "initial_model_score": planned["initial_score"], "branches": branch_rows})
                if h.parameter_hash(model) != before:
                    raise ValueError("model hash changed")
                starts, chains = np.asarray(starts), np.asarray(chains)
                goal = np.asarray(h.GOAL[:2])
                physical = {tick: h.physical.replay_position_endpoints(starts, chains[:, :tick])
                            for tick in (12, 17, 22)}
                for root_index, row in enumerate(rows):
                    for branch in row["branches"]:
                        k = root_index*8 + branch["index"]
                        branch["physical_distances"] = {str(t): float(np.linalg.norm(physical[t][k, :2]-goal))
                                                         for t in physical}
                        branch["physical_endpoints"] = {str(t): physical[t][k].tolist() for t in physical}
                    b = row["branches"]
                    row["ranking"] = {}
                    for tick, actual_slot, model_key in ((17, 1, "model_prefix_score"), (22, 2, "model_terminal_score")):
                        pred = [x[model_key] for x in b]
                        actual = [x["actual_rgb_scores"][actual_slot] for x in b]
                        dist = [x["physical_distances"][str(tick)] for x in b]
                        row["ranking"][str(tick)] = {"model_vs_actual": reversals(pred, actual),
                                                       "actual_vs_physical": reversals(actual, dist)}
                    row["physical_terminal_oracle_index"] = min(range(8), key=lambda i: (b[i]["physical_distances"]["22"], i))
                journal.record("completed", model_hash_before=before, model_hash_after=h.parameter_hash(model),
                               branches=16, goal_encodes=1, feedback_encodes=48, physical_steps=352)
            receipt = {"schema": "pc_spectral_prefix_same_action_physical.v1", "rows": rows,
                       "source_sha256": prior.sha(SOURCE), "protocol_sha256": prior.sha(PROTOCOL),
                       "journal_sha256": prior.sha(OUTPUT / "events.jsonl"), "independent_confirmation": False}
            with (OUTPUT / "result.json").open("x") as out:
                json.dump(receipt, out, indent=2, allow_nan=False)
                out.write("\n")
        except BaseException as exc:
            journal.record("failed", error_type=type(exc).__name__, message=str(exc))
            raise


if __name__ == "__main__":
    main()
