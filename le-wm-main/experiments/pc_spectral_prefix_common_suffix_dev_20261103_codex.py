"""Joint model-only prefix selection with equal multistart suffix work."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability.pc_spectral_seeded_feedback_cem_20261103_codex import (
    seeded_cem, spectral_candidates,
)

ROOT, h = prior.ROOT, prior.h
SOURCE = ROOT / "results/pc_pointmaze_shadow_closed_loop_dev_20261103_CODEX.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SPECTRAL_PREFIX_COMMON_SUFFIX_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_spectral_prefix_common_suffix_dev_20261103_CODEX.json"
NAMES = ("zero", "greedy", "spectral_plus_025", "spectral_minus_025",
         "spectral_plus_05", "spectral_minus_05", "random_0", "random_1")


def select(first, terminal, initial):
    first, terminal = np.asarray(first, dtype=float), np.asarray(terminal, dtype=float)
    if first.shape != (8,) or terminal.shape != (8,) or not np.isfinite(first).all() or not np.isfinite(terminal).all() or not np.isfinite(initial):
        raise ValueError("invalid candidate scores")
    selected = int(np.argmin(terminal))
    eligible = [i for i in range(2, 8) if first[i] >= initial and terminal[i] < min(terminal[:2])]
    qualified = selected in eligible
    return {"selected_index": selected, "eligible_indices": eligible,
            "selected_is_qualified": qualified, "abstained_reconfiguration": not qualified}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text())
    model_source, archive, _ = prior.files()
    prior.load_sealed(prior.CONTEXT)
    snapshots = []
    for row in source["rows"]:
        step = row["steps"][0]
        path = Path(step["decision_path"])
        if prior.sha(path) != step["decision_sha256"]:
            raise ValueError("snapshot hash mismatch")
        snapshots.append((row["name"], path))
    if len(snapshots) != 2:
        raise ValueError("unexpected roots")
    print("Compare model-generated prefixes with identical two-start suffix work; no physical dispatch.", flush=True)
    rows = []
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        before = h.parameter_hash(model)
        if before != model_source["model_before"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (mean.cpu().tolist() != archive["provenance"]["action_mean"]
                or std.cpu().tolist() != archive["provenance"]["action_std"]
                or not bool(torch.isfinite(std).all() and (std > 0).all())
                or not np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, atol=1e-7, rtol=0)
                or not np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, atol=1e-7, rtol=0)):
            raise ValueError("normalization mismatch")
        lower, upper = ((-1-mean)/std).repeat(5), ((1-mean)/std).repeat(5)
        zero = (-mean/std).repeat(5)
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        camera, old = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal = obstacle["goal_rgb"]
        goal_state, _ = h.estimate_state(goal, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
        for root_index, (name, path) in enumerate(snapshots):
            with np.load(path, allow_pickle=False) as saved:
                state = h.causal_state(saved["previous_rgb"], saved["current_rgb"], metadata, offset, .05)
                np.testing.assert_allclose(state, saved["rgb_derived_state"], rtol=0, atol=1e-7)
                with torch.no_grad():
                    context = h.encode_with_state(model, saved["current_rgb"], state, device)
                    initial = float(h.projected_cost(context["proprio"].reshape(1, 256, 16), target, pinv, pos_std)[0])
                def predict(packets):
                    with torch.no_grad():
                        predicted = unroll_with_action_history(model, context, packets[:0], packets, 1)["proprio"]
                        return h.projected_cost(predicted[-1].reshape(packets.shape[1], 256, 16), target, pinv, pos_std)
                def response(action):
                    future = action.reshape(1, 1, 10)
                    pred = unroll_with_action_history(model, context, future[:0], future, 1)["proprio"][-1]
                    return (pred.reshape(256, 16).mean(dim=0) @ pinv.T)[:2] * pos_std
                greedy_noise = torch.as_tensor(saved["noise"], dtype=torch.float32, device=device)
                greedy = seeded_cem(lambda x: predict(x.unsqueeze(0)), zero, lower, upper, greedy_noise)
                spectrum = spectral_candidates(response, mean, std)
                generator = torch.Generator(device="cpu").manual_seed(2026110610 + root_index)
                random_raw = torch.rand((2, 10), generator=generator).to(device) - .5
                bank = torch.cat((zero[None], greedy["chosen"][None], spectrum["normalized"],
                                  (random_raw - mean.repeat(5)) / std.repeat(5)))
                if bank.shape != (8, 10) or bool((bank < lower).any() or (bank > upper).any()):
                    raise ValueError("invalid prefix bank")
                suffix_noises = [torch.randn((3, 16, 10), generator=torch.Generator().manual_seed(
                    2026110700 + root_index * 10 + start), dtype=torch.float32).to(device) for start in range(2)]
                prefix_scores = predict(bank.unsqueeze(0))
                candidates, terminal = [], []
                started = time.monotonic()
                for index, prefix in enumerate(bank):
                    count = {"samples": 0, "batches": 0}
                    def evaluate(suffix):
                        count["samples"] += len(suffix)
                        count["batches"] += 1
                        packet = torch.stack((prefix.expand(len(suffix), -1), suffix))
                        return predict(packet)
                    searches = [seeded_cem(evaluate, zero, lower, upper, noise) for noise in suffix_noises]
                    winner = int(float(searches[1]["final_scores"].min()) < float(searches[0]["final_scores"].min()))
                    chosen = searches[winner]
                    value = float(chosen["final_scores"].min())
                    if count != {"samples": 100, "batches": 8}:
                        raise RuntimeError("suffix budget mismatch")
                    terminal.append(value)
                    candidates.append({"name": NAMES[index], "prefix_normalized": prefix.detach().cpu().tolist(),
                                       "prefix_raw": (prefix * std.repeat(5) + mean.repeat(5)).detach().cpu().tolist(),
                                       "prefix_score": float(prefix_scores[index]), "terminal_score": value,
                                       "suffix_normalized": chosen["chosen"].cpu().tolist(),
                                       "suffix_raw": (chosen["chosen"] * std.repeat(5) + mean.repeat(5)).cpu().tolist(),
                                       "winning_start": winner, "counts": count,
                                       "searches": [{"final_scores": s["final_scores"].cpu().tolist(),
                                                     "trace": [[t.cpu().tolist() for t in x] for x in s["trace"]]}
                                                    for s in searches]})
                torch.cuda.synchronize()
                decision = select(prefix_scores.cpu().numpy(), terminal, initial)
                rows.append({"root": name, "snapshot_sha256": prior.sha(path), "initial_score": initial,
                             "candidates": candidates, "decision": decision,
                             "spectral_singular_values": spectrum["singular_values"].cpu().tolist(),
                             "structure_counts": spectrum["structure_counts"], "greedy_counts": greedy["cem_counts"],
                             "prefix_score_samples": 8, "suffix_samples": 800,
                             "suffix_predicted_packet_transitions": 1600,
                             "suffix_seconds_including_logging": time.monotonic() - started})
                print(f"{name}: selected={NAMES[decision['selected_index']]} qualified={decision['selected_is_qualified']}", flush=True)
        after = h.parameter_hash(model)
        if after != before:
            raise ValueError("parameter hash changed")
    result = {"schema": "pc_spectral_prefix_common_suffix_dev.v1", "protocol_sha256": prior.sha(PROTOCOL),
              "source_sha256": prior.sha(SOURCE), "runner_sha256": prior.sha(Path(__file__)),
              "model_hash_before": before, "model_hash_after": after, "rows": rows,
              "physical_execution": False, "independent_confirmation": False}
    with OUTPUT.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
