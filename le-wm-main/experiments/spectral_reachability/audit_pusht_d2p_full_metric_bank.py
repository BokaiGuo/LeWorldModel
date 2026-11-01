"""CUDA-only whole-bank projected/full-metric suffix causality diagnostic.

All four existing calibration roots and all sixteen fixed proposals are kept.
No outcome, candidate filtering by physical labels, training or gate override.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.audit_pusht_reachability_horizon import collect_nonterminal_boundary_contact_roots
from experiments.pusht_protocol_contracts import root_fingerprint
from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_pusht_d2p_p1_physical_calibration as physical
from experiments.spectral_reachability import pusht_d2p_full_metric_suffix as solver

MODES = ("random8_one_shot", "full192_one_shot", "full192_gn4")
OFFICIAL_SHA = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True)+"\n")


def solve(model, emb, actions, goal, *, mode, seed):
    if mode not in MODES:
        raise ValueError("explicit fixed solver mode required")
    projection = (p1._frozen_projection(latent_dim=192, sketch=8, device=goal.device, seed=seed)
                  if mode == MODES[0] else torch.eye(192, device=goal.device))
    reach = p1._reachability(model, goal, emb, actions, horizon=1, damping=.01, projection=projection)
    params = reach.pop("dls_params")[0]
    nonlinear = None
    if mode == MODES[2]:
        params, nonlinear = solver.refine(
            lambda value: p1._suffix_terminal(model, emb, actions, value[None])[0],
            goal, params, damping=.01, iterations=4)
    with torch.no_grad():
        predicted = p1._suffix_terminal(model, emb, actions, params[None])[0]
    return torch.tanh(params).detach().cpu().numpy(), {
        "mode": mode, "linear": reach, "nonlinear": nonlinear,
        "predicted_terminal_vector": predicted.detach().cpu().tolist(),
        "predicted_goal_cost": float((predicted-goal).square().mean().detach()),
        "projection_sha256": physical._sha256(projection.detach().cpu().numpy())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = physical._load_preflight(args.p1_evidence)
    protocol = reference["protocol"]
    if (protocol["cohort"] != "calibration" or protocol["root_count"] != 4
            or protocol["prefix_proposals"] != 16 or protocol["sketch_dims"] != 8
            or protocol["damping"] != .01 or protocol["suffix_horizon"] != 1):
        raise ValueError("fixed four-root/sixteen-candidate known calibration diagnostic required")
    if p1._sha256(args.policy/"weights.pt") != OFFICIAL_SHA:
        raise ValueError("official original frozen weights required")
    seal = physical._seal_run(args.output, args.p1_evidence, None)
    extra_sources = {"auditor": Path(__file__).resolve(), "solver": Path(solver.__file__).resolve()}
    for name, path in extra_sources.items():
        saved = args.output/"prerun_inputs"/(name+".py")
        saved.write_bytes(path.read_bytes())
    extra_hashes = {name: p1._sha256(path) for name, path in extra_sources.items()}
    plan = {"schema": "pusht_full_metric_bank_protocol.v1", "modes": MODES,
        "roots": [r["root_sha256"] for r in reference["rows"]], "proposal_indices": list(range(16)),
        "Q": "random8 legacy or exact full192 identity; same .01 damping", "gn_iterations": 4,
        "line_search": [1., .5, .25, .125], "zero_incumbent": True,
        "expected_real_branches": 192, "known_calibration_only": True, "outcome_opened": False,
        "no_physical_labels_in_selection": True, "weights_sha256": OFFICIAL_SHA,
        "extra_source_sha256": extra_hashes,
        "scope": "All-bank solver/metric diagnosis; no P1 admission or equal-compute efficacy claim."}
    write_new(args.output/"protocol.json", plan)
    stages = args.output/"stages"
    stages.mkdir()
    print("HOST CUDA: same4 known roots x ALL16 proposals x random8/full192/GN4 suffixes. No training, no outcome, no label-selected roots.", flush=True)
    host = require_cuda_allocation("cuda")
    device = torch.device("cuda")
    model = load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    before = p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    original = model.predict
    samples = 0
    def counted(emb, actions):
        nonlocal samples
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += len(emb)
        return original(emb, actions)
    model.predict = counted
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    transform = p0.image_transform()
    rows = []
    try:
        roots = collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(protocol["root_seed"]), oversample_factor=4)
        for index, (root, ref) in enumerate(zip(roots, reference["rows"], strict=True)):
            if root_fingerprint(root) != ref["root_sha256"]:
                raise ValueError("known calibration root reconstruction changed")
            frames, history, goal_rgb = physical._reset_and_warmup(env, root, history_len=3)
            emb, acts = p1._encode_context(model, frames, history, transform, device)
            goal = p0._goal_latent(model, goal_rgb, transform=transform, device=device)
            gen = torch.Generator(device=device).manual_seed(int(protocol["root_seed"])+index)
            raw = 2*torch.rand((16, 1, 2), generator=gen, device=device)-1
            bank = torch.atanh(raw.clamp(-1+torch.finfo(torch.float32).eps, 1-torch.finfo(torch.float32).eps))
            bank[0].zero_()
            if physical._sha256(bank.cpu().numpy()) != ref["compilation"]["proposal_bank_sha256"]:
                raise ValueError("same-bank reconstruction changed")
            predictions = []
            for candidate in range(16):
                with torch.no_grad():
                    successor, successor_actions = p1._advance(model, emb, acts, bank[candidate:candidate+1])
                suffixes = {}
                for mode in MODES:
                    action, receipt = solve(model, successor, successor_actions, goal, mode=mode,
                                             seed=int(protocol["root_seed"])+index+10_000)
                    suffixes[mode] = {"action": action.tolist(), **receipt}
                predictions.append({"proposal_index": candidate,
                    "prefix_action": torch.tanh(bank[candidate, 0]).cpu().tolist(),
                    "predicted_prefix_vector": successor[0, -1].cpu().tolist(), "suffixes": suffixes})
            root_plan = {"root_id": index, "root_sha256": ref["root_sha256"],
                "goal_vector": goal.cpu().tolist(), "legacy_selected": ref["compilation"]["selected_proposal_index"],
                "predictions": predictions, "future_physical_labels_used": False}
            write_new(stages/f"root_{index}.plan.json", root_plan)
            for candidate, predicted in enumerate(predictions):
                branches = {}
                for mode in MODES:
                    frames, history, actual_goal_rgb = physical._reset_and_warmup(env, root, history_len=3)
                    p0.packets.execute_packet(env, p0.packets.repeat_command(np.asarray(predicted["prefix_action"], np.float32)))
                    successor_rgb = np.asarray(env.render(), np.uint8).copy()
                    prefix_state = p0.packets.state_of(env)
                    observed_emb, observed_acts = p1._encode_context(model, frames[1:]+[successor_rgb],
                        np.concatenate((history[1:], [predicted["prefix_action"]]), axis=0), transform, device)
                    suffix, receipt = solve(model, observed_emb, observed_acts, goal, mode=mode,
                        seed=int(protocol["root_seed"])+index+10_000)
                    stage = {"root_id": index, "proposal_index": candidate, "mode": mode,
                        "root_plan_sha256": p1._sha256(stages/f"root_{index}.plan.json"),
                        "successor_rgb_sha256": physical._sha256(successor_rgb),
                        "observed_successor_vector": observed_emb[0, -1].cpu().tolist(),
                        "suffix_action": suffix.tolist(), "forecast": receipt,
                        "future_suffix_observations_used": False}
                    path = stages/f"root_{index}.candidate_{candidate}.{mode}.suffix.json"
                    write_new(path, stage)
                    p0.packets.execute_packet(env, p0.packets.repeat_command(suffix[0]))
                    final_rgb = np.asarray(env.render(), np.uint8).copy()
                    with torch.no_grad():
                        final_emb = model.encode({"pixels": p0._image_tensor(final_rgb, transform=transform, device=device)})["emb"][0, 0]
                    branches[mode] = {"stage_path": str(path.resolve()), "stage_sha256": p1._sha256(path),
                        "successor_rgb_sha256": stage["successor_rgb_sha256"], "successor_state": prefix_state.tolist(),
                        "prefix_pose": physical._pose_label(prefix_state, root["goal"]),
                        "observed_prefix_vector": observed_emb[0, -1].cpu().tolist(),
                        "suffix_action": suffix.tolist(), "suffix_receipt": receipt,
                        "final_state": p0.packets.state_of(env).tolist(),
                        "final_pose": physical._pose_label(p0.packets.state_of(env), root["goal"]),
                        "final_vector": final_emb.cpu().tolist(), "final_rgb_sha256": physical._sha256(final_rgb)}
                if len({b["successor_rgb_sha256"] for b in branches.values()}) != 1:
                    raise RuntimeError("solver-mode replay prefixes differ")
                rows.append({"root_id": index, "root_sha256": ref["root_sha256"], "proposal_index": candidate,
                    "prefix_action": predicted["prefix_action"], "goal_vector": goal.cpu().tolist(),
                    "predicted": predicted, "branches": branches})
            print(json.dumps({"root": index, "all_candidates": 16, "real_branches_completed": len(rows)*3, "predict_samples": samples}), flush=True)
    finally:
        env.close()
        model.predict = original
    gates = {"cuda_only": True, "all64_root_candidates": len(rows) == 64,
        "all192_real_branches": sum(len(r["branches"]) for r in rows) == 192,
        "parameters_unchanged": before == p0._parameter_hash(model),
        "buffers_unchanged": all(torch.equal(buffers[k], v) for k,v in model.named_buffers()),
        "all_modules_eval": all(not m.training for m in model.modules()),
        "prerun_seal_unchanged": physical._seal_unchanged(seal),
        "extra_sources_unchanged": all(p1._sha256(path) == extra_hashes[name] for name,path in extra_sources.items()),
        "weights_file_unchanged": p1._sha256(args.policy/"weights.pt") == OFFICIAL_SHA}
    write_new(args.output/"report.json", {"schema": "pusht_full_metric_bank.v1", "status": "completed" if all(gates.values()) else "integrity_fail",
        "cuda_host": host, "gates": gates, "protocol": plan, "actual_predict_samples": samples,
        "upstream_path": str(args.p1_evidence.resolve()), "upstream_sha256": p1._sha256(args.p1_evidence),
        "rows": rows, "p1_admitted": False})
    print(json.dumps({"gates": gates, "predict_samples": samples, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
