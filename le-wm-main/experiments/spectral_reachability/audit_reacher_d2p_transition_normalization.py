"""Host-CUDA one-step BN regression on actual transition and joint trainers.

Uses real original source training clips, plus the original independent pair
pool for the joint trainer. This checks implementation, not generalization.
No checkpoint is exported. Both policies start from the same official state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import train_reacher_d2p_dynamics_repair as dynamics
from experiments.spectral_reachability import train_reacher_d2p_joint_repair as joint
from experiments.spectral_reachability import train_reacher_d2p_counterfactual_repair as pairs
from experiments.spectral_reachability import reacher_d2p_training_normalization as normalization
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

digest = dynamics._sha256  # Stream the 93-GB official source; never read_bytes().


def select_training_rows(base, excluded):
    """Metadata-only admission: two clips from each of four training episodes."""
    groups = {}
    for i, (episode, _) in enumerate(base.clip_indices):
        episode = int(episode)
        if episode % 10 >= 8 or episode in excluded:
            continue
        if episode not in groups and len(groups) == 4:
            continue
        group = groups.setdefault(episode, [])
        if len(group) < 2:
            group.append(i)
        if len(groups) == 4 and all(len(g) == 2 for g in groups.values()):
            return groups
    raise ValueError("four admitted real training episodes with two clips required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--exclusion-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".protocol.json").exists():
        raise FileExistsError("new immutable integration-test outputs required")
    manifest, exclusion = (json.loads(p.read_text()) for p in (args.repair_manifest, args.exclusion_protocol))
    inputs = {"repair_manifest": args.repair_manifest, "exclusion_protocol": args.exclusion_protocol,
        "transition_source": Path(manifest["transition_dataset"]), "pair_source": Path(manifest["counterfactual_pool"]),
        "pair_manifest": Path(manifest["counterfactual_manifest"]), "adapter": Path(manifest["adapter_artifact"]),
        "base_weights": Path(manifest["base_checkpoint"]) / "weights.pt"}
    print("HOST CUDA integration test: one actual transition/joint optimizer step per fixed BN policy. Verify real sources; no checkpoint export, no P1 or efficacy claim.", flush=True)
    hashes = {k: digest(p) for k, p in inputs.items()}
    for key, expected in (("transition_source", "transition_dataset_sha256"), ("pair_source", "counterfactual_pool_sha256"),
            ("pair_manifest", "counterfactual_manifest_sha256"), ("adapter", "adapter_sha256"), ("base_weights", "base_weights_sha256")):
        if hashes[key] != manifest[expected]:
            raise ValueError("original training input changed: " + key)
    provenance = json.loads(inputs["pair_manifest"].read_text())
    if provenance.get("status") != "completed" or provenance.get("p0_p1_calibration_or_outcome_roots_used") is not False:
        raise ValueError("original independent pair-pool provenance required")
    excluded = set(exclusion["excluded_source_episodes"]) | {r["source_episode"] for r in exclusion["source_roots"]}
    base = dynamics._build_dataset(inputs["transition_source"])
    groups = select_training_rows(base, excluded)
    indices = [i for group in groups.values() for i in group]
    modules = {"auditor": Path(__file__), "dynamics": Path(dynamics.__file__), "joint": Path(joint.__file__),
        "counterfactual": Path(pairs.__file__), "normalization": Path(normalization.__file__)}
    protocol = {"inputs": {k: {"path": str(p.resolve()), "sha256": hashes[k]} for k, p in inputs.items()},
        "implementation_sha256": {k: digest(p) for k, p in modules.items()}, "seed": 2026102611,
        "trainers": ["dynamics", "joint"], "policies": list(normalization.BN_POLICIES), "optimizer_steps_each": 1,
        "transition_source_clips": {str(k): [list(map(int, base.clip_indices[i])) for i in v] for k, v in groups.items()},
        "excluded_source_episodes": sorted(excluded), "batch_size": 8,
        "scope": "Actual CUDA optimizer integration only; training split, no exported model, no generalization or P-stage admission."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output.with_suffix(".protocol.json"), protocol)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = (torch.as_tensor(a, device="cuda", dtype=torch.float32) for a in shared._load_action_stats(inputs["adapter"], "swm/ReacherDMControl-v0"))
    pair_set, rows = pairs._Pairs(inputs["pair_source"], split="train"), []
    try:
        transition_batch = next(iter(DataLoader(Subset(base, indices), batch_size=8, num_workers=0)))
        pair_batch = next(iter(DataLoader(pair_set, batch_size=8, shuffle=False, num_workers=0)))
        for name in protocol["trainers"]:
            for policy in normalization.BN_POLICIES:
                torch.manual_seed(protocol["seed"])
                torch.cuda.manual_seed_all(protocol["seed"])
                model = shared.load_compatible_pretrained(inputs["base_weights"].parent, device="cuda").eval()
                trainable = dynamics._freeze_visual_target(model)
                before, params = normalization.normalization_snapshot(model), shared._parameter_hash(model)
                visual = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("encoder.", "projector."))}
                optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=1e-4)
                predict, samples = model.predict, 0

                def counted(emb, act):
                    nonlocal samples
                    if emb.device.type != "cuda" or act.device.type != "cuda":
                        raise RuntimeError("CPU neural fallback forbidden")
                    samples += int(emb.shape[0])
                    return predict(emb, act)

                model.predict = counted
                if name == "dynamics":
                    metrics = dynamics._epoch(model=model, loader=[transition_batch], device=torch.device("cuda"),
                        action_mean=mean, action_std=std, optimizer=optimizer, ranking_temperature=.05,
                        max_batches=1, epoch=1, split="train", batchnorm_policy=policy)
                else:
                    metrics = joint._run_epoch(model=model, transition_loader=[transition_batch], pair_loader=[pair_batch],
                        retention_pair_loader=None, device=torch.device("cuda"), mean=mean, std=std,
                        transition_temperature=.05, counterfactual_temperature=.05, counterfactual_weight=1.,
                        optimizer=optimizer, steps=1, batchnorm_policy=policy)
                after = normalization.normalization_snapshot(model)
                changed = [k for k in before if not torch.equal(before[k], after[k])]
                gates = {"finite_metrics": bool(np.isfinite(list(metrics.values())).all()),
                    "parameters_updated": shared._parameter_hash(model) != params,
                    "visual_state_unchanged": all(torch.equal(model.state_dict()[k], v) for k, v in visual.items()),
                    "finite_trainable_gradients": all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable),
                    "exact_forward_accounting": samples == (16 if name == "dynamics" else 32),
                    "declared_bn_behavior": not changed if policy == "frozen-base-statistics" else bool(changed)}
                rows.append({"trainer": name, "policy": policy, "metrics": metrics, "actual_predict_samples": samples,
                    "changed_normalization_buffers": changed, "gates": gates,
                    "normalization_receipt": normalization.verify_normalization(model, before, policy)})
                print(json.dumps(rows[-1]), flush=True)
                model.predict = predict
                del optimizer, model
    finally:
        pair_set.handle.close()
        if base.h5_file is not None:
            base.h5_file.close()
    gates = {"cuda_only": True, "all_four_optimizer_steps": len(rows) == 4,
        "all_integration_gates": all(all(v is True for v in r["gates"].values()) for r in rows),
        "source_clips_disjoint": not (set(groups) & excluded) and all(k % 10 < 8 for k in groups),
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items()),
        "base_checkpoint_unchanged": digest(inputs["base_weights"]) == hashes["base_weights"]}
    write_new(args.output, {"schema": "reacher_d2p_transition_normalization_guard.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "protocol_sha256": digest(args.output.with_suffix(".protocol.json")), "rows": rows, "gates": gates})
    print(json.dumps({"gates": gates}), flush=True)


if __name__ == "__main__":
    main()
