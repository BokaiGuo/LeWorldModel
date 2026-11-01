"""Real-source CUDA one-batch integration test of the repair BN contract.

Runs the ACTUAL counterfactual trainer for one optimizer step under each
predeclared normalization policy. No checkpoint is saved and no planning or
generalization efficacy is inferred from this implementation test.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import train_reacher_d2p_counterfactual_repair as trainer
from experiments.spectral_reachability import reacher_d2p_training_normalization as normalization
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".protocol.json").exists():
        raise FileExistsError("new immutable guard-test outputs required")
    manifest = json.loads(args.repair_manifest.read_text())
    pool, pool_manifest = (Path(manifest[k]) for k in ("counterfactual_pool", "counterfactual_manifest"))
    source = json.loads(pool_manifest.read_text())
    if source.get("status") != "completed" or source.get("p0_p1_calibration_or_outcome_roots_used") is not False:
        raise ValueError("independent completed training-only pair source required")
    if digest(pool_manifest) != manifest["counterfactual_manifest_sha256"] or digest(pool) != manifest["counterfactual_pool_sha256"]:
        raise ValueError("original training pair provenance changed")
    policy, adapter = Path(manifest["base_checkpoint"]), Path(manifest["adapter_artifact"])
    if digest(policy / "weights.pt") != manifest["base_weights_sha256"] or digest(adapter) != manifest["adapter_sha256"]:
        raise ValueError("official base or normalization changed")
    protocol = {"repair_manifest_sha256": digest(args.repair_manifest), "pair_pool_sha256": digest(pool),
        "base_weights_sha256": manifest["base_weights_sha256"], "adapter_sha256": manifest["adapter_sha256"],
        "batch_size": 8, "batches": 1, "seed": 2026102607, "policies": list(normalization.BN_POLICIES),
        "source_rows": "first eight deterministic training-split rows; no P0/P1 roots",
        "trainer_sha256": digest(Path(trainer.__file__)), "normalization_sha256": digest(Path(normalization.__file__)),
        "auditor_sha256": digest(Path(__file__)), "scope": "Real-data CUDA optimizer integration test only; no exported checkpoint, calibration, efficacy or theory verdict."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output.with_suffix(".protocol.json"), protocol)
    print("HOST CUDA real-source one-batch optimizer check: frozen-base vs legacy BN modes. Uses only independent original TRAINING pair pool. No model exported and no planning claim.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = (torch.as_tensor(a, device="cuda", dtype=torch.float32) for a in shared._load_action_stats(adapter, "swm/ReacherDMControl-v0"))
    dataset = trainer._Pairs(pool, split="train")
    if len(dataset) < 8:
        raise ValueError("eight source training examples required")
    rows = []
    try:
        for bn_policy in normalization.BN_POLICIES:
            torch.manual_seed(protocol["seed"])
            torch.cuda.manual_seed_all(protocol["seed"])
            model = shared.load_compatible_pretrained(policy, device="cuda").eval()
            trainable = trainer._freeze_visual_target(model)
            params_before = shared._parameter_hash(model)
            visual = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("encoder.", "projector."))}
            before = normalization.normalization_snapshot(model)
            optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=1e-4)
            predict, samples = model.predict, 0

            def counted_predict(emb, act):
                nonlocal samples
                if emb.device.type != "cuda" or act.device.type != "cuda":
                    raise RuntimeError("CPU neural fallback forbidden")
                samples += int(emb.shape[0])
                return predict(emb, act)

            model.predict = counted_predict
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            metrics = trainer._run_epoch(model=model, loader=itertools.islice(loader, 1), device=torch.device("cuda"),
                mean=mean, std=std, optimizer=optimizer, temperature=.05, batchnorm_policy=bn_policy)
            after = normalization.normalization_snapshot(model)
            changed = [k for k in before if not torch.equal(before[k], after[k])]
            gates = {"finite_loss": bool(np.isfinite(list(metrics.values())).all()),
                "optimizer_updated_parameters": shared._parameter_hash(model) != params_before,
                "visual_encoder_projector_unchanged": all(torch.equal(model.state_dict()[k], v) for k, v in visual.items()),
                "all_trainable_gradients_finite": all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable),
                "predict_sample_accounting": samples == 16,
                "declared_bn_behavior": not changed if bn_policy == "frozen-base-statistics" else bool(changed)}
            row = {"policy": bn_policy, "metrics": metrics, "changed_normalization_buffers": changed,
                "normalization_receipt": normalization.verify_normalization(model, before, bn_policy), "gates": gates,
                "actual_predict_samples": samples}
            rows.append(row)
            print(json.dumps(row), flush=True)
            model.predict = predict
            del optimizer, model
    finally:
        dataset.handle.close()
    gates = {"cuda_only": True, "both_policies_actual_training_step": len(rows) == 2,
        "all_integration_gates": all(all(r["gates"].values()) for r in rows),
        "trainer_unchanged": digest(Path(trainer.__file__)) == protocol["trainer_sha256"],
        "normalization_unchanged": digest(Path(normalization.__file__)) == protocol["normalization_sha256"],
        "base_checkpoint_unchanged": digest(policy / "weights.pt") == protocol["base_weights_sha256"]}
    write_new(args.output, {"schema": "reacher_d2p_training_normalization_guard.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "rows": rows, "gates": gates})
    print(json.dumps({"gates": gates}), flush=True)


if __name__ == "__main__":
    main()
