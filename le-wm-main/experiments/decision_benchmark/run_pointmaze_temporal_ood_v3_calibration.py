"""Run v3's frozen calibration-only temporal authorization decision.

V3 never imports outcome labels into calibration.  The predeclared selector is
``history`` iff the calibration paired-regret bootstrap lower interval is
strictly positive; otherwise the action-only fallback is selected.  A frozen
offline package is materialized only after that gate *and* the untouched v3
outcome interval are positive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import ACTION_OOD_FAMILY
from experiments.decision_benchmark.build_pointmaze_temporal_ood_v3_panel import SCHEMA as V3_BANK_SCHEMA, V3_ACTION_OOD_FAMILY
from experiments.decision_benchmark.run_pointmaze_temporal_ood_test import (
    FEATURES,
    _feature,
    _training_features,
    run as run_four_arm_test,
)
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _load_records
from experiments.decision_benchmark.train_candidate_effect_adapter import SEEDS, fit_effect_model
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_temporal_ood_v3_calibration.v1"


def _compatibility_view(source: Path, output: Path) -> Path:
    payload = json.loads(source.read_text())
    metadata = dict(payload["metadata"])
    if metadata.get("candidate_family") != V3_ACTION_OOD_FAMILY:
        raise ValueError("v3 context has an unexpected candidate family")
    metadata.update({"underlying_candidate_family": V3_ACTION_OOD_FAMILY, "candidate_family": ACTION_OOD_FAMILY,
        "evaluator_compatibility": "metadata-only v3->shared-four-arm evaluator bridge"})
    path = output / "temporal_ood_v3_evaluator_compatible_context.json"
    path.write_text(json.dumps({"metadata": metadata, "records": payload["records"]}, indent=2, sort_keys=True) + "\n")
    return path


def _freeze_history_ensemble(source_dataset: Path, source_arrays: Path, output: Path, *, epochs: int) -> dict[str, Any]:
    """Serialize the train-only history ensemble after authorization, verbatim."""

    _metadata, records = _load_records(source_dataset)
    arrays = np.load(source_arrays, allow_pickle=False)
    train_records = [row for row in records if str(row["split"]) == "train"]
    x, y = _training_features(train_records, arrays, "history_latent_action")
    package = output / "frozen_temporal_policy_v3"
    models = package / "models"
    models.mkdir(parents=True, exist_ok=False)
    manifests = []
    for seed in SEEDS:
        model, normalization = fit_effect_model(x, y, seed=seed, epochs=epochs, device_name="cuda")
        path = models / f"history_effect_seed_{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "normalization": normalization, "feature_family": "history_latent_action"}, path)
        manifests.append({"seed": seed, "path": str(path.resolve()), "input_dim": normalization["input_dim"]})
    package_manifest = {"schema": "frozen_temporal_policy_v3.v1", "selected_family": "history_latent_action",
        "selection_source": "v3 calibration only", "deployment_status": "offline_only_not_live_deployment",
        "feature_contract": "four causal renderer-backed frozen latents plus actions on [t-15,t)", "ensemble": manifests}
    (package / "manifest.json").write_text(json.dumps(package_manifest, indent=2, sort_keys=True) + "\n")
    return {"path": str(package.resolve()), "manifest": package_manifest}


def run(source_dataset: Path, source_arrays: Path, temporal_context: Path, temporal_outcomes: Path, output: Path, *, epochs: int = 300, bootstrap_samples: int = 5000, device: str = "cuda") -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal v3 decision requires CUDA")
    cuda = require_cuda_allocation(device)
    metadata, records = _load_records(temporal_context)
    if metadata.get("schema") != V3_BANK_SCHEMA or metadata.get("renderer_backed_frozen_observation_control") is not True:
        raise ValueError("v3 runner requires the declared renderer-backed v3 context")
    counts = {name: len({str(row["instance_id"]) for row in records if str(row["split"]) == name}) for name in ("train", "calibration", "outcome")}
    if counts["calibration"] < 40:
        raise ValueError("v3 requires at least 40 calibration roots")
    if counts != {"train": 64, "calibration": 48, "outcome": 32}:
        raise ValueError(f"v3 root counts drifted from predeclared panel: {counts}")
    output.mkdir(parents=True, exist_ok=True)
    compatible = _compatibility_view(temporal_context, output)
    report = run_four_arm_test(source_dataset, source_arrays, compatible, temporal_outcomes, output, epochs=epochs, bootstrap_samples=bootstrap_samples, device=device)
    calibration = report["calibration_only_history_policy"]
    outcome = report["paired_outcome_regret_gain_over_action_only"]["history_latent_action"]
    gate_pass = calibration["decision"] == "history_permitted"
    outcome_pass = float(outcome["lower"]) > 0.0
    # The sole selected policy is fixed by calibration.  Outcome labels cannot
    # switch it; they only audit the independently selected policy.
    decision = "history" if gate_pass else "action_only"
    package = _freeze_history_ensemble(source_dataset, source_arrays, output, epochs=epochs) if gate_pass and outcome_pass else None
    result = {"schema": SCHEMA, "cuda_preflight": cuda, "native_context": str(temporal_context.resolve()),
        "compatible_context": str(compatible.resolve()), "four_arm_report": str((output / "temporal_ood_report.json").resolve()),
        "predeclared_selection": {"comparison": "history_latent_action vs action_only", "authorization": "calibration paired-regret bootstrap lower CI > 0", "selected_policy": decision, "gate_passed": gate_pass,
            "calibration_roots_only": counts["calibration"], "prior_outcome_labels_used_in_calibration": False},
        "held_out_outcome_audit": {"history_vs_action_only": outcome, "outcome_positive": outcome_pass, "outcome_roots": counts["outcome"]},
        "policy_status": "frozen_temporal_policy_created_offline_only" if package else "no_frozen_temporal_policy",
        "policy_package": package,
        "four_arms": list(FEATURES),
        "claim_boundary": "A successful package is an offline renderer-backed PointMaze action-OOD artifact, not a live visual deployment claim."}
    (output / "temporal_ood_v3_calibration_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "temporal_ood_v3_policy_status.json").write_text(json.dumps({"policy_status": result["policy_status"], "policy_package": package, "reason": "requires both predeclared v3 calibration authorization and independent v3 outcome lower CI > 0"}, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-arrays", type=Path, required=True)
    parser.add_argument("--temporal-context", type=Path, required=True)
    parser.add_argument("--temporal-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
