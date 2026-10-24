"""Audit whether a P1 proposal-pool selector survived a dynamics repair.

P1's selected packet is a deterministic function of the planning checkpoint,
the RGB/action-history root, and the frozen proposal bank.  A repair that
updates the predictor or action encoder can therefore change the deployment
packet law even if its damping value is unchanged.  This CUDA-only diagnostic
replays sealed source roots and compares their source selector indices to the
post-repair selector.  It is not a calibration or outcome experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.build_reacher_d2p_p1_proposal_pair_pool import _require_cuda, _sha256_file
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds
from experiments.spectral_reachability.reacher_d2p_context import collect_executed_context


SCHEMA = "reacher_d2p_p1_selector_stability_audit.v1"
POOL_SCHEMA = "reacher_d2p_p1_proposal_pair_pool.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_rows(manifest_path: Path) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != POOL_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("selector-stability source must be a completed P1 proposal pool")
    pool = Path(manifest.get("pairs_hdf5", "")).resolve()
    if not pool.is_file() or manifest.get("pairs_hdf5_sha256") != _sha256(pool):
        raise ValueError("selector-stability source pool differs from its manifest")
    contract = manifest.get("proposal_contract")
    required = {"history_len", "action_block", "damping", "proposals", "proposal_distribution", "selection_metric"}
    if not isinstance(contract, dict) or not required <= set(contract):
        raise ValueError("selector-stability source lacks its selector contract")
    with h5py.File(pool, "r") as handle:
        if not {"source_seed", "selected_proposal_index"} <= set(handle):
            raise ValueError("selector-stability source is missing sealed selected-index rows")
        seeds = np.asarray(handle["source_seed"], dtype=np.int64)
        indices = np.asarray(handle["selected_proposal_index"], dtype=np.int64)
    if len(seeds) < 2 or len(seeds) != len(indices) or len(seeds) != int(manifest.get("selected_pairs", -1)):
        raise ValueError("selector-stability source row count is invalid")
    if len(set(map(int, seeds))) != len(seeds):
        raise ValueError("selector-stability source contains duplicate root seeds")
    return manifest, [(int(seed), int(index)) for seed, index in zip(seeds, indices, strict=True)]


def audit(*, policy: Path, adapter_artifact: Path, source_manifest: Path, output: Path, device_name: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable selector-stability output: {output}")
    device = _require_cuda(device_name)
    policy, adapter_artifact, source_manifest = policy.resolve(), adapter_artifact.resolve(), source_manifest.resolve()
    manifest, source_rows = _source_rows(source_manifest)
    contract = manifest["proposal_contract"]
    if int(contract["history_len"]) != 3 or int(contract["action_block"]) != 5:
        raise ValueError("selector-stability source has incompatible Reacher context geometry")
    install_missing_field_compat()
    mean, std = _load_action_stats(adapter_artifact, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    try:
        action_dim = int(np.prod(env.action_space.shape))
        p0._validate_reacher_action_protocol(model, mean, std, action_dim=action_dim, action_block=int(contract["action_block"]))
        low, high = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
        rows: list[dict[str, Any]] = []
        for seed, source_index in source_rows:
            context = collect_executed_context(env, seed=seed, history_len=3, action_block=5)
            pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), device)
            _, receipt = p1._compile_reconfiguration(
                model, pixels, goal, history, reconfiguration_horizon=1, future_horizon=2,
                action_lower=lower, action_upper=upper, damping=float(contract["damping"]),
                proposals=int(contract["proposals"]), non_greedy_margin=1.0e-6,
                reachability_margin=1.0e-6, seed=seed,
                proposal_distribution=str(contract["proposal_distribution"]),
                selection_metric=str(contract["selection_metric"]),
            )
            selected = receipt.get("selected_proposal_index")
            rows.append({
                "seed": seed, "source_selected_proposal_index": source_index,
                "deployed_selected_proposal_index": selected,
                "selection_agrees": selected is not None and int(selected) == source_index,
                "deployed_non_greedy_witness_found": bool(receipt.get("non_greedy_witness_found", False)),
            })
    finally:
        env.close()
    agreement = sum(bool(row["selection_agrees"]) for row in rows)
    result = {
        "schema": SCHEMA, "status": "completed", "cuda_host": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "source_manifest": str(source_manifest), "source_manifest_sha256": _sha256(source_manifest),
        "source_policy_weights_sha256": contract.get("weights_sha256"), "deployed_policy": str(policy),
        "deployed_policy_weights_sha256": _sha256_file(policy / "weights.pt"),
        "selector_contract": {key: contract[key] for key in ("history_len", "action_block", "damping", "proposals", "proposal_distribution", "selection_metric")},
        "selection": {"roots": len(rows), "agreement_count": agreement, "agreement_rate": agreement / len(rows), "all_source_selections_stable": agreement == len(rows)},
        "rows": rows,
        "claim_boundary": "CUDA source-root selector-stability diagnostic only. It neither reuses nor opens P0/P1 calibration/outcome roots and makes no planning-efficacy claim.",
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = audit(policy=args.policy, adapter_artifact=args.adapter_artifact, source_manifest=args.source_manifest, output=args.output, device_name=args.device)
    print(json.dumps(result["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
