"""Fresh CUDA physical diagnostic for source-supported Reacher P1 packets.

Both the causal warmup and the nonzero first-prefix bank are drawn from finite
consecutive official training packets.  This tests a protocol repair only; it
is deliberately not a sealed P1 calibration or outcome receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds
from experiments.spectral_reachability.reacher_d2p_context import collect_executed_context
from experiments.spectral_reachability import reacher_d2p_context as context_runtime
from experiments.spectral_reachability.audit_reacher_d2p_runtime_support import _source_state_context
from experiments.spectral_reachability.reacher_d2p_packet_support import WARMUP_POLICY, load_history_packet_pool, load_packet_pool, proposal_parameters, receipt_source, sample_history_packets
from experiments.spectral_reachability.reacher_d2p_qpos_readout import PREDICTION_READOUT_TASK, ReacherQposReadoutCostModel, load_qpos_readout
from experiments.spectral_reachability.reacher_d2p_reconfiguration_reference import FUTURE_REFERENCES
from experiments.spectral_reachability.reacher_d2p_packet_support import SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _disjoint_reference_binding(paths: list[Path], seeds: set[int]) -> tuple[set[int], list[dict]]:
    episodes, bindings = set(), []
    for path in paths:
        report = json.loads(path.read_text())
        rows = report.get("rows", [])
        if report.get("status") != "completed" or not rows or any(row.get("source_episode") is None for row in rows):
            raise ValueError("completed source-episode reference required")
        if seeds & {int(row["seed"]) for row in rows}:
            raise ValueError("confirmation seeds overlap reference roots")
        episodes.update(int(row["source_episode"]) for row in rows)
        bindings.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return episodes, bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--qpos-readout", type=Path, default=None)
    parser.add_argument("--predicted-qpos-readout", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026102520)
    parser.add_argument("--proposals", type=int, default=128)
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--root-context", choices=("executed_source_history", "official_source_state_history"), default="executed_source_history")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--future-reference", choices=FUTURE_REFERENCES, default="zero_prefix")
    parser.add_argument("--disjoint-reference", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("source-supported P1 audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.roots < 4 or args.proposals < 2 or args.output.exists():
        raise ValueError("roots >= 4, proposals >= 2, and a new output are required")
    policy, adapter, output = args.policy.resolve(), args.adapter_artifact.resolve(), args.output.resolve()
    qpos_readout = args.qpos_readout.resolve() if args.qpos_readout is not None else None
    predicted_qpos_readout = args.predicted_qpos_readout.resolve() if args.predicted_qpos_readout is not None else None
    if (qpos_readout is None) != (predicted_qpos_readout is None):
        raise ValueError("encoder and predicted qpos readouts must be supplied together")
    if args.disjoint_reference and args.root_context != "official_source_state_history":
        raise ValueError("episode-disjoint confirmation requires official source-state roots")
    excluded_episodes, reference_bindings = _disjoint_reference_binding(
        args.disjoint_reference, set(range(args.seed, args.seed + args.roots)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sealed_protocol = {
        "root_seeds": list(range(args.seed, args.seed + args.roots)),
        "root_context": args.root_context, "proposals": args.proposals,
        "damping": args.damping, "future_reference": args.future_reference,
        "recompiled_suffix_solver": p1.SUFFIX_SOLVER,
        "weights_sha256": _sha256(policy / "weights.pt"),
        "adapter_sha256": _sha256(adapter),
        "selector_sha256": _sha256(Path(p1.__file__)),
        "runner_sha256": _sha256(Path(runner.__file__)),
        "context_module_sha256": _sha256(Path(context_runtime.__file__)),
        "auditor_sha256": _sha256(Path(__file__)),
        "disjoint_reference_bindings": reference_bindings,
        "excluded_source_episodes": sorted(excluded_episodes),
        "claim_boundary": "Diagnostic only, not calibration, outcome, or resource-matched efficacy.",
    }
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(sealed_protocol, indent=2, sort_keys=True) + "\n")
    root_records = output.with_suffix(".roots")
    root_records.mkdir(exist_ok=False)
    def write_root(row):
        with (root_records / f"root_{row['seed']}.json").open("x") as handle:
            handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"seed": row["seed"], "source_episode": row["source_episode"],
            "eligible": row["reconfiguration"] is not None,
            "selected_final_qpos_distance": None if row["reconfiguration"] is None else row["reconfiguration"]["physical"]["final_qpos_distance"],
            "greedy_final_qpos_distance": row["same_bank_greedy"]["physical"]["final_qpos_distance"]}), flush=True)
    print("Testing the fixed backtracking solver on every declared root; diagnostic, not formal P1 efficacy.", flush=True)
    install_missing_field_compat(); device = torch.device("cuda"); torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    dataset, digest = receipt_source(adapter)
    pool, support = load_packet_pool(dataset, expected_sha256=digest)
    histories, history_support = load_history_packet_pool(dataset, expected_sha256=digest, history_len=3, verify_source=False)
    model = load_compatible_pretrained(policy, device=device).eval(); model.requires_grad_(False); model.interpolate_pos_encoding = True
    if qpos_readout is not None:
        model = ReacherQposReadoutCostModel(
            model, load_qpos_readout(qpos_readout),
            load_qpos_readout(predicted_qpos_readout, expected_task=PREDICTION_READOUT_TASK),
        ).to(device).eval()
    hash_before = p1._parameter_hash(model)
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    lower_np, upper_np = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std)); rows: list[dict[str, Any]] = []
    seen_episodes = set()
    try:
        for offset in range(args.roots):
            seed = int(args.seed + offset)
            if args.root_context == "official_source_state_history":
                context, warmup_index = _source_state_context(env, dataset=dataset, seed=seed), None
                episode = context["source_episode"]
                if args.disjoint_reference and (episode in excluded_episodes or episode in seen_episodes):
                    raise RuntimeError("source-episode overlap; no resampling or root removal permitted")
                seen_episodes.add(episode)
            else:
                packets, warmup_index = sample_history_packets(histories, seed=seed)
                context = collect_executed_context(env, seed=seed, history_len=3, action_block=5, warmup_packets=packets, warmup_policy=WARMUP_POLICY)
            bank, bank_receipt = proposal_parameters(pool, proposals=args.proposals, horizon=1, seed=seed, mean=mean, std=std, lower=lower, upper=upper)
            selection_receipts: list[dict[str, Any]] = []
            def selected_program(pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor, root_seed: int):
                params, receipt = p1._compile_reconfiguration(model, pixels, goal, history, reconfiguration_horizon=1, future_horizon=2, action_lower=lower, action_upper=upper, damping=args.damping, proposals=args.proposals, non_greedy_margin=1.0e-6, reachability_margin=0.0, seed=root_seed, proposal_bank=bank, future_reference=args.future_reference, proposal_distribution=SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION)
                selection_receipts.append(receipt)
                if params is None:
                    raise RuntimeError("source-supported P1 bank produced no eligible non-greedy prefix")
                return params, receipt
            def zero_program(_pixels: torch.Tensor, _goal: torch.Tensor, history: torch.Tensor, _root_seed: int):
                return torch.zeros((1, 1, history.shape[-1]), device=device), {"program_type": "physical_zero_prefix_control"}
            def source_matched_greedy_program(
                pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor, _root_seed: int,
            ):
                return runner._source_matched_greedy_program(
                    model, pixels, goal, history, proposal_bank=bank,
                    action_lower=lower, action_upper=upper,
                )
            zero = runner._two_stage(name="source_supported_zero", first_program=zero_program, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=5, progress_horizon=2, damping=args.damping, device=device)
            greedy = runner._two_stage(name="source_supported_same_bank_greedy", first_program=source_matched_greedy_program, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=5, progress_horizon=2, damping=args.damping, device=device)
            try:
                selected = runner._two_stage(name="source_supported_reconfiguration", first_program=selected_program, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=5, progress_horizon=2, damping=args.damping, device=device)
            except RuntimeError as error:
                if "no eligible non-greedy prefix" not in str(error):
                    raise
                rows.append({"seed": seed, "root_context": context["context_mode"], "root_warmup_policy": context["warmup_policy"], "source_episode": context.get("source_episode"), "source_start_row": context.get("source_start_row"), "contiguous_warmup_sequence_index": warmup_index, "proposal_support": bank_receipt, "reconfiguration": None, "zero_prefix": zero, "same_bank_greedy": greedy, "ineligible_reason": str(error)})
                rows[-1]["model_selection"] = selection_receipts[-1]
                write_root(rows[-1])
                continue
            rows.append({"seed": seed, "root_context": context["context_mode"], "root_warmup_policy": context["warmup_policy"], "source_episode": context.get("source_episode"), "source_start_row": context.get("source_start_row"), "contiguous_warmup_sequence_index": warmup_index, "proposal_support": bank_receipt, "reconfiguration": selected, "zero_prefix": zero, "same_bank_greedy": greedy, "ineligible_reason": None})
            rows[-1]["model_selection"] = selection_receipts[-1]
            write_root(rows[-1])
    finally:
        env.close()
    eligible = [row for row in rows if row["reconfiguration"] is not None]
    observed = np.asarray([row["reconfiguration"]["observed_first_chunk"]["candidate_goal_cost"] - row["zero_prefix"]["observed_first_chunk"]["candidate_goal_cost"] for row in eligible], dtype=np.float64)
    physical = np.asarray([row["reconfiguration"]["physical"]["after_first_chunk_qpos_distance"] - row["zero_prefix"]["physical"]["after_first_chunk_qpos_distance"] for row in eligible], dtype=np.float64)
    greedy_observed = np.asarray([row["reconfiguration"]["observed_first_chunk"]["candidate_goal_cost"] - row["same_bank_greedy"]["observed_first_chunk"]["candidate_goal_cost"] for row in eligible], dtype=np.float64)
    greedy_physical = np.asarray([row["reconfiguration"]["physical"]["after_first_chunk_qpos_distance"] - row["same_bank_greedy"]["physical"]["after_first_chunk_qpos_distance"] for row in eligible], dtype=np.float64)
    result: dict[str, Any] = {"schema": "reacher_d2p_source_supported_p1_diagnostic.v5", "status": "completed", "scope": "fresh CUDA physical diagnostic of a source-supported runtime repair; not a sealed P1 calibration or outcome receipt", "inputs": {"policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"), "adapter": str(adapter), "adapter_sha256": _sha256(adapter), "qpos_readout": str(qpos_readout) if qpos_readout is not None else None, "qpos_readout_sha256": _sha256(qpos_readout) if qpos_readout is not None else None, "predicted_qpos_readout": str(predicted_qpos_readout) if predicted_qpos_readout is not None else None, "predicted_qpos_readout_sha256": _sha256(predicted_qpos_readout) if predicted_qpos_readout is not None else None, "source_packet_support": support, "contiguous_warmup_support": history_support}, "protocol": {"device": str(device), "roots": int(args.roots), "root_seeds": [row["seed"] for row in rows], "root_context": args.root_context, "proposal_distribution": "source_locked_training_packet_support_with_raw_zero_at_index_0", "comparison_control": "same-bank frozen-model immediate-greedy first packet", "proposals": int(args.proposals), "damping": float(args.damping), "selection_goal_coordinate": "separate_source_calibrated_encoder_and_prediction_qpos" if qpos_readout is not None else "frozen_encoder_l2"}, "aggregate": {"eligible_root_count": int(len(eligible)), "ineligible_root_count": int(len(rows) - len(eligible)), "observed_selected_minus_zero_goal_cost": observed.tolist(), "physical_selected_minus_zero_qpos_distance": physical.tolist(), "observed_non_greedy_every_eligible_root": bool(len(eligible) and np.all(observed >= -1.0e-8)), "physical_non_greedy_every_eligible_root": bool(len(eligible) and np.all(physical >= -1.0e-8)), "model_and_physical_order_agree_every_eligible_root": bool(len(eligible) and np.all((observed >= 0.0) == (physical >= 0.0))), "observed_selected_minus_same_bank_greedy_goal_cost": greedy_observed.tolist(), "physical_selected_minus_same_bank_greedy_qpos_distance": greedy_physical.tolist(), "same_bank_observed_non_greedy_every_eligible_root": bool(len(eligible) and np.all(greedy_observed >= -1.0e-8)), "same_bank_physical_non_greedy_every_eligible_root": bool(len(eligible) and np.all(greedy_physical >= -1.0e-8)), "same_bank_model_and_physical_order_agree_every_eligible_root": bool(len(eligible) and np.all((greedy_observed >= 0.0) == (greedy_physical >= 0.0)))}, "rows": rows}
    result["protocol"]["future_reference"] = args.future_reference
    result["protocol"]["recompiled_suffix_solver"] = p1.SUFFIX_SOLVER
    result["sealed_protocol"] = sealed_protocol
    result["gates"] = {"cuda_only": str(device) == "cuda", "weights_unchanged": hash_before == p1._parameter_hash(model),
                       "all_declared_roots_retained": [row["seed"] for row in rows] == sealed_protocol["root_seeds"]}
    if args.disjoint_reference:
        result["gates"]["reference_episodes_disjoint"] = not bool(excluded_episodes & seen_episodes)
        result["gates"]["unique_confirmation_episodes"] = len(seen_episodes) == args.roots
    result["cuda_host"] = {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda}
    if not all(result["gates"].values()):
        result["status"] = "fail"
    result["inputs"]["auditor_sha256"] = _sha256(Path(__file__))
    result["inputs"]["selector_sha256"] = _sha256(Path(p1.__file__))
    result["claim_boundary"] = "Diagnostic only. Same-bank future-reference rejection is an abstention, not a passed P1 or evidence of efficacy."
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
