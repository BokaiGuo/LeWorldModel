"""Paired physical development follow-up to a sealed model-only selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_p1_predicted_successor_cem as selector
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner


def bound_indices(row: dict, bank: torch.Tensor) -> dict[str, int]:
    """Bind stored selection and same-bank greedy without looking at outcomes."""
    selection = row["selection"]
    records = selection["proposal_rows"]
    if len(records) != len(bank) or [r["proposal_index"] for r in records] != list(range(len(bank))):
        raise ValueError("proposal rows do not match reconstructed bank")
    costs = np.asarray([r["immediate_goal_cost"] for r in records], dtype=np.float64)
    if not np.isfinite(costs).all():
        raise ValueError("non-finite saved immediate cost")
    selected = selection["selected_proposal_index"]
    if selected is None:
        if row["selected_params"] is not None or selection["eligible_count"] != 0:
            raise ValueError("inconsistent ineligible selection")
        selected = 0
    else:
        if not isinstance(selected, int) or not 0 <= selected < len(bank):
            raise ValueError("invalid selected index")
        saved = torch.as_tensor(row["selected_params"], dtype=bank.dtype, device=bank.device)
        if saved.shape != bank[selected:selected + 1].shape or not torch.equal(saved, bank[selected:selected + 1]):
            raise ValueError("selected action does not match reconstructed bank")
    return {"selected": selected, "zero": 0, "same_bank_greedy": int(np.argmin(costs))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite physical development output")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required; no CPU fallback")
    source = json.loads(args.source.read_text())
    protocol = source["protocol"]
    policy, stats = Path(protocol["policy"]), Path(protocol["action_stats"])
    for key, path in (("weights_sha256", policy / "weights.pt"),
                      ("action_stats_sha256", stats),
                      ("selector_sha256", Path(selector.__file__))):
        if setup._sha256_file(path) != source["inputs"][key]:
            raise RuntimeError(f"sealed input mismatch: {key}")
    if source["frozen_parameters_unchanged"] is not True:
        raise RuntimeError("source model was not frozen")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    pool, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != source["parameter_hash_before"]:
        raise RuntimeError("model parameter hash differs from sealed selection")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65),
    )
    rows = []
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for source_row in source["rows"]:
            seed = source_row["seed"]
            packets, history_index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history",
            )
            bank, bank_receipt = setup.proposal_parameters(
                pool, proposals=protocol["proposals"], horizon=1, seed=seed,
                mean=mean, std=std, lower=lower, upper=upper,
            )
            if (int(history_index) != source_row["history_index"]
                    or context["root_rgb_sha256"] != source_row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != source_row["goal_rgb_sha256"]
                    or bank_receipt != source_row["bank_receipt"]):
                raise RuntimeError(f"sealed root, goal, history or bank mismatch at seed {seed}")
            indices = bound_indices(source_row, bank)
            methods = {}
            for name, index in indices.items():
                def fixed_first(pixels, goal, history, root_seed, *, candidate=index):
                    return bank[candidate:candidate + 1].detach(), {
                        "selection": "sealed model report", "proposal_index": candidate,
                        "ineligible_zero_fallback": name == "selected" and source_row["selected_params"] is None,
                    }
                methods[name] = runner._two_stage(
                    name=name, first_program=fixed_first, env=env, seed=seed, context=context,
                    preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                    action_block=5, progress_horizon=2, damping=0.0, device=torch.device("cuda"),
                    progress_solver="cem", cem_samples=protocol["samples"],
                    cem_steps=protocol["steps"], cem_elite=protocol["elite"],
                )
                print(json.dumps({"seed": seed, "arm": name, "physical": methods[name]["physical"]}), flush=True)
            rows.append({"seed": seed, "eligible": source_row["selected_params"] is not None,
                         "indices": indices, "methods": methods})
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen parameters changed")
    report = {
        "scope": "paired development diagnostic, not PC2/PC3 admission or independent confirmation",
        "candidate_origin": "source action packets, not spectral programs",
        "ineligible_rule": "selected uses neutral-parameter prefix; all roots retained",
        "selection_source_sha256": setup._sha256_file(args.source),
        "script_sha256": setup._sha256_file(Path(__file__)),
        "model_parameter_sha256_before": before, "model_parameter_sha256_after": after,
        "feedback": "each arm replans from its own observed RGB and executed action history",
        "cost_scope": "source selector and suffix costs recorded separately; full resource fairness not certified",
        "mechanism_scope": "independent suffix optimization only; no common-suffix intervention",
        "training_root_independence": "unknown",
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
