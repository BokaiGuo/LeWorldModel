"""CUDA replay audit for immutable Reacher D2P action witnesses.

This is deliberately not another planner or physical evaluation.  It reloads
the frozen checkpoint, reconstructs each recorded RGB/action context, and
recomputes the frozen terminal objective from the receipt's *recorded
normalized action witness*.  It then independently checks the corresponding
raw witness, action bounds, root RGB digests, and the recorded D2P--CEM model
cost ordering.  A mismatch is a code/provenance failure, not an efficacy
result.
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
from experiments.run_live_reacher_planning import (
    ReacherGoalImageWrapper,
    _load_action_stats,
    install_missing_field_compat,
)
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as runner


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_receipt(report: dict[str, Any]) -> None:
    if report.get("schema") != "d2p_reacher_p0_goal_progress.v1":
        raise ValueError("input is not a Reacher D2P P0 receipt")
    protocol, inputs = report.get("protocol", {}), report.get("inputs", {})
    if not protocol.get("outcome_only") or not protocol.get("future_suffix_only_optimized"):
        raise ValueError("witness replay requires an outcome-only, future-suffix receipt")
    if inputs.get("runner_sha256") != runner._sha256_file(Path(runner.__file__)):
        raise ValueError("receipt runner hash differs from the current replay adapter")
    if protocol.get("action_parameterization") != "tanh mapped to environment raw action box after checkpoint normalization":
        raise ValueError("receipt action parameterization is incompatible")
    if protocol.get("context_mode") != "executed_warmup":
        raise ValueError("witness replay requires an executed temporal context")
    if protocol.get("rollout_root_conditioning") != "observed_post_warmup_rgb_anchor":
        raise ValueError("witness replay requires an observed-root-anchored rollout")
    if protocol.get("execution_root_reconstruction") != "evaluator_only_observed_root_snapshot_restore":
        raise ValueError("witness replay requires an observed-root execution snapshot contract")
    if not protocol.get("warmup_policy"):
        raise ValueError("receipt does not declare the warmup policy")
    if not report.get("rows"):
        raise ValueError("receipt has no roots")


def _frozen_cost_from_normalized_witness(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    goal: torch.Tensor,
    history: torch.Tensor,
    normalized_future: torch.Tensor,
) -> torch.Tensor:
    """Replay a receipt suffix already mapped through the action ``tanh``.

    ``runner._cost_fn`` accepts unconstrained parameters and intentionally
    applies ``_bounded_normalized_actions``.  Receipts persist the *mapped*
    normalized witness, so replay must concatenate it directly rather than
    applying that mapping a second time.
    """

    if normalized_future.ndim != 2 or normalized_future.shape[-1] != history.shape[-1]:
        raise ValueError("normalized witness has incompatible shape")
    if pixels.shape[1] != history.shape[1] or pixels.shape[1] < 2:
        raise ValueError("root-anchored replay context is incompatible")
    encoded = runner._latent_context(model, pixels, history)
    action_context = torch.cat([history[:, 1:], normalized_future[:1][None]], dim=1)
    emb_context = encoded
    with torch.no_grad():
        terminal = None
        history_size = int(pixels.shape[1])
        for step in range(normalized_future.shape[0]):
            action_emb = model.action_encoder(action_context[:, -history_size:])
            terminal = model.predict(emb_context[:, -history_size:], action_emb)[:, -1]
            if step + 1 < normalized_future.shape[0]:
                emb_context = torch.cat([emb_context, terminal[:, None]], dim=1)
                action_context = torch.cat([action_context, normalized_future[step + 1:step + 2][None]], dim=1)
        if terminal is None or terminal.ndim != 2 or terminal.shape[0] != 1:
            raise RuntimeError("unexpected root-anchored replay terminal shape")
        return (terminal[0] - runner._goal_latent(model, goal)).square().mean()


def _model_cost_order_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Count a replayed D2P--CEM ordering without treating either sign as failure.

    The audit proves that stored action witnesses and frozen scores replay.  A
    particular score direction is an experimental observation, not an
    implementation invariant: after a valid D2P proposal-screen repair it is
    entirely possible that D2P scores lower on every root.  Requiring a CEM
    win would therefore turn the expected ordering into a spurious audit
    failure.
    """

    d2p_lower = cem_lower = ties = 0
    for row in rows:
        methods = row["methods"]
        d2p = float(methods["d2p_goal_progress"]["replayed_terminal_cost"])
        cem = float(methods["cem"]["replayed_terminal_cost"])
        if not np.isfinite(d2p) or not np.isfinite(cem):
            raise ValueError("replayed D2P/CEM terminal cost is non-finite")
        if d2p < cem:
            d2p_lower += 1
        elif cem < d2p:
            cem_lower += 1
        else:
            ties += 1
    return d2p_lower, cem_lower, ties


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.atol <= 0.0:
        raise ValueError("atol must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("witness replay requires host CUDA; CPU fallback is forbidden")

    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_receipt(report)
    protocol, inputs = report["protocol"], report["inputs"]
    policy, adapter = Path(inputs["checkpoint"]), Path(inputs["adapter_artifact"])
    if _digest(adapter) != inputs.get("adapter_sha256"):
        raise RuntimeError("adapter artifact hash differs from the immutable receipt")
    if _digest(policy / "weights.pt") != inputs.get("weights_sha256"):
        raise RuntimeError("checkpoint weights hash differs from the immutable receipt")

    install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if runner._parameter_hash(model) != report.get("parameter_hash_before"):
        raise RuntimeError("reloaded frozen parameter hash differs from the immutable receipt")
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    action_lower = torch.as_tensor(protocol["model_action_box_normalized"][0], device=device)
    action_upper = torch.as_tensor(protocol["model_action_box_normalized"][1], device=device)
    raw_low = np.asarray(protocol["environment_action_box_raw"][0], dtype=np.float32)
    raw_high = np.asarray(protocol["environment_action_box_raw"][1], dtype=np.float32)
    history_len = int(protocol["history_len"])
    future_horizon, action_block = int(protocol["future_horizon"]), int(protocol["action_block"])
    rows: list[dict[str, Any]] = []
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    try:
        for row in report["rows"]:
            seed = int(row["seed"])
            context = runner._root_context(env, seed, history_len, action_block)
            recorded_context = row.get("executed_context", {})
            expected_context = {
                key: context[key] for key in (
                    "context_mode", "warmup_policy", "warmup_steps",
                    "history_rgb_sha256", "root_rgb_sha256", "goal_rgb_sha256",
                    "warmup_actions_sha256",
                )
            }
            if recorded_context != expected_context:
                raise RuntimeError(f"seed {seed}: recorded temporal context does not replay")
            snapshot = row.get("execution_snapshot")
            if not isinstance(snapshot, dict):
                raise RuntimeError(f"seed {seed}: receipt has no evaluator-only execution snapshot")
            restored_root, restored_goal = runner.restore_execution_snapshot(
                env, seed=seed, snapshot=snapshot,
            )
            if runner._rgb_digest(restored_root) != recorded_context["root_rgb_sha256"]:
                raise RuntimeError(f"seed {seed}: execution snapshot does not reproduce recorded root RGB")
            if runner._rgb_digest(restored_goal) != recorded_context["goal_rgb_sha256"]:
                raise RuntimeError(f"seed {seed}: execution snapshot does not reproduce recorded goal RGB")
            goal = context["goal"]
            pixels, goal_tensor, history = runner._prepare(
                preprocessor, context["frames"], context["root_frame"], goal,
                np.asarray(context["history_action_blocks"], dtype=np.float32), device,
            )
            root_methods: dict[str, Any] = {}
            for name, recorded in row["methods"].items():
                normalized = np.asarray(recorded["action_witness_normalized"], dtype=np.float32)
                expected_shape = (future_horizon, action_lower.numel())
                if normalized.shape != expected_shape:
                    raise RuntimeError(f"seed {seed} {name}: normalized witness shape {normalized.shape} != {expected_shape}")
                normalized_tensor = torch.as_tensor(normalized, device=device)
                replay_cost = float(_frozen_cost_from_normalized_witness(
                    model, pixels, goal_tensor, history, normalized_tensor,
                ).detach())
                stored_cost = recorded.get("predicted_terminal_cost")
                cost_error = None if stored_cost is None else abs(replay_cost - float(stored_cost))
                raw = (normalized * np.asarray(std, dtype=np.float32).reshape(1, -1) + np.asarray(mean, dtype=np.float32).reshape(1, -1))
                raw = raw.reshape(future_horizon, -1, raw_low.size).reshape(-1, raw_low.size)
                stored_raw = np.asarray(recorded["action_witness_raw"], dtype=np.float32)
                raw_error = float(np.max(np.abs(raw - stored_raw)))
                identity = recorded["execution_identity"]
                identity_pass = (
                    identity.get("reset_rgb_sha256") == runner._rgb_digest(restored_root)
                    and identity.get("goal_rgb_sha256") == runner._rgb_digest(restored_goal)
                )
                root_methods[name] = {
                    "replayed_terminal_cost": replay_cost,
                    "stored_terminal_cost": stored_cost,
                    "terminal_cost_absolute_error": cost_error,
                    "raw_action_max_absolute_error": raw_error,
                    "raw_actions_within_environment_box": bool(
                        np.all(raw >= raw_low - args.atol) and np.all(raw <= raw_high + args.atol)
                    ),
                    "execution_identity_replayed": identity_pass,
                }
            rows.append({"seed": seed, "methods": root_methods})
    finally:
        env.close()

    compared = [row["methods"] for row in rows]
    all_costs_match = all(
        method["terminal_cost_absolute_error"] is None or method["terminal_cost_absolute_error"] <= args.atol
        for methods in compared for method in methods.values()
    )
    all_raw_match = all(method["raw_action_max_absolute_error"] <= args.atol for methods in compared for method in methods.values())
    all_boxes = all(method["raw_actions_within_environment_box"] for methods in compared for method in methods.values())
    all_identity = all(method["execution_identity_replayed"] for methods in compared for method in methods.values())
    d2p_lower, cem_lower, cost_ties = _model_cost_order_counts(rows)
    gates = {
        "cuda_only": True,
        "all_recorded_terminal_costs_replay_within_atol": all_costs_match,
        "all_normalized_to_raw_witnesses_replay_within_atol": all_raw_match,
        "all_witnesses_within_environment_action_box": all_boxes,
        "all_execution_rgb_identities_replay": all_identity,
        "replayed_d2p_cem_model_cost_order_is_finite": True,
    }
    output = {
        "schema": "d2p_reacher_p0_witness_replay_audit.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "frozen-model receipt replay only; no planner optimization, qpos access, or physical efficacy claim",
        "source_report": str(report_path),
        "source_report_sha256": _digest(report_path),
        "runner_sha256": runner._sha256_file(Path(runner.__file__)),
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "atol": args.atol,
        "gates": gates,
        "d2p_lower_model_cost_root_count": d2p_lower,
        "cem_lower_model_cost_root_count": cem_lower,
        "model_cost_tie_root_count": cost_ties,
        "root_count": len(rows),
        "rows": rows,
        "claim_boundary": "A passing audit excludes stale terminal costs, normalized/raw action reshaping, action-box, and root-identity mismatches as explanations of this receipt. It does not establish D2P efficacy or a theory claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"], "gates": gates,
        "d2p_lower_model_cost_root_count": d2p_lower,
        "cem_lower_model_cost_root_count": cem_lower,
        "model_cost_tie_root_count": cost_ties,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
