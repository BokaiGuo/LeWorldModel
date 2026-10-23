"""Build a support-expanded, runtime-matched PointMaze effect policy.

The previous online policy had one source feature per train root.  That is an
appropriate fixed-bank *ranking* table, but it is not a deployment support
set: a receding-horizon controller visits new local states after its first
block.  This builder adds only causal pre-action prefixes and one-block pulse
counterfactuals from declared train/effect-calibration roots.  It never opens
the visual-gate or held-out planning roots.

The generated policy keeps the existing ``OnlineEffectPolicy`` serialization
contract so the unmodified strict RGB planner is its sole deployment surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from experiments.decision_benchmark.fit_pointmaze_online_effect_policy import (
    SCHEMA,
    OnlineEffectPolicy,
    build as fit_online_effect_policy,
    fit_nearest_support,
    support_distance,
)
from experiments.decision_benchmark.task_contrast import calibrate_root_score_bound
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    action_sequence_basis,
    predict_effect_model,
)
from experiments.run_live_pointmaze_planning import (
    FRAME_STRIDE,
    GOAL_STATE,
    DatasetMatchedRenderer,
    LegacyPointMazeRuntime,
    _make_context,
    _make_gdd_source_contract_context,
    _pool_visual,
    _load_jepa_model,
    action_bank,
    fit_visual_calibration,
    green_center,
    load_episode,
    load_episode_states,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = tuple(range(1800, 1854))
DEFAULT_EFFECT_CALIBRATION = tuple(range(1854, 1868))
FORBIDDEN_RUNTIME_ROOTS = frozenset(range(1868, 1900))


def parse_roots(value: str, label: str) -> list[int]:
    roots = [int(token.strip()) for token in str(value).split(",") if token.strip()]
    if not roots or len(roots) != len(set(roots)):
        raise ValueError(f"{label} must be a non-empty duplicate-free integer list")
    return roots


def validate_collection_partition(
    train_roots: Iterable[int], effect_calibration_roots: Iterable[int],
) -> None:
    """Reject any accidental use of visual-gate or held-out roots."""

    train = set(map(int, train_roots))
    calibration = set(map(int, effect_calibration_roots))
    if not train or not calibration:
        raise ValueError("support expansion needs non-empty train and effect-calibration roots")
    if train & calibration:
        raise ValueError("train and effect-calibration roots must be disjoint")
    forbidden = (train | calibration) & FORBIDDEN_RUNTIME_ROOTS
    if forbidden:
        raise ValueError(
            "support expansion may not read visual-gate or held-out roots: "
            f"{sorted(forbidden)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefix_actions(current: np.ndarray, root: int, prefix_count: int) -> list[np.ndarray]:
    """Deterministic causal exploration prefixes from the live pulse family."""

    actions: list[np.ndarray] = []
    state = np.asarray(current, dtype=np.float64).copy()
    # The same goal-relative pulse parameterization and magnitude envelope as
    # the strict live bank.  The deterministic index prevents task-outcome
    # selection while covering direction changes across roots and prefixes.
    for depth in range(int(prefix_count)):
        bank = action_bank(state[:2], GOAL_STATE, directions=8, magnitudes=[0.25, 0.5, 0.75, 1.0])
        actions.append(np.asarray(bank[(int(root) * 11 + depth * 7) % len(bank)], dtype=np.float64))
        state = state  # explicit: dynamics advances outside this pure scheduler
    return actions


def _context_proprio(
    renderer: DatasetMatchedRenderer,
    calibration: Any,
    previous_state: np.ndarray,
    current_state: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Produce the RGB-derived deployment chart; state only drives rendering."""

    previous_image = renderer.render(previous_state)
    current_image = renderer.render(current_state)
    previous_center = green_center(previous_image)
    current_center = green_center(current_image)
    previous_xy = calibration.position(previous_center)
    current_xy = calibration.position(current_center)
    previous_proprio = np.concatenate([previous_xy, np.zeros(2, dtype=np.float64)])
    current_proprio = np.concatenate([
        current_xy,
        calibration.velocity((current_center[0] - previous_center[0], current_center[1] - previous_center[1])),
    ])
    return [previous_image, current_image], [previous_proprio, current_proprio]


def _encode_projection(
    model: Any,
    transform: Any,
    pca_mean: np.ndarray,
    pca_basis: np.ndarray,
    images: list[np.ndarray],
    proprios: list[np.ndarray],
) -> np.ndarray:
    context = _make_context(images, proprios)
    context = _make_gdd_source_contract_context(context, transform)
    with torch.inference_mode():
        encoded = model.encode(context)
        pooled = _pool_visual(encoded["visual"][:, -1]).detach().cpu().numpy()[0]
    return (np.asarray(pooled, dtype=np.float64) - pca_mean) @ pca_basis.T


def _pulse_rows_for_prefix(
    *,
    root: int,
    split: str,
    prefix: int,
    source_state: np.ndarray,
    source_projection: np.ndarray,
    current_xy: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Counterfactually execute the *same* 32 live one-block pulses."""

    bank = action_bank(current_xy, GOAL_STATE, directions=8, magnitudes=[0.25, 0.5, 0.75, 1.0])
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for candidate_id, action in enumerate(bank):
        runtime = LegacyPointMazeRuntime(source_state)
        terminal = runtime.step_block(action, steps=FRAME_STRIDE)
        key = f"realized_state::support_episode_{root}_prefix_{prefix:02d}/{candidate_id:03d}"
        arrays[key] = np.asarray(terminal, dtype=np.float64)
        terminal_cost = float(np.linalg.norm(np.asarray(terminal[:2]) - GOAL_STATE))
        rows.append({
            "action_sequence": np.repeat(np.asarray(action, dtype=np.float64)[None], FRAME_STRIDE, axis=0).tolist(),
            "candidate_id": int(candidate_id),
            "candidate_generator": "runtime_one_block_pulse_v1",
            "candidate_bank_coordinate_frame": "goal_relative",
            "candidate_bank_hash": hashlib.sha256(np.asarray(bank, dtype=np.float32).tobytes()).hexdigest()[:20],
            "cem_stage": "support_expanded_local_prefix_counterfactual",
            "context_length": 2,
            "environment": "PointMaze",
            "goal_id": "legacy_pointmaze_goal_1p2_1p2",
            "goal_raw": GOAL_STATE.tolist(),
            "instance_id": f"pointmaze_episode_{root}",
            "prefix_id": int(prefix),
            "prefix_state_role": "causal_pre_action_runtime_state",
            "source_frame": int(prefix * FRAME_STRIDE),
            "source_latent_model": np.asarray(source_projection, dtype=np.float64).tolist(),
            "source_history_model": np.asarray(source_projection, dtype=np.float64).tolist(),
            "source_state_raw": np.asarray(source_state, dtype=np.float64).tolist(),
            "realized_terminal_state_ref": f"arrays.npz::{key}",
            "real_task_cost": terminal_cost,
            "real_success": bool(terminal_cost <= 0.25),
            "split": split,
            "start_episode_id": str(root),
            "total_action_steps": FRAME_STRIDE,
            "action_support": {"chart": "raw_control", "duration_steps": FRAME_STRIDE},
            "privileged_state_role": "offline_support_expansion_label_only",
        })
    return rows, arrays


def _calibrate_expanded_gate(
    *,
    policy_path: Path,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    train_roots: list[int],
    calibration_roots: list[int],
    alpha: float,
) -> dict[str, Any]:
    """Replace initial-state support by an independently calibrated prefix gate."""

    payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected fitted policy serialization")
    train_set, calibration_set = set(train_roots), set(calibration_roots)
    def root_of(row: dict[str, Any]) -> int:
        return int(row["start_episode_id"])

    train_states = np.stack([
        np.asarray(row["source_latent_model"], dtype=np.float64)
        for row in rows if root_of(row) in train_set
    ])
    train_bases = np.stack([
        action_sequence_basis(np.asarray(row["action_sequence"], dtype=np.float64))
        for row in rows if root_of(row) in train_set
    ])
    latent_support = fit_nearest_support(train_states, quantile=0.98)
    action_support = fit_nearest_support(train_bases, quantile=0.98)
    train_latent_threshold = float(latent_support["threshold"])
    train_action_threshold = float(action_support["threshold"])
    calibration_states = [
        np.asarray(row["source_latent_model"], dtype=np.float64)
        for row in rows if root_of(row) in calibration_set
    ]
    calibration_bases = [
        action_sequence_basis(np.asarray(row["action_sequence"], dtype=np.float64))
        for row in rows if root_of(row) in calibration_set
    ]
    calibration_latent_distances = np.asarray([
        support_distance(latent_support, value) for value in calibration_states
    ])
    calibration_action_distances = np.asarray([
        support_distance(action_support, value) for value in calibration_bases
    ])
    # The effect-calibration set alone chooses the deployment radii.  Use a
    # fixed 90% marginal target, never a held-out planning outcome.
    latent_support["threshold"] = float(np.quantile(calibration_latent_distances, 0.90, method="higher"))
    action_support["threshold"] = float(np.quantile(calibration_action_distances, 0.90, method="higher"))

    policy = OnlineEffectPolicy.load(policy_path, torch.device("cpu"))
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if root_of(row) in calibration_set:
            grouped[(root_of(row), int(row["prefix_id"]))].append(row)
    root_risks: dict[int, list[float]] = defaultdict(list)
    for (root, _prefix), group in grouped.items():
        source = np.asarray(group[0]["source_latent_model"], dtype=np.float64)
        source_xy = np.asarray(group[0]["source_state_raw"], dtype=np.float64)[:2]
        actions = np.asarray([row["action_sequence"][0] for row in group], dtype=np.float64)
        report = policy.score(source, source_xy, GOAL_STATE, actions)
        selected = int(np.argmin(np.asarray(report["costs"], dtype=np.float64)))
        costs = np.asarray([float(row["real_task_cost"]) for row in group], dtype=np.float64)
        supported = (
            support_distance(latent_support, source) <= float(latent_support["threshold"])
            and np.all(np.asarray(report["action_support_distances"]) <= float(action_support["threshold"]) + 1.0e-12)
        )
        if supported:
            root_risks[root].append(float(costs[selected] - costs.min()))
    if set(root_risks) != calibration_set:
        missing = sorted(calibration_set - set(root_risks))
        raise ValueError(f"support gate rejected every calibration prefix for roots {missing}")
    per_root = np.asarray([max(root_risks[root]) for root in sorted(calibration_set)], dtype=np.float64)
    regret_calibration = calibrate_root_score_bound(per_root, alpha=alpha)
    payload["latent_support"] = latent_support
    payload["action_support"] = action_support
    payload["regret_calibration"] = regret_calibration
    payload["support_expansion"] = {
        "schema": "pointmaze.support_expanded_gate.v1",
        "source": "declared_train_and_effect_calibration_prefixes_only",
        "prefix_collection": "causal_pre_action_local_rollouts_matching_runtime_one_block_pulse_v1",
        "train_root_count": len(train_set),
        "effect_calibration_root_count": len(calibration_set),
        "train_prefix_state_count": int(len({(root_of(row), int(row["prefix_id"])) for row in rows if root_of(row) in train_set})),
        "effect_calibration_prefix_state_count": int(len({(root_of(row), int(row["prefix_id"])) for row in rows if root_of(row) in calibration_set})),
        "latent_training_threshold": train_latent_threshold,
        "action_training_threshold": train_action_threshold,
        "latent_threshold": float(latent_support["threshold"]),
        "action_threshold": float(action_support["threshold"]),
        "latent_calibration_distance_p50": float(np.quantile(calibration_latent_distances, .5)),
        "latent_calibration_distance_p90": float(np.quantile(calibration_latent_distances, .9)),
        "action_calibration_distance_p50": float(np.quantile(calibration_action_distances, .5)),
        "action_calibration_distance_p90": float(np.quantile(calibration_action_distances, .9)),
        "calibration_quantile": .90,
        "regret_calibration": regret_calibration,
        "gate_calibration_outcome_roots_used": sorted(calibration_set),
        "held_out_planning_roots_read": False,
    }
    torch.save(payload, policy_path)
    return dict(payload["support_expansion"])


def build_support_expanded_policy(
    *,
    dataset_root: Path,
    checkpoint: Path,
    dinov2_source: Path,
    base_effect_policy: Path,
    output_dir: Path,
    train_roots: list[int],
    effect_calibration_roots: list[int],
    prefix_count: int = 4,
    epochs: int = 180,
    alpha: float = .10,
) -> dict[str, Any]:
    validate_collection_partition(train_roots, effect_calibration_roots)
    if int(prefix_count) < 2:
        raise ValueError("support expansion needs at least initial plus one prefix state")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    base = torch.load(base_effect_policy, map_location="cpu", weights_only=False)
    pca_mean = np.asarray(base["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(base["pca_basis"], dtype=np.float64)
    model, transform, _stats, _action_mean, _action_std = _load_jepa_model(
        ROOT, dataset_root, checkpoint, dinov2_source, device,
    )
    # Renderer and RGB->proprio chart are fitted only from train roots. They
    # are collection infrastructure, never a test-time state channel.
    renderer_images, _renderer_states = load_episode(dataset_root, train_roots[0])
    calibration = fit_visual_calibration(dataset_root, train_roots[:min(16, len(train_roots))], max_frames=512)
    renderer = DatasetMatchedRenderer.from_episode(renderer_images, calibration)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for split, roots in (("train", train_roots), ("calibration", effect_calibration_roots)):
        for root in roots:
            initial = load_episode_states(dataset_root, root)[0]
            runtime = LegacyPointMazeRuntime(initial)
            previous = np.asarray(runtime.state, dtype=np.float64).copy()
            for prefix in range(int(prefix_count)):
                source = np.asarray(runtime.state, dtype=np.float64).copy()
                images, proprios = _context_proprio(renderer, calibration, previous, source)
                projection = _encode_projection(model, transform, pca_mean, pca_basis, images, proprios)
                current_xy = np.asarray(proprios[-1], dtype=np.float64)[:2]
                prefix_rows, prefix_arrays = _pulse_rows_for_prefix(
                    root=root, split=split, prefix=prefix, source_state=source,
                    source_projection=projection, current_xy=current_xy,
                )
                rows.extend(prefix_rows)
                arrays.update(prefix_arrays)
                if prefix + 1 < int(prefix_count):
                    # Recompute in the new pre-action chart after every
                    # prefix.  This is the same goal-relative family that the
                    # live controller regenerates at each replanning block.
                    live_bank = action_bank(current_xy, GOAL_STATE, directions=8, magnitudes=[0.25, 0.5, 0.75, 1.0])
                    prefix_action = live_bank[(int(root) * 11 + prefix * 7) % len(live_bank)]
                    previous = source
                    runtime.step_block(prefix_action, steps=FRAME_STRIDE)
    metadata = {
        "candidate_family": "runtime_one_block_pulse_v1",
        "candidate_bank_coordinate_frame": "goal_relative",
        "total_action_steps": FRAME_STRIDE,
        "context_length": 2,
        "history_windows": 1,
        "execution_contract": "one_replanning_block_before_reobservation",
        "executed_blocks_before_reobservation": 1,
        "support_expansion": "causal_prefix_local_state_action_rollouts",
        "declared_train_roots": train_roots,
        "declared_effect_calibration_roots": effect_calibration_roots,
        "forbidden_gate_and_test_roots": sorted(FORBIDDEN_RUNTIME_ROOTS),
        "held_out_planning_outcomes_read": False,
    }
    dataset = {"metadata": metadata, "records": rows}
    dataset_path = output_dir / "support_expanded_candidate_dataset.json"
    arrays_path = output_dir / "support_expanded_arrays.npz"
    dataset_path.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(arrays_path, **arrays)
    policy_path = output_dir / "support_expanded_effect_policy.pt"
    fit_summary_path = output_dir / "fit_summary.json"
    fit_summary = fit_online_effect_policy(
        candidate_dataset=dataset_path,
        arrays=arrays_path,
        source_model_checkpoint=base_effect_policy,
        output=policy_path,
        summary_output=fit_summary_path,
        calibration_candidate_dataset=dataset_path,
        calibration_arrays=arrays_path,
        epochs=int(epochs), alpha=float(alpha), device="cuda",
        feature_mode="frozen_latent_action_basis", calibration_splits=["calibration"],
    )
    gate = _calibrate_expanded_gate(
        policy_path=policy_path, rows=rows, arrays=arrays,
        train_roots=train_roots, calibration_roots=effect_calibration_roots, alpha=float(alpha),
    )
    report = {
        "schema": "pointmaze.support_expanded_effect_policy_build.v1",
        "status": "frozen",
        "policy": str(policy_path.resolve()),
        "policy_sha256": _sha256(policy_path),
        "source_data": {"dataset": str(dataset_path.resolve()), "arrays": str(arrays_path.resolve())},
        "base_effect_policy": str(base_effect_policy.resolve()),
        "base_effect_policy_sha256": _sha256(base_effect_policy),
        "protocol": {
            "train_roots": train_roots,
            "effect_calibration_roots": effect_calibration_roots,
            "prefix_count": int(prefix_count),
            "one_block_pulse_steps": FRAME_STRIDE,
            "candidate_count_per_prefix": 32,
            "held_out_gate_or_test_roots_read": False,
            "gpu_training_required": True,
        },
        "counts": {
            "records": len(rows), "prefix_states": len(rows) // 32,
            "train_prefix_states": len(train_roots) * int(prefix_count),
            "effect_calibration_prefix_states": len(effect_calibration_roots) * int(prefix_count),
        },
        "fit_summary": fit_summary,
        "separate_support_gate": gate,
        "claim_boundary": (
            "The effect policy is frozen before visual-gate and held-out planning roots are opened. "
            "This expands runtime pulse support; it does not by itself establish coverage transport or planning benefit."
        ),
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/point_maze_tail1800_1899")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar")
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--base-effect-policy", type=Path, default=ROOT / "results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-roots", default=",".join(map(str, DEFAULT_TRAIN)))
    parser.add_argument("--effect-calibration-roots", default=",".join(map(str, DEFAULT_EFFECT_CALIBRATION)))
    parser.add_argument("--prefix-count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--alpha", type=float, default=.10)
    args = parser.parse_args()
    report = build_support_expanded_policy(
        dataset_root=args.dataset_root.resolve(), checkpoint=args.checkpoint.resolve(),
        dinov2_source=args.dinov2_source.resolve(), base_effect_policy=args.base_effect_policy.resolve(),
        output_dir=args.output_dir.resolve(), train_roots=parse_roots(args.train_roots, "train roots"),
        effect_calibration_roots=parse_roots(args.effect_calibration_roots, "effect-calibration roots"),
        prefix_count=args.prefix_count, epochs=args.epochs, alpha=args.alpha,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
