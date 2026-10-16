"""Evaluate a PointMaze adapter on an independent balanced action intervention.

The learned-adapter panel was previously limited to observed trajectory blocks.
That is enough for calibration, but it is not an external action-identifiability
test.  This runner uses source roots from a separate enriched export, applies a
fixed ``+a/-a/zero`` fan to the adapter, and obtains the intervention labels by
stepping the PointMaze simulator offline.  The simulator state never enters the
adapter input.

The runner also reports a PointMaze-specific held-out horizon consequence on
the same new roots.  It intentionally does not manufacture PushT-style angle
or block metrics; the record builder accepts this declared progress metric
family explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.empirical_spectral_audit.pointmaze_action_charts import action_blocks_for_chart
from experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel import (
    HORIZONS,
    _project,
    _predict,
    _task_readout_with_center,
    LinearAdapter,
    MLPAdapter,
    ResidualMLPAdapter,
    StandardizedMLPAdapter,
    TaskAlignedMLPAdapter,
)


def _model_class(architecture: str):
    if architecture == "linear":
        return LinearAdapter
    if architecture == "mlp":
        return MLPAdapter
    if architecture == "residual_mlp":
        return ResidualMLPAdapter
    if architecture in {"standardized_mlp", "task_aligned_mlp", "task_effect_mlp", "task_effect_lowrank_mlp"}:
        return StandardizedMLPAdapter if architecture == "standardized_mlp" else TaskAlignedMLPAdapter
    if architecture == "task_effect_linear":
        return LinearAdapter
    raise ValueError(f"unsupported adapter architecture: {architecture}")


def _load_adapter(checkpoint: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    training = dict(payload["adapter_training"])
    rank = int(payload["rank"])
    architecture = str(payload["architecture"])
    task_effect = architecture in {"task_effect_mlp", "task_effect_linear", "task_effect_lowrank_mlp"}
    output_dim = len(HORIZONS) if task_effect else rank * len(HORIZONS)
    history_windows = int(training.get("history_windows", 1))
    declared_input = training.get("x_mean")
    input_dim = (
        len(np.asarray(declared_input))
        if declared_input is not None
        else rank * history_windows + 4 * 10
    )
    if training.get("feature_mode") == "train_action_pca4":
        input_dim = rank + len(np.asarray(training["action_basis"]))
    model = _model_class(architecture)(input_dim, output_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _root_key(row: dict[str, Any]) -> str:
    return f"{int(row['episode_id'])}:{int(row.get('window_start', 0))}"


def _fit_readouts(fit_rows: list[dict[str, Any]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    source_states = np.asarray([row["source_state_raw"] for row in fit_rows], dtype=np.float64)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for horizon in HORIZONS:
        targets = np.asarray([row["latent_by_horizon"][str(horizon)]["target"] for row in fit_rows], dtype=np.float64)
        target_states = np.asarray([
            row["latent_by_horizon"][str(horizon)]["target_state_raw"] for row in fit_rows
        ], dtype=np.float64)
        out[horizon] = _task_readout_with_center(targets, source_states, target_states)
    return out


def _external_features(
    rows: list[dict[str, Any]],
    pca_mean: np.ndarray,
    pca_basis: np.ndarray,
    action_blocks: np.ndarray,
    action_feature_chart: str = "raw",
) -> np.ndarray:
    source = np.asarray([_project(row["source_latent"], pca_mean, pca_basis) for row in rows], dtype=np.float32)
    actions = np.asarray(action_blocks, dtype=np.float32)
    actions = action_blocks_for_chart(rows, actions, action_feature_chart)
    return np.concatenate([source, actions.reshape(len(rows), -1)], axis=1)


def _intervention_action_blocks(action: np.ndarray) -> np.ndarray:
    """Build a support-consistent four-block intervention sequence."""
    block = np.tile(np.asarray(action, dtype=np.float64), 5)
    return np.repeat(block.reshape(1, -1), max(HORIZONS), axis=0)


def _legacy_pointmaze_xml() -> str:
    """Reproduce the released legacy U-maze XML without mujoco-py/d4rl."""
    maze = (
        "#####\\"
        "#GOO#\\"
        "###O#\\"
        "#OOO#\\"
        "#####"
    )
    lines = maze.split("\\")
    walls = []
    for w, line in enumerate(lines):
        for h, tile in enumerate(line):
            if tile == "#":
                walls.append(
                    f'<geom name="wall_{w}_{h}" type="box" material="wall" '
                    f'pos="{w + 1.0} {h + 1.0} 0" size="0.5 0.5 0.2" '
                    'contype="1" conaffinity="1"/>'
                )
    return """<mujoco model="point_maze">
<compiler inertiafromgeom="true" angle="radian" coordinate="local"/>
<option timestep="0.01" gravity="0 0 0" iterations="20" integrator="Euler"/>
<default>
  <joint damping="1" limited="false"/>
  <geom friction=".5 .1 .1" density="1000" margin=".002" condim="1" contype="2" conaffinity="1"/>
</default>
<asset>
  <material name="wall" rgba=".7 .5 .3 1"/>
</asset>
<worldbody>
  <geom name="ground" type="plane" size="40 40 .25" pos="0 0 -.1" contype="0" conaffinity="0"/>
  <body name="particle" pos="1.2 1.2 0">
    <geom name="particle_geom" type="sphere" size=".1" contype="1" conaffinity="1"/>
    <joint name="ball_x" type="slide" pos="0 0 0" axis="1 0 0"/>
    <joint name="ball_y" type="slide" pos="0 0 0" axis="0 1 0"/>
  </body>
  {walls}
</worldbody>
<actuator>
  <motor joint="ball_x" ctrlrange="-1 1" ctrllimited="true" gear="100"/>
  <motor joint="ball_y" ctrlrange="-1 1" ctrllimited="true" gear="100"/>
</actuator>
</mujoco>""".format(walls="\n  ".join(walls))


def _simulate_effects(rows: list[dict[str, Any]], fan_actions: list[np.ndarray]) -> np.ndarray:
    """Return offline effects aligned with the adapter's four action blocks.

    ``action_blocks_raw`` contains four consecutive five-step blocks and the
    adapter predicts horizons ``{1, 2, 4}`` in that block clock.  The old
    evaluator simulated only the first block, which made an apparently
    multi-horizon action intervention incomparable with the predictor input.
    Each fan action is now repeated over all four blocks and the physical
    effect is recorded after 1, 2, and 4 blocks.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    effects = []
    for row in rows:
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        if source.shape[0] != 4:
            raise ValueError("PointMaze source_state_raw must have four entries")
        for action in fan_actions:
            data.qpos[:] = source[:2]
            data.qvel[:] = source[2:4]
            mujoco.mj_forward(model, data)
            before = source[:2].copy()
            horizon_effects = []
            data.ctrl[:] = np.asarray(action, dtype=np.float64)
            for block_index in range(max(HORIZONS)):
                for _ in range(5):
                    mujoco.mj_step(model, data)
                if block_index + 1 in HORIZONS:
                    horizon_effects.append(np.asarray(data.qpos[:2], dtype=np.float64) - before)
            effects.append(horizon_effects)
    return np.asarray(effects, dtype=np.float64).reshape(len(rows), len(fan_actions), len(HORIZONS), 2)


def _balanced_action_risk(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    """Compute symmetric relative +a/-a/zero effect error and sign accuracy."""
    pair_pred = predicted[:, 0] - predicted[:, 1]
    pair_actual = actual[:, 0] - actual[:, 1]
    denom = np.maximum(np.linalg.norm(pair_actual, axis=1), 0.10)
    pair_error = np.linalg.norm(pair_pred - pair_actual, axis=1) / denom
    zero_error = np.linalg.norm(predicted[:, 2] - actual[:, 2], axis=1) / np.maximum(
        np.linalg.norm(pair_actual, axis=1), 0.10
    )
    cosine = np.sum(pair_pred * pair_actual, axis=1) / np.maximum(
        np.linalg.norm(pair_pred, axis=1) * np.linalg.norm(pair_actual, axis=1), 1e-8
    )
    # ATM measures the action-conditioned effect, not the model's ability to
    # reproduce passive drift.  The latter is a separate observable reported
    # below.  Mixing it into the primary risk previously penalized a valid
    # action-effect predictor whenever the source had nonzero velocity.
    risk = float(np.mean(pair_error))
    identifiability = float(np.mean(cosine > 0.0))
    details = {
        "pair_error_mean": float(np.mean(pair_error)),
        "zero_effect_error_mean": float(np.mean(zero_error)),
        "effect_cosine_mean": float(np.mean(cosine)),
        "directional_pair_count": int(len(pair_error)),
    }
    return risk, identifiability, details


def _planning_consequence(
    rows: list[dict[str, Any]],
    predicted: dict[int, np.ndarray],
    readouts: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    progress = []
    actual = []
    for horizon in HORIZONS:
        decoder, center, target_center = readouts[horizon]
        latent = np.asarray(predicted[horizon], dtype=np.float64)
        decoded = (latent - center.reshape(1, -1)) @ decoder + target_center.reshape(1, -1)
        progress.append(np.sum(decoded, axis=1))
        actual.append(np.asarray([
            np.sum(np.asarray(row["latent_by_horizon"][str(horizon)]["target_state_raw"], dtype=np.float64)[:2]
                   - np.asarray(row["source_state_raw"], dtype=np.float64)[:2])
            for row in rows
        ]))
    predicted_matrix = np.stack(progress, axis=1)
    actual_matrix = np.stack(actual, axis=1)
    selected = np.argmax(predicted_matrix, axis=1)
    chosen = actual_matrix[np.arange(len(rows)), selected]
    oracle = np.max(actual_matrix, axis=1)
    return {
        "metric_family": "pointmaze_horizon_progress",
        "planning_risk": float(np.mean(oracle - chosen)),
        "mean_chosen_progress": float(np.mean(chosen)),
        "mean_oracle_progress": float(np.mean(oracle)),
        "selected_horizon_counts": {str(h): int(np.sum(selected == i)) for i, h in enumerate(HORIZONS)},
        "paired_roots": int(len(rows)),
        "protocol": {
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "test_planning_used_for_snapshot_selection": False,
            "simulator_state_used_for": "offline consequence scoring only",
            "metric_fit_split": "diagnostic fit rows only",
        },
    }


def evaluate(
    checkpoint: Path,
    fit_export: Path,
    outcome_export: Path,
    external_export: Path,
    output: Path,
    *,
    device: str = "cpu",
    action_magnitude: float = 0.75,
    max_roots: int | None = None,
) -> dict[str, Any]:
    fit = load_payload(fit_export)
    outcome = load_payload(outcome_export)
    external = load_payload(external_export)
    if str(Path(fit["checkpoint"]).resolve()) != str(checkpoint.resolve()):
        raise ValueError("fit export checkpoint does not match adapter checkpoint")
    if str(Path(outcome["checkpoint"]).resolve()) != str(checkpoint.resolve()):
        raise ValueError("outcome export checkpoint does not match adapter checkpoint")
    used = {_root_key(row) for row in fit["enriched_records"]} | {_root_key(row) for row in outcome["enriched_records"]}
    rows = [row for row in external["enriched_records"] if int(row.get("window_start", 0)) != 0 and _root_key(row) not in used]
    if max_roots is not None:
        rows = rows[: int(max_roots)]
    if len(rows) < 4:
        raise ValueError("external export has fewer than four root-disjoint source roots")
    model, payload = _load_adapter(checkpoint, device)
    action_feature_chart = str(payload.get("adapter_training", {}).get("action_feature_chart", "raw"))
    pca_mean = np.asarray(payload["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(payload["pca_basis"], dtype=np.float64)
    readouts = _fit_readouts(fit["enriched_records"])
    # Diagonal directions keep both action coordinates excited.  The old
    # axis fan included an exact zero coordinate and, together with its
    # zero-filled future blocks, created a preventable action-support gap.
    directions = (
        np.asarray([1.0, 1.0]) / np.sqrt(2.0),
        np.asarray([1.0, -1.0]) / np.sqrt(2.0),
    )
    all_predicted, all_actual = [], []
    for direction in directions:
        fan_actions = [
            action_magnitude * direction,
            -action_magnitude * direction,
            np.zeros(2, dtype=np.float64),
        ]
        # A model input is four consecutive blocks, each containing five
        # two-dimensional actions.  Repeat the intervention over all blocks;
        # do not silently leave the remaining 30 features at zero.
        blocks = np.asarray([_intervention_action_blocks(action) for action in fan_actions], dtype=np.float32)
        predicted_fan = {h: [] for h in HORIZONS}
        actual_fan = _simulate_effects(rows, fan_actions)
        for fan_index in range(3):
            x = _external_features(
                rows,
                pca_mean,
                pca_basis,
                np.repeat(blocks[fan_index][None], len(rows), axis=0),
                action_feature_chart,
            )
            prediction = _predict(model, x, int(pca_basis.shape[0]), device, payload["adapter_training"])
            for h in HORIZONS:
                decoder, center, target_center = readouts[h]
                predicted_fan[h].append(
                    (prediction[h] - center.reshape(1, -1)) @ decoder
                    + target_center.reshape(1, -1)
                )
        all_predicted.append(np.stack([np.stack(predicted_fan[h], axis=1) for h in HORIZONS], axis=2))
        all_actual.append(actual_fan)
    predicted = np.mean(np.stack(all_predicted, axis=1), axis=1)  # [N, 3, H, 2]
    actual = np.mean(np.stack(all_actual, axis=1), axis=1)
    horizon_risks = {}
    horizon_details = {}
    for index, h in enumerate(HORIZONS):
        risk_h, ident_h, details_h = _balanced_action_risk(predicted[:, :, index], actual[:, :, index])
        horizon_risks[str(h)] = risk_h
        horizon_details[str(h)] = {"action_identifiability": ident_h, **details_h}
    risk = float(np.mean(list(horizon_risks.values())))
    identifiability = float(np.mean([item["action_identifiability"] for item in horizon_details.values()]))
    details = {
        "horizon_action_inverse_risk": horizon_risks,
        "horizon_action_details": horizon_details,
        "pair_error_mean": risk,
        "zero_effect_error_mean": float(np.mean([item["zero_effect_error_mean"] for item in horizon_details.values()])),
        "effect_cosine_mean": float(np.mean([item["effect_cosine_mean"] for item in horizon_details.values()])),
        "directional_pair_count": int(len(rows) * len(directions) * len(HORIZONS)),
    }
    observed_blocks = np.asarray([row["action_blocks_raw"] for row in rows], dtype=np.float32)
    observed_predicted = _predict(
        model,
        _external_features(rows, pca_mean, pca_basis, observed_blocks, action_feature_chart),
        int(pca_basis.shape[0]),
        device,
        payload["adapter_training"],
    )
    planning = _planning_consequence(rows, observed_predicted, readouts)
    report = {
        "schema": "learned_snapshot_external_evidence.v2",
        "checkpoint": str(checkpoint.resolve()),
        "architecture": str(fit["architecture"]),
        "seed": int(fit["seed"]),
        "environment": "PointMaze",
        "training_stage": int(fit["training_stage"]),
        "action_distribution": str(fit["action_distribution"]),
        "action_risk": {
            "action_inverse_risk": risk,
            "action_identifiability": identifiability,
            "paired_roots": int(len(rows) * len(directions)),
            **details,
            "protocol": {
                "split": "root-disjoint external windows from a separate frozen export",
                "intervention": "+a/-a/zero balanced diagonal fan, repeated over four five-step blocks",
                "action_magnitude": float(action_magnitude),
                "risk_definition": "mean over horizons of relative error of predicted (+a)-(-a) effect; zero branch is baseline-drift diagnostic only",
                "horizon_clock": "one adapter action block equals five simulator steps; labels at blocks 1, 2, and 4",
                "action_support_repair": "all four model action blocks populated; diagonal directions avoid exact zero coordinates",
                "simulator_state_used_for": "offline intervention labels only",
                "model_runtime_input": "source latent derived from RGB plus action fan",
                "model_action_feature_chart": action_feature_chart,
            },
        },
        "planning_consequence": planning,
        "protocol": {
            "checkpoint_immutable": True,
            "diagnostic_export_not_reused_for_external_metrics": True,
            "external_source_root_disjoint_from_fit_and_outcome": True,
            "action_risk_split": "independent source-root balanced intervention",
            "planning_split": "same independent external roots; observed-action horizon consequence",
            "evidence_path": str((output / "external_evidence.json").resolve()),
        },
        "claim_boundary": "PointMaze root-disjoint action-balanced external evidence; no universal or cross-environment claim.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "external_evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path, required=True)
    parser.add_argument("--outcome-export", type=Path, required=True)
    parser.add_argument("--external-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-magnitude", type=float, default=0.75)
    parser.add_argument("--max-roots", type=int, default=None)
    args = parser.parse_args()
    report = evaluate(**vars(args))
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "paired_roots": report["action_risk"]["paired_roots"],
        "action_inverse_risk": report["action_risk"]["action_inverse_risk"],
        "planning_risk": report["planning_consequence"]["planning_risk"],
    }, indent=2))


if __name__ == "__main__":
    main()
