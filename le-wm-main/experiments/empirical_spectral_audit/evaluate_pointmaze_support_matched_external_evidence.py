"""Evaluate PointMaze external action transfer with support-matched sequences.

The legacy external audit used a diagonal constant action repeated five times
inside every action block.  The learned adapter, however, is trained on
``action_blocks_raw`` containing five potentially different actions per block.
This runner keeps the frozen checkpoint and all train/calibration/outcome
splits unchanged, but replaces that out-of-support intervention with a
predeclared signed version of each external root's observed four-block action
sequence.  The simulator is used only to create offline intervention labels.

The output is a diagnostic, not a planning claim.  It reports both replay
consistency of the observed sequence and the support-matched action inverse
risk, making simulator/action-sequence mismatch visible rather than folding it
into a single negative score.
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
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    HORIZONS,
    _balanced_action_risk,
    _external_features,
    _fit_readouts,
    _load_adapter,
    _planning_consequence,
    _predict,
    _root_key,
)


def _legacy_pointmaze_xml() -> str:
    maze = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    walls = []
    for w, line in enumerate(maze.split("\\")):
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
<default><joint damping="1" limited="false"/>
<geom friction=".5 .1 .1" density="1000" margin=".002" condim="1" contype="2" conaffinity="1"/>
</default><asset><material name="wall" rgba=".7 .5 .3 1"/></asset>
<worldbody><geom name="ground" type="plane" size="40 40 .25" pos="0 0 -.1" contype="0" conaffinity="0"/>
<body name="particle" pos="1.2 1.2 0"><geom name="particle_geom" type="sphere" size=".1" contype="1" conaffinity="1"/>
<joint name="ball_x" type="slide" pos="0 0 0" axis="1 0 0"/>
<joint name="ball_y" type="slide" pos="0 0 0" axis="0 1 0"/></body>
{walls}</worldbody><actuator><motor joint="ball_x" ctrlrange="-1 1" ctrllimited="true" gear="100"/>
<motor joint="ball_y" ctrlrange="-1 1" ctrllimited="true" gear="100"/></actuator></mujoco>""".format(
        walls="\n".join(walls)
    )


def _simulate_sequence_effects(
    rows: list[dict[str, Any]],
    action_sequences: np.ndarray,
) -> np.ndarray:
    """Replay [root, branch, block, five-step action] sequences."""
    if action_sequences.ndim != 4 or action_sequences.shape[1:] != (3, 4, 10):
        raise ValueError(f"expected action sequence shape [N,3,4,10], got {action_sequences.shape}")
    import mujoco

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    output = []
    for row_index, row in enumerate(rows):
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        if source.shape != (4,):
            raise ValueError("PointMaze source_state_raw must have four entries")
        branch_effects = []
        for branch in range(3):
            data.qpos[:] = source[:2]
            data.qvel[:] = source[2:4]
            mujoco.mj_forward(model, data)
            before = source[:2].copy()
            horizon_effects = []
            blocks = action_sequences[row_index, branch].reshape(4, 5, 2)
            for block_index, block in enumerate(blocks):
                for action in block:
                    data.ctrl[:] = action
                    # PointMaze stores one control per environment step, and
                    # the released environment uses frame_skip=5.  The
                    # previous support-matched evaluator advanced only one
                    # MuJoCo step, making its intervention labels
                    # incomparable with the predictor's five-step blocks.
                    for _ in range(5):
                        mujoco.mj_step(model, data)
                if block_index + 1 in HORIZONS:
                    horizon_effects.append(data.qpos[:2].copy() - before)
            branch_effects.append(horizon_effects)
        output.append(branch_effects)
    return np.asarray(output, dtype=np.float64)


def _load_rows(
    fit_export: Path,
    outcome_export: Path,
    external_export: Path,
    max_roots: int | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    fit = load_payload(fit_export)
    outcome = load_payload(outcome_export)
    external = load_payload(external_export)
    used = {_root_key(row) for row in fit["enriched_records"]}
    used |= {_root_key(row) for row in outcome["enriched_records"]}
    rows = [
        row
        for row in external["enriched_records"]
        if int(row.get("window_start", 0)) != 0 and _root_key(row) not in used
    ]
    if max_roots is not None:
        rows = rows[: int(max_roots)]
    if len(rows) < 4:
        raise ValueError("external export has fewer than four root-disjoint rows")
    return fit, outcome, rows


def evaluate(
    checkpoint: Path,
    fit_export: Path,
    outcome_export: Path,
    external_export: Path,
    output: Path,
    *,
    device: str = "cuda",
    sequence_scale: float = 0.5,
    max_roots: int | None = None,
) -> dict[str, Any]:
    if not np.isfinite(sequence_scale) or not 0.0 < sequence_scale <= 1.0:
        raise ValueError("sequence_scale must lie in (0, 1]")
    fit, outcome, rows = _load_rows(fit_export, outcome_export, external_export, max_roots)
    if str(Path(fit["checkpoint"]).resolve()) != str(checkpoint.resolve()):
        raise ValueError("fit export checkpoint does not match adapter checkpoint")
    model, payload = _load_adapter(checkpoint, device)
    pca_mean = np.asarray(payload["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(payload["pca_basis"], dtype=np.float64)
    readouts = _fit_readouts(fit["enriched_records"])

    observed = np.asarray([row["action_blocks_raw"] for row in rows], dtype=np.float32)
    if observed.shape[1:] != (4, 10):
        raise ValueError(f"expected external action blocks [N,4,10], got {observed.shape}")
    action_sequences = np.stack(
        [sequence_scale * observed, -sequence_scale * observed, np.zeros_like(observed)],
        axis=1,
    )
    actual = _simulate_sequence_effects(rows, action_sequences)
    predicted_by_branch: list[dict[int, np.ndarray]] = []
    for branch in range(3):
        features = _external_features(
            rows,
            pca_mean,
            pca_basis,
            action_sequences[:, branch],
            str(payload.get("adapter_training", {}).get("action_feature_chart", "raw")),
        )
        latent = _predict(
            model,
            features,
            int(pca_basis.shape[0]),
            device,
            payload["adapter_training"],
        )
        decoded: dict[int, np.ndarray] = {}
        for horizon in HORIZONS:
            decoder, center, target_center = readouts[horizon]
            decoded[horizon] = (
                (latent[horizon] - center.reshape(1, -1)) @ decoder
                + target_center.reshape(1, -1)
            )
        predicted_by_branch.append(decoded)
    predicted = np.stack(
        [np.stack([predicted_by_branch[branch][h] for h in HORIZONS], axis=1)
         for branch in range(3)],
        axis=1,
    )

    horizon_risk: dict[str, float] = {}
    horizon_details: dict[str, dict[str, Any]] = {}
    for index, horizon in enumerate(HORIZONS):
        risk, identifiability, details = _balanced_action_risk(
            predicted[:, :, index], actual[:, :, index]
        )
        horizon_risk[str(horizon)] = risk
        horizon_details[str(horizon)] = {
            "action_identifiability": identifiability,
            **details,
        }

    observed_features = _external_features(
        rows,
        pca_mean,
        pca_basis,
        observed,
        str(payload.get("adapter_training", {}).get("action_feature_chart", "raw")),
    )
    observed_latent = _predict(
        model,
        observed_features,
        int(pca_basis.shape[0]),
        device,
        payload["adapter_training"],
    )
    planning = _planning_consequence(rows, observed_latent, readouts)
    # Replay the *full* observed sequence separately.  The intervention branch
    # may use a predeclared scale, so comparing its scaled trajectory against
    # the dataset target would confound replay mismatch with the intervention.
    observed_replay = _simulate_sequence_effects(
        rows,
        np.stack([observed, np.zeros_like(observed), np.zeros_like(observed)], axis=1),
    )
    observed_replay_errors = []
    for row_index, row in enumerate(rows):
        for horizon_index, horizon in enumerate(HORIZONS):
            target_delta = (
                np.asarray(row["latent_by_horizon"][str(horizon)]["target_state_raw"], dtype=np.float64)[:2]
                - np.asarray(row["source_state_raw"], dtype=np.float64)[:2]
            )
            replay_delta = observed_replay[row_index, 0, horizon_index]
            observed_replay_errors.append(float(np.linalg.norm(replay_delta - target_delta)))
    report = {
        "schema": "pointmaze_support_matched_external_evidence.v2",
        "checkpoint": str(checkpoint.resolve()),
        "architecture": str(fit["architecture"]),
        "seed": int(fit["seed"]),
        "environment": "PointMaze",
        "action_risk": {
            "action_inverse_risk": float(np.mean(list(horizon_risk.values()))),
            "action_identifiability": float(np.mean([
                item["action_identifiability"] for item in horizon_details.values()
            ])),
            "horizon_action_inverse_risk": horizon_risk,
            "horizon_action_details": horizon_details,
            "paired_roots": int(len(rows)),
        },
        "replay_consistency": {
            "observed_sequence_state_l2_mean": float(np.mean(observed_replay_errors)),
            "observed_sequence_state_l2_median": float(np.median(observed_replay_errors)),
            "observed_sequence_state_l2_max": float(np.max(observed_replay_errors)),
            "metric": "modern-MuJoCo replay of observed action_blocks_raw versus released target_state_raw",
        },
        "planning_consequence": planning,
        "protocol": {
            "checkpoint_immutable": True,
            "root_disjoint": True,
            "outcome_used_for_selection": False,
            "model_runtime_input": "source latent plus root-specific action sequence",
            "intervention": "observed four-block sequence, signed +/- and zero branches",
            "sequence_scale": float(sequence_scale),
            "action_support": "same temporal block structure as training; no constant-repeat substitution",
            "steps_per_action": 5,
            "simulator_state_used_for": "offline intervention labels and replay consistency only",
            "device": device,
        },
        "claim_boundary": (
            "Root-disjoint support-matched PointMaze action-transfer diagnostic. "
            "It does not establish causal physics, visual online planning, or a "
            "universal world model."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "support_matched_external_evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path, required=True)
    parser.add_argument("--outcome-export", type=Path, required=True)
    parser.add_argument("--external-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-scale", type=float, default=0.5)
    parser.add_argument("--max-roots", type=int, default=100)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report = evaluate(
        args.checkpoint,
        args.fit_export,
        args.outcome_export,
        args.external_export,
        args.output,
        device=args.device,
        sequence_scale=args.sequence_scale,
        max_roots=args.max_roots,
    )
    print(json.dumps({
        "action_inverse_risk": report["action_risk"]["action_inverse_risk"],
        "action_identifiability": report["action_risk"]["action_identifiability"],
        "replay_l2_mean": report["replay_consistency"]["observed_sequence_state_l2_mean"],
        "planning_risk": report["planning_consequence"]["planning_risk"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
