"""Fresh-root C3 calibration test for the frozen V20 visible-motion repair.

This runner never retrains the V20 recurrent/action model and never changes
its 0.02 visual-motion route.  It constructs a new root-disjoint panel, uses
only its calibration roots to freeze a marginal fixed-bank contrast/risk
gate, and opens final outcome labels only after that gate is persisted.

The resulting certificate is deliberately marginal and offline; it is not a
permit-conditional safety guarantee, online MPC result, or universal claim.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder
from experiments.decision_benchmark import run_htsc_matched_transfer as htsc
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.decision_benchmark import run_pointmaze_mixed_velocity_repairability_v20_cuda as v20
from experiments.decision_benchmark import run_unseen_action_bank_pointmaze as simulator
from experiments.gdd_wm import preflight_pointmaze_c3_calibrated_reliability_v21 as preflight
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "point_maze_v67_c3_calibrated_reliability_20261023"
OUTPUT = ROOT / "results" / "pointmaze_c3_calibrated_reliability_v21_20261023_cuda"
V20_PANEL = ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_20261023_cuda"
PREFLIGHT = ROOT / "results" / "pointmaze_c3_calibrated_reliability_v21_static_preflight_20261023.json"
ROOT_RANGE, SPLIT_COUNTS = (28_040, 28_872), {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611503, 202611504, 202611505
POLICY = {"kind": "history_energy_gate", "direction": "recurrent_when_high", "threshold": 0.02,
          "motion_source": "source_history_visual_state", "fixed_threshold": True}
INVERSE_POLICY = {**POLICY, "direction": "recurrent_when_low"}
PREFLIGHT_MODULE = preflight
POOL_MODULE = "experiments.decision_benchmark.build_pointmaze_v21_c3_reliability_pool"
FIXED_HELD_ACTIONS: np.ndarray | None = None
ALLOWED_PRIOR_ACTION_HASHES: set[str] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _contexts() -> tuple[Path, ...]:
    contexts = {*v20._prior_contexts(), (V20_PANEL / "temporal_ood_v6_context.json").resolve()}
    if not all(path.is_file() for path in contexts):
        raise FileNotFoundError("V21 requires all V20-frozen prior contexts and V20 context")
    return tuple(sorted(contexts))


def _ensure_preflight(contexts: tuple[Path, ...]) -> dict[str, Any]:
    result = PREFLIGHT_MODULE.preflight(
        dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
        held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    if result["status"] != "passed_static_preflight":
        raise RuntimeError(f"V21 static preflight rejected: {result['checks']}")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if PREFLIGHT.exists() and PREFLIGHT.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing V21 preflight differs from its deterministic frozen contract")
    PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    if not PREFLIGHT.exists():
        PREFLIGHT.write_text(encoded, encoding="utf-8")
    return result


def _build(contexts: tuple[Path, ...]) -> dict[str, Any]:
    source, arrays, context = (OUTPUT / name for name in (
        "temporal_ood_v6_source_dataset.json", "temporal_ood_v6_source_arrays.npz", "temporal_ood_v6_context.json"))
    if OUTPUT.exists():
        if not (source.is_file() and arrays.is_file() and context.is_file()):
            raise RuntimeError("V21 output exists but lacks a complete sealed feature panel")
        return {"resumed_existing_sealed_build": True}
    inputs = {
        "encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
        "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
        "dinov2": ROOT / "external" / "local_dinov2",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V21 requires bound feature inputs: {missing}")
    return builder.build_panel(
        DATASET, list(contexts), OUTPUT, encoder_checkpoint=inputs["encoder"], pca_checkpoint=inputs["pca"],
        dinov2_source=inputs["dinov2"], candidate_generalization_regime="shared_candidate_in_support",
        candidate_design="velocity_matched_braking_pairs", goal_qpos=[0.8, 0.8], device="cuda",
        root_seed=ROOT_SEED, held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        split_counts=SPLIT_COUNTS, root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
        frozen_preflight_ledger=PREFLIGHT,
        fixed_held_actions=FIXED_HELD_ACTIONS,
        allowed_prior_action_hashes=ALLOWED_PRIOR_ACTION_HASHES,
    )


def _labels(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | list[float]]]:
    values: dict[str, dict[str, float | list[float]]] = {}
    for row in rows:
        terminal = simulator._simulate_sequence(np.asarray(row["source_state_raw"], dtype=np.float64),
                                                np.asarray(row["action_sequence"], dtype=np.float64))
        goal = np.asarray(row["goal_raw"], dtype=np.float64)
        values[interaction._key(row)] = {"real_task_cost": float(np.linalg.norm(terminal[:2] - goal[:2])),
                                         "realized_terminal_state": np.asarray(terminal, dtype=np.float64).tolist()}
    return values


def _costs(rows: list[dict[str, Any]], frozen: dict[str, Any], *, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The causal router and recurrent predictor are intentionally separate
    # interfaces.  V20 routes by renderer-visible motion but its frozen model
    # may legitimately consume latent, visual, or hybrid history.  Replaying
    # any other source would turn C3 into a new model-selection experiment.
    history_source = str(frozen.get("selected_history_source", ""))
    if history_source not in interaction.HISTORY_SOURCES:
        raise ValueError("V20 frozen selection has no supported history source")
    model_path, model_hash = interaction._resolve_frozen_checkpoint(
        frozen, V20_PANEL / "temporal_ood_v6_frozen_selection.json", expected_schema=interaction.FROZEN_SCHEMA,
    )
    recurrent, action, fit = interaction._load_frozen_model(
        model_path, model_hash, device=device, expected_contrast_weight=float(frozen["selected_contrast_weight"]),
        expected_history_source=history_source,
    )
    rec_effect, action_effect, state, goal = interaction._predict(
        recurrent, action, fit, rows, history_source=history_source)
    rec_cost = interaction._cost(state, goal, rec_effect, float(frozen["selected_temperature"]))
    action_cost = interaction._cost(state, goal, action_effect, 1.0)
    return rec_cost, action_cost, interaction._policy_cost(rows, rec_cost, action_cost, POLICY)


def _records(rows: list[dict[str, Any]], labels: dict[str, dict[str, float | list[float]]], routed: np.ndarray,
             fallback: np.ndarray) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups = interaction._group(rows)
    for root, root_rows in groups.items():
        indices = [rows.index(row) for row in root_rows]
        scores = np.asarray([routed[index] for index in indices], dtype=np.float64)
        ordered = np.sort(scores)
        risk = float(1.0 / max(float(ordered[1] - ordered[0]), 1.0e-6))
        for index in indices:
            row = rows[index]
            label = labels[interaction._key(row)]
            split = "test" if str(row["split"]) == "outcome" else str(row["split"])
            result.append({"root_id": root, "split": split, "candidate_id": int(row["candidate_id"]),
                           "action_sequence": row["action_sequence"], "wm_score": float(routed[index]),
                           "fallback_score": float(fallback[index]), "true_cost": float(label["real_task_cost"]),
                           "success": bool(float(label["real_task_cost"]) <= 0.25), "risk_score": risk,
                           "support_passed": True})
    return result


def main() -> int:
    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    pool = importlib.import_module(POOL_MODULE)
    pool.build()
    contexts, static = _contexts(), _ensure_preflight(_contexts())
    build = _build(contexts)
    context_path = OUTPUT / "temporal_ood_v6_context.json"
    _meta, rows = interaction._read_panel(context_path, interaction.CONTEXT_SCHEMA)
    frozen_path = V20_PANEL / "temporal_ood_v6_frozen_selection.json"
    frozen = _read(frozen_path)
    if str(frozen.get("selected_history_source", "")) not in interaction.HISTORY_SOURCES:
        raise RuntimeError("V21 may replay only a V20 frozen model with a declared history source")
    train_calibration = [row for row in rows if str(row["split"]) != "outcome"]
    calibration_labels = _labels(train_calibration)
    rec_tc, act_tc, route_tc = _costs(train_calibration, frozen, device="cuda")
    calibration_records = _records(train_calibration, calibration_labels, route_tc, act_tc)
    calibration_only = [row for row in calibration_records if row["split"] == "calibration"]
    bound = htsc.calibrate_contrast_bound(
        np.asarray([row["true_cost"] for row in calibration_only]), np.asarray([row["wm_score"] for row in calibration_only]),
        np.asarray([row["root_id"] for row in calibration_only]), sorted({row["root_id"] for row in calibration_only}), alpha=0.1,
    )
    root_risk = np.asarray([next(row["risk_score"] for row in calibration_only if row["root_id"] == root)
                            for root in sorted({row["root_id"] for row in calibration_only})])
    risk = htsc.calibrate_root_score_bound(root_risk, alpha=0.1)
    gate_path = OUTPUT / "v21_frozen_c3_gate.json"
    gate = {"schema": "pointmaze_c3_frozen_gate.v1", "v20_model_sha256": _sha256(frozen_path),
            "policy": POLICY, "contrast": bound, "risk": risk, "calibration_root_count": int(root_risk.size),
            "test_outcomes_opened": False,
            "claim_boundary": "Marginal fixed-bank calibration gate, not permit-conditional coverage."}
    encoded = json.dumps(gate, indent=2, sort_keys=True) + "\n"
    if gate_path.exists() and gate_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("V21 frozen C3 gate differs from its sealed calibration")
    if not gate_path.exists():
        gate_path.write_text(encoded, encoding="utf-8")
    outcome_rows = [row for row in rows if str(row["split"]) == "outcome"]
    outcome_path = OUTPUT / "v21_outcome_labels.json"
    if outcome_path.exists():
        raise FileExistsError("V21 final outcome labels already exist; refusing to reopen immutable outcomes")
    outcome_labels = _labels(outcome_rows)
    outcome_path.write_text(json.dumps({"schema": "pointmaze_c3_outcome_labels.v1", "gate_sha256": _sha256(gate_path),
                                        "labels": outcome_labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rec_o, act_o, route_o = _costs(outcome_rows, frozen, device="cuda")
    all_records = calibration_records + _records(outcome_rows, outcome_labels, route_o, act_o)
    report = htsc.evaluate_records(all_records, alpha=0.1, risk_quantile=0.9, bootstrap_draws=10_000, seed=202611506)
    if report["calibration"]["contrast"]["bound"] != bound["bound"] or report["calibration"]["risk_threshold"] != risk["bound"]:
        raise RuntimeError("V21 evaluator recalibration does not reproduce the sealed gate")
    report.update({"schema": "pointmaze_c3_calibrated_reliability_v21.v1", "runtime": runtime,
                   "cudnn_enabled": False, "build": build, "static_preflight": static,
                   "frozen_gate": str(gate_path.resolve()), "outcome_labels": str(outcome_path.resolve()),
                   "claim_boundary": "Fresh-root C3 offline fixed-bank result only. The contrast certificate is marginal, not permit-conditional; no online MPC, contact, or universal claim."})
    report_path = OUTPUT / "v21_c3_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(report_path), "release_rate": report["test"]["release_rate"],
                      "paired_regret_gain": report["test"]["paired_regret_gain_mean"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
