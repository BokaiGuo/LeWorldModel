"""Run V23: independently fitted, fixed-bank epsilon-regret C3 evaluation.

V22 rejected a stronger unique-winner release because near-tied physical
actions are finer than its simultaneous contrast envelope.  V23 does not
relax that envelope.  It trains one predeclared rootwise-contrast repair on
fresh training roots, calibrates it on fresh calibration roots, then releases
only when the same fixed-bank envelope implies a predeclared epsilon-regret
bound.  Outcome labels remain sealed until the gate has been written.
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
from experiments.decision_benchmark import run_pointmaze_mixed_velocity_repairability_v20_cuda as v20
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.decision_benchmark import run_unseen_action_bank_pointmaze as simulator
from experiments.gdd_wm import preflight_pointmaze_c3_regret_budget_v23 as preflight
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "point_maze_v72_c3_regret_budget_20261023"
OUTPUT = ROOT / "results" / "pointmaze_c3_regret_budget_v23_20261023_cuda"
PREFLIGHT = ROOT / "results" / "pointmaze_c3_regret_budget_v23_static_preflight_20261023.json"
ROOT_RANGE, SPLIT_COUNTS = (30_664, 31_496), {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611704, 202611705, 202611706
EPOCHS, HIDDEN, CONTRAST_WEIGHT, TEMPERATURE = 160, 32, 1.0, 1.0
REGRET_BUDGET = float(preflight.REGRET_BUDGET)
POLICY = {"kind": "recurrent"}
POOL_MODULE = "experiments.decision_benchmark.build_pointmaze_v23_c3_regret_budget_pool"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _contexts() -> tuple[Path, ...]:
    required = {
        *v20._prior_contexts(),
        ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_20261023_cuda" / "temporal_ood_v6_context.json",
        ROOT / "results" / "pointmaze_c3_calibrated_reliability_v21_20261023_cuda" / "temporal_ood_v6_context.json",
        ROOT / "results" / "pointmaze_c3_action_supported_v22_20261023_cuda" / "temporal_ood_v6_context.json",
    }
    paths = tuple(sorted(path.resolve() for path in required))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V23 requires every frozen predecessor context: {missing}")
    return paths


def _ensure_preflight(contexts: tuple[Path, ...]) -> dict[str, Any]:
    result = preflight.preflight(
        dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
        held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    if result["status"] != "passed_static_preflight":
        raise RuntimeError(f"V23 static preflight rejected: {result['checks']}")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if PREFLIGHT.exists() and PREFLIGHT.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing V23 preflight differs from its frozen contract")
    PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    if not PREFLIGHT.exists():
        PREFLIGHT.write_text(encoded, encoding="utf-8")
    return result


def _build(contexts: tuple[Path, ...]) -> dict[str, Any]:
    names = ("temporal_ood_v6_source_dataset.json", "temporal_ood_v6_source_arrays.npz",
             "temporal_ood_v6_context.json", "temporal_ood_v6_calibration_outcomes.json",
             "temporal_ood_v6_manifest.json")
    if OUTPUT.exists():
        if not all((OUTPUT / name).is_file() for name in names):
            raise RuntimeError("V23 output exists but lacks a complete sealed pre-outcome panel")
        return {"resumed_existing_sealed_build": True}
    inputs = {
        "encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
        "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
        "dinov2": ROOT / "external" / "local_dinov2",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V23 requires bound feature inputs: {missing}")
    return builder.build_panel(
        DATASET, list(contexts), OUTPUT, encoder_checkpoint=inputs["encoder"], pca_checkpoint=inputs["pca"],
        dinov2_source=inputs["dinov2"], candidate_generalization_regime="shared_candidate_in_support",
        candidate_design="velocity_matched_braking_pairs", goal_qpos=[0.8, 0.8], device="cuda",
        root_seed=ROOT_SEED, held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        split_counts=SPLIT_COUNTS, root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
        frozen_preflight_ledger=PREFLIGHT,
    )


def _labels(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | list[float]]]:
    result: dict[str, dict[str, float | list[float]]] = {}
    for row in rows:
        terminal = simulator._simulate_sequence(
            np.asarray(row["source_state_raw"], dtype=np.float64),
            np.asarray(row["action_sequence"], dtype=np.float64),
        )
        goal = np.asarray(row["goal_raw"], dtype=np.float64)
        result[interaction._key(row)] = {
            "real_task_cost": float(np.linalg.norm(terminal[:2] - goal[:2])),
            "realized_terminal_state": np.asarray(terminal, dtype=np.float64).tolist(),
        }
    return result


def _costs(rows: list[dict[str, Any]], frozen: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_source = str(frozen.get("selected_history_source", ""))
    model_path, model_hash = interaction._resolve_frozen_checkpoint(
        frozen, OUTPUT / "temporal_ood_v6_frozen_selection.json", expected_schema=interaction.FROZEN_SCHEMA,
    )
    recurrent, action, fit = interaction._load_frozen_model(
        model_path, model_hash, device="cuda", expected_contrast_weight=CONTRAST_WEIGHT,
        expected_history_source=history_source,
    )
    recurrent_effect, action_effect, state, goal = interaction._predict(
        recurrent, action, fit, rows, history_source=history_source,
    )
    recurrent_cost = interaction._cost(state, goal, recurrent_effect, TEMPERATURE)
    action_cost = interaction._cost(state, goal, action_effect, 1.0)
    return recurrent_cost, action_cost, recurrent_cost


def _records(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, float | list[float]]], routed: np.ndarray,
    fallback: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = interaction._group(rows)
    positions = {id(row): index for index, row in enumerate(rows)}
    for root, root_rows in groups.items():
        indices = [positions[id(row)] for row in root_rows]
        scores = np.asarray([routed[index] for index in indices], dtype=np.float64)
        ordered = np.sort(scores)
        risk = float(1.0 / max(float(ordered[1] - ordered[0]), 1.0e-6))
        for index in indices:
            row, label = rows[index], labels[interaction._key(rows[index])]
            output.append({
                "root_id": root, "split": "test" if str(row["split"]) == "outcome" else str(row["split"]),
                "candidate_id": int(row["candidate_id"]), "action_sequence": row["action_sequence"],
                "wm_score": float(routed[index]), "fallback_score": float(fallback[index]),
                "true_cost": float(label["real_task_cost"]), "success": bool(float(label["real_task_cost"]) <= .25),
                "risk_score": risk, "support_passed": True,
            })
    return output


def main() -> int:
    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    pool = importlib.import_module(POOL_MODULE)
    pool_result = pool.build()
    contexts = _contexts()
    static = _ensure_preflight(contexts)
    build = _build(contexts)
    source, arrays = OUTPUT / "temporal_ood_v6_source_dataset.json", OUTPUT / "temporal_ood_v6_source_arrays.npz"
    context, calibration = OUTPUT / "temporal_ood_v6_context.json", OUTPUT / "temporal_ood_v6_calibration_outcomes.json"
    manifest, frozen_path = OUTPUT / "temporal_ood_v6_manifest.json", OUTPUT / "temporal_ood_v6_frozen_selection.json"
    if not frozen_path.exists():
        interaction.calibrate_fixed_policy(
            source, arrays, context, calibration, frozen_path, hidden=HIDDEN, contrast_weight=CONTRAST_WEIGHT,
            temperature=TEMPERATURE, history_source="source_history_model", epochs=EPOCHS, device="cuda",
            manifest=manifest, protocol_reference="V23 fixed rootwise task-contrast repair; no calibration model selection",
            policy=POLICY,
        )
    frozen = _read(frozen_path)
    _meta, rows = interaction._read_panel(context, interaction.CONTEXT_SCHEMA)
    non_outcome = [row for row in rows if str(row["split"]) != "outcome"]
    calibration_labels = _labels(non_outcome)
    _recurrent, action, routed = _costs(non_outcome, frozen)
    calibration_records = _records(non_outcome, calibration_labels, routed, action)
    calibration_only = [row for row in calibration_records if row["split"] == "calibration"]
    contrast = htsc.calibrate_contrast_bound(
        np.asarray([row["true_cost"] for row in calibration_only]), np.asarray([row["wm_score"] for row in calibration_only]),
        np.asarray([row["root_id"] for row in calibration_only]), sorted({row["root_id"] for row in calibration_only}), alpha=.1,
    )
    root_risk = np.asarray([next(row["risk_score"] for row in calibration_only if row["root_id"] == root)
                            for root in sorted({row["root_id"] for row in calibration_only})])
    risk = htsc.calibrate_root_score_bound(root_risk, alpha=.1)
    gate_path = OUTPUT / "v23_frozen_c3_gate.json"
    gate = {
        "schema": "pointmaze_c3_regret_budget_frozen_gate.v1", "frozen_model_sha256": _sha256(frozen_path),
        "policy": POLICY, "contrast": contrast, "risk": risk, "certificate_kind": "regret_budget",
        "regret_budget": REGRET_BUDGET, "calibration_root_count": int(root_risk.size),
        "test_outcomes_opened": False,
        "claim_boundary": "Marginal fixed-bank epsilon-regret gate; not permit-conditional coverage or online MPC.",
    }
    encoded = json.dumps(gate, indent=2, sort_keys=True) + "\n"
    if gate_path.exists() and gate_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing V23 gate differs from its sealed calibration")
    if not gate_path.exists():
        gate_path.write_text(encoded, encoding="utf-8")
    outcome_rows = [row for row in rows if str(row["split"]) == "outcome"]
    outcome_path = OUTPUT / "v23_outcome_labels.json"
    if outcome_path.exists():
        raise FileExistsError("V23 final outcome labels already exist; refusing to reopen immutable outcomes")
    outcome_labels = _labels(outcome_rows)
    outcome_path.write_text(json.dumps({"schema": "pointmaze_c3_regret_budget_outcome_labels.v1", "gate_sha256": _sha256(gate_path), "labels": outcome_labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rec_o, act_o, route_o = _costs(outcome_rows, frozen)
    report = htsc.evaluate_records(
        calibration_records + _records(outcome_rows, outcome_labels, route_o, act_o), alpha=.1, risk_quantile=.9,
        bootstrap_draws=10_000, seed=202611707, certificate_kind="regret_budget", regret_budget=REGRET_BUDGET,
    )
    if report["calibration"]["contrast"]["bound"] != contrast["bound"] or report["calibration"]["risk_threshold"] != risk["bound"]:
        raise RuntimeError("V23 evaluator does not reproduce sealed calibration")
    report.update({
        "schema": "pointmaze_c3_regret_budget_v23.v1", "runtime": runtime, "cudnn_enabled": False,
        "pool": pool_result, "build": build, "static_preflight": static, "frozen_gate": str(gate_path.resolve()),
        "outcome_labels": str(outcome_path.resolve()),
        "claim_boundary": "Fresh-root offline fixed-bank epsilon-regret C3 test only; no permit-conditional, live MPC, contact, or universal guarantee.",
    })
    report_path = OUTPUT / "v23_c3_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(report_path), "release_rate": report["test"]["release_rate"],
                      "paired_regret_gain": report["test"]["paired_regret_gain_mean"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
