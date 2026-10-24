"""Run V24: fresh-root motion-routed, stratum-calibrated epsilon-regret C3.

V23's opened audit showed that a single mixed-motion envelope transported well
only to the high-motion roots where recurrent repair had decision value. V24
predeclares the repair route and calibrates the two observable strata
separately before opening any outcome labels.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder
from experiments.decision_benchmark import run_htsc_matched_transfer as htsc
from experiments.decision_benchmark import run_pointmaze_c3_regret_budget_v23_cuda as v23
from experiments.decision_benchmark import run_pointmaze_mixed_velocity_repairability_v20_cuda as v20
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.decision_benchmark.root_observation_identity import assert_unique_root_observations
from experiments.gdd_wm import preflight_pointmaze_c3_stratified_repair_v24 as preflight
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "point_maze_v75_c3_stratified_repair_20261023"
OUTPUT = ROOT / "results" / "pointmaze_c3_stratified_repair_v24_20261023_cuda"
PREFLIGHT = ROOT / "results" / "pointmaze_c3_stratified_repair_v24_static_preflight_20261023.json"
ROOT_RANGE, SPLIT_COUNTS = (32_456, 33_288), {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611804, 202611805, 202611806
EPOCHS, HIDDEN, CONTRAST_WEIGHT, TEMPERATURE = 160, 32, 1.0, 1.0
REGRET_BUDGET = float(preflight.REGRET_BUDGET)
POLICY = {"kind": "history_energy_gate", "direction": "recurrent_when_high", "threshold": .02,
          "motion_source": "source_history_visual_state", "fixed_threshold": True}
POOL_MODULE = "experiments.decision_benchmark.build_pointmaze_v24_c3_stratified_repair_pool"
# Successors with deliberately paired physical sources set this mapping before
# calling ``main``.  The default preserves the original one-root-per-unit
# protocol for historical runs.
CLUSTER_BY_ROOT: dict[str, str] | None = None


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def _contexts() -> tuple[Path, ...]:
    required = {
        *v20._prior_contexts(),
        ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_20261023_cuda" / "temporal_ood_v6_context.json",
        ROOT / "results" / "pointmaze_c3_calibrated_reliability_v21_20261023_cuda" / "temporal_ood_v6_context.json",
        ROOT / "results" / "pointmaze_c3_action_supported_v22_20261023_cuda" / "temporal_ood_v6_context.json",
        ROOT / "results" / "pointmaze_c3_regret_budget_v23_20261023_cuda" / "temporal_ood_v6_context.json",
    }
    paths = tuple(sorted(path.resolve() for path in required))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(f"V24 requires every frozen predecessor context: {missing}")
    return paths


def _ensure_preflight(contexts: tuple[Path, ...]) -> dict[str, Any]:
    result = preflight.preflight(dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
                                 held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
                                 root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS)
    if result["status"] != "passed_static_preflight": raise RuntimeError(f"V24 static preflight rejected: {result['checks']}")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if PREFLIGHT.exists() and PREFLIGHT.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing V24 preflight differs from its frozen contract")
    PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    if not PREFLIGHT.exists(): PREFLIGHT.write_text(encoded, encoding="utf-8")
    return result


def _build(contexts: tuple[Path, ...]) -> dict[str, Any]:
    names = ("temporal_ood_v6_source_dataset.json", "temporal_ood_v6_source_arrays.npz", "temporal_ood_v6_context.json",
             "temporal_ood_v6_calibration_outcomes.json", "temporal_ood_v6_manifest.json")
    if OUTPUT.exists():
        if not all((OUTPUT / name).is_file() for name in names): raise RuntimeError("V24 output lacks complete sealed pre-outcome panel")
        return {"resumed_existing_sealed_build": True}
    inputs = {"encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
              "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
              "dinov2": ROOT / "external" / "local_dinov2"}
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing: raise FileNotFoundError(f"V24 requires bound feature inputs: {missing}")
    return builder.build_panel(DATASET, list(contexts), OUTPUT, encoder_checkpoint=inputs["encoder"], pca_checkpoint=inputs["pca"],
                               dinov2_source=inputs["dinov2"], candidate_generalization_regime="shared_candidate_in_support",
                               candidate_design="velocity_matched_braking_pairs", goal_qpos=[.8, .8], device="cuda",
                               root_seed=ROOT_SEED, held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
                               split_counts=SPLIT_COUNTS, root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
                               frozen_preflight_ledger=PREFLIGHT)


def _strata(rows: list[dict[str, Any]]) -> dict[str, str]:
    mask = interaction._history_gate_mask(rows, POLICY)
    result: dict[str, str] = {}
    for row, enabled in zip(rows, mask):
        root = str(row["instance_id"])
        value = "high_motion" if bool(enabled) else "direct_zero_motion"
        if root in result and result[root] != value: raise ValueError("V24 stratum is not root-constant")
        result[root] = value
    if set(result.values()) != {"high_motion", "direct_zero_motion"}: raise ValueError("V24 needs both predeclared strata")
    return result


def _records(rows: list[dict[str, Any]], labels: dict[str, dict[str, float | list[float]]], routed: np.ndarray, fallback: np.ndarray) -> list[dict[str, Any]]:
    records = v23._records(rows, labels, routed, fallback)
    mapping = _strata(rows)
    for row in records: row["stratum"] = mapping[str(row["root_id"])]
    return records


def _costs(rows: list[dict[str, Any]], frozen: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_source = str(frozen.get("selected_history_source", ""))
    model_path, model_hash = interaction._resolve_frozen_checkpoint(
        frozen, OUTPUT / "temporal_ood_v6_frozen_selection.json", expected_schema=interaction.FROZEN_SCHEMA,
    )
    recurrent, action, fit = interaction._load_frozen_model(
        model_path, model_hash, device="cuda", expected_contrast_weight=CONTRAST_WEIGHT,
        expected_history_source=history_source,
    )
    rec_effect, act_effect, state, goal = interaction._predict(recurrent, action, fit, rows, history_source=history_source)
    rec_cost, act_cost = interaction._cost(state, goal, rec_effect, TEMPERATURE), interaction._cost(state, goal, act_effect, 1.0)
    return rec_cost, act_cost, interaction._policy_cost(rows, rec_cost, act_cost, POLICY)


def _calibration_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in ("high_motion", "direct_zero_motion"):
        rows = [row for row in records if row["split"] == "calibration" and row["stratum"] == stratum]
        roots = sorted({str(row["root_id"]) for row in rows})
        root_contrast = htsc.calibrate_contrast_bound(np.asarray([row["true_cost"] for row in rows]), np.asarray([row["wm_score"] for row in rows]),
                                                      np.asarray([row["root_id"] for row in rows]), roots, alpha=.1)
        clusters = {root: root for root in roots} if CLUSTER_BY_ROOT is None else {
            root: CLUSTER_BY_ROOT[root] for root in roots
        }
        contrast = htsc._clustered_conformal_calibration(root_contrast, clusters, alpha=.1)
        root_risks = {
            root: next(float(row["risk_score"]) for row in rows if str(row["root_id"]) == root)
            for root in roots
        }
        risks = np.asarray(list(htsc._cluster_maximum(root_risks, clusters).values()))
        risk = htsc.calibrate_root_score_bound(risks, alpha=.1)
        if contrast["bound"] is None or risk["bound"] is None: raise RuntimeError("V24 stratum calibration is insufficient")
        result[stratum] = {"contrast": contrast, "risk": risk, "calibration_root_count": len(roots),
                           "calibration_cluster_count": len(set(clusters.values()))}
    return result


def _combine(strata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    roots = [row for report in strata.values() for row in report["test"]["roots"]]
    coverage_roots = [row for report in strata.values() for row in report["test"]["conformal_outcome_coverage"]["roots"]]
    gains, successes = np.asarray([row["paired_regret_gain"] for row in roots]), np.asarray([row["paired_success_gain"] for row in roots])
    covered = [bool(row["covered"]) for row in coverage_roots]
    count = len(roots)
    failures = {key: sum(report["test"]["gate_failure_counts"][key] for report in strata.values()) for key in ("support_failure", "uncertainty_failure", "certificate_margin_failure")}
    return {"root_count": count, "released_root_count": int(sum(row["status"] == "permit" for row in roots)),
            "release_rate": float(np.mean([row["status"] == "permit" for row in roots])), "roots": roots,
            "gate_failure_counts": failures, "paired_regret_gain_mean": float(gains.mean()), "paired_success_gain_mean": float(successes.mean()),
            "paired_regret_gain_bootstrap": htsc._bootstrap_mean(gains, draws=10_000, seed=202611807),
            "paired_success_gain_bootstrap": htsc._bootstrap_mean(successes, draws=10_000, seed=202611808),
            "posthoc_utility_ranking_failure_count": int(sum(row["posthoc_utility_ranking_failure"] for row in roots)),
            "conformal_outcome_coverage": {"outcome_root_count": count, "covered_root_count": int(sum(covered)),
                                             "empirical_root_coverage": float(np.mean(covered)),
                                             "covered_root_regret_bound_all_hold": bool(all(row["regret_bound_holds_when_covered"] for row in coverage_roots if row["regret_bound_holds_when_covered"] is not None)),
                                             "roots": coverage_roots}}


def main() -> int:
    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    pool = importlib.import_module(POOL_MODULE).build()
    contexts, static = _contexts(), _ensure_preflight(_contexts())
    build = _build(contexts)
    source, arrays, context = OUTPUT / "temporal_ood_v6_source_dataset.json", OUTPUT / "temporal_ood_v6_source_arrays.npz", OUTPUT / "temporal_ood_v6_context.json"
    calibration, manifest, frozen_path = OUTPUT / "temporal_ood_v6_calibration_outcomes.json", OUTPUT / "temporal_ood_v6_manifest.json", OUTPUT / "temporal_ood_v6_frozen_selection.json"
    if not frozen_path.exists():
        interaction.calibrate_fixed_policy(source, arrays, context, calibration, frozen_path, hidden=HIDDEN, contrast_weight=CONTRAST_WEIGHT,
                                           temperature=TEMPERATURE, history_source="source_history_model", epochs=EPOCHS, device="cuda", manifest=manifest,
                                           protocol_reference="V24 fixed visual-motion route with stratum-specific C3 calibration", policy=POLICY)
    frozen = _read(frozen_path)
    _meta, rows = interaction._read_panel(context, interaction.CONTEXT_SCHEMA)
    # A new root ID is not an independent conformal draw when the renderer
    # input and physical source state are byte-for-byte the same.
    assert_unique_root_observations(rows)
    nonoutcome = [row for row in rows if str(row["split"]) != "outcome"]
    calibration_labels = v23._labels(nonoutcome)
    rec, act, routed = _costs(nonoutcome, frozen)
    del rec
    calibration_records = _records(nonoutcome, calibration_labels, routed, act)
    gate = _calibration_gate(calibration_records)
    gate_path = OUTPUT / "v24_frozen_stratified_gate.json"
    payload = {"schema": "pointmaze_c3_stratified_repair_frozen_gate.v1", "frozen_model_sha256": v23._sha256(frozen_path), "policy": POLICY,
               "certificate_kind": "regret_budget", "regret_budget": REGRET_BUDGET, "strata": gate, "test_outcomes_opened": False,
               "claim_boundary": "Stratum-wise marginal fixed-bank epsilon-regret calibration; not conditional coverage or online MPC."}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if gate_path.exists() and gate_path.read_text(encoding="utf-8") != encoded: raise RuntimeError("existing V24 gate differs from sealed calibration")
    if not gate_path.exists(): gate_path.write_text(encoded, encoding="utf-8")
    outcome_rows = [row for row in rows if str(row["split"]) == "outcome"]
    outcome_path = OUTPUT / "v24_outcome_labels.json"
    if outcome_path.exists(): raise FileExistsError("V24 final outcome labels already exist; refusing to reopen immutable outcomes")
    outcome_labels = v23._labels(outcome_rows)
    outcome_path.write_text(json.dumps({"schema": "pointmaze_c3_stratified_repair_outcome_labels.v1", "gate_sha256": v23._sha256(gate_path), "labels": outcome_labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rec_o, act_o, routed_o = _costs(outcome_rows, frozen)
    del rec_o
    all_records = calibration_records + _records(outcome_rows, outcome_labels, routed_o, act_o)
    reports: dict[str, dict[str, Any]] = {}
    for stratum in ("high_motion", "direct_zero_motion"):
        report = htsc.evaluate_records([row for row in all_records if row["stratum"] == stratum], alpha=.1, risk_quantile=.9,
                                       bootstrap_draws=10_000, seed=202611809 if stratum == "high_motion" else 202611810,
                                       certificate_kind="regret_budget", regret_budget=REGRET_BUDGET,
                                       cluster_by_root=None if CLUSTER_BY_ROOT is None else {
                                           str(row["root_id"]): CLUSTER_BY_ROOT[str(row["root_id"])]
                                           for row in all_records if row["stratum"] == stratum
                                       })
        if report["calibration"]["contrast"]["bound"] != gate[stratum]["contrast"]["bound"] or report["calibration"]["risk_threshold"] != gate[stratum]["risk"]["bound"]:
            raise RuntimeError("V24 evaluator does not reproduce sealed stratum calibration")
        reports[stratum] = report
    combined = _combine(reports)
    report = {"schema": "pointmaze_c3_stratified_repair_v24.v1", "runtime": runtime, "cudnn_enabled": False, "pool": pool, "build": build,
              "static_preflight": static, "frozen_gate": str(gate_path.resolve()), "outcome_labels": str(outcome_path.resolve()),
              "protocol": {"train_calibration_test_root_disjoint": True, "candidate_generation": "fixed before calibration and test scoring",
                           "route": preflight.ROUTE_CONTRACT, "calibration": "separate fixed-bank rootwise envelopes per observable motion stratum",
                           "certificate_kind": "regret_budget", "regret_budget": REGRET_BUDGET, "test_outcome_used_for_selection": False},
              "calibration": {"by_stratum": {name: report["calibration"] for name, report in reports.items()}}, "test": combined, "strata": reports,
              "claim_boundary": "Fresh-root offline fixed-bank stratum-calibrated C3 test only; no conditional safety, live MPC, contact, or universal claim."}
    report_path = OUTPUT / "v24_c3_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(report_path), "release_rate": combined["release_rate"], "paired_regret_gain": combined["paired_regret_gain_mean"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
