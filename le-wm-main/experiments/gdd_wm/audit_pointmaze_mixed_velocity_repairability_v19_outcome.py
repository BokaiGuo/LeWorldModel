"""Read-only root-cluster audit for the frozen V19 prospective C2 route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v12_outcome as root_audit
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "results" / "pointmaze_mixed_velocity_repairability_v19_20261023_cuda"
SOURCE = PANEL / "temporal_ood_v6_source_dataset.json"
ARRAYS = PANEL / "temporal_ood_v6_source_arrays.npz"
CONTEXT = PANEL / "temporal_ood_v6_context.json"
FROZEN = PANEL / "temporal_ood_v6_frozen_selection.json"
LABELS = PANEL / "score" / "temporal_ood_v6_outcome_labels.json"
SCORE = PANEL / "score" / "temporal_ood_v6_score_report.json"
MANIFEST = PANEL / "temporal_ood_v6_manifest.json"
OUTPUT = PANEL / "v19_outcome_root_cluster_audit.json"
POLICY = {
    "kind": "history_energy_gate", "direction": "recurrent_when_high", "threshold": 0.02,
    "motion_source": "source_history_visual_state", "fixed_threshold": True,
}


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


def _root_metric(selection: dict[str, Any]) -> dict[str, dict[str, float]]:
    roots = selection.get("roots")
    if not isinstance(roots, dict) or not roots:
        raise ValueError("selection lacks per-root outcomes")
    return {str(root): {"regret": float(value["regret"])} for root, value in roots.items()}


def _strata(records: list[dict[str, Any]], routed_mask: np.ndarray) -> dict[str, list[str]]:
    grouped = interaction._group(records)
    result = {"history_routed": [], "action_fallback": []}
    index_by_root: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        index_by_root.setdefault(str(row["instance_id"]), []).append(index)
    for root in grouped:
        values = routed_mask[np.asarray(index_by_root[root], dtype=np.int64)]
        if not np.all(values == values[0]):
            raise ValueError("V19 route must be candidate-invariant within a root")
        result["history_routed" if bool(values[0]) else "action_fallback"].append(root)
    if set(result["history_routed"]) & set(result["action_fallback"]):
        raise ValueError("V19 strata overlap")
    return result


def _selection_for_roots(records: list[dict[str, Any]], cost: np.ndarray, actual: np.ndarray, roots: list[str]) -> dict[str, Any]:
    wanted = set(roots)
    indices = [index for index, row in enumerate(records) if str(row["instance_id"]) in wanted]
    return interaction._selection([records[index] for index in indices], cost[indices], actual[indices])


def _gain(method: dict[str, Any], baseline: dict[str, Any], *, seed: int) -> dict[str, float | int | str]:
    result = root_audit._paired_root_bootstrap(
        _root_metric(method), _root_metric(baseline), key="regret", higher_is_better=False, seed=seed,
    )
    result["interpretation"] = "positive means the first-listed policy has lower root regret than the second-listed policy"
    return result


def _gain_difference(
    high_method: dict[str, Any], high_baseline: dict[str, Any],
    low_method: dict[str, Any], low_baseline: dict[str, Any], *, seed: int,
) -> dict[str, float | int | str]:
    """Bootstrap the incremental repair gain in high-motion over direct roots."""
    def paired(method: dict[str, Any], baseline: dict[str, Any]) -> np.ndarray:
        method_metric, baseline_metric = _root_metric(method), _root_metric(baseline)
        roots = sorted(method_metric)
        if roots != sorted(baseline_metric):
            raise ValueError("stratum comparison has mismatched root identities")
        return np.asarray(
            [baseline_metric[root]["regret"] - method_metric[root]["regret"] for root in roots],
            dtype=np.float64,
        )

    high, low = paired(high_method, high_baseline), paired(low_method, low_baseline)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(high, size=len(high), replace=True).mean()
        - rng.choice(low, size=len(low), replace=True).mean()
        for _ in range(root_audit.BOOTSTRAP_DRAWS)
    ])
    return {
        "high_root_count": int(len(high)), "low_root_count": int(len(low)),
        "mean_gain_difference": float(high.mean() - low.mean()),
        "lower": float(np.quantile(draws, 0.025)), "upper": float(np.quantile(draws, 0.975)),
        "interpretation": "positive means recurrent repair reduces regret more in history-routed roots than action-fallback roots",
    }


def audit(*, device: str = "cuda", comparison_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    if str(device) != "cuda":
        raise ValueError("V19 outcome audit requires host CUDA; no CPU fallback")
    required = (SOURCE, ARRAYS, CONTEXT, FROZEN, LABELS, SCORE, MANIFEST)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V19 outcome audit requires sealed artifacts: {missing}")
    frozen, labels_payload, score_payload = _read(FROZEN), _read(LABELS), _read(SCORE)
    if frozen.get("selected_policy") != POLICY or frozen.get("history_gate_enabled") is not True:
        raise ValueError("V19 does not bind the predeclared visual-motion route")
    if labels_payload.get("generated_after_frozen_selection_sha256") != _sha256(FROZEN):
        raise ValueError("V19 outcome labels are not bound to frozen route receipt")
    if score_payload.get("frozen_selection_sha256") != _sha256(FROZEN):
        raise ValueError("V19 score is not bound to frozen route receipt")
    if score_payload.get("active_policy") not in {"history_energy_gate", "action_only_fallback"}:
        raise ValueError("V19 score has an invalid active policy")

    calibration_path = Path(str(frozen.get("calibration_labels_path", "")))
    manifest = interaction._validate_manifest(MANIFEST, SOURCE, ARRAYS, CONTEXT, calibration_path)
    expected = interaction._validated_root_counts(manifest["root_counts"])
    cuda_runtime = require_cuda_recurrent_cell(device)
    _meta, source_rows = interaction._read_panel(SOURCE, interaction.SOURCE_SCHEMA)
    context_meta, context_rows = interaction._read_panel(CONTEXT, interaction.CONTEXT_SCHEMA)
    interaction._validate_builder_contract(source_rows, context_rows, context_meta, strict_counts=True, expected_root_counts=expected)
    model_path, model_sha = interaction._resolve_frozen_checkpoint(frozen, FROZEN, expected_schema=interaction.FROZEN_SCHEMA)
    recurrent, action, fit = interaction._load_frozen_model(
        model_path, model_sha, device=device, expected_contrast_weight=1.0, expected_history_source="source_history_model",
    )
    rows = [row for row in context_rows if str(row["split"]) == "outcome"]
    labels = labels_payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("V19 outcome labels lack labels")
    actual = np.asarray([float(labels[interaction._key(row)]["real_task_cost"]) for row in rows], dtype=np.float64)
    prediction, action_prediction, state, goal = interaction._predict(recurrent, action, fit, rows, history_source="source_history_model")
    recurrent_cost = interaction._cost(state, goal, prediction, 1.0)
    action_cost = interaction._cost(state, goal, action_prediction, 1.0)
    routed_cost = interaction._policy_cost(rows, recurrent_cost, action_cost, POLICY)
    routed_mask = interaction._history_gate_mask(rows, POLICY)
    stratified_roots = _strata(rows, routed_mask)
    selections = {
        "routed": interaction._selection(rows, routed_cost, actual),
        "action_only": interaction._selection(rows, action_cost, actual),
        "recurrent": interaction._selection(rows, recurrent_cost, actual),
    }
    strata: dict[str, Any] = {}
    for index, (name, roots) in enumerate(stratified_roots.items()):
        routed = _selection_for_roots(rows, routed_cost, actual, roots)
        fallback = _selection_for_roots(rows, action_cost, actual, roots)
        recurrent_selection = _selection_for_roots(rows, recurrent_cost, actual, roots)
        strata[name] = {
            "root_count": len(roots), "routed": routed, "action_only": fallback, "recurrent": recurrent_selection,
            "routed_over_action_only": _gain(routed, fallback, seed=202611310 + index),
            "recurrent_over_action_only": _gain(recurrent_selection, fallback, seed=202611320 + index),
            "routed_over_recurrent": _gain(routed, recurrent_selection, seed=202611330 + index),
        }
    routed_over_action = _gain(selections["routed"], selections["action_only"], seed=202611309)
    recurrent_over_action = _gain(selections["recurrent"], selections["action_only"], seed=202611319)
    routed_over_recurrent = _gain(selections["routed"], selections["recurrent"], seed=202611329)
    repair_gain_heterogeneity = _gain_difference(
        strata["history_routed"]["recurrent"], strata["history_routed"]["action_only"],
        strata["action_fallback"]["recurrent"], strata["action_fallback"]["action_only"],
        seed=202611339,
    )
    equal_budget_control: dict[str, Any] | None = None
    if comparison_policy is not None:
        control_mask = interaction._history_gate_mask(rows, comparison_policy)
        if not np.isclose(float(np.mean(control_mask)), float(np.mean(routed_mask)), rtol=0.0, atol=0.0):
            raise ValueError("equal-budget control does not use the same recurrent fraction as the declared route")
        control_cost = interaction._policy_cost(rows, recurrent_cost, action_cost, comparison_policy)
        control_selection = interaction._selection(rows, control_cost, actual)
        equal_budget_control = {
            "policy": dict(comparison_policy), "recurrent_fraction": float(np.mean(control_mask)),
            "selection": control_selection,
            "routed_over_control": _gain(selections["routed"], control_selection, seed=202611349),
        }
    result = {
        "schema": "pointmaze_mixed_velocity_repairability_v19_outcome_root_cluster_audit.v1",
        "panel": str(PANEL.resolve()), "frozen_selection_sha256": _sha256(FROZEN),
        "outcome_labels_sha256": _sha256(LABELS), "score_report_sha256": _sha256(SCORE),
        "cuda_runtime": cuda_runtime, "root_counts": expected,
        "routed_mask_fraction": float(np.mean(routed_mask)), "selections": selections,
        "routed_over_action_only": routed_over_action,
        "recurrent_over_action_only": recurrent_over_action,
        "routed_over_recurrent": routed_over_recurrent,
        "repair_gain_heterogeneity": repair_gain_heterogeneity,
        "strata": strata,
        "outcome_gain_authorized": bool(float(routed_over_action["lower"]) > 0.0),
        "recurrent_dominates_route": bool(float(routed_over_recurrent["upper"]) < 0.0),
        "repair_gain_higher_when_history_routed": bool(float(repair_gain_heterogeneity["lower"]) > 0.0),
        "claim_boundary": "Read-only post-outcome audit of a pre-fixed V19 mechanism route. It is a C2 mechanism test only if the route beats action-only and the reported repair-gain heterogeneity supports its predeclared mechanism; it does not establish dominance over always-recurrent repair, C3 calibration, online MPC, contact transfer, or universal repairability.",
    }
    if equal_budget_control is not None:
        result["equal_budget_control"] = equal_budget_control
        result["route_beats_equal_budget_control"] = bool(float(equal_budget_control["routed_over_control"]["lower"]) > 0.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="new immutable audit path")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable V19 outcome audit {output}")
    result = audit()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "outcome_gain_authorized": result["outcome_gain_authorized"],
        "routed_over_action_only": result["routed_over_action_only"],
        "routed_over_recurrent": result["routed_over_recurrent"],
        "repair_gain_heterogeneity": result["repair_gain_heterogeneity"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
