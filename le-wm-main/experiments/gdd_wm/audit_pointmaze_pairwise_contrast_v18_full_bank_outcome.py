"""Read-only full-bank decision audit for independently rooted V18 C1 evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v12_outcome as root_audit
from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v18_outcome as pair_audit
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "results" / "pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_20261023_cuda"
SOURCE = PANEL / "temporal_ood_v6_source_dataset.json"
ARRAYS = PANEL / "temporal_ood_v6_source_arrays.npz"
CONTEXT = PANEL / "temporal_ood_v6_context.json"
FROZEN = PANEL / "temporal_ood_v6_frozen_selection.json"
LABELS = PANEL / "score" / "temporal_ood_v6_outcome_labels.json"
SCORE = PANEL / "score" / "temporal_ood_v6_score_report.json"
MANIFEST = PANEL / "temporal_ood_v6_manifest.json"
OUTPUT = PANEL / "v18_full_bank_outcome_root_cluster_audit.json"
BOOTSTRAP_DRAWS = 10_000


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _root_metrics(selection: dict[str, Any]) -> dict[str, dict[str, float]]:
    roots = selection.get("roots")
    if not isinstance(roots, dict) or not roots:
        raise ValueError("selection lacks per-root outcome records")
    result: dict[str, dict[str, float]] = {}
    for root, row in roots.items():
        if not isinstance(row, dict):
            raise ValueError("per-root selection record must be an object")
        regret = float(row["regret"])
        if not np.isfinite(regret) or regret < 0.0:
            raise ValueError("root regret must be finite and non-negative")
        result[str(root)] = {"regret": regret, "top1_accuracy": float(bool(row["ranking_correct"]))}
    return result


def audit(*, device: str = "cuda") -> dict[str, Any]:
    """Recompute V18 frozen predictions and root-cluster full-bank CIs."""
    if str(device) != "cuda":
        raise ValueError("V18 full-bank audit requires host CUDA; no CPU fallback")
    required = (SOURCE, ARRAYS, CONTEXT, FROZEN, LABELS, SCORE, MANIFEST)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V18 full-bank audit requires sealed artifacts: {missing}")
    frozen, labels_payload, score_payload = _read_object(FROZEN), _read_object(LABELS), _read_object(SCORE)
    if labels_payload.get("generated_after_frozen_selection_sha256") != pair_audit._sha256(FROZEN):
        raise ValueError("outcome labels are not bound to V18 frozen selection")
    if score_payload.get("frozen_selection_sha256") != pair_audit._sha256(FROZEN):
        raise ValueError("score report is not bound to V18 frozen selection")
    if score_payload.get("outcome_labels_sha256") != pair_audit._sha256(LABELS):
        raise ValueError("score report is not bound to V18 outcome labels")
    if score_payload.get("active_policy") != "recurrent":
        raise ValueError("V18 score report did not activate its authorized recurrent policy")

    calibration_path = Path(str(frozen.get("calibration_labels_path", "")))
    manifest = interaction._validate_manifest(MANIFEST, SOURCE, ARRAYS, CONTEXT, calibration_path)
    expected_counts = interaction._validated_root_counts(manifest["root_counts"])
    if manifest.get("candidate_design") != "velocity_matched_braking_pairs":
        raise ValueError("V18 full-bank audit requires the declared velocity-alias bank")
    cuda_runtime = require_cuda_recurrent_cell(device)
    _source_meta, source_rows = interaction._read_panel(SOURCE, interaction.SOURCE_SCHEMA)
    context_meta, context_rows = interaction._read_panel(CONTEXT, interaction.CONTEXT_SCHEMA)
    interaction._validate_builder_contract(source_rows, context_rows, context_meta, strict_counts=True, expected_root_counts=expected_counts)
    model_path, model_sha = interaction._resolve_frozen_checkpoint(frozen, FROZEN, expected_schema=interaction.FROZEN_SCHEMA)
    history_source = str(frozen["selected_history_source"])
    recurrent, action, fit = interaction._load_frozen_model(
        model_path, model_sha, device=device,
        expected_contrast_weight=float(frozen["selected_contrast_weight"]),
        expected_history_source=history_source,
    )
    rows = [row for row in context_rows if str(row["split"]) == "outcome"]
    labels = labels_payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("outcome label payload lacks labels")
    actual = np.asarray([float(labels[interaction._key(row)]["real_task_cost"]) for row in rows], dtype=np.float64)
    prediction, action_prediction, state, goal = interaction._predict(recurrent, action, fit, rows, history_source=history_source)
    recurrent_cost = interaction._cost(state, goal, prediction, float(frozen["selected_temperature"]))
    action_cost = interaction._cost(state, goal, action_prediction, 1.0)
    recurrent_selection = interaction._selection(rows, recurrent_cost, actual)
    action_selection = interaction._selection(rows, action_cost, actual)
    recurrent_root = _root_metrics(recurrent_selection)
    action_root = _root_metrics(action_selection)
    top1_gain = root_audit._paired_root_bootstrap(recurrent_root, action_root, key="top1_accuracy", higher_is_better=True, seed=202611293)
    regret_reduction = root_audit._paired_root_bootstrap(recurrent_root, action_root, key="regret", higher_is_better=False, seed=202611294)
    return {
        "schema": "pointmaze_pairwise_contrast_v18_full_bank_outcome_root_cluster_audit.v1",
        "panel": str(PANEL.resolve()),
        "frozen_selection_sha256": pair_audit._sha256(FROZEN),
        "outcome_labels_sha256": pair_audit._sha256(LABELS),
        "score_report_sha256": pair_audit._sha256(SCORE),
        "root_counts": expected_counts,
        "cuda_runtime": cuda_runtime,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "recomputed_selection": {"recurrent": recurrent_selection, "action_only": action_selection},
        "root_clustered_gain": {"top1_accuracy": top1_gain, "regret_reduction": regret_reduction},
        "full_bank_gain_authorized": bool(float(top1_gain["lower"]) > 0.0 and float(regret_reduction["lower"]) > 0.0),
        "claim_boundary": (
            "Read-only held-out V18 full-bank consequence audit. It is an independently rooted fixed shared candidate-bank result, not C2 repairability, HTSC, online MPC, live visual deployment, or a universal-world-model result."
        ),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable V18 full-bank audit {OUTPUT}")
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "full_bank_gain_authorized": result["full_bank_gain_authorized"], "root_clustered_gain": result["root_clustered_gain"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
