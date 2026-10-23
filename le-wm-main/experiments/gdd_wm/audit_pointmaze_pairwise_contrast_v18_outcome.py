"""Read-only root-cluster audit for the independent sealed V18 C1 outcome.

V18 uses fresh roots and a policy fixed from V17 before V18 calibration labels
or outcomes were opened.  This audit performs no selection; it only reloads
the bound checkpoint and quantifies held-out uncertainty by resampling roots,
never individual action pairs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v12_outcome as root_audit
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "results" / "pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_20261023_cuda"
SOURCE = PANEL / "temporal_ood_v6_source_dataset.json"
ARRAYS = PANEL / "temporal_ood_v6_source_arrays.npz"
CONTEXT = PANEL / "temporal_ood_v6_context.json"
FROZEN = PANEL / "temporal_ood_v6_frozen_selection.json"
LABELS = PANEL / "score" / "temporal_ood_v6_outcome_labels.json"
MANIFEST = PANEL / "temporal_ood_v6_manifest.json"
OUTPUT = PANEL / "v18_outcome_root_cluster_audit.json"
MARGIN = 0.003
BOOTSTRAP_DRAWS = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit(*, device: str = "cuda") -> dict[str, Any]:
    """Audit already sealed V18 evidence; host CUDA is mandatory."""
    if str(device) != "cuda":
        raise ValueError("V18 outcome audit requires host CUDA; no CPU fallback")
    required = (SOURCE, ARRAYS, CONTEXT, FROZEN, LABELS, MANIFEST)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V18 outcome audit requires sealed artifacts: {missing}")

    frozen, label_payload = _read_object(FROZEN), _read_object(LABELS)
    if label_payload.get("generated_after_frozen_selection_sha256") != _sha256(FROZEN):
        raise ValueError("outcome labels are not bound to this frozen V18 selection")
    if frozen.get("outcome_labels_opened") is not False or frozen.get("outcome_labels_generated") is not False:
        raise ValueError("frozen V18 selection must precede outcome-label opening")
    if frozen.get("selected_policy") != {"kind": "recurrent"}:
        raise ValueError("V18 audit expects its frozen recurrent policy")
    if not bool(frozen.get("authorization", {}).get("permitted", False)):
        raise ValueError("V18 fixed-policy calibration did not authorize recurrent evaluation")
    if frozen.get("selected_history_source") != "source_history_model":
        raise ValueError("V18 must use the V17-frozen source-history model")

    calibration_path = Path(str(frozen.get("calibration_labels_path", "")))
    manifest = interaction._validate_manifest(MANIFEST, SOURCE, ARRAYS, CONTEXT, calibration_path)
    expected_counts = interaction._validated_root_counts(manifest["root_counts"])
    if frozen.get("manifest_sha256") != _sha256(MANIFEST):
        raise ValueError("frozen selection manifest does not match V18 audit manifest")
    if manifest.get("candidate_design") != "velocity_matched_braking_pairs":
        raise ValueError("V18 audit requires the declared velocity-alias candidate design")

    cuda_runtime = require_cuda_recurrent_cell(device)
    _source_meta, source_rows = interaction._read_panel(SOURCE, interaction.SOURCE_SCHEMA)
    context_meta, context_rows = interaction._read_panel(CONTEXT, interaction.CONTEXT_SCHEMA)
    interaction._validate_builder_contract(
        source_rows, context_rows, context_meta,
        strict_counts=True, expected_root_counts=expected_counts,
    )
    model_path, model_sha = interaction._resolve_frozen_checkpoint(
        frozen, FROZEN, expected_schema=interaction.FROZEN_SCHEMA,
    )
    history_source = str(frozen["selected_history_source"])
    recurrent, action, fit = interaction._load_frozen_model(
        model_path, model_sha, device=device,
        expected_contrast_weight=float(frozen["selected_contrast_weight"]),
        expected_history_source=history_source,
    )
    rows = [row for row in context_rows if str(row["split"]) == "outcome"]
    labels = label_payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("outcome label payload lacks labels")
    actual = np.asarray([float(labels[interaction._key(row)]["real_task_cost"]) for row in rows], dtype=np.float64)
    prediction, action_prediction, state, goal = interaction._predict(
        recurrent, action, fit, rows, history_source=history_source,
    )
    recurrent_cost = interaction._cost(state, goal, prediction, float(frozen["selected_temperature"]))
    action_cost = interaction._cost(state, goal, action_prediction, 1.0)
    recurrent_root = root_audit._root_pair_metrics(rows, recurrent_cost, actual)
    action_root = root_audit._root_pair_metrics(rows, action_cost, actual)
    sign_gain = root_audit._paired_root_bootstrap(
        recurrent_root, action_root, key="sign_accuracy", higher_is_better=True, seed=202611291,
    )
    mae_gain = root_audit._paired_root_bootstrap(
        recurrent_root, action_root, key="contrast_mae", higher_is_better=False, seed=202611292,
    )
    return {
        "schema": "pointmaze_pairwise_contrast_v18_outcome_root_cluster_audit.v1",
        "panel": str(PANEL.resolve()),
        "frozen_selection_sha256": _sha256(FROZEN),
        "outcome_labels_sha256": _sha256(LABELS),
        "manifest_sha256": _sha256(MANIFEST),
        "root_counts": expected_counts,
        "cuda_runtime": cuda_runtime,
        "margin": MARGIN,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "recurrent_pairwise": interaction.pairwise_counterfactual_contrast_metrics(rows, recurrent_cost, actual, margin=MARGIN),
        "action_only_pairwise": interaction.pairwise_counterfactual_contrast_metrics(rows, action_cost, actual, margin=MARGIN),
        "root_clustered_gain": {"sign_accuracy": sign_gain, "contrast_mae": mae_gain},
        "outcome_gain_authorized": bool(float(sign_gain["lower"]) > 0.0),
        "claim_boundary": (
            "Read-only post-outcome uncertainty audit for independently rooted V18 C1 confirmation. It changes neither the V17-frozen policy nor V18 authorization, and is not C2 repairability, C3 planning, closed-loop deployment, or a universal-world-model result."
        ),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable V18 outcome audit {OUTPUT}")
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "outcome_gain_authorized": result["outcome_gain_authorized"], "root_clustered_gain": result["root_clustered_gain"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
