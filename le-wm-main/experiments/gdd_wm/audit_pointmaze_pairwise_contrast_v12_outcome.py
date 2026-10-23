"""Root-clustered, read-only uncertainty audit for V12's sealed C1 outcome.

The paired candidates within a root share the same visual history, so the
2,048 held-out pair scores are not independent evidence. This audit reloads
the already frozen V12 model and already materialized outcome labels, then
reports root-cluster bootstrap intervals for both sign accuracy and contrast
MAE. It never changes model selection, authorization, or deployment policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "results" / "pointmaze_pairwise_contrast_v12_balance_20261022_cuda"
SOURCE, ARRAYS, CONTEXT = (
    PANEL / "temporal_ood_v6_source_dataset.json",
    PANEL / "temporal_ood_v6_source_arrays.npz",
    PANEL / "temporal_ood_v6_context.json",
)
FROZEN = PANEL / "temporal_ood_v6_frozen_selection.json"
LABELS = PANEL / "score" / "temporal_ood_v6_outcome_labels.json"
MANIFEST = PANEL / "temporal_ood_v6_manifest.json"
OUTPUT = PANEL / "v12_outcome_root_cluster_audit.json"
MARGIN, BOOTSTRAP_DRAWS = 0.003, 10_000


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


def _root_pair_metrics(records: list[dict[str, Any]], predicted: np.ndarray, actual: np.ndarray) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[int, int]] = defaultdict(dict)
    for index, row in enumerate(records):
        root, candidate_id = str(row["instance_id"]), int(row["candidate_id"])
        if candidate_id in positions[root]:
            raise ValueError(f"duplicate candidate {candidate_id} under root {root}")
        positions[root][candidate_id] = index
    result: dict[str, dict[str, float]] = {}
    for root, indexed in sorted(positions.items()):
        ids = sorted(indexed)
        if ids != list(range(len(ids))) or len(ids) % 2:
            raise ValueError("V12 audit requires contiguous, adjacent action pairs")
        signs, errors = [], []
        for left in range(0, len(ids), 2):
            right = left + 1
            truth_delta = float(actual[indexed[left]] - actual[indexed[right]])
            if abs(truth_delta) < MARGIN:
                continue
            predicted_delta = float(predicted[indexed[left]] - predicted[indexed[right]])
            signs.append(float(np.sign(truth_delta) == np.sign(predicted_delta)))
            errors.append(abs(predicted_delta - truth_delta))
        if not signs:
            raise ValueError(f"root {root} has no outcome pair beyond the frozen margin")
        result[root] = {
            "sign_accuracy": float(np.mean(signs)), "contrast_mae": float(np.mean(errors)),
            "valid_pair_count": float(len(signs)),
        }
    return result


def _paired_root_bootstrap(
    method: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]], *,
    key: str, higher_is_better: bool, seed: int,
) -> dict[str, float | int]:
    roots = sorted(method)
    if roots != sorted(baseline):
        raise ValueError("method/baseline root identities differ")
    raw = np.asarray([method[root][key] - baseline[root][key] for root in roots], dtype=np.float64)
    gain = raw if higher_is_better else -raw
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(gain, size=len(gain), replace=True).mean() for _ in range(BOOTSTRAP_DRAWS)])
    return {
        "root_count": int(len(gain)), "mean_gain": float(gain.mean()),
        "lower": float(np.quantile(draws, 0.025)), "upper": float(np.quantile(draws, 0.975)),
        "interpretation": f"positive means recurrent has {'higher' if higher_is_better else 'lower'} {key}",
    }


def audit(*, device: str = "cuda") -> dict[str, Any]:
    if str(device) != "cuda":
        raise ValueError("V12 outcome audit requires host CUDA; no CPU fallback")
    required = (SOURCE, ARRAYS, CONTEXT, FROZEN, LABELS, MANIFEST)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V12 outcome audit requires sealed artifacts: {missing}")
    frozen, label_payload = _read_object(FROZEN), _read_object(LABELS)
    if label_payload.get("generated_after_frozen_selection_sha256") != _sha256(FROZEN):
        raise ValueError("outcome labels are not bound to this frozen selection")
    if frozen.get("outcome_labels_opened") is not False:
        raise ValueError("frozen selection must precede outcome-label opening")
    calibration_path = Path(str(frozen.get("calibration_labels_path", "")))
    manifest = interaction._validate_manifest(MANIFEST, SOURCE, ARRAYS, CONTEXT, calibration_path)
    expected_root_counts = interaction._validated_root_counts(manifest["root_counts"])
    if frozen.get("manifest_sha256") != _sha256(MANIFEST):
        raise ValueError("frozen selection manifest does not match V12 audit manifest")
    cuda_runtime = require_cuda_recurrent_cell(device)
    _source_meta, source_rows = interaction._read_panel(SOURCE, interaction.SOURCE_SCHEMA)
    context_meta, context_rows = interaction._read_panel(CONTEXT, interaction.CONTEXT_SCHEMA)
    interaction._validate_builder_contract(
        source_rows, context_rows, context_meta, strict_counts=True,
        expected_root_counts=expected_root_counts,
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
    recurrent_root = _root_pair_metrics(rows, recurrent_cost, actual)
    action_root = _root_pair_metrics(rows, action_cost, actual)
    return {
        "schema": "pointmaze_pairwise_contrast_v12_outcome_root_cluster_audit.v1",
        "panel": str(PANEL.resolve()), "frozen_selection_sha256": _sha256(FROZEN),
        "outcome_labels_sha256": _sha256(LABELS), "manifest_sha256": _sha256(MANIFEST),
        "root_counts": expected_root_counts, "cuda_runtime": cuda_runtime,
        "margin": MARGIN, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "recurrent_pairwise": interaction.pairwise_counterfactual_contrast_metrics(rows, recurrent_cost, actual, margin=MARGIN),
        "action_only_pairwise": interaction.pairwise_counterfactual_contrast_metrics(rows, action_cost, actual, margin=MARGIN),
        "root_clustered_gain": {
            "sign_accuracy": _paired_root_bootstrap(recurrent_root, action_root, key="sign_accuracy", higher_is_better=True, seed=202611221),
            "contrast_mae": _paired_root_bootstrap(recurrent_root, action_root, key="contrast_mae", higher_is_better=False, seed=202611222),
        },
        "claim_boundary": "Read-only post-outcome uncertainty audit for sealed C1 V12. It changes neither hyperparameters nor authorization and is not a C2 repairability, C3 planning, closed-loop, or universal-world-model result.",
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable V12 outcome audit {OUTPUT}")
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "root_clustered_gain": result["root_clustered_gain"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
