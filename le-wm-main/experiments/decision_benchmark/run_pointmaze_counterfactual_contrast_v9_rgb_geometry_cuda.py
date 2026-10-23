"""Run a fresh RGB-geometry repair test after V8's latent-only failure.

V8 ruled out its original candidate-bank explanation: the fixed bank passed
two-fold, outcome-blind headroom checks, yet frozen-latent history did not
beat its action-only control.  The matched builder already records a causal
``source_history_visual_state`` series extracted from rendered RGB frames, but
the V6--V8 repair family never consumed that field.  V9 tests three *declared*
history families under one frozen calibration protocol:

* frozen latent history;
* RGB-derived geometry history; and
* their concatenation.

This is a representation-observability intervention, not an outcome-guided
fix.  The RGB geometry channel is renderer-calibrated and is explicitly not a
claim about a frozen-latent-only LeWM or a universal visual world model.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder
from experiments.decision_benchmark import run_pointmaze_counterfactual_contrast_v8_robust_cuda as v8
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.decision_benchmark.extend_pointmaze_state_action_pool import build_extension
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = ROOT / "data" / "point_maze_v27_robust_counterfactual_extension_20261022"
DATASET = ROOT / "data" / "point_maze_v28_rgb_geometry_counterfactual_extension_20261022"
OUTPUT = ROOT / "results" / "pointmaze_counterfactual_contrast_v9_rgb_geometry_20261022_cuda"
ROOT_RANGE = (11_320, 11_832)
SPLIT_COUNTS = {"train": 128, "calibration": 192, "outcome": 64}
EXTENSION_SEED = 202610491
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202610492, 202610493, 202610494
HISTORY_SOURCES = (
    "source_history_model",
    "source_history_visual_state",
    "hybrid_latent_visual_state",
)
V8_CONTEXT = v8.OUTPUT / "temporal_ood_v6_context.json"
CONTEXTS = tuple(dict.fromkeys([*v8.CONTEXTS, V8_CONTEXT]))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ensure_fresh_extension() -> dict[str, object]:
    expected = {
        "source_root": str(BASE_DATASET.resolve()),
        "seed": EXTENSION_SEED,
        "new_episode_count": ROOT_RANGE[1] - ROOT_RANGE[0],
        "base_root_count": ROOT_RANGE[0],
        "extended_root_count": ROOT_RANGE[1],
        "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
    }
    manifest = DATASET / "extension_manifest.json"
    if DATASET.exists():
        if not manifest.is_file():
            raise RuntimeError("V9 extension directory exists without an immutable manifest")
        payload = _read_json(manifest)
        mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatch:
            raise RuntimeError(f"V9 extension contract mismatch: {mismatch}")
        return {"resumed_existing_extension": True, "manifest": str(manifest.resolve())}
    return build_extension(
        BASE_DATASET, DATASET, episode_count=ROOT_RANGE[1] - ROOT_RANGE[0], seed=EXTENSION_SEED,
    )


def main() -> int:
    source = OUTPUT / "temporal_ood_v6_source_dataset.json"
    arrays = OUTPUT / "temporal_ood_v6_source_arrays.npz"
    context = OUTPUT / "temporal_ood_v6_context.json"
    calibration = OUTPUT / "temporal_ood_v6_calibration_outcomes.json"
    manifest = OUTPUT / "temporal_ood_v6_manifest.json"
    frozen = OUTPUT / "temporal_ood_v6_frozen_selection.json"
    outcome = OUTPUT / "score" / "temporal_ood_v6_outcome_labels.json"
    expected_build = (source, arrays, context, calibration, manifest)
    if outcome.exists():
        raise FileExistsError(f"refusing to reuse completed immutable V9 output path: {OUTPUT}")
    inputs = {
        "base_dataset": BASE_DATASET,
        "encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
        "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
        "dinov2": ROOT / "external" / "local_dinov2",
        "v8_context": V8_CONTEXT,
    }
    missing = [str(path) for path in [*inputs.values(), *CONTEXTS] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V9 requires bound historical inputs: {missing}")

    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    extension = _ensure_fresh_extension()

    if OUTPUT.exists():
        if not all(path.is_file() for path in expected_build):
            raise RuntimeError("V9 partial output is not a build-complete resumable state")
        model_path = interaction._frozen_model_path(frozen)
        score_dir = OUTPUT / "score"
        if not frozen.exists() and model_path.exists():
            raise RuntimeError("V9 found an unbound calibration checkpoint without frozen selection")
        if score_dir.exists() and not frozen.exists():
            raise RuntimeError("V9 found a partial score directory without frozen selection")
        build: dict[str, object] = {"resumed_existing_sealed_build": True}
    else:
        build = builder.build_panel(
            DATASET, list(CONTEXTS), OUTPUT,
            encoder_checkpoint=inputs["encoder"], pca_checkpoint=inputs["pca"], dinov2_source=inputs["dinov2"],
            candidate_generalization_regime="shared_candidate_in_support",
            candidate_design="robust_goal_balanced_interaction_pairs", goal_qpos=[0.8, 0.8], device="cuda",
            root_seed=ROOT_SEED, held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
            split_counts=SPLIT_COUNTS, root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
        )
    build_report = _read_json(OUTPUT / "build_v6_matched_report.json")
    robust_bank = v8._validate_robust_bank_contract(build_report)
    if frozen.exists():
        calibration_result = {"resumed_frozen_selection": str(frozen.resolve())}
    else:
        calibration_result = interaction.calibrate(
            source, arrays, context, calibration, frozen, epochs=160, device="cuda",
            observability_suite=True, history_gate=True, manifest=manifest,
            history_sources=HISTORY_SOURCES,
        )
    score = interaction.score(
        source, arrays, context, frozen, OUTPUT / "score", device="cuda",
        observability_suite=True, history_gate=True, manifest=manifest,
    )
    (OUTPUT / "v9_cuda_runtime.json").write_text(json.dumps({
        "schema": "pointmaze_counterfactual_contrast_v9_rgb_geometry_runtime.v1",
        "cuda_runtime": runtime, "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "recurrent_implementation": "manually unrolled torch.nn.GRUCell",
        "history_sources": list(HISTORY_SOURCES),
        "rgb_geometry_contract": "causal source_history_visual_state captured by the matched renderer; no source_state_raw is an input to the deployable candidate arms",
        "extension": extension, "build_report": build, "robust_bank_preflight": robust_bank,
        "calibration_result": calibration_result,
        "frozen_selection": str(frozen.resolve()),
        "score_report": str((OUTPUT / "score" / "temporal_ood_v6_score_report.json").resolve()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "active_policy": score["active_policy"],
        "authorization": calibration_result.get("authorization"),
        "selected_history_source": score["selected_history_source"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
