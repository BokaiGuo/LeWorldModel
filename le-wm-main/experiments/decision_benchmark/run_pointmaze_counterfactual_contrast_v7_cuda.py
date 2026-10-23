"""Execute the sealed V7 counterfactual-contrast panel in one CUDA process.

The managed runtime can expose NVIDIA device nodes to a direct Python process
while dropping them for a shell child with rewritten library variables.  V7
therefore has a Python entry point: it keeps the process that passed CUDA
preflight through visual encoding, calibration, and sealed outcome scoring.
The model uses manually-unrolled ``GRUCell`` modules and explicitly disables
cuDNN, because the visible system cuDNN is older than this PyTorch build.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "pointmaze_counterfactual_contrast_v7_20261022_cuda"
CONTEXTS = [
    ROOT / relative for relative in (
        "results/pointmaze_temporal_ood_20261021_cuda/temporal_ood_context.json",
        "results/pointmaze_temporal_ood_v2_20261021_cuda/temporal_ood_v2_context.json",
        "results/pointmaze_temporal_ood_v3_20261021_cuda/temporal_ood_v3_context.json",
        "results/pointmaze_temporal_ood_v4_20261021_cuda/temporal_ood_v4_context.json",
        "results/pointmaze_temporal_ood_v5_20261021_cuda/temporal_ood_v5_context.json",
        "results/pointmaze_temporal_ood_v6_1_matched_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_2_matched_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_3_matched_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_4_goal_sensitivity_20261021_cuda/goal_sensitivity_context.json",
        "results/pointmaze_temporal_ood_v6_4_in_support_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_5_observability_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_6_history_gate_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_7_state_interaction_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_8_goal_sensitivity_20261021_cuda_retry/goal_sensitivity_context.json",
        "results/pointmaze_temporal_ood_v6_8_state_interaction_20261021_cuda/temporal_ood_v6_context.json",
        "results/pointmaze_temporal_ood_v6_matched_20261021_cuda/temporal_ood_v6_context.json",
    )
]


def main() -> int:
    source = OUTPUT / "temporal_ood_v6_source_dataset.json"
    arrays = OUTPUT / "temporal_ood_v6_source_arrays.npz"
    context = OUTPUT / "temporal_ood_v6_context.json"
    calibration = OUTPUT / "temporal_ood_v6_calibration_outcomes.json"
    manifest = OUTPUT / "temporal_ood_v6_manifest.json"
    frozen = OUTPUT / "temporal_ood_v6_frozen_selection.json"
    outcome = OUTPUT / "score" / "temporal_ood_v6_outcome_labels.json"
    expected_build = (source, arrays, context, calibration, manifest)
    if OUTPUT.exists() and outcome.exists():
        raise FileExistsError(f"refusing to reuse completed immutable V7 output path: {OUTPUT}")
    inputs = {
        "dataset": ROOT / "data" / "point_maze_tail1800_1899",
        "encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
        "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
        "dinov2": ROOT / "external" / "local_dinov2",
    }
    missing = [str(path) for path in [*inputs.values(), *CONTEXTS] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V7 requires existing bound inputs: {missing}")

    # Conv/attention retain CUDA execution through native kernels, while this
    # prevents the incompatible system cuDNN from being selected.  The exact
    # unrolled-GRUCell CUDA path is preflighted before artifacts are written.
    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    if OUTPUT.exists():
        if not all(path.is_file() for path in expected_build) or frozen.exists():
            raise RuntimeError("V7 partial output is not a build-complete, pre-calibration resumable state")
        build = {"resumed_existing_sealed_build": True}
    else:
        build = builder.build_panel(
            inputs["dataset"], CONTEXTS, OUTPUT,
            encoder_checkpoint=inputs["encoder"], pca_checkpoint=inputs["pca"], dinov2_source=inputs["dinov2"],
            candidate_generalization_regime="shared_candidate_in_support",
            candidate_design="state_interaction_pairs", goal_qpos=[0.8, 0.8], device="cuda",
        )
    calibration_result = interaction.calibrate(
        source, arrays, context, calibration, frozen, epochs=100, device="cuda",
        observability_suite=True, history_gate=True, manifest=manifest,
    )
    score = interaction.score(
        source, arrays, context, frozen, OUTPUT / "score", device="cuda",
        observability_suite=True, history_gate=True, manifest=manifest,
    )
    runtime_path = OUTPUT / "v7_cuda_runtime.json"
    runtime_path.write_text(json.dumps({
        "schema": "pointmaze_counterfactual_contrast_v7_runtime.v1",
        "cuda_runtime": runtime,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "recurrent_implementation": "manually unrolled torch.nn.GRUCell",
        "build_report": build,
        "frozen_selection": str(frozen.resolve()),
        "score_report": str((OUTPUT / "score" / "temporal_ood_v6_score_report.json").resolve()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "active_policy": score["active_policy"],
        "authorization": calibration_result["authorization"],
        "selected_contrast_weight": calibration_result["selected_contrast_weight"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
