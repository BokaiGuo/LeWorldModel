"""Run a large fresh confirmation of V9's pre-outcome fixed latent policy.

V9's calibration authorization was non-positive, so its controller correctly
fell back.  Separately, its sealed outcome showed a positive recurrent versus
action-only difference.  V10 does not promote that outcome.  Instead it binds
the exact policy selected by V9 *before* V9 outcomes existed, removes all new
model/feature/policy search, and gives that one policy a larger independent
authorization panel followed by a larger outcome panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder
from experiments.decision_benchmark import run_pointmaze_counterfactual_contrast_v8_robust_cuda as v8
from experiments.decision_benchmark import run_pointmaze_counterfactual_contrast_v9_rgb_geometry_cuda as v9
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.decision_benchmark.extend_pointmaze_state_action_pool import build_extension
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = v9.DATASET
DATASET = ROOT / "data" / "point_maze_v30_fixed_counterfactual_confirmation_extension_20261022"
OUTPUT = ROOT / "results" / "pointmaze_counterfactual_contrast_v10_fixed_confirmation_20261022_cuda"
# The robust two-fold candidate design consumes 64 roots disjoint from all
# train/calibration/outcome roots.  The 1,024 evaluation roots below therefore
# require a 1,088-root extension.  V10's first, never-built v29 draft supplied
# only 1,024 roots and correctly failed before writing a panel; preserve that
# unused draft rather than overwriting it, and bind this executable protocol to
# a fresh v30 extension.
ROOT_RANGE = (11_832, 12_920)
SPLIT_COUNTS = {"train": 256, "calibration": 512, "outcome": 256}
EXTENSION_SEED = 202610501
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202610502, 202610503, 202610504
FIXED_POLICY = {
    "history_source": "source_history_model",
    "hidden": 16,
    "contrast_weight": 1.0,
    "temperature": 0.75,
    "policy": {"kind": "recurrent"},
}
V9_FROZEN = v9.OUTPUT / "temporal_ood_v6_frozen_selection.json"
V9_CONTEXT = v9.OUTPUT / "temporal_ood_v6_context.json"
CONTEXTS = tuple(dict.fromkeys([*v9.CONTEXTS, V9_CONTEXT]))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _v9_preoutcome_reference() -> str:
    """Bind V10 to V9's sealed pre-outcome selection, never V9 outcomes."""
    if not V9_FROZEN.is_file():
        raise FileNotFoundError(f"missing V9 frozen selection: {V9_FROZEN}")
    frozen = _read_json(V9_FROZEN)
    expected = {
        "selected_history_source": FIXED_POLICY["history_source"],
        "selected_hidden": FIXED_POLICY["hidden"],
        "selected_contrast_weight": FIXED_POLICY["contrast_weight"],
        "selected_temperature": FIXED_POLICY["temperature"],
        "selected_policy": FIXED_POLICY["policy"],
        "outcome_labels_opened": False,
        "outcome_labels_generated": False,
    }
    mismatch = {key: (frozen.get(key), value) for key, value in expected.items() if frozen.get(key) != value}
    if mismatch:
        raise ValueError(f"V10 fixed-policy contract does not match V9 pre-outcome selection: {mismatch}")
    return f"{V9_FROZEN.resolve()}#sha256={interaction._sha256(V9_FROZEN)}"


def _ensure_fresh_extension() -> dict[str, object]:
    expected = {
        "source_root": str(BASE_DATASET.resolve()), "seed": EXTENSION_SEED,
        "new_episode_count": ROOT_RANGE[1] - ROOT_RANGE[0], "base_root_count": ROOT_RANGE[0],
        "extended_root_count": ROOT_RANGE[1], "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
    }
    manifest = DATASET / "extension_manifest.json"
    if DATASET.exists():
        if not manifest.is_file():
            raise RuntimeError("V10 extension directory exists without an immutable manifest")
        payload = _read_json(manifest)
        mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatch:
            raise RuntimeError(f"V10 extension contract mismatch: {mismatch}")
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
        raise FileExistsError(f"refusing to reuse completed immutable V10 output path: {OUTPUT}")
    inputs = {
        "base_dataset": BASE_DATASET,
        "encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
        "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
        "dinov2": ROOT / "external" / "local_dinov2",
        "v9_context": V9_CONTEXT,
    }
    missing = [str(path) for path in [*inputs.values(), *CONTEXTS] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V10 requires bound historical inputs: {missing}")
    protocol_reference = _v9_preoutcome_reference()

    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    extension = _ensure_fresh_extension()
    if OUTPUT.exists():
        if not all(path.is_file() for path in expected_build):
            raise RuntimeError("V10 partial output is not a build-complete resumable state")
        model_path = interaction._frozen_model_path(frozen)
        score_dir = OUTPUT / "score"
        if not frozen.exists() and model_path.exists():
            raise RuntimeError("V10 found an unbound calibration checkpoint without frozen selection")
        if score_dir.exists() and not frozen.exists():
            raise RuntimeError("V10 found a partial score directory without frozen selection")
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
    robust_bank = v8._validate_robust_bank_contract(_read_json(OUTPUT / "build_v6_matched_report.json"))
    if frozen.exists():
        calibration_result = {"resumed_frozen_selection": str(frozen.resolve())}
    else:
        calibration_result = interaction.calibrate_fixed_policy(
            source, arrays, context, calibration, frozen, epochs=160, device="cuda", manifest=manifest,
            protocol_reference=protocol_reference,
            history_source=str(FIXED_POLICY["history_source"]), hidden=int(FIXED_POLICY["hidden"]),
            contrast_weight=float(FIXED_POLICY["contrast_weight"]), temperature=float(FIXED_POLICY["temperature"]),
        )
    score = interaction.score(source, arrays, context, frozen, OUTPUT / "score", device="cuda", manifest=manifest)
    (OUTPUT / "v10_cuda_runtime.json").write_text(json.dumps({
        "schema": "pointmaze_counterfactual_contrast_v10_fixed_confirmation_runtime.v1",
        "cuda_runtime": runtime, "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "recurrent_implementation": "manually unrolled torch.nn.GRUCell",
        "fixed_policy": FIXED_POLICY, "v9_preoutcome_reference": protocol_reference,
        "extension": extension, "build_report": build, "robust_bank_preflight": robust_bank,
        "calibration_result": calibration_result, "frozen_selection": str(frozen.resolve()),
        "score_report": str((OUTPUT / "score" / "temporal_ood_v6_score_report.json").resolve()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "active_policy": score["active_policy"],
        "authorization": calibration_result.get("authorization"), "fixed_policy": FIXED_POLICY,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
