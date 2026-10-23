"""Run V11's fresh, host-CUDA pairwise counterfactual-identifiability test.

V10 established that a 16-way action-bank gain cannot be authorized when its
full-bank lower-tail margin is too small.  V11 does not relax that planning
criterion.  Instead it asks the narrower C1 question exposed by the static
audit: can a visual history model rank the predeclared, same-summary action
pairs whose preferred member changes across states?

The primary held-out outcomes are pairwise sign accuracy, contrast MAE, and
valid-pair coverage.  The usual full-bank regret remains in the score artifact
only as a secondary diagnostic; this driver makes no planning-success claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder
from experiments.decision_benchmark import extend_pointmaze_state_action_pool as extension_tool
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v11 as v11
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = ROOT / "data" / "point_maze_v53_probe_balanced_residual_v16_extension_20261022"
DATASET = ROOT / "data" / "point_maze_v54_pairwise_contrast_v11_extension_20261022"
PREFLIGHT = ROOT / "results" / "pointmaze_pairwise_contrast_v11_static_preflight_20261022.json"
OUTPUT = ROOT / "results" / "pointmaze_pairwise_contrast_v11_20261022_cuda"
ROOT_RANGE = (19_720, 20_552)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
EXTENSION_SEED = 202611201
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611202, 202611203, 202611204
EPOCHS = 160
RUN_LABEL = "V11"
RUN_SCHEMA = "pointmaze_pairwise_contrast_v11_runtime.v1"
RUNTIME_FILENAME = "v11_cuda_runtime.json"
PREFLIGHT_MODULE = v11
CANDIDATE_DESIGN = "pairwise_contrast_robust_interaction_pairs"
HISTORY_GATE = False
CLAIM_BOUNDARY = (
    "C1 counterfactual-identifiability evaluation only; not a C2 repairability, "
    "C3 planning, live-visual, or universal-world-model result."
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prior_contexts() -> tuple[Path, ...]:
    """Freeze every compatible historical panel identity before root selection."""
    # Once the V11 static ledger exists, it is the sole admissible identity
    # universe for every resume.  Rediscovering `results/**/context*.json`
    # after panel construction would include V11's own context and make its
    # previously frozen root range appear unavailable.
    if PREFLIGHT.is_file():
        ledger = _read_json(PREFLIGHT)
        frozen = ledger.get("prior_context_sha256")
        if not isinstance(frozen, dict) or not frozen:
            raise RuntimeError("V11 frozen preflight lacks its prior-context hash ledger")
        contexts = tuple(Path(str(path)).resolve() for path in sorted(frozen))
        missing = [str(path) for path in contexts if not path.is_file()]
        if missing:
            raise RuntimeError(f"V11 frozen preflight contexts are unavailable: {missing}")
        mismatched = {
            str(path): (frozen[str(path)], _sha256(path))
            for path in contexts if frozen.get(str(path)) != _sha256(path)
        }
        if mismatched:
            raise RuntimeError(f"V11 frozen preflight context bytes changed: {mismatched}")
        return contexts
    found: list[Path] = []
    for path in sorted((ROOT / "results").rglob("*context*.json")):
        if path.resolve().is_relative_to(OUTPUT.resolve()):
            continue
        try:
            payload = _read_json(path)
            metadata, records = payload.get("metadata"), payload.get("records")
            if (
                isinstance(metadata, dict)
                and isinstance(metadata.get("candidate_family"), str)
                and isinstance(records, list)
                and all(
                    isinstance(row, dict)
                    and "start_episode_id" in row
                    and "candidate_action_hash" in row
                    and "candidate_template_fingerprint" in row
                    for row in records
                )
            ):
                found.append(path.resolve())
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            continue
    if not found:
        raise RuntimeError("V11 needs at least one compatible prior context for identity exclusion")
    return tuple(found)


def _ensure_extension() -> dict[str, Any]:
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
            raise RuntimeError("V11 extension directory exists without immutable manifest")
        payload = _read_json(manifest)
        mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatch:
            raise RuntimeError(f"V11 extension contract mismatch: {mismatch}")
        return {"resumed_existing_extension": True, "manifest": str(manifest.resolve())}
    return extension_tool.build_extension(
        BASE_DATASET, DATASET, episode_count=ROOT_RANGE[1] - ROOT_RANGE[0], seed=EXTENSION_SEED,
    )


def _ensure_preflight(contexts: tuple[Path, ...]) -> dict[str, Any]:
    result = PREFLIGHT_MODULE.preflight(
        dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
        held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    if result["status"] != "passed_static_preflight":
        raise RuntimeError(f"{RUN_LABEL} pairwise static preflight rejected before model fitting: {result['checks']}")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if PREFLIGHT.exists():
        existing = _read_json(PREFLIGHT)
        # A host-only admission wrapper may append immutable execution
        # provenance (for example extension receipt and context count) after
        # this core preflight was computed.  Those audit-only fields neither
        # affect root/action selection nor appear in the builder contract.
        # Compare the canonical preflight payload, not incidental receipt
        # annotations, so a valid sealed ledger cannot spuriously block CUDA.
        for key in ("extension", "prior_context_count"):
            existing.pop(key, None)
        if json.dumps(existing, indent=2, sort_keys=True) + "\n" != encoded:
            raise RuntimeError(f"existing {RUN_LABEL} preflight does not match the deterministic frozen contract")
    else:
        PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
        PREFLIGHT.write_text(encoded, encoding="utf-8")
    return result


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
        raise FileExistsError(f"refusing to reopen immutable V11 outcome path: {OUTPUT}")
    inputs = {
        "base_dataset": BASE_DATASET,
        "encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
        "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
        "dinov2": ROOT / "external" / "local_dinov2",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V11 requires bound inputs: {missing}")

    # GPU admission precedes extension and all learned work.  The driver is
    # only launched from its host wrapper, never as a sandbox CPU fallback.
    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    extension = _ensure_extension()
    contexts = _prior_contexts()
    preflight = _ensure_preflight(contexts)

    if OUTPUT.exists():
        if not all(path.is_file() for path in expected_build):
            raise RuntimeError("V11 partial output is not a build-complete resumable state")
        model_path = interaction._frozen_model_path(frozen)
        if not frozen.exists() and model_path.exists():
            raise RuntimeError("V11 found an unbound calibration checkpoint without frozen selection")
        if (OUTPUT / "score").exists() and not frozen.exists():
            raise RuntimeError("V11 found score state without its frozen selection")
        build: dict[str, Any] = {"resumed_existing_sealed_build": True}
    else:
        build = builder.build_panel(
            DATASET, list(contexts), OUTPUT,
            encoder_checkpoint=inputs["encoder"], pca_checkpoint=inputs["pca"], dinov2_source=inputs["dinov2"],
            candidate_generalization_regime="shared_candidate_in_support",
            candidate_design=CANDIDATE_DESIGN, goal_qpos=[0.8, 0.8], device="cuda",
            root_seed=ROOT_SEED, held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
            split_counts=SPLIT_COUNTS, root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
            frozen_preflight_ledger=PREFLIGHT,
        )

    if frozen.exists():
        calibration_result: dict[str, Any] = {"resumed_frozen_selection": str(frozen.resolve())}
    else:
        calibration_result = interaction.calibrate_pairwise_contrast(
            source, arrays, context, calibration, frozen, epochs=EPOCHS, device="cuda", manifest=manifest,
        )
    score = interaction.score(
        source, arrays, context, frozen, OUTPUT / "score", device="cuda", manifest=manifest,
        history_gate=bool(HISTORY_GATE),
    )
    pairwise = score["pairwise_counterfactual_contrast"]
    if not all(value.get("status") == "evaluated" for value in pairwise.values()):
        raise RuntimeError("V11 pairwise score is unexpectedly not applicable despite its frozen pair contract")
    runtime_path = OUTPUT / RUNTIME_FILENAME
    runtime_path.write_text(json.dumps({
        "schema": RUN_SCHEMA,
        "cuda_runtime": runtime, "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "recurrent_implementation": "manually unrolled torch.nn.GRUCell",
        "extension": extension, "prior_context_count": len(contexts),
        "prior_context_sha256": {str(path): _sha256(path) for path in contexts},
        "static_preflight": preflight, "build_report": build,
        "calibration_result": calibration_result, "frozen_selection": str(frozen.resolve()),
        "primary_endpoint": "held-out pairwise sign accuracy, contrast MAE, and valid-pair coverage",
        "secondary_diagnostic": "full-bank planner regret only",
        "score_report": str((OUTPUT / "score" / "temporal_ood_v6_score_report.json").resolve()),
        "claim_boundary": CLAIM_BOUNDARY,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "active_policy": score["active_policy"],
        "authorization": calibration_result.get("authorization"), "pairwise": pairwise,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
