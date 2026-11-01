"""Development-only static dev303 blind-RGB candidate diagnostic; no physics."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from . import p1_rgb_boundary_contact_witness_20261102_exclusive as witness

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results/p1_dev303_blind_adapter_20261101_codex_MAIN02.json"
NPZ = ROOT / "results/p1_dev303_20261101_codex_blind_inputs.npz"
MANIFEST_SHA = "b7b5a1b2394acdad13981c1fb4258d97941e0f1863e816e9dc86d85b91b460dc"
NPZ_SHA = "092af72253c9b0463e5c7a5814052599843dc4b6a979bd040d7f318c93dcddd9"
MOTION_SOURCE = ROOT / "experiments/spectral_reachability/p1_dev302_mixed_prefix_bank_v5_20261101_codex.py"
CONFIG = {
    "schema": witness.CONFIG_SCHEMA,
    "current_action_pixels": 29.567 * (224.0 / 512.0),
    "previous_action_pixels": 0.02993 * 374.1 * (224.0 / 512.0),
    "radius_pixels": 6.5,
    # New development protocol parameters; not old-guard calibrated values.
    "clearance_pixels": 1.0, "steps": 5, "max_action": 1.0,
    "min_component_pixels": 8, "max_candidates": 8,
    "target_separation_pixels": 4.0, "min_goal_displacement_pixels": 1.0,
    "min_alignment": 0.5, "model_id": "dev302_v5_noncontact_rgb_motion_development_only",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def static_check(manifest_raw, npz_raw, motion_source_sha):
    if sha(manifest_raw) != MANIFEST_SHA or sha(npz_raw) != NPZ_SHA:
        raise ValueError("fixed manifest or blind NPZ SHA mismatch")
    manifest = json.loads(manifest_raw)
    if not (manifest["blind_sha256"] == NPZ_SHA and manifest["seed"] == 301000003
            and manifest["P1_admission"] is False
            and manifest["independent_confirmation"] is False):
        raise ValueError("development manifest boundary")
    with np.load(BytesIO(npz_raw), allow_pickle=False) as archive:
        if set(archive.files) != {"rgb_history", "raw_action_history", "goal_rgb"}:
            raise ValueError("blind array whitelist")
        history = archive["rgb_history"]
        actions = archive["raw_action_history"]
        goal = archive["goal_rgb"]
    if not (history.shape == (3, 224, 224, 3) and history.dtype == np.uint8
            and goal.shape == (224, 224, 3) and goal.dtype == np.uint8
            and actions.shape == (10, 2) and actions.dtype == np.float32
            and np.isfinite(actions).all() and not actions.any()):
        raise ValueError("blind RGB/action history")
    pixel_sha = [sha(x.tobytes()) for x in (*history, goal)]
    if pixel_sha != manifest["pixel_sha256"] or sha(actions.tobytes()) != manifest["action_sha256"]:
        raise ValueError("blind pixel/action manifest binding")
    config_sha = sha(witness.config_bytes(CONFIG))
    result = witness.generate(history[-1], goal, CONFIG, config_sha)
    counts = {}
    for label, rgb in (("current", history[-1]), ("goal", goal)):
        counts[label] = {
            "object_exact_rgb_pixels": int(np.all(rgb == witness.OBJECT, axis=2).sum()),
            "pusher_exact_rgb_pixels": int(np.all(rgb == witness.PUSHER, axis=2).sum()),
        }
    return {
        "schema": "p1_dev303_rgb_boundary_static_20261102_exclusive.v1",
        "classification": "development_only_post_development_not_independent",
        "manifest_sha256": MANIFEST_SHA, "blind_npz_sha256": NPZ_SHA,
        "source_receipt_sha256": manifest["source_receipt_sha256"],
        "source_chain_reverified": False,
        "motion_source_sha256_observed_not_authority_verified": motion_source_sha,
        "config": CONFIG, "config_sha256": config_sha,
        "config_note": "motion coefficients from dev302 V5; all other thresholds are development protocol choices",
        "goal_format": {"dtype": str(goal.dtype), "shape": list(goal.shape),
                        "rgb_exact_object_color": list(witness.OBJECT),
                        "rgb_exact_pusher_color": list(witness.PUSHER), "counts": counts},
        "action_history": {"shape": list(actions.shape), "all_zero": True,
                           "last_action": actions[-1].tolist(),
                           "action_sha256": sha(actions.tobytes()),
                           "zero_last_action_proves_zero_physical_velocity": False},
        "result": result, "candidate_count": len(result["candidates"]),
        "new_physics_calls": 0, "new_neural_calls": 0, "outcomes_read": False,
        "independent_confirmation": False, "P1_admission": False,
        "physical_execution_authorized": False,
    }


def main():
    value = static_check(MANIFEST.read_bytes(), NPZ.read_bytes(), sha(MOTION_SOURCE.read_bytes()))
    print(json.dumps(value, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
