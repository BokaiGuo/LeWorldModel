"""V10 design sidecar for matched-budget cp5 RGB suffix replanning.

This module freezes a design and provides pure geometry/scoring helpers. It
does not initialize CUDA, execute physics, or modify the V8/V9 implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_v10_cp5_rgb_suffix_design_20261101_parallel"
OUTPUT = ROOT / "results" / f"{NAME}_PREP_MAIN01.json"
V8_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v8_20261101_codex_PREP_MAIN01.json"
V8_PREP_SHA = "40d6f517ced03a790c21e2ec4d4a8ecfeb69269937df0e56de203e180c36a394"
V8_REPORT = ROOT / "results/p1_dev302_mixed_bank_cuda_v8_20261101_codex_RUN_MAIN01/report.json"
V8_REPORT_SHA = "547d657e5155df3a6cca88969045b04d9bd1b365135f54b95958311119d7b18c"
SOURCE_PINS = {
    "experiments/spectral_reachability/p1_dev302_mixed_prefix_bank_v8_20261101_codex.py":
        "e73f26b5993802ba6aa7b12ee33e41ff3509848c7da89296417070762cfc04a1",
    "experiments/spectral_reachability/p1_dev302_mixed_bank_cuda_v8_20261101_codex.py":
        "f9c73ef587c8a6b7eea1d7a19f9ad73379037192cad808a6c9f2a3088b8c019d",
    "experiments/spectral_reachability/p1_unified_cp5_feedback_cem_20261101_codex.py":
        "f0510356e73d01f020f302e70bd0001a88e46f2e2f326b41d7d98145a86c8b1f",
}
INDEX = 16
PREFIX_SHA = "0db61c3546f87f17bdf69efa4b55e6a8e9678ae7b4627c1c3cebd731d58bdb36"
CONTACT_STEPS = 3
CONTACT_GAP_PIXELS = 6.5
MIN_NORMAL_ALIGNMENT = 0.8
MAX_TANGENTIAL_PIXELS = 0.75
MIN_FORWARD_PIXELS = 0.25
ITERATIONS, SAMPLES, ELITES, VARIANTS, PACKETS = 5, 64, 8, 5, 2
SUFFIX_STEPS = 10
RNG_SEED = 30220261102
NATIVE_DELTA_PER_CURRENT_ACTION = 29.567
NATIVE_DELTA_PER_PREVIOUS_VELOCITY = 0.02993
NATIVE_VELOCITY_PER_PREVIOUS_ACTION = 374.1
PIXELS_PER_NATIVE = 224.0 / 512.0


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path):
    raw = Path(path).read_bytes()
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def support_normal(object_xy, goal_object_xy):
    current = np.asarray(object_xy, dtype=np.float64)
    goal = np.asarray(goal_object_xy, dtype=np.float64)
    require(current.shape == goal.shape == (2,) and
            np.isfinite(current).all() and np.isfinite(goal).all(), "finite XY")
    delta = goal - current
    norm = float(np.linalg.norm(delta))
    require(norm > 1e-9, "nonzero task support direction")
    return delta / norm


def support_face_pixels(object_pixels, normal):
    pixels = np.asarray(object_pixels, dtype=np.float64)
    direction = np.asarray(normal, dtype=np.float64)
    require(
        pixels.ndim == 2
        and pixels.shape[1] == 2
        and len(pixels) >= 1
        and direction.shape == (2,)
        and np.isfinite(pixels).all()
        and np.isfinite(direction).all()
        and abs(float(np.linalg.norm(direction)) - 1.0) <= 1e-6,
        "support-face geometry ABI",
    )
    center = pixels.mean(axis=0)
    projections = (pixels - center) @ direction
    trailing = float(projections.min())
    face = pixels[projections <= trailing + 1.0]
    require(len(face) >= 1, "nonempty support face")
    return face


def predict_pusher_centers(start, actions, previous_action):
    position = np.asarray(start, dtype=np.float64).copy()
    values = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64).copy()
    require(
        position.shape == previous.shape == (2,)
        and values.ndim == 2
        and values.shape[1] == 2
        and np.isfinite(position).all()
        and np.isfinite(previous).all()
        and np.isfinite(values).all(),
        "pusher response ABI",
    )
    rows = []
    for action in values:
        native_delta = (
            NATIVE_DELTA_PER_CURRENT_ACTION * action
            + NATIVE_DELTA_PER_PREVIOUS_VELOCITY
            * NATIVE_VELOCITY_PER_PREVIOUS_ACTION
            * previous
        )
        position = position + native_delta * PIXELS_PER_NATIVE
        rows.append(position.copy())
        previous = action
    return np.asarray(rows)


def project_early_motion(actions, normal, previous_action):
    """Project the first contact steps in pixel-motion space, not raw-action space."""
    raw = np.asarray(actions, dtype=np.float64)
    direction = np.asarray(normal, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64).copy()
    require(
        raw.shape == (SUFFIX_STEPS, 2)
        and direction.shape == previous.shape == (2,)
        and np.isfinite(raw).all()
        and np.isfinite(direction).all()
        and np.isfinite(previous).all()
        and abs(float(np.linalg.norm(direction)) - 1.0) <= 1e-6,
        "early-motion projection ABI",
    )
    alpha = NATIVE_DELTA_PER_CURRENT_ACTION * PIXELS_PER_NATIVE
    beta = (
        NATIVE_DELTA_PER_PREVIOUS_VELOCITY
        * NATIVE_VELOCITY_PER_PREVIOUS_ACTION
        * PIXELS_PER_NATIVE
    )
    projected = raw.copy()
    for step in range(CONTACT_STEPS):
        original_delta = alpha * projected[step] + beta * previous
        normal_pixels = max(float(original_delta @ direction), MIN_FORWARD_PIXELS)
        tangent = original_delta - float(original_delta @ direction) * direction
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > MAX_TANGENTIAL_PIXELS:
            tangent *= MAX_TANGENTIAL_PIXELS / tangent_norm
        desired_delta = normal_pixels * direction + tangent
        projected[step] = np.clip((desired_delta - beta * previous) / alpha, -1.0, 1.0)
        previous = projected[step]
    return projected.astype(np.float32)


def contact_hold(pusher_centers, object_pixels, normal):
    centers = np.asarray(pusher_centers, dtype=np.float64)
    pixels = support_face_pixels(object_pixels, normal)
    direction = np.asarray(normal, dtype=np.float64)
    require(centers.ndim == pixels.ndim == 2 and centers.shape[1] == pixels.shape[1] == 2
            and len(centers) >= CONTACT_STEPS and len(pixels) >= 1
            and direction.shape == (2,) and np.isfinite(centers).all()
            and np.isfinite(pixels).all() and np.isfinite(direction).all()
            and abs(float(np.linalg.norm(direction)) - 1.0) <= 1e-6,
            "contact geometry ABI")
    rows = []
    for step, center in enumerate(centers[:CONTACT_STEPS], start=1):
        deltas = pixels - center
        distances = np.linalg.norm(deltas, axis=1)
        index = int(np.argmin(distances))
        gap = float(distances[index])
        alignment = float(deltas[index] @ direction / gap) if gap > 1e-12 else 1.0
        rows.append({
            "step": step,
            "gap_pixels": gap,
            "normal_alignment": alignment,
            "valid": gap <= CONTACT_GAP_PIXELS and alignment >= MIN_NORMAL_ALIGNMENT,
        })
    return {"valid": all(row["valid"] for row in rows), "steps": rows}


def robust_full_rollout_scores(costs, contact_valid):
    values = np.asarray(costs, dtype=np.float64)
    valid = np.asarray(contact_valid, dtype=np.bool_)
    require(
        values.ndim == 2
        and values.shape[1] == VARIANTS
        and valid.shape == values.shape
        and np.isfinite(values).all(),
        "five finite full-rollout and contact variants per candidate",
    )
    scores = values.mean(axis=1) + np.ptp(values, axis=1)
    return np.where(valid.all(axis=1), scores, np.inf)


def _source():
    require(sha(V8_PREP) == V8_PREP_SHA and sha(V8_REPORT) == V8_REPORT_SHA,
            "V8 sealed artifact SHA")
    for relative, expected in SOURCE_PINS.items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                f"V8 source SHA {relative}")
    prepared, report = strict_json(V8_PREP), strict_json(V8_REPORT)
    require(prepared["schema"] ==
            "p1_dev302_mixed_bank_cuda_v8_20261101_codex.protocol.v1" and
            prepared["P1_admission"] is False and
            len(prepared["prefix_bank"]["candidates"]) == 17, "V8 prepared contract")
    require(report["schema"] ==
            "p1_dev302_mixed_bank_cuda_v8_20261101_codex.report.v1" and
            report["status"] == "model_only_complete" and
            report["P1_admission"] is False and report["selected_index"] == INDEX and
            report["physical_outcomes_used"] is False, "V8 model-only report")
    row = report["rows"][INDEX]
    candidate = prepared["prefix_bank"]["candidates"][INDEX]
    require(row["index"] == candidate["index"] == INDEX and
            row["label"] == candidate["label"] == "rgb_delayed_x23_neg04" and
            row["actions_sha256"] == candidate["actions_sha256"] == PREFIX_SHA and
            row["actions"] == candidate["actions"] and
            row["contact_feasible_prior"] is True and
            candidate["task_support_alignment"] >= 0.5, "V8 index16 identity")
    old_tail = np.asarray(row["tail_selected"]["actions"], dtype=np.float32)
    require(old_tail.shape == (PACKETS, 5, 2) and np.isfinite(old_tail).all() and
            np.max(np.abs(old_tail)) <= 1, "V8 old tail ABI")
    return candidate, row, old_tail.reshape(SUFFIX_STEPS, 2)


def build():
    candidate, row, old_tail = _source()
    budget = {
        "iterations": ITERATIONS,
        "samples": SAMPLES,
        "elites": ELITES,
        "variants": VARIANTS,
        "packets": PACKETS,
        "full_suffix_steps": SUFFIX_STEPS,
        "action_encoder_calls": ITERATIONS * PACKETS,
        "predict_calls": ITERATIONS * PACKETS,
        "predict_samples": ITERATIONS * PACKETS * SAMPLES * VARIANTS,
        "wall_seconds": 1800,
    }
    return {
        "schema": NAME + ".design.v1",
        "status": "v10_minimal_implementation_design_frozen_not_run",
        "v8_source": {
            "selected_index": INDEX,
            "label": candidate["label"],
            "prefix_actions": candidate["actions"],
            "prefix_actions_sha256": candidate["actions_sha256"],
            "old_tail_actions": old_tail.tolist(),
            "old_tail_robust_score": row["tail_selected"]["robust_score"],
            "task_support_alignment": candidate["task_support_alignment"],
            "predicted_final_contact_gap_pixels":
                candidate["rgb_dynamic_final_object_center_to_pixel_gap"],
        },
        "cp5_information_boundary": {
            "execute": "exact V8 index16 five-step prefix",
            "planner_inputs": [
                "actual cp5 RGB",
                "actual five-action prefix history",
                "goal RGB",
                "sealed frozen model and normalization",
            ],
            "simulator_state_used_by_planner": False,
            "post_cp5_outcome_used": False,
            "feedback_count": 1,
        },
        "task_support_geometry": {
            "normal": "normalize(goal_object_centroid_xy - cp5_object_centroid_xy)",
            "support_face": (
                "RGB object pixels in the trailing one-pixel band along the task normal"
            ),
            "pusher_motion_model": (
                "the V8 RGB-calibrated one-step-memory response, initialized with the "
                "actually executed fifth prefix action"
            ),
            "candidate_projection": (
                "solve each of the first three raw actions so predicted pixel displacement "
                "has at least 0.25 px forward motion and at most 0.75 px tangential motion"
            ),
            "contact_vector": "nearest trailing support-face pixel minus predicted pusher center",
            "first_contact_steps": CONTACT_STEPS,
            "maximum_gap_pixels": CONTACT_GAP_PIXELS,
            "minimum_normal_alignment": MIN_NORMAL_ALIGNMENT,
            "all_five_robust_variants_must_pass": True,
            "invalid_candidate_score": "positive infinity",
            "RGB_only": True,
        },
        "matched_design": {
            "common_inputs": (
                "same replayed index16 cp5 state, model, normalization, old-tail/zero seeds, "
                "CEM random draws, perturbations, horizon and stopping rules"
            ),
            "arms": {
                "v10_contact_normal": {
                    "candidate_transform": (
                        "project first three suffix actions through the frozen V8 pixel-motion "
                        "response, including prefix-step-5 velocity carry; reject unless RGB "
                        "support-face contact-hold passes for every robust variant"
                    ),
                    "budget": dict(budget),
                },
                "matched_unconstrained": {
                    "candidate_transform": "identity; same raw common-random-number bank",
                    "budget": dict(budget),
                },
            },
            "iteration0_seeds": ["V8 index16 old tail", "zero two-packet tail"],
            "common_random_numbers_seed": RNG_SEED,
            "no_dummy_compute": True,
            "projection_CPU_cost_reported_separately": True,
        },
        "robust_full_rollout": {
            "variants": [
                "nominal",
                "plus_0p05_both_axes",
                "minus_0p05_both_axes",
                "plus_0p05_checkerboard",
                "minus_0p05_checkerboard",
            ],
            "rollout": (
                "all ten suffix actions from actual cp5 latent for every variant; feasibility "
                "is checked on the first three predicted pusher centers for every variant"
            ),
            "terminal_cost": "terminal latent goal MSE",
            "score": "mean five terminal costs plus max-minus-min range",
            "selection": "minimum finite score; stable iteration/candidate tie order",
            "partial_or_one-packet_scoring": "prohibited",
            "prefix_velocity_reset": "prohibited",
        },
        "minimal_runtime_changes": [
            {
                "file": "experiments/spectral_reachability/"
                        "p1_v10_cp5_rgb_suffix_runner_20261101_parallel.py",
                "change": (
                    "new runner only: bind V8 index16, replay prefix, capture actual cp5 RGB, "
                    "materialize two matched CEM arms under cooperative CUDA/physics locks"
                ),
            },
            {
                "file": "experiments/spectral_reachability/"
                        "p1_v10_cp5_rgb_suffix_geometry_20261101_parallel.py",
                "change": (
                    "new RGB-only support-normal extraction, first-three-step contact projection "
                    "and five-variant contact validation"
                ),
            },
            {
                "file": "tests/test_p1_v10_cp5_rgb_suffix_runner_20261101_parallel.py",
                "change": "new contract, budget, replay and fail-closed tests",
            },
        ],
        "implementation_order": [
            "freeze cp5 replay/source receipt without executing a suffix",
            "unit-test RGB geometry and all-variant contact rejection",
            "prepare both matched CUDA plans without physics",
            "saved-validate exact equal counters and full-rollout journals",
            "only then authorize paired physical suffix execution",
        ],
        "failure_rules": {
            "cp5_replay_identity_mismatch": "abort both arms",
            "RGB_segmentation_failure": "abort both arms; no simulator-state fallback",
            "no_finite_contact_normal_candidate": "V10 abstains and is scored failure",
            "matched_counterpart_failure": "paired comparison censored; preserve both receipts",
            "budget_or_counter_mismatch": "engineering invalid, no scientific comparison",
        },
        "bindings": {
            str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
            str(V8_PREP): V8_PREP_SHA,
            str(V8_REPORT): V8_REPORT_SHA,
            **{str(ROOT / path): value for path, value in SOURCE_PINS.items()},
        },
        "V9_files_read_or_modified": False,
        "cuda_initialized": False,
        "physics_calls": 0,
        "execution_authorized": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }


def write_new(path, value):
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        require(os.write(fd, payload) == len(payload), "complete write")
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(args.output.name == OUTPUT.name, "exclusive V10 design output")
    value = build()
    digest = write_new(args.output, value)
    print(json.dumps({"status": value["status"], "output": str(args.output),
                      "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
