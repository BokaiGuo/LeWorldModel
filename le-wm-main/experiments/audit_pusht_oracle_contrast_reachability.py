"""Audit whether measurable matched torque contrast predicts PushT reachability.

This is a standalone privileged mechanism audit.  It fixes one contact action
bank before seeing any result: zero, ``+0.55`` and ``-0.55`` tangential pulses
with the same inward component, each lasting two simulator steps.  For every
fresh contact root we first reset to that root to measure the matched contrast,
then (from another reset to the unchanged root) run a real-successor,
two-pulse receding-horizon oracle using precisely that same three-action bank.

The contrast rule is predeclared and is never used by the oracle.  It only
partitions outcome roots after the fact, so this audit can falsify (but cannot
prove) claims that measurable contact contrast is necessary or sufficient for
strict pose reachability.  There is no RGB input, visual ranker, learned model,
or deployable-policy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import stable_worldmodel  # noqa: F401 - registers PushT

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_orientation_response import edge_chart, matched_actions  # noqa: E402
from experiments.audit_pusht_reachability_horizon import (  # noqa: E402
    action_bank_fingerprint,
    is_pose_success,
    real_successor,
    select_horizon_oracle,
)
from experiments.audit_pusht_torque_contrast_sensitivity import (  # noqa: E402
    SweepCell,
    contrast_metrics,
    root_digest,
    run_matched_triplet,
    triplet_specs,
)
from experiments.gpu_preflight import require_cuda_allocation  # noqa: E402
from experiments.run_pusht_mode_aware_shared_candidate import (  # noqa: E402
    angle_error,
    make_boundary_contact_roots,
    pose_cost,
)


ARTIFACT_SCHEMA = "pusht_oracle_contrast_reachability.v1"
PROBE_AMPLITUDE = 0.55
PROBE_STEPS = 2
CANDIDATE_IDS = ("zero_drift_reference", "positive_contact_moment", "negative_contact_moment")
RUNTIME_CONTRACT = {
    "uses_privileged_state": True,
    "uses_privileged_real_successors": True,
    "uses_visual_or_learned_ranker": False,
    "deployable_controller": False,
    "oracle_receives_eligibility_label": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--calibration-roots", type=int, default=16)
    parser.add_argument("--outcome-roots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026102117)
    parser.add_argument("--macro-steps", type=int, default=12)
    parser.add_argument("--inward-magnitude", type=float, default=0.18)
    parser.add_argument("--min-effect", type=float, default=2.0e-4)
    parser.add_argument("--min-effect-to-noise", type=float, default=2.0)
    args = parser.parse_args()
    if args.calibration_roots < 2 or args.outcome_roots < 2:
        parser.error("calibration-roots and outcome-roots must both be at least two")
    if args.macro_steps < 1:
        parser.error("macro-steps must be positive")
    if not 0.0 < args.inward_magnitude <= 1.0:
        parser.error("inward-magnitude must lie in (0, 1]")
    if args.min_effect < 0.0 or args.min_effect_to_noise < 0.0:
        parser.error("eligibility thresholds must be non-negative")
    return args


def root_identity(root: dict[str, np.ndarray]) -> str:
    """Hash the complete intervention root, including its task goal.

    A state-only digest is insufficient for a planning protocol: the same
    physical state paired with a different goal is a different root-level
    decision problem.  The digest is intentionally exact and is used only for
    split audits, never as an oracle input.
    """
    digest = hashlib.sha256()
    for key in ("state", "goal"):
        value = np.ascontiguousarray(np.asarray(root[key], dtype=np.float64))
        digest.update(key.encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def candidate_manifest(bank: np.ndarray) -> list[dict[str, Any]]:
    """Attach globally frozen IDs to the local-frame action bank.

    Values may change when the contact frame changes.  What is frozen is the
    candidate *meaning*, ordering, amplitude and pulse duration—not an
    impossible global XY vector that would lose its torque interpretation at
    a different edge of the T block.
    """
    values = np.asarray(bank, dtype=np.float64)
    if values.shape != (len(CANDIDATE_IDS), 2):
        raise ValueError("candidate bank must contain exactly zero/positive/negative moment actions")
    return [
        {"candidate_id": candidate_id, "action": [float(v) for v in action]}
        for candidate_id, action in zip(CANDIDATE_IDS, values)
    ]


def candidate_manifest_fingerprint(bank: np.ndarray) -> str:
    """Hash exact IDs and exact local-frame vectors in their declared order."""
    encoded = json.dumps(candidate_manifest(bank), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probe_action_bank(env: gym.Env, state: np.ndarray, *, inward_magnitude: float) -> tuple[dict[str, Any], np.ndarray]:
    """Return the fixed semantic ordered bank at one contact state.

    The numeric actions are deterministically expressed in that state's local
    contact frame.  This preserves the same torque semantics at every edge;
    it is not a globally fixed Cartesian action-vector bank.
    """

    chart = edge_chart(env, np.asarray(state, dtype=np.float64))
    actions = matched_actions(chart, float(inward_magnitude), PROBE_AMPLITUDE)
    bank = np.asarray([actions["zero"], actions["plus"], actions["minus"]], dtype=np.float64)
    if bank.shape != (3, 2) or not np.isfinite(bank).all():
        raise ValueError("the predeclared matched action bank must be finite [zero, plus, minus]")
    if not bool(actions["pair_valid"]):
        raise ValueError("the predeclared matched action pair is not a valid opposite-moment pair")
    if not np.allclose(bank[0], 0.0):
        raise ValueError("the first action-bank member must be the zero drift reference")
    if not np.isclose(np.linalg.norm(bank[1]), np.linalg.norm(bank[2])):
        raise ValueError("matched plus/minus actions must have equal norm")
    return actions, bank


def contrast_record(env: gym.Env, root: dict[str, np.ndarray], *, root_id: int, root_seed: int, args: argparse.Namespace) -> dict[str, Any]:
    """Measure the predeclared same-state contrast without changing the root."""

    state = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    actions, bank = probe_action_bank(env, state, inward_magnitude=float(args.inward_magnitude))
    cell = SweepCell(PROBE_AMPLITUDE, PROBE_STEPS)
    specs = triplet_specs(state, actions, cell, int(root_id), int(root_seed))
    probes = run_matched_triplet(env, state, goal, specs)
    metrics = contrast_metrics(
        actions,
        probes,
        min_effect=float(args.min_effect),
        min_effect_to_noise=float(args.min_effect_to_noise),
    )
    return {
        "root_identity_sha256": root_identity(root),
        "root_state_digest": root_digest(state),
        "edge_id": str(edge_chart(env, state)["edge_id"]),
        "contact_gap": float(edge_chart(env, state)["contact_gap"]),
        "probe_action_bank_sha256": action_bank_fingerprint(bank),
        "probe_candidate_manifest_sha256": candidate_manifest_fingerprint(bank),
        "probe_candidates": candidate_manifest(bank),
        "probe_action_bank": bank.tolist(),
        "same_state_pairing_verified": True,
        "probe": probes,
        **metrics,
    }


def run_two_pulse_oracle(env: gym.Env, root: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    """Run a privileged two-pulse receding-horizon oracle on the declared bank.

    At each replanning state the bank is regenerated by the same fixed matched
    construction.  Its initial member order and exact hash must therefore
    match the contrast probe performed from this root.
    """

    current = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    _initial_actions, initial_bank = probe_action_bank(env, current, inward_magnitude=float(args.inward_magnitude))
    initial_hash = action_bank_fingerprint(initial_bank)
    initial_manifest_hash = candidate_manifest_fingerprint(initial_bank)
    trace: list[dict[str, Any]] = []
    for macro_step in range(int(args.macro_steps)):
        if is_pose_success(current, goal):
            break

        def bank_fn(state: np.ndarray) -> np.ndarray:
            _actions, bank = probe_action_bank(env, state, inward_magnitude=float(args.inward_magnitude))
            return bank

        choice = select_horizon_oracle(
            current,
            horizon=2,
            action_bank_fn=bank_fn,
            successor_fn=lambda state, action: real_successor(env, state, goal, action, PROBE_STEPS),
            cost_fn=lambda state: pose_cost(state, goal),
            beam_width=1,
        )
        before = current.copy()
        current = real_successor(env, current, goal, choice.first_action, PROBE_STEPS)
        trace.append({
            "macro_step": int(macro_step),
            "bank_sha256": action_bank_fingerprint(bank_fn(before)),
            "candidate_manifest_sha256": candidate_manifest_fingerprint(bank_fn(before)),
            "candidate_ids": list(CANDIDATE_IDS),
            "candidate_count": 3,
            "selected_first_index": int(choice.first_index),
            "selected_sequence_indices": list(choice.sequence_indices),
            "selected_first_candidate_id": CANDIDATE_IDS[int(choice.first_index)],
            "selected_sequence_candidate_ids": [CANDIDATE_IDS[int(index)] for index in choice.sequence_indices],
            "lookahead_terminal_cost": float(choice.terminal_cost),
            "action": choice.first_action.tolist(),
            "angle_before": float(angle_error(before[4], goal[4])),
            "angle_after": float(angle_error(current[4], goal[4])),
            "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
        })
    return {
        "initial_action_bank_sha256": initial_hash,
        "initial_candidate_manifest_sha256": initial_manifest_hash,
        "initial_candidates": candidate_manifest(initial_bank),
        "initial_candidate_count": 3,
        "initial_cost": float(pose_cost(root["state"], goal)),
        "final_cost": float(pose_cost(current, goal)),
        "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
        "final_angle": float(angle_error(current[4], goal[4])),
        "pose_success": bool(is_pose_success(current, goal)),
        "trace": trace,
    }


def assert_root_disjoint(*splits: list[dict[str, np.ndarray]]) -> None:
    """Fail closed when source states or complete task roots overlap.

    Sharing a physical source state with a different goal would still leak the
    rendered/history context across calibration and outcome cohorts.  The
    complete identity is recorded for traceability, while state identity is
    the stricter split key.
    """

    root_sets = [{root_identity(root) for root in split} for split in splits]
    state_sets = [{root_digest(np.asarray(root["state"], dtype=np.float64)) for root in split} for split in splits]
    if any(len(values) != len(split) for values, split in zip(root_sets, splits)):
        raise ValueError("a root split contains duplicate complete task roots")
    if any(len(values) != len(split) for values, split in zip(state_sets, splits)):
        raise ValueError("a root split contains duplicate source states")
    for first_index in range(len(splits)):
        for second_index in range(first_index + 1, len(splits)):
            if root_sets[first_index] & root_sets[second_index] or state_sets[first_index] & state_sets[second_index]:
                raise ValueError("root splits overlap in source state or complete task root; refusing non-disjoint protocol")


def collect_nonterminal_contact_roots(env: gym.Env, count: int, seed: int) -> list[dict[str, np.ndarray]]:
    """Create a fixed cohort whose roots require an actual oracle decision.

    A root already satisfying the strict terminal pose predicate cannot be
    evidence that a candidate bank is reachable.  Oversampling happens before
    contrast/oracle evaluation and filtering depends solely on the declared
    terminal predicate, never on a later contrast or success outcome.
    """
    proposals = make_boundary_contact_roots(env, max(int(count) * 4, int(count) + 8), int(seed))
    selected = [
        {"state": np.asarray(root["state"], dtype=np.float64).copy(), "goal": np.asarray(root["goal"], dtype=np.float64).copy()}
        for root in proposals
        if not is_pose_success(np.asarray(root["state"]), np.asarray(root["goal"]))
    ][:int(count)]
    if len(selected) != int(count):
        raise RuntimeError(f"insufficient nonterminal boundary-contact roots: requested={count}, obtained={len(selected)}")
    if any(is_pose_success(root["state"], root["goal"]) for root in selected):
        raise RuntimeError("terminal root passed into oracle-reachability panel")
    return selected


def empirical_necessity_sufficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report finite-panel falsification, never a proof of necessity/sufficiency."""

    contingency = {
        "eligible_reachable": 0,
        "eligible_unreachable": 0,
        "ineligible_reachable": 0,
        "ineligible_unreachable": 0,
    }
    for row in rows:
        eligible = bool(row["contrast"]["eligible_for_later_visual_ranker"])
        reachable = bool(row["two_pulse_oracle"]["pose_success"])
        key = f"{'eligible' if eligible else 'ineligible'}_{'reachable' if reachable else 'unreachable'}"
        contingency[key] += 1
    eligible_count = contingency["eligible_reachable"] + contingency["eligible_unreachable"]
    ineligible_count = contingency["ineligible_reachable"] + contingency["ineligible_unreachable"]
    necessary_falsified = contingency["ineligible_reachable"] > 0
    sufficient_falsified = contingency["eligible_unreachable"] > 0
    return {
        "contingency": contingency,
        "eligible_count": int(eligible_count),
        "ineligible_count": int(ineligible_count),
        "oracle_reachable_count": int(contingency["eligible_reachable"] + contingency["ineligible_reachable"]),
        "empirical_necessity": {
            "falsified_on_this_panel": bool(necessary_falsified),
            "status": "falsified" if necessary_falsified else "not_falsified_not_proven",
            "reason": (
                "At least one non-eligible root reached strict pose success under the same privileged oracle."
                if necessary_falsified
                else "No non-eligible root reached strict pose success in this finite panel; that is not a proof of necessity."
            ),
        },
        "empirical_sufficiency": {
            "falsified_on_this_panel": bool(sufficient_falsified),
            "status": "falsified" if sufficient_falsified else ("unevaluable_no_eligible_roots" if eligible_count == 0 else "not_falsified_not_proven"),
            "reason": (
                "At least one eligible root still failed strict pose reachability under the same privileged oracle."
                if sufficient_falsified
                else (
                    "No eligible roots were observed, so sufficiency is unevaluable."
                    if eligible_count == 0
                    else "Every eligible root reached success in this finite panel; that is not a proof of sufficiency."
                )
            ),
        },
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cuda = require_cuda_allocation(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    # The audit never renders or decodes images.  Keep the environment itself
    # in state-only mode so an accidental visual-planner dependency cannot be
    # hidden behind an RGB render-mode setting.
    env = gym.make("swm/PushT-v1")
    try:
        calibration_seed = int(args.seed) + 100_000
        outcome_seed = int(args.seed) + 200_000
        calibration_roots = collect_nonterminal_contact_roots(env, int(args.calibration_roots), calibration_seed)
        outcome_roots = collect_nonterminal_contact_roots(env, int(args.outcome_roots), outcome_seed)
        assert_root_disjoint(calibration_roots, outcome_roots)
        calibration = [
            contrast_record(env, root, root_id=index, root_seed=calibration_seed, args=args)
            for index, root in enumerate(calibration_roots)
        ]
        rows: list[dict[str, Any]] = []
        for root_id, root in enumerate(outcome_roots):
            contrast = contrast_record(env, root, root_id=root_id, root_seed=outcome_seed, args=args)
            oracle = run_two_pulse_oracle(env, root, args)
            if contrast["probe_action_bank_sha256"] != oracle["initial_action_bank_sha256"]:
                raise RuntimeError("contrast and two-pulse oracle did not use the same ordered initial action bank")
            if contrast["probe_candidate_manifest_sha256"] != oracle["initial_candidate_manifest_sha256"]:
                raise RuntimeError("contrast and two-pulse oracle did not use the same exact initial candidate identifiers")
            rows.append({
                "root_id": f"outcome:{root_id:04d}",
                "root_identity_sha256": contrast["root_identity_sha256"],
                "root_state_digest": contrast["root_state_digest"],
                "contrast": contrast,
                "two_pulse_oracle": oracle,
            })
    finally:
        env.close()
    association = empirical_necessity_sufficiency(rows)
    report = {
        "schema": ARTIFACT_SCHEMA,
        "claim_boundary": (
            "Privileged same-bank mechanism audit only. It can falsify finite-panel necessity or sufficiency of measurable contrast for this two-pulse oracle, "
            "but evaluates no visual/learned ranker and establishes neither deployable planning nor world-model causal value."
        ),
        "protocol": {
            "environment": "swm/PushT-v1",
            "root_protocol": "boundary_contact",
                "terminal_root_exclusion": "all roots fail the strict pose-success predicate before any contrast probe or oracle decision",
            "root_split": {
                "calibration_seed": calibration_seed,
                "outcome_seed": outcome_seed,
                "calibration_roots": int(len(calibration_roots)),
                "outcome_roots": int(len(rows)),
                "root_disjointness_verified": True,
                "root_identity": "sha256(exact source state || exact goal state)",
                "cross_cohort_exclusion": "both exact source-state digest and complete state+goal identity must be disjoint",
                "calibration_root_ids": [f"calibration:{index:04d}" for index in range(len(calibration_roots))],
                "outcome_root_ids": [f"outcome:{index:04d}" for index in range(len(outcome_roots))],
            },
            "predeclared_probe": {
                "amplitude": PROBE_AMPLITUDE,
                "pulse_steps": PROBE_STEPS,
                "inward_magnitude": float(args.inward_magnitude),
                "ordered_bank": ["zero", "plus", "minus"],
                "candidate_ids": list(CANDIDATE_IDS),
                "fixed_bank_contract": (
                    "Candidate IDs, order, amplitude and pulse duration are frozen globally. "
                    "The two non-zero vectors are deterministically re-expressed in each current contact frame; "
                    "an exact local candidate manifest is recorded at every oracle decision."
                ),
                "eligibility_thresholds": {
                    "min_effect": float(args.min_effect),
                    "min_effect_to_noise": float(args.min_effect_to_noise),
                    "requires_mode_agreement": True,
                },
            },
            "oracle": {
                "horizon": 2,
                "receding_macro_steps": int(args.macro_steps),
                "bank_contract": "The contrast triplet and the first oracle decision share one exact ordered candidate manifest. Later decisions use the same frozen semantic IDs and deterministic local-frame construction; eligibility is never passed to the oracle.",
            },
            "runtime_contract": RUNTIME_CONTRACT,
            "cuda_preflight": cuda,
            "simulator_note": "PushT/Pymunk successor physics remains CPU-bound; CUDA was allocated for mandatory runtime integrity before collecting rows.",
            "visual_separation": "No render call, RGB feature, visual ranker, learned model, or decoder is imported or evaluated by this script.",
        },
        "calibration_contrast_screen": {
            "eligible_count": int(sum(bool(row["eligible_for_later_visual_ranker"]) for row in calibration)),
            "root_count": int(len(calibration)),
            "records": calibration,
            "use": "Descriptive independent split only; thresholds were fixed and no calibration result selects an oracle action or modifies the outcome rule.",
        },
        "outcome_association": association,
        "sanity_checks": {
            "all_calibration_roots_nonterminal": bool(not any(is_pose_success(root["state"], root["goal"]) for root in calibration_roots)),
            "all_outcome_roots_nonterminal": bool(not any(is_pose_success(root["state"], root["goal"]) for root in outcome_roots)),
            "all_outcome_initial_manifests_match": bool(all(
                row["contrast"]["probe_candidate_manifest_sha256"] == row["two_pulse_oracle"]["initial_candidate_manifest_sha256"]
                for row in rows
            )),
        },
        "rows": rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "# PushT oracle contrast-to-reachability audit", "",
        f"- Fresh outcome roots: `{len(rows)}`; independent contrast-screen roots: `{len(calibration)}`.",
        f"- Fixed matched bank: `zero, +{PROBE_AMPLITUDE}, -{PROBE_AMPLITUDE}` with `{PROBE_STEPS}`-step pulses.",
        f"- Contingency: `{association['contingency']}`.",
        f"- Necessity: `{association['empirical_necessity']['status']}`.",
        f"- Sufficiency: `{association['empirical_sufficiency']['status']}`.", "",
        report["claim_boundary"], "",
    ]
    (args.output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "outcome_association": association}, indent=2))


if __name__ == "__main__":
    main()
