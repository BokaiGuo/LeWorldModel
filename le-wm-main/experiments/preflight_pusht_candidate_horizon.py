"""Fail-closed candidate-bank/horizon feasibility preflight for learned PushT.

Before attributing a learned PushT pose-control failure to a visual
representation or candidate ranker, this program runs a *privileged*
real-successor ceiling on the identical declared root cohort, contact-frame
candidate-bank construction, pulse duration, and macro-step budget.  A
positive ceiling is only a feasibility upper bound; it never evaluates a
learned policy.  A negative calibration ceiling prohibits a learned panel from
being labelled ``ranking_limited`` under this protocol.

The script is deliberately CUDA-gated even though Pymunk successor rollouts
are CPU physics.  It must not silently turn a planned learned/GPU panel into a
CPU result with a different runtime contract.
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
import stable_worldmodel  # noqa: F401 -- registers swm/PushT-v1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_reachability_horizon import (  # noqa: E402
    collect_nonterminal_boundary_contact_roots,
    resolve_device,
    run_root,
)
from experiments.pusht_protocol_contracts import root_fingerprint  # noqa: E402


SCHEMA = "pusht_candidate_horizon_preflight.v1"
ORACLE_ARMS = ("one_pulse", "two_pulse", "three_pulse_bounded")


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    """Hash canonical declared settings used by both oracle and learned runs."""

    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def cohort_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose only equality/audit digests needed by a later learned runner."""

    manifest: list[dict[str, Any]] = []
    for row in rows:
        if bool(row.get("initial_pose_success", True)):
            raise ValueError("preflight row is terminal and cannot be used for learned ranking")
        manifest.append({
            "root_id": int(row["root_id"]),
            "root_sha256": str(row["root_sha256"]),
            "root_action_bank_sha256": str(row["root_action_bank_sha256"]),
            "root_candidate_count": int(row["root_candidate_count"]),
            "oracle_route": str(row["routing"]["route"]),
            "oracle_reachable": bool(row["routing"]["oracle_reachable"]),
            "oracle_arm_pose_success": {
                arm: bool(row["arms"][arm]["pose_success"])
                for arm in ORACLE_ARMS
            },
        })
    if len({item["root_sha256"] for item in manifest}) != len(manifest):
        raise ValueError("preflight cohort reuses a physical root")
    return manifest


def authorize_calibration(
    manifest: list[dict[str, Any]], *, minimum_reachable_roots: int, minimum_reachable_rate: float,
    required_oracle_arm: str = "two_pulse",
) -> dict[str, Any]:
    """Freeze authorization from calibration feasibility alone, never outcome."""

    if required_oracle_arm not in ORACLE_ARMS:
        raise ValueError(f"required_oracle_arm must be one of {ORACLE_ARMS}")
    if minimum_reachable_roots < 1 or not 0.0 < minimum_reachable_rate <= 1.0:
        raise ValueError("invalid predeclared calibration feasibility threshold")
    total = len(manifest)
    if total < 1:
        raise ValueError("calibration manifest cannot be empty")
    # Compatibility fallback keeps the pure helper convenient for older unit
    # fixtures; real preflight manifests always include per-arm outcomes.
    reachable = int(sum(bool(row.get("oracle_arm_pose_success", {}).get(
        required_oracle_arm, row["oracle_reachable"],
    )) for row in manifest))
    rate = float(reachable / total)
    authorized = bool(reachable >= int(minimum_reachable_roots) and rate >= float(minimum_reachable_rate))
    return {
        "authorized": authorized,
        "reachable_roots": reachable,
        "root_count": total,
        "reachable_rate": rate,
        "predeclared_minimum_reachable_roots": int(minimum_reachable_roots),
        "predeclared_minimum_reachable_rate": float(minimum_reachable_rate),
        "required_oracle_arm": required_oracle_arm,
        "decision": (
            "oracle_feasibility_authorized"
            if authorized
            else "candidate_or_horizon_limited"
        ),
        "interpretation": (
            "At least one correct ranking exists on enough calibration roots under the declared bank/horizon. "
            "This is a privileged feasibility ceiling, not learned-policy evidence."
            if authorized
            else
            "The declared bank/horizon lacks enough privileged calibration reachability. "
            "A learned 0/N result must not be attributed to visual ranking or latent failure."
        ),
    }


def validate_learned_runtime_contract(
    preflight: dict[str, Any], runtime: dict[str, Any], *, cohort: str = "outcome",
) -> None:
    """Reject a learned panel unless its roots and candidate contract match.

    The learned runner supplies only hashes and declared parameters; it never
    receives privileged successors or feasibility labels at runtime.
    """

    if preflight.get("schema") != SCHEMA:
        raise ValueError("unsupported preflight schema")
    authorization = preflight.get("calibration_authorization", {})
    if not bool(authorization.get("authorized", False)):
        raise ValueError("candidate_or_horizon_limited preflight forbids learned ranking attribution")
    expected = preflight.get("protocol", {})
    if str(runtime.get("protocol_sha256")) != str(expected.get("protocol_sha256")):
        raise ValueError("learned runtime candidate-bank/horizon protocol fingerprint differs from preflight")
    expected_rows = preflight.get("cohorts", {}).get(str(cohort), [])
    supplied = runtime.get("roots", [])
    expected_pairs = [
        (str(row["root_sha256"]), str(row["root_action_bank_sha256"]), int(row["root_candidate_count"]))
        for row in expected_rows
    ]
    supplied_pairs = [
        (str(row["root_sha256"]), str(row["root_action_bank_sha256"]), int(row["root_candidate_count"]))
        for row in supplied
    ]
    if supplied_pairs != expected_pairs:
        raise ValueError("learned runtime roots or ordered candidate banks differ from preflight")


def reconstruct_frozen_cohort(
    env: gym.Env, preflight: dict[str, Any], *, cohort: str,
) -> list[dict[str, np.ndarray]]:
    """Recreate and verify the exact nonterminal roots declared by preflight.

    This is the bridge from a privileged preflight to a learned runner.  The
    runner receives source roots only through the deterministic experiment
    protocol, never through oracle successor calls or reachability labels.
    A changed root generator, seed, terminal rule, or ordering fails before
    model fitting/evaluation rather than silently changing the estimand.
    """

    if preflight.get("schema") != SCHEMA:
        raise ValueError("unsupported preflight schema")
    if cohort not in {"calibration", "outcome"}:
        raise ValueError("cohort must be calibration or outcome")
    cohort_contract = preflight.get("protocol", {}).get("cohort_contract", {}).get(cohort)
    if not isinstance(cohort_contract, dict):
        raise ValueError("preflight lacks an immutable cohort reconstruction contract")
    roots = collect_nonterminal_boundary_contact_roots(
        env,
        count=int(cohort_contract["root_count"]),
        seed=int(cohort_contract["root_seed"]),
        oversample_factor=int(cohort_contract["oversample_factor"]),
    )
    expected = preflight.get("cohorts", {}).get(cohort, [])
    actual_hashes = [root_fingerprint(root) for root in roots]
    expected_hashes = [str(row["root_sha256"]) for row in expected]
    if actual_hashes != expected_hashes:
        raise ValueError("reconstructed root cohort differs from immutable preflight manifest")
    return roots


def validate_mode_aware_runner_parameters(
    preflight: dict[str, Any], *, root_protocol: str, boundary_action_frame: bool,
    inward_magnitude: float, tangent_magnitudes: tuple[float, ...],
    contact_normal_magnitudes: tuple[float, ...], pulse_steps: int,
    closure_steps: int, lookahead: int,
) -> None:
    """Ensure the mode-aware learned loop has the same feasibility contract.

    The check intentionally rejects a broader oracle horizon than the learned
    selector's lookahead: h=3 oracle feasibility cannot license an h=2
    learned-panel attribution.
    """

    if preflight.get("schema") != SCHEMA:
        raise ValueError("unsupported preflight schema")
    protocol = preflight.get("protocol", {})
    config = protocol.get("action_bank_config", {})
    expected_arm = str(protocol.get("required_oracle_arm", "two_pulse"))
    arm_to_lookahead = {"one_pulse": 1, "two_pulse": 2, "three_pulse_bounded": 3}
    if root_protocol != "boundary_contact" or not boundary_action_frame:
        raise ValueError("preflight-compatible runner requires boundary-contact roots and action frame")
    if int(pulse_steps) != int(protocol.get("pulse_steps", -1)):
        raise ValueError("runner pulse_steps differs from oracle preflight")
    if int(closure_steps) != int(protocol.get("macro_steps", -1)):
        raise ValueError("runner closure_steps differs from oracle preflight macro budget")
    if int(lookahead) != arm_to_lookahead.get(expected_arm):
        raise ValueError("runner lookahead differs from the authorized oracle arm")
    if not np.isclose(float(inward_magnitude), float(config.get("inward_magnitude", np.nan))):
        raise ValueError("runner inward magnitude differs from oracle preflight")
    if tuple(float(item) for item in tangent_magnitudes) != tuple(config.get("tangent_magnitudes", ())):
        raise ValueError("runner tangent magnitudes differ from oracle preflight")
    if tuple(float(item) for item in contact_normal_magnitudes) != tuple(config.get("contact_normal_magnitudes", ())):
        raise ValueError("runner contact normal magnitudes differ from oracle preflight")


def _root_disjoint(first: list[dict[str, np.ndarray]], second: list[dict[str, np.ndarray]]) -> None:
    from experiments.audit_pusht_reachability_horizon import root_fingerprint

    overlap = {root_fingerprint(root) for root in first} & {root_fingerprint(root) for root in second}
    if overlap:
        raise ValueError("calibration and outcome feasibility cohorts overlap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--calibration-roots", type=int, default=16)
    parser.add_argument("--outcome-roots", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=202610231)
    parser.add_argument("--outcome-seed", type=int, default=202610232)
    parser.add_argument("--minimum-reachable-roots", type=int, default=1)
    parser.add_argument("--minimum-reachable-rate", type=float, default=0.10)
    parser.add_argument("--required-oracle-arm", choices=ORACLE_ARMS, default="two_pulse")
    parser.add_argument("--macro-steps", type=int, default=18)
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument("--three-pulse-beam-width", type=int, default=8)
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--contact-normal-magnitudes", type=float, nargs="+", default=[0.02, 0.04, 0.08])
    args = parser.parse_args()
    if min(args.calibration_roots, args.outcome_roots, args.macro_steps, args.pulse_steps) < 1:
        parser.error("root counts, macro steps, and pulse steps must be positive")
    if args.three_pulse_beam_width < 1:
        parser.error("three-pulse beam width must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable preflight output: {args.output}")
    _device, runtime = resolve_device(args.device)
    started = time.perf_counter()
    protocol = {
        "environment": "swm/PushT-v1",
        "root_protocol": "unique_nonterminal_boundary_contact",
        "oracle_arms": ["one_pulse", "two_pulse", "three_pulse_bounded"],
        "required_oracle_arm": str(args.required_oracle_arm),
        "macro_steps": int(args.macro_steps),
        "pulse_steps": int(args.pulse_steps),
        "three_pulse_beam_width": int(args.three_pulse_beam_width),
        "action_bank_config": {
            "boundary_action_frame": True,
            "inward_magnitude": float(args.inward_magnitude),
            "tangent_magnitudes": [float(value) for value in args.tangent_magnitudes],
            "contact_normal_magnitudes": [float(value) for value in args.contact_normal_magnitudes],
        },
        "cohort_contract": {
            "calibration": {
                "root_seed": int(args.calibration_seed),
                "root_count": int(args.calibration_roots),
                "oversample_factor": 4,
            },
            "outcome": {
                "root_seed": int(args.outcome_seed),
                "root_count": int(args.outcome_roots),
                "oversample_factor": 4,
            },
        },
    }
    protocol["protocol_sha256"] = protocol_fingerprint(protocol)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        calibration_roots = collect_nonterminal_boundary_contact_roots(
            env, count=int(args.calibration_roots), seed=int(args.calibration_seed), oversample_factor=4,
        )
        outcome_roots = collect_nonterminal_boundary_contact_roots(
            env, count=int(args.outcome_roots), seed=int(args.outcome_seed), oversample_factor=4,
        )
        _root_disjoint(calibration_roots, outcome_roots)
        calibration_rows = [run_root(env, index, root, args) for index, root in enumerate(calibration_roots)]
        outcome_rows = [run_root(env, index, root, args) for index, root in enumerate(outcome_roots)]
    finally:
        env.close()
    calibration_manifest = cohort_manifest(calibration_rows)
    outcome_manifest = cohort_manifest(outcome_rows)
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Privileged candidate-bank/horizon feasibility gate only. It is not a learned/RGB/LeWM planning result, "
            "does not expose successors to a learned ranker, and does not demonstrate physical causality or universal world modeling."
        ),
        "protocol": {**protocol, "terminal_root_exclusion": "all roots are nonterminal before any oracle decision", **runtime},
        "cohorts": {"calibration": calibration_manifest, "outcome": outcome_manifest},
        "calibration_authorization": authorize_calibration(
            calibration_manifest,
            minimum_reachable_roots=int(args.minimum_reachable_roots),
            minimum_reachable_rate=float(args.minimum_reachable_rate),
            required_oracle_arm=str(args.required_oracle_arm),
        ),
        "outcome_ceiling": {
            "required_oracle_arm": str(args.required_oracle_arm),
            "reachable_roots": int(sum(
                item["oracle_arm_pose_success"][str(args.required_oracle_arm)]
                for item in outcome_manifest
            )),
            "root_count": int(len(outcome_manifest)),
            "note": "outcome ceiling is diagnostic only and cannot change calibration authorization",
        },
        "sanity_checks": {
            "root_disjoint": True,
            "terminal_roots_excluded": True,
            "all_oracle_arms_share_each_root_bank": True,
            "authorization_uses_calibration_only": True,
            "learned_runtime_requires_exact_protocol_and_root_bank_manifest": True,
        },
        "wall_seconds": float(time.perf_counter() - started),
    }
    args.output.mkdir(parents=True)
    path = args.output / "preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "authorization": report["calibration_authorization"]}, indent=2))


if __name__ == "__main__":
    main()
