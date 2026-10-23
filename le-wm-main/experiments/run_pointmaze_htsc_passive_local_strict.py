"""Strict RGB evaluation of passive verified local residual repair.

Unlike the V5--V8 active-probe panels, the passive arm never executes an
extra information action.  It initially behaves as HTSC fallback, records the
RGB successor of each ordinary executed action, and may re-rank only after a
rank-two, prior-anchored residual chart is supported by those already-charged
actions.  This directly tests whether the active probe's physical budget was
the limiting cost without relaxing support or post-action verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VISUAL_METHOD = "visual_dynamics"
ADAPTER_METHOD = "effect_adapter_rgb"
HTSC_METHOD = "effect_adapter_rgb_htsc"
PASSIVE_METHOD = "effect_adapter_rgb_online_prior_anchored_passive_residual_verified_local"
FULL_METHOD = "effect_adapter_rgb_htsc_consistency"
METHODS = (VISUAL_METHOD, ADAPTER_METHOD, HTSC_METHOD, PASSIVE_METHOD, FULL_METHOD)
TOTAL_BLOCK_BUDGET = 25
BOOTSTRAP_DRAWS = 10_000


def parse_roots(value: str, label: str) -> list[int]:
    try:
        roots = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as error:
        raise ValueError(f"{label} must be comma-separated integer roots") from error
    if not roots or len(set(roots)) != len(roots) or any(root < 0 for root in roots):
        raise ValueError(f"{label} must be non-empty, unique, and non-negative")
    return roots


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_live(
    *, dataset_root: Path, state_root: Path, checkpoint: Path, dinov2_source: Path,
    effect_policy: Path, output: Path, calibration_roots: list[int], test_roots: list[int],
    method: str, consistency_roots: list[int] | None,
) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "experiments.run_live_pointmaze_planning",
        "--dataset-root", str(dataset_root.resolve()),
        "--test-state-root", str(state_root.resolve()),
        "--test-visual-source", "dataset_matched_renderer",
        "--checkpoint", str(checkpoint.resolve()),
        "--dinov2-source", str(dinov2_source.resolve()),
        "--effect-policy-checkpoint", str(effect_policy.resolve()),
        "--output", str(output.resolve()),
        "--calibration-episodes", ",".join(map(str, calibration_roots)),
        "--test-episodes", ",".join(map(str, test_roots)),
        "--max-blocks", str(TOTAL_BLOCK_BUDGET - 1),
        "--strict-shared-root-candidate-bank", "--methods", method,
    ]
    if consistency_roots is not None:
        command.extend([
            "--effect-consistency-calibration-episodes",
            ",".join(map(str, consistency_roots)),
        ])
    _write_json(output.with_suffix(".command.json"), {"command": command})
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def _rows_by_root(report: dict[str, Any], method: str) -> dict[int, dict[str, Any]]:
    rows = [row for row in report.get("rows", []) if row.get("method") == method]
    result = {int(row["episode_id"]): row for row in rows}
    if len(result) != len(rows) or not result:
        raise ValueError(f"{method}: missing unique root rows")
    return result


def _bootstrap(values: np.ndarray, *, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("paired values must be non-empty")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _paired(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired methods must share exact root IDs")
    roots = sorted(left)
    distance = np.asarray([
        float(left[root]["final_distance"]) - float(right[root]["final_distance"])
        for root in roots
    ])
    success = np.asarray([
        float(left[root]["success"]) - float(right[root]["success"])
        for root in roots
    ])
    return {
        "root_count": int(len(roots)),
        "final_distance_delta_left_minus_right": float(distance.mean()),
        "final_distance_delta_95pct_bootstrap": _bootstrap(distance, seed=9107),
        "success_delta_left_minus_right": float(success.mean()),
        "success_delta_95pct_bootstrap": _bootstrap(success, seed=9108),
    }


def validate(rows_by_method: dict[str, dict[int, dict[str, Any]]], roots: list[int]) -> None:
    expected = set(roots)
    for method, rows in rows_by_method.items():
        if set(rows) != expected:
            raise ValueError(f"{method}: root set differs from frozen panel")
        for root, row in rows.items():
            contract = row.get("policy_contract", {})
            if bool(contract.get("simulator_state_in_policy_input", True)):
                raise ValueError(f"{method} root {root}: simulator state leaked into policy")
            if not bool(contract.get("strict_shared_root_candidate_bank", False)):
                raise ValueError(f"{method} root {root}: missing fixed candidate bank")
            charged = int(row["warmup_blocks"]) + int(row["active_probe_blocks"]) + int(row["blocks_executed"])
            if charged > TOTAL_BLOCK_BUDGET:
                raise ValueError(f"{method} root {root}: exceeded shared physical budget")
            if method == PASSIVE_METHOD:
                if int(row["active_probe_blocks"]) != 0:
                    raise ValueError("passive local method spent an undeclared probe action")
                if not bool(contract.get("passive_local_observation_only", False)):
                    raise ValueError("passive local method lacks its no-extra-action contract")
    for root in roots:
        digests = {
            str(rows_by_method[method][root]["policy_contract"].get("candidate_bank_digest"))
            for method in METHODS
        }
        if len(digests) != 1 or None in digests:
            raise ValueError(f"root {root}: candidate bank differs across methods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--test-state-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--effect-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-calibration-roots", required=True)
    parser.add_argument("--consistency-calibration-roots", required=True)
    parser.add_argument("--test-roots", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    calibration_roots = parse_roots(args.visual_calibration_roots, "visual calibration roots")
    consistency_roots = parse_roots(args.consistency_calibration_roots, "consistency calibration roots")
    test_roots = parse_roots(args.test_roots, "test roots")
    named = {"visual": set(calibration_roots), "consistency": set(consistency_roots), "test": set(test_roots)}
    if any(named[left] & named[right] for left in named for right in named if left < right):
        raise ValueError("visual, consistency, and held-out roots must be disjoint")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse immutable output directory: {args.output_dir}")
    if not args.source_manifest.is_file():
        raise FileNotFoundError(f"missing source manifest: {args.source_manifest}")
    # Do not create an immutable scientific-result directory until the exact
    # CUDA/cuDNN execution contract has succeeded.  In particular, a managed
    # bwrap process must fail before it can leave a directory that could be
    # mistaken for an interrupted host experiment.
    subprocess.run(
        [sys.executable, "-m", "experiments.gpu_preflight", "--device", "cuda", "--recurrent-smoke"],
        cwd=ROOT, check=True,
    )
    args.output_dir.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "experiments.gpu_preflight", "--device", "cuda", "--recurrent-smoke",
         "--output", str(args.output_dir / "gpu_preflight.json")],
        cwd=ROOT, check=True,
    )
    _write_json(args.output_dir / "root_provenance.json", {
        "schema": "pointmaze.passive_local.root_provenance.v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "test_state_root": str(args.test_state_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "visual_calibration_roots": calibration_roots,
        "consistency_calibration_roots": consistency_roots,
        "held_out_test_roots": test_roots,
        "passive_method_predeclared": PASSIVE_METHOD,
        "held_out_outcomes_read_before_execution": False,
    })
    reports: dict[str, dict[str, Any]] = {}
    output = args.output_dir / "held_out"
    for method in METHODS:
        reports[method] = _run_live(
            dataset_root=args.dataset_root, state_root=args.test_state_root,
            checkpoint=args.checkpoint, dinov2_source=args.dinov2_source,
            effect_policy=args.effect_policy_checkpoint,
            output=output / f"{method}.json", calibration_roots=calibration_roots,
            test_roots=test_roots, method=method,
            consistency_roots=consistency_roots if method == FULL_METHOD else None,
        )
    rows = {method: _rows_by_root(reports[method], method) for method in METHODS}
    validate(rows, test_roots)
    passive_rows = rows[PASSIVE_METHOD]
    gates = [
        trace.get("action_transfer_gate", {})
        for row in passive_rows.values() for trace in row.get("trace", [])
        if isinstance(trace, dict)
    ]
    report = {
        "schema": "pointmaze.htsc_passive_local.strict_rgb_closed_loop.v1",
        "status": "completed",
        "methods": {method: method for method in METHODS},
        "split_protocol": {
            "visual_calibration_roots": calibration_roots,
            "consistency_calibration_roots": consistency_roots,
            "held_out_test_roots": test_roots,
            "pairwise_disjoint": True,
        },
        "budget_contract": {
            "total_physical_block_budget": TOTAL_BLOCK_BUDGET,
            "passive_information_actions": 0,
            "local_chart_updates_only_from_executed_RGB_successors": True,
            "all_methods_rgb_only": True,
        },
        "paired_passive_vs_htsc": _paired(passive_rows, rows[HTSC_METHOD]),
        "passive_mechanism": {
            "observed_update_decisions": int(sum(bool(gate.get("passive_local_observed_update", False)) for gate in gates)),
            "local_verified_release_decisions": int(sum(bool(gate.get("local_verified_release", False)) for gate in gates)),
            "local_override_decisions": int(sum(
                str(gate.get("fallback_reason", "")).endswith("residual_local_override")
                for gate in gates
            )),
            "no_extra_probe_action_count": int(sum(int(row["active_probe_blocks"]) for row in passive_rows.values())),
        },
        "aggregate": {method: reports[method].get("aggregates", {}).get(method) for method in METHODS},
        "artifacts": {method: str((output / f"{method}.json").resolve()) for method in METHODS},
        "claim_boundary": (
            "A positive paired consequence would establish only a passive, RGB-observed local repair in this "
            "legacy PointMaze protocol; it would not establish general counterfactual sufficiency."
        ),
    }
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps({"output": str(args.output_dir / "report.json"), "paired": report["paired_passive_vs_htsc"], "mechanism": report["passive_mechanism"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
