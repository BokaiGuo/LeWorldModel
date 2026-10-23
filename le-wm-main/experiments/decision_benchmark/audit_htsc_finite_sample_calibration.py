"""Audit the finite-sample HTSC fixed-bank calibration contract.

This is a deliberately narrow calibration audit.  It freezes an
intervention-supervised latent-plus-action effect score after fitting on
development *train* roots, calibrates only on development *calibration* roots,
and evaluates simultaneous candidate-bank contrast coverage on a disjoint
external-root bank.  It neither changes the live planner nor converts the
adapter into an RGB-only world-model claim.

The script also contains a synthetic selection-shift stress test.  The test is
not a model-performance result: it checks that a finite-sample fixed-bank
certificate has the expected coverage under the declared candidate distribution
and can fail after a larger post-score proposal bank is substituted.  The
runtime certificate therefore rejects adaptive candidate generation rather than
silently applying the fixed-bank threshold to CEM-like proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    certify_candidate_bank,
    evaluate_contrast_bound,
)
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    SEEDS,
    _features,
    _fit_predict,
    action_sequence_basis,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "htsc.finite_sample_calibration_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty candidate dataset")
    return [dict(row) for row in records]


def _canonical_action(row: dict[str, Any]) -> str:
    value = np.asarray(row["action_sequence"], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("candidate action sequences must be finite [steps, 2] arrays")
    return json.dumps(value.round(12).tolist(), separators=(",", ":"))


def _group_by_root(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["instance_id"])].append(row)
    for root in groups:
        groups[root].sort(key=lambda row: int(row["candidate_id"]))
    return dict(sorted(groups.items()))


def _latent_representation_signature(records: list[dict[str, Any]]) -> dict[str, int]:
    """Require one frozen latent coordinate shape before score fitting.

    Concatenating two different encoder coordinates merely because both are
    numeric would manufacture a false external-calibration experiment.  The
    signature intentionally checks the input representation rather than adding
    a learned projection between panels.
    """

    dimensions = {
        int(np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1).size)
        for row in records
    }
    if len(dimensions) != 1:
        raise ValueError("each panel must use one fixed source-latent dimension")
    return {"source_latent_dimension": int(next(iter(dimensions)))}


def validate_fixed_bank_protocol(
    development: list[dict[str, Any]],
    external: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless splits, roots, and action identities are declared."""

    development_splits = {str(row.get("split")) for row in development}
    if development_splits - {"train", "calibration"}:
        raise ValueError("development data must contain only train and calibration rows")
    if {str(row.get("split")) for row in external} != {"external_outcome"}:
        raise ValueError("external data must contain only external_outcome rows")
    dev_groups = _group_by_root(development)
    ext_groups = _group_by_root(external)
    if set(dev_groups) & set(ext_groups):
        raise ValueError("development and external roots must be disjoint")
    if not dev_groups or not ext_groups:
        raise ValueError("both development and external root panels are required")

    expected_ids: tuple[int, ...] | None = None
    action_by_id: dict[int, str] = {}
    split_by_root: dict[str, str] = {}
    for panel, groups in (("development", dev_groups), ("external", ext_groups)):
        for root, rows in groups.items():
            root_splits = {str(row["split"]) for row in rows}
            if len(root_splits) != 1:
                raise ValueError(f"{panel} root {root!r} appears in more than one split")
            split_by_root[root] = next(iter(root_splits))
            candidate_ids = tuple(int(row["candidate_id"]) for row in rows)
            if len(candidate_ids) < 2 or len(set(candidate_ids)) != len(candidate_ids):
                raise ValueError(f"{panel} root {root!r} needs at least two distinct candidates")
            if expected_ids is None:
                expected_ids = candidate_ids
            elif candidate_ids != expected_ids:
                raise ValueError("all roots must share exactly the same candidate IDs")
            for row in rows:
                candidate_id = int(row["candidate_id"])
                action = _canonical_action(row)
                previous = action_by_id.setdefault(candidate_id, action)
                if previous != action:
                    raise ValueError("candidate IDs must map to one common action sequence across roots")
    if not any(split == "train" for split in split_by_root.values()):
        raise ValueError("development panel has no train roots")
    if not any(split == "calibration" for split in split_by_root.values()):
        raise ValueError("development panel has no calibration roots")
    return {
        "candidate_ids": list(expected_ids or ()),
        "candidate_action_sha256": {
            str(key): hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in sorted(action_by_id.items())
        },
        "candidate_bank_sha256": hashlib.sha256(
            json.dumps(action_by_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "development_root_count": len(dev_groups),
        "external_root_count": len(ext_groups),
        "train_root_count": int(sum(split == "train" for split in split_by_root.values())),
        "calibration_root_count": int(sum(split == "calibration" for split in split_by_root.values())),
        "root_overlap_count": 0,
    }


def _predicted_task_cost(records: list[dict[str, Any]], effect: np.ndarray) -> np.ndarray:
    if len(records) != len(effect):
        raise ValueError("records and effect predictions must align")
    source = np.asarray([row["source_state_raw"] for row in records], dtype=np.float64)[:, :2]
    goal = np.asarray([row["goal_raw"] for row in records], dtype=np.float64)[:, :2]
    return np.linalg.norm(source + np.asarray(effect, dtype=np.float64) - goal, axis=1)


def _frozen_latent_action_basis_inputs(records: list[dict[str, Any]]) -> np.ndarray:
    """Build outcome-free external inputs without opening realized successors.

    `_features` is appropriate for train roots because it constructs supervised
    effect targets.  Calling it on held roots would deserialize their realized
    state merely to discard the target, which weakens the no-outcome-read
    protocol.  This input-only mirror intentionally uses exactly its deployed
    feature family: frozen source latent plus the deterministic action basis.
    """

    values = []
    for row in records:
        source = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1)
        action = action_sequence_basis(np.asarray(row["action_sequence"], dtype=np.float64))
        values.append(np.concatenate([source, action]))
    return np.asarray(values, dtype=np.float32)


def _row_arrays(records: list[dict[str, Any]], predicted_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row["real_task_cost"]) for row in records], dtype=np.float64),
        np.asarray(predicted_cost, dtype=np.float64),
        np.asarray([str(row["instance_id"]) for row in records], dtype=object),
    )


def simulate_fixed_vs_adaptive_selection_shift(
    *,
    alpha: float = 0.1,
    calibration_roots: int = 19,
    fixed_candidate_count: int = 4,
    adaptive_proposal_count: int = 32,
    trials: int = 4000,
    seed: int = 2026102107,
) -> dict[str, Any]:
    """Stress the declared-bank assumption with continuous synthetic errors.

    Each root is exchangeable before the proposal-bank substitution.  The fixed
    audit has exactly the bank size used at calibration; the adaptive audit
    replaces it with a larger post-calibration proposal pool.
    This is a selection/distribution shift demonstration, not a claim that all
    CEM implementations have this numerical failure rate.
    """

    if adaptive_proposal_count <= fixed_candidate_count:
        raise ValueError("adaptive_proposal_count must exceed fixed_candidate_count")
    if trials < 1:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)

    def root_error(candidate_count: int) -> float:
        # A root's simultaneous contrast error equals the maximum pairwise
        # discrepancy produced by an unobserved action-specific score error.
        task_cost = rng.normal(loc=0.0, scale=1.0, size=candidate_count)
        score_noise = rng.normal(loc=0.0, scale=0.35, size=candidate_count)
        predicted_cost = task_cost + score_noise
        contrast_error = (task_cost[:, None] - task_cost[None, :]) - (
            predicted_cost[:, None] - predicted_cost[None, :]
        )
        return float(np.max(np.abs(contrast_error)))

    fixed_covered = 0
    adaptive_covered = 0
    finite_bounds = 0
    for _ in range(int(trials)):
        calibration_scores = np.asarray(
            [root_error(fixed_candidate_count) for _ in range(int(calibration_roots))],
            dtype=np.float64,
        )
        # This import path is intentionally exercised through calibration to
        # validate the exact same finite-sample rank as deployment.
        calibration = calibrate_contrast_bound(
            real_cost=np.column_stack([
                calibration_scores, np.zeros_like(calibration_scores),
            ]).reshape(-1),
            predicted_cost=np.zeros(2 * calibration_scores.size, dtype=np.float64),
            group_id=np.repeat(np.arange(calibration_scores.size), 2),
            calibration_groups=np.arange(calibration_scores.size),
            alpha=alpha,
        )
        bound = calibration["bound"]
        if bound is None:
            continue
        finite_bounds += 1
        fixed_covered += int(root_error(fixed_candidate_count) <= float(bound))
        adaptive_covered += int(root_error(adaptive_proposal_count) <= float(bound))
    if finite_bounds == 0:
        raise RuntimeError("simulation unexpectedly has no finite calibration bounds")
    return {
        "schema": "htsc.fixed_vs_adaptive_selection_shift_simulation.v1",
        "alpha": float(alpha),
        "calibration_root_count": int(calibration_roots),
        "fixed_candidate_count": int(fixed_candidate_count),
        "adaptive_proposal_count": int(adaptive_proposal_count),
        "trials": int(trials),
        "finite_bound_trial_count": int(finite_bounds),
        "fixed_declared_bank_empirical_coverage": float(fixed_covered / finite_bounds),
        "adaptive_proposal_bank_naive_empirical_coverage": float(adaptive_covered / finite_bounds),
        "adaptive_runtime_policy": "must abstain: adaptive_candidate_generation_not_certified",
        "claim_boundary": (
            "Continuous synthetic stress test of candidate-bank distribution shift. "
            "It validates the fail-closed contract, not a physical-control result "
            "or a universal statement about adaptive planners."
        ),
    }


def run_audit(
    development_dataset: Path,
    development_arrays: Path,
    external_dataset: Path,
    *,
    alpha: float,
    epochs: int,
    max_external_roots: int | None,
    simulation_trials: int,
    device: str,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    """Fit once on development train roots and audit frozen scores on fresh roots."""

    runtime = require_cuda_allocation(device)
    development_all = _load_records(development_dataset)
    external_all = _load_records(external_dataset)
    development = [
        row for row in development_all if str(row.get("split")) in {"train", "calibration"}
    ]
    external = [row for row in external_all if str(row.get("split")) == "external_outcome"]
    if len(development) != len(development_all):
        # A mixed source file is allowed only when its held rows are never
        # admitted into this audit.  The count is recorded as a provenance
        # warning rather than claiming the file itself was outcome-free.
        source_panel_note = "source file contained excluded non-development rows"
    else:
        source_panel_note = "development source contained train/calibration rows only"
    if max_external_roots is not None:
        kept = set(sorted(_group_by_root(external))[: int(max_external_roots)])
        external = [row for row in external if str(row["instance_id"]) in kept]
    full_protocol = validate_fixed_bank_protocol(development, external)
    if candidate_count is not None:
        count = int(candidate_count)
        available = full_protocol["candidate_ids"]
        if count < 2 or count > len(available):
            raise ValueError("candidate_count must lie between two and the declared bank size")
        retained = set(available[:count])
        development = [row for row in development if int(row["candidate_id"]) in retained]
        external = [row for row in external if int(row["candidate_id"]) in retained]
    protocol = validate_fixed_bank_protocol(development, external)
    development_representation = _latent_representation_signature(development)
    external_representation = _latent_representation_signature(external)
    if development_representation != external_representation:
        raise ValueError(
            "development and external panels use incompatible frozen latent coordinates; "
            "refusing any learned projection or zero-padding repair"
        )
    protocol["frozen_representation"] = development_representation
    if not development_arrays.is_file():
        raise FileNotFoundError("development arrays are required for supervised effect fitting")

    dev_arrays = np.load(development_arrays, allow_pickle=False)
    dev_x, dev_y, _, _ = _features(development, dev_arrays, "frozen_latent_action_basis")
    ext_x = _frozen_latent_action_basis_inputs(external)
    train_mask = np.asarray([str(row["split"]) == "train" for row in development])
    calibration_mask = np.asarray([str(row["split"]) == "calibration" for row in development])
    if not train_mask.any() or not calibration_mask.any():
        raise ValueError("development data needs both train and calibration roots")
    combined_x = np.concatenate([dev_x, ext_x], axis=0)
    members, member_training = [], []
    for seed in SEEDS:
        prediction, metadata = _fit_predict(
            dev_x[train_mask], dev_y[train_mask], combined_x,
            seed=seed, epochs=int(epochs), device_name=device,
        )
        members.append(prediction)
        member_training.append(metadata)
    mean_effect = np.mean(np.stack(members, axis=0), axis=0)
    dev_cost = _predicted_task_cost(development, mean_effect[: len(development)])
    external_cost = _predicted_task_cost(external, mean_effect[len(development) :])
    cal_real, cal_predicted, cal_groups = _row_arrays(
        [row for row, keep in zip(development, calibration_mask) if keep], dev_cost[calibration_mask]
    )
    calibration_groups = sorted(set(cal_groups.tolist()))
    calibration = calibrate_contrast_bound(
        cal_real, cal_predicted, cal_groups, calibration_groups, alpha=alpha,
    )
    ext_real, ext_predicted, ext_groups = _row_arrays(external, external_cost)
    coverage = evaluate_contrast_bound(ext_real, ext_predicted, ext_groups, calibration["bound"])

    by_root = _group_by_root(external)
    certificates = []
    for root, rows in by_root.items():
        indices = sorted(
            [index for index, row in enumerate(external) if str(row["instance_id"]) == root],
            key=lambda index: int(external[index]["candidate_id"]),
        )
        # This audit has no independently calibrated deployment support model.
        # Keep action release fail-closed while still reporting the purely
        # mathematical fixed-bank regret inequality on covered roots.
        certificate = certify_candidate_bank(
            external_cost[indices], calibration["bound"], support_passed=False,
            candidate_ids=[int(row["candidate_id"]) for row in rows], candidate_bank_mode="fixed",
        )
        adaptive = certify_candidate_bank(
            external_cost[indices], calibration["bound"], support_passed=False,
            candidate_ids=[int(row["candidate_id"]) for row in rows], candidate_bank_mode="adaptive",
        )
        certificates.append({
            "root_id": root,
            "fixed_certificate": certificate,
            # This is not a runtime release: it answers whether the score
            # margins would clear the same mathematical contrast threshold if
            # an independently valid support gate were later supplied.
            "support_passed_counterfactual_status": certify_candidate_bank(
                external_cost[indices], calibration["bound"], support_passed=True,
                candidate_ids=[int(row["candidate_id"]) for row in rows], candidate_bank_mode="fixed",
            )["status"],
            "adaptive_counterfactual_status": adaptive["status"],
            "adaptive_counterfactual_reason": adaptive["reason"],
        })
    tolerance = 0.25
    finite_bounds = [
        float(row["fixed_certificate"]["joint_regret_upper_bound"])
        for row in certificates
        if row["fixed_certificate"]["joint_regret_upper_bound"] is not None
    ]
    non_vacuous = [value for value in finite_bounds if value <= tolerance]
    simulation = simulate_fixed_vs_adaptive_selection_shift(
        alpha=alpha, calibration_roots=len(calibration_groups), trials=simulation_trials,
    )
    return {
        "schema": SCHEMA,
        "runtime": runtime,
        "protocol": {
            **protocol,
            "score": "ensemble intervention-supervised frozen-latent plus action-basis effect adapter",
            "score_training_roots": "development train only",
            "calibration_roots": "development calibration only",
            "outcome_roots": "external_outcome only",
            "candidate_bank": "one declared world-fixed action sequence per ID",
            "candidate_bank_selection": (
                "full declared bank" if candidate_count is None
                else f"predeclared candidate-ID prefix [0,{int(candidate_count) - 1}]"
            ),
            "outcome_labels_used_for_fit_or_calibration": False,
            "deployment_support_gate": "not calibrated in this audit; release intentionally fail-closed",
            "source_panel_note": source_panel_note,
            "predeclared_non_vacuous_regret_tolerance": tolerance,
        },
        "provenance": {
            "development_dataset": str(development_dataset.resolve()),
            "development_dataset_sha256": _sha256(development_dataset),
            "development_arrays": str(development_arrays.resolve()),
            "development_arrays_sha256": _sha256(development_arrays),
            "external_dataset": str(external_dataset.resolve()),
            "external_dataset_sha256": _sha256(external_dataset),
            "external_realized_successor_arrays_opened": False,
        },
        "calibration": calibration,
        "independent_external_root_coverage": coverage,
        "candidate_bank_certificates": {
            "root_count": len(certificates),
            "permit_count": int(sum(row["fixed_certificate"]["status"] == "permit" for row in certificates)),
            "support_passed_counterfactual_permit_count": int(sum(
                row["support_passed_counterfactual_status"] == "permit" for row in certificates
            )),
            "finite_joint_regret_bound_count": len(finite_bounds),
            "non_vacuous_joint_regret_bound_count": len(non_vacuous),
            "non_vacuous_joint_regret_bound_rate": float(len(non_vacuous) / len(certificates)),
            "adaptive_counterfactual_abstain_count": int(sum(
                row["adaptive_counterfactual_status"] == "abstain" for row in certificates
            )),
            "rows": certificates,
        },
        "member_training": member_training,
        "selection_shift_simulation": simulation,
        "claim_boundary": (
            "This validates finite-sample marginal simultaneous contrast calibration for one frozen, "
            "declared fixed candidate bank and reports empirical coverage on disjoint external roots. "
            "It does not establish permit-conditional safety, action-conditional coverage, adaptive-CEM "
            "coverage, RGB-only deployment, closed-loop improvement, or a frozen-world-model causal benefit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--development-arrays", type=Path, required=True)
    parser.add_argument("--external-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-external-roots", type=int, default=32)
    parser.add_argument("--simulation-trials", type=int, default=4000)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--candidate-count", type=int, default=None, help="predeclared prefix of the common fixed candidate bank")
    args = parser.parse_args()
    if not 0.0 < float(args.alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if int(args.epochs) < 1:
        raise ValueError("epochs must be positive")
    result = run_audit(
        args.development_dataset, args.development_arrays,
        args.external_dataset,
        alpha=float(args.alpha), epochs=int(args.epochs),
        max_external_roots=args.max_external_roots,
        simulation_trials=int(args.simulation_trials), device=args.device,
        candidate_count=args.candidate_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "finite_bound": result["calibration"]["bound"],
        "external_coverage": result["independent_external_root_coverage"]["empirical_root_coverage"],
        "permit_count": result["candidate_bank_certificates"]["permit_count"],
        "support_passed_counterfactual_permit_count": result["candidate_bank_certificates"]["support_passed_counterfactual_permit_count"],
        "non_vacuous_bound_count": result["candidate_bank_certificates"]["non_vacuous_joint_regret_bound_count"],
        "fixed_simulated_coverage": result["selection_shift_simulation"]["fixed_declared_bank_empirical_coverage"],
        "adaptive_simulated_coverage": result["selection_shift_simulation"]["adaptive_proposal_bank_naive_empirical_coverage"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
