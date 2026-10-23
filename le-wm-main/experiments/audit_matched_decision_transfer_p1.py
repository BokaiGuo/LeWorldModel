"""Matched learned-model P1 audit: does a pre-outcome diagnostic add beyond ATM?

The audit deliberately makes a smaller claim than a universal checkpoint
selector.  It asks a specific reviewer-facing question on an immutable panel:
after exact matching on architecture, stage, horizon support, latent dimension,
and action support -- and pairwise matching on prediction MSE and ATM -- does
the finite-horizon action-conditioned innovation diagnostic improve *held-out*
within-stratum risk selection?

``P1`` is the finite-horizon action-conditioned innovation concentration
(FH-AIC).  For each declared horizon it is the squared Frobenius correlation
between centered deployed action sequences and centered predictor residuals,
then averaged over the fixed horizon bank.  It is computed only from the fit
export.  A high value means that prediction errors systematically vary with
the actions whose consequences the planner must rank; it is therefore
pre-declared as higher risk.  This differs from ATM, which is an independent
action-effect inverse-risk measurement.

No unmatched correlation is emitted.  If the available checkpoint panel cannot
form a valid MSE+ATM matched seed stratum, the report records that structural
deficiency and leaves selection statistics unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_VERSION = "matched_decision_transfer_p1.v1"
DEFAULT_MSE_TOLERANCE = 0.05
DEFAULT_ATM_TOLERANCE = 0.05
DEFAULT_MIN_CANDIDATES = 3


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _relative_distance(left: float, right: float) -> float:
    return float(abs(left - right) / max((abs(left) + abs(right)) / 2.0, 1e-12))


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so exact diagnostic ties cannot depend on file order."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and np.isclose(values[order[start]], values[order[end]], rtol=0.0, atol=1e-12):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _action_support_hash(rows: list[dict[str, Any]]) -> str:
    """Hash the complete ordered action support, rather than only its name."""
    actions = np.asarray([row["action_blocks_normalized"] for row in rows], dtype=np.float64)
    if actions.ndim != 3 or actions.shape[0] < 2:
        raise ValueError("fit export must contain at least two action-block rows")
    # The row order is an episode/window ordering, not an action-support
    # definition.  Sort canonical byte rows to make the support identity
    # invariant to presentation order while retaining all action sequences.
    flattened = np.ascontiguousarray(actions.reshape(len(actions), -1))
    canonical = np.asarray(sorted(tuple(float(x) for x in row) for row in flattened), dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()


def finite_horizon_action_conditioned_innovation(
    fit_payload: dict[str, Any], horizon_support: tuple[int, ...],
) -> tuple[float, dict[str, float]]:
    """Return P1 from a fit export, without opening its outcome counterpart.

    For each horizon h, the action prefix through h blocks is paired with the
    h-step latent innovation (target minus prediction).  The normalized
    cross-covariance has an exact [0, 1] Frobenius bound by Cauchy--Schwarz.
    It needs no learned hyperparameter or held-out planning label.
    """
    rows = fit_payload.get("enriched_records")
    if not isinstance(rows, list) or len(rows) < 8:
        raise ValueError("fit export requires at least eight enriched records")
    values: dict[str, float] = {}
    for horizon in horizon_support:
        actions, innovations = [], []
        for row in rows:
            blocks = np.asarray(row["action_blocks_normalized"], dtype=np.float64)
            if blocks.ndim != 2 or blocks.shape[0] < horizon:
                raise ValueError(f"fit action block support lacks horizon {horizon}")
            item = row["latent_by_horizon"].get(str(horizon))
            if not isinstance(item, dict):
                raise ValueError(f"fit export lacks latent horizon {horizon}")
            predicted = np.asarray(item["predicted"], dtype=np.float64)
            target = np.asarray(item["target"], dtype=np.float64)
            if predicted.shape != target.shape or predicted.ndim != 1:
                raise ValueError(f"invalid latent innovation at horizon {horizon}")
            actions.append(blocks[:horizon].reshape(-1))
            innovations.append(target - predicted)
        action_matrix = np.asarray(actions, dtype=np.float64)
        innovation_matrix = np.asarray(innovations, dtype=np.float64)
        action_matrix -= action_matrix.mean(axis=0, keepdims=True)
        innovation_matrix -= innovation_matrix.mean(axis=0, keepdims=True)
        denominator = float(np.sum(action_matrix**2) * np.sum(innovation_matrix**2))
        if denominator <= 1e-18:
            raise ValueError(f"degenerate action or innovation variation at horizon {horizon}")
        cross = action_matrix.T @ innovation_matrix
        value = float(np.sum(cross**2) / denominator)
        values[str(horizon)] = float(np.clip(value, 0.0, 1.0))
    return float(np.mean(list(values.values()))), values


@dataclass(frozen=True)
class Candidate:
    source_path: str
    checkpoint: str
    environment: str
    architecture: str
    seed: str
    stage: int
    latent_dimension: int
    action_distribution: str
    horizon_support: tuple[int, ...]
    action_support_hash: str
    mse: float
    atm: float
    p1: float
    p1_by_horizon: dict[str, float]
    planning_risk: float

    @property
    def stratum_key(self) -> tuple[str, str, int, int, str, tuple[int, ...], str]:
        return (
            self.environment,
            self.architecture,
            self.stage,
            self.latent_dimension,
            self.action_distribution,
            self.horizon_support,
            self.action_support_hash,
        )


def _resolve_export(record_path: Path, declared: str) -> Path:
    path = Path(declared)
    if path.is_absolute():
        return path
    # Records live under ``results/...`` while their provenance is rooted at
    # the le-wm project root.  Walk upward until the declared relative path is
    # found; this keeps copied result panels reproducible.
    for parent in (record_path.parent, *record_path.parents):
        candidate = parent / path
        if candidate.is_file():
            return candidate
    raise ValueError(f"fit export does not exist: {declared}")


def load_candidate(path: Path) -> Candidate:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") != "real_checkpoint_selection.v1":
        raise ValueError("incompatible record_schema")
    record = payload.get("record")
    protocol = payload.get("protocol")
    if not isinstance(record, dict) or not isinstance(protocol, dict):
        raise ValueError("record/protocol must be objects")
    factors = record.get("model_factors", {})
    diagnostics = record.get("diagnostics", {})
    heldout = record.get("heldout", {})
    required = {
        "architecture": record.get("architecture"),
        "checkpoint": record.get("checkpoint"),
        "environment": protocol.get("environment"),
        "latent_dimension": factors.get("latent_dimension"),
        "training_stage": factors.get("training_stage"),
        "action_distribution": factors.get("action_distribution"),
        "prediction_mse": diagnostics.get("prediction_mse"),
        "action_inverse_risk": diagnostics.get("action_inverse_risk"),
        "planning_risk": heldout.get("planning_risk"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for name in ("prediction_mse", "action_inverse_risk", "planning_risk"):
        if not _finite(required[name]):
            raise ValueError(f"non-finite {name}")
    if protocol.get("independent_external_evidence") is not True:
        raise ValueError("independent_external_evidence is not certified")
    provenance = record.get("provenance", {})
    fit_path = _resolve_export(path, str(provenance.get("fit_export", "")))
    fit_payload = json.loads(fit_path.read_text())
    horizons = tuple(sorted(int(item) for item in fit_payload.get("horizons", [])))
    if not horizons:
        raise ValueError("fit export has no horizon support")
    p1, by_horizon = finite_horizon_action_conditioned_innovation(fit_payload, horizons)
    rows = fit_payload["enriched_records"]
    checkpoint = str(required["checkpoint"])
    seed = str(record.get("group", checkpoint)).split("seed_")[-1].split("/")[0]
    if seed == str(record.get("group", checkpoint)):
        seed = str(fit_payload.get("seed", "unknown"))
    return Candidate(
        source_path=str(path), checkpoint=checkpoint, environment=str(required["environment"]),
        architecture=str(required["architecture"]), seed=seed, stage=int(required["training_stage"]),
        latent_dimension=int(required["latent_dimension"]), action_distribution=str(required["action_distribution"]),
        horizon_support=horizons, action_support_hash=_action_support_hash(rows),
        mse=float(required["prediction_mse"]), atm=float(required["action_inverse_risk"]),
        p1=p1, p1_by_horizon=by_horizon, planning_risk=float(required["planning_risk"]),
    )


def _maximum_clique(candidates: list[Candidate], mse_tolerance: float, atm_tolerance: float) -> tuple[list[int], int]:
    adjacency = [set() for _ in candidates]
    edge_count = 0
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if (
                _relative_distance(candidates[left].mse, candidates[right].mse) <= mse_tolerance
                and _relative_distance(candidates[left].atm, candidates[right].atm) <= atm_tolerance
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
                edge_count += 1
    best: tuple[int, ...] = ()

    def search(current: tuple[int, ...], possible: set[int]) -> None:
        nonlocal best
        if not possible:
            ordered = tuple(sorted(current))
            candidate_seeds = len({candidates[index].seed for index in ordered})
            best_seeds = len({candidates[index].seed for index in best})
            if (len(ordered), candidate_seeds, tuple(candidates[index].checkpoint for index in ordered)) > (
                len(best), best_seeds, tuple(candidates[index].checkpoint for index in best)
            ):
                best = ordered
            return
        for vertex in sorted(tuple(possible)):
            search(current + (vertex,), possible & adjacency[vertex])
            possible.remove(vertex)

    search((), set(range(len(candidates))))
    return list(best), edge_count


def _bootstrap(values: list[float], seed: int, draws: int = 20_000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_strata": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()), "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)), "n_strata": int(len(array)),
    }


def _sign_flip(values: list[float], seed: int, draws: int = 40_000) -> dict[str, Any]:
    result = _bootstrap(values, seed + 1, draws)
    if not values:
        return {**result, "p_one_sided": None, "positive_gate": False}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = float(array.mean())
    null = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(array))) @ array / len(array)
    p_value = float((1 + np.sum(null >= observed - 1e-15)) / (draws + 1))
    return {
        **result, "p_one_sided": p_value,
        "positive_gate": bool(observed > 0.0 and result["ci95_low"] > 0.0 and p_value < 0.05),
    }


def audit_candidates(
    candidates: Iterable[Candidate], *, mse_tolerance: float = DEFAULT_MSE_TOLERANCE,
    atm_tolerance: float = DEFAULT_ATM_TOLERANCE, min_candidates: int = DEFAULT_MIN_CANDIDATES,
    seed: int = 20261021,
) -> dict[str, Any]:
    """Run P1 selection only within the pre-declared strict matched strata."""
    candidates = list(candidates)
    if mse_tolerance < 0.0 or atm_tolerance < 0.0 or min_candidates < 2:
        raise ValueError("invalid matching thresholds")
    groups: dict[tuple[Any, ...], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.stratum_key].append(candidate)

    strata = []
    deltas: list[float] = []
    deficiencies: Counter[str] = Counter()
    for key, pool in sorted(groups.items(), key=lambda item: repr(item[0])):
        clique, edge_count = _maximum_clique(pool, mse_tolerance, atm_tolerance)
        matched = [pool[index] for index in clique]
        row: dict[str, Any] = {
            "environment": key[0], "architecture": key[1], "training_stage": key[2],
            "latent_dimension": key[3], "action_distribution": key[4],
            "horizon_support": list(key[5]), "action_support_sha256": key[6],
            "candidate_count": len(pool), "pairwise_mse_atm_compatible_edges": edge_count,
            "eligible_count": len(matched), "eligible_seeds": sorted({item.seed for item in matched}),
            "eligible_checkpoints": [item.checkpoint for item in matched], "identifiable": False,
            "reason_if_gated": None, "selection": None,
        }
        if len(matched) < min_candidates:
            row["reason_if_gated"] = "insufficient_joint_mse_atm_matched_candidates"
            deficiencies[row["reason_if_gated"]] += 1
        elif len({item.seed for item in matched}) < min_candidates:
            row["reason_if_gated"] = "insufficient_distinct_seeds_in_matched_clique"
            deficiencies[row["reason_if_gated"]] += 1
        else:
            mse = np.asarray([item.mse for item in matched])
            atm = np.asarray([item.atm for item in matched])
            p1 = np.asarray([item.p1 for item in matched])
            risk = np.asarray([item.planning_risk for item in matched])
            baseline_index = int(np.argmin((_rank(mse) + _rank(atm)) / 2.0))
            p1_index = int(np.argmin((_rank(mse) + _rank(atm) + _rank(p1)) / 3.0))
            oracle_index = int(np.argmin(risk))
            baseline_regret = float(risk[baseline_index] - risk[oracle_index])
            p1_regret = float(risk[p1_index] - risk[oracle_index])
            delta = float(baseline_regret - p1_regret)
            row.update({
                "identifiable": True,
                "selection": {
                    "directions": {"prediction_mse": "lower", "atm_action_inverse_risk": "lower", "p1_fh_action_conditioned_innovation": "lower"},
                    "candidates": [
                        {"checkpoint": item.checkpoint, "seed": item.seed, "prediction_mse": item.mse,
                         "atm_action_inverse_risk": item.atm, "p1_fh_action_conditioned_innovation": item.p1,
                         "p1_by_horizon": item.p1_by_horizon, "heldout_planning_risk": item.planning_risk}
                        for item in matched
                    ],
                    "baseline_mse_plus_atm": {"checkpoint": matched[baseline_index].checkpoint, "regret": baseline_regret},
                    "p1_augmented_mse_atm": {"checkpoint": matched[p1_index].checkpoint, "regret": p1_regret},
                    "oracle": {"checkpoint": matched[oracle_index].checkpoint, "planning_risk": float(risk[oracle_index])},
                    "incremental_delta_risk_baseline_minus_p1": delta,
                },
            })
            deltas.append(delta)
        strata.append(row)

    identifiable = [row for row in strata if row["identifiable"]]
    inferential = _sign_flip(deltas, seed)
    matched_architectures = sorted({row["architecture"] for row in identifiable})
    # Passing this pilot requires replication across architectures as well as
    # a positive independent outcome gate.  It remains a pilot, not a
    # cross-environment or universal transfer certificate.
    sufficient_replication = len(identifiable) >= 4 and len(matched_architectures) >= 2
    positive = bool(sufficient_replication and inferential["positive_gate"])
    if not identifiable:
        status = "structural_deficiency_no_valid_matched_strata"
    elif not sufficient_replication:
        status = "inconclusive_insufficient_matched_replication"
    elif positive:
        status = "passed_matched_p1_incremental_risk_pilot"
    else:
        status = "negative_matched_p1_incremental_risk_pilot"
    return {
        "artifact": "matched_decision_transfer_p1_audit",
        "protocol_version": PROTOCOL_VERSION,
        "protocol": {
            "estimand": "held-out delta R = regret(MSE+ATM) - regret(MSE+ATM+P1), within strict matched seed strata",
            "p1_definition": "mean_h ||X_h^T E_h||_F^2 / (||X_h||_F^2 ||E_h||_F^2), fit export only",
            "matching": {
                "environment": "exact", "architecture": "exact", "training_stage": "exact",
                "latent_dimension": "exact", "action_distribution": "exact",
                "horizon_support": "exact", "action_support": "complete normalized sequence support hash exact",
                "prediction_mse": f"pairwise symmetric relative distance <= {mse_tolerance}",
                "atm_action_inverse_risk": f"pairwise symmetric relative distance <= {atm_tolerance}",
            },
            "outcome_access": "only after all selectors are fixed from pre-outcome diagnostics",
            "negative_result_policy": "no unmatched correlation or pooled selector statistic",
            "replication_unit": "matched checkpoint stratum, never episode",
        },
        "panel": {
            "record_count": len(candidates), "raw_architectures": sorted({item.architecture for item in candidates}),
            "raw_environments": sorted({item.environment for item in candidates}),
            "raw_stages": sorted({item.stage for item in candidates}), "total_hard_strata": len(strata),
            "identifiable_strata": len(identifiable), "identifiable_architectures": matched_architectures,
            "structural_deficiencies": dict(sorted(deficiencies.items())),
        },
        "strata": strata,
        "heldout_incremental_risk": inferential if identifiable else {**inferential, "status": "not_estimated_without_valid_matched_strata"},
        "summary": {
            "status": status, "claim_ready": positive,
            "replication_gate": {"min_identifiable_strata": 4, "min_architectures": 2, "passed": sufficient_replication},
            "claim_boundary": "P1 is only a PointMaze fixed-action-bank checkpoint-selection pilot. It does not establish cross-environment transfer, a universal beyond-ATM diagnostic, a causal world model, or a controller improvement.",
            "limitations": [
                "One environment is represented, so cross-environment generalization is not estimable.",
                "The panel is intervention-supervised task-effect adaptation, not a raw frozen-world-model checkpoint panel.",
                "The fixed horizon bank is matched as one deployment contract; this audit cannot certify a separate h=1, h=2, or h=4 selector.",
                "Action-support identity is exact inside this panel, so support-shift robustness is intentionally outside the estimand.",
                "Spectrum, optimizer history, and representation gauge are not additionally matched; the conditional claim is limited to the declared MSE, ATM, architecture, stage, horizon, and support controls.",
            ],
        },
    }


def _record_paths(roots: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.rglob("checkpoint_record.json")))
        else:
            raise FileNotFoundError(root)
    return sorted(set(paths))


def run(record_roots: Iterable[Path], *, mse_tolerance: float, atm_tolerance: float, min_candidates: int, seed: int) -> dict[str, Any]:
    paths = _record_paths(record_roots)
    candidates, rejected = [], []
    for path in paths:
        try:
            candidates.append(load_candidate(path))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    report = audit_candidates(candidates, mse_tolerance=mse_tolerance, atm_tolerance=atm_tolerance, min_candidates=min_candidates, seed=seed) if candidates else {
        "artifact": "matched_decision_transfer_p1_audit", "protocol_version": PROTOCOL_VERSION,
        "summary": {"status": "structural_deficiency_no_loadable_records", "claim_ready": False},
    }
    report["input"] = {"record_paths": [str(path) for path in paths], "loaded_records": len(candidates), "rejected_records": rejected}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mse-tolerance", type=float, default=DEFAULT_MSE_TOLERANCE)
    parser.add_argument("--atm-tolerance", type=float, default=DEFAULT_ATM_TOLERANCE)
    parser.add_argument("--min-candidates", type=int, default=DEFAULT_MIN_CANDIDATES)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    print("Running strict matched learned-model P1 audit: MSE+ATM baseline versus MSE+ATM+pre-outcome P1.")
    report = run(args.record_root, mse_tolerance=args.mse_tolerance, atm_tolerance=args.atm_tolerance, min_candidates=args.min_candidates, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["summary"]["status"], "loaded_records": report["input"]["loaded_records"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
