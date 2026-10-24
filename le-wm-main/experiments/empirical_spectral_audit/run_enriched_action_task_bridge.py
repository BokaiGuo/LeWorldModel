"""Diagnose and repair the pooled-latent/action-metric interface.

The older learned-checkpoint export kept only one pooled latent vector per
episode.  This runner consumes the enriched records produced by
``run_jepa_wm_temporal_eval.py`` and performs two linked checks:

1. an action-sensitivity audit against zero and within-episode-deranged action
   controls; and
2. a preregistered fixed-spectrum probe in a low-rank source-latent chart,
   using train-only decoder-pullback and action-interaction metrics.

The external metric is fitted only from the diagnostic split.  Held-out
multi-step transfer is never used to choose a metric, weight, or snapshot.
This makes the script useful for finding whether the old negative result was a
measurement/interface failure, while keeping the result conditional rather
than claiming a universal certificate from a single pilot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gram = x.T @ x / max(len(x), 1)
    scale = max(float(np.trace(gram)) / max(x.shape[1], 1), 1e-12)
    return np.linalg.solve(gram + ridge * scale * np.eye(x.shape[1]), x.T @ y / max(len(x), 1))


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / max(denom, 1e-12))


def _task_weights(dim: int) -> np.ndarray:
    weights = np.asarray([0.25, 0.25, 1.0, 1.0, 4.0, 0.25, 0.25], dtype=np.float64)
    return weights if dim == len(weights) else np.ones(dim, dtype=np.float64)


def _weighted_mse(pred: np.ndarray, target: np.ndarray) -> float:
    weights = _task_weights(target.shape[1])
    return float(np.mean((pred - target) ** 2 * weights[None, :]))


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Tie-safe enough Spearman correlation for finite candidate audits."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    left_order = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(np.float64)
    right_order = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(np.float64)
    return float(np.corrcoef(left_order, right_order)[0, 1])


def _orthogonal(dim: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def _plane_rotation(dim: int, first: int, second: int, angle: float) -> np.ndarray:
    """Return a deterministic two-plane rotation in the declared chart."""
    q = np.eye(dim, dtype=np.float64)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    q[first, first] = c
    q[second, second] = c
    q[first, second] = -s
    q[second, first] = s
    return q


def _candidate_operators(anchor: np.ndarray, candidates: int, seed: int, family: str) -> list[np.ndarray]:
    """Create a spectrum-preserving candidate family.

    ``random`` retains the original broad null.  ``local`` is the matched-
    quality family: it enumerates small, deterministic rotations in every
    coordinate plane before falling back to the broad null.  This prevents a
    validation-MSE gate from degenerating to the anchor-only candidate.
    """
    if candidates < 1:
        raise ValueError("candidates must be positive")
    operators = [anchor]
    if family == "random":
        for j in range(1, candidates):
            q = _orthogonal(len(anchor), 10000 * seed + j)
            operators.append(q @ anchor @ q.T)
        return operators
    if family != "local":
        raise ValueError(f"unknown candidate family: {family}")
    planes = [(i, j) for i in range(len(anchor)) for j in range(i + 1, len(anchor))]
    angles = (-0.20, -0.14, -0.10, -0.07, -0.05, -0.03, -0.02, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.20)
    for index in range(candidates - 1):
        plane = planes[index % len(planes)]
        angle = angles[(index // len(planes)) % len(angles)]
        q = _plane_rotation(len(anchor), *plane, angle)
        operators.append(q @ anchor @ q.T)
    return operators


def _geometry(cov: np.ndarray, operator: np.ndarray, metric: np.ndarray) -> dict[str, float]:
    experiments_root = Path(__file__).resolve().parents[1]
    if str(experiments_root) not in sys.path:
        sys.path.insert(0, str(experiments_root))
    from task_metric_diagnostics import metric_geometry

    values = metric_geometry(cov, operator, metric, rank=min(4, len(operator)), horizons=range(1, 5))
    return {key: float(value) for key, value in values.items()}


def _joint_geometry(
    cov: np.ndarray,
    action_cov: np.ndarray,
    pair: tuple[np.ndarray, np.ndarray],
    metric: np.ndarray,
) -> dict[str, float]:
    """Evaluate geometry of the augmented action-conditioned operator.

    Rows use ``x_next = x @ A + u @ B``.  In column convention the same map is
    the square augmented operator ``[[A.T, B.T], [0, 0]]``.  The block-diagonal
    metric keeps the declared latent task metric and a fixed standardized
    action metric separate, so the diagnostic cannot silently discard ``B``.
    """
    state_operator, action_operator = pair
    state_dim, action_dim = state_operator.shape[0], action_operator.shape[0]
    augmented_cov = np.zeros((state_dim + action_dim, state_dim + action_dim), dtype=np.float64)
    augmented_cov[:state_dim, :state_dim] = cov
    augmented_cov[state_dim:, state_dim:] = action_cov
    augmented_metric = np.zeros_like(augmented_cov)
    augmented_metric[:state_dim, :state_dim] = metric
    augmented_metric[state_dim:, state_dim:] = np.eye(action_dim, dtype=np.float64)
    augmented_operator = np.zeros_like(augmented_cov)
    augmented_operator[:state_dim, :state_dim] = state_operator.T
    augmented_operator[:state_dim, state_dim:] = action_operator.T
    return _geometry(augmented_cov, augmented_operator, augmented_metric)


def _geometry_arrays(
    geometries: list[dict[str, float]],
    prefix: str,
) -> dict[str, np.ndarray]:
    """Expand the non-normal geometry dictionary into selector arrays."""
    keys = (
        "task_metric_alignment",
        "task_metric_commutator",
        "task_metric_schur_offdiagonal_energy",
        "task_metric_departure_from_normality",
        "task_metric_transient_amplification",
        "task_metric_spectral_radius",
    )
    return {
        f"{prefix}_{key.removeprefix('task_metric_')}": np.asarray([item[key] for item in geometries])
        for key in keys
    }


def _load(path: Path, horizon: int, downstream_horizon: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload.get("enriched_records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no enriched_records")
    out = {}
    for row in rows:
        episode = int(row["episode_id"])
        sample_id = str(row.get("sample_id", f"{episode}:{row.get('window_start', row.get('source_frame', 0))}"))
        latent_value = row.get("source_latent")
        latent = None if latent_value is None else np.asarray(latent_value, dtype=np.float64)
        state0 = np.asarray(row["source_state_raw"], dtype=np.float64)
        h = row["latent_by_horizon"]
        raw_blocks = np.asarray(row["action_blocks_raw"], dtype=np.float64)
        if raw_blocks.ndim != 2 or raw_blocks.shape[1] % 2 != 0:
            raise ValueError(f"{sample_id} has invalid flattened action blocks: {raw_blocks.shape}")
        action_sequence = raw_blocks.reshape(raw_blocks.shape[0], -1, 2).mean(axis=1)
        predictor_blocks_value = row.get("predictor_action_blocks_normalized")
        predictor_action_sequence = None
        if predictor_blocks_value is not None:
            predictor_blocks = np.asarray(predictor_blocks_value, dtype=np.float64)
            if predictor_blocks.shape != raw_blocks.shape:
                raise ValueError(
                    f"{sample_id} predictor action blocks do not match realized action-block shape: "
                    f"{predictor_blocks.shape} != {raw_blocks.shape}"
                )
            predictor_action_sequence = predictor_blocks.reshape(predictor_blocks.shape[0], -1, 2).mean(axis=1)
        out[sample_id] = {
            "sample_id": sample_id,
            "episode_id": episode,
            "latent": latent,
            "pred": np.asarray(h[str(horizon)]["predicted"], dtype=np.float64),
            "target": np.asarray(h[str(horizon)]["target"], dtype=np.float64),
            "pred_downstream": np.asarray(h[str(downstream_horizon)]["predicted"], dtype=np.float64),
            "target_downstream": np.asarray(h[str(downstream_horizon)]["target"], dtype=np.float64),
            "downstream": np.asarray(h[str(downstream_horizon)]["target"], dtype=np.float64),
            "delta_state": np.asarray(h[str(horizon)]["target_state_raw"], dtype=np.float64) - state0,
            "delta_state_downstream": np.asarray(h[str(downstream_horizon)]["target_state_raw"], dtype=np.float64) - state0,
            "proprio": np.asarray(row.get("source_proprio_raw", []), dtype=np.float64),
            "actions": action_sequence[0],
            "action_sequence": action_sequence,
            "predictor_action_sequence": predictor_action_sequence,
            "predictor_action_mode": row.get("predictor_action_mode"),
        }
    return {"payload": payload, "rows": out}


def _metric_matrices(
    x: np.ndarray,
    proprio: np.ndarray,
    action: np.ndarray,
    delta_state: np.ndarray,
    floor: float,
    ridge: float,
    latent_action_effect: np.ndarray | None = None,
    proprio_residual: bool = True,
) -> dict[str, np.ndarray]:
    """Fit external metrics after the source latent has been PCA projected."""
    weights = _task_weights(delta_state.shape[1])
    # Remove the directly observed proprioceptive component first.  The
    # residual is the external task effect that the visual latent is expected
    # to explain; fitting the metric to the raw delta would reward a missing
    # modality rather than diagnose the visual representation.
    proprio_fit = _ridge(proprio, delta_state, ridge)
    residual = delta_state - proprio @ proprio_fit if proprio_residual else delta_state
    task_decoder = _ridge(x, residual, ridge)
    task = task_decoder @ np.diag(weights) @ task_decoder.T

    # The interaction term isolates state-dependent action effects.  The
    # action-only main effect is deliberately retained in the regression but
    # is not allowed to define a latent metric by itself.
    action_z = (action - action.mean(0, keepdims=True)) / np.maximum(action.std(0, keepdims=True), 1e-6)
    interaction = (x[:, :, None] * action_z[:, None, :]).reshape(len(x), -1)
    design = np.concatenate([x, action_z, interaction], axis=1)
    fit = _ridge(design, residual, ridge)
    interaction_fit = fit[x.shape[1] + action.shape[1] :].reshape(x.shape[1], action.shape[1], -1)
    action_metric = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for component in range(action.shape[1]):
        jacobian = interaction_fit[:, component, :]
        action_metric += float(np.var(action_z[:, component])) * jacobian @ np.diag(weights) @ jacobian.T

    def stabilize(matrix: np.ndarray) -> np.ndarray:
        matrix = 0.5 * (matrix + matrix.T)
        scale = max(float(np.trace(matrix)) / max(len(matrix), 1), 1e-12)
        return matrix + float(floor) * scale * np.eye(len(matrix))

    metrics = {"task_pullback": stabilize(task), "action_interaction_pullback": stabilize(action_metric)}
    if latent_action_effect is not None:
        if latent_action_effect.shape != x.shape:
            raise ValueError("latent_action_effect must have the same shape as projected source latent")
        effect_decoder = _ridge(latent_action_effect, residual, ridge)
        metrics["observed_zero_action_effect_pullback"] = stabilize(
            effect_decoder @ np.diag(weights) @ effect_decoder.T
        )
    return metrics


def _attach_observed_zero_effect(
    observed_rows: dict[str, dict[str, np.ndarray]],
    zero_rows: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Attach a matched observed-minus-zero predicted latent effect.

    The zero-action control is a paired intervention, not a separate dataset:
    the source observation, target state, and episode must be identical for
    every sample.  Some intervention exports serialize an unpooled source
    token grid, so the observed source vector is made canonical and the
    control's source serialization is never used.  The returned effect is only
    later consumed on the training episodes by ``_one_split``; attaching it to
    all rows here keeps the split logic explicit and prevents test outcomes
    from entering metric fitting.
    """
    observed_ids = set(observed_rows)
    zero_ids = set(zero_rows)
    missing = sorted(observed_ids - zero_ids)
    extra = sorted(zero_ids - observed_ids)
    if missing or extra:
        raise ValueError(
            "observed/zero controls must have identical sample_id sets; "
            f"missing_zero={missing[:5]}, extra_zero={extra[:5]}"
        )
    max_source_latent_gap = 0.0
    source_latent_shape_mismatches = 0
    max_target_gap = 0.0
    max_downstream_target_gap = 0.0
    for sample_id in sorted(observed_ids):
        observed = observed_rows[sample_id]
        zero = zero_rows[sample_id]
        if int(observed["episode_id"]) != int(zero["episode_id"]):
            raise ValueError(f"episode mismatch for matched control {sample_id}")
        # The intervention export may have serialized an unpooled token grid
        # for the source context while the observed export stores its pooled
        # source vector.  This is an export-format mismatch, not an action
        # effect.  The observed source vector is the canonical paired input;
        # never use the control's source latent in the metric fit.
        observed_source = np.asarray(observed["latent"])
        zero_source = np.asarray(zero["latent"])
        if observed_source.ndim != 1:
            raise ValueError(f"observed source latent must be a vector for matched control {sample_id}")
        if observed_source.shape != zero_source.shape:
            source_latent_shape_mismatches += 1
            feature_axes = [axis for axis, width in enumerate(zero_source.shape) if width == len(observed_source)]
            if len(feature_axes) != 1:
                raise ValueError(
                    f"cannot canonicalize control source latent for {sample_id}: "
                    f"observed={observed_source.shape}, control={zero_source.shape}"
                )
            zero_source = np.moveaxis(zero_source, feature_axes[0], -1).reshape(-1, len(observed_source)).mean(axis=0)
        source_gap = float(np.max(np.abs(observed_source - zero_source)))
        max_source_latent_gap = max(max_source_latent_gap, source_gap)
        if source_gap > 2e-6:
            raise ValueError(f"source latent mismatch for matched control {sample_id}: {source_gap}")
        observed_predictor_actions = observed.get("predictor_action_sequence")
        zero_predictor_actions = zero.get("predictor_action_sequence")
        if observed_predictor_actions is None or zero_predictor_actions is None:
            raise ValueError(
                "matched zero-action control lacks predictor_action_blocks_normalized; "
                "rerun the evaluator rather than inferring its intervention from realised actions"
            )
        if observed_predictor_actions.shape != zero_predictor_actions.shape:
            raise ValueError(f"predictor action shape mismatch for matched control {sample_id}")
        action_gap = float(np.max(np.abs(observed["action_sequence"] - zero["action_sequence"])))
        if action_gap > 1e-8:
            raise ValueError(f"realized action mismatch for matched control {sample_id}: {action_gap}")
        zero_predictor_max_abs = float(np.max(np.abs(zero_predictor_actions)))
        if zero_predictor_max_abs > 1e-8:
            raise ValueError(f"zero control fed non-zero predictor actions for {sample_id}: {zero_predictor_max_abs}")
        for field, label in (
            ("target", "target latent"),
            ("target_downstream", "downstream target latent"),
        ):
            left = observed[field]
            right = zero[field]
            if left is None or right is None:
                raise ValueError(f"{label} is missing for matched control {sample_id}")
            gap = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            if field == "target":
                max_target_gap = max(max_target_gap, gap)
            else:
                max_downstream_target_gap = max(max_downstream_target_gap, gap)
            if gap > 1e-8:
                raise ValueError(f"{label} mismatch for matched control {sample_id}: {gap}")
        # Canonicalize the control row to the observed source representation;
        # only its intervention prediction is used below.
        zero["latent"] = observed["latent"]
        observed["latent_action_effect"] = np.asarray(observed["pred"] - zero["pred"], dtype=np.float64)
        observed["latent_action_effect_downstream"] = np.asarray(
            observed["pred_downstream"] - zero["pred_downstream"], dtype=np.float64
        )
    return {
        "matched_samples": len(observed_ids),
        "max_source_latent_gap": max_source_latent_gap,
        "source_latent_shape_mismatches": source_latent_shape_mismatches,
        "max_target_latent_gap": max_target_gap,
        "max_downstream_target_gap": max_downstream_target_gap,
        "control_predictor_action_max_abs": 0.0,
        "definition": "observed predicted latent minus matched zero-action predicted latent",
    }


def _one_split(
    rows: dict[str, dict[str, np.ndarray]],
    split_seed: int,
    rank: int,
    candidates: int,
    floor: float,
    ridge: float,
    candidate_family: str,
    match_tolerance: float,
    metric_horizon: str,
    metric_target: str,
) -> dict[str, Any]:
    ids = np.asarray(sorted(rows), dtype=object)
    episode_ids = np.asarray(sorted({int(rows[str(i)]["episode_id"]) for i in ids}), dtype=int)
    order = np.random.default_rng(split_seed).permutation(episode_ids)
    train_end = max(6, int(round(0.6 * len(episode_ids))))
    valid_end = max(train_end + 2, int(round(0.8 * len(episode_ids))))
    train_episodes, valid_episodes, test_episodes = order[:train_end], order[train_end:valid_end], order[valid_end:]
    row_episodes = np.asarray([int(rows[str(i)]["episode_id"]) for i in ids], dtype=int)
    train = np.flatnonzero(np.isin(row_episodes, train_episodes))
    valid = np.flatnonzero(np.isin(row_episodes, valid_episodes))
    test = np.flatnonzero(np.isin(row_episodes, test_episodes))
    if any(rows[str(i)]["latent"] is None for i in ids):
        raise ValueError("observed bridge records must contain source_latent")
    source = np.stack([rows[str(i)]["latent"] for i in ids])
    mean = source[train].mean(0)
    _, _, vh = np.linalg.svd(source[train] - mean, full_matrices=False)
    basis = vh[: min(rank, len(vh), len(train) - 1)]
    if len(basis) < 3:
        raise ValueError("insufficient source-latent rank for bridge")
    project = lambda values: (np.asarray(values) - mean) @ basis.T
    x_all = project(source)
    pred_all = project(np.stack([rows[str(i)]["pred"] for i in ids]))
    target_all = project(np.stack([rows[str(i)]["target"] for i in ids]))
    downstream_all = project(np.stack([rows[str(i)]["downstream"] for i in ids]))
    action_all = np.stack([rows[str(i)]["actions"] for i in ids])
    action_sequence_all = np.stack([rows[str(i)]["action_sequence"] for i in ids])
    delta_one_all = np.stack([rows[str(i)]["delta_state"] for i in ids])
    delta_downstream_all = np.stack([rows[str(i)]["delta_state_downstream"] for i in ids])
    proprio_all = np.stack([rows[str(i)]["proprio"] for i in ids])
    if proprio_all.shape[1] == 0:
        raise ValueError("observed bridge records must contain source_proprio_raw")
    x_train, x_valid, x_test = x_all[train], x_all[valid], x_all[test]
    pred_train = pred_all[train]
    target_valid, target_test = target_all[valid], target_all[test]
    downstream_test = downstream_all[test]
    effect_one_all = np.stack([rows[str(i)]["latent_action_effect"] for i in ids]) if "latent_action_effect" in rows[str(ids[0])] else None
    effect_downstream_all = np.stack([rows[str(i)]["latent_action_effect_downstream"] for i in ids]) if "latent_action_effect_downstream" in rows[str(ids[0])] else None
    if metric_horizon == "downstream":
        metric_x = x_all[train]
        metric_proprio = proprio_all[train]
        metric_action = action_all[train]
        delta_all = delta_downstream_all
        effect_all = effect_downstream_all
    elif metric_horizon == "integrated":
        metric_x = np.concatenate([x_all[train], x_all[train]], axis=0)
        metric_proprio = np.concatenate([proprio_all[train], proprio_all[train]], axis=0)
        metric_action = np.concatenate([action_all[train], action_all[train]], axis=0)
        delta_all = np.concatenate([delta_one_all[train], delta_downstream_all[train]], axis=0)
        effect_all = None if effect_one_all is None or effect_downstream_all is None else np.concatenate([effect_one_all[train], effect_downstream_all[train]], axis=0)
    else:
        metric_x = x_all[train]
        metric_proprio = proprio_all[train]
        metric_action = action_all[train]
        delta_all = delta_one_all
        effect_all = effect_one_all
    effect_projected = None if effect_all is None else effect_all @ basis.T
    metric_train = _metric_matrices(
        metric_x,
        metric_proprio,
        metric_action,
        delta_all if metric_horizon == "integrated" else delta_all[train],
        floor,
        ridge,
        None if effect_projected is None else (
            effect_projected if metric_horizon == "integrated" else effect_projected[train]
        ),
        proprio_residual=metric_target == "proprio_residual",
    )
    anchor = _ridge(x_train, pred_train, ridge)
    candidates_ops = _candidate_operators(anchor, candidates, split_seed, candidate_family)
    cov = x_train.T @ x_train / max(len(x_train) - 1, 1)
    validation = np.asarray([np.mean((x_all[valid] @ op - target_valid) ** 2) for op in candidates_ops])
    test_one = np.asarray([np.mean((x_test @ op - target_test) ** 2) for op in candidates_ops])
    test_multi = np.asarray([np.mean((x_test @ np.linalg.matrix_power(op, 4) - downstream_test) ** 2) for op in candidates_ops])
    train_multi = np.asarray([
        np.mean((x_train @ np.linalg.matrix_power(op, 4) - downstream_all[train]) ** 2)
        for op in candidates_ops
    ])
    physical_decoder = _ridge(x_train, delta_downstream_all[train], ridge)
    physical_target_test = delta_downstream_all[test]
    test_multi_physical = np.asarray([
        _weighted_mse(x_test @ np.linalg.matrix_power(op, 4) @ physical_decoder, physical_target_test)
        for op in candidates_ops
    ])
    train_multi_physical = np.asarray([
        _weighted_mse(x_train @ np.linalg.matrix_power(op, 4) @ physical_decoder, delta_downstream_all[train])
        for op in candidates_ops
    ])
    identity = np.eye(len(basis))
    geometry: dict[str, np.ndarray] = {}
    for name, metric in {**metric_train, "euclidean": identity}.items():
        geometry.update(_geometry_arrays([_geometry(cov, op, metric) for op in candidates_ops], name))
    floor_valid = validation.min() * match_tolerance
    feasible = np.flatnonzero(validation <= floor_valid)
    rng = np.random.default_rng(80000 + split_seed)
    oracle = int(np.argmin(test_multi))
    physical_oracle = int(np.argmin(test_multi_physical))
    def physical_readout_r2(target: np.ndarray) -> dict[str, float]:
        """Evaluate the declared physical target at the same horizon as selection.

        The previous summary always reported the one-step readout even when
        selection used an h=4 or integrated task metric.  That silently turned
        a finite-horizon diagnostic into a one-step proxy.
        """
        return {
            "latent_only": _r2(target[test], x_test @ _ridge(x_train, target[train], ridge)),
            "proprio_only": _r2(
                target[test], proprio_all[test] @ _ridge(proprio_all[train], target[train], ridge)
            ),
            "joint": _r2(
                target[test],
                np.concatenate([x_test, proprio_all[test]], axis=1) @ _ridge(
                    np.concatenate([x_train, proprio_all[train]], axis=1), target[train], ridge
                ),
            ),
        }

    result: dict[str, Any] = {
        "split_seed": split_seed,
        "n_train": len(train), "n_validation": len(valid), "n_test": len(test),
        "rank": len(basis), "candidate_count": len(candidates_ops),
        "eigenvalue_match_max_abs": float(max(np.max(np.abs(np.sort_complex(np.linalg.eigvals(anchor)) - np.sort_complex(np.linalg.eigvals(op)))) for op in candidates_ops)),
        "metric_condition_numbers": {name: float(np.linalg.cond(metric)) for name, metric in metric_train.items()},
        "heldout_physical_readout_r2": {
            "one_step": physical_readout_r2(delta_one_all),
            "downstream": physical_readout_r2(delta_downstream_all),
        },
        "metric_horizon": metric_horizon,
        "metric_target": metric_target,
        "oracle": oracle,
        "physical_oracle": physical_oracle,
        "selectors": {},
        "candidate_transfer": test_multi.tolist(),
        "candidate_physical_transfer": test_multi_physical.tolist(),
        "candidate_validation_mse": validation.tolist(),
        "validation_feasible_count": int(len(feasible)),
        "candidate_geometry": {name: values.tolist() for name, values in geometry.items()},
        "matched_rank_correlation": {
            name: _rank_correlation(values[feasible], test_multi[feasible])
            for name, values in geometry.items()
        },
        "matched_physical_rank_correlation": {
            name: _rank_correlation(values[feasible], test_multi_physical[feasible])
            for name, values in geometry.items()
        },
    }
    selectors = {
        "validation_mse": int(np.argmin(validation)),
        "random_equal_spectrum": int(rng.integers(0, candidates)),
        "train_rollout_mse": int(np.argmin(train_multi)),
        "train_physical_rollout_mse": int(np.argmin(train_multi_physical)),
    }
    for name, values in geometry.items():
        selectors[name] = int(np.argmin(values))
        selectors[f"matched_{name}"] = int(feasible[np.argmin(values[feasible])])
    for name, selected in selectors.items():
        result["selectors"][name] = {
            "candidate": selected,
            "test_one_step_mse": float(test_one[selected]),
            "test_transfer_mse": float(test_multi[selected]),
            "regret": float(test_multi[selected] - test_multi[oracle]),
            "is_oracle": bool(selected == oracle),
            "physical_transfer_mse": float(test_multi_physical[selected]),
            "physical_regret": float(test_multi_physical[selected] - test_multi_physical[physical_oracle]),
            "is_physical_oracle": bool(selected == physical_oracle),
        }
    # Action-conditioned repair: the old static operator marginalized the
    # action input, so it could not represent the very action effect whose
    # metric was being used for selection.  Fit a joint linear interface
    # y_{t+1}=A_x x_t + B_u u_t, preserve the spectrum of A_x, and transform
    # B_u covariantly with the same latent rotation.
    action_train = action_sequence_all[train, 0]
    action_valid = action_sequence_all[valid]
    action_test = action_sequence_all[test]
    # Keep the visual anchor fixed.  Estimating A_x and B_u jointly is
    # ill-conditioned when the action signal is weak (as in PointMaze) and
    # creates a spurious unstable A_x whose powers dominate the h=4 outcome.
    # Fit only the action residual, so the repair cannot trade away the
    # already-identified visual predictor for a noisy action coefficient.
    anchor_x = anchor
    anchor_u = _ridge(action_train, pred_train - x_train @ anchor_x, ridge)
    # Build joint candidates explicitly so B_u receives the same latent
    # coordinate rotation Q as A_x.  The local family is deterministic; for
    # the broad null we use the historical seeded orthogonal draws.
    joint_ops = [(anchor_x, anchor_u)]
    if candidate_family == "local":
        planes = [(i, j) for i in range(len(basis)) for j in range(i + 1, len(basis))]
        angles = (-0.20, -0.14, -0.10, -0.07, -0.05, -0.03, -0.02, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.20)
        for index in range(candidates - 1):
            plane = planes[index % len(planes)]
            angle = angles[(index // len(planes)) % len(angles)]
            q = _plane_rotation(len(basis), *plane, angle)
            joint_ops.append((q @ anchor_x @ q.T, anchor_u @ q.T))
    else:
        for j in range(1, candidates):
            q = _orthogonal(len(basis), 20000 * split_seed + j)
            joint_ops.append((q @ anchor_x @ q.T, anchor_u @ q.T))

    def one_step_joint(x_values: np.ndarray, action_values: np.ndarray, pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return x_values @ pair[0] + action_values @ pair[1]

    def rollout_joint(x_values: np.ndarray, action_values: np.ndarray, pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        state = x_values.copy()
        for step in range(min(4, action_values.shape[1])):
            state = state @ pair[0] + action_values[:, step] @ pair[1]
        return state

    joint_validation = np.asarray([np.mean((one_step_joint(x_all[valid], action_valid[:, 0], pair) - target_valid) ** 2) for pair in joint_ops])
    joint_test_one = np.asarray([np.mean((one_step_joint(x_test, action_test[:, 0], pair) - target_test) ** 2) for pair in joint_ops])
    joint_test_multi = np.asarray([np.mean((rollout_joint(x_test, action_test, pair) - downstream_test) ** 2) for pair in joint_ops])
    joint_train_multi = np.asarray([
        np.mean((rollout_joint(x_train, action_sequence_all[train], pair) - downstream_all[train]) ** 2)
        for pair in joint_ops
    ])
    joint_test_multi_physical = np.asarray([
        _weighted_mse(rollout_joint(x_test, action_test, pair) @ physical_decoder, physical_target_test)
        for pair in joint_ops
    ])
    joint_train_multi_physical = np.asarray([
        _weighted_mse(rollout_joint(x_train, action_sequence_all[train], pair) @ physical_decoder, delta_downstream_all[train])
        for pair in joint_ops
    ])
    joint_geometry: dict[str, np.ndarray] = {}
    for name, metric in {**metric_train, "euclidean": identity}.items():
        joint_geometry.update(_geometry_arrays([_geometry(cov, pair[0], metric) for pair in joint_ops], name))
    action_centered = action_train - action_train.mean(axis=0, keepdims=True)
    action_cov = action_centered.T @ action_centered / max(len(action_centered) - 1, 1)
    for name, metric in {**metric_train, "euclidean": identity}.items():
        joint_geometry.update(_geometry_arrays(
            [_joint_geometry(cov, action_cov, pair, metric) for pair in joint_ops],
            f"joint_{name}",
        ))
    joint_oracle = int(np.argmin(joint_test_multi))
    joint_physical_oracle = int(np.argmin(joint_test_multi_physical))
    joint_feasible = np.flatnonzero(joint_validation <= joint_validation.min() * match_tolerance)
    joint_selectors = {
        "action_conditioned_validation_mse": int(np.argmin(joint_validation)),
        "action_conditioned_random_equal_spectrum": int(rng.integers(0, candidates)),
        "action_conditioned_train_rollout_mse": int(np.argmin(joint_train_multi)),
        "action_conditioned_train_physical_rollout_mse": int(np.argmin(joint_train_multi_physical)),
    }
    for name, values in joint_geometry.items():
        joint_selectors[f"action_conditioned_{name}"] = int(np.argmin(values))
        joint_selectors[f"action_conditioned_matched_{name}"] = int(joint_feasible[np.argmin(values[joint_feasible])])
    result["action_conditioned"] = {
        "oracle": joint_oracle,
        "physical_oracle": joint_physical_oracle,
        "feasible_count": int(len(joint_feasible)),
        "candidate_validation_mse": joint_validation.tolist(),
        "candidate_geometry": {name: values.tolist() for name, values in joint_geometry.items()},
        "matched_rank_correlation": {
            name: _rank_correlation(values[joint_feasible], joint_test_multi[joint_feasible])
            for name, values in joint_geometry.items()
        },
        "matched_physical_rank_correlation": {
            name: _rank_correlation(values[joint_feasible], joint_test_multi_physical[joint_feasible])
            for name, values in joint_geometry.items()
        },
        "selectors": {
            name: {
                "candidate": selected,
                "test_one_step_mse": float(joint_test_one[selected]),
                "test_transfer_mse": float(joint_test_multi[selected]),
                "regret": float(joint_test_multi[selected] - joint_test_multi[joint_oracle]),
                "is_oracle": bool(selected == joint_oracle),
                "physical_transfer_mse": float(joint_test_multi_physical[selected]),
                "physical_regret": float(joint_test_multi_physical[selected] - joint_test_multi_physical[joint_physical_oracle]),
                "is_physical_oracle": bool(selected == joint_physical_oracle),
            }
            for name, selected in joint_selectors.items()
        },
        "eigenvalue_match_max_abs": float(max(np.max(np.abs(np.sort_complex(np.linalg.eigvals(anchor_x)) - np.sort_complex(np.linalg.eigvals(pair[0])))) for pair in joint_ops)),
    }
    result["geometry_ranges"] = {name: [float(values.min()), float(values.max())] for name, values in geometry.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--zero", type=Path, default=None)
    parser.add_argument("--shuffle", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--downstream-horizon", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=17)
    parser.add_argument("--candidate-family", choices=["random", "local"], default="random")
    parser.add_argument("--match-tolerance", type=float, default=1.05)
    parser.add_argument("--metric-horizon", choices=["one_step", "downstream", "integrated"], default="one_step")
    parser.add_argument("--metric-target", choices=["proprio_residual", "full_delta"], default="proprio_residual")
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=0.05)
    args = parser.parse_args()
    observed = _load(args.observed, args.horizon, args.downstream_horizon)
    rows = observed["rows"]
    summary: dict[str, Any] = {"observed": {}}
    effect_audit = None
    for label, path in (("zero", args.zero), ("shuffle", args.shuffle)):
        if path is None:
            continue
        control = _load(path, args.horizon, args.downstream_horizon)["rows"]
        common = sorted(set(rows) & set(control))
        if not common:
            raise ValueError(f"no matched episodes between observed and {label}")
        observed_pred = np.stack([rows[i]["pred"] for i in common])
        control_pred = np.stack([control[i]["pred"] for i in common])
        target = np.stack([rows[i]["target"] for i in common])
        summary[label] = {
            "matched_episodes": len(common),
            "latent_prediction_mse": float(np.mean((control_pred - target) ** 2)),
            "latent_action_gap_rmse": float(np.sqrt(np.mean((observed_pred - control_pred) ** 2))),
            "observed_prediction_mse": float(np.mean((observed_pred - target) ** 2)),
        }
        if label == "zero":
            effect_audit = _attach_observed_zero_effect(rows, control)
    if effect_audit is None:
        raise ValueError("the enriched bridge requires a matched zero-action control")
    summary["observed_zero_action_effect"] = effect_audit
    splits = [_one_split(
        rows, seed, args.rank, args.candidates, args.floor, args.ridge,
        args.candidate_family, args.match_tolerance, args.metric_horizon, args.metric_target,
    ) for seed in range(5)]
    physical_readout_horizon = "one_step" if args.metric_horizon == "one_step" else "downstream"
    summary["fixed_spectrum_bridge"] = {
        "split_count": len(splits),
        "mean_regret": {name: float(np.mean([s["selectors"][name]["regret"] for s in splits])) for name in splits[0]["selectors"]},
        "mean_physical_regret": {name: float(np.mean([s["selectors"][name]["physical_regret"] for s in splits])) for name in splits[0]["selectors"]},
        "oracle_hit_rate": {name: float(np.mean([s["selectors"][name]["is_oracle"] for s in splits])) for name in splits[0]["selectors"]},
        "physical_oracle_hit_rate": {name: float(np.mean([s["selectors"][name]["is_physical_oracle"] for s in splits])) for name in splits[0]["selectors"]},
        "mean_physical_readout_r2": {
            name: float(np.mean([
                s["heldout_physical_readout_r2"][physical_readout_horizon][name] for s in splits
            ]))
            for name in ("latent_only", "proprio_only", "joint")
        },
        "physical_readout_horizon": physical_readout_horizon,
        "action_conditioned_mean_regret": {
            name: float(np.mean([s["action_conditioned"]["selectors"][name]["regret"] for s in splits]))
            for name in splits[0]["action_conditioned"]["selectors"]
        },
        "action_conditioned_mean_physical_regret": {
            name: float(np.mean([s["action_conditioned"]["selectors"][name]["physical_regret"] for s in splits]))
            for name in splits[0]["action_conditioned"]["selectors"]
        },
        "action_conditioned_oracle_hit_rate": {
            name: float(np.mean([s["action_conditioned"]["selectors"][name]["is_oracle"] for s in splits]))
            for name in splits[0]["action_conditioned"]["selectors"]
        },
        "action_conditioned_physical_oracle_hit_rate": {
            name: float(np.mean([s["action_conditioned"]["selectors"][name]["is_physical_oracle"] for s in splits]))
            for name in splits[0]["action_conditioned"]["selectors"]
        },
    }
    report = {
        "protocol": {
            "observed": str(args.observed), "zero_control": str(args.zero) if args.zero else None,
            "shuffle_control": str(args.shuffle) if args.shuffle else None,
            "metric_fit": "train-only external raw-state pullback and state-action interaction pullback",
            "control_contract": "matched zero-control exports must record the actual predictor input blocks",
            "selection": "fixed-spectrum orthogonal conjugates; validation MSE matching then metric alignment",
            "candidate_family": args.candidate_family,
            "match_tolerance": args.match_tolerance,
            "metric_horizon": args.metric_horizon,
            "metric_target": args.metric_target,
            "outcome": "held-out one-step and four-step latent transfer MSE",
            "negative_result_policy": "old pooled-latent results are not overwritten; this is a new enriched interface audit",
        },
        "summary": summary,
        "splits": splits,
        "claim_boundary": "Pilot bridge evidence only; cross-environment and independently trained-checkpoint generality remain unproven until the same enriched protocol is run across environments and architectures.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "enriched_action_task_bridge.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Enriched action/task bridge audit", "", "## Action-sensitivity controls", "", "| condition | matched episodes | prediction MSE | observed-vs-control latent gap |", "|---|---:|---:|---:|"]
    for label, item in summary.items():
        if label in {"observed", "fixed_spectrum_bridge", "observed_zero_action_effect"}:
            continue
        lines.append(f"| {label} | {item['matched_episodes']} | {item['latent_prediction_mse']:.6g} | {item['latent_action_gap_rmse']:.6g} |")
    lines += ["", "## Fixed-spectrum bridge", "", "| selector | mean regret | oracle hit rate |", "|---|---:|---:|"]
    for name, value in summary["fixed_spectrum_bridge"]["mean_regret"].items():
        lines.append(f"| {name} | {value:.6g} | {summary['fixed_spectrum_bridge']['oracle_hit_rate'][name]:.4g} |")
    lines += ["", "## Action-conditioned repair", "", "| selector | mean regret | oracle hit rate |", "|---|---:|---:|"]
    for name, value in summary["fixed_spectrum_bridge"]["action_conditioned_mean_regret"].items():
        lines.append(f"| {name} | {value:.6g} | {summary['fixed_spectrum_bridge']['action_conditioned_oracle_hit_rate'][name]:.4g} |")
    readout = summary["fixed_spectrum_bridge"]["mean_physical_readout_r2"]
    lines += ["", f"Held-out {summary['fixed_spectrum_bridge']['physical_readout_horizon']} physical readout R2 (latent/proprio/joint): {readout['latent_only']:.6g} / {readout['proprio_only']:.6g} / {readout['joint']:.6g}", "", report["claim_boundary"], ""]
    (args.output / "enriched_action_task_bridge.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
