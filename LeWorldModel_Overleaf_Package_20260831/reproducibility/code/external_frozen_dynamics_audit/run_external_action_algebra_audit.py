"""Cross-fitted action algebra for frozen external LeWM trajectories.

This audit asks a stronger question than whether latent increments project onto
one transferred subspace: can a low-dimensional, action-conditioned transition
map explain and compose the observed latent trajectory?  For each held-out
episode we fit, in a source-episode PCA coordinate system,

    T_u(z) = A z + B u + b,

and compare it with a state-only control.  We report action sensitivity under
zero and permuted actions, direct-vs-composed two/four/eight-step maps, and a
negated-action return diagnostic.  All fits are leave-one-episode-out.  The
result is a frozen latent-operator diagnostic: the affine action parameter is
not a causal environment model, and negating a relative action is only an
algebraic inverse proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    _bootstrap_clustered,
    _load_trajectories,
    _pca_fit,
    _pca_transform,
)


HORIZONS = (2, 4, 8)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _design(states: np.ndarray, actions: np.ndarray | None = None) -> np.ndarray:
    """Build a centered-regression design without an explicit intercept."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if actions is None:
        return states
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] != states.shape[0]:
        raise ValueError("actions must align with states")
    return np.concatenate([states, actions], axis=1)


def _fit_map(features: np.ndarray, target: np.ndarray, ridge: float = 1e-2) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if features.ndim != 2 or target.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError("features and target must be aligned matrices")
    if features.shape[0] <= features.shape[1]:
        raise ValueError("regression is underdetermined")
    x_mean = features.mean(axis=0)
    y_mean = target.mean(axis=0)
    x = features - x_mean
    y = target - y_mean
    gram = x.T @ x
    scale = max(float(np.trace(gram)) / max(features.shape[1], 1), 1e-8)
    coef = np.linalg.solve(gram + float(ridge) * scale * np.eye(features.shape[1]), x.T @ y)
    return {"x_mean": x_mean, "y_mean": y_mean, "coef": coef}


def _apply_map(fit: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - fit["x_mean"]) @ fit["coef"] + fit["y_mean"]


def _mse(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.square(np.asarray(predicted) - np.asarray(target)).mean())


def _r2(predicted: np.ndarray, target: np.ndarray) -> float:
    numerator = np.square(np.asarray(predicted) - np.asarray(target)).sum()
    denominator = max(float(np.square(np.asarray(target) - np.asarray(target).mean(axis=0)).sum()), 1e-12)
    return float(1.0 - numerator / denominator)


def _relative_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(predicted) - np.asarray(target)) / max(np.linalg.norm(target), 1e-12))


def _fit_horizon_map(
    states: np.ndarray, actions: np.ndarray, horizon: int, ridge: float
) -> dict[str, np.ndarray]:
    source = states[:, :-horizon].reshape(-1, states.shape[-1])
    # Attach each action sequence to the corresponding state.  Both source
    # and target are flattened in (window, start) order, so the action tensor
    # must be flattened in the same order.  Concatenating one start at a time
    # would silently pair an episode's state with another episode's actions
    # whenever there is more than one start per window.
    action_seq = np.stack(
        [
            actions[:, start : start + horizon, :].reshape(-1, horizon * actions.shape[-1])
            for start in range(states.shape[1] - horizon)
        ],
        axis=1,
    ).reshape(-1, horizon * actions.shape[-1])
    target = states[:, horizon:].reshape(-1, states.shape[-1])
    return _fit_map(_design(source, action_seq), target, ridge=ridge)


def _apply_composed_map(
    fit: dict[str, np.ndarray], state: np.ndarray, action_sequence: np.ndarray, state_dim: int
) -> np.ndarray:
    current = np.asarray(state, dtype=np.float64)
    for action in np.asarray(action_sequence, dtype=np.float64):
        current = _apply_map(fit, _design(current.reshape(1, state_dim), action.reshape(1, -1)))[0]
    return current


def _cluster_metric(values: np.ndarray, groups: np.ndarray, seed: int, repeats: int) -> dict[str, Any]:
    return _bootstrap_clustered(np.asarray(values, dtype=np.float64), np.asarray(groups), repeats, seed)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(x), _rank(y))


def _evaluate_source(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    source_name: str,
    pca_dimension: int,
    ridge: float,
) -> dict[str, Any]:
    train_states_raw = np.stack([row["true"] for row in train_rows] + [row["predicted"] for row in train_rows])
    pca = _pca_fit(train_states_raw, pca_dimension)
    train_states = np.stack([_pca_transform(row[source_name], pca) for row in train_rows])
    test_states = np.stack([_pca_transform(row[source_name], pca) for row in test_rows])
    train_actions = np.stack([row["action_features"] for row in train_rows])
    test_actions = np.stack([row["action_features"] for row in test_rows])
    dimension = train_states.shape[-1]

    x1 = train_states[:, :-1].reshape(-1, dimension)
    u1 = train_actions.reshape(-1, train_actions.shape[-1])
    y1 = train_states[:, 1:].reshape(-1, dimension)
    state_fit = _fit_map(_design(x1), y1, ridge=ridge)
    action_fit = _fit_map(_design(x1, u1), y1, ridge=ridge)

    test_x1 = test_states[:, :-1].reshape(-1, dimension)
    test_u1 = test_actions.reshape(-1, test_actions.shape[-1])
    test_y1 = test_states[:, 1:].reshape(-1, dimension)
    state_pred = _apply_map(state_fit, _design(test_x1))
    action_pred = _apply_map(action_fit, _design(test_x1, test_u1))
    zero_pred = _apply_map(action_fit, _design(test_x1, np.zeros_like(test_u1)))
    permuted_u = np.roll(test_u1, 1, axis=0)
    permuted_pred = _apply_map(action_fit, _design(test_x1, permuted_u))

    records: dict[str, Any] = {
        "pca_dimension": int(dimension),
        "one_step": {
            "state_only_mse": _mse(state_pred, test_y1),
            "action_conditioned_mse": _mse(action_pred, test_y1),
            "state_only_r2": _r2(state_pred, test_y1),
            "action_conditioned_r2": _r2(action_pred, test_y1),
            "action_gain_over_state_mse": _mse(state_pred, test_y1) - _mse(action_pred, test_y1),
            "zero_action_mse": _mse(zero_pred, test_y1),
            "permuted_action_mse": _mse(permuted_pred, test_y1),
            "zero_minus_observed_mse": _mse(zero_pred, test_y1) - _mse(action_pred, test_y1),
            "permuted_minus_observed_mse": _mse(permuted_pred, test_y1) - _mse(action_pred, test_y1),
        },
        "composition": {},
        "negated_action_return": {},
    }

    for horizon in HORIZONS:
        direct_fit = _fit_horizon_map(train_states, train_actions, horizon, ridge=ridge)
        source = test_states[:, :-horizon].reshape(-1, dimension)
        sequences = np.stack(
            [test_actions[:, start : start + horizon, :].reshape(-1, horizon * test_actions.shape[-1])
             for start in range(test_states.shape[1] - horizon)], axis=1
        )
        # sequences has shape (windows, starts, flattened_actions); select the
        # corresponding start for the flattened source rows.
        sequences = sequences.reshape(-1, horizon, test_actions.shape[-1])
        target = test_states[:, horizon:].reshape(-1, dimension)
        direct_pred = _apply_map(direct_fit, _design(source, sequences.reshape(-1, horizon * test_actions.shape[-1])))
        composed = np.stack(
            [_apply_composed_map(action_fit, source[i], sequences[i], dimension) for i in range(source.shape[0])]
        )
        records["composition"][f"h{horizon}"] = {
            "direct_mse": _mse(direct_pred, target),
            "composed_mse": _mse(composed, target),
            "direct_r2": _r2(direct_pred, target),
            "composed_r2": _r2(composed, target),
            "composition_defect": _relative_error(composed, direct_pred),
            "composed_minus_direct_mse": _mse(composed, target) - _mse(direct_pred, target),
        }

    # Algebraic inverse proxy: apply u then -u from each held-out state.  It is
    # explicitly not called a physical inverse because PushT contact dynamics
    # need not be reversible under action negation.
    initial = test_states[:, 0]
    forward = np.stack([
        _apply_map(action_fit, _design(initial[i : i + 1], test_actions[i, 0].reshape(1, -1)))[0]
        for i in range(len(test_rows))
    ])
    returned = np.stack([
        _apply_map(action_fit, _design(forward[i : i + 1], (-test_actions[i, 0]).reshape(1, -1)))[0]
        for i in range(len(test_rows))
    ])
    records["negated_action_return"] = {
        "mean_relative_return_defect": float(np.mean([_relative_error(returned[i], initial[i]) for i in range(len(initial))])),
        "mean_forward_displacement": float(np.mean(np.linalg.norm(forward - initial, axis=1))),
        "mean_return_displacement": float(np.mean(np.linalg.norm(returned - initial, axis=1))),
    }
    return records


def _crossfit(
    trajectories: list[dict[str, Any]],
    pca_dimension: int,
    ridge: float,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    episode_ids = np.asarray(sorted({int(row["episode_id"]) for row in trajectories}), dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for test_episode in episode_ids.tolist():
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        fold = {"test_episode": int(test_episode), "windows": len(test_rows)}
        for source_name in ("true", "predicted"):
            fold[source_name] = _evaluate_source(train_rows, test_rows, source_name, pca_dimension, ridge)
        folds.append(fold)

    def aggregate(source: str, path: str) -> dict[str, Any]:
        values = [fold[source][path.split("/")[0]][path.split("/")[1]] for fold in folds]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "episode_bootstrap": _cluster_metric(np.asarray(values), episode_ids, 20261004, bootstrap_repeats),
        }

    summary: dict[str, Any] = {"by_source": {}, "episode_associations": {}}
    for source in ("true", "predicted"):
        summary["by_source"][source] = {}
        for metric in (
            "one_step/state_only_mse", "one_step/action_conditioned_mse",
            "one_step/state_only_r2", "one_step/action_conditioned_r2",
            "one_step/action_gain_over_state_mse", "one_step/zero_minus_observed_mse",
            "one_step/permuted_minus_observed_mse",
            "negated_action_return/mean_relative_return_defect",
        ):
            key = metric.replace("/", "__")
            summary["by_source"][source][key] = aggregate(source, metric)
        for horizon in HORIZONS:
            for metric in ("direct_mse", "composed_mse", "composition_defect", "composed_minus_direct_mse"):
                values = np.asarray([fold[source]["composition"][f"h{horizon}"][metric] for fold in folds])
                summary["by_source"][source][f"h{horizon}__{metric}"] = _cluster_metric(
                    values, episode_ids, 20261010 + horizon, bootstrap_repeats
                )

    # Association is computed from held-out episode means.  The h=4/h=8 error
    # is the actual model-vs-target latent rollout MSE, not a regression target.
    for source in ("true", "predicted"):
        comp = np.asarray([fold[source]["composition"]["h8"]["composition_defect"] for fold in folds])
        errors = []
        for episode in episode_ids.tolist():
            rows = [row for row in trajectories if int(row["episode_id"]) == episode]
            errors.append(float(np.mean([np.square(row["true"][8] - row["predicted"][8]).mean() for row in rows])))
        errors = np.asarray(errors)
        summary["episode_associations"][source] = {
            "composition_defect_h8_vs_rollout_mse_h8_pearson": _pearson(comp, errors),
            "composition_defect_h8_vs_rollout_mse_h8_spearman": _spearman(comp, errors),
            "episodes": int(len(episode_ids)),
        }
    return {"folds": folds, "summary": summary}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]
    lines = [
        "# External frozen LeWM action-algebra audit",
        "",
        "All maps are fit in fold-specific PCA coordinates and evaluated leave-one-episode-out. Values are episode-bootstrap means with 95% CIs unless stated otherwise.",
        "",
        "| source | state-only h=1 MSE | action-conditioned h=1 MSE | action gain | zero-action penalty | permuted-action penalty |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in ("true", "predicted"):
        item = summary["by_source"][source]
        def f(key: str) -> str:
            value = item[key]
            return f"{value['mean']:.6g} [{value['episode_bootstrap']['ci95'][0]:.6g}, {value['episode_bootstrap']['ci95'][1]:.6g}]"
        lines.append(
            f"| {source} | {f('one_step__state_only_mse')} | {f('one_step__action_conditioned_mse')} | "
            f"{f('one_step__action_gain_over_state_mse')} | {f('one_step__zero_minus_observed_mse')} | "
            f"{f('one_step__permuted_minus_observed_mse')} |"
        )
    lines.extend([
        "",
        "| source | horizon | direct MSE | composed MSE | composition defect | composed − direct MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for source in ("true", "predicted"):
        item = summary["by_source"][source]
        for horizon in HORIZONS:
            def g(metric: str) -> str:
                value = item[f"h{horizon}__{metric}"]
                return f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
            lines.append(f"| {source} | {horizon} | {g('direct_mse')} | {g('composed_mse')} | {g('composition_defect')} | {g('composed_minus_direct_mse')} |")
    lines.extend([
        "",
        "| source | negated-action return defect | Pearson(comp h=8, rollout MSE h=8) | Spearman(comp h=8, rollout MSE h=8) |",
        "| --- | ---: | ---: | ---: |",
    ])
    for source in ("true", "predicted"):
        item = summary["by_source"][source]
        association = summary["episode_associations"][source]
        lines.append(
            f"| {source} | {item['negated_action_return__mean_relative_return_defect']['mean']:.6g} | "
            f"{association['composition_defect_h8_vs_rollout_mse_h8_pearson']:.6g} | "
            f"{association['composition_defect_h8_vs_rollout_mse_h8_spearman']:.6g} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The action-conditioned map is a cross-fitted low-dimensional latent diagnostic. A positive action gain or action-order penalty shows that the fitted latent transition uses aligned action features; it does not establish a causal physical action effect.",
        "",
        "Composition defect compares repeated one-step maps with separately fitted direct horizon maps. The negated-action experiment is an algebraic inverse proxy, not a claim that PushT contact dynamics are reversible. The episode correlations are descriptive and do not establish that composition error mediates rollout error.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run_report(
    val_root: Path,
    archive: Path,
    policy: str,
    output: Path,
    windows_per_episode: int = 2,
    pca_dimension: int = 16,
    ridge: float = 1e-2,
    device_name: str = "cpu",
    normalization_mode: str = "public",
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy

    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        episode_count = len(pickle.load(handle))
    model = load_policy(policy, device=device_name)
    trajectories = _load_trajectories(
        model,
        val_root,
        torch.device(device_name),
        normalization_mode,
        windows_per_episode,
        max(HORIZONS),
    )
    window_counts = {
        episode_id: sum(int(row["episode_id"]) == episode_id for row in trajectories)
        for episode_id in range(episode_count)
    }
    if any(count <= 0 or count > windows_per_episode for count in window_counts.values()):
        raise ValueError("invalid observed window count for one or more episodes")
    crossfit = _crossfit(trajectories, pca_dimension, ridge, bootstrap_repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact": "external_frozen_lewm_action_algebra_audit",
        "status": "frozen_external_action_conditioned_latent_operator_diagnostic",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "episodes": int(episode_count),
            "windows": int(len(trajectories)),
            "windows_per_episode_requested": int(windows_per_episode),
            "windows_per_episode_observed": sorted(set(window_counts.values())),
        },
        "model": {"policy": policy, "device": device_name},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "horizons": list(HORIZONS),
            "windows_per_episode": int(windows_per_episode),
            "action_feature": "mean of each normalized five-frame action block aligned to z_t -> z_{t+1}",
            "operator": "T_u(z)=A z+B u+b fitted by ridge regression",
            "state_only_control": "T(z)=A z+b with identical folds and ridge",
            "crossfit": "leave-one-episode-out",
            "pca_dimension": int(pca_dimension),
            "ridge": float(ridge),
            "action_normalization": normalization_mode,
            "bootstrap": "episode-cluster bootstrap",
            "bootstrap_repeats": int(bootstrap_repeats),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "This is a frozen external latent-operator diagnostic, not a retraining or online-control experiment.",
            "Action sensitivity and composition are conditional on the observed external trajectories and do not establish causal controllability.",
            "The negated-action return statistic is an algebraic inverse proxy; it is not a physical reversibility claim.",
            "Episode-level correlations are descriptive and do not establish mediation or planning benefit.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(
        "Frozen external action-algebra audit completed with leave-one-episode-out fits.\n"
        "Interpret action sensitivity and composition as latent diagnostics, not causal control evidence.\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--pca-dimension", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        archive=args.archive,
        policy=args.policy,
        output=args.output,
        windows_per_episode=args.windows_per_episode,
        pca_dimension=args.pca_dimension,
        ridge=args.ridge,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
