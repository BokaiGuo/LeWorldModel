"""Fit a low-rank view operator in frozen I-JEPA feature space.

This is a bounded learned-representation anchor for the paper's operator-pair
question.  A frozen image encoder supplies source and transformed-view
features; a ridge linear map is fitted in the source PCA coordinates on a
training split and evaluated on held-out pairs.  We then compare its
covariance commutator with spectrum-preserving random basis rotations.  The
script does not claim that the fitted map is the JEPA training dynamics or a
policy/world-model transition.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.theory_spectral_transfer.run_ijepa_feature_anchor import (
    extract_features,
    choose_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/ijepa_vith16_1k")
    parser.add_argument("--dataset", default="sklearn-digits")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_anchor"),
    )
    parser.add_argument("--max-images", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--operator-rank", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--random-draws", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--view-transform", choices=("hflip", "color", "crop"), default="hflip")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _fit_operator(source: np.ndarray, target: np.ndarray, rank: int, ridge: float):
    mean = source.mean(axis=0, keepdims=True)
    centered = source - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    q = min(int(rank), centered.shape[0] - 1, centered.shape[1])
    basis = vt[:q].T
    eigvals = (singular[:q] ** 2) / max(1, centered.shape[0] - 1)
    x = centered @ basis
    y = (target - mean) @ basis
    gram = x.T @ x + float(ridge) * np.eye(q)
    operator = np.linalg.solve(gram, x.T @ y).T
    covariance = (x.T @ x) / max(1, x.shape[0] - 1)
    return mean, basis, eigvals, covariance, operator, x, y


def _commutator_norm(covariance: np.ndarray, operator: np.ndarray) -> float:
    denom = np.linalg.norm(covariance, ord="fro") * np.linalg.norm(operator, ord="fro")
    if denom <= 0:
        return 0.0
    return float(np.linalg.norm(covariance @ operator - operator @ covariance, ord="fro") / denom)


def _random_rotation(q: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(q, q))
    orthogonal, signs = np.linalg.qr(matrix)
    diagonal = np.sign(np.diag(signs))
    diagonal[diagonal == 0] = 1.0
    return orthogonal @ np.diag(diagonal)


def _top_lift(values: np.ndarray, top_k: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values**2))
    if total <= 0:
        return 0.0
    k = min(int(top_k), values.shape[1])
    return float(np.sum(values[:, :k] ** 2) / total / (k / values.shape[1]))


def run(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    # Reuse the exact deterministic feature extraction path from the existing
    # I-JEPA anchor so the two learned-representation measurements are paired.
    source, target, labels = extract_features(args, device)
    n = source.shape[0]
    if n < 8:
        raise ValueError("at least eight image pairs are required for a holdout fit")
    order = rng.permutation(n)
    holdout = min(max(int(round(n * args.holdout_fraction)), 2), n - 2)
    train_idx, test_idx = order[:-holdout], order[-holdout:]
    mean, basis, eigvals, covariance, operator, x_train, y_train = _fit_operator(
        source[train_idx], target[train_idx], args.operator_rank, args.ridge
    )
    x_test = (source[test_idx] - mean) @ basis
    y_test = (target[test_idx] - mean) @ basis
    y_pred = x_test @ operator.T
    baseline = y_test - y_test.mean(axis=0, keepdims=True)
    mse = float(np.mean((y_pred - y_test) ** 2))
    base_mse = float(np.mean(baseline**2))
    test_r2 = float(1.0 - mse / max(base_mse, 1e-12))
    observed_comm = _commutator_norm(covariance, operator)
    null_comm = []
    for _ in range(int(args.random_draws)):
        rotation = _random_rotation(operator.shape[0], rng)
        null_comm.append(_commutator_norm(covariance, rotation @ operator @ rotation.T))
    null = np.asarray(null_comm, dtype=np.float64)
    predicted_displacement = y_pred - x_test
    top_k = min(int(args.top_k), basis.shape[1])
    observed_lift = _top_lift(predicted_displacement, top_k)
    # A sample-level permutation keeps feature norms and the source covariance
    # fixed while removing the paired view relation.
    perm = rng.permutation(x_test.shape[0])
    shuffled_lift = _top_lift(y_test[perm] - x_test, top_k)
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "view_transform": args.view_transform,
            "images": int(n),
            "train_images": int(train_idx.size),
            "test_images": int(test_idx.size),
            "feature_dim": int(source.shape[1]),
            "operator_rank": int(basis.shape[1]),
            "top_k": int(top_k),
            "ridge": float(args.ridge),
            "holdout_fraction": float(args.holdout_fraction),
            "seed": int(args.seed),
            "device": str(device),
        },
        "holdout": {
            "mse": mse,
            "baseline_mse": base_mse,
            "r2": test_r2,
        },
        "operator_pair": {
            "covariance_trace": float(np.trace(covariance)),
            "operator_frobenius": float(np.linalg.norm(operator, ord="fro")),
            "commutator_norm": observed_comm,
            "operator_eigenvalues": [float(v) for v in np.linalg.eigvals(operator).real],
            "covariance_eigenvalues": [float(v) for v in eigvals],
        },
        "spectrum_preserving_null": {
            "mean": float(null.mean()),
            "p05": float(np.quantile(null, 0.05)),
            "p95": float(np.quantile(null, 0.95)),
            "draws": int(null.size),
        },
        "transfer_readout": {
            "heldout_predicted_displacement_top_k_lift": observed_lift,
            "heldout_pair_shuffled_top_k_lift": shuffled_lift,
        },
        "evidence_boundary": (
            "Held-out low-rank linear view-operator anchor in frozen I-JEPA feature space; "
            "not JEPA training-quality, action-conditioned dynamics, planning, or theorem proof."
        ),
    }


def write_markdown(path: Path, result: dict[str, object]) -> None:
    cfg = result["config"]
    holdout = result["holdout"]
    pair = result["operator_pair"]
    null = result["spectrum_preserving_null"]
    transfer = result["transfer_readout"]
    lines = [
        "# Frozen I-JEPA View-Operator Anchor",
        "",
        f"Dataset: {cfg['dataset']} ({cfg['view_transform']}), {cfg['images']} pairs; "
        f"rank-{cfg['operator_rank']} fit with {cfg['test_images']} held-out pairs.",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| held-out R2 | {holdout['r2']:.6f} |",
        f"| observed covariance-operator commutator | {pair['commutator_norm']:.6f} |",
        f"| spectrum-preserving random-basis mean | {null['mean']:.6f} |",
        f"| random-basis P05--P95 | {null['p05']:.6f}--{null['p95']:.6f} |",
        f"| held-out predicted-displacement top-{cfg['top_k']} lift | {transfer['heldout_predicted_displacement_top_k_lift']:.6f} |",
        f"| held-out pair-shuffled top-{cfg['top_k']} lift | {transfer['heldout_pair_shuffled_top_k_lift']:.6f} |",
        "",
        "The fitted operator is evaluated on held-out view pairs. The commutator is compared with orthogonal conjugates that preserve the fitted operator spectrum. This is a learned-representation anchor for the covariance--predictor measurement question; it is not evidence for JEPA training quality or a world-model control claim.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    (args.output_dir / "jepa_operator_anchor.json").write_text(json.dumps(result, indent=2) + "\n")
    write_markdown(args.output_dir / "jepa_operator_anchor.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
