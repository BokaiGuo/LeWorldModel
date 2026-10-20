"""Train a frozen RGB-successor latent adapter from joined observations.

The adapter is deliberately separate from GDD projection.  It uses only the
train-root rows of RGB successor joins to predict a latent successor delta
from the source latent and the complete action sequence.  A train-only latent
to XY readout turns the predicted successor into a task-conditioned score at
runtime.  Calibration and outcome rows remain evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.decision_benchmark.train_candidate_effect_adapter import action_sequence_basis


class SuccessorDeltaMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _array_key(reference: str) -> str:
    parts = str(reference).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid realized latent reference: {reference}")
    return parts[1]


def _load_rows(dataset: Path, arrays: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{dataset} has no records")
    values = np.load(arrays, allow_pickle=False)
    source_array_cache: dict[Path, Any] = {}
    output: list[dict[str, Any]] = []
    for row in rows:
        key = _array_key(row["realized_latent_ref"])
        if key not in values.files:
            raise ValueError(f"{arrays} lacks {key}")
        updated = dict(row)
        updated["_realized_latent"] = np.asarray(values[key], dtype=np.float64).reshape(-1)
        source_dataset = Path(str(row.get("source_candidate_dataset", dataset)))
        if not source_dataset.is_absolute():
            source_dataset = dataset.parent / source_dataset
        source_arrays_path = source_dataset.parent / "arrays.npz"
        if source_arrays_path not in source_array_cache:
            source_array_cache[source_arrays_path] = np.load(source_arrays_path, allow_pickle=False)
        state_key = _array_key(row["realized_terminal_state_ref"])
        source_arrays = source_array_cache[source_arrays_path]
        if state_key not in source_arrays.files:
            raise ValueError(f"{source_arrays_path} lacks {state_key}")
        updated["_realized_state"] = np.asarray(source_arrays[state_key], dtype=np.float64).reshape(-1)[:2]
        output.append(updated)
    return output


def _features(rows: list[dict[str, Any]], feature_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if feature_mode not in {"basis", "raw"}:
        raise ValueError("feature_mode must be basis or raw")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    source_latents: list[np.ndarray] = []
    source_states: list[np.ndarray] = []
    roots: list[str] = []
    for row in rows:
        source = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1)
        realized = np.asarray(row["_realized_latent"], dtype=np.float64).reshape(-1)
        sequence = np.asarray(row["action_sequence"], dtype=np.float64)
        if source.shape != realized.shape or source.size < 2:
            raise ValueError("source and realized latent dimensions disagree")
        basis = action_sequence_basis(sequence) if feature_mode == "basis" else sequence.reshape(-1)
        xs.append(np.concatenate([source, basis]))
        ys.append(realized - source)
        source_latents.append(source)
        source_states.append(np.asarray(row["_realized_state"], dtype=np.float64).reshape(-1)[:2])
        roots.append(str(row["instance_id"]))
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            np.asarray(source_latents, dtype=np.float64), np.asarray(source_states, dtype=np.float64), roots)


def _fit_readout(source_latents: np.ndarray, states: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = source_latents[indices]
    y = states[indices]
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)
    design = np.c_[np.ones(len(x)), (x - mean) / scale]
    beta = np.linalg.solve(design.T @ design + 1.0e-3 * np.eye(design.shape[1]), design.T @ y)
    return beta, mean, scale


def build(
    datasets: list[Path],
    arrays: list[Path],
    output: Path,
    *,
    seed: int = 8411,
    epochs: int = 400,
    device: str = "cuda",
    feature_mode: str = "basis",
) -> dict[str, Any]:
    if len(datasets) != len(arrays) or not datasets:
        raise ValueError("datasets and arrays must be non-empty and aligned")
    rows = [row for dataset, array in zip(datasets, arrays) for row in _load_rows(dataset, array)]
    x, y, source_latents, source_states, roots = _features(rows, feature_mode)
    train = np.asarray([i for i, row in enumerate(rows) if str(row.get("split")) == "train"], dtype=np.int64)
    calibration = np.asarray([i for i, row in enumerate(rows) if str(row.get("split")) == "calibration"], dtype=np.int64)
    outcome = np.asarray([i for i, row in enumerate(rows) if str(row.get("split")) == "outcome"], dtype=np.int64)
    if len(train) < 16 or len(calibration) < 8 or len(outcome) < 8:
        raise ValueError("successor adapter requires train, calibration, and outcome support")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed(seed)
    x_mean = x[train].mean(axis=0)
    x_scale = np.maximum(x[train].std(axis=0), 1.0e-3)
    y_mean = y[train].mean(axis=0)
    y_scale = np.maximum(y[train].std(axis=0), 1.0e-3)
    model = SuccessorDeltaMLP(x.shape[1], y.shape[1]).to(requested)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    xt = torch.as_tensor((x[train] - x_mean) / x_scale, dtype=torch.float32, device=requested)
    yt = torch.as_tensor((y[train] - y_mean) / y_scale, dtype=torch.float32, device=requested)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(xt) - yt) ** 2)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        prediction = model(torch.as_tensor((x - x_mean) / x_scale, dtype=torch.float32, device=requested)).cpu().numpy()
    prediction = prediction * y_scale + y_mean
    beta, readout_mean, readout_scale = _fit_readout(source_latents, source_states, train)

    def readout(values: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(len(values)), (values - readout_mean) / readout_scale] @ beta

    def root_eval(indices: np.ndarray) -> dict[str, Any]:
        grouped: dict[str, list[int]] = {}
        for index in indices.tolist():
            grouped.setdefault(roots[index], []).append(index)
        regrets: list[float] = []
        hits: list[bool] = []
        for group in grouped.values():
            predicted_terminal = source_latents[group] + prediction[group]
            pred_xy = readout(predicted_terminal)
            target = np.asarray([rows[group[0]]["goal_raw"]], dtype=np.float64)[:, :2]
            pred_cost = np.linalg.norm(pred_xy - target, axis=1)
            real_cost = np.asarray([float(rows[i]["real_task_cost"]) for i in group])
            selected = int(np.argmin(pred_cost))
            regrets.append(float(real_cost[selected] - real_cost.min()))
            hits.append(bool(selected == int(np.argmin(real_cost))))
        return {"root_count": len(grouped), "mean_regret": float(np.mean(regrets)),
                "top1_exact": float(np.mean(hits)), "regrets": regrets}

    artifact = {
        "schema": "gdd_wm_rgb_successor_latent_adapter.v0.1",
        "model_state": model.state_dict(),
        "input_dim": int(x.shape[1]), "latent_dim": int(y.shape[1]),
        "x_mean": x_mean.tolist(), "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(), "y_scale": y_scale.tolist(),
        "readout_beta": beta.tolist(), "readout_mean": readout_mean.tolist(), "readout_scale": readout_scale.tolist(),
        "action_feature": "action_sequence_basis_v1" if feature_mode == "basis" else "action_sequence_raw_v1",
        "feature_mode": feature_mode,
        "training_root_only": True,
        "source_datasets": [str(path.resolve()) for path in datasets],
        "seed": int(seed), "epochs": int(epochs),
        "claim_boundary": "RGB successor-supervised adapter; not universal GDD or causal latent validation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    summary = {"schema": "gdd_wm_rgb_successor_latent_adapter_summary.v0.1",
               "artifact": str(output.resolve()), "row_count": len(rows),
               "train_root_count": len(set(np.asarray(roots)[train])),
               "calibration": root_eval(calibration), "outcome": root_eval(outcome),
               "claim_boundary": artifact["claim_boundary"]}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--arrays", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8411)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-mode", choices=("basis", "raw"), default="basis")
    args = parser.parse_args()
    print(json.dumps(build(args.datasets, args.arrays, args.output, seed=args.seed, epochs=args.epochs, device=args.device, feature_mode=args.feature_mode), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
