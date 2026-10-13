"""Bidirectional cross-fitted operator-direction ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analyze_temporal_operator_controls import analyze_horizon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--primary-rank", type=int, default=8)
    parser.add_argument("--primary-ridge", type=float, default=1e-2)
    args = parser.parse_args()
    if len(args.input) != len(args.label):
        raise ValueError("Provide one --label per --input")
    # Reuse the same deterministic split/control protocol as the main control.
    control_args = argparse.Namespace(
        ranks=args.ranks,
        ridge=args.ridge,
        folds=5,
        split_seeds=[0, 1, 2],
        control_seeds=[0, 1, 2, 3],
        primary_rank=args.primary_rank,
        primary_ridge=args.primary_ridge,
    )
    output = {
        "protocol": {
            "direction_pairs": ["target_to_predicted", "predicted_to_target"],
            "ranks": args.ranks,
            "ridge": args.ridge,
            "primary_setting": {"rank": args.primary_rank, "ridge": args.primary_ridge},
            "same_folds_and_controls": True,
        },
        "models": {},
        "evidence_boundary": "Directionality estimator ablation on stored checkpoint latents; not a causal or universal theorem.",
    }
    for label, path in zip(args.label, args.input):
        payload = json.loads(path.read_text())
        vectors = payload.get("operator_fit_vectors")
        if vectors is None:
            raise ValueError(f"{path} lacks operator_fit_vectors")
        model = {"source": str(path), "horizons": {}}
        for horizon in sorted(vectors["target"], key=int):
            target = torch.tensor(vectors["target"][horizon], dtype=torch.float64)
            predicted = torch.tensor(vectors["predicted"][horizon], dtype=torch.float64)
            forward = analyze_horizon(target, predicted, control_args)
            reverse = analyze_horizon(predicted, target, control_args)
            model["horizons"][horizon] = {
                "target_to_predicted": forward,
                "predicted_to_target": reverse,
            }
        output["models"][label] = model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    compact = {}
    primary = f"rank={args.primary_rank},ridge={args.primary_ridge:g}"
    for label, model in output["models"].items():
        compact[label] = {}
        for horizon, value in model["horizons"].items():
            compact[label][horizon] = {
                direction: value[direction]["primary"]["paired_r2"]["mean"]
                for direction in ["target_to_predicted", "predicted_to_target"]
            }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
