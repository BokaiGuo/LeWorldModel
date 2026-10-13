"""External action-alignment controls for the frozen LeWM temporal audit.

The existing external temporal audit compares observed actions with a zero
action input ablation.  This companion audit keeps the released action
marginal and the same action-block preprocessing, but assigns an action block
from another window to the target observation window.  It therefore tests
whether the frozen predictor benefits from the correct action--observation
alignment rather than merely from receiving nonzero action features.

This remains a frozen latent diagnostic.  It is not an environment
intervention, a causal-control estimate, or an online-planning evaluation.
"""

from __future__ import annotations

import argparse
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

from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    PUBLIC_ACTION_MEAN,
    PUBLIC_ACTION_STD,
    _bootstrap_clustered,
    _load_video_frames,
    _make_action_blocks,
    _predict_horizons,
    _window_starts,
)


CONDITIONS = ("observed", "zero", "shift_next", "random_derangement")


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _random_derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("a derangement requires at least two windows")
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    for _ in range(1000):
        permutation = rng.permutation(size)
        if not np.any(permutation == identity):
            return permutation
    raise RuntimeError("failed to generate a derangement")


def _predict_with_actions(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    action_blocks: torch.Tensor,
    horizons: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    with torch.inference_mode():
        action_embedding = model.action_encoder(action_blocks.to(device))
        return _predict_horizons(
            model,
            embedding,
            action_embedding,
            horizons,
            CONTEXT_LENGTH,
        )


def run_report(
    val_root: Path,
    policy: str,
    output: Path,
    horizons: list[int],
    windows_per_episode: int,
    device_name: str,
    permutation_seed: int,
    normalization_mode: str,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy
    from utils import get_img_preprocessor

    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_policy(policy, device=device_name)
    device = torch.device(device_name)
    actions = torch.load(
        val_root / "rel_actions.pth", map_location="cpu", weights_only=False
    ).float()
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(v) for v in pickle.load(handle)]
    all_actions = actions / 100.0
    if normalization_mode == "public":
        action_mean, action_std = PUBLIC_ACTION_MEAN, PUBLIC_ACTION_STD
    elif normalization_mode == "val_zscore":
        action_mean = all_actions.mean(dim=(0, 1))
        action_std = all_actions.std(dim=(0, 1)).clamp_min(1e-6)
    elif normalization_mode == "raw_div100":
        action_mean, action_std = torch.zeros(2), torch.ones(2)
    else:
        raise ValueError(f"unsupported normalization mode: {normalization_mode}")
    preprocess = get_img_preprocessor("pixels", "pixels", 224)

    windows: list[dict[str, Any]] = []
    for episode_id, length in enumerate(lengths):
        starts = _window_starts(
            length,
            context_length=CONTEXT_LENGTH,
            max_horizon=max(horizons),
            windows_per_episode=windows_per_episode,
        )
        print(f"episode={episode_id} length={length} windows={len(starts)}", flush=True)
        for start in starts:
            count = CONTEXT_LENGTH + max(horizons)
            frame_indices = [FRAME_STRIDE * (start + i) for i in range(count)]
            frames = _load_video_frames(
                val_root / "obses" / f"episode_{episode_id:03d}.mp4", frame_indices
            )
            pixels = preprocess({"pixels": frames})["pixels"].unsqueeze(0).to(device)
            action_blocks = _make_action_blocks(
                actions[episode_id],
                start,
                count,
                "observed",
                normalization_mode=normalization_mode,
                action_mean=action_mean,
                action_std=action_std,
            )
            with torch.inference_mode():
                embedding = model.encode({"pixels": pixels})["emb"].detach()
            windows.append(
                {
                    "episode_id": int(episode_id),
                    "start_observation": int(start),
                    "embedding": embedding,
                    "action_blocks": action_blocks,
                }
            )

    if len(windows) < 2:
        raise ValueError("at least two windows are required")
    derangement = _random_derangement(len(windows), permutation_seed)
    records: list[dict[str, Any]] = []
    for target_index, window in enumerate(windows):
        condition_blocks = {
            "observed": window["action_blocks"],
            "zero": torch.zeros_like(window["action_blocks"]),
            "shift_next": windows[(target_index + 1) % len(windows)]["action_blocks"],
            "random_derangement": windows[int(derangement[target_index])]["action_blocks"],
        }
        source_indices = {
            "observed": target_index,
            "zero": None,
            "shift_next": (target_index + 1) % len(windows),
            "random_derangement": int(derangement[target_index]),
        }
        for condition in CONDITIONS:
            predictions = _predict_with_actions(
                model,
                window["embedding"],
                condition_blocks[condition].unsqueeze(0),
                horizons,
                device,
            )
            for horizon in horizons:
                target = window["embedding"][:, CONTEXT_LENGTH - 1 + horizon]
                mse = float(torch.square(predictions[horizon] - target).mean())
                records.append(
                    {
                        "window_index": int(target_index),
                        "episode_id": window["episode_id"],
                        "start_observation": window["start_observation"],
                        "condition": condition,
                        "action_source_window": source_indices[condition],
                        "horizon": int(horizon),
                        "mse": mse,
                    }
                )

    summary: dict[str, Any] = {
        "by_condition_horizon": {},
        "paired_delta_vs_observed": {},
    }
    for condition in CONDITIONS:
        for horizon in horizons:
            key = f"{condition}/h{horizon}"
            rows = [
                row
                for row in records
                if row["condition"] == condition and row["horizon"] == horizon
            ]
            values = np.asarray([row["mse"] for row in rows], dtype=np.float64)
            groups = np.asarray([row["episode_id"] for row in rows])
            summary["by_condition_horizon"][key] = {
                "mse": _bootstrap_clustered(
                    values, groups, repeats=2000, seed=20265003 + horizon
                ),
                "windows": int(values.size),
            }
            if condition != "observed":
                baseline = np.asarray(
                    [
                        row["mse"]
                        for row in records
                        if row["condition"] == "observed"
                        and row["horizon"] == horizon
                    ],
                    dtype=np.float64,
                )
                summary["paired_delta_vs_observed"][key] = _bootstrap_clustered(
                    values - baseline,
                    groups,
                    repeats=2000,
                    seed=20266003 + horizon,
                )

    report = {
        "artifact": "external_lewm_action_alignment_permutation_audit",
        "status": "frozen_external_action_alignment_control",
        "source": {"validation_root": str(val_root), "episodes": len(lengths)},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": horizons,
            "windows_per_episode": int(windows_per_episode),
            "conditions": list(CONDITIONS),
            "permutation_seed": int(permutation_seed),
            "random_derangement": derangement.tolist(),
            "bootstrap": "episode-cluster paired bootstrap, 2000 replicates",
            "device": device_name,
            "policy": policy,
            "action_preprocessing": "public release normalization used by the external temporal audit",
            "action_normalization": {
                "mode": normalization_mode,
                "raw_action_divisor": 100.0,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": (
                    "public JEPA-WM PushT dataset loader constants"
                    if normalization_mode == "public"
                    else "validation split statistics"
                    if normalization_mode == "val_zscore"
                    else "unit scale after raw action division"
                ),
            },
        },
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "The shifted and deranged arms preserve released action blocks but break their alignment with the target observation window.",
            "This is a frozen latent input control, not an intervention on the physical environment.",
            "The result does not establish causal controllability, online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# External frozen LeWM action-alignment permutation audit",
        "",
        f"Action normalization: `{report['protocol']['action_normalization']['mode']}`. The observed arm uses the action block aligned with each target window. The zero arm removes action input, while shifted and deranged arms preserve released action blocks but assign them to a different target window.",
        "",
        "| condition | h | MSE | 95% cluster CI | delta vs observed | delta 95% CI |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for key, value in report["summary"]["by_condition_horizon"].items():
        condition, horizon = key.split("/h")
        mse = value["mse"]
        if condition == "observed":
            lines.append(
                f"| {condition} | {horizon} | {mse['mean']:.6g} | [{mse['ci95'][0]:.6g}, {mse['ci95'][1]:.6g}] | -- | -- |"
            )
        else:
            delta = report["summary"]["paired_delta_vs_observed"][key]
            lines.append(
                f"| {condition} | {horizon} | {mse['mean']:.6g} | [{mse['ci95'][0]:.6g}, {mse['ci95'][1]:.6g}] | "
                f"{delta['mean']:.6g} | [{delta['ci95'][0]:.6g}, {delta['ci95'][1]:.6g}] |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--permutation-seed", type=int, default=20261003)
    parser.add_argument(
        "--action-normalization",
        choices=["public", "val_zscore", "raw_div100"],
        default="public",
    )
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        policy=args.policy,
        output=args.output,
        horizons=args.horizons,
        windows_per_episode=args.windows_per_episode,
        device_name=args.device,
        permutation_seed=args.permutation_seed,
        normalization_mode=args.action_normalization,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
