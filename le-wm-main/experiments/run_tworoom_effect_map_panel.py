"""Run a frozen paired comparison of direction and local-effect RGB gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUNNER = Path(__file__).with_name("run_live_tworoom_planning.py")


def _command(args: argparse.Namespace, output: Path, use_effect_map: bool) -> list[str]:
    command = [
        sys.executable, str(RUNNER),
        "--device", args.device,
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--max-episode-steps", str(args.max_episode_steps),
        "--horizon", str(args.horizon),
        "--receding-horizon", str(args.receding_horizon),
        "--action-block", str(args.action_block),
        "--num-samples", str(args.num_samples),
        "--n-steps", str(args.n_steps),
        "--topk", str(args.topk),
        "--output", str(output),
        "--hybrid-conflict-cos", str(args.hybrid_conflict_cos),
        "--hybrid-model-min-norm", str(args.hybrid_model_min_norm),
    ]
    if use_effect_map:
        command.append("--hybrid-use-effect-map")
    return command


def _load(path: Path) -> dict[str, Any]:
    artifact = path / "planning.json" if path.is_dir() else path
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid panel artifact: {artifact}")
    return payload


def _paired(left: list[int], right: list[int]) -> dict[str, int]:
    if len(left) != len(right):
        raise ValueError("paired outcome vectors do not have equal length")
    delta = np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64)
    return {
        "left_wins": int(np.sum(delta > 0)),
        "right_wins": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20269200)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hybrid-conflict-cos", type=float, default=0.0)
    parser.add_argument("--hybrid-model-min-norm", type=float, default=0.08)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for the paired effect-map panel but unavailable")
    if args.episodes < 2 or args.topk >= args.num_samples:
        raise ValueError("require episodes >= 2 and topk < num_samples")
    args.output.mkdir(parents=True, exist_ok=True)
    direction_path = args.output / "direction_gate.json"
    effect_path = args.output / "effect_map_gate.json"
    for path, enabled in ((direction_path, False), (effect_path, True)):
        subprocess.run(_command(args, path, enabled), cwd=RUNNER.parent.parent, check=True)
    direction = _load(direction_path)
    effect = _load(effect_path)
    direction_artifact = direction_path / "planning.json"
    effect_artifact = effect_path / "planning.json"
    seed_order = direction["hybrid_rgb_fallback"]["seeds"]
    if seed_order != effect["hybrid_rgb_fallback"]["seeds"]:
        raise RuntimeError("paired panel arms used different seed orders")
    direction_success = direction["hybrid_rgb_fallback"]["episode_successes"]
    effect_success = effect["hybrid_rgb_fallback"]["episode_successes"]
    report = {
        "schema": "tworoom_effect_map_paired_panel.v0.1",
        "protocol": {
            "device": args.device,
            "seed": args.seed,
            "episodes": args.episodes,
            "same_fresh_roots": True,
            "only_difference": "root-local effect-map gate enabled versus disabled",
            "outcome_used_for_parameter_selection": False,
            "effect_gate": {
                "minimum_rows": 6,
                "minimum_rank": 2,
                "maximum_condition_number": 100.0,
                "maximum_normalized_residual": 0.75,
            },
        },
        "direction_gate": direction["hybrid_rgb_fallback"],
        "effect_map_gate": effect["hybrid_rgb_fallback"],
        "paired_effect_minus_direction": {
            "seed_order": seed_order,
            **_paired(effect_success, direction_success),
            "success_rate_delta": float(
                np.mean(np.asarray(effect_success) - np.asarray(direction_success))
            ),
        },
        "artifacts": {
            "direction": str(direction_artifact),
            "effect_map": str(effect_artifact),
        },
        "claim_boundary": (
            "paired TwoRoom mechanism comparison; no universal world-model or "
            "cross-environment claim"
        ),
    }
    (args.output / "panel.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(report["paired_effect_minus_direction"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
