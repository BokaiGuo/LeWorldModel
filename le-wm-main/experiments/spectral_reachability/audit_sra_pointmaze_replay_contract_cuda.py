"""CUDA-host-gated audit of PointMaze state/action replay used by SRA P1/P2.

The SRA learned model is not evaluated here.  This separate contract audit
checks the *evaluator* dependency: raw recorded actions must replay from raw
recorded states into the recorded +5/+10 successors under the same legacy
MuJoCo XML that P1/P2 use.  A mismatch invalidates physical-label claims; it
is not evidence for or against spectral planning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability.pointmaze_physical_evaluator import build_model, replay_recorded_actions
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, require_cuda, sha256_file, write_json


def parse_ints(value: str, label: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be nonempty and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--roots", default="1880,1881,1882,1883,1884,1885,1886,1887,1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899")
    parser.add_argument("--frames", default="0,5,10,15,20,25,30,35,40")
    parser.add_argument("--horizons", default="5,10")
    parser.add_argument("--state-tolerance", type=float, default=2.0e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots, frames, horizons = parse_ints(args.roots, "roots"), parse_ints(args.frames, "frames"), parse_ints(args.horizons, "horizons")
    if args.output_dir.exists():
        raise FileExistsError(f"immutable replay-contract output exists: {args.output_dir}")
    if any(horizon <= 0 for horizon in horizons) or args.state_tolerance <= 0:
        raise ValueError("horizons and tolerance must be positive")
    # This is a host-executed physical evaluator audit, but require a real CUDA
    # allocation so it cannot be mistaken for a replacement CPU learned run.
    device = require_cuda()
    states_path, actions_path = args.dataset_root / "states.pth", args.dataset_root / "actions.pth"
    if not states_path.is_file() or not actions_path.is_file():
        raise FileNotFoundError("PointMaze root requires states.pth and actions.pth")
    states = torch.load(states_path, map_location="cpu", weights_only=True).numpy().astype(np.float64)
    actions = torch.load(actions_path, map_location="cpu", weights_only=True).numpy().astype(np.float64)
    if states.ndim != 3 or states.shape[-1] != 4 or actions.shape[:2] != states.shape[:2] or actions.shape[-1] != 2:
        raise ValueError(f"unexpected state/action layout: {states.shape}, {actions.shape}")
    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml
    xml = _legacy_pointmaze_xml()
    physics = build_model()
    rows = []
    for root in roots:
        if root < 0 or root >= len(states):
            raise ValueError(f"root {root} outside dataset")
        for frame in frames:
            for horizon in horizons:
                if frame < 0 or frame + horizon >= states.shape[1]:
                    continue
                predicted = replay_recorded_actions(physics, states[root, frame], actions[root, frame : frame + horizon])
                target = states[root, frame + horizon]
                error = predicted - target
                rows.append({"root": root, "frame": frame, "horizon": horizon, "state_l2_error": float(np.linalg.norm(error)), "position_l2_error": float(np.linalg.norm(error[:2])), "velocity_l2_error": float(np.linalg.norm(error[2:])), "predicted_state": predicted.tolist(), "recorded_state": target.tolist()})
    if not rows:
        raise RuntimeError("no valid root/frame/horizon replay rows")
    errors = np.asarray([row["state_l2_error"] for row in rows])
    by_horizon = {str(horizon): [row["state_l2_error"] for row in rows if row["horizon"] == horizon] for horizon in horizons}
    passed = bool(float(errors.max()) <= args.state_tolerance)
    report = {
        "schema": "sra_pointmaze_legacy_replay_contract.v1",
        "status": "pass" if passed else "fail",
        "scope": "evaluator state/action contract only; learned P0/P1/P2 computations remain CUDA-only and are not rerun here",
        "cuda_host_preflight": {"device": str(device), "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "inputs": {"dataset": str(args.dataset_root), "states_sha256": sha256_file(states_path), "actions_sha256": sha256_file(actions_path), "legacy_xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(), "roots": roots, "frames": frames, "horizons": horizons},
        "summary": {"rows": len(rows), "state_tolerance": args.state_tolerance, "max_state_l2_error": float(errors.max()), "mean_state_l2_error": float(errors.mean()), "per_horizon_max_state_l2_error": {key: float(np.max(values)) for key, values in by_horizon.items()}},
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"output": str(args.output_dir), "status": report["status"], "summary": report["summary"]}, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("PointMaze replay contract failed: P1/P2 physical labels are invalid pending a simulator/action-layout repair")


if __name__ == "__main__":
    main()
