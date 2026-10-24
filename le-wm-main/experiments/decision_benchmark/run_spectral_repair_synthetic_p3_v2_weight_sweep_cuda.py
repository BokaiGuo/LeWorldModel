"""Run a declared CUDA-only task-metric sweep for the corrected P3v2 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.decision_benchmark.run_spectral_repair_synthetic_p3_v2_cuda import run


WEIGHTS = (64.0, 256.0, 1024.0, 4096.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=202610240)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P3v2 task-metric sweep: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for weight in WEIGHTS:
        child = args.output / f"task_weight_{int(weight)}"
        # Common random numbers isolate the task metric.  Advancing the seed
        # by weight confounds a purported decision-value effect with a new
        # synthetic action/candidate world.
        result = run(output=child, device=args.device, seed=int(args.seed), task_weight=weight)
        summary = result["summary"]
        records.append({
            "task_output_0_weight": weight,
            "seed": int(args.seed),
            "status": result["status"],
            "decision_weighted_rank1_regret": summary["decision_weighted_rank1"]["mean_realized_regret"],
            "ordinary_svd_rank1_regret": summary["ordinary_svd_rank1"]["mean_realized_regret"],
            "pca_axis_rank1_regret": summary["pca_axis_rank1"]["mean_realized_regret"],
            "random_rank1_regret": summary["random_rank1"]["mean_realized_regret"],
        })
    payload = {
        "schema": "spectral_repair_synthetic_p3_v2_task_metric_sweep_cuda.v2",
        "status": "completed_task_metric_mechanism_sweep",
        "weights": list(WEIGHTS),
        "common_random_numbers": True,
        "records": records,
        "claim_boundary": "All declared task-metric settings are reported on the same synthetic action/candidate world. This controlled CUDA calibration locates the regime in which the low-energy mode is actually task-critical; it is not a learned-world-model repair result.",
    }
    (args.output / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
