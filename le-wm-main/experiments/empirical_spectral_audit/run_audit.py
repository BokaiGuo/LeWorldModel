"""Run the separated empirical LeWM spectral audit over a latents.npz file.

Example:
    python experiments/empirical_spectral_audit/run_audit.py \
        --latents results/pusht/lewm/latents.npz \
        --output results/pusht/lewm/empirical_spectral_audit.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit import EmpiricalAuditConfig, load_latent_arrays, run_empirical_audit
from experiments.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, help="path to latents.npz")
    parser.add_argument("--output", required=True, help="path to write empirical_spectral_audit.json")
    parser.add_argument("--condition", choices=["states", "actions", "latents", "goals", "episode", "phase", "global"], default="states")
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--min-cell-size", type=int, default=256)
    parser.add_argument("--band-mode", choices=["rank", "energy"], default="rank")
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1])
    parser.add_argument("--no-interventions", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="deterministic row cap for large latent NPZ files; 0 loads all")
    parser.add_argument("--seed", type=int, default=3072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = load_latent_arrays(args.latents, max_samples=args.max_samples, seed=args.seed)
    config = EmpiricalAuditConfig(
        condition=args.condition,
        clusters=args.clusters,
        ks=tuple(args.ks),
        min_cell_size=args.min_cell_size,
        band_mode=args.band_mode,
        horizons=tuple(args.horizons),
        seed=args.seed,
        include_interventions=not args.no_interventions,
    )
    report = run_empirical_audit(arrays, config)
    if args.max_samples:
        report["input_sampling"] = {
            "max_samples": int(args.max_samples),
            "num_loaded_samples": int(arrays["latents"].shape[0]),
            "seed": int(args.seed),
        }
    save_json(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
