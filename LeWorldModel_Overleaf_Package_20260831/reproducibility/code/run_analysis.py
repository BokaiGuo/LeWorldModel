"""Run the theory-only LeWM spectral-transfer simulation.

Example:
    python experiments/theory_spectral_transfer/run_analysis.py \
        --output results/theory_spectral_transfer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.theory_spectral_transfer import TransferConfig, spectral_transfer_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--intrinsic-dim", type=float, default=2.0)
    parser.add_argument("--input-exponent", type=float, default=1.0)
    parser.add_argument("--encoder-exponent", type=float, default=1.0)
    parser.add_argument("--tail-encoder-exponent", type=float, default=None)
    parser.add_argument("--breakpoint", type=int, default=None)
    parser.add_argument("--predictor-exponent", type=float, default=1.0)
    parser.add_argument("--heat-time", type=float, default=0.0)
    parser.add_argument("--heat-exponent", type=float, default=1.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--slope-window", type=int, nargs=2, action="append", metavar=("START", "STOP"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        encoder_exponent=args.encoder_exponent,
        tail_encoder_exponent=args.tail_encoder_exponent,
        breakpoint=args.breakpoint,
        predictor_exponent=args.predictor_exponent,
        heat_time=args.heat_time,
        heat_exponent=args.heat_exponent,
    )
    windows = args.slope_window or [(1, None)]
    report = spectral_transfer_report(config, slope_windows=windows, horizons=args.horizons)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
