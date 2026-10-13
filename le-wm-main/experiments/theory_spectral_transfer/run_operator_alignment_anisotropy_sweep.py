"""Run a continuous anisotropy sweep for same-spectrum operator alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.theory_spectral_transfer import TransferConfig, operator_alignment_anisotropy_sweep_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/theory_operator_alignment_anisotropy_sweep_20260829",
        help="output directory for JSON and Markdown assets",
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--intrinsic-dim", type=float, default=2.0)
    parser.add_argument("--input-exponents", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--encoder-exponent", type=float, default=1.0)
    parser.add_argument("--predictor-exponent", type=float, default=1.0)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 15, 30, 45, 60, 75, 90])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--random-trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(output: Path, report: dict[str, object]) -> None:
    rows = report["rows"]
    lines = [
        "# Anisotropy Sweep",
        "",
        "## Object",
        "",
        "- continuous anisotropy sweep for same-spectrum operator alignment",
        "",
        "## Readout",
        "",
        "| input exponent | latent stable rank | peak commutator | peak angle | peak top-band lift |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["input_exponent"]),
                    _fmt(row["latent_stable_rank"]),
                    _fmt(row["peak_commutator_norm"]),
                    _fmt(row["peak_angle_degrees"]),
                    _fmt(row["peak_top_band_lift"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    (output / "operator_alignment_anisotropy_sweep.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponents[0] if args.input_exponents else 1.0,
        encoder_exponent=args.encoder_exponent,
        predictor_exponent=args.predictor_exponent,
    )
    report = operator_alignment_anisotropy_sweep_report(
        config=config,
        input_exponents=args.input_exponents,
        angles=args.angles,
        horizons=args.horizons,
        ks=args.ks,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    (output / "operator_alignment_anisotropy_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
