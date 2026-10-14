"""Run an ATM-style audit on an existing enriched latent export.

This is post-hoc analysis: it never re-runs the learned model and therefore
does not silently move learned inference to CPU.  It requires source latent,
true dynamics, predicted dynamics, and actions.  Candidate/root identifiers
are required for the separate finite-horizon ranking audit; without them that
part is explicitly unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.atm_cross_domain import inverse_risk_matrix


def audit_export(path: Path, *, max_samples: int = 20000, seed: int = 20261014) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"latents", "dynamics", "predicted_dynamics", "actions"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"latent export missing required fields: {missing}")
        arrays = {name: np.asarray(data[name]) for name in required}
        n = len(arrays["latents"])
        if any(len(value) != n for value in arrays.values()):
            raise ValueError("latent export fields have inconsistent row counts")
        rng = np.random.default_rng(int(seed))
        indices = np.arange(n) if not max_samples or n <= max_samples else np.sort(rng.choice(n, max_samples, replace=False))
        source = arrays["latents"][indices]
        true_successor = source + arrays["dynamics"][indices]
        predicted_successor = source + arrays["predicted_dynamics"][indices]
        actions = arrays["actions"][indices]
    fit = np.zeros(len(indices), dtype=bool)
    fit[: max(1, int(0.6 * len(indices)))] = True
    evaluation = ~fit
    report = inverse_risk_matrix(
        source,
        true_successor,
        predicted_successor,
        actions,
        fit_mask=fit,
        eval_mask=evaluation,
    )
    report.update(
        {
            "artifact": "atm_cross_domain_export_audit",
            "input": str(path.resolve()),
            "sample_count": int(len(indices)),
            "fit_count": int(np.sum(fit)),
            "evaluation_count": int(np.sum(evaluation)),
            "sampling": {"seed": int(seed), "max_samples": int(max_samples), "without_replacement": True},
            "ranking_audit": {
                "status": "unavailable_missing_candidate_root_identifiers",
                "required_fields": ["candidate_id", "root_id", "real_task_cost", "predicted_task_cost"],
                "claim_boundary": "the four inverse risks do not establish candidate-wise finite-horizon ranking",
            },
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    report = audit_export(args.input, max_samples=args.max_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"matrix": report["inverse_risk_matrix"], "ranking": report["ranking_audit"]}, indent=2))


if __name__ == "__main__":
    main()
