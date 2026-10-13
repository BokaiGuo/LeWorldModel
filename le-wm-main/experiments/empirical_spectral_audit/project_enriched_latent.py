"""Project a canonical enriched export into a declared latent sub-chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload, make_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    payload = load_payload(args.input)
    dimension = int(payload["latent_dimension"])
    indices = [int(index) for index in args.indices]
    if not indices or min(indices) < 0 or max(indices) >= dimension:
        raise ValueError("indices must be non-empty and within the source latent dimension")
    rows = json.loads(json.dumps(payload["enriched_records"]))
    for row in rows:
        row["source_latent"] = [row["source_latent"][index] for index in indices]
        for item in row["latent_by_horizon"].values():
            item["predicted"] = [item["predicted"][index] for index in indices]
            item["target"] = [item["target"][index] for index in indices]
    split = dict(payload["split"])
    split["latent_projection"] = indices
    protocol = dict(payload.get("protocol", {}))
    protocol["latent_projection"] = indices
    result = make_payload(
        architecture=payload["architecture"], checkpoint=payload["checkpoint"],
        environment=payload["environment"], seed=payload["seed"],
        training_stage=payload["training_stage"], latent_dimension=len(indices),
        action_distribution=payload["action_distribution"], split=split,
        horizons=payload["horizons"], enriched_records=rows,
        checkpoint_sha256=payload.get("checkpoint_sha256"),
        dataset_sha256=payload.get("dataset_sha256"), protocol=protocol,
        study_scope=payload["study_scope"], claim_boundary=payload["claim_boundary"],
        runtime=payload["runtime"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"input": str(args.input), "output": str(args.output), "latent_dimension": len(indices)}))


if __name__ == "__main__":
    main()
