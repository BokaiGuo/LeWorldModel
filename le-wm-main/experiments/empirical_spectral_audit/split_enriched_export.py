"""Create deterministic episode-disjoint fit/outcome exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonicalize_enriched_export import canonicalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--fit-fraction", type=float, default=0.6)
    args = parser.parse_args()
    if not 0.0 < args.fit_fraction < 1.0:
        parser.error("fit-fraction must be between 0 and 1")
    source = json.loads(args.source.read_text())
    rows = source.get("enriched_records")
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("at least four enriched rows are required for an episode-disjoint split")
    episodes = sorted({int(row["episode_id"]) for row in rows})
    split_at = max(1, min(len(episodes) - 1, int(round(len(episodes) * args.fit_fraction))))
    fit_ids, outcome_ids = episodes[:split_at], episodes[split_at:]
    for name, ids in (("fit", fit_ids), ("outcome", outcome_ids)):
        selected = [row for row in rows if int(row["episode_id"]) in set(ids)]
        temporary = args.output_root / f".{name}_source.json"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({"enriched_records": selected, "protocol": source.get("protocol", {})}))
        payload = canonicalize(
            temporary,
            architecture=str(source["architecture"]),
            checkpoint=str(source["checkpoint"]),
            environment=str(source["environment"]),
            seed=str(source["seed"]),
            training_stage=int(source["training_stage"]),
            action_distribution=str(source["action_distribution"]),
            split_name=name,
            episode_ids=ids,
        )
        target = args.output_root / f"{name}.json"
        target.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.unlink()
    (args.output_root / "split_manifest.json").write_text(json.dumps({"fit_episode_ids": fit_ids, "outcome_episode_ids": outcome_ids}, indent=2) + "\n")
    print(json.dumps({"fit": len(fit_ids), "outcome": len(outcome_ids), "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()

