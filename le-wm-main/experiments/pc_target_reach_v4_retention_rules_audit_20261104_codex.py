"""Independently reconstruct both frozen v4 retention rules."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_target_reach_v4_e0_20261104_codex as full
from experiments import pc_target_reach_v4_position_retention_e0_20261104_codex as alt

OUTPUT = alt.PREFIX.with_name(alt.PREFIX.name + "_rules_audit.json")


def reconstruct(scores, positions, responses, use_local):
    chosen = [int(np.argmin(scores))]
    if use_local:
        index = int(np.argmin([r["linear_reach_value"] for r in responses]))
        if index not in chosen:
            chosen.append(index)
        features = np.asarray([np.asarray(r["matrix"]).reshape(-1) for r in responses])
    else:
        features = np.asarray(positions)
    while len(chosen) < min(4, len(scores)):
        distances = np.linalg.norm(features[:, None] - features[chosen][None], axis=2).min(axis=1)
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return chosen


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = full.source.prior.base.sha
    paths = [p.with_name(p.name + "_plans.json") for p in (full.PREFIX, alt.PREFIX)]
    audits = [p.with_name(p.name + "_audit.json") for p in (full.PREFIX, alt.PREFIX)]
    datasets = [json.loads(p.read_text()) for p in paths]
    for path, audit in zip(paths, audits, strict=True):
        if json.loads(audit.read_text())["plans_sha256"] != sha(path):
            raise ValueError("prediction audit binding mismatch")
    for a, b in zip(datasets[0]["rows"], datasets[1]["rows"], strict=True):
        if a["root"] != b["root"] or a["task"] != b["task"]:
            raise ValueError("task mismatch")
        ar, br = a["result"], b["result"]
        if (ar["first_bank"] != br["first_bank"] or ar["successor_responses"] != br["successor_responses"]
                or ar["prefix_scores"] != br["prefix_scores"]):
            raise ValueError("pre-retention input mismatch")
        requested = [item["prefix"] for item in ar["first_bank"]]
        matches = [batch for batch in a["trace"] if batch["requested"] == requested]
        if len(matches) != 1:
            raise ValueError("cannot identify candidate position batch")
        positions = matches[0]["positions"]
        if (reconstruct(ar["prefix_scores"], positions, ar["successor_responses"], True) != ar["retained"]
                or reconstruct(br["prefix_scores"], positions, br["successor_responses"], False) != br["retained"]):
            raise ValueError("retention rule mismatch")
    if len(datasets[0]["rows"]) != 24 or len(datasets[1]["rows"]) != 24:
        raise ValueError("incomplete panel")
    full.source.prior.base.write(OUTPUT, {"plans_sha256": [sha(p) for p in paths], "verified_roots": 24,
                                         "scope": "retention reconstruction from recorded response and position"})
    print("Both retention rules independently reconstructed on 24 roots.", flush=True)


if __name__ == "__main__":
    main()
