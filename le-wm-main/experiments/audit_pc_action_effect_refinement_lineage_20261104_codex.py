"""Post-hoc reconstruction of refinement start lineage from recorded calls."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_action_effect_ranked_end_to_end_dev_20261104_codex as panel


OUTPUT = panel.OUT / "refinement_lineage_audit.json"
ARMS = ("random_min", "random_effect", "spectral_min", "spectral_effect")


def reconstruct(row: dict, calls: list[tuple]) -> dict:
    result = row["result"]
    child = result["child"]
    starts = child["refinement_starts"]
    count = 1 + sum(len(s["generations"]) for s in starts)
    # The last call is the incumbent/comparison paired check; immediately
    # before it is the refinement seed batch followed by each start's batches.
    if len(calls) < count + 1:
        raise ValueError("missing refinement calls")
    refinement_calls = calls[-(count + 1):-1]
    seed_actions, _, seed_scores = refinement_calls[0]
    if len(seed_actions) != len(starts):
        raise ValueError("refinement seed count mismatch")
    np.testing.assert_allclose(seed_scores, [s["seed_score"] for s in starts], rtol=0, atol=1e-12)
    incumbent = np.asarray(child["incumbent"], dtype=np.float32)
    incumbent_score = float(child["paired_scores"][0])
    target = np.asarray(result["selected"], dtype=np.float32)
    per_start = []
    cursor = 1
    for index, start in enumerate(starts):
        seed = seed_actions[index]
        batches = refinement_calls[cursor:cursor + len(start["generations"])]
        cursor += len(start["generations"])
        matches = []
        if np.array_equal(seed, incumbent) and abs(seed_scores[index] - incumbent_score) < 1e-12:
            matches.append("seed")
        for j, (actions, _, scores) in enumerate(batches):
            if len(actions) != start["generations"][j]["samples"]:
                raise ValueError("generation size mismatch")
            if np.any(np.all(actions == incumbent, axis=(1, 2)) & (np.abs(scores-incumbent_score) < 1e-12)):
                matches.append(f"generation_{j}")
        per_start.append({
            "index": index, "source": start["source"], "samples": start["samples"],
            "seed_score": start["seed_score"], "best_score": start["best_score"],
            "incumbent_matches": matches,
        })
    if cursor != len(refinement_calls):
        raise ValueError("unconsumed refinement calls")
    if child["incumbent_source"] == "full_refine" and not any(s["incumbent_matches"] for s in per_start):
        raise ValueError("refinement incumbent not found in refinement calls")
    if not np.array_equal(target, calls[-1][0][int(child["abstained"])]):
        raise ValueError("returned action does not match paired call")
    return {
        "root": row["root"], "arm": row["arm"], "score": result["selected_score"],
        "abstained": result["abstained"], "incumbent_source": child["incumbent_source"],
        "starts": per_start,
        "selected_action_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    panel.prior.prior.seal.verify(manifest["dependencies"])
    prior_audit = json.loads((panel.OUT / "audit.json").read_text())
    if prior_audit["status"] != "passed" or prior_audit["run_count"] != 80:
        raise ValueError("full-panel call audit missing")
    rows = []
    hashes = {}
    for root in range(16):
        for arm in ARMS:
            path = panel.OUT / f"plan_{root:02d}_{arm}.json"
            row = json.loads(path.read_text())
            if (row["root"], row["arm"], row["status"]) != (root, arm, "completed"):
                raise ValueError("incomplete source")
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            calls = list(panel.prior.prior.old.previous.fixed.read_calls(
                panel.OUT / f"plan_{root:02d}_{arm}_calls"
            ))
            rows.append(reconstruct(row, calls))
    summaries = {}
    for direction in ("random", "spectral"):
        selected = [r for r in rows if r["arm"].startswith(direction)]
        start_counts = collections.Counter(s["source"] for r in selected for s in r["starts"])
        winner_counts = collections.Counter(s["source"] for r in selected for s in r["starts"]
                                            if s["incumbent_matches"])
        same = 0
        for root in range(16):
            left = next(r for r in selected if r["root"] == root and r["arm"].endswith("_min"))
            right = next(r for r in selected if r["root"] == root and r["arm"].endswith("_effect"))
            same += left["selected_action_sha256"] == right["selected_action_sha256"]
        summaries[direction] = {
            "runs": len(selected), "same_selected_action_pairs": same,
            "abstentions": sum(r["abstained"] for r in selected),
            "starts_by_source": dict(start_counts),
            "starts_matching_incumbent_by_source": dict(winner_counts),
            "incumbent_sources": dict(collections.Counter(r["incumbent_source"] for r in selected)),
        }
    output = {
        "schema": "pc_action_effect_refinement_lineage_audit.v1",
        "status": "completed", "scope": "post-hoc action-log lineage, not causal attribution",
        "source_row_sha256": hashes, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows, "summaries": summaries,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
