"""Attribute nonprogress-prefix coverage in sealed multidirection search traces."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_multidirection_successor_exact_dev_20261103_codex as source

PROTOCOL = source.PROTOCOL.parent / "PC_MULTIDIRECTION_NONPROGRESS_COVERAGE_ATTRIBUTION_PROTOCOL_20261103_CODEX.md"
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_nonprogress_coverage_attribution.json")
EPS = 1e-10


def analyze(row):
    task, result = row["task"], row["result"]
    goal = np.asarray(task["goal"])
    initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))

    def prefix_stats(actions):
        model = np.asarray(actions, dtype=np.float32)
        positions = source.previous.prior.exact.endpoints(task, model[:, :5])
        distances = np.linalg.norm(positions - goal, axis=1)
        nonprogress = distances >= initial - EPS
        nonzero = np.any(model[:, :5] != 0, axis=(1, 2))
        return distances, nonprogress, nonzero

    first = np.asarray([item["prefix"] for item in result["first_bank"]], dtype=np.float32)
    retained = np.asarray([item["prefix"] for item in result["bank"]], dtype=np.float32)
    first_d, first_np, first_nz = prefix_stats(first)
    ret_d, ret_np, ret_nz = prefix_stats(retained)
    chains, recorded_scores, trace_locations = [], [], []
    for batch_index, batch in enumerate(row["trace"]):
        actions = np.asarray(batch["model"], dtype=np.float32)
        requested = np.asarray(batch["requested"])
        np.testing.assert_array_equal(requested.astype(np.float32), actions)
        if actions.shape[1] != 15:
            continue
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("invalid full chain")
        scores = np.asarray(batch["scores"])
        endpoints = source.previous.prior.exact.endpoints(task, actions)
        np.testing.assert_allclose(np.linalg.norm(endpoints - goal, axis=1), scores, rtol=0, atol=1e-12)
        chains.extend(actions)
        recorded_scores.extend(scores)
        trace_locations.extend((batch_index, i) for i in range(len(actions)))
    chains = np.asarray(chains, dtype=np.float32)
    scores = np.asarray(recorded_scores)
    distances, nonprogress, nonzero = prefix_stats(chains)
    eligible = nonprogress & nonzero
    best = int(np.argmin(scores))
    best_np = int(np.flatnonzero(eligible)[np.argmin(scores[eligible])]) if eligible.any() else None
    selected = np.asarray(result["selected"]["chain"], dtype=np.float32)
    selected_score = float(result["selected"]["terminal_score"])
    selected_dist, selected_np, selected_nz = prefix_stats(selected[None])
    matching = np.all(chains == selected, axis=(1, 2))
    if not matching.any() or not np.any(np.abs(scores[matching] - selected_score) <= 1e-12):
        raise ValueError("selected chain not present with sealed score")
    return {"root": row["root"], "task": task, "initial_distance": initial,
            "first_bank": {"distances": first_d.tolist(), "nonprogress": first_np.tolist(),
                           "nonzero": first_nz.tolist(), "retained_indices": result["retained"]},
            "retained_bank": {"distances": ret_d.tolist(), "nonprogress": ret_np.tolist(),
                              "nonzero": ret_nz.tolist()},
            "full": {"samples": len(scores), "nonprogress": int(nonprogress.sum()),
                     "nonzero_nonprogress": int(eligible.sum()), "scores": scores.tolist(),
                     "prefix_distances": distances.tolist(), "nonprogress_mask": nonprogress.tolist(),
                     "nonzero_mask": nonzero.tolist(), "trace_locations": trace_locations,
                     "best_index": best, "best_score": float(scores[best]),
                     "best_nonzero_nonprogress_index": best_np,
                     "best_nonzero_nonprogress_score": None if best_np is None else float(scores[best_np])},
            "selected": {"distance": float(selected_dist[0]), "nonprogress": bool(selected_np[0]),
                         "nonzero": bool(selected_nz[0]), "score": selected_score,
                         "score_gap_from_scored_best": selected_score - float(scores[best])}}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = source.sha
    paths = (Path(__file__), PROTOCOL, source.PLANS, source.AUDIT)
    hashes = {str(p): sha(p) for p in paths}
    data = json.loads(source.PLANS.read_text())
    audit = json.loads(source.AUDIT.read_text())
    if data["dependencies"] != source.dependencies() or audit["plans_sha256"] != sha(source.PLANS):
        raise ValueError("source seal mismatch")
    print("Attributing sealed nonprogress-prefix coverage and final selection; no new planning.", flush=True)
    rows = [analyze(row) for row in data["rows"]]
    if len(rows) != 24 or len({r["root"] for r in rows}) != 24:
        raise ValueError("panel mismatch")
    if hashes != {str(p): sha(p) for p in paths} or data["dependencies"] != source.dependencies():
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, allow_nan=False, separators=(",", ":"))
    print(f"Sealed attribution for {len(rows)} exposed tasks.", flush=True)


if __name__ == "__main__":
    main()
