"""Attribute sealed multibranch search and final selection without new search."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_diverse_beam_exact_dev_20261103_codex as exp

PROTOCOL = exp.prior.ROOT.parent / "docs/PC_RESPONSE_DIVERSE_BEAM_TRACE_ATTRIBUTION_PROTOCOL_20261103_CODEX.md"
OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_trace_attribution.json")


def analyze(row):
    task, result = row["task"], row["result"]
    batches = [b for b in row["trace"] if np.asarray(b["requested"]).shape[1:] == (15, 2)]
    cursor, records = 0, []
    for candidate, program in zip(result["bank"], result["programs"], strict=True):
        if candidate["name"] != program["name"]:
            raise ValueError("candidate identity mismatch")
        remaining, chunks = program["samples"], []
        while remaining:
            if cursor >= len(batches):
                raise ValueError("missing suffix batch")
            batch = batches[cursor]
            size = len(batch["scores"])
            if size != min(12, remaining):
                raise ValueError("suffix allocation mismatch")
            requested, actions = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), actions)
            np.testing.assert_array_equal(requested[:, :10],
                                          np.broadcast_to(candidate["prefix"], requested[:, :10].shape))
            positions = exp.prior.exact.endpoints(task, actions)
            scores = np.linalg.norm(positions - task["goal"], axis=1)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            chunks.append((requested, positions, scores))
            cursor += 1
            remaining -= size
        actions = np.concatenate([x[0] for x in chunks])
        positions = np.concatenate([x[1] for x in chunks])
        scores = np.concatenate([x[2] for x in chunks])
        index = int(np.argmin(scores))
        np.testing.assert_array_equal(actions[index], program["chain"])
        np.testing.assert_allclose(scores[index], program["terminal_score"], atol=1e-12, rtol=0)
        prefix = np.asarray(candidate["prefix"], dtype=np.float32)
        prefix_position = exp.prior.exact.endpoints(task, prefix[None])[0]
        records.append({"name": candidate["name"], "prefix_position": prefix_position.tolist(),
                        "prefix_far": bool(prefix_position[0] > 0), "samples": len(scores),
                        "full_far": int(np.count_nonzero(positions[:, 0] > 0)),
                        "full_success": int(np.count_nonzero(scores < .05)),
                        "best_score": float(scores[index]), "best_chain": actions[index].tolist()})
    first = min(range(len(records)), key=lambda i: records[i]["best_score"])
    np.testing.assert_array_equal(records[first]["best_chain"],
                                  min(result["programs"], key=lambda p: p["terminal_score"])["chain"])
    if cursor >= len(batches):
        raise ValueError("missing final comparison")
    refinement = batches[cursor:-1]
    comparison = batches[-1]
    if len(comparison["scores"]) != 1:
        raise ValueError("comparison batch mismatch")
    np.testing.assert_array_equal(comparison["requested"][0], result["comparison_chain"])
    np.testing.assert_allclose(comparison["scores"][0], result["comparison_score"], atol=1e-12, rtol=0)
    all_scores, all_actions = [], []
    for batch in batches:
        requested, actions = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
        positions = exp.prior.exact.endpoints(task, actions)
        scores = np.linalg.norm(positions - task["goal"], axis=1)
        np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
        all_scores.extend(scores.tolist())
        all_actions.extend(requested.tolist())
    index = int(np.argmin(all_scores))
    selected_score = result["selected"]["terminal_score"]
    return {"root": row["root"], "candidates": records, "refinement_input_index": first,
            "refinement_count": sum(len(b["scores"]) for b in refinement),
            "total_full_samples": len(all_scores), "best_recorded_score": all_scores[index],
            "best_recorded_chain": all_actions[index], "selected_score": selected_score,
            "selection_gap": selected_score - all_scores[index],
            "selected_equals_best": bool(np.array_equal(result["selected"]["chain"], all_actions[index]))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = (Path(__file__), PROTOCOL, exp.PLANS, exp.AUDIT)
    hashes = {str(p): exp.prior.sha(p) for p in files}
    data = json.loads(exp.PLANS.read_text())
    if data["dependencies"] != exp.dependencies() or json.loads(exp.AUDIT.read_text())["plans_sha256"] != exp.prior.sha(exp.PLANS):
        raise ValueError("source seal mismatch")
    print("Replaying sealed candidate and full-chain samples; no new search.", flush=True)
    rows = [analyze(row) for row in data["rows"] if row["arm"] == "response_diverse"]
    if len(rows) != 24 or hashes != {str(p): exp.prior.sha(p) for p in files}:
        raise ValueError("source drift or unexpected panel")
    with OUTPUT.open("x") as stream:
        json.dump({"sources": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Reconstructed {len(rows)} decisions.", flush=True)


if __name__ == "__main__":
    main()
