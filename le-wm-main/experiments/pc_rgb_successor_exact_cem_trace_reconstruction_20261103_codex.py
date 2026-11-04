"""Reconstruct the sealed fixed-prefix CEM draws; no optimization is rerun."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_fixed_prefix_diag_20261103_codex as source

PROTOCOL = source.prior.ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_CEM_TRACE_RECONSTRUCTION_PROTOCOL_20261103_CODEX.md"
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_cem_trace_reconstruction.json")


def reconstruct(row, run):
    prefix = np.asarray(run["prefix"], dtype=float)
    witness = np.asarray(row["witness"], dtype=float)
    remaining = 15 - len(prefix)
    rng = np.random.default_rng(2026121500 + row["root"])
    loc = np.zeros((remaining, 2))
    scale = np.full((remaining, 2), 1 / 3)
    incumbent = float("inf")
    batches = []
    for index, batch in enumerate(run["trace"]):
        actions = np.asarray(batch["actions"], dtype=float)
        scores = np.asarray(batch["scores"], dtype=float)
        size = len(scores)
        if actions.shape != (size, 15, 2) or size != 12 or not np.isfinite(scores).all():
            raise ValueError("invalid recorded batch")
        expected = np.clip(loc + rng.normal(size=(size, remaining, 2)) * scale, -1, 1)
        expected[0] = loc
        np.testing.assert_array_equal(actions[:, :len(prefix)], np.broadcast_to(prefix, actions[:, :len(prefix)].shape))
        np.testing.assert_array_equal(actions[:, len(prefix):], expected)
        elite_indices = np.argsort(scores, kind="stable")[:4]
        incumbent = min(incumbent, float(scores.min()))
        entry = {"batch": index + 1, "scale_median": float(np.median(scale)),
                 "scale_min": float(np.min(scale)), "scale_lt_001_fraction": float(np.mean(scale < .01)),
                 "elite_indices": elite_indices.tolist(), "elite_scores": scores[elite_indices].tolist(),
                 "incumbent": incumbent}
        if run["name"] in ("witness_first5", "witness_first10"):
            delta = np.abs((witness[len(prefix):] - loc) / scale)
            entry.update(witness_z_median=float(np.median(delta)), witness_z_max=float(np.max(delta)))
        batches.append(entry)
        elite = expected[elite_indices]
        loc = elite.mean(axis=0)
        scale = np.maximum(elite.std(axis=0, ddof=0), 1e-5)
    if len(batches) != 32:
        raise ValueError("wrong batch count")
    np.testing.assert_allclose(incumbent, run["checkpoints"]["384"]["score"], atol=1e-12, rtol=0)
    return {"root": row["root"], "name": run["name"], "batches": batches}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sources = {str(p): source.prior.sha(p) for p in (Path(__file__), PROTOCOL, source.PLANS, source.AUDIT)}
    data = json.loads(source.PLANS.read_text())
    audit = json.loads(source.AUDIT.read_text())
    if data["dependencies"] != source.dependencies() or audit["plans_sha256"] != sources[str(source.PLANS)]:
        raise ValueError("source seal mismatch")
    if len(data["rows"]) != 24 or len(audit["rows"]) != 144:
        raise ValueError("unexpected source size")
    print("Reconstructing recorded draws and elite distributions; no new search.", flush=True)
    rows = [reconstruct(row, run) for row in data["rows"] for run in row["runs"]]
    if sources != {str(p): source.prior.sha(p) for p in (Path(__file__), PROTOCOL, source.PLANS, source.AUDIT)}:
        raise ValueError("source drift")
    with OUTPUT.open("x") as stream:
        json.dump({"sources": sources, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Reconstructed {len(rows)} searches and {sum(len(r['batches']) for r in rows)} batches.", flush=True)


if __name__ == "__main__":
    main()
