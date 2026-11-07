"""Evaluation-only success/score decomposition for nine already-replayed chains."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL = ROOT / "results/pc_tworoom_hold_first_pool_physical_20261107_codex.json"
ALIGN = ROOT / "results/pc_tworoom_first_pool_triple_alignment_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_RGB_FIRST_POOL_SCORING_FAILURE_POSTHOC_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_first_pool_scoring_failure_posthoc_20261107_codex.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def evaluate(model, encoded, successes):
    model = np.asarray(model, dtype=float)
    encoded = np.asarray(encoded, dtype=float)
    success = np.asarray(successes, dtype=bool)
    if model.shape != (3, 3) or encoded.shape != (3, 3) or success.shape != (3, 3, 25):
        raise ValueError("invalid candidate shapes")
    if not np.isfinite(model).all() or not np.isfinite(encoded).all():
        raise ValueError("nonfinite costs")
    selected = {"model": int(np.argmin(model)), "actual_encoding": int(np.argmin(encoded))}
    ever = success.any(axis=2).reshape(-1)
    endpoint = success[:, :, -1].reshape(-1)
    return {"model_index": selected["model"], "encoded_index": selected["actual_encoding"],
            "ever_success": ever.tolist(), "endpoint_success": endpoint.tolist(),
            "ever_oracle": bool(ever.any()), "endpoint_oracle": bool(endpoint.any()),
            "ever_model": bool(ever[selected["model"]]), "ever_encoded": bool(ever[selected["actual_encoding"]]),
            "endpoint_model": bool(endpoint[selected["model"]]),
            "endpoint_encoded": bool(endpoint[selected["actual_encoding"]])}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    physical = json.loads(PHYSICAL.read_text())
    alignment = json.loads(ALIGN.read_text())
    if alignment["source_sha256"] != sha(PHYSICAL):
        raise ValueError("physical source hash mismatch")
    if len(physical["rows"]) != len(alignment["rows"]):
        raise ValueError("different root counts")
    print("Post-hoc nine-chain scoring decomposition; actual encodings are evaluation-only.", flush=True)
    rows = []
    for p, a in zip(physical["rows"], alignment["rows"], strict=True):
        if p["root"] != a["root"]:
            raise ValueError("root mismatch")
        traces = [[p["physical"][i][j]["success"] for j in range(3)] for i in range(3)]
        distances = np.asarray([[p["physical"][i][j]["distance"][-1] for j in range(3)] for i in range(3)])
        if not np.array_equal(distances, np.asarray(a["physical_distance"])):
            raise ValueError("physical endpoint mismatch")
        rows.append({"root": p["root"], "initial_success": p["initial_success"],
                     **evaluate(a["model_cost"], a["encoded_cost"], traces)})
    eligible = [r for r in rows if not r["initial_success"]]
    summary = {"roots_total": len(rows), "initial_success": len(rows) - len(eligible),
               "eligible_roots": len(eligible)}
    for when in ("ever", "endpoint"):
        for strategy in ("oracle", "model", "encoded"):
            summary[f"{when}_{strategy}"] = sum(r[f"{when}_{strategy}"] for r in eligible)
        paired = [(r[f"{when}_model"], r[f"{when}_encoded"]) for r in eligible]
        summary[f"{when}_model_only"] = sum(m and not e for m, e in paired)
        summary[f"{when}_encoded_only"] = sum(e and not m for m, e in paired)
    print(json.dumps(summary, indent=2), flush=True)
    with OUT.open("x") as stream:
        json.dump({"scope": "post-hoc nine-chain subset; actual encoding evaluation-only",
                   "inputs_sha256": {str(p): sha(p) for p in (PHYSICAL, ALIGN, PROTOCOL, Path(__file__))},
                   "summary": summary, "rows": rows}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
