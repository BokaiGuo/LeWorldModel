"""Decision-ranking audit on an already exposed privileged contact bank."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_jepa_v22_contact_latent_error_20261105_codex as latent

ROOT = latent.ROOT
OUT = ROOT / "results/pc_jepa_privileged_bank_success_rank_posthoc_v2_20261105_codex"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_group(indices, scores, success):
    indices = np.asarray(indices, dtype=int)
    scores, success = np.asarray(scores), np.asarray(success, dtype=bool)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("invalid candidate set")
    if not np.isfinite(scores[indices]).all():
        raise ValueError("nonfinite score")
    order = sorted(indices.tolist(), key=lambda i: (float(scores[i]), i))
    successful = [i for i in order if success[i]]
    first = successful[0] if successful else None
    if first is None:
        best_rank = None
        success_margin = None
    else:
        # Count ties explicitly; do not imply a unique ordering within them.
        best_rank = {"strictly_better_count": int(sum(scores[i] < scores[first] for i in order)),
                     "tied_count": int(sum(scores[i] == scores[first] for i in order))}
        success_margin = float(scores[first] - scores[order[0]])
    return {"selected_index": order[0], "selected_success": bool(success[order[0]]),
            "selected_score": float(scores[order[0]]), "best_success_index": first,
            "best_success_rank": best_rank, "best_success_score_margin": success_margin,
            "successes_in_top_1_3_5": [int(sum(success[i] for i in order[:k])) for k in (1, 3, 5)],
            "order": order}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    latent.verify()
    source, contact = latent.OUT, latent.SOURCE
    result = json.loads((source / "report.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    physical = json.loads((contact / "physical.json").read_text())
    contact_audit = json.loads((contact / "audit.json").read_text())
    assert result["status"] == "completed" and audit["status"] == "passed"
    assert physical["status"] == "completed" and contact_audit["status"] == "passed"
    assert contact_audit["physical_sha256"] == sha(contact / "physical.json")
    assert physical["arrays_sha256"] == sha(contact / "physical_arrays.npz")
    assert result["latents_sha256"] == sha(source / "output/latents.npz")
    assert len(result["rows"]) == len(physical["rows"]) == 64
    with np.load(source / "output/latents.npz", allow_pickle=False) as data:
        pred, actual, goal = (data[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
    predicted_scores = ((pred - goal) ** 2).mean(axis=(1, 2))
    actual_scores = ((actual - goal) ** 2).mean(axis=(1, 2))
    success = np.asarray([r["metrics"]["success"] for r in physical["rows"]], dtype=bool)
    for i, (p, row) in enumerate(zip(physical["rows"], result["rows"], strict=True)):
        assert row["index"] == p["index"] == i
        assert row["physical"] == p["metrics"]
        assert np.isclose(row["predicted_goal_mse"], predicted_scores[i], rtol=1e-12, atol=1e-12)
        assert np.isclose(row["actual_goal_mse"], actual_scores[i], rtol=1e-12, atol=1e-12)
    groups = {"all_64": np.arange(64), "witness_prefix_all_16": np.arange(48, 64),
              "witness_prefix_construction_7": np.arange(49, 56),
              "witness_prefix_validation_8": np.arange(56, 64)}
    rows = {}
    for name, indices in groups.items():
        rows[name] = {"count": len(indices), "success_count": int(success[indices].sum()),
                      "predicted": rank_group(indices, predicted_scores, success),
                      "actual_encoded": rank_group(indices, actual_scores, success)}
    files = [Path(__file__), source / "manifest.json", source / "report.json", source / "audit.json",
             source / "output/latents.npz", source / "output/receipt.json", contact / "physical.json",
             contact / "physical_arrays.npz", contact / "audit.json"]
    report = {"status": "completed", "groups": rows, "hashes": {str(p): sha(p) for p in files},
              "scope": "posthoc single exposed root; privileged candidates evaluation only; all queried"}
    OUT.mkdir()
    with (OUT / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
