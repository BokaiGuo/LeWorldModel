"""Attribute candidate-generation, retention, refinement and selection losses."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_discovery_attribution_dev as panel


def main():
    output = panel.OUT / "loss_attribution_posthoc.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    rows = []
    for root in range(len(panel.prior.tasks())):
        for arm in panel.ARMS:
            record = json.loads((panel.OUT / f"plan_{root:02d}_{arm}.json").read_text())
            if record["status"] != "completed":
                rows.append({"root": root, "arm": arm, "status": record["status"]})
                continue
            result = record["result"]
            if arm == "global_cem":
                rows.append({"root": root, "arm": arm, "status": "baseline_no_recursive_layers",
                             "selected_score": result["selected_score"],
                             "starts": [{"source": "global_bank", "seed_score": result["bank_scores"][i]}
                                        for i in result["bank_indices"]]})
                continue
            child = result["child"]
            layers = []
            for layer in child["layers"]:
                candidates = layer["candidates"]
                scores = np.asarray([item["score"] for item in candidates], float)
                retained = set(layer["retained"])
                best_i = int(np.argmin(scores))
                retained_best = min(scores[list(retained)])
                layers.append({
                    "depth": layer["depth"], "candidate_count": len(candidates),
                    "candidate_best": float(scores[best_i]),
                    "retained_best": float(retained_best),
                    "retained_best_gap": float(retained_best - scores[best_i]),
                    "best_retained": best_i in retained,
                    "response_added": layer["response_added"],
                    "response_removed": layer["response_removed"],
                    "reachability_count": len(layer["reachability"]),
                    "probe_shortlist": layer["probe_shortlist"],
                    "retained": layer["retained"],
                })
            starts = child["refinement_starts"]
            rows.append({
                "root": root, "arm": arm, "status": "completed",
                "selected_score": result["selected_score"],
                "incumbent_source": child["incumbent_source"],
                "global_choice_score": result["global_proposals"]["scores"][
                    result["global_proposals"]["choice"]],
                "layers": layers,
                "refinement_starts": [{"source": item["source"], "seed_score": item["seed_score"],
                                      "best_score": item["best_score"],
                                      "improvement": item["seed_score"] - item["best_score"]}
                                     for item in starts],
                "selected_from_refinement": min(item["best_score"] for item in starts),
            })
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(output, {"status": "completed",
        "scope": "posthoc sealed-record loss attribution; no gate reclassification",
        "source_plans_sha256": panel.seal.digest(panel.OUT / "plans_audit.json"),
        "rows": rows})
    print(f"wrote {len(rows)} loss-attribution records", flush=True)


if __name__ == "__main__":
    main()
