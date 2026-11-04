"""Independent loop-based reconstruction of post-hoc factorial analysis."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_retention_factorial_posthoc_20261104_codex as target
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUTPUT = target.OUT.with_name(target.OUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(target.OUT.read_text())
    seal.verify(data["seal"])
    paired = json.loads(target.prior.DATA.read_text())
    pairs = target.prior.source()
    arm_values = {k: [] for k in data["summary"]}
    near_ties = []
    for row, prev, (source, _) in zip(data["rows"], paired["rows"], pairs, strict=True):
        r = source["result"]
        request = [item["prefix"] for item in r["first_bank"]]
        batches = [b for b in source["trace"] if b["requested"] == request]
        if len(batches) != 1 or len(prev["candidates"]) != 9 or row["root"] != source["root"]:
            raise ValueError("source mismatch")
        positions = batches[0]["positions"]
        responses = r["successor_responses"]
        greedy = min(range(9), key=lambda i: (r["prefix_scores"][i], i))
        best = min(range(9), key=lambda i: (responses[i]["linear_reach_value"], i))
        if row["greedy"] != greedy or row["local_best"] != best:
            raise ValueError("seed mismatch")
        for local in (False, True):
            for feature in ("response", "position"):
                key = f"{'local' if local else 'no_local'}_{feature}"
                features = [np.asarray(responses[i]["matrix"]).ravel() if feature == "response"
                            else np.asarray(positions[i]) for i in range(9)]
                chosen = [greedy]
                if local and best != greedy:
                    chosen.append(best)
                actual = row["arms"][key]
                while len(chosen) < 4:
                    remaining = [i for i in range(9) if i not in chosen]
                    distances = {i: min(float(np.linalg.norm(features[i] - features[j])) for j in chosen)
                                 for i in remaining}
                    independently_chosen = max(remaining, key=lambda i: (distances[i], -i))
                    saved = actual["retained"][len(chosen)]
                    if saved not in distances:
                        raise ValueError("duplicate or invalid saved index")
                    gap = distances[independently_chosen] - distances[saved]
                    if gap > 1e-12:
                        raise ValueError(f"distance maximum mismatch root={row['root']} arm={key}: {gap}")
                    if saved != independently_chosen:
                        near_ties.append({"root": row["root"], "arm": key, "gap": gap,
                                          "saved": saved, "independent": independently_chosen})
                    chosen.append(saved)
                if actual["retained"] != chosen:
                    raise ValueError("seed or order mismatch")
                values = [min(prev["candidates"][i]["repeats"][rep]["score"] for i in chosen)
                          for rep in range(3)]
                np.testing.assert_array_equal(values, actual["per_seed_best"])
                arm_values[key].append(values)
    if len(data["rows"]) != 24:
        raise ValueError("incomplete panel")
    for key, values in arm_values.items():
        x = np.asarray(values)
        root_means = x.mean(axis=1)
        expected = data["summary"][key]
        np.testing.assert_allclose([root_means.mean(), np.median(root_means)],
                                   [expected["mean"], expected["median"]], rtol=0, atol=1e-15)
        if int((root_means < .05).sum()) != expected["root_mean_under_005"] or int((x < .05).sum()) != expected["replicates_under_005"]:
            raise ValueError("summary mismatch")
    seal.verify(data["seal"])
    target.prior.write(OUTPUT, {"analysis_sha256": target.prior.sha(target.OUT),
                                "audit_script_sha256": target.prior.sha(Path(__file__)),
                                "roots": 24, "sets_rebuilt": 96, "scored_root_repeats": 288,
                                "distance_tolerance": 1e-12, "near_ties": near_ties})
    print("Rebuilt 96 sets and 288 root-repeat arm scores.")


if __name__ == "__main__":
    main()
