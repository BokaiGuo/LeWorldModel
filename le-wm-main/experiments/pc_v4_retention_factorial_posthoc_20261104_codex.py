"""Post-hoc factorial reconstruction on sealed paired candidate outcomes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_paired_retention_quality_20261104_codex as prior
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_RETENTION_FACTORIAL_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v4_retention_factorial_posthoc_20261104_codex.json"


def retain(scores, responses, positions, local, feature):
    values = np.asarray([r["linear_reach_value"] for r in responses])
    features = (np.asarray([np.asarray(r["matrix"]).reshape(-1) for r in responses])
                if feature == "response" else np.asarray(positions))
    chosen = [int(np.argmin(scores))]
    best = int(np.argmin(values))
    if local and best not in chosen:
        chosen.append(best)
    while len(chosen) < 4:
        d = np.linalg.norm(features[:, None] - features[chosen][None], axis=2).min(axis=1)
        d[chosen] = -np.inf
        chosen.append(int(np.argmax(d)))
    return chosen


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if json.loads(prior.AUDIT.read_text())["plans_sha256"] != prior.sha(prior.DATA):
        raise ValueError("paired results do not match audit")
    prior.seal.verify(json.loads(prior.MANIFEST.read_text())["seal"])
    pairs = prior.source()
    data = json.loads(prior.DATA.read_text())
    dependencies = seal.capture([Path(__file__), PROTOCOL, prior.DATA, prior.AUDIT,
                                 prior.MANIFEST, *prior.PLANS, *prior.AUDITS, prior.RULES])
    rows = []
    for data_row, (source, other) in zip(data["rows"], pairs, strict=True):
        r = source["result"]
        requested = [c["prefix"] for c in r["first_bank"]]
        batches = [b for b in source["trace"] if b["requested"] == requested]
        if len(batches) != 1 or source["root"] != data_row["root"]:
            raise ValueError("source position or root mismatch")
        if (data_row["task"] != source["task"] or data_row["retained"] != [r["retained"], other["result"]["retained"]]
                or [c["prefix"] for c in data_row["candidates"]] != requested):
            raise ValueError("paired candidate linkage mismatch")
        positions = batches[0]["positions"]
        arms = {}
        for local in (False, True):
            for feature in ("response", "position"):
                key = f"{'local' if local else 'no_local'}_{feature}"
                chosen = retain(r["prefix_scores"], r["successor_responses"], positions, local, feature)
                scores = np.asarray([[x["score"] for x in c["repeats"]] for c in data_row["candidates"]])
                arms[key] = {"retained": chosen, "per_seed_best": scores[chosen].min(axis=0).tolist()}
        if (arms["local_response"]["retained"] != r["retained"]
                or arms["no_local_position"]["retained"] != other["result"]["retained"]):
            raise ValueError("existing rules cannot be reconstructed")
        rows.append({"root": source["root"], "greedy": int(np.argmin(r["prefix_scores"])),
                     "local_best": int(np.argmin([x["linear_reach_value"] for x in r["successor_responses"]])),
                     "arms": arms})
    if len(rows) != 24 or {r["root"] for r in rows} != set(range(24)):
        raise ValueError("incomplete roots")
    summary = {}
    for key in rows[0]["arms"]:
        values = np.asarray([r["arms"][key]["per_seed_best"] for r in rows])
        root_means = values.mean(axis=1)
        summary[key] = {"mean": float(root_means.mean()), "median": float(np.median(root_means)),
                        "root_mean_under_005": int((root_means < .05).sum()),
                        "replicates_under_005": int((values < .05).sum())}
    seal.verify(dependencies)
    prior.write(OUT, {"scope": "post-hoc fixed-bank exploratory analysis", "seal": dependencies,
                      "summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
