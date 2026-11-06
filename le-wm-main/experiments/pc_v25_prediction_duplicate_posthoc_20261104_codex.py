"""Audit repeated exact predictions in the sealed v25 same-chain plans."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as panel


def inspect(calls):
    seen = {}
    repeated_samples = repeated_tokens = distinct_samples = 0
    repeated_batches = 0
    for actions, features, scores in calls:
        batch_repeated = False
        for action, feature, score in zip(actions, features, scores, strict=True):
            key = (action.shape, action.dtype.str, action.tobytes())
            if key in seen:
                reference_feature, reference_score = seen[key]
                np.testing.assert_array_equal(feature, reference_feature)
                np.testing.assert_equal(score, reference_score)
                repeated_samples += 1
                repeated_tokens += action.shape[0] // 5
                batch_repeated = True
            else:
                seen[key] = (feature.copy(), float(score))
                distinct_samples += 1
        repeated_batches += int(batch_repeated)
    return {"distinct_samples": distinct_samples, "repeated_samples": repeated_samples,
            "repeated_prediction_tokens": repeated_tokens, "batches_with_repetition": repeated_batches}


def main():
    output = panel.OUT / "prediction_duplicate_posthoc.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    plans = json.loads((panel.OUT / "plans.json").read_text())
    audit = json.loads((panel.OUT / "plans_audit.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    if audit["plans_sha256"] != panel.seal.digest(panel.OUT / "plans.json"):
        raise ValueError("plans not audited")
    rows = []
    for row in plans["rows"]:
        if row["status"] != "completed":
            rows.append({"root": row["root"], "status": "failed"})
            continue
        calls = panel.fixed.read_calls(panel.OUT / f"plan_root{row['root']:02d}_calls")
        rows.append({"root": row["root"], "status": "completed", **inspect(calls)})
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(output, {"scope": "exact-action and exact-output duplicates within each fixed context",
        "plans_sha256": panel.seal.digest(panel.OUT / "plans.json"),
        "analysis_sha256": panel.seal.digest(Path(__file__)), "rows": rows})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
