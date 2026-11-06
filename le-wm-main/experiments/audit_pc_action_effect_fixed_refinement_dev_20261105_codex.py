"""Re-evaluate and replay fixed-opportunity CEM journals."""
import hashlib
import json

import numpy as np

from experiments import pc_action_effect_fixed_refinement_dev_20261105_codex as run
from experiments import pc_prediction_journal_audit_v1 as journal_audit


def main():
    output = run.OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((run.OUT / "manifest.json").read_text())
    result = json.loads((run.OUT / "result.json").read_text())
    if result["status"] != "completed" or len(result["rows"]) != 64:
        raise ValueError("result incomplete")
    if manifest["code_sha256"] != run.digest(run.Path(run.__file__)):
        raise ValueError("runner changed")
    if manifest["optimizer_sha256"] != run.digest(run.Path(run.optimizer.__file__)):
        raise ValueError("optimizer changed")
    for name, digest in manifest["source_rows"].items():
        if run.digest(run.prior.OUT / name) != digest:
            raise ValueError("source plan changed")
    total = 0
    identities = set()
    for row in result["rows"]:
        root, arm = row["root"], row["arm"]
        if (root, arm) in identities:
            raise ValueError("duplicate run")
        identities.add((root, arm))
        task = manifest["tasks"][root]
        source = json.loads((run.prior.OUT / f"plan_{root:02d}_{arm}.json").read_text())
        starts, indices = run.starts_from_result(source["result"])
        np.testing.assert_array_equal(starts, np.asarray(row["initial_actions"], np.float32))
        if indices != row["global_indices"] or row["seed"] != 2046000000 + root:
            raise ValueError("source/seed mismatch")
        directory = run.OUT / f"root{root:02d}_{arm}_calls"
        info = journal_audit.audit_journal(directory)
        if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
            raise ValueError("journal budget mismatch")
        calls = list(run.prior.prior.prior.old.previous.fixed.read_calls(directory))
        iterator = iter(calls)
        predictor = run.prior.prior.prior.old.previous.fixed.predictor(task)

        def evaluate(actions):
            saved, features, scores = next(iterator)
            np.testing.assert_array_equal(actions, saved)
            actual_features, actual_scores = predictor(saved.copy())
            np.testing.assert_allclose(features, actual_features, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, actual_scores, atol=1e-12, rtol=0)
            return scores

        rebuilt = run.optimizer.optimize(
            evaluate, np.empty((0, 2), np.float32), starts, run.SAMPLES,
            seed=row["seed"], batch_size=12, elites=4,
        )
        if next(iterator, None) is not None:
            raise ValueError("extra journal call")
        if (rebuilt["score"] != row["score"] or rebuilt["actions"].tolist() != row["actions"]
                or rebuilt["starts"] != row["starts"]
                or min(r["best_score"] for r in rebuilt["starts"][2:]) != row["retained_best_score"]):
            raise ValueError("fixed optimization replay mismatch")
        total += info["tokens"]
    expected = {(r, a) for r in range(16) for a in run.ARMS}
    if identities != expected or total != result["diagnostic_prediction_tokens"]:
        raise ValueError("coverage or total cost mismatch")
    audit = {"status": "passed", "rows": len(identities), "prediction_tokens": total,
             "result_sha256": hashlib.sha256((run.OUT / "result.json").read_bytes()).hexdigest(),
             "scope": "same-workspace exact-predictor re-evaluation and optimizer replay, not independent confirmation"}
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
