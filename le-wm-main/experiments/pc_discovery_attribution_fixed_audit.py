"""Audit fixed-opportunity calls and shared-start deterministic optimization."""
import json
import numpy as np

from experiments import pc_discovery_attribution_dev as source


def main():
    out = source.OUT
    output = out / "fixed_audit_v2.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((out / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    rows = []
    for root, task in enumerate(source.prior.tasks()):
        common_actions, common_starts, common_batches = None, None, None
        for arm in source.ARMS:
            row = json.loads((out / f"fixed_{root:02d}_{arm}.json").read_text())
            if row["status"] != "completed":
                rows.append({"root": root, "arm": arm, "status": row["status"]})
                continue
            plan = json.loads((out / f"plan_{root:02d}_{arm}.json").read_text())
            starts, labels = source.fixed_starts(root, arm, plan)
            np.testing.assert_array_equal(starts, row["initial_actions"])
            if labels != row["labels"]:
                raise ValueError("labels mismatch")
            if common_actions is None:
                common_actions, common_starts = starts[:2].copy(), row["starts"][:2]
            else:
                np.testing.assert_array_equal(common_actions, starts[:2])
                if common_starts != row["starts"][:2]:
                    raise ValueError("shared-start CEM differs")
            directory = out / f"fixed_{root:02d}_{arm}_calls"
            info = source.prior.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"] or row["tokens"] != source.SAMPLES * 3:
                raise ValueError("budget mismatch")
            calls = list(source.prior.old.previous.fixed.read_calls(directory))
            # Four starts receive 156 post-seed samples each, in 13 batches;
            # seed batch differs in its last two rows and is checked above.
            batches = [call[0] for call in calls[1:27]]
            if len(batches) != 26:
                raise ValueError("shared-start batches missing")
            if common_batches is None:
                common_batches = batches
            else:
                for left, right in zip(common_batches, batches, strict=True):
                    np.testing.assert_array_equal(left, right)
            for actions, features, scores in calls:
                path, _ = source.prior.old.OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], features, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(features-task["goal"], axis=1), atol=1e-12, rtol=0)
            replay = iter(calls)
            def predict(actions):
                saved, features, scores = next(replay)
                np.testing.assert_array_equal(actions, saved)
                return features, scores
            result = source.fixed_optimize(predict, starts, root)
            if (next(replay, None) is not None or result["score"] != row["score"] or
                    result["starts"] != row["starts"]):
                raise ValueError("CEM replay mismatch")
            np.testing.assert_array_equal(result["actions"], row["actions"])
            rows.append({"root": root, "arm": arm, "status": "completed", "score": row["score"],
                         "fixed_sha256": source.seal.digest(out / f"fixed_{root:02d}_{arm}.json")})
        print(f"audited fixed root={root}", flush=True)
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(output, {"status": "passed", "rows": rows,
        "scope": "same-environment geometry and deterministic replay, not independent replication"})


if __name__ == "__main__":
    main()
