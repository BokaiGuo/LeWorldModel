"""Audit recorded budgets, returned actions, and treatment activation."""
import collections
import json
from pathlib import Path

import numpy as np

from experiments import pc_action_effect_end_to_end_dev_20261104_codex as panel
from experiments import pc_prediction_journal_audit_v1 as journal_audit


def main():
    output = panel.OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    panel.prior.prior.seal.verify(manifest["dependencies"])
    if manifest["tasks"] != panel.prior.tasks() or tuple(manifest["arms"]) != panel.ARMS:
        raise ValueError("manifest mismatch")
    outcomes = {}
    activation = collections.Counter()
    for root, task in enumerate(manifest["tasks"]):
        first = None
        for arm in panel.ARMS:
            row = json.loads((panel.OUT / f"plan_{root:02d}_{arm}.json").read_text())
            if (row["root"], row["arm"], row["task"], row["status"]) != (root, arm, task, "completed"):
                raise ValueError("failed or mismatched run")
            directory = panel.OUT / f"plan_{root:02d}_{arm}_calls"
            info = journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
                raise ValueError("cost mismatch")
            calls = list(panel.prior.prior.old.previous.fixed.read_calls(directory))
            if first is None:
                first = calls[0][0]
            elif not np.array_equal(first, calls[0][0]):
                raise ValueError("global proposal bank mismatch")
            selected = np.asarray(row["result"]["selected"], np.float32)
            selected_score = row["result"]["selected_score"]
            found = False
            best = float("inf")
            for actions, features, scores in calls:
                path, _ = panel.prior.prior.old.OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], features, rtol=0, atol=1e-12)
                np.testing.assert_allclose(scores, np.linalg.norm(features - task["goal"], axis=1), rtol=0, atol=1e-12)
                if actions.shape[1] == 15:
                    best = min(best, float(scores.min()))
                    found |= bool(np.any(np.all(actions == selected, axis=(1, 2)) & (np.abs(scores-selected_score) < 1e-12)))
            if not found or selected_score > best + 1e-12:
                raise ValueError("returned plan not a scored best complete chain")
            outcomes[(root, arm)] = row
        for direction in ("random", "spectral"):
            min_layers = outcomes[(root, direction + "_min")]["result"]["child"]["layers"]
            effect_layers = outcomes[(root, direction + "_effect")]["result"]["child"]["layers"]
            for left, right in zip(min_layers, effect_layers, strict=True):
                if left["retained"] != right["retained"]:
                    activation[direction + "_changed_layers"] += 1
                if right["depth"] == 2:
                    if right["reachability"]:
                        raise ValueError("terminal depth cannot have a remaining-tail probe")
                    continue
                records = right["reachability"]
                greedy = min(range(len(right["candidates"])), key=lambda i: right["candidates"][i]["score"])
                threshold = min(r["probe_min"] for r in records) + 0.05
                eligible = sum(r["candidate"] != greedy and r["probe_min"] <= threshold for r in records)
                activation[f"{direction}_qualifying_{eligible}"] += 1
    if len(outcomes) != len(manifest["tasks"]) * len(panel.ARMS):
        raise ValueError("run coverage mismatch")
    result = {
        "status": "passed",
        "scope": "same-environment log/dynamics/selection audit; not independent replication or full cost",
        "run_count": len(outcomes),
        "activation": dict(activation),
        "warning": "absence of treatment changes makes this panel non-identifying for retention",
    }
    panel.prior.prior.journal.write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
