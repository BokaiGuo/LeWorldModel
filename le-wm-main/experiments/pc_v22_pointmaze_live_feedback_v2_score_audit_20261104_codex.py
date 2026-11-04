"""Distinguish repeated-score drift from discarded distinct action chains."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_pointmaze_live_feedback_v2_20261104_codex as run

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_score_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    rows = []
    for arm in run.ARMS:
        decisions, _, _ = run.paths(arm)
        for step, path in enumerate(decisions):
            data = json.loads(path.read_text())
            selected = np.asarray(data["result"]["selected"]["chain"], dtype=np.float32)
            selected_score = float(data["result"]["selected"]["terminal_score"])
            values = [(float(score), np.array_equal(np.asarray(action, dtype=np.float32), selected))
                      for batch in data["trace"] if len(batch["raw"][0]) == len(selected)
                      for action, score in zip(batch["raw"], batch["scores"], strict=True)]
            if not values or not all(np.isfinite(score) for score, _ in values):
                raise ValueError("missing or nonfinite full-chain scores")
            best = min(score for score, _ in values)
            same = [score for score, equal in values if equal]
            distinct = [score for score, equal in values if not equal]
            if not same or selected_score not in same:
                raise ValueError("selected score absent from trace")
            rows.append({"arm": arm, "step": step, "selected_score": selected_score,
                         "minimum_scored": best, "minimum_scored_is_selected_action":
                         any(equal and score == best for score, equal in values),
                         "selected_repeat_score_range": [min(same), max(same)],
                         "best_distinct_score": min(distinct) if distinct else None,
                         "decision_sha256": sha(path)})
    run.old.write_json(OUTPUT, {"rows": rows, "script_sha256": sha(Path(__file__)),
                                "limitations": ["observational audit; does not identify numerical drift cause",
                                                "not an independent model prediction replay"]})
    print("scored-action audit:", [(r["arm"], r["step"], r["minimum_scored_is_selected_action"]) for r in rows])


if __name__ == "__main__":
    main()
