"""Recompute every stored E0 call and verify budget, incumbent and shared probes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_multi_direction_goal_combination_dev_20261105_codex as exp


def audit():
    summary = json.loads((exp.OUT / "summary.json").read_text())
    manifest = json.loads((exp.OUT / "manifest.json").read_text())
    seal = exp.fixed.prior.prior.prior.seal
    seal.verify(manifest["dependencies"])
    seal.verify(manifest["source_dependencies"])
    rows = summary["rows"]
    if len(rows) != 192:
        raise ValueError("wrong row count")
    errors, probes = [], {}
    max_feature_err = max_score_err = 0.0
    for row in rows:
        name = f"root{row['root']:02d}_rep{row['repeat']}_{row['arm']}"
        try:
            if row["status"] != "completed":
                raise ValueError("incomplete run")
            path = exp.OUT / f"{name}_calls"
            action_paths = sorted(path.glob("*_actions.npz"))
            predictor = exp.fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][row["root"]])
            all_actions, all_scores = [], []
            for ap in action_paths:
                idx = ap.name.split("_")[0]
                with np.load(ap) as data:
                    actions = data["actions"]
                with np.load(path / f"{idx}_result.npz") as data:
                    features, scores = data["features"], data["scores"]
                replay_feature, replay_score = predictor(actions)
                max_feature_err = max(max_feature_err, float(np.max(np.abs(features - replay_feature))))
                max_score_err = max(max_score_err, float(np.max(np.abs(scores - replay_score))))
                if not np.allclose(features, replay_feature, rtol=1e-8, atol=1e-9):
                    raise ValueError("feature replay mismatch")
                if not np.allclose(scores, replay_score, rtol=1e-8, atol=1e-9):
                    raise ValueError("score replay mismatch")
                all_actions.extend(actions)
                all_scores.extend(scores)
            expected = 157 if row["arm"].endswith("_total") else (109 if row["arm"].startswith("probe") else 121)
            if len(all_scores) != expected or row["tokens"] != 3 * expected:
                raise ValueError("budget mismatch")
            final = np.asarray(row["result"]["final_actions"])
            final_score = row["result"]["final_score"]
            if abs(min(all_scores) - final_score) > 1e-8:
                raise ValueError("final score not scored incumbent")
            if not any(np.allclose(a, final, rtol=0, atol=1e-7) and abs(float(s) - final_score) < 1e-8
                       for a, s in zip(all_actions, all_scores)):
                raise ValueError("returned chain not scored")
            with np.load(action_paths[1]) as data:
                probe = data["actions"].copy()
            key = (row["root"], row["repeat"])
            if key in probes and not np.array_equal(probes[key], probe):
                raise ValueError("probe actions differ across arms")
            probes[key] = probe
        except Exception as exc:
            errors.append({"run": name, "error": repr(exc)})
    result = {"status": "passed" if not errors else "failed", "rows": len(rows), "errors": errors,
              "max_feature_replay_error": max_feature_err, "max_score_replay_error": max_score_err,
              "same_implementation_replay_not_independent": True}
    with (exp.OUT / "audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
