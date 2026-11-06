"""Replay prediction journals and calculate root-paired budget curves."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_good_chain_budget_curve_dev_20261105_codex as run

OUT, fixed = run.OUT, run.fixed


def main():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    fixed.prior.prior.prior.seal.verify(manifest["dependencies"])
    fixed.prior.prior.prior.seal.verify(manifest["source_dependencies"])
    if len(summary["rows"]) != 16*2*len(run.BUDGETS)*len(run.ARMS):
        raise ValueError("wrong row count")
    previous_rows = json.loads((run.prior.OUT / "summary.json").read_text())["rows"]
    previous = {(r["root"], r["repeat"], r["arm"]): r for r in previous_rows}
    checked, v2_matches = 0, 0
    for row in summary["rows"]:
        root, repeat, budget, arm = (row[k] for k in ("root", "repeat", "budget", "arm"))
        name = f"r{root:02d}_rep{repeat}_n{budget}_{arm}"
        if row != json.loads((OUT / f"{name}.json").read_text()) or row["status"] != "completed":
            raise ValueError("row mismatch or failure")
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = fixed.starts_from_result(source["result"])[0][0]
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        log = OUT / f"{name}_calls"
        best, best_action, tokens = float("inf"), None, 0
        first = None
        for j in range(row["calls"]):
            stem = f"{j:06d}"
            request = json.loads((log / f"{stem}_attempt.json").read_text())
            response = json.loads((log / f"{stem}_returned.json").read_text())
            action_path, result_path = log / request["actions"], log / response["result"]
            if run.prior.digest(action_path) != request["sha256"] or run.prior.digest(result_path) != response["sha256"]:
                raise ValueError("journal hash mismatch")
            with np.load(action_path, allow_pickle=False) as data:
                actions = data["actions"].copy()
            with np.load(result_path, allow_pickle=False) as data:
                features, scores = data["features"].copy(), data["scores"].copy()
            replay_features, replay_scores = predictor(actions)
            np.testing.assert_allclose(features, replay_features, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, replay_scores, rtol=0, atol=1e-12)
            if first is None:
                first = actions[0].copy()
            k = int(np.argmin(scores))
            if scores[k] < best:
                best, best_action = float(scores[k]), actions[k].copy()
            tokens += len(actions)*3
        if (tokens != budget*3 or row["tokens"] != tokens or not np.array_equal(first, parent)
                or abs(best-row["result"]["final_score"]) > 1e-12):
            raise ValueError("budget, parent, or incumbent mismatch")
        returned = np.asarray(row["result"]["final_actions"], np.float32)
        _, returned_score = predictor(returned[None])
        if abs(float(returned_score[0]) - best) > 1e-12:
            raise ValueError("returned score mismatch")
        if budget == 157:
            old = previous[(root, repeat, f"{arm}_total")]
            if (old["status"] != "completed" or old["result"]["final_score"] != row["result"]["final_score"]
                    or old["result"]["final_actions"] != row["result"]["final_actions"]):
                raise ValueError("prior 471-token result mismatch")
            v2_matches += 1
        checked += 1
    curves, pairwise = {}, {}
    for budget in run.BUDGETS:
        curves[str(budget*3)] = {}
        root_scores = {}
        for arm in run.ARMS:
            rows = [r for r in summary["rows"] if r["budget"] == budget and r["arm"] == arm]
            values = np.array([r["result"]["final_score"] for r in rows]).reshape(16, 2)
            root_scores[arm] = values.mean(axis=1)
            curves[str(budget*3)][arm] = {"mean_final": float(values.mean()),
                "success_over_32": int((values < .05).sum()),
                "roots_success_both": int(np.all(values < .05, axis=1).sum()),
                "mean_cpu_journal_seconds": float(np.mean([r["elapsed_seconds"] for r in rows]))}
        for arm in ("random", "cem"):
            delta = root_scores["response"] - root_scores[arm]
            pairwise[f"{budget*3}_response_minus_{arm}"] = {"mean": float(delta.mean()),
                "response_better_roots": int((delta < -1e-12).sum()),
                "response_worse_roots": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()), "per_root": delta.tolist()}
    result = {"status": "passed", "rows": checked, "v2_471_exact_matches": v2_matches,
              "summary_sha256": run.prior.digest(OUT / "summary.json"),
              "auditor_sha256": run.prior.digest(Path(__file__)), "curves": curves, "paired": pairwise,
              "scope": "same predictor implementation; finite budget and exposed tasks"}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
