"""Replay all recorded predictions and audit direction edit comparisons."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_cheap_direction_edit_dev_20261105_codex as run


def main():
    out, fixed, prior = run.OUT, run.fixed, run.prior
    output = out / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((out / "manifest.json").read_text())
    rows = json.loads((out / "summary.json").read_text())["rows"]
    fixed.prior.prior.prior.seal.verify(manifest["dependencies"])
    fixed.prior.prior.prior.seal.verify(manifest["source_dependencies"])
    if len(rows) != 256:
        raise ValueError("incomplete rows")
    previous_rows = json.loads((prior.OUT / "summary.json").read_text())["rows"]
    previous = {(r["root"], r["repeat"], r["arm"]): r for r in previous_rows}
    seen, matched = set(), 0
    for row in rows:
        root, repeat, arm = (row[k] for k in ("root", "repeat", "arm"))
        key = (root, repeat, arm)
        if key in seen:
            raise ValueError("duplicate row")
        seen.add(key)
        name = f"root{root:02d}_rep{repeat}_{arm}"
        if row != json.loads((out / f"{name}.json").read_text()) or row["status"] != "completed":
            raise ValueError("incomplete or divergent row")
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = fixed.starts_from_result(source["result"])[0][0]
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        log = out / f"{name}_calls"
        best, tokens, first = float("inf"), 0, None
        for j in range(row["calls"]):
            stem = f"{j:06d}"
            request = json.loads((log / f"{stem}_attempt.json").read_text())
            response = json.loads((log / f"{stem}_returned.json").read_text())
            action_path, result_path = log / request["actions"], log / response["result"]
            if prior.digest(action_path) != request["sha256"] or prior.digest(result_path) != response["sha256"]:
                raise ValueError("journal hash mismatch")
            with np.load(action_path, allow_pickle=False) as data:
                actions = data["actions"].copy()
            with np.load(result_path, allow_pickle=False) as data:
                features, scores = data["features"].copy(), data["scores"].copy()
            a, b = predictor(actions)
            np.testing.assert_allclose(features, a, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, b, rtol=0, atol=1e-12)
            if first is None:
                first = actions[0]
            best = min(best, float(np.min(scores)))
            tokens += len(actions)*3
        expected = {"cheap_fixed": 309, "response_fixed": 471, "random_fixed": 291,
                    "cem_fixed": 252}.get(arm, 471)
        if (tokens != expected or row["tokens"] != tokens or not np.array_equal(first, parent)
                or abs(best-row["result"]["final_score"]) > 1e-12):
            raise ValueError("parent, budget, or incumbent mismatch")
        _, final_score = predictor(np.asarray(row["result"]["final_actions"], np.float32)[None])
        if abs(float(final_score[0])-best) > 1e-12:
            raise ValueError("returned chain mismatch")
        if not arm.startswith("cheap"):
            old = previous[key]
            if (old["result"]["final_score"] != row["result"]["final_score"] or
                    old["result"]["final_actions"] != row["result"]["final_actions"]):
                raise ValueError("V2 control changed")
            matched += 1
    if seen != {(r, rep, a) for r in range(16) for rep in range(2) for a in run.ARMS}:
        raise ValueError("incomplete design")
    summaries, paired = {}, {}
    scores = {}
    for arm in run.ARMS:
        selected = sorted((r for r in rows if r["arm"] == arm), key=lambda x: (x["root"], x["repeat"]))
        values = np.array([r["result"]["final_score"] for r in selected]).reshape(16, 2)
        scores[arm] = values.mean(axis=1)
        summaries[arm] = {"mean_final": float(values.mean()), "success_over_32": int((values < .05).sum()),
                          "mean_seconds": float(np.mean([r["elapsed_seconds"] for r in selected])),
                          "tokens": selected[0]["tokens"]}
    for mode in ("fixed", "total"):
        for method in ("response", "random", "cem"):
            delta = scores[f"cheap_{mode}"] - scores[f"{method}_{mode}"]
            paired[f"cheap_minus_{method}_{mode}"] = {
                "mean": float(delta.mean()), "better_roots": int((delta < -1e-12).sum()),
                "worse_roots": int((delta > 1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()),
                "per_root": delta.tolist()}
    result = {"status": "passed", "rows": len(seen), "v2_exact_matches": matched,
              "summary_sha256": prior.digest(out / "summary.json"),
              "auditor_sha256": prior.digest(Path(__file__)), "summaries": summaries, "paired": paired,
              "scope": "same predictor implementation, exposed tasks, not independent confirmation"}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
