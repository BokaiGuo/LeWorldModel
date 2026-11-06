"""Replay paired candidate records, reconstruct witnesses and summarize comparisons."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_paired_segment_repair_dev_20261105_codex as run


def main():
    out, old, fixed = run.OUT, run.old, run.fixed
    manifest = json.loads((out / "manifest.json").read_text())
    rows = json.loads((out / "summary.json").read_text())["rows"]
    seal = fixed.prior.prior.prior.seal
    seal.verify(manifest["dependencies"])
    seal.verify(manifest["source_dependencies"])
    prior_rows = json.loads((old.OUT / "summary.json").read_text())["rows"]
    cem = {(r["root"], r["repeat"]): r for r in prior_rows if r["arm"] == "cem_total"}
    seen, witness_runs, witness_pairs, selected, final_exact = set(), [], 0, 0, 0
    for row in rows:
        root, rep, arm = row["root"], row["repeat"], row["arm"]
        key = root, rep, arm
        if key in seen or row["status"] != "completed":
            raise ValueError("duplicate or failed row")
        seen.add(key)
        name = f"root{root:02d}_rep{rep}_{arm}"
        if row != json.loads((out / f"{name}.json").read_text()):
            raise ValueError("row mismatch")
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = fixed.starts_from_result(source["result"])[0][0]
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        singles, pairs, extras, segments = run.make_candidates(parent, row["proposal_seed"])
        log, tokens, best = out / f"{name}_calls", 0, float("inf")
        observed_scores = []
        for j in range(row["calls"]):
            stem = f"{j:06d}"
            request = json.loads((log / f"{stem}_attempt.json").read_text())
            response = json.loads((log / f"{stem}_returned.json").read_text())
            action_path, result_path = log / request["actions"], log / response["result"]
            if old.prior.digest(action_path) != request["sha256"] or old.prior.digest(result_path) != response["sha256"]:
                raise ValueError("journal hash mismatch")
            with np.load(action_path, allow_pickle=False) as f:
                actions = f["actions"].copy()
            with np.load(result_path, allow_pickle=False) as f:
                features, scores = f["features"].copy(), f["scores"].copy()
            if j < 3:
                expected = (parent[None], singles, pairs if arm == "pair" else extras)[j]
                np.testing.assert_array_equal(actions, expected)
                observed_scores.append(scores.copy())
            replay_f, replay_s = predictor(actions)
            np.testing.assert_allclose(features, replay_f, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, replay_s, rtol=0, atol=1e-12)
            tokens += 3*len(actions)
            best = min(best, float(scores.min()))
        if tokens != 471 or tokens != row["tokens"] or abs(best-row["result"]["final_score"]) > 1e-12:
            raise ValueError("budget or incumbent mismatch")
        result = row["result"]
        _, returned = predictor(np.asarray(result["final_actions"], np.float32)[None])
        if abs(float(returned[0])-best) > 1e-12:
            raise ValueError("returned chain mismatch")
        all_scores = np.concatenate(observed_scores)
        if int(np.argmin(all_scores)) != result["start_index"] or abs(all_scores.min()-result["start_score"]) > 1e-12:
            raise ValueError("start mismatch")
        if arm == "pair":
            p, s, t = observed_scores
            flags = []
            for i, (a, b) in enumerate(segments):
                valid = np.any(singles[2*i, 5*a:5*a+5] != parent[5*a:5*a+5]) and \
                        np.any(singles[2*i+1, 5*b:5*b+5] != parent[5*b:5*b+5])
                flags.append(bool(valid and t[i] <= p[0]-.01 and p[0]-s[2*i] < .01 and p[0]-s[2*i+1] < .01))
            if flags != result["witnesses"]:
                raise ValueError("witness mismatch")
            witness_pairs += sum(flags)
            if any(flags):
                witness_runs.append([root, rep])
            selected += int(result["start_index"] >= 25 and flags[result["start_index"]-25])
            final = np.asarray(result["final_actions"], np.float32)
            final_exact += int(any(flag and np.array_equal(final, pairs[i]) for i, flag in enumerate(flags)))
    if seen != {(r, p, a) for r in range(16) for p in range(2) for a in ("pair", "single")}:
        raise ValueError("incomplete design")
    groups, values = {}, {}
    for arm in ("pair", "single", "cem"):
        cells = np.array([((cem[(r, p)] if arm == "cem" else
                            next(x for x in rows if (x["root"], x["repeat"], x["arm"]) == (r, p, arm)))
                           ["result"]["final_score"])
                          for r in range(16) for p in range(2)]).reshape(16, 2)
        values[arm] = cells.mean(axis=1)
        groups[arm] = {"mean_final": float(cells.mean()), "success_over_32": int((cells < .05).sum())}
    paired = {}
    for other in ("single", "cem"):
        d = values["pair"] - values[other]
        paired[other] = {"mean_delta": float(d.mean()), "pair_better_roots": int((d < -1e-12).sum()),
                         "pair_worse_roots": int((d > 1e-12).sum()),
                         "ties": int((np.abs(d) <= 1e-12).sum()), "per_root": d.tolist()}
    result = {"status": "passed", "rows": len(seen), "groups": groups, "paired": paired,
              "witness_pairs": witness_pairs, "witness_runs": witness_runs,
              "witness_root_count": len({x[0] for x in witness_runs}),
              "witness_selected_start_count": selected, "witness_exact_final_count": final_exact,
              "summary_sha256": old.prior.digest(out / "summary.json"),
              "auditor_sha256": old.prior.digest(Path(__file__)),
              "scope": "same predictor implementation and previously exposed tasks"}
    with (out / "audit.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
