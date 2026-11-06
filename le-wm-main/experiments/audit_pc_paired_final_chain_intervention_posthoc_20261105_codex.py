"""Replay intervention journals and independently recompute comparisons."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_paired_final_chain_intervention_posthoc_20261105_codex as run


def main():
    out, source, fixed = run.OUT, run.source, run.fixed
    manifest = json.loads((out / "manifest.json").read_text())
    rows = json.loads((out / "summary.json").read_text())["rows"]
    seal = fixed.prior.prior.prior.seal
    seal.verify(manifest["dependencies"])
    seal.verify(manifest["source_dependencies"])
    if len(rows) != 32:
        raise ValueError("incomplete panel")
    seen = set()
    for row in rows:
        root, rep = row["root"], row["repeat"]
        if (root, rep) in seen or row["status"] != "completed":
            raise ValueError("duplicate or failed row")
        seen.add((root, rep))
        name = f"root{root:02d}_rep{rep}"
        if row != json.loads((out / f"{name}.json").read_text()):
            raise ValueError("row mismatch")
        plan = json.loads((source.OUT / f"{name}_pair.json").read_text())
        source_parent = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = fixed.starts_from_result(source_parent["result"])[0][0]
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        log, counter, tokens = out / f"{name}_calls", [0], [0]

        def predict(actions):
            j = counter[0]
            stem = f"{j:06d}"
            req = json.loads((log / f"{stem}_attempt.json").read_text())
            resp = json.loads((log / f"{stem}_returned.json").read_text())
            a_path, r_path = log / req["actions"], log / resp["result"]
            if source.old.prior.digest(a_path) != req["sha256"] or source.old.prior.digest(r_path) != resp["sha256"]:
                raise ValueError("hash mismatch")
            with np.load(a_path, allow_pickle=False) as f:
                recorded = f["actions"].copy()
            with np.load(r_path, allow_pickle=False) as f:
                features, scores = f["features"].copy(), f["scores"].copy()
            np.testing.assert_array_equal(actions, recorded)
            f2, s2 = predictor(recorded)
            np.testing.assert_allclose(features, f2, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, s2, rtol=0, atol=1e-12)
            if recorded.shape[1] % 5:
                raise ValueError("non-packet action length")
            tokens[0] += len(recorded)*(recorded.shape[1] // 5)
            counter[0] += 1
            return features, scores

        rebuilt = run.intervene(predict, manifest["tasks"][root], plan["result"]["final_actions"], parent, root, rep)
        if rebuilt != row["result"] or counter[0] != row["calls"] or tokens[0] != row["tokens"] or tokens[0] != 3265:
            raise ValueError("replay, budget, or result mismatch")
        result, chain = row["result"], np.asarray(plan["result"]["final_actions"], np.float32)
        if result["selected_chain"] != chain.tolist():
            raise ValueError("selected chain mismatch")
        controls = result["controls"]
        for name in ("parent", "zero"):
            a, b = controls["selected"], controls[name]
            common = (a["common_scores"][0]-a["common_scores"][1]) - (b["common_scores"][0]-b["common_scores"][1])
            replan = np.mean([x["score"] for x in b["runs"]]) - np.mean([x["score"] for x in a["runs"]])
            check = result[name + "_comparison"]
            if (not np.isclose(common, check["common_improvement_delta"], atol=1e-12, rtol=0)
                    or not np.isclose(replan, check["replan_score_improvement"], atol=1e-12, rtol=0)
                    or (common >= .01) != check["common_at_least_001"]
                    or (replan >= .01) != check["replan_at_least_001"]):
                raise ValueError("comparison calculation mismatch")
    if seen != {(r, p) for r in range(16) for p in range(2)}:
        raise ValueError("panel missing")
    nonprogress = [r for r in rows if r["result"]["nonprogress_001"]]
    summary = {"chains": len(rows), "task_roots": 16, "nonprogress": len(nonprogress),
               "success": sum(r["result"]["chain_score"] < .05 for r in rows)}
    for name in ("parent", "zero"):
        common = [r for r in rows if r["result"][name+"_comparison"]["common_at_least_001"]]
        replan = [r for r in rows if r["result"][name+"_comparison"]["replan_at_least_001"]]
        both = [r for r in rows if r["result"]["nonprogress_001"] and
                r["result"][name+"_comparison"]["common_at_least_001"] and
                r["result"][name+"_comparison"]["replan_at_least_001"]]
        summary[name] = {"common": len(common), "replan": len(replan), "nonprogress_and_both": len(both),
                         "nonprogress_and_both_roots": len({r["root"] for r in both}),
                         "rows": [[r["root"], r["repeat"]] for r in both]}
    allboth = [r for r in rows if r["result"]["nonprogress_001"] and all(
        r["result"][n+"_comparison"]["common_at_least_001"] and
        r["result"][n+"_comparison"]["replan_at_least_001"] for n in ("parent", "zero"))]
    summary["both_controls_intersection"] = len(allboth)
    summary["both_controls_roots"] = len({r["root"] for r in allboth})
    result = {"status": "passed", "summary": summary,
              "summary_sha256": source.old.prior.digest(out / "summary.json"),
              "auditor_sha256": source.old.prior.digest(Path(__file__)),
              "scope": "same predictor and intervention implementation; not independent replication or PC2"}
    with (out / "audit.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
