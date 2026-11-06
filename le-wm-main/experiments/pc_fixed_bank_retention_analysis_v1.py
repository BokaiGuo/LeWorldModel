"""Post-hoc fixed-bank retention analysis; no model or environment calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(selection, result):
    if result.get("status") != "completed":
        raise ValueError("completed fixed-bank result required")
    layer = selection["layer"]
    bank = layer["candidates"]
    entries = result["diagnostic"]["candidates"]
    if len(bank) != len(entries) or not bank:
        raise ValueError("candidate bank mismatch")
    scores = []
    for index, entry in enumerate(entries):
        if entry["index"] != index or not entry["runs"]:
            raise ValueError("invalid candidate order or empty runs")
        runs = [float(run["score"]) for run in entry["runs"]]
        best = float(entry["best_score"])
        if not all(math.isfinite(value) for value in runs) or best != min(runs):
            raise ValueError("invalid candidate suffix score")
        scores.append(best)
    best = min(scores)
    rules = {}
    for name, indices in selection["selections"].items():
        if (not indices or len(set(indices)) != len(indices)
                or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(scores)
                       for i in indices)):
            raise ValueError("invalid retained indices")
        winner = min(indices, key=lambda i: (scores[i], i))
        reported = result["diagnostic"]["rules"][name]
        if (reported["indices"] != indices or reported["winner"] != winner
                or reported["best_score"] != scores[winner]):
            raise ValueError("saved rule disagrees with candidate results")
        rules[name] = {"indices": indices, "winner": winner, "score": scores[winner],
                       "gap_to_bank_best": scores[winner] - best}
    if len({len(rule["indices"]) for rule in rules.values()}) != 1:
        raise ValueError("unequal retention capacity")
    comparisons = {}
    for baseline in ("greedy", "diverse", "random", "lookahead"):
        response = set(rules["response"]["indices"])
        other = set(rules[baseline]["indices"])
        added, removed = sorted(response - other), sorted(other - response)
        comparisons[baseline] = {
            "response_added": [{"index": i, "suffix_score": scores[i]} for i in added],
            "response_removed": [{"index": i, "suffix_score": scores[i]} for i in removed],
            "response_minus_baseline": rules["response"]["score"] - rules[baseline]["score"],
        }
    return {"candidate_count": len(bank), "best_suffix_score_in_generated_bank": best,
            "rules": rules, "response_comparisons": comparisons,
            "samples_per_candidate": result["diagnostic"]["samples_per_candidate"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit_path = args.source / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "passed":
        raise ValueError("source audit not passed")
    rows, sources = [], {str(audit_path.resolve()): digest(audit_path)}
    for audited in audit["rows"]:
        root = audited["root"]
        if audited["status"] != "completed":
            rows.append({"root": root, "status": audited["status"]})
            continue
        result_path = args.source / f"root{root:02d}_result.json"
        result = json.loads(result_path.read_text())
        selection_path = args.source / result["selection"]
        if digest(selection_path) != result["selection_sha256"]:
            raise ValueError("selection hash mismatch")
        selection = json.loads(selection_path.read_text())
        if selection["root"] != root or result["root"] != root or selection["task"] != result["task"]:
            raise ValueError("root or task mismatch")
        summary = analyze(selection, result)
        if (audited["candidate_count"] != summary["candidate_count"]
                or audited["all_bank_best"] != summary["best_suffix_score_in_generated_bank"]
                or audited["rules"] != result["diagnostic"]["rules"]):
            raise ValueError("result differs from audited summary")
        rows.append({"root": root, "status": "completed", **summary})
        for path in (result_path, selection_path):
            sources[str(path.resolve())] = digest(path)
    output = {"scope": "post-hoc fixed generated bank; not equal-total-cost or proof of coverage",
              "source_sha256": sources, "rows": rows}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Analyzed {len(rows)} recorded roots without new model calls: {args.output}")


if __name__ == "__main__":
    main()
