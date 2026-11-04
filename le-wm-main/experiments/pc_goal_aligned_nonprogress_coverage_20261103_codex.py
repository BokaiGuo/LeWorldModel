"""Post-hoc attribution of sealed goal-aligned search, without replanning."""
import json
from pathlib import Path

from experiments import pc_goal_aligned_successor_dev_20261103_codex as source
from experiments import pc_multidirection_nonprogress_coverage_attribution_20261103_codex as coverage

PROTOCOL = source.PROTOCOL.parent / "PC_GOAL_ALIGNED_NONPROGRESS_COVERAGE_PROTOCOL_20261103_CODEX.md"
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_nonprogress_coverage.json")
ARMS = ("goal", "svd", "random")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = source.sha
    paths = (Path(__file__), Path(coverage.__file__), PROTOCOL, source.PLANS, source.AUDIT)
    hashes = {str(p): sha(p) for p in paths}
    data, audit = (json.loads(p.read_text()) for p in (source.PLANS, source.AUDIT))
    if data["dependencies"] != source.dependencies() or audit["plans_sha256"] != sha(source.PLANS):
        raise ValueError("source seal mismatch")
    rows = []
    for row in data["rows"]:
        if row["arm"] in ARMS:
            result = coverage.analyze(row)
            result["arm"] = row["arm"]
            rows.append(result)
    if len(rows) != 72 or len({(x["arm"], x["root"]) for x in rows}) != 72:
        raise ValueError("panel mismatch")
    summary = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        condition = lambda x: any(a and b for a, b in zip(x["nonprogress"], x["nonzero"], strict=True))
        best = [r["full"]["best_nonzero_nonprogress_score"] for r in subset]
        summary[arm] = {
            "first_roots": sum(condition(r["first_bank"]) for r in subset),
            "retained_roots": sum(condition(r["retained_bank"]) for r in subset),
            "retained_missing_roots": [r["root"] for r in subset if not condition(r["retained_bank"])],
            "scored_roots": sum(x is not None for x in best),
            "selected_roots": sum(r["selected"]["nonprogress"] and r["selected"]["nonzero"] for r in subset),
            "full_samples": sum(r["full"]["samples"] for r in subset),
            "nonprogress_nonzero_samples": sum(r["full"]["nonzero_nonprogress"] for r in subset),
            "nonprogress_best_mean": sum(x for x in best if x is not None) / sum(x is not None for x in best) if any(x is not None for x in best) else None,
            "max_selected_gap": max(abs(r["selected"]["score_gap_from_scored_best"]) for r in subset)}
    if hashes != {str(p): sha(p) for p in paths} or data["dependencies"] != source.dependencies():
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "summary": summary, "rows": rows}, stream, allow_nan=False, separators=(",", ":"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
