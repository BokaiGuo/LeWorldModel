"""Post-hoc evaluator-only useful-prefix recall on the sealed exhaustive B bank."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_full_b_pool_diagnostic_20261107_codex as source
from experiments.pc_input_permissions import file_hash

q = source.q
OUT = q.ROOT / "results/pc_tworoom_retention_b_utility_audit_20261107_codex.json"
PROTOCOL = q.ROOT.parent / "docs/PC_TWOROOM_RETENTION_B_UTILITY_AUDIT_PROTOCOL_20261107_CODEX.md"


def utility_metrics(term, retained):
    term = np.asarray(term)
    retained = np.asarray(retained)
    if term.shape != (8, 8, 25) or term.dtype != np.bool_:
        raise ValueError("expected boolean 8x8x25 termination array")
    if (retained.shape != (2,) or not np.issubdtype(retained.dtype, np.integer)
            or len(set(retained.tolist())) != 2 or np.any(retained < 0) or np.any(retained >= 8)):
        raise ValueError("expected two distinct retained indices")
    ever = term.any(axis=2)
    post_prefix = ~term[:, :, :10].any(axis=2) & term[:, :, 10:].any(axis=2)
    result = {}
    for name, successful in (("ever", ever), ("post_prefix", post_prefix)):
        useful = successful.any(axis=1)
        count = int(useful.sum())
        retained_count = int(useful[retained].sum())
        result[name] = {"useful_prefixes": count, "retained_useful_prefixes": retained_count,
                        "recall": retained_count / count if count else None,
                        "pool_success": bool(useful.any()),
                        "retained_success": bool(useful[retained].any()),
                        "retained_successful_chains": int(successful[retained].sum())}
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source.verify()
    plan = json.loads(q.PLAN.read_text())
    prior = json.loads(q.PHYSICAL.read_text())
    original_report = json.loads(source.RESULT.read_text())
    if (prior["plan_sha256"] != file_hash(q.PLAN) or prior["pool_sha256"] != file_hash(q.POOL)
            or original_report["seal_sha256"] != file_hash(source.SEAL)
            or original_report["physical_sha256"] != file_hash(source.PHYS)):
        raise ValueError("physical evidence binding mismatch")
    if [r["root"] for r in plan["rows"]] != list(range(10, 50)):
        raise ValueError("incomplete fixed panel")
    previous = {r["root"]: r for r in prior["rows"]}
    rows = []
    with np.load(q.old.POOL, allow_pickle=False) as pool, np.load(source.PHYS, allow_pickle=False) as physical:
        for task in plan["rows"]:
            root = task["root"]
            actions = pool[f"r{root}_b"].reshape(8, 8, 25, 2)
            if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                raise ValueError("illegal B actions")
            if not np.array_equal(actions[:, :, :10], np.broadcast_to(actions[:, :1, :10], (8, 8, 10, 2))):
                raise ValueError("B candidates do not share their declared prefix")
            term = physical[f"r{root}_term"].reshape(8, 8, 25)
            arms = {}
            for arm in q.METHODS:
                kept = task["decisions"][arm]["retained"]
                metrics = utility_metrics(term, kept)
                selected = previous[root]["outcomes"][arm]
                arms[arm] = {"retained": kept, "metrics": metrics,
                             "actual_selected_ever": selected["ever"],
                             "actual_selected_final": selected["final"]}
            rows.append({"root": root, "arms": arms})
    summary = {}
    for arm in q.METHODS:
        summary[arm] = {}
        for label in ("ever", "post_prefix"):
            vals = [r["arms"][arm]["metrics"][label] for r in rows]
            recall = [v["recall"] for v in vals if v["recall"] is not None]
            summary[arm][label] = {
                "roots_with_pool_success": sum(v["pool_success"] for v in vals),
                "roots_with_retained_success": sum(v["retained_success"] for v in vals),
                "root_macro_recall": float(np.mean(recall)) if recall else None,
                "eligible_roots": len(recall),
                "actual_failure_with_retained_success": [
                    r["root"] for r in rows
                    if not r["arms"][arm]["actual_selected_ever"]
                    and r["arms"][arm]["metrics"][label]["retained_success"]]}
        summary[arm]["actual_selected_success"] = sum(r["arms"][arm]["actual_selected_ever"] for r in rows)
    with OUT.open("x") as f:
        json.dump({"scope": "post-hoc exposed development; physical oracle not a planner",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "plan_sha256": file_hash(q.PLAN), "physical_sha256": file_hash(source.PHYS),
                   "pool_sha256": file_hash(q.old.POOL), "summary": summary, "rows": rows},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
