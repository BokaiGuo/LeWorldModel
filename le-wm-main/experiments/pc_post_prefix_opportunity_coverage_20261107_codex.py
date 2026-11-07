"""Evaluate saved prefix retention choices against a previously replayed B pool."""
import json
from pathlib import Path

from experiments import pc_temporal_prefix_mask_audit_20261107_codex as temporal
from experiments import pc_tworoom_temporal_retention_fixed_pool_20261107_codex as old
from experiments.pc_input_permissions import file_hash

PROTOCOL = temporal.run.ROOT.parent / "docs/PC_POST_PREFIX_OPPORTUNITY_COVERAGE_PROTOCOL_20261107_CODEX.md"
OUT = temporal.run.ROOT / "results/pc_post_prefix_opportunity_coverage_20261107_codex.json"
METHODS = ("endpoint", "full", "post")


def summarize(rows):
    episodes = {}
    for row in rows:
        episode = episodes.setdefault(row["episode"], {"initial_success": row["episode_initial_success"],
                                                       **{m: False for m in METHODS}})
        if episode["initial_success"] != row["episode_initial_success"]:
            raise ValueError("initial success mismatch")
        for m in METHODS:
            episode[m] |= row["coverage"][m]["tail"]
    non_initial = [e for e in episodes.values() if not e["initial_success"]]
    return {"decisions": len(rows), "episodes": len(episodes), "non_initial_episodes": len(non_initial),
            "decision_ever": {m: sum(r["coverage"][m]["ever"] for r in rows) for m in METHODS},
            "decision_tail": {m: sum(r["coverage"][m]["tail"] for r in rows) for m in METHODS},
            "episode_tail_non_initial": {m: sum(e[m] for e in non_initial) for m in METHODS},
            "post_only_vs_full": sum(e["post"] and not e["full"] for e in non_initial),
            "full_only_vs_post": sum(e["full"] and not e["post"] for e in non_initial)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, gpu, physical = (json.loads(p.read_text()) for p in (old.PLAN, temporal.OUT, old.pool.OUTPUT))
    if (plan["source_sha256"] != file_hash(old.pool.run.OUTPUT)
            or plan["protocol_sha256"] != file_hash(old.PROTOCOL)
            or plan["runner_sha256"] != file_hash(Path(old.__file__))
            or gpu["protocol_sha256"] != file_hash(temporal.PROTOCOL)
            or gpu["runner_sha256"] != file_hash(Path(temporal.__file__))
            or gpu["source_sha256"] != file_hash(old.pool.run.OUTPUT)
            or physical["source_sha256"] != file_hash(old.pool.run.OUTPUT)
            or physical["protocol_sha256"] != file_hash(old.pool.PROTOCOL)
            or physical["runner_sha256"] != file_hash(Path(old.pool.__file__))):
        raise ValueError("source drift")
    gpu_index = {(r["episode"], r["iteration"]): r for r in gpu["rows"]}
    phys_index = {(r["episode"], r["iteration"]): r for r in physical["entries"]}
    if len(gpu_index) != 119 or len(phys_index) != 119 or len(plan["decisions"]) != 119:
        raise ValueError("unexpected number of decisions")
    rows = []
    for decision in plan["decisions"]:
        key = decision["episode"], decision["iteration"]
        g, p = gpu_index[key], phys_index[key]
        if g["full_kept"] != decision["temporal_kept"] or decision["baseline"] != p["retained"]["lookahead"]:
            raise ValueError("retention identity mismatch")
        selected = {"endpoint": decision["baseline"], "full": g["full_kept"], "post": g["post_kept"]}
        rows.append({"episode": key[0], "iteration": key[1],
                     "episode_initial_success": decision["episode_initial_success"],
                     "selected": selected, "coverage": {m: old.coverage(p, selected[m]) for m in METHODS}})
    result = {"scope": "post-hoc exposed fixed-pool coverage; not actual closed-loop benefit",
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
              "gpu_sha256": file_hash(temporal.OUT), "plan_sha256": file_hash(old.PLAN),
              "physical_sha256": file_hash(old.pool.OUTPUT), "summary": summarize(rows), "rows": rows}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
