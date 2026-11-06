"""Fresh-random-stream suffix optimization for fixed retention selections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_candidate_suffix_quality_posthoc as quality
from experiments import pc_discovery_attribution_dev as panel
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_action_opportunity_retention_absolute_probe_audit_20261104_codex.json"
OUT = ROOT / "results/pc_action_opportunity_fresh_cem_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_ACTION_OPPORTUNITY_FRESH_CEM_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("random_direction", "spectral")
POLICIES = ("absolute_probe_min", "relative_improvement", "effect_complement")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(root: int, depth: int) -> int:
    if root not in range(16) or depth not in (0, 1):
        raise ValueError("invalid root or depth")
    return 2045000000 + root * 100 + depth


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    selection = json.loads(SOURCE.read_text())
    if selection["status"] != "completed":
        raise ValueError("selection source incomplete")
    selection_index = {}
    for row in selection["rows"]:
        key = (row["root"], row["arm"], row["depth"], row["policy"])
        if key in selection_index or row["policy"] not in POLICIES:
            raise ValueError(f"duplicate or unknown selection {key}")
        if len(row["selected_indices"]) != 2 or len(set(row["selected_indices"])) != 2:
            raise ValueError(f"invalid slots {key}")
        selection_index[key] = row
    if len(selection_index) != 16 * 2 * 2 * 3:
        raise ValueError("incomplete selection coverage")
    tasks = panel.prior.tasks()
    if len(tasks) != 16:
        raise ValueError("unexpected task count")
    OUT.mkdir()
    source_hashes = {str(SOURCE.relative_to(ROOT)): sha(SOURCE), str(PROTOCOL): sha(PROTOCOL)}
    result_rows = []
    group_rows = []
    for root, task in enumerate(tasks):
        for arm in ARMS:
            path = quality.OUT / f"root{root:02d}_{arm}.json"
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
            source = json.loads(path.read_text())
            if (source["status"], source["root"], source["arm"]) != ("completed", root, arm):
                raise ValueError(f"invalid quality source {path}")
            indexed = {(int(x["depth"]), int(x["index"])): x for x in source["candidates"]}
            if len(indexed) != len(source["candidates"]):
                raise ValueError("duplicate candidate key")
            for depth in (0, 1):
                selected = {p: selection_index[(root, arm, depth, p)]["selected_indices"] for p in POLICIES}
                union = sorted(set().union(*map(set, selected.values())))
                results = {}
                for index in union:
                    item = indexed[(depth, index)]
                    if item.get("status") not in (None, "completed"):
                        raise ValueError("candidate failed")
                    prefix = np.asarray(item["prefix"], dtype=np.float32)
                    if prefix.shape != ((depth + 1) * 5, 2):
                        raise ValueError("prefix length mismatch")
                    directory = OUT / f"root{root:02d}_{arm}_d{depth}_c{index:03d}_calls"
                    recorder = panel.journal.PredictionJournal(
                        directory, panel.prior.old.previous.fixed.predictor(task),
                        action_dim=2, packet_length=5, token_cap=100000,
                    )
                    tail = np.zeros((15 - len(prefix), 2), dtype=np.float32)
                    initial = float(recorder(np.concatenate((prefix, tail))[None])[1][0])
                    if abs(initial - float(item["initial_score"])) > 1e-7:
                        raise ValueError("zero-tail score mismatch")
                    chain, score, generations = cem.cem(
                        tail, prefix, 120, np.random.default_rng(seed_for(root, depth)),
                        recorder, initial, 4,
                    )
                    if not np.isfinite(score) or recorder.tokens != 363:
                        raise ValueError("invalid score or unexpected token count")
                    results[index] = float(score)
                    result_rows.append({
                        "root": root, "arm": arm, "depth": depth, "index": index,
                        "seed": seed_for(root, depth),
                        "prefix_sha256_float32": hashlib.sha256(prefix.tobytes()).hexdigest(),
                        "initial_score": initial, "best_score": float(score),
                        "success": bool(score < 0.05), "chain": chain.tolist(),
                        "generations": generations, "tokens": recorder.tokens,
                        "calls": recorder.calls,
                    })
                for policy in POLICIES:
                    best = min(results[i] for i in selected[policy])
                    group_rows.append({
                        "root": root, "arm": arm, "depth": depth, "policy": policy,
                        "selected_indices": selected[policy],
                        "best_score": best, "success": bool(best < 0.05),
                    })
            print(f"completed root={root} arm={arm}", flush=True)
    root_rows = []
    for root in range(16):
        for arm in ARMS:
            mean = {
                p: float(np.mean([r["best_score"] for r in group_rows
                                  if r["root"] == root and r["arm"] == arm and r["policy"] == p]))
                for p in POLICIES
            }
            root_rows.append({"root": root, "arm": arm, "mean_by_policy": mean,
                              "effect_minus_absolute": mean["effect_complement"] - mean["absolute_probe_min"]})
    summaries = {}
    for arm in ARMS:
        selected = [r for r in root_rows if r["arm"] == arm]
        diffs = [r["effect_minus_absolute"] for r in selected]
        summaries[arm] = {
            "root_count": len(selected),
            "mean_root_score": {p: float(np.mean([r["mean_by_policy"][p] for r in selected])) for p in POLICIES},
            "effect_minus_absolute": float(np.mean(diffs)),
            "effect_better_roots": sum(d < -1e-12 for d in diffs),
            "effect_worse_roots": sum(d > 1e-12 for d in diffs),
            "ties": sum(abs(d) <= 1e-12 for d in diffs),
            "success_groups": {p: sum(r["success"] for r in group_rows if r["arm"] == arm and r["policy"] == p)
                               for p in POLICIES},
        }
    result = {
        "schema": "pc_action_opportunity_fresh_cem_dev.v1", "status": "completed",
        "scope": "fixed development candidates, common fresh random streams, finite suffix solver; not end-to-end",
        "protocol_sha256": sha(PROTOCOL), "code_sha256": sha(Path(__file__)),
        "optimizer_sha256": sha(Path(cem.__file__)), "source_hashes": source_hashes,
        "candidate_rows": result_rows, "group_rows": group_rows, "root_rows": root_rows,
        "summaries": summaries, "diagnostic_prediction_tokens": sum(r["tokens"] for r in result_rows),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    print("Testing retained candidates with a fresh common finite-budget CEM random stream.", flush=True)
    main()
