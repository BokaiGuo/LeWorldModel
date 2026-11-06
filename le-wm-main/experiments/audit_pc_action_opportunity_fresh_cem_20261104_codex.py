"""Read-only reconstruction of fresh-CEM calls and paired group summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_candidate_suffix_quality_posthoc as quality
from experiments import pc_discovery_attribution_dev as panel
from experiments import pc_prediction_journal_audit_v1 as journal_audit
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/pc_action_opportunity_fresh_cem_dev_20261104_codex"
SOURCE = RUN / "result.json"
OUTPUT = RUN / "independent_audit_v2.json"


def verify_hashes(hashes: dict[str, str]) -> None:
    for filename, expected in hashes.items():
        path = Path(filename)
        if not path.is_absolute():
            path = ROOT / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {path}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(SOURCE.read_text())
    if result["status"] != "completed":
        raise ValueError("run did not complete")
    verify_hashes(result["source_hashes"])
    if hashlib.sha256(Path(cem.__file__).read_bytes()).hexdigest() != result["optimizer_sha256"]:
        raise ValueError("optimizer changed since run")
    tasks = panel.prior.tasks()
    scores = {}
    total_tokens = 0
    for row in result["candidate_rows"]:
        root, arm, depth, index = (row[k] for k in ("root", "arm", "depth", "index"))
        key = (root, arm, depth, index)
        if key in scores:
            raise ValueError(f"duplicate candidate {key}")
        source = json.loads((quality.OUT / f"root{root:02d}_{arm}.json").read_text())
        matching = [c for c in source["candidates"] if c["depth"] == depth and c["index"] == index]
        if len(matching) != 1:
            raise ValueError(f"candidate binding mismatch {key}")
        prefix = np.asarray(matching[0]["prefix"], dtype=np.float32)
        if hashlib.sha256(prefix.tobytes()).hexdigest() != row["prefix_sha256_float32"]:
            raise ValueError(f"prefix mismatch {key}")
        if row["seed"] != 2045000000 + root * 100 + depth:
            raise ValueError(f"seed mismatch {key}")
        directory = RUN / f"root{root:02d}_{arm}_d{depth}_c{index:03d}_calls"
        info = journal_audit.audit_journal(directory)
        if (len(info["calls"]) != row["calls"] or info["tokens"] != row["tokens"]
                or any(call["status"] != "returned" for call in info["calls"])):
            raise ValueError(f"journal mismatch {key}")
        predictor = panel.prior.old.previous.fixed.predictor(tasks[root])
        index_call = 0

        def recorded_predict(actions):
            nonlocal index_call
            stem = directory / f"{index_call:06d}"
            with np.load(f"{stem}_actions.npz", allow_pickle=False) as f:
                saved_actions = f["actions"].copy()
            with np.load(f"{stem}_result.npz", allow_pickle=False) as f:
                saved_features, saved_scores = f["features"].copy(), f["scores"].copy()
            np.testing.assert_array_equal(np.asarray(actions, dtype=np.float32), saved_actions)
            actual_features, actual_scores = predictor(saved_actions.copy())
            np.testing.assert_allclose(actual_features, saved_features, rtol=0, atol=1e-10)
            np.testing.assert_allclose(actual_scores, saved_scores, rtol=0, atol=1e-10)
            index_call += 1
            return saved_features, saved_scores

        tail = np.zeros((15 - len(prefix), 2), dtype=np.float32)
        initial = float(recorded_predict(np.concatenate((prefix, tail))[None])[1][0])
        chain, score, generations = cem.cem(
            tail, prefix, 120, np.random.default_rng(row["seed"]), recorded_predict, initial, 4,
        )
        if (index_call != len(info["calls"]) or initial != row["initial_score"]
                or score != row["best_score"] or chain.tolist() != row["chain"]
                or generations != row["generations"] or bool(score < 0.05) != row["success"]):
            raise ValueError(f"solver replay mismatch {key}")
        scores[key] = score
        total_tokens += info["tokens"]
    if total_tokens != result["diagnostic_prediction_tokens"]:
        raise ValueError("total token mismatch")
    selection = json.loads(
        (ROOT / "results/pc_action_opportunity_retention_absolute_probe_audit_20261104_codex.json").read_text()
    )
    selected = {
        (r["root"], r["arm"], r["depth"], r["policy"]): r["selected_indices"]
        for r in selection["rows"]
    }
    if len(selected) != 192:
        raise ValueError("selection coverage mismatch")
    seen_groups = set()
    group_scores = {}
    for group in result["group_rows"]:
        root, arm, depth, policy = (group[k] for k in ("root", "arm", "depth", "policy"))
        group_key = (root, arm, depth, policy)
        if group_key in seen_groups:
            raise ValueError("duplicate group")
        seen_groups.add(group_key)
        if group["selected_indices"] != selected[group_key]:
            raise ValueError(f"selection binding mismatch {group_key}")
        score = min(scores[(root, arm, depth, i)] for i in group["selected_indices"])
        if score != group["best_score"] or (score < 0.05) != group["success"]:
            raise ValueError(f"summary mismatch {group_key}")
        group_scores[group_key] = score
    if seen_groups != set(selected):
        raise ValueError("group coverage mismatch")
    if len(result["root_rows"]) != 32:
        raise ValueError("root coverage mismatch")
    for row in result["root_rows"]:
        root, arm = row["root"], row["arm"]
        means = {
            p: float(np.mean([group_scores[(root, arm, depth, p)] for depth in (0, 1)]))
            for p in ("absolute_probe_min", "relative_improvement", "effect_complement")
        }
        if means != row["mean_by_policy"]:
            raise ValueError("root mean mismatch")
        if means["effect_complement"] - means["absolute_probe_min"] != row["effect_minus_absolute"]:
            raise ValueError("root comparison mismatch")
    audit = {
        "schema": "pc_action_opportunity_fresh_cem_independent_audit.v1",
        "status": "passed",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidates": len(scores), "groups": len(seen_groups), "prediction_tokens": total_tokens,
        "scope": "same-workspace independent analysis, exact-dynamics re-evaluation and deterministic solver replay; not independent scientific replication",
        "limitations": "uses current exact predictor and original optimizer; not a frozen clean-environment dependency package",
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
