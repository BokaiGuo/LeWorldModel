"""Independently audit saved replay comparisons without rerunning the model."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_unified_rgb_frozen_model_replay_20261103_codex.json"
OUTPUT = ROOT / "results/pc_unified_rgb_frozen_model_replay_20261103_codex_audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a - b)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = json.loads(SOURCE.read_text())
    sources = payload["dependencies"]
    require(all(sha(Path(path)) == digest for path, digest in sources.items()),
            "dependency mismatch")
    require(payload["tolerances"] == {"prediction_abs": 2e-5, "selected_action_abs": 1e-5},
            "unexpected thresholds")
    decisions = {}
    for path, digest in sources.items():
        if path.endswith(".json"):
            try:
                candidate = json.loads(Path(path).read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(candidate, dict) and "result" in candidate and "ticks" in candidate:
                decisions[digest] = candidate
    rows = payload["rows"]
    require(sorted((r["root"], r["step"]) for r in rows)
            == [(root, step) for root in range(4) for step in range(3)], "panel mismatch")
    stats = {"fixed_batches": 0, "fixed_passing": 0, "fixed_tokens": 0,
             "fixed_position_max_abs": 0.0, "fixed_score_max_abs": 0.0,
             "recompiled_batches": 0, "recompiled_samples": 0, "recompiled_tokens": 0,
             "matched_selected": 0, "matched_abstention": 0, "matched_batch_count": 0,
             "selected_action_max_abs": 0.0, "selected_score_max_abs": 0.0,
             "paired_action_max_abs": 0.0, "paired_position_max_abs": 0.0,
             "paired_score_max_abs": 0.0, "paired_shape_mismatches": 0}
    for row in rows:
        old = decisions[row["decision_sha256"]]
        saved = old["result"]
        require(row["ticks"] == old["ticks"] and row["input_rgb_sha256"] == old["input_rgb_sha256"]
                and row["history_actions"] == old["history_actions"], "context mismatch")
        old_trace = saved["prediction_trace"]
        fixed = row["fixed_batches"]
        require(len(fixed) == len(old_trace), "fixed batch count mismatch")
        fixed_tokens = 0
        for index, (f, batch) in enumerate(zip(fixed, old_trace)):
            actions = np.asarray(batch["model_raw_float32_actions"])
            require(f["index"] == index and f["count"] == len(actions)
                    and f["horizon"] == actions.shape[1], "fixed batch metadata mismatch")
            require(f["within_tolerance"] == (f["position_max_abs"] <= 2e-5
                    and f["score_max_abs"] <= 2e-5), "fixed tolerance mismatch")
            fixed_tokens += len(actions) * actions.shape[1] // 5
            stats["fixed_passing"] += int(f["within_tolerance"])
            for key in ("position", "score"):
                stat = "fixed_" + key + "_max_abs"
                stats[stat] = max(stats[stat], f[key + "_max_abs"])
        require(fixed_tokens == row["fixed_tokens"], "fixed token mismatch")
        stats["fixed_batches"] += len(fixed)
        stats["fixed_tokens"] += fixed_tokens
        new = row["recompiled"]
        trace = new["prediction_trace"]
        counts = new["counts"]
        samples = sum(len(batch["model_raw_float32_actions"]) for batch in trace)
        tokens = sum(len(batch["model_raw_float32_actions"])
                     * np.asarray(batch["model_raw_float32_actions"]).shape[1] // 5 for batch in trace)
        require(counts["prediction_calls"] == len(trace) and counts["trajectory_samples"] == samples
                and counts["prediction_token_samples"] == tokens and tokens <= counts["token_cap"],
                "recompiled counts mismatch")
        require(new["context_id"] == row["decision_sha256"] and new["root"] == row["root"]
                and new["step"] == row["step"], "compiled context mismatch")
        chain = new["core_result"]["selected"]["chain"]
        require(new["dispatched_actions"] == chain[:5] and new["remaining_tail"] == chain[5:],
                "dispatch mismatch")
        action_error = maximum(chain, saved["selected"]["chain"])
        score_error = abs(new["core_result"]["selected"]["terminal_score"]
                          - saved["selected"]["terminal_score"])
        require(action_error == row["selected_action_max_abs"]
                and score_error == row["selected_score_abs"], "selected comparison mismatch")
        require(row["selected_within_tolerance"] == (action_error <= 1e-5 and score_error <= 2e-5)
                and row["abstention_matches"] == (new["core_result"]["abstained"] == saved["abstained"])
                and row["same_batch_count"] == (len(trace) == len(old_trace)), "decision flag mismatch")
        for index, (comparison, a, b) in enumerate(zip(row["batch_comparison"], trace, old_trace)):
            aa, bb = np.asarray(a["model_raw_float32_actions"]), np.asarray(b["model_raw_float32_actions"])
            require(comparison["index"] == index and comparison["shape_match"] == (aa.shape == bb.shape),
                    "batch shape record mismatch")
            if aa.shape != bb.shape:
                stats["paired_shape_mismatches"] += 1
                continue
            for key, first, second in (("action", aa, bb), ("position", a["positions"], b["positions"]),
                                       ("score", a["scores"], b["scores"])):
                error = maximum(first, second)
                require(error == comparison[key + "_max_abs"], "batch error mismatch")
                stat = "paired_" + key + "_max_abs"
                stats[stat] = max(stats[stat], error)
        require(len(row["batch_comparison"]) == min(len(trace), len(old_trace)), "pair count mismatch")
        stats["recompiled_batches"] += len(trace)
        stats["recompiled_samples"] += samples
        stats["recompiled_tokens"] += tokens
        stats["matched_selected"] += int(row["selected_within_tolerance"])
        stats["matched_abstention"] += int(row["abstention_matches"])
        stats["matched_batch_count"] += int(row["same_batch_count"])
        stats["selected_action_max_abs"] = max(stats["selected_action_max_abs"], action_error)
        stats["selected_score_max_abs"] = max(stats["selected_score_max_abs"], score_error)
    result = {"source_sha256": sha(SOURCE), "audit_script_sha256": sha(Path(__file__)),
              "dependency_count": len(sources), "contexts": len(rows), "stats": stats,
              "limitations": ["Fixed-request outputs were not saved: their errors cannot be independently recomputed.",
                              "Hashes verify listed files only, not all transitive dependencies or data provenance.",
                              "No model execution or physical execution was performed by this audit."]}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
