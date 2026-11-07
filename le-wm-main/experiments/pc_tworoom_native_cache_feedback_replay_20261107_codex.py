"""Replay exposed feedback cases with native cached LeWM rollout."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import pc_latent_rollout_safe_20261107_codex as cached
from experiments import pc_tworoom_real_vs_imagined_feedback_20261107_codex as original
from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_NATIVE_CACHE_FEEDBACK_REPLAY_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_native_cache_feedback_replay_20261107_codex.json"
AUDIT = ROOT / "results/pc_tworoom_native_cache_feedback_replay_20261107_codex_audit.json"


def compare(reference, candidate):
    if reference["track"] != candidate["track"] or reference["model_sha256"] != candidate["model_sha256"]:
        raise ValueError("track or model mismatch")
    old = {row["root"]: row for row in reference["rows"]}
    new = {row["root"]: row for row in candidate["rows"]}
    if old.keys() != new.keys() or len(old) != len(reference["rows"]) or len(new) != len(candidate["rows"]):
        raise ValueError("root mismatch or duplicate")
    rows = []
    for root in old:
        for arm in ("real_rgb", "imagined"):
            a, b = old[root]["arms"][arm], new[root]["arms"][arm]
            errors = []
            for key in ("ever_success", "queries", "packet_steps", "initial_rgb_sha256"):
                if a[key] != b[key]:
                    errors.append(key)
            if len(a["trace"]) != len(b["trace"]) or len(a["decisions"]) != len(b["decisions"]):
                errors.append("trace or decision count")
            score_diff, distance_diff = 0.0, 0.0
            for left, right in zip(a["trace"], b["trace"]):
                if left["action"] != right["action"] or left["terminated"] != right["terminated"] or left["truncated"] != right["truncated"]:
                    errors.append("physical trace")
                distance_diff = max(distance_diff, abs(left["distance"] - right["distance"]))
            for left, right in zip(a["decisions"], b["decisions"]):
                for key in ("iteration", "remaining", "winner", "actions", "actions_sha256", "old_tail_sha256",
                            "queries", "packet_steps", "forward_batches", "propagation_rollouts"):
                    if left[key] != right[key]:
                        errors.append(key)
                if len(left["all_scores"]) != len(right["all_scores"]):
                    errors.append("score count")
                else:
                    score_diff = max(score_diff, float(np.max(np.abs(np.asarray(left["all_scores"])
                                                                      - np.asarray(right["all_scores"])))))
                score_diff = max(score_diff, abs(left["score"] - right["score"]))
            if score_diff > 0.001 or distance_diff > 0.0001:
                errors.append("numeric tolerance")
            rows.append({"root": root, "arm": arm, "steps": len(b["trace"]),
                         "decisions": len(b["decisions"]), "ever_success": b["ever_success"],
                         "max_score_error": score_diff, "max_distance_error": distance_diff,
                         "errors": sorted(set(errors))})
    return rows


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("new output exists")
    reference = original.OUTPUT
    source_files = [Path(__file__), Path(original.__file__), Path(cached.__file__), PROTOCOL, reference]
    before = {str(path): file_hash(path) for path in source_files}
    print("Replay two exposed roots; compare physical traces and complete decision scores.", flush=True)
    # Replace a function binding only for this single-process invocation.
    # The shared model's encode method is never replaced by the new function.
    with patch.object(original, "latent_rollout", cached.rollout_single_latent), \
            patch.object(original, "OUTPUT", OUTPUT), patch.object(original, "PROTOCOL", PROTOCOL):
        original.main()
    after = {str(path): file_hash(path) for path in source_files}
    if before != after:
        raise ValueError("source drift during run")
    rows = compare(json.loads(reference.read_text()), json.loads(OUTPUT.read_text()))
    with AUDIT.open("x") as stream:
        json.dump({"track": "rgb_only_dev", "source_sha256": before, "rows": rows,
                   "all_pass": all(not row["errors"] for row in rows),
                   "scope": "retrospective two-root same-environment implementation regression"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
