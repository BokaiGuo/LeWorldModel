"""Compare sealed cached-control episodes against the uncached control run."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_incumbent_rgb_feedback_20261107_codex as baseline
from experiments import pc_tworoom_rgb_cache_actual_feedback_20261107_codex as cached
from experiments.pc_input_permissions import file_hash

OUT = cached.ROOT / "results/pc_tworoom_rgb_cache_actual_feedback_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    a, b = json.loads(baseline.OUT.read_text()), json.loads(cached.OUT.read_text())
    for record, runner in ((a, baseline), (b, cached)):
        for key, path in (("protocol_sha256", runner.PROTOCOL), ("runner_sha256", runner.__file__),
                          ("planner_sha256", baseline.planner.__file__)):
            if record[key] != file_hash(Path(path)):
                raise ValueError(f"binding mismatch: {key}")
    for key in ("track", "input_sha256", "model_sha256", "normalizer_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"run mismatch: {key}")
    rows, max_error, max_distance = [], 0.0, 0.0
    for x, y in zip(a["rows"], b["rows"], strict=True):
        if (x["root"], x["row"]) != (y["root"], y["row"]):
            raise ValueError("root mismatch")
        for arm in baseline.planner.ARMS:
            old, new = x["arms"][arm], y["arms"][arm]
            if len(old["trace"]) != len(new["trace"]) or len(old["decisions"]) != len(new["decisions"]):
                raise ValueError("trajectory length mismatch")
            for p, q in zip(old["trace"], new["trace"], strict=True):
                max_distance = max(max_distance, abs(p["distance"] - q["distance"]))
                if (p["action"] != q["action"] or p["terminated"] != q["terminated"]
                        or p["truncated"] != q["truncated"] or abs(p["distance"] - q["distance"]) > 1e-6):
                    raise ValueError("physical trajectory mismatch")
            for p, q in zip(old["decisions"], new["decisions"], strict=True):
                v, w = np.asarray(p["all_scores"]), np.asarray(q["all_scores"])
                max_error = max(max_error, float(np.max(np.abs(v - w))))
                if (not np.allclose(v, w, rtol=1e-5, atol=1e-4)
                        or p["winner"] != q["winner"] or p["actions_sha256"] != q["actions_sha256"]
                        or p["retained"] != q["retained"] or p["dispatched"] != q["dispatched"]):
                    raise ValueError("planning mismatch")
                if (q["queries"] != 680 or q["encoder_calls"] != 2 or q["cache_misses"] != 1
                        or q["cache_hits"] != q["forward_calls"] - 1):
                    raise ValueError("cache accounting mismatch")
            rows.append({"root": x["root"], "arm": arm, "success": new["ever_success"],
                         "steps": len(new["trace"]), "decisions": len(new["decisions"]),
                         "encoder_calls": sum(t["encoder_calls"] for t in new["decisions"]),
                         "wall_seconds": new["wall_seconds"]})
    output = {"baseline_sha256": file_hash(baseline.OUT), "cached_sha256": file_hash(cached.OUT),
              "auditor_sha256": file_hash(__file__), "rows": rows, "max_score_error": max_error,
              "max_distance_error": max_distance, "scope": "four exposed roots, actual feedback regression"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
