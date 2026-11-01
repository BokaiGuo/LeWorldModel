"""Read-only comparison of actual resource vectors in a saved P3 consumer.

This checks a pinned consumer artifact, not its underlying journal or model.
Resource axes overlap and must never be summed into FLOPs.
"""
import argparse
import hashlib
import json
from pathlib import Path

AXES = ("action_encoder", "predict", "predictor", "pred_proj", "jacobian",
        "vjp_directions", "grad_api", "raw_backward_hook", "svd", "metric")
ARMS = ("uniform", "signal", "reverse")
POINTS = (20, 30)


def audit(data, sha256):
    if (data.get("status") != "semantic_saved_checked_not_scientifically_admitted"
            or data.get("scientific_pass") is not False
            or data.get("physical_certified") is not False
            or data.get("model_arrays_independently_recomputed") is not False):
        raise ValueError("unexpected consumer status or admission")
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("expected six distinct point/arm rows")
    vectors = {}
    for row in rows:
        point, arm, counts = row.get("point"), row.get("arm"), row.get("attempted")
        if type(point) is not int or point not in POINTS or arm not in ARMS:
            raise ValueError("unexpected point or arm")
        if (point, arm) in vectors:
            raise ValueError("duplicate point/arm")
        if not isinstance(counts, dict) or set(counts) != set(AXES):
            raise ValueError("resource axes incomplete or unexpected")
        if any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("resource counts must be nonnegative integers")
        vectors[point, arm] = counts
    comparisons = []
    for point in POINTS:
        for arm in ("signal", "reverse"):
            base, candidate = vectors[point, "uniform"], vectors[point, arm]
            delta = {axis: candidate[axis] - base[axis] for axis in AXES}
            comparisons.append({"point": point, "baseline": "uniform", "arm": arm,
                                "actual_axis_delta": delta,
                                "same_recorded_vector": all(v == 0 for v in delta.values())})
    return {"consumer_sha256": sha256, "comparisons": comparisons,
            "scope": "pinned saved-consumer counts only; no independent journal replay",
            "equal_total_compute_certified": False, "scientific_pass": False,
            "unknown": ["per-arm wall", "FLOPs", "energy", "model load",
                        "dynamic source closure"],
            "warning": "nested resource axes are not additive"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consumer", type=Path)
    parser.add_argument("--sha256", required=True, help="expected SHA-256 of consumer bytes")
    args = parser.parse_args()
    raw = args.consumer.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.sha256:
        raise ValueError("consumer SHA-256 mismatch")
    print(json.dumps(audit(json.loads(raw), digest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
