"""Retrospective, CPU-only six-slot resource census; never grants admission."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

EXPECTED = "b45348e4dd8e74ee735d6c165148539b346fd21dd3b9aeb8e552ad524ec0ae1b"
ARMS = ("joint", "composition", "grasp")
AXES = ("predictor_samples", "AD_primal", "VJP", "J", "FLOPs")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(data):
    if data.get("schema") != "p2_saved_fair_pair_20261029.analysis":
        raise ValueError("unexpected schema")
    if data.get("planned_horizon_raw_steps") != 15 or data.get("known_roots") != 2:
        raise ValueError("unexpected protocol")
    expected = {(r, t, frozenset(pair)) for r in (0, 1) for t in (5, 10, 15)
                for pair in itertools.combinations(ARMS, 2)}
    seen, vectors, missing = set(), {}, []
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 18:
        raise ValueError("expected 18 pairs")
    for row in pairs:
        key = (row["root"], row["step"], frozenset((row["left"], row["right"])))
        if key not in expected or key in seen or type(row["comparable"]) is not bool:
            raise ValueError("invalid or duplicate pair")
        seen.add(key)
        if not row["comparable"]:
            missing.append({"root": row["root"], "step": row["step"],
                            "arms": sorted(key[2]), "reasons": row["missing_reasons"]})
        for side in ("left", "right"):
            arm, resources = row[side], row["resources_" + side]
            vector = {}
            for axis in AXES:
                if axis not in resources:
                    raise ValueError("missing resource axis")
                value = resources[axis]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("invalid resource value")
                vector[axis] = value
            identity = (row["root"], arm)
            if identity in vectors and vectors[identity] != vector:
                raise ValueError("inconsistent repeated resource vector")
            vectors[identity] = vector
    if seen != expected or len(missing) != 2:
        raise ValueError("incomplete pairing census")
    comparisons = []
    for root in (0, 1):
        a, b = vectors[root, "joint"], vectors[root, "composition"]
        comparisons.append({"root": root, "joint": a, "composition": b,
                            "axes": {k: ("unknown" if a[k] is None or b[k] is None
                                         else "equal" if a[k] == b[k] else "different")
                                     for k in AXES}})
    return {"schema": "p2_exclusive_resource_gap.v1", "pairs": len(seen),
            "comparable": len(seen) - len(missing), "missing": missing,
            "vectors": [{"root": r, "arm": a, "resources": vectors[r, a]}
                        for r in (0, 1) for a in ARMS],
            "joint_vs_composition": comparisons, "P2_admission": False,
            "fair_total_compute_verified": False, "upstream_reverified": False,
            "unmeasured_or_unverified": ["complete physical cost", "total FLOPs",
                "complete wall-time accounting", "candidate-level resource matching",
                "independent roots", "P1 admission"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if sha(raw) != EXPECTED:
        raise ValueError("input digest mismatch")
    report = audit(json.loads(raw))
    report["input_sha256"] = sha(raw)
    report["implementation_sha256"] = sha(Path(__file__).read_bytes())
    report["retrospective_known_roots"] = True
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
