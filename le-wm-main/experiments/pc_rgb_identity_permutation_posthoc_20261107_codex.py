"""Enumerate rowwise probe-identity permutations on the exposed first pool."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL = ROOT / "results/pc_tworoom_hold_first_pool_physical_20261107_codex.json"
RUN = ROOT / "results/pc_tworoom_rgb_hold_retention_feedback_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_RGB_IDENTITY_PERMUTATION_POSTHOC_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_identity_permutation_posthoc_20261107_codex.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retain(costs, swaps):
    # The archived scorer returned float32; preserve its arithmetic and tie behavior.
    costs = np.asarray(costs, dtype=np.float32)
    if costs.shape != (3, 3) or not np.isfinite(costs).all():
        raise ValueError("expected finite 3x3 costs")
    if len(swaps) != 3 or any(x not in (0, 1) for x in swaps):
        raise ValueError("expected three binary swaps")
    first = int(np.argsort(costs.min(axis=1), kind="stable")[0])
    effects = costs[:, :1] - costs[:, 1:]
    effects = np.stack([row[::-1] if swap else row for row, swap in zip(effects, swaps, strict=True)])
    distances = np.square(effects - effects[first]).sum(axis=1)
    distances[first] = -np.inf
    return [first, int(np.argmax(distances))]


def analyze(physical, original):
    decisions = original["arms"]
    costs = np.asarray(decisions["effect"]["decisions"][0]["probe_cost"], dtype=np.float32)
    for arm in ("effect", "lookahead", "shuffled"):
        if decisions[arm]["decisions"][0]["probe_cost"] != costs.tolist():
            raise ValueError("candidate scores differ between arms")
    if not np.array_equal(costs[:, :1] - costs[:, 1:], np.asarray(physical["predicted_effects"])):
        raise ValueError("effects do not match stored candidate costs")
    labels = []
    for i, group in enumerate(physical["physical"]):
        if len(group) != 5 or any(len(entry["success"]) != 25 for entry in group):
            raise ValueError("invalid physical trace")
        prefix = any(group[0]["success"][:10])
        if prefix != physical["prefix_success"][i] or any(
            entry["success"][:10] != group[0]["success"][:10] for entry in group
        ):
            raise ValueError("physical prefix inconsistency")
        labels.append(not physical["initial_success"] and not prefix and
                      any(any(group[j]["success"][10:]) for j in (3, 4)))
    if labels != physical["useful_prefixes_independent_tails"]:
        raise ValueError("physical useful labels mismatch")
    results = []
    for swaps in itertools.product((0, 1), repeat=3):
        kept = retain(costs, swaps)
        results.append({"swaps": list(swaps), "retained": kept,
                        "useful_retained": sum(labels[i] for i in kept)})
    identity = results[0]
    if identity["retained"] != decisions["effect"]["decisions"][0]["retained"]:
        raise ValueError("identity differs from original solver")
    for arm in ("effect", "lookahead", "shuffled"):
        kept = decisions[arm]["decisions"][0]["retained"]
        if sum(labels[i] for i in kept) != physical["recall"][arm]["useful_retained"]:
            raise ValueError("recorded recall mismatch")
    baseline = identity["useful_retained"]
    return {"root": physical["root"], "useful": labels, "identity": identity,
            "lookahead": decisions["lookahead"]["decisions"][0]["retained"],
            "permutations": results,
            "better_than_identity": sum(r["useful_retained"] > baseline for r in results),
            "equal_to_identity": sum(r["useful_retained"] == baseline for r in results),
            "worse_than_identity": sum(r["useful_retained"] < baseline for r in results),
            "selection_changes": sum(r["retained"] != identity["retained"] for r in results)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    physical, original = (json.loads(p.read_text()) for p in (PHYSICAL, RUN))
    if physical["source_sha256"] != sha(RUN):
        raise ValueError("physical source does not match original run")
    if len(physical["rows"]) != len(original["rows"]):
        raise ValueError("root count mismatch")
    rows = []
    for p, o in zip(physical["rows"], original["rows"], strict=True):
        if (p["root"], p["row"]) != (o["root"], o["row"]):
            raise ValueError("root mismatch")
        rows.append(analyze(p, o))
    print("Post-hoc identity permutation sensitivity; physical labels are evaluation-only.", flush=True)
    summary = {
        "roots": len(rows),
        "roots_with_useful_prefix": sum(any(r["useful"]) for r in rows),
        "roots_selection_changes": sum(r["selection_changes"] > 0 for r in rows),
        "roots_some_shuffle_better": sum(r["better_than_identity"] > 0 for r in rows),
        "roots_some_shuffle_worse": sum(r["worse_than_identity"] > 0 for r in rows),
        "identity_useful_retained": sum(r["identity"]["useful_retained"] for r in rows),
        "all_shuffle_mean_useful_retained": sum(
            sum(p["useful_retained"] for p in r["permutations"]) / 8 for r in rows),
        "lookahead_useful_retained": sum(sum(r["useful"][i] for i in r["lookahead"]) for r in rows),
    }
    print(json.dumps(summary, indent=2), flush=True)
    result = {"scope": "post-hoc diagnostic; no additional model or environment calls",
              "inputs_sha256": {str(p): sha(p) for p in (PHYSICAL, RUN, PROTOCOL, Path(__file__))},
              "summary": summary, "rows": rows}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
