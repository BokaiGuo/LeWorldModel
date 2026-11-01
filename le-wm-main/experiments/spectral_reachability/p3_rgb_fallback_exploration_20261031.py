"""Retrospective cached-old-suffix fallback, using saved real physical branches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import p3_rgb_guarded_h3_physical_20261031 as source

ROOT = source.ROOT
NAME = "p3_rgb_fallback_exploration_20261031"
GUARDED = ROOT / "results/p3_rgb_guarded_h3_summary_20261031_RUN01.json"
BASE = ROOT / "results/p3_h3_real_context_physical_summary_mixed_20261031_RUN01.json"
GUARDED_SHA = "c8d4b1090da484ae417a87b7c13fa7142aa6cc55e881478f4441e1273427cd90"
BASE_SHA = "e4a75c39eb152643d40f2779245907f59922e7681197c2de0b9ba4cbbdefe477"
METRICS = ("object_position_error", "object_angle_error_wrapped",
           "pusher_position_error", "native_joint_xy_error")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(guarded, prior):
    require(guarded["pair_key"] == prior["pair_key"]
            and guarded["identity"] == prior["identity"], "pair identity")
    require(prior["old_status"] == "horizon_completed"
            and prior["old_returned"] == 20 and prior["old_endpoint"] is not None,
            "cached suffix physical endpoint")
    old = prior["old_endpoint"]
    if guarded["new_status"] == "rgb_preflight_abstain":
        require(guarded["old_status"] == "rgb_preflight_abstain"
                and guarded["old_returned"] == guarded["new_returned"] == 15
                and guarded["old_endpoint"] is None and guarded["new_endpoint"] is None,
                "paired guard")
        return "cached_old_counterfactual", old, old
    require(guarded["old_status"] == guarded["new_status"] == "horizon_completed"
            and guarded["old_returned"] == guarded["new_returned"] == 20
            and guarded["old_endpoint"] == old and guarded["new_endpoint"] is not None
            and prior["new_status"] == "horizon_completed"
            and prior["new_endpoint"] == guarded["new_endpoint"], "admitted replay")
    return "guarded_new_observed", guarded["new_endpoint"], old


def analyze(guarded, prior):
    require(guarded["schema"] == "p3_rgb_guarded_h3_summary_20261031"
            and prior["schema"] == "p3_h3_real_context_physical_summary_mixed_20261031"
            and all(x["P3_admission"] is False and x["independent_confirmation"] is False
                    and x["equal_budget_certified"] is False for x in (guarded, prior)),
            "sources")
    def key(row):
        return (row["identity"]["logical_pair"], row["identity"]["source_variant"])
    left = {key(row): row for row in guarded["rows"]}
    right = {key(row): row for row in prior["rows"]}
    require(len(left) == len(right) == len(guarded["rows"]) == len(prior["rows"]) == 52
            and set(left) == set(right), "fixed roster")
    rows, unique = [], {}
    for identity in sorted(left):
        a, b = left[identity], right[identity]
        origin, endpoint, baseline = policy(a, b)
        delta = {metric: endpoint[metric] - baseline[metric] for metric in METRICS}
        item = {"identity": a["identity"], "pair_key": a["pair_key"],
                "origin": origin, "endpoint": endpoint, "old_endpoint": baseline,
                "delta_vs_old": delta, "success": endpoint["native_success"]}
        comparable = {k: item[k] for k in ("origin", "endpoint", "old_endpoint", "delta_vs_old", "success")}
        if item["pair_key"] in unique:
            require(unique[item["pair_key"]] == comparable, "duplicate actions disagree")
        else:
            unique[item["pair_key"]] = comparable
        rows.append(item)
    require(len(unique) == 26, "unique pair roster")
    outcomes = list(unique.values())
    summary = {"unique_pairs": len(outcomes),
               "selected_origins": dict(Counter(x["origin"] for x in outcomes)),
               "native_successes": sum(x["success"] for x in outcomes),
               "delta_directions": {
                   m: dict(Counter("lower" if x["delta_vs_old"][m] < -1e-6 else
                                   "higher" if x["delta_vs_old"][m] > 1e-6 else "tie"
                                   for x in outcomes)) for m in METRICS}}
    return {"schema": NAME, "status": "complete", "rows": rows, "summary": summary,
            "P1_admission": False, "P3_admission": False, "independent_confirmation": False,
            "equal_budget_certified": False,
            "limitations": "Retrospective branch substitution, not executed online; two known roots, "
                           "repeated actions, not safety or equal-budget evidence."}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    require(out.is_absolute() and out.parent == ROOT / "results"
            and out.name.startswith(NAME + "_") and out.suffix == ".json"
            and not out.exists() and not out.is_symlink(), "exclusive output")
    paths = ((GUARDED, GUARDED_SHA), (BASE, BASE_SHA))
    require(all(sha(p) == h for p, h in paths), "source hashes")
    print("Compare existing guarded new branches and cached-old physical outcomes; no online fallback.", flush=True)
    result = analyze(*(json.loads(p.read_bytes()) for p, _ in paths))
    result["sources"] = {str(p): h for p, h in paths}
    result["code_sha256"] = sha(Path(__file__).resolve())
    require(all(sha(p) == h for p, h in paths), "source drift")
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"summary": result["summary"], "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
