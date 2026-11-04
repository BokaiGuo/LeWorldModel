"""Audit resource replay accounting from original traces and saved timings."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_e0_resource_replay_dev_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.OUT.with_name(run.OUT.stem + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest, result, plans, baselines = (json.loads(p.read_text()) for p in
        (run.MANIFEST, run.OUT, run.panel.OUT, run.baseline.OUT))
    seal.verify(manifest["dependencies"])
    if result["manifest_sha256"] != seal.digest(run.MANIFEST):
        raise ValueError("manifest mismatch")
    original = {(r["root"], "v7"): r for r in plans["rows"]}
    original.update({(r["root"], r["arm"]): r for r in baselines["rows"]})
    records = {(r["root"], r["arm"]): r for r in result["rows"]}
    if len(records) != 54 or set(records) != set(original):
        raise ValueError("panel coverage mismatch")
    structural_tokens = []
    for key, row in records.items():
        source, arm = original[key], key[1]
        calls = len(source["trace"])
        samples = sum(len(b["raw"]) for b in source["trace"])
        tokens = sum(len(b["raw"])*len(b["raw"][0])//5 for b in source["trace"])
        if (row["trace_batches_verified"] != calls or row["counts"]["prediction_calls"] != calls
                or row["counts"]["trajectory_samples"] != samples
                or row["counts"]["prediction_token_samples"] != tokens or tokens != 1920):
            raise ValueError("prediction accounting mismatch")
        if row["planning_ns"] <= 0 or row["counts"]["prediction_seconds"] < 0:
            raise ValueError("invalid measured duration")
        timings = row["inclusive_call_timing"]
        if any(t["calls"] < 1 or not 0 <= t["inclusive_ns"] <= row["planning_ns"]
               for t in timings.values()):
            raise ValueError("invalid inclusive time")
        if arm == "v7":
            result_row = source["result"]
            probes = result_row["first_response"] + result_row["successor_responses"]
            if (len(probes) != timings["response"]["calls"]
                    or timings["svd"]["calls"] != result_row["structure"]["svd"]
                    or result_row["structure"]["fd_action_samples"] != 21*len(probes)):
                raise ValueError("response/SVD accounting mismatch")
            first = source["trace"][:2]
            successor = source["trace"][3:3+len(result_row["successor_responses"])]
            selected = first + successor
            if any(len(b["raw"]) != 21 for b in selected):
                raise ValueError("probe trace layout changed")
            structural_tokens.append(sum(len(b["raw"])*len(b["raw"][0])//5 for b in selected))
        elif timings["baseline_cem"]["calls"] != 3 or "svd" in timings:
            raise ValueError("unexpected baseline operations")
    for arm in run.ARMS:
        data = [r for (root, name), r in records.items() if name == arm]
        median = np.median([r["planning_ns"] for r in data])/1e9
        np.testing.assert_allclose(result["summary"][arm]["median_planning_seconds"], median, rtol=0, atol=1e-15)
    timing = {arm: [records[(root, arm)]["planning_ns"]/1e9 for root in range(18)] for arm in run.ARMS}
    run.panel.source.write(OUT, {"result_sha256": seal.digest(run.OUT),
        "script_sha256": seal.digest(Path(__file__)), "verified_plans": len(records),
        "fd_prediction_tokens": structural_tokens, "planning_seconds": timing,
        "scope": "recounts original trace and saved durations; does not independently rerun profiling"})
    print("Audited all 54 plans, trace budgets, structural calls and saved medians.", flush=True)


if __name__ == "__main__":
    main()
