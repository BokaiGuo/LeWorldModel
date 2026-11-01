"""Strictly consume all matched-scale physical probes and retain paired responses."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_matched_suffix_scale_20261030 as experiment

runner, consume, require = experiment.runner, experiment.consume, experiment.require


def response(anchor, endpoints):
    require(len(endpoints) == 21, "fixed endpoints")
    goal = np.asarray(anchor["goal"], dtype=np.float64)
    require(goal.shape == (7,) and np.isfinite(goal).all(), "goal")
    if endpoints[0] is None:
        return [None] * 21
    base = np.asarray(endpoints[0], dtype=np.float64)
    require(base.shape == (7,) and np.isfinite(base).all(), "zero endpoint")
    def metrics(state):
        state = np.asarray(state, dtype=np.float64)
        require(state.shape == (7,) and np.isfinite(state).all(), "physical endpoint")
        angle = float(state[4] - goal[4])
        return float(np.linalg.norm(state[2:4] - goal[2:4])), abs(math.atan2(math.sin(angle), math.cos(angle)))
    position0, angle0 = metrics(base)
    values = []
    for endpoint in endpoints:
        if endpoint is None:
            values.append(None)
        else:
            pos, angle = metrics(endpoint)
            values.append(dict(position_error=pos, angle_error=angle,
                               position_gain_vs_zero=position0 - pos, angle_gain_vs_zero=angle0 - angle))
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.prepared)
    require(runner.physical.sha(source) == args.sha256, "prepare SHA")
    p = runner.physical.native.read(source, cap=32 * 1024**2)
    require(p == experiment.build(), "source and prepare identity")
    directory = Path(args.directory)
    binding_path = directory / "run_binding.json"
    binding_sha = runner.physical.sha(binding_path)
    binding = runner.physical.native.read(binding_path, cap=32 * 1024**2)
    require(binding["prepared"] == str(source.resolve()) and binding["prepare_sha256"] == args.sha256 and
            binding["fixed_branches"] == 336 and
            all(binding["source_code_bindings"].get(k) == v for k, v in p["bindings"].items()), "run binding")
    rows, hashes = [], []
    for anchor, probes in zip(p["anchors"], p["probe_sets"], strict=True):
        path = directory / f"anchor_{anchor['index']:02d}"
        digest = runner.physical.sha(path / "report.json")
        endpoints = consume.check_shard(path, anchor, probes)
        require(runner.physical.sha(path / "report.json") == digest, "report drift")
        rows.append(dict(anchor=anchor["index"], root_slot=anchor["root_slot"], condition=anchor["condition"],
                         arm=anchor["arm"], role=anchor["role"], probes=probes,
                         complete_endpoints=sum(x is not None for x in endpoints),
                         response=response(anchor, endpoints)))
        hashes.append(dict(anchor=anchor["index"], report_sha256=digest))
    pairs = []
    for i in range(0, 16, 2):
        a, b = rows[i:i+2]
        require((a["role"], b["role"]) == ("incumbent", "model_selected") and
                all(a[k] == b[k] for k in ("root_slot", "condition", "arm", "probes")), "pair identity")
        values = []
        for probe, left, right in zip(a["probes"], a["response"], b["response"], strict=True):
            item = dict(name=probe["name"], selected_minus_incumbent=None)
            if left is not None and right is not None:
                item["selected_minus_incumbent"] = {key: right[key] - left[key] for key in left}
            values.append(item)
        pairs.append(dict(root_slot=a["root_slot"], condition=a["condition"], arm=a["arm"], responses=values))
    for path, digest in binding["source_code_bindings"].items():
        require(runner.physical.sha(path) == digest, "final source identity")
    require(runner.physical.sha(binding_path) == binding_sha, "binding identity")
    output = Path(args.output)
    require(output.is_absolute() and output.parent == experiment.ROOT / "results" and
            output.name.startswith(experiment.NAME + "_analysis_") and not output.exists()
            and not output.is_symlink(), "exclusive output")
    result = dict(schema=experiment.NAME + ".analysis", rows=rows, pairs=pairs, shard_hashes=hashes,
                  prepare_sha256=args.sha256, run_binding_sha256=binding_sha,
                  analysis_sha256=runner.physical.sha(Path(__file__)), P1_admission=False,
                  independent_confirmation=False, known_roots=2)
    raw = runner.physical.native.raw(result)
    with output.open("xb") as stream:
        require(stream.write(raw) == len(raw), "full result write")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    runner.prior.base.paired.fsync_dir(output.parent)
    print(json.dumps(dict(path=str(output), sha256=runner.physical.sha(output))))


if __name__ == "__main__":
    main()
