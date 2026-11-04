"""Check shared initial context and audited v7/CEM Reacher development outcomes."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_reacher_matched_warm_cem_dev_20261104_codex as cem
from experiments import pc_v7_reacher_rgb_live_20261104_codex as v7

OUTPUT = cem.PREFIX.with_name(cem.PREFIX.name + "_paired_comparison.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifests = [json.loads(path.read_text()) for path in (v7.MANIFEST, cem.MANIFEST)]
    if any(m["root"] != 20263160 or m["budget"] != 1920 for m in manifests):
        raise ValueError("root or budget mismatch")
    if manifests[0]["method"] != v7.METHOD or manifests[1]["method"] != cem.prior.method.METHOD:
        raise ValueError("method mismatch")
    v7_context = v7.PREFIX.with_name(v7.PREFIX.name + "_response_context0.npz")
    cem_context = cem.PREFIX.with_name(cem.PREFIX.name + "_smooth_cem_context0.npz")
    with np.load(v7_context, allow_pickle=False) as left, np.load(cem_context, allow_pickle=False) as right:
        if set(left.files) != set(right.files) or not all(np.array_equal(left[k], right[k]) for k in left.files):
            raise ValueError("initial observation or warmup mismatch")
    outcomes, dependencies = {}, {}
    for name, config, arm in (("v7", v7, "response"), ("cem", cem, "smooth_cem")):
        manifest = config.MANIFEST
        audit = config.PREFIX.with_name(config.PREFIX.name + "_audit.json")
        prediction = config.PREFIX.with_name(config.PREFIX.name + "_prediction_replay.json")
        physical = config.PREFIX.with_name(config.PREFIX.name + "_physical_replay.json")
        execution = config.PREFIX.with_name(config.PREFIX.name + f"_{arm}_execution.json")
        a, p, r, e = (json.loads(path.read_text()) for path in (audit, prediction, physical, execution))
        if (a["manifest_sha256"] != sha(manifest) or p["audit_sha256"] != sha(audit)
                or r["audit_sha256"] != sha(audit) or e["manifest_sha256"] != sha(manifest)
                or len(p["rows"]) != 3 or len(r["rows"]) != 1):
            raise ValueError(f"{name} audit binding mismatch")
        outcomes[name] = [x["distance"] for x in e["initial_and_block_outcomes"]]
        dependencies.update({str(path): sha(path) for path in (manifest, audit, prediction, physical, execution)})
    if outcomes["v7"][0] != outcomes["cem"][0]:
        raise ValueError("initial physical metric mismatch")
    result = {"scope": "one exposed development root; separate feedback trajectories",
              "shared_initial_arrays": ["frames", "root", "goal", "history"],
              "v7_distances": outcomes["v7"], "cem_distances": outcomes["cem"],
              "final_distance_difference_v7_minus_cem": outcomes["v7"][-1]-outcomes["cem"][-1],
              "context_sha256": {"v7": sha(v7_context), "cem": sha(cem_context)},
              "dependencies": dependencies}
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
