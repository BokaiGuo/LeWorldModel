"""Independently verify both PointMaze initial-context hashes and paired frames."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex as configured
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = configured.PREFIX.with_name(configured.PREFIX.name + "_pair_binding.json")


def resolve(prefix):
    make = lambda suffix: prefix.with_name(prefix.name + suffix)
    return {key: make(suffix) for key, suffix in (
        ("manifest", "_manifest.json"), ("audit", "_audit.json"),
        ("execution", "_response_execution.json"), ("images", "_response_images.npz"),
        ("decision0", "_response_decision0.json"))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    a, b = resolve(configured.v7.PREFIX), resolve(configured.PREFIX)
    for group in (a, b):
        audit, receipt = [json.loads(group[key].read_text()) for key in ("audit", "execution")]
        if (audit["manifest_sha256"] != seal.digest(group["manifest"])
                or audit["execution_sha256"] != seal.digest(group["execution"])
                or audit["images_sha256"] != seal.digest(group["images"])
                or receipt["images_sha256"] != seal.digest(group["images"])
                or receipt["rows"][0]["decision_sha256"] != seal.digest(group["decision0"])):
            raise ValueError("paired input hash mismatch")
    with np.load(a["images"], allow_pickle=False) as one, np.load(b["images"], allow_pickle=False) as two:
        np.testing.assert_array_equal(one["rgb"][:16], two["rgb"][:16])
    first, second = [json.loads(group["decision0"].read_text())["binding"] for group in (a, b)]
    keys = ("ticks", "rgb_sha256", "previous_rgb_sha256", "history_actions", "goal_sha256",
            "goal_position", "model_hash", "checkpoint_sha256", "action_mean", "action_std")
    if any(first[key] != second[key] for key in keys):
        raise ValueError("initial context mismatch")
    deps = seal.capture([Path(__file__), *a.values(), *b.values()])
    seal.verify(deps)
    configured.live.old.write_json(OUT, {"dependencies": deps, "initial_frames_equal": 16,
                                         "initial_fields_equal": list(keys),
                                         "scope": "one exposed root; record binding only"})
    print("Both source hashes and paired initial contexts verified.", flush=True)


if __name__ == "__main__":
    main()
