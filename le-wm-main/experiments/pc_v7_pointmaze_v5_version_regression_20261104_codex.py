"""Bind v7-v5 action equivalence on one exposed PointMaze root."""
import json

import numpy as np

from experiments import pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex as v5
from experiments import pc_v7_pointmaze_rgb_live_dev_20261104_codex as v7
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = v7.PREFIX.with_name(v7.PREFIX.name + "_v5_version_regression.json")


def path(prefix, suffix):
    return prefix.with_name(prefix.name + suffix)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    v7_audit = path(v7.PREFIX, "_audit.json")
    v5_audit = path(v5.PREFIX, "_audit.json")
    for prefix, audit_path in ((v7.PREFIX, v7_audit), (v5.PREFIX, v5_audit)):
        audit = json.loads(audit_path.read_text())
        if audit["manifest_sha256"] != seal.digest(path(prefix, "_manifest.json")):
            raise ValueError("manifest binding mismatch")
    deps = seal.capture([v7_audit, v5_audit, path(v7.PREFIX, "_manifest.json"),
                         path(v5.PREFIX, "_manifest.json"),
                         *[path(prefix, f"_response_decision{i}.json")
                           for prefix in (v7.PREFIX, v5.PREFIX) for i in range(3)]])
    same = []
    for i in range(3):
        current, old = [json.loads(path(prefix, f"_response_decision{i}.json").read_text())
                        for prefix in (v7.PREFIX, v5.PREFIX)]
        a = np.asarray(current["result"]["selected"]["chain"], np.float32)
        b = np.asarray(old["result"]["selected"]["chain"], np.float32)
        same.append(bool(np.array_equal(a, b)))
    seal.verify(deps)
    v7.live.old.write_json(OUT, {"dependencies": deps, "float32_action_chain_identical": same,
        "scope": "same exposed root only, not independent replication or algorithm equivalence"})
    print(f"Float32 selected chains identical: {same}", flush=True)


if __name__ == "__main__":
    main()
