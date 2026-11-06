"""Reconstruct seeded global/local suffix proposals from saved traces."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_suffix_global_exploration_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    result = json.loads((out / "result.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    if audit["status"] != "passed" or audit["result_sha256"] != seal.digest(out / "result.json"):
        raise ValueError("source audit mismatch")
    seal.verify(manifest["dependencies"])
    checked = 0
    for row in result["rows"]:
        if row["status"] != "completed":
            continue
        if seal.digest(out / row["trace"]) != row["trace_sha256"]:
            raise ValueError("altered trace")
        with np.load(out / row["trace"], allow_pickle=False) as saved:
            for run, label in zip(row["runs"], row["labels"][1:], strict=True):
                if run["arm"] != "global_exploration":
                    continue
                seed = 2030020000 + 10 * row["root"] + run["restart"]
                rng = np.random.default_rng(seed)
                loc = np.zeros((10, 2), np.float64)
                scale = np.full((10, 2), .35)
                incumbent = np.zeros((10, 2), np.float32)
                best = float(row["initial_score"])
                indices = range(label["first_batch"], label["last_batch"])
                if len(run["generations"]) != len(indices):
                    raise ValueError("generation count mismatch")
                for index, generation in zip(indices, run["generations"], strict=True):
                    actions, scores = saved[f"actions_{index}"], saved[f"scores_{index}"]
                    n = len(actions)
                    global_count = min(n - 1, int(np.floor(n * .25)))
                    local_count = n - global_count
                    expected = np.empty((n, 10, 2), np.float64)
                    expected[:local_count] = loc + rng.normal(size=(local_count, 10, 2)) * scale
                    if global_count:
                        expected[local_count:] = rng.uniform(-1., 1., size=(global_count, 10, 2))
                    expected[0] = incumbent
                    expected = np.clip(expected, -1., 1.).astype(np.float32)
                    np.testing.assert_array_equal(expected, actions[:, 5:])
                    np.testing.assert_allclose(generation["mean_before"], loc, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(generation["scale_before"], scale, rtol=0, atol=1e-12)
                    if generation["size"] != n or generation["global_count"] != global_count:
                        raise ValueError("invalid recorded allocation")
                    choice = int(np.argmin(scores))
                    if scores[choice] < best:
                        incumbent, best = expected[choice].copy(), float(scores[choice])
                    elite = expected[np.argsort(scores, kind="stable")[:min(n, 4)]].astype(np.float64)
                    loc, scale = elite.mean(axis=0), np.maximum(elite.std(axis=0), .07)
                    checked += 1
                if best != run["best"]:
                    raise ValueError("best score mismatch")
    if checked != 4 * 3 * 10:
        raise ValueError("unexpected number of generations")
    seal.verify(manifest["dependencies"])
    record = {"status": "passed", "generations": checked,
              "result_sha256": seal.digest(out / "result.json"), "audit_sha256": seal.digest(out / "audit.json"),
              "audit_script_sha256": seal.digest(Path(__file__)),
              "scope": "seeded saved-sampling reconstruction, not optimizer sufficiency"}
    source.source.prior.prior.prior.writer(out / "sampling_audit.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
