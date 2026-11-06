"""Reconstruct finite-difference actions and scored incumbents from traces."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_suffix_fd_baseline_e0_dev_20261104_codex as source
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
            raise ValueError("trace mismatch")
        with np.load(out / row["trace"], allow_pickle=False) as saved:
            for run, label in zip(row["runs"], row["labels"], strict=True):
                selected = np.asarray(run["chain"], np.float32)
                matches = []
                for index in [0, *range(label["first_batch"], label["last_batch"])]:
                    actions, scores = saved[f"actions_{index}"], saved[f"scores_{index}"]
                    matches.extend(float(scores[j]) for j in np.flatnonzero(np.all(actions == selected, axis=(1, 2))))
                if not matches or not any(abs(x-run["best"]) <= 1e-12 for x in matches):
                    raise ValueError("selected chain was not scored")
                if run["arm"] != "fd":
                    continue
                index = label["first_batch"]
                for generation in run["generations"]:
                    center = np.asarray(generation["center"], np.float32)
                    dim = center.size
                    axis = np.eye(dim).reshape(dim, *center.shape)
                    plus = np.clip(center[None] + .05 * axis, -1, 1).astype(np.float32)
                    minus = np.clip(center[None] - .05 * axis, -1, 1).astype(np.float32)
                    probes, scores = saved[f"actions_{index}"], saved[f"scores_{index}"]
                    np.testing.assert_array_equal(probes[:, 5:], np.concatenate((plus, minus)))
                    denom = (plus.reshape(dim, dim)[np.arange(dim), np.arange(dim)]
                             - minus.reshape(dim, dim)[np.arange(dim), np.arange(dim)])
                    gradient = (scores[:dim] - scores[dim:]) / denom.astype(np.float64)
                    np.testing.assert_allclose(gradient, generation["gradient"], atol=1e-12, rtol=0)
                    index += 1
                    n = generation["line_samples"]
                    if n:
                        steps = (.1, .25, .5, 1.)[:n]
                        expected = np.stack([np.clip(center - step*(gradient/np.max(np.abs(gradient))).reshape(center.shape),
                                                     -1, 1) for step in steps]).astype(np.float32)
                        np.testing.assert_array_equal(saved[f"actions_{index}"][:, 5:], expected)
                        np.testing.assert_allclose(saved[f"scores_{index}"], generation["line_scores"], atol=1e-12, rtol=0)
                        index += 1
                    checked += 1
                if index != label["last_batch"]:
                    raise ValueError("unaccounted FD batch")
    seal.verify(manifest["dependencies"])
    record = {"status": "passed", "fd_generations": checked, "result_sha256": seal.digest(out / "result.json"),
              "source_audit_sha256": seal.digest(out / "audit.json"), "script_sha256": seal.digest(Path(__file__)),
              "scope": "saved finite-difference and selection reconstruction"}
    source.prior.source.prior.prior.prior.writer(out / "trace_audit.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
