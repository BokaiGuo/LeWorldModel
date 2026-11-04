"""Independently check the saved diagnostic's exact response probe inputs."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_response_retention_fixedbank_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    output = source.OUT / "probe_input_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    result = json.loads((source.OUT / "result.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if (result["manifest_sha256"] != seal.digest(source.OUT / "manifest.json")
            or audit["result_sha256"] != seal.digest(source.OUT / "result.json")):
        raise ValueError("source evidence binding mismatch")
    axis = np.eye(20).reshape(20, 10, 2)
    suffix = np.concatenate((np.zeros((1, 10, 2)), .05 * axis, -.05 * axis)).astype(np.float32)
    checked = 0
    for row in result["rows"]:
        if row["status"] != "completed" or seal.digest(source.OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("incomplete or modified source trace")
        with np.load(source.OUT / row["trace"], allow_pickle=False) as data:
            for candidate in row["candidates"]:
                batch = candidate["response_batch"]
                actions, scores = data[f"actions_{batch}"], data[f"scores_{batch}"]
                np.testing.assert_array_equal(actions[:, 5:], suffix)
                np.testing.assert_array_equal(actions[:, :5],
                    np.broadcast_to(np.asarray(candidate["prefix"], np.float32), (41, 5, 2)))
                if float(scores[0]) != candidate["zero_tail_score"]:
                    raise ValueError("zero-tail incumbent mismatch")
                checked += 1
    seal.verify(manifest["dependencies"])
    source.prior.prior.prior.writer(output, {"status": "passed", "candidate_probes": checked,
        "result_sha256": seal.digest(source.OUT / "result.json"), "audit_sha256": seal.digest(source.OUT / "audit.json"),
        "audit_script_sha256": seal.digest(Path(__file__)),
        "scope": "saved probe input and binding consistency; not independent replication"})
    print(f"verified {checked} response probe batches")


if __name__ == "__main__":
    main()
