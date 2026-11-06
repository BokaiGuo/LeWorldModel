"""Describe stopped-bank coverage of privileged witness prefixes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_retention_feasibility_posthoc_20261104_codex as witness
from experiments import pc_response_retention_fixedbank_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_RESPONSE_RETENTION_WITNESS_COVERAGE_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = source.ROOT / "results/pc_response_retention_witness_coverage_posthoc_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    result = json.loads((source.OUT / "result.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    saved = json.loads(witness.OUT.read_text())
    seal.verify(manifest["dependencies"])
    seal.verify(saved["dependencies"])
    result_hash = seal.digest(source.OUT / "result.json")
    if (result["manifest_sha256"] != seal.digest(source.OUT / "manifest.json")
            or audit["status"] != "passed" or audit["result_sha256"] != result_hash
            or saved["source_result_sha256"] != result_hash
            or saved["source_audit_sha256"] != seal.digest(source.OUT / "audit.json")):
        raise ValueError("source binding mismatch")
    old = source.prior.prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(source.__file__), Path(witness.__file__),
                         Path(old.previous.independent.__code__.co_filename)])
    rows = []
    for root, (row, feasible) in enumerate(zip(result["rows"], saved["rows"], strict=True)):
        if row["status"] != "completed" or row["root"] != root or feasible["root"] != root or row["task"] != feasible["task"]:
            raise ValueError("invalid source task")
        prefixes = np.asarray([c["prefix"] for c in row["candidates"]], np.float32)
        paths, _ = old.previous.independent(row["task"]["start"], prefixes)
        states = paths[:, -1]
        suffix_bests = [min(run["best"] for run in c["runs"]) for c in row["candidates"]]
        groups = {"all": list(range(len(prefixes))), **row["selected"]}
        attempts = []
        for item in feasible["attempts"]:
            if not item.get("success"):
                continue
            actions = np.asarray(item["actions"], np.float32)
            replay, _ = old.previous.independent(row["task"]["start"], actions[None])
            np.testing.assert_allclose(replay[0], item["path"], atol=1e-12, rtol=0)
            target = replay[0, 5]
            rms = np.sqrt(np.mean((prefixes - actions[None, :5]) ** 2, axis=(1, 2)))
            pos = np.linalg.norm(states[:, :2] - target[:2], axis=1)
            heading = np.abs((states[:, 2] - target[2] + np.pi) % (2 * np.pi) - np.pi)
            group_rows = {}
            for name, indices in groups.items():
                indexes = np.asarray(indices, dtype=int)
                group_rows[name] = {}
                for metric, distances in (("action_rms", rms), ("position", pos), ("heading", heading)):
                    index = int(indexes[np.argmin(distances[indexes])])
                    group_rows[name][metric] = {"distance": float(distances[index]), "candidate": index,
                                                "suffix_best": float(suffix_bests[index])}
            attempts.append({"height": item["height"], "witness_prefix_state": target.tolist(), "groups": group_rows})
        rows.append({"root": root, "candidate_states": states.tolist(),
                     "candidate_suffix_bests": suffix_bests, "witnesses": attempts})
    seal.verify(deps)
    old.writer(OUT, {"source_result_sha256": result_hash,
        "feasibility_sha256": seal.digest(witness.OUT), "dependencies": deps, "rows": rows,
        "scope": "post-hoc privileged coverage only, no planner input or causal claim"})
    print(json.dumps([{"root": row["root"], "witnesses": [
        {"height": a["height"], "all": a["groups"]["all"]} for a in row["witnesses"]]} for row in rows], indent=2))


if __name__ == "__main__":
    main()
