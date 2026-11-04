"""Post-seal privileged feasibility witnesses, never planning inputs."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_response_retention_fixedbank_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_V22_RESPONSE_RETENTION_FEASIBILITY_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = source.ROOT / "results/pc_v22_response_retention_feasibility_posthoc_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    result = json.loads((source.OUT / "result.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    probes = json.loads((source.OUT / "probe_input_audit.json").read_text())
    seal.verify(manifest["dependencies"])
    result_hash = seal.digest(source.OUT / "result.json")
    if (result["manifest_sha256"] != seal.digest(source.OUT / "manifest.json")
            or audit["status"] != "passed" or audit["result_sha256"] != result_hash
            or probes["status"] != "passed" or probes["candidate_probes"] != 40
            or probes["result_sha256"] != result_hash
            or probes["audit_sha256"] != seal.digest(source.OUT / "audit.json")):
        raise ValueError("source is not sealed and audited")
    if any(row["status"] != "completed" for row in result["rows"]):
        raise ValueError("source panel incomplete")
    old = source.prior.prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(source.__file__),
        Path(old.exact.__file__), Path(old.exact.wall.__file__),
        Path(old.exact.wall.physics.__file__), Path(old.previous.independent.__code__.co_filename)])
    rows = []
    for root, task in enumerate(source.tasks()):
        attempts = []
        for item in old.exact.wall.feasibility(task):
            record = {"height": item["height"], "steps_needed": item["steps_needed"],
                      "eligible": bool(item["within_horizon"] and item["steps_needed"] <= 15)}
            if record["eligible"]:
                actions = (np.asarray(item["actions"][:15], np.float64) / old.exact.wall.HIGH).astype(np.float32)
                if actions.shape != (15, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                    raise ValueError("invalid witness")
                positions = old.exact.endpoints(task, actions[None])
                path, blocked = old.previous.independent(task["start"], actions[None])
                np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                distance = float(np.linalg.norm(positions[0] - task["goal"]))
                record.update({"actions": actions.tolist(), "path": path[0].tolist(),
                               "blocked": blocked[0].tolist(), "distance": distance, "success": distance < .05})
            attempts.append(record)
        rows.append({"root": root, "task": task, "attempts": attempts})
    seal.verify(deps)
    old.writer(OUT, {"source_result_sha256": result_hash,
        "source_audit_sha256": seal.digest(source.OUT / "audit.json"),
        "probe_audit_sha256": seal.digest(source.OUT / "probe_input_audit.json"),
        "dependencies": deps, "rows": rows, "scope": "privileged diagnostic only"})
    print(json.dumps([{"root": row["root"], "successes": sum(a.get("success", False) for a in row["attempts"]),
                       "attempts": len(row["attempts"])} for row in rows], indent=2))


if __name__ == "__main__":
    main()
