"""Check every shared-tail optimizer incumbent is an actually scored chain."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_modecount_common_tail_replan_posthoc_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.OUT / "selection_audit.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((run.OUT / "manifest.json").read_text())
    result = json.loads((run.OUT / "result.json").read_text())
    audit = json.loads((run.OUT / "audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if audit["result_sha256"] != seal.digest(run.OUT / "result.json"):
        raise ValueError("missing source audit")
    checked = 0
    for row in result["rows"]:
        if row["status"] != "completed":
            continue
        if seal.digest(run.OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("trace changed")
        with np.load(run.OUT / row["trace"], allow_pickle=False) as trace:
            for name in run.NAMES:
                runs = row["controls"][name]["runs"]
                for item in runs:
                    selected = np.asarray(item["chain"], np.float32)
                    found = False
                    for i in range(item["first_batch"], item["last_batch"]):
                        actions, scores = trace[f"actions_{i}"], trace[f"scores_{i}"]
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                             & (np.abs(scores-item["score"]) <= 1e-12)))
                    if not found or item["score"] > item["initial_score"] + 1e-12:
                        raise ValueError("selected chain not scored or incumbent lost")
                    checked += 1
                np.testing.assert_allclose(row["controls"][name]["mean_score"],
                                           np.mean([item["score"] for item in runs]), atol=1e-12, rtol=0)
    seal.verify(manifest["dependencies"])
    run.source.mechanism.writer(OUT, {"result_sha256": seal.digest(run.OUT / "result.json"),
                                     "score_audit_sha256": seal.digest(run.OUT / "audit.json"),
                                     "script_sha256": seal.digest(Path(__file__)), "checked_runs": checked,
                                     "scope": "saved action/score identity, not independent optimizer replay"})
    print(f"Verified {checked} selected scored incumbents.")


if __name__ == "__main__":
    main()
