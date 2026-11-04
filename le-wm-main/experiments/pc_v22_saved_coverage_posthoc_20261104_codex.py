"""Descriptive action-coverage analysis of the sealed v22 paired panel."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_reserve_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.OUT / "coverage_posthoc.json"
PROTOCOL = source.ROOT.parent / "docs/PC_V22_SAVED_COVERAGE_DIAGNOSTIC_POSTHOC_PROTOCOL_20261104_CODEX.md"


def main():
    manifest_path = source.OUT / "manifest.json"
    plans_path = source.OUT / "plans.json"
    audit_path = source.OUT / "audit.json"
    manifest = json.loads(manifest_path.read_text())
    plans = json.loads(plans_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(plans_path):
        raise ValueError("paired panel audit missing or mismatched")
    seal.verify(manifest["dependencies"])
    by_key = {(r["root"], r["arm"]): r for r in plans["rows"]}
    if len(by_key) != len(manifest["tasks"]) * len(source.ARMS):
        raise ValueError("duplicate or missing rows")
    rows = []
    for root, task in enumerate(manifest["tasks"]):
        method, baseline = (by_key[(root, arm)] for arm in ("v22", "multistart_cem"))
        for row in (method, baseline):
            if (row["status"] != "completed" or row["task"] != task
                    or seal.digest(source.OUT / row["trace"]) != row["trace_sha256"]):
                raise ValueError("failed or altered source")
        ref_result = baseline["result"]["result"]
        reference = np.asarray(ref_result["selected"], dtype=np.float32)
        if reference.shape != (15, 2):
            raise ValueError("unexpected reference shape")
        layer = method["result"]["result"]["child"]["layers"][0]
        bank = np.asarray([item["chain"] for item in layer["candidates"]], dtype=np.float32)
        retained = np.asarray(layer["retained"], dtype=int)
        if bank.ndim != 3 or bank.shape[1:] != (5, 2) or not len(retained):
            raise ValueError("invalid candidate bank")
        if len(set(retained.tolist())) != len(retained) or np.any((retained < 0) | (retained >= len(bank))):
            raise ValueError("invalid retained indices")
        rms = np.sqrt(np.mean((bank.astype(np.float64) - reference[None, :5]) ** 2, axis=(1, 2)))
        full_scores, exact = [], 0
        with np.load(source.OUT / method["trace"], allow_pickle=False) as trace:
            for i, call in enumerate(method["result"]["calls"]):
                actions, scores = trace[f"actions_{i}"], trace[f"scores_{i}"]
                if len(actions) != call["samples"] or len(scores) != len(actions):
                    raise ValueError("trace and call ledger differ")
                if actions.shape[1:] == reference.shape:
                    full_scores.extend(np.asarray(scores, np.float64).tolist())
                    exact += int(np.all(actions == reference, axis=(1, 2)).sum())
        if not full_scores or not np.isfinite(full_scores).all():
            raise ValueError("missing full-horizon scores")
        all_index = int(np.argmin(rms))
        kept_index = int(retained[np.argmin(rms[retained])])
        rows.append({"root": root, "reference_score": ref_result["selected_score"],
                     "reference_success": ref_result["selected_score"] < .05,
                     "candidate_count": len(bank), "retained": retained.tolist(),
                     "nearest_candidate": all_index, "candidate_rms": float(rms[all_index]),
                     "nearest_retained": kept_index, "retained_rms": float(rms[kept_index]),
                     "exact_prefix_matches": int(np.count_nonzero(rms == 0)),
                     "full_chain_samples": len(full_scores), "best_full_chain_score": min(full_scores),
                     "successful_full_chain_samples": sum(x < .05 for x in full_scores),
                     "exact_reference_full_chain_matches": exact})
    seal.verify(manifest["dependencies"])
    result = {"scope": "post hoc saved-data description, not causal attribution or confirmation",
              "plans_sha256": seal.digest(plans_path), "audit_sha256": seal.digest(audit_path),
              "protocol_sha256": seal.digest(PROTOCOL), "rows": rows}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
