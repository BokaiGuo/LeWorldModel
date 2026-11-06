"""Post-hoc comparison using only sealed v13 lookahead and compiler records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_v13_e0_prospective_mechanism_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V13_SAVED_RETENTION_SUFFIX_DIAGNOSTIC_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v13_saved_retention_suffix_20261104_codex.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    names = ("manifest.json", "plans.json", "plans_audit.json",
             "interventions.json", "interventions_audit.json")
    inputs = [Path(__file__), PROTOCOL, *(SOURCE / name for name in names)]
    hashes = {str(p): digest(p) for p in inputs}
    plans, pa, interventions, ia = [
        json.loads((SOURCE / name).read_text())
        for name in ("plans.json", "plans_audit.json", "interventions.json", "interventions_audit.json")
    ]
    if (pa["status"] != "passed" or ia["status"] != "passed"
            or pa["plans_sha256"] != digest(SOURCE / "plans.json")
            or ia["interventions_sha256"] != digest(SOURCE / "interventions.json")
            or plans["manifest_sha256"] != digest(SOURCE / "manifest.json")):
        raise ValueError("source seal or audit mismatch")
    pmap = {r["root"]: r for r in plans["rows"]}
    imap = {r["root"]: r for r in interventions["rows"]}
    if set(pmap) != set(range(12)) or set(imap) != set(range(12)):
        raise ValueError("missing or duplicated roots")
    if len(pmap) != len(plans["rows"]) or len(imap) != len(interventions["rows"]):
        raise ValueError("duplicate roots")
    output = []
    for root in range(12):
        plan, intervention = pmap[root], imap[root]
        if plan["status"] != "completed" or intervention["status"] != "completed":
            raise ValueError("incomplete source")
        data = intervention["data"]
        if data["task"] != plan["task"]:
            raise ValueError("task mismatch")
        layer = plan["result"]["child"]["layers"][0]
        candidates = layer["candidates"]
        bank, runs = data["bank"], data["selection_runs"]
        if len(candidates) != 10 or len(bank) != 11 or len(runs) != 11:
            raise ValueError("unexpected bank size")
        retained = np.asarray(layer["retained"], dtype=int)
        if len(retained) != 4 or len(np.unique(retained)) != 4 or np.any((retained < 0) | (retained >= 10)):
            raise ValueError("invalid retention")
        scores = []
        for index, (item, run) in enumerate(zip(bank, runs, strict=True)):
            prefix = np.asarray(item["prefix"], np.float32)
            chain = np.asarray(run["chain"], np.float32)
            if prefix.shape != (5, 2) or chain.shape != (15, 2) or not np.array_equal(prefix, chain[:5]):
                raise ValueError("lookahead prefix mismatch")
            if sum(g["size"] for g in run["generations"]) != 119:
                raise ValueError("unexpected suffix sample budget")
            if index and (item["name"] != f"response_{index - 1}"
                          or not np.array_equal(prefix, np.asarray(candidates[index-1]["chain"], np.float32))):
                raise ValueError("candidate bank mismatch")
            scores.append(float(run["score"]))
        scores = np.asarray(scores)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite score")
        scores, zero_score = scores[1:], float(scores[0])
        greedy = np.argsort([c["score"] for c in candidates], kind="stable")[:4]
        all_best, kept_best, greedy_best = (float(np.min(scores[i])) for i in
                                             (np.arange(10), retained, greedy))
        output.append({"root": root, "all_best": all_best, "retained_best": kept_best,
                       "greedy_best": greedy_best, "zero_best": zero_score,
                       "retained_gap": kept_best - all_best,
                       "greedy_gap": greedy_best - all_best,
                       "any_success": all_best < .05,
                       "retained_success": kept_best < .05,
                       "greedy_success": greedy_best < .05,
                       "candidate_scores": scores.tolist(), "retained": retained.tolist(),
                       "greedy": greedy.tolist()})
    if hashes != {str(p): digest(p) for p in inputs}:
        raise RuntimeError("source changed during analysis")
    with OUT.open("x") as stream:
        json.dump({"scope": "post-hoc; no new search or rollout", "sha256": hashes,
                   "rows": output}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name in ("retained", "greedy"):
        print(name, "strict_loss", sum(r[f"{name}_gap"] > 1e-12 for r in output),
              "mean_gap", np.mean([r[f"{name}_gap"] for r in output]),
              "success", sum(r[f"{name}_success"] for r in output))
    print("any_success", sum(r["any_success"] for r in output), "output", OUT)


if __name__ == "__main__":
    main()
