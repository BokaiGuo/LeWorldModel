"""Branch statistics on sealed suffix-search trajectories, no new planning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import pc_v13_fixedbank_suffix_expression_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SUFFIX_EXPRESSION_SAVED_BRANCH_DIAG_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_suffix_expression_saved_branch_diag_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result_path, audit_path = source.OUT / "result.json", source.OUT / "audit.json"
    result, audit = json.loads(result_path.read_text()), json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["result_sha256"] != seal.digest(result_path):
        raise ValueError("invalid source audit")
    deps = seal.capture([Path(__file__), PROTOCOL, result_path, audit_path, source.OUT / "manifest.json",
                         Path(source.prior.previous.independent.__code__.co_filename),
                         *(source.OUT / r["trace"] for r in result["rows"])])
    if len(result["rows"]) != 4:
        raise ValueError("missing roots")
    rows, total = [], 0
    for root, row in enumerate(result["rows"]):
        if row["root"] != root or row["status"] != "completed":
            raise ValueError("bad root")
        if seal.digest(source.OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        seen = set()
        with np.load(source.OUT / row["trace"], allow_pickle=False) as data:
            for candidate in row["candidates"]:
                ib = candidate["initial_batch"]
                if ib in seen:
                    raise ValueError("repeated initial batch")
                seen.add(ib)
                for run in candidate["runs"]:
                    counts = {"samples": 0, "blocked": 0, "ever_right": 0, "success": 0}
                    best_cross = None
                    for i in range(run["first_batch"], run["last_batch"]):
                        if i in seen:
                            raise ValueError("duplicate batch")
                        seen.add(i)
                        actions, stored = data[f"actions_{i}"], data[f"scores_{i}"]
                        if not np.all(actions[:, :5] == np.asarray(candidate["prefix"], np.float32)):
                            raise ValueError("wrong prefix")
                        path, blocked = source.prior.previous.independent(row["task"]["start"], actions)
                        scores = np.linalg.norm(path[:, -1, :2] - row["task"]["goal"], axis=1)
                        np.testing.assert_allclose(scores, stored, atol=1e-12, rtol=0)
                        crossed = np.any(path[:, 1:, 0] > 0, axis=1)
                        counts["samples"] += len(actions)
                        counts["blocked"] += int(np.any(blocked, axis=1).sum())
                        counts["ever_right"] += int(crossed.sum())
                        counts["success"] += int((scores < .05).sum())
                        if np.any(crossed):
                            value = float(scores[crossed].min())
                            best_cross = value if best_cross is None else min(best_cross, value)
                    if counts["samples"] != 480:
                        raise ValueError("wrong proposal count")
                    total += counts["samples"]
                    rows.append({"root": root, "candidate": candidate["index"], "knots": run["knots"],
                                 "restart": run["restart"], **counts, "best_cross": best_cross})
            if seen != set(range(row["generation_batches"], row["batches"])):
                raise ValueError("suffix batch coverage incomplete")
    seal.verify(deps)
    summary = {}
    for knots in (4, 10):
        group = [x for x in rows if x["knots"] == knots]
        crosses = [x["best_cross"] for x in group if x["best_cross"] is not None]
        summary[str(knots)] = {key: sum(x[key] for x in group)
                               for key in ("samples", "blocked", "ever_right", "success")}
        summary[str(knots)].update({"candidate_root_pairs_with_crossing":
            len({(x["root"], x["candidate"]) for x in group if x["ever_right"]}),
            "roots_with_crossing": len({x["root"] for x in group if x["ever_right"]}),
            "best_cross": min(crosses) if crosses else None})
    source.prior.writer(OUT, {"dependencies": deps, "rows": rows, "summary": summary,
                              "scope": "posthoc on saved samples; no new search"})
    print(json.dumps({"samples": total, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
