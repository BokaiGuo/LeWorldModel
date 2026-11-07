"""Audit historical timed-prefix candidate identity and shortened old-tail equivalence."""
import json

import numpy as np

from experiments import pc_tworoom_timed_fallback_clock_audit_20261107_codex as prior
from experiments.pc_input_permissions import file_hash
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored

OUTPUT = prior.ROOT / "results/pc_tworoom_timed_all_horizons_clock_audit_20261107_codex.json"


def audit(record):
    prior.audit(record)
    rows = []
    for root in record["rows"]:
        for arm, outcome in root["arms"].items():
            legacy_tail = fixed_tail = None
            for d in outcome["decisions"]:
                remaining = d["remaining_steps"]
                prefix = d["prefix_steps"]
                count = 13 if remaining == 5 else 3
                build_prefix = 5 if remaining == 5 else prefix
                old = prior.timed.build(d["candidate_seed"], legacy_tail, build_prefix, count=count)
                fixed = prior.timed.build(d["candidate_seed"], fixed_tail, build_prefix, count=count)
                if remaining == 5 and legacy_tail is None:
                    old[0][0] = 0
                    fixed[0][0] = 0
                # Every candidate evaluated by the planner has the same scored prefix.
                lengths = (remaining, remaining - build_prefix, remaining - build_prefix)
                for before, after, length in zip(old, fixed, lengths, strict=True):
                    np.testing.assert_array_equal(before[:, :length], after[:, :length])
                parents, probes, tails = old
                parent_cost = np.asarray(d["parent_cost"])
                if remaining == 5:
                    best = ScoredCandidate("fallback", int(np.argmin(parent_cost)), float(parent_cost.min()))
                    chosen = parents[best.index]
                else:
                    retained = d["retained"]
                    bank = np.concatenate((np.broadcast_to(parents[:, None, :prefix],
                                                             (3, 2, prefix, 2)),
                                           np.broadcast_to(probes[None], (3, *probes.shape))), axis=2)
                    tail_chains = np.concatenate((np.broadcast_to(parents[retained, None, :prefix],
                                                                    (2, 2, prefix, 2)),
                                                  np.broadcast_to(tails[None], (2, *tails.shape))), axis=2)
                    tail_chains = tail_chains.reshape(4, 25, 2)
                    candidates = [ScoredCandidate("parent", i, float(v)) for i, v in enumerate(parent_cost)]
                    candidates += [ScoredCandidate("probe", i, float(v))
                                   for i, v in enumerate(np.asarray(d["probe_cost"]).ravel())]
                    candidates += [ScoredCandidate("tail", i, float(v))
                                   for i, v in enumerate(d["tail_cost"])]
                    best = select_scored(candidates)
                    chosen = {"parent": parents, "probe": bank.reshape(6, 25, 2),
                              "tail": tail_chains}[best.source][best.index]
                if (d["choice"] != {"source": best.source, "index": best.index, "cost": best.cost}
                        or not np.array_equal(chosen, np.asarray(d["actions"], dtype=np.float32))):
                    raise ValueError("selected candidate does not match logged scored candidates")
                dispatched = len(d["dispatched"])
                legacy_tail = chosen[dispatched:].copy()
                fixed_tail = chosen[:remaining][dispatched:].copy()
                rows.append({"root": root["root"], "arm": arm, "remaining": remaining,
                             "returned": len(chosen), "dispatched": dispatched, "choice": best.source})
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record = json.loads(prior.SOURCE.read_text())
    rows = audit(record)
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(prior.SOURCE), "runner_sha256": file_hash(__file__),
                   "decisions": len(rows), "rows": rows,
                   "scope": "deterministic candidate and stored-score audit; no model rerun"},
                  stream, indent=2, allow_nan=False)
    print(f"Audited {len(rows)} decisions; candidate arrays and scored choices agree after trimming old tails.")


if __name__ == "__main__":
    main()
