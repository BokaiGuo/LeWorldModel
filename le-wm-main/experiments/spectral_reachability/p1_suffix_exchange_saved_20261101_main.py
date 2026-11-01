"""Recompute complete prefix/tail coverage, model-only selection and cost."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import p1_suffix_exchange_q_20261101_main as host
from . import p1_horizon_q_saved_20261101_main as prior


def check(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_bytes())
    records = []
    for row in report["rows"]:
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] == "not_started":
                continue
            h, _ = host.parent.parent.carry.dimensions(node["boundary"])
            trace = Path(node["receipt"]["trace"])
            events = [json.loads(line) for line in trace.read_bytes().splitlines()]
            def one(kind):
                values = [e["payload"] for e in events if e["kind"] == kind]
                prior.saved.need(len(values) == 1, kind + " multiplicity")
                return values[0]
            if h == 1:
                prior.saved.need(not any(e["kind"] == "suffix_exchange" for e in events), "no H1 exchange")
                continue
            exchange = one("suffix_exchange")
            before = np.asarray(one("before_exchange_optimized_candidates")["raw"], np.float32)
            after = np.asarray(one("optimized_candidates")["raw"], np.float32)
            scores, indices = exchange["scores"], host.best_per_prefix(exchange["scores"])
            prior.saved.need(exchange["horizon"] == h and exchange["tail_indices"] == indices
                             and exchange["physical_labels_used"] is False, "exchange rule")
            expected = np.stack([np.concatenate((before[i, :1], before[j, 1:]))
                                 for i, j in enumerate(indices)])
            prior.saved.need(np.array_equal(expected, after), "exact prefix/tail source")
            original = one("before_exchange_final_selection")
            prior.saved.need(np.array_equal(np.asarray(scores).reshape(8, 8).diagonal(),
                                             np.asarray(original["scores"])), "diagonal replay equality")
            final = one("final_selection")
            best = [scores[8*i+j] for i, j in enumerate(indices)]
            winner = min(range(8), key=lambda i: (best[i], i))
            prior.saved.need(final["scores"] == best and final["index"] == winner, "final model selection")
            records.append(dict(arm=row["arm"], boundary=node["boundary"],
                                prefix_index=winner, tail_index=indices[winner],
                                previous_winner=original["index"],
                                previous_score=min(original["scores"]), exchanged_score=best[winner]))
    with patch.object(prior.host, "cost", host.cost):
        result = prior.check(directory)
    result["exchange_decisions"] = records
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(nodes=len(result["nodes"]), output=str(args.output))))
