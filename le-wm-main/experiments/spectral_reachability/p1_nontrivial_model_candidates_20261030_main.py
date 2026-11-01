"""Describe all saved candidate prefixes versus zero and the incumbent."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from . import p1_diverse_cuda_replay_20261030_main as source

NAME = "p1_nontrivial_model_candidates_20261030_main"
ROOT = source.ROOT


def analyze(prefixes, stage, initial, scores):
    if len(prefixes) != 20 or len(stage) != 20 or len(scores) != 20:
        raise ValueError("20 candidate denominator")
    prefixes = np.asarray(prefixes, dtype="<f4")
    stage = np.asarray(stage, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if (prefixes.shape != (20, 5, 2) or not np.isfinite(prefixes).all()
            or (np.abs(prefixes) > 1).any() or not np.isfinite(stage).all()
            or not np.isfinite(scores).all() or not np.isfinite(initial)):
        raise ValueError("candidate input")
    greedy = prefixes[0]
    result = []
    for i in range(20):
        action = prefixes[i]
        result.append(dict(index=i, zero=bool(np.all(action == 0)),
                           same_as_incumbent=bool(np.array_equal(action, greedy)),
                           model_non_improving_stage=bool(stage[i] >= initial),
                           stage_delta=float(stage[i]-initial),
                           score_delta_vs_incumbent=float(scores[i]-scores[0]),
                           score_beats_incumbent=bool(scores[i] < scores[0])))
    return result


def build(backtrack, report, prepared):
    arms = []
    for i in range(8):
        src = report["rows"][i]
        candidates = src["result"]["candidates"]
        group = backtrack["rows"][20*i:20*(i+1)]
        if len(group) != 20 or len(candidates) != 20:
            raise ValueError("candidate roster")
        rows = analyze([x["prefix"] for x in candidates], [x["stage_Q"] for x in candidates],
                       src["result"]["initial_Q"], [x["best_regularized"] for x in group])
        arms.append(dict(slot=src["slot"], arm=src["arm"],
                         task_binding=prepared["requests"][i//2]["task"]["binding"],
                         candidates=rows))
    return dict(schema=NAME, arms=arms, P1_admission=False, independent_confirmation=False,
                interpretation="Saved model scores only; no physical gate, no threshold or selector change.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    if not path.is_absolute() or path.parent != ROOT / "results" or not path.name.startswith(NAME+"_") or path.exists():
        raise ValueError("exclusive output")
    _, backtrack, report, prepared = source.sources()
    value = build(backtrack, report, prepared)
    value.update(source_backtrack_sha256=source.prior.inventory.SHA,
                 source_report_sha256=source.prior.one.source.REPORT_SHA,
                 code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps({"sha256": source.prior.one.saved.write(path, value), "arms": len(value["arms"])}))


if __name__ == "__main__":
    main()
