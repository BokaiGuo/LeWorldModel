"""Exploratory full-bank score selection with no forced reconfiguration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_diverse_cuda_replay_20261030_main as source
from . import p1_nontrivial_model_candidates_20261030_main as candidate_audit

NAME = "p1_full_bank_selector_20261030_main"
ROOT = source.ROOT


def choose(prefixes, scores):
    x = np.asarray(prefixes, dtype="<f4")
    y = np.asarray(scores, dtype=np.float64)
    if (x.shape != (20, 5, 2) or y.shape != (20,) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any(np.abs(x) > 1)):
        raise ValueError("invalid complete candidate bank")
    # Zero and incumbent remain eligible. Never force a non-greedy witness.
    return min(range(20), key=lambda i: (y[i], i))


def build(backtrack, report, prepared, prior):
    rows = []
    if len(report["rows"]) != 8 or len(backtrack["rows"]) != 160 or len(prior["rows"]) != 8:
        raise ValueError("fixed bank roster")
    for arm, old in enumerate(prior["rows"]):
        candidates = report["rows"][arm]["result"]["candidates"]
        group = backtrack["rows"][20 * arm:20 * (arm + 1)]
        prefixes = [x["prefix"] for x in candidates]
        scores = [x["best_regularized"] for x in group]
        index = choose(prefixes, scores)
        stage = candidate_audit.analyze(prefixes, [x["stage_Q"] for x in candidates],
                                        report["rows"][arm]["result"]["initial_Q"], scores)[index]
        action = np.concatenate((np.asarray(prefixes[index], dtype="<f4"),
                                 np.asarray(group[index]["final"]["raw"], dtype="<f4").reshape(10, 2)))
        if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("invalid witness")
        rows.append(dict(slot=old["slot"], arm=old["arm"], task_binding=old["task_binding"],
                         selected_index=index, previous_four_indices=old["selected_indices"],
                         previous_selected_index=old["chosen_index"], descriptor=stage,
                         score=scores[index], incumbent_score=scores[0],
                         raw15_float32_hex=action.tobytes().hex(),
                         raw15_sha256=hashlib.sha256(action.tobytes()).hexdigest()))
    return dict(schema=NAME, rows=rows, score="saved best_regularized",
                policy="minimum score over all 20; ties use original index; zero and incumbent allowed",
                P1_admission=False, physical_authorized=False, independent_confirmation=False,
                limitation="Post-outcome known-root development; all 20 optimization costs already incurred.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    if not path.is_absolute() or path.parent != ROOT / "results" or not path.name.startswith(NAME + "_") or path.exists():
        raise ValueError("exclusive output")
    _, backtrack, report, prepared = source.sources()
    prior = source.prior.one.saved.read_verified(source.JOIN, source.JOIN_SHA)
    result = build(backtrack, report, prepared, prior)
    result.update(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_join_sha256=source.JOIN_SHA,
                  source_backtrack_sha256=source.prior.inventory.SHA,
                  source_report_sha256=source.prior.one.source.REPORT_SHA)
    print("Descriptive full-bank selection; no neural inference or physical execution.", flush=True)
    print(json.dumps({"sha256": source.prior.one.saved.write(path, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
