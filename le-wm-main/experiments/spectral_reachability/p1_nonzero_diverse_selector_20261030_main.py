"""Exploratory diverse selector that abstains rather than calls zero reconfiguration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from . import p1_response_diverse_backtrack_join_20261030_main as joined

NAME = "p1_nonzero_diverse_selector_20261030_main"
ROOT = joined.ROOT
JOIN_SHA = "ad5b2df40dc14d73e37c1b1b4913d76c20f224090e705892acd29f7852140d67"


def select(indices, prefixes, scores):
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 or i >= len(prefixes) or i >= len(scores) for i in indices)):
        raise ValueError("candidate roster")
    accepted = []
    for i in indices:
        prefix = np.asarray(prefixes[i], dtype="<f4")
        if prefix.shape != (5, 2) or not np.isfinite(prefix).all() or (np.abs(prefix) > 1).any():
            raise ValueError("candidate prefix")
        if not np.isfinite(scores[i]):
            raise ValueError("candidate score")
        if np.any(prefix != 0):
            accepted.append(i)
    return min(accepted, key=lambda i: (scores[i], i)) if accepted else None


def build(backtrack, report, prepared, panel):
    verified = joined.build(backtrack, report, prepared, panel)
    rows = []
    for arm, old in enumerate(verified["rows"]):
        group = backtrack["rows"][20*arm:20*(arm+1)]
        candidates = report["rows"][arm]["result"]["candidates"]
        chosen = select(old["selected_indices"], [c["prefix"] for c in candidates],
                        [x["best_regularized"] for x in group])
        item = dict(slot=old["slot"], arm=old["arm"], selected_indices=old["selected_indices"],
                    prior_chosen_index=old["chosen_index"], nonzero_chosen_index=chosen,
                    abstained=chosen is None, task_binding=old["task_binding"])
        if chosen is not None:
            prefix = np.asarray(candidates[chosen]["prefix"], dtype="<f4")
            suffix = np.asarray(group[chosen]["final"]["raw"], dtype="<f4").reshape(10, 2)
            witness = np.concatenate((prefix, suffix))
            if not np.isfinite(witness).all() or (np.abs(witness) > 1).any():
                raise ValueError("witness")
            item.update(raw15_f32_hex=witness.tobytes().hex(),
                        raw15_sha256=hashlib.sha256(witness.tobytes()).hexdigest(),
                        score=group[chosen]["best_regularized"], model_stage_uphill=group[chosen]["stage_uphill"])
        rows.append(item)
    return dict(schema=NAME, rows=rows, numerical_zero_rejected=True,
                no_nonzero_candidate="abstain", P1_admission=False, physical_authorized=False,
                independent_confirmation=False,
                limitation="Post-outcome development on known roots; nonzero does not establish mechanism.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    if not path.is_absolute() or path.parent != ROOT / "results" or not path.name.startswith(NAME+"_") or path.exists():
        raise ValueError("exclusive output")
    backtrack = joined.need_read(ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json",
                                 joined.inventory.SHA)
    report = joined.need_read(joined.panel.source.REPORT, joined.panel.source.REPORT_SHA)
    receipt = joined.need_read(joined.panel.source.RECEIPT, joined.panel.source.RECEIPT_SHA)
    if receipt["report_sha256"] != joined.panel.source.REPORT_SHA:
        raise ValueError("source receipt")
    prepared = joined.need_read(report["prepared"], report["prepare_sha256"])
    joined.panel.source.saved.analyze(report, receipt, prepared)
    joined.panel.source.host.verify(report["prepared"], report["prepare_sha256"])
    panel = joined.need_read(ROOT / "results/p1_task_response_saved_panel_20261030_main_MAIN01.json", joined.PANEL_SHA)
    old = joined.need_read(ROOT / "results/p1_response_diverse_backtrack_join_20261030_main_RUN01.json", JOIN_SHA)
    if {k: v for k, v in old.items() if k != "code_sha256"} != joined.build(backtrack, report, prepared, panel):
        raise ValueError("old selection source")
    result = build(backtrack, report, prepared, panel)
    result.update(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_join_sha256=JOIN_SHA)
    print(json.dumps({"sha256": joined.panel.source.saved.write(path, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
