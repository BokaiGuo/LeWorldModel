"""Bind response-diverse prefixes to the already optimized fixed suffix panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_backtrack_action_inventory_20261030 as inventory
from . import p1_task_response_saved_panel_20261030_main as panel

NAME = "p1_response_diverse_backtrack_join_20261030_main"
PANEL_SHA = "940c854036dc660cc3c0217f1486abf401c0148dd8fa3078ec90c301df16cc88"
ROOT = panel.source.ROOT
need_read = panel.source.saved.read_verified


def build(backtrack, report, prepared, selected):
    inventory.verifier.verify(backtrack, report, prepared)
    expected = panel.analyze(report, prepared)
    if expected != selected or len(selected["arms"]) != 8:
        raise ValueError("diversity panel identity")
    rows = []
    for arm_index, choice in enumerate(selected["arms"]):
        group = backtrack["rows"][20 * arm_index:20 * (arm_index + 1)]
        src = report["rows"][arm_index]
        indices = choice["selected_indices"]
        if len(group) != 20 or len(indices) != 4 or len(set(indices)) != 4 or \
                any(type(i) is not int or not 0 <= i < 20 for i in indices) or \
                (choice["slot"], choice["arm"]) != (src["slot"], src["arm"]):
            raise ValueError("arm or candidate identity")
        winner = min(indices, key=lambda i: (group[i]["best_regularized"], i))
        global_winner = min(range(20), key=lambda i: (group[i]["best_regularized"], i))
        candidate = src["result"]["candidates"][winner]
        item = group[winner]
        prefix = np.asarray(candidate["prefix"], dtype="<f4")
        suffix = np.asarray(item["final"]["raw"], dtype="<f4").reshape(10, 2)
        raw = np.concatenate((prefix, suffix))
        if prefix.shape != (5, 2) or not np.isfinite(raw).all() or not (np.abs(raw) <= 1).all():
            raise ValueError("invalid witness")
        rows.append({"slot": choice["slot"], "arm": choice["arm"], "selected_indices": indices,
                     "chosen_index": winner, "global_index": global_winner,
                     "candidate_0_score": group[0]["best_regularized"],
                     "chosen_score": item["best_regularized"],
                     "global_score": group[global_winner]["best_regularized"],
                     "model_stage_uphill": item["stage_uphill"],
                     "raw15_f32_hex": raw.tobytes().hex(),
                     "raw15_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                     "task_binding": prepared["requests"][arm_index // 2]["task"]["binding"]})
    return {"schema": NAME, "status": "descriptive_saved_join", "rows": rows,
            "source_backtrack_sha256": inventory.SHA, "source_response_panel_sha256": PANEL_SHA,
            "P1_admission": False, "physical_authorized": False, "independent_confirmation": False,
            "limitations": ["post-outcome known roots and repeated arms",
                            "all 20 candidates were optimized, not a four-candidate cost measurement",
                            "no new physical execution or model calls",
                            "inherits source and runtime closure limitations"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != ROOT / "results" or not out.name.startswith(NAME + "_"):
        raise ValueError("exclusive output scope")
    backtrack = need_read(ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", inventory.SHA)
    report = need_read(panel.source.REPORT, panel.source.REPORT_SHA)
    receipt = need_read(panel.source.RECEIPT, panel.source.RECEIPT_SHA)
    if receipt["report_sha256"] != panel.source.REPORT_SHA:
        raise ValueError("receipt")
    prepared = need_read(report["prepared"], report["prepare_sha256"])
    panel.source.saved.analyze(report, receipt, prepared)
    panel.source.host.verify(report["prepared"], report["prepare_sha256"])
    selected = need_read(ROOT / "results/p1_task_response_saved_panel_20261030_main_MAIN01.json", PANEL_SHA)
    result = build(backtrack, report, prepared, selected)
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest = panel.source.saved.write(out, result)
    print(json.dumps({"sha256": digest, "rows": len(result["rows"]),
                      "new_indices": sum(x["chosen_index"] != x["global_index"] for x in result["rows"])}))


if __name__ == "__main__":
    main()
