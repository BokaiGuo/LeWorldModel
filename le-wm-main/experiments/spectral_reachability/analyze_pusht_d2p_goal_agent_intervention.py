"""Independent saved-array verification of fixed-action goal intervention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_pusht_d2p_physical_calibration import digest

MODES = ("random8_one_shot", "full192_one_shot", "full192_gn4")
GATES = {"cuda_only", "all20_goal_forwards", "parameters_unchanged", "buffers_unchanged",
         "all_modules_eval", "sources_unchanged", "weights_unchanged", "config_unchanged", "bank_unchanged"}


def analyze(path):
    report = json.loads(path.read_text())
    protocol = json.loads((path.parent/"protocol.json").read_text())
    if (report["schema"] != "pusht_d2p_goal_agent_intervention.v1" or report["status"] != "completed"
            or report["protocol"] != protocol or set(report["gates"]) != GATES
            or any(v is not True for v in report["gates"].values()) or report["P1_admission"] is not False
            or report["goal_encoder_forwards"] != 20 or report["new_predict_samples"] != 0
            or report["new_physical_future_branches"] != 0):
        raise ValueError("complete CUDA goal-only intervention contract required")
    bank_path = Path(protocol["bank_path"])
    if digest(bank_path) != protocol["bank_sha256"]:
        raise ValueError("unchanged action bank required")
    for name, sha in protocol["implementation_sha256"].items():
        if digest(path.parent/"implementation"/(name+".py")) != sha:
            raise ValueError("sealed intervention source changed")
    bank = json.loads(bank_path.read_text())
    roots = sorted(report["rows"], key=lambda r: r["root_id"])
    if [r["root_id"] for r in roots] != list(range(4)):
        raise ValueError("all four unique known roots required")
    changed = {mode: [] for mode in MODES}
    all_rows = []
    offsets = [[0., 0.], [-32., 0.], [32., 0.], [0., -32.], [0., 32.]]
    if protocol["goal_agent_xy_offsets"] != offsets:
        raise ValueError("fixed symmetric intervention required")
    for root in roots:
        index = root["root_id"]
        if (root["root_sha256"] != protocol["root_ids"][index]
                or [v["offset"] for v in root["variants"]] != offsets):
            raise ValueError("complete root/goal intervention join required")
        group = sorted([r for r in bank["rows"] if r["root_id"] == index], key=lambda r: r["proposal_index"])
        if [r["proposal_index"] for r in group] != list(range(16)):
            raise ValueError("all fixed candidate actions required")
        original = np.asarray(root["variants"][0]["goal_vector"], float)
        np.testing.assert_allclose(original, group[0]["goal_vector"], rtol=1e-6, atol=1e-6)
        original_choices = {}
        for variant_index, variant in enumerate(root["variants"]):
            goal = np.asarray(variant["goal_vector"], float)
            if goal.shape != (192,) or not np.isfinite(goal).all():
                raise ValueError("finite full-coordinate goal required")
            choices = {}
            for mode in MODES:
                predictions = np.asarray([r["predicted"]["suffixes"][mode]["predicted_terminal_vector"] for r in group], float)
                values = np.sum(np.square(predictions-goal), axis=1)/192
                stored = variant["model_fixed_actions"][mode]
                np.testing.assert_allclose(values, stored["terminal_costs"], rtol=1e-12, atol=1e-12)
                choice = int(np.argmin(values))
                if choice != stored["model_terminal_cost_argmin"]:
                    raise ValueError("independent fixed-action argmin differs")
                choices[mode] = choice
                if variant_index == 0:
                    original_choices[mode] = choice
                elif choice != original_choices[mode]:
                    changed[mode].append({"root_id": index, "offset": variant["offset"],
                                          "original_argmin": original_choices[mode], "intervened_argmin": choice})
            all_rows.append({"root_id": index, "offset": variant["offset"], "argmins": choices,
                             "goal_embedding_shift_norm": float(np.linalg.norm(goal-original))})
    counts = {mode: len(items) for mode, items in changed.items()}
    if counts != report["argmin_changes_from_original"]:
        raise ValueError("aggregate intervention counts differ")
    return {"schema": "pusht_d2p_goal_agent_intervention_analysis.v1", "report_sha256": digest(path),
            "goal_variants": 20, "independently_recomputed_costs": 960,
            "argmin_changes_from_original": counts, "changes": changed, "rows": all_rows,
            "P1_admission": False, "scope": "Known-bank nuisance sensitivity, not original selector changes or evidence of an improved controller."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable independent analysis required")
    result = analyze(args.report)
    result["analyzer_sha256"] = digest(Path(__file__))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["argmin_changes_from_original"]))


if __name__ == "__main__":
    main()
