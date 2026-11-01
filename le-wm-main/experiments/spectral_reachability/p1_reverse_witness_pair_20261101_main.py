"""Compare the preselected plan and supplied witness at the same fifteen steps."""
import argparse
import json
from pathlib import Path

from . import p1_reverse_control_20261101_main as control
from . import p1_reverse_intervention_rgb_20261101_main as rgb
from . import p1_reverse_prediction_saved_20261101_main as prediction


def analyze():
    root = control.ROOT
    actual_path = root / "results/p1_reverse_intervention_rgb_20261101_main_MAIN01.json"
    witness_path = root / "results/p1_reverse_witness_prediction_20261101_main_MAIN01.json"
    check = prediction.check(witness_path)
    physical = rgb.analyze(rgb.DIRECTORY)
    actual = json.loads(actual_path.read_bytes())
    witness = json.loads(witness_path.read_bytes())
    selected = physical["model_selected"]
    row = next(r for r in physical["rows"] if r["candidate"] == selected and r["suffix"] == "own")
    if row["endpoint15"] is None or row["status"] != "completed15":
        raise ValueError("selected fifteen-step endpoint unavailable")
    frame = next(f for f in actual["frames"] if f["candidate"] == selected
                 and f["suffix"] == "own" and f["step"] == 15)
    supplied = next(r for r in witness["rows"] if r["mode"] == "witness_open" and r["step"] == 15)
    model_margin = physical["predicted_scores"][selected] - supplied["predicted_goal_q"]
    actual_margin = actual["encoded"][frame["pixel_sha256"]]["q_score"] - supplied["observed_goal_q"]
    sources = {**actual["sources"], **witness["sources"], **physical["sources"]}
    for path in (actual_path, witness_path):
        sources[str(path)] = control.target_definition.encoder.base.sha(path)
    if any(control.target_definition.encoder.base.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    return dict(selected_candidate=selected, horizon_steps=15,
                selected_predicted_q=physical["predicted_scores"][selected],
                witness_predicted_q=supplied["predicted_goal_q"],
                selected_actual_q=actual["encoded"][frame["pixel_sha256"]]["q_score"],
                witness_actual_q=supplied["observed_goal_q"],
                model_selected_minus_witness=model_margin, actual_selected_minus_witness=actual_margin,
                ranking_reversal=model_margin * actual_margin < 0,
                witness_saved_check=check, sources=sources, P1_admission=False,
                independent_confirmation=False, privileged_witness_only=True,
                caution="Diagnostic alternative was not an admissible planner candidate; no outcome-selected deployment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "witness_saved_check")}), flush=True)
