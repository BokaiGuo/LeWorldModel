"""Offline fixed-panel RGB rollout preparation. No torch, physics or CUDA."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_causal_panel_bridge_20261030_codex_v1 as bridge

STEM = "p5_rgb_rollout_contract_20261030_codex_v1"
PREVIOUS = bridge.ROOT / "results/p5_rgb_causal_panel_bridge_20261030_codex_v1_MAIN01.json"
PREVIOUS_SHA = "559c7c12667e39a320e27151e08c9022ba719f19444b78b08e8a392aed1ba324"


def check(test, message):
    if not test:
        raise ValueError(message)


def load_sources():
    raw = PREVIOUS.read_bytes()
    check(bridge.digest(raw) == PREVIOUS_SHA, "previous preparation SHA")
    return (bridge.read_json(raw), bridge.read_json(bridge.read_fixed("physical_receipt")),
            bridge.read_json(bridge.read_fixed("panel")))


def validate_input(row, inputs):
    check(set(inputs) == {"previous_image", "image", "actions"}, "input fields")
    for key, record in (("previous_image", "previous_rgb"), ("image", "current_rgb")):
        a = inputs[key]
        check(isinstance(a, np.ndarray) and a.dtype == np.dtype("<f4") and
              a.shape == (1, 3, 64, 64) and np.isfinite(a).all() and
              bridge.digest(a.tobytes(order="C")) == row[record]["model_input_chw_float32_sha256"],
              key + " identity")
    a = inputs["actions"]
    check(isinstance(a, np.ndarray) and a.dtype == np.dtype("<f4") and
          a.shape == (1, 5, 2) and np.isfinite(a).all() and
          bridge.digest(a.tobytes(order="C")) == row["actions_sha256"], "actions identity")


def materialize(row):
    old = dict(row, action_raw_float32=row["actions"][0],
               action_raw_float32_sha256=row["first_action_sha256"])
    inputs = bridge.materialize_inputs(old)
    inputs.pop("action")
    inputs["actions"] = np.asarray(row["actions"], dtype="<f4")[None, ...]
    validate_input(row, inputs)
    return inputs


def prepare(previous, physical, panel):
    check(previous["status"] == "cpu_strict_load_and_input_only_prepared_not_prediction" and
          physical["status"] == "observed_raw164_physical_recorded_not_admitted" and
          panel["status"] == "saved_input_physical_identity_bound_not_scientific_admission",
          "source statuses")
    check(previous["branch_denominator"] == physical["fixed_denominator"] ==
          panel["branch_denominator"] == 164 and previous["pair_denominator"] ==
          panel["pair_denominator"] == 80, "denominator")
    check(len(previous["rows"]) == len(physical["rows"]) == len(panel["rows"]) == 164
          and len(panel["pairs"]) == 80 and previous["pairs"] == panel["pairs"], "row counts")
    rows = []
    for i, (old, p, panel_row) in enumerate(zip(previous["rows"], physical["rows"], panel["rows"], strict=True)):
        keys = ("index", "root_slot", "condition", "task_id", "selection_digest")
        check(all(old[k] == p[k] == panel_row[k] == p["plan"][k] for k in keys) and
              (i, i // 82, ("off", "on")[(i // 41) % 2]) ==
              (p["index"], p["root_slot"], p["condition"]), "branch identity")
        check(p["status"] == "horizon_completed" and p["own_returned"] == 10 and
              p["initial_exact"] is True, "physical completeness")
        actions = np.asarray(p["plan"]["commands"], dtype="<f4")
        check(actions.shape == (10, 2) and np.isfinite(actions).all() and
              bridge.digest(actions.tobytes()) == p["plan"]["raw_float32_sha256"] ==
              panel_row["raw10_sha256"], "raw10 identity")
        check(bridge.digest(actions[5].tobytes()) == old["action_raw_float32_sha256"], "action6")
        for step, key in ((4, "previous_rgb"), (5, "current_rgb")):
            frames = [f for f in p["trajectory"] if (f["phase"], f["step"], f["condition"],
                      f["overlay_index"]) == ("plan", step, p["condition"], int(p["condition"] == "on"))]
            check(len(frames) == 1 and frames[0]["file_sha256"] == old[key]["file_sha256"] and
                  frames[0]["pixel_sha256"] == old[key]["pixel_sha256"] and
                  frames[0]["png"] == Path(old[key]["path"]).name, "history binding")
        suffix = actions[5:10].copy()
        row = {k: old[k] for k in (*keys, "previous_rgb", "current_rgb")}
        row.update(actions=suffix.tolist(), actions_sha256=bridge.digest(suffix.tobytes()),
                   first_action_sha256=old["action_raw_float32_sha256"])
        materialize(row)
        rows.append(row)
    for n, pair in enumerate(panel["pairs"]):
        slot, local = divmod(n, 20)
        coord = local % 10
        minus, plus = 41 * slot + 1 + 2 * local, 41 * slot + 2 + 2 * local
        check((pair["slot"], pair["coord"], pair["h"], pair["minus_index"], pair["plus_index"]) ==
              (slot, coord, (0.01, 0.001)[local // 10], minus, plus), "pair schedule")
        a = np.asarray(rows[minus]["actions"], dtype="<f4").reshape(-1)
        b = np.asarray(rows[plus]["actions"], dtype="<f4").reshape(-1)
        base = np.asarray(rows[41 * slot]["actions"], dtype="<f4").reshape(-1)
        check(all(a[j].tobytes() == b[j].tobytes() == base[j].tobytes()
                  for j in range(10) if j != coord) and a[coord] < b[coord], "paired perturbation")
        for key in ("previous_rgb", "current_rgb"):
            check(rows[minus][key]["model_input_chw_float32_sha256"] ==
                  rows[plus][key]["model_input_chw_float32_sha256"], "paired history")
    return {"schema": STEM, "status": "offline_inputs_prepared_no_execution_no_admission",
            "source_sha256": {"previous": PREVIOUS_SHA, "physical": bridge.FILES["physical_receipt"][1],
                              "panel": bridge.FILES["panel"][1], "checkpoint": bridge.FILES["checkpoint"][1],
                              "model_source": bridge.FILES["model_source"][1]},
            "branch_denominator": 164, "pair_denominator": 80, "steps_per_branch": 5,
            "baseline_indices": [0, 41, 82, 123], "rows": rows, "pairs": panel["pairs"],
            "normalization_from_prior_strict_load": previous["model"]["normalization"],
            "planned_calls": {"encode_state": 164, "predict_next": 820},
            "actual_calls": {"neural": 0, "cuda": 0, "physics": 0, "training": 0},
            "execution_authorized": False, "P5_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check(args.output.name.startswith(STEM + "_"), "exclusive output name")
    bridge.read_fixed("checkpoint")
    bridge.read_fixed("model_source")
    result = prepare(*load_sources())
    raw = bridge.canonical(result)
    with args.output.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"sha256": bridge.digest(raw), "branches": 164, "pairs": 80,
                      "neural_calls": 0, "cuda_calls": 0}))


if __name__ == "__main__":
    main()
