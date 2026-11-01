"""CPU-only 164-branch input census and bounded-resource preparation; no launch."""
import argparse
import datetime
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_launch_preflight_gate_20261030_codex_v1 as gate

host, wiring, abi = gate.host, gate.wiring, gate.abi
NAME = "p5_rgb_production_readiness_20261030_codex_v1"
GATE_SHA = "f23ce5fae14fc0c7d0e379feafb80a574054396e138982015600b665011d4ab5"


def census(panel):
    abi.check(panel["branch_denominator"] == 164 and panel["pair_denominator"] == 80 and
              len(panel["rows"]) == 164 and len(panel["pairs"]) == 80, "fixed denominator")
    rows = panel["rows"]
    conditions = {x: {"branches": 0, "actions": 0, "outside_unit_box": 0} for x in ("off", "on")}
    for i, row in enumerate(rows):
        condition = ("off", "on")[(i // 41) % 2]
        abi.check(row["index"] == i and row["root_slot"] == i // 82 and
                  row["condition"] == condition, "branch order")
        actions = np.asarray(row["actions"], dtype="<f4")
        abi.check(actions.shape == (5, 2) and np.isfinite(actions).all() and
                  abi.bridge.digest(actions.tobytes()) == row["actions_sha256"], "action binding")
        group = conditions[condition]
        group["branches"] += 1
        group["actions"] += len(actions)
        group["outside_unit_box"] += int((abs(actions) > 1).sum())
    for index, pair in enumerate(panel["pairs"]):
        slot, local = divmod(index, 20)
        coord = local % 10
        minus, plus = slot * 41 + 1 + 2 * local, slot * 41 + 2 + 2 * local
        abi.check((pair["slot"], pair["coord"], pair["h"], pair["minus_index"], pair["plus_index"]) ==
                  (slot, coord, (0.01, 0.001)[local // 10], minus, plus), "pair schedule")
        a, b = (np.asarray(rows[i]["actions"], dtype="<f4").reshape(-1) for i in (minus, plus))
        baseline = np.asarray(rows[slot * 41]["actions"], dtype="<f4").reshape(-1)
        abi.check(a[coord] < b[coord] and
                  all(a[j].tobytes() == b[j].tobytes() == baseline[j].tobytes()
                      for j in range(10) if j != coord), "pair coordinates")
        for key in ("previous_rgb", "current_rgb"):
            abi.check(rows[minus][key]["model_input_chw_float32_sha256"] ==
                      rows[plus][key]["model_input_chw_float32_sha256"], "pair history")
    differences = {key: sum(
        rows[base + offset][key]["model_input_chw_float32_sha256"] !=
        rows[base + 41 + offset][key]["model_input_chw_float32_sha256"]
        for base in (0, 82) for offset in range(41)) for key in ("previous_rgb", "current_rgb")}
    return {"branch_denominator": 164, "pair_denominator": 80, "conditions": conditions,
            "off_on_input_hash_differences": differences}


def prepare(disk_floor_mib):
    abi.check(abi.bridge.digest(Path(gate.__file__).read_bytes()) == GATE_SHA, "launcher gate SHA")
    host_report = host.inspect(disk_floor_mib)
    abi.check(host_report["source_check"]["ok"] is True, "sealed inputs/sources unavailable")
    panel = wiring.runner.load_prepared()
    baseline = abi.bridge.read_json(wiring.acc.read_new_file(wiring.BASE_PATH, wiring.BASE_SHA))
    abi.check(baseline["parameter_count"] == 93842 and baseline["state_dict_entries"] == 26,
              "checkpoint baseline")
    return {"schema": NAME, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "offline_preparation_not_execution_authorization",
            "source_sha256": {"gate": GATE_SHA, "host_preflight": gate.HOST_SHA,
                              "wiring": host.WIRING_SHA, "baseline": wiring.BASE_SHA,
                              "panel": wiring.runner.PREP_SHA,
                              "checkpoint": baseline["checkpoint_sha256"]},
            "input_census": census(panel), "host_preflight": host_report,
            "planned_operation_caps": {"encode_state": 164, "predict_next": 820,
                                       "maximum_neural_interface_calls": 984, "branches": 164,
                                       "frames_per_branch": 2, "actions_per_branch": 5},
            "logical_bytes": {"model_parameters_float32": 93842 * 4,
                              "images_per_branch_float32": 2 * 3 * 64 * 64 * 4,
                              "actions_per_branch_float32": 5 * 2 * 4,
                              "six_outputs_per_branch_float32": 6 * 6 * 4,
                              "all_transition_outputs_float32": 164 * 5 * 6 * 4},
            "validation_caps_not_writer_quotas": {"journal_max_bytes": 16 * 1024 * 1024,
                                                  "each_snapshot_max_bytes": 1024 * 1024},
            "unknown_not_bounded": ["CUDA runtime and allocator memory", "GPU free memory and contention",
                                    "neural compute and model wall time", "CPU RSS and page cache",
                                    "stdout/stderr and filesystem journal growth", "GPU execution health"],
            "note": "logical tensor bytes and consumer size limits are not device-memory or disk upper bounds",
            "P5_admission": False, "gpu_calls": 0, "neural_forward_calls": 0, "physics_calls": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-disk-free-mib", type=int, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    name = args.output_name
    abi.check(Path(name).name == name and name.startswith(NAME + "_") and name.endswith(".json"), "output name")
    report = prepare(args.min_disk_free_mib)
    print(wiring.gate.write_new(abi.bridge.ROOT / "results" / name, report))


if __name__ == "__main__":
    main()
