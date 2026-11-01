"""Execute all fixed H1 raw-chart comparisons from authenticated physical roots."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from . import codex_p3_online_budget_physical_host_20261030_v1 as host
from . import p3_raw_chart_nonlinear_20261030_main as source

ROOT = host.ROOT
NAME = "p3_raw_h1_physical_20261031_main"
NONLINEAR = ROOT / "results/p3_raw_chart_nonlinear_20261030_main_RUN01.json"
NONLINEAR_SHA = "44cc243acc1c260ff84b9263227cad14508d8704cd5292244b894eba9b0d7839"
PROTOCOL = ROOT.parent / "docs/D2P_P3_RAW_H1_PHYSICAL_PROTOCOL_20261031_MAIN.md"
require = host.require


def build_plans(roster, result, physical):
    require(result["schema"] == source.NAME and result["status"] == "complete_cuda_model_only"
            and len(roster) == len(result["rows"]) == 72, "model source roster")
    references = {p["root_slot"]: p for p in physical["plans"]}
    plans = []
    for (index, slot, task, horizon, row), saved in zip(roster, result["rows"], strict=True):
        require((saved["decision_index"], saved["slot"], saved["root"], saved["policy"],
                 saved["horizon"], saved["candidate"])
                == (index, slot["slot"], slot["root"], slot["policy"], horizon, row["index"]),
                "model row identity")
        root = slot["root"]
        require(task["binding"]["root_sha256"] == references[root]["root_sha256"]
                and task["binding"]["task_id"] == references[root]["task_id"], "physical root identity")
        old = np.asarray(row["suffix_actions"], dtype="<f4")
        require(np.array_equal(old, np.asarray(saved["old_action"], dtype="<f4")), "old action binding")
        if horizon != 1:
            continue
        prefix = np.asarray(row["actions"], dtype="<f4")
        require(prefix.shape == old.shape == (5, 2), "H1 packet shape")
        for variant in ("old", "new"):
            tail = np.asarray(saved[variant + "_action"], dtype="<f4")
            commands = np.concatenate((prefix, tail), axis=0)
            require(tail.shape == (5, 2) and np.isfinite(commands).all()
                    and (np.abs(commands) <= 1).all(), "action bounds")
            plan = {
                "root_slot": root, "root_sha256": references[root]["root_sha256"],
                "task_id": references[root]["task_id"], "arm": f"d{index}_c{row['index']}_{variant}",
                "decision_index": index, "policy": slot["policy"], "candidate": row["index"],
                "variant": variant, "commands": commands.astype(float).tolist(),
                "source_candidate_sha256": host.kernel.source.digest(commands.astype(float).tolist()),
                "raw_float32_sha256": hashlib.sha256(commands.tobytes()).hexdigest(),
            }
            plan["selection_digest"] = host.kernel.source.digest(plan)
            plans.append(plan)
    require(len(plans) == 72, "complete 4x9x2 H1 roster")
    return plans


def prepare():
    require(host.sha(NONLINEAR) == NONLINEAR_SHA, "model result hash")
    result = json.loads(NONLINEAR.read_bytes())
    roster = source.load_roster()
    physical, data = host.inputs()
    plans = build_plans(roster, result, physical)
    bindings = dict(physical["bindings"])
    for path in (NONLINEAR, source.JOURNAL, Path(source.__file__), Path(__file__), PROTOCOL):
        bindings[str(path.resolve())] = host.sha(path)
    require(bindings[str(NONLINEAR)] == NONLINEAR_SHA and
            all(host.sha(p) == h for p, h in bindings.items()), "source drift")
    return plans, bindings, physical["contract"], data


def run(output):
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_RUN_") and not os.path.lexists(output),
            "exclusive output")
    require(shutil.disk_usage(output.parent).free >= 12 * host.helper.CAP + host.helper.FLOOR,
            "all-batch disk reservation")
    plans, bindings, contract, data = prepare()
    output.mkdir(mode=0o700)
    receipts = []
    for batch in range(12):
        host.NAME = NAME + f"_B{batch:02}"
        p = {"schema": host.NAME + ".prepare", "contract": copy.deepcopy(contract),
             "plans": copy.deepcopy(plans[6 * batch:6 * batch + 6]), "bindings": bindings,
             "physical_execution": False, "scientific_pass": False,
             "source_nonlinear_sha256": NONLINEAR_SHA}
        directory = output / f"batch_{batch:02}"
        store = host.helper.Store(directory)
        store.put("prepare.json", p)

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                            with_target=False, render_action=False)

        result = host.execute(p, data, store, factory)
        receipts.append({"batch": batch, "status": result["status"],
                         "report_sha256": host.sha(directory / "report.json")})
        print(json.dumps(receipts[-1]), flush=True)
        if result["status"] != "recorded_not_admitted":
            break
    final = {"schema": NAME, "status": "recorded_not_admitted" if len(receipts) == 12
             and all(x["status"] == "recorded_not_admitted" for x in receipts) else "partial_or_failed",
             "batches": receipts, "fixed_branches": 72, "P1_admission": False,
             "P3_admission": False, "independent_confirmation": False, "equal_budget_certified": False}
    digest = host.bind.host.write_new(output / "batch_report.json", final)
    print(json.dumps({"status": final["status"], "sha256": digest}), flush=True)
    return final["status"] == "recorded_not_admitted"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replay every fixed H1 prefix with both old and raw-chart suffixes.", flush=True)
    raise SystemExit(0 if run(args.output) else 1)
