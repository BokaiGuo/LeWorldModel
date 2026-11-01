"""Run only the ten H1 batches not attempted by the sealed partial run."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np

from . import p3_raw_h1_physical_20261031_main as initial

host, ROOT, require = initial.host, initial.ROOT, initial.require
NAME = "p3_raw_h1_physical_continue_20261031_main"
PREVIOUS = ROOT / "results/p3_raw_h1_physical_20261031_main_RUN_01"
PREVIOUS_SHA = "bf8ad50b0054e84772772b9ffffb823b50f0b88b8fcf26e07967285729a17429"
CONTRACT = ROOT.parent / "docs/D2P_P3_RAW_H1_PHYSICAL_CONTINUATION_20261031_MAIN.md"


def run(output):
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_RUN_") and not os.path.lexists(output), "exclusive output")
    require(host.sha(PREVIOUS / "batch_report.json") == PREVIOUS_SHA, "prior partial hash")
    previous = json.loads((PREVIOUS / "batch_report.json").read_bytes())
    require(previous["status"] == "partial_or_failed" and
            [x["batch"] for x in previous["batches"]] == [0, 1] and
            all(host.sha(PREVIOUS / f"batch_{x['batch']:02}" / "report.json") == x["report_sha256"]
                for x in previous["batches"]), "prior attempt roster")
    require(shutil.disk_usage(output.parent).free >= 10 * host.helper.CAP + host.helper.FLOOR, "disk reserve")
    plans, bindings, contract, data = initial.prepare()
    bindings.update({str(Path(__file__).resolve()): host.sha(Path(__file__)),
                     str(CONTRACT): host.sha(CONTRACT),
                     str(PREVIOUS / "batch_report.json"): PREVIOUS_SHA})
    output.mkdir(mode=0o700)
    receipts = []
    for batch in range(2, 12):
        host.NAME = NAME + f"_B{batch:02}"
        p = {"schema": host.NAME + ".prepare", "contract": copy.deepcopy(contract),
             "plans": copy.deepcopy(plans[6 * batch:6 * batch + 6]), "bindings": bindings,
             "physical_execution": False, "scientific_pass": False,
             "source_nonlinear_sha256": initial.NONLINEAR_SHA, "prior_partial_sha256": PREVIOUS_SHA}
        directory = output / f"batch_{batch:02}"
        store = host.helper.Store(directory)
        store.put("prepare.json", p)

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                            with_target=False, render_action=False)

        result = host.execute(p, data, store, factory)
        failures = [r for r in result["rows"] if r["status"] == "failed"]
        permitted = (not result["errors"] and failures and all(
            r["first_error"] and r["first_error"]["message"] ==
            "RGB/canvas/state/pure-render guard; failed image retained"
            and not r["secondary_errors"] and
            np.isfinite(r["last_observation"]["state"]).all() and
            any(x < 0 or x > 512 for x in r["last_observation"]["state"][:4]) and
            r["last_observation"]["state"] == r["last_observation"]["observation_state"]
            for r in failures))
        receipts.append({"batch": batch, "status": result["status"],
                         "guard_partial": bool(permitted),
                         "report_sha256": host.sha(directory / "report.json")})
        print(json.dumps(receipts[-1]), flush=True)
        if result["status"] != "recorded_not_admitted" and not permitted:
            break
    final = {"schema": NAME, "status": "all_attempted_not_admitted" if len(receipts) == 10
             else "partial_or_failed", "batches": receipts, "fixed_total_branches": 72,
             "P1_admission": False, "P3_admission": False, "independent_confirmation": False}
    digest = host.bind.host.write_new(output / "batch_report.json", final)
    print(json.dumps({"status": final["status"], "sha256": digest}), flush=True)
    return final["status"] == "all_attempted_not_admitted"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Run fixed unattempted H1 batches; retain native guard partials.", flush=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
