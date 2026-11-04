"""Continue the risk-selected physical paths to the next real RGB boundary."""
import argparse

import numpy as np

from . import p1_dev305_contact_model_select_20261102 as planner
from . import p1_dev305_contact_second_boundary_20261102 as prior
from . import p1_dev305_contact_second_boundary_physical_20261102 as physical
from . import p1_dev305_second_boundary_risk_saved_20261102 as risk_saved

base, need, run, ROOT = prior.base, prior.need, prior.run, prior.ROOT
NAME = "p1_dev305_contact_third_boundary_20261102"
AUTHORITY = "DEV305_CONTACT_THIRD_BOUNDARY_CUDA"
ORIGINAL_PROTOCOL = planner.protocol


def protocol():
    # The reused frozen-model planner temporarily swaps its protocol function.
    # Restore it while revalidating the archived source chain.
    current = planner.protocol
    planner.protocol = ORIGINAL_PROTOCOL
    try:
        checked = risk_saved.analyze()
        old = base.read(ROOT / "results" / f"{prior.NAME}_PREP_DEV01.json", physical.PREP_SHA)
        physical_spec = base.read(ROOT / "results" / f"{physical.NAME}_PREP_DEV01.json",
                                  risk_saved.physical.CONTRACT_SHA)
        executed = base.read(ROOT / "results" / f"{physical.NAME}_RUN_DEV01/report.json",
                             risk_saved.physical.REPORT_SHA)
    finally:
        planner.protocol = current
    directory = ROOT / "results" / f"{physical.NAME}_RUN_DEV01"
    selected = [row for row in checked["rows"] if row["selected"]]
    need(len(selected) == len(old["roles"]) == 3, "risk-selected roster")
    roles = []
    for previous in old["roles"]:
        choice = next(row for row in selected if row["arm"] == previous["arm"])
        key = choice["arm"] + "__" + choice["candidate"]
        replay = next(row for row in physical_spec["roles"] if row["arm"] == key)
        actual = next(row for row in executed["rows"] if row["arm"] == key)
        need(actual["status"] == "fixed_horizon_completed" and len(actual["observations"]) == 6,
             "completed risk-selected packet")
        actions = np.asarray([step["action"] for step in actual["transitions"]], dtype="<f4")
        need(actions.shape == (5, 2)
             and np.array_equal(actions, np.asarray(replay["tail"], dtype="<f4")), "dispatched actions")
        frame = actual["observations"][-1]
        image = {"path": str((directory / frame["file"]).resolve()),
                 "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}
        pixels = run.coherent.base.frame(image)
        need(prior.previous.source.visibility.inspect(pixels)["eligible"], "current RGB")
        direction = run.coherent.base.first.engine._geometry(pixels)["contact_direction_raw_xy"]
        history = np.concatenate((np.asarray(previous["actual_history"], dtype="<f4")[-5:], actions))
        need(np.array_equal(history[:5], np.asarray(replay["history"], dtype="<f4")[-5:]),
             "previous packet history")
        roles.append({"arm": previous["arm"], "images": [*previous["images"][-2:], image],
                      "actual_history": history.tolist(), "actions": planner.packets(direction).tolist(),
                      "previous_candidate": choice["candidate"]})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "planner_sha256": base.sha(planner.__file__), "risk_report_sha256": risk_saved.REPORT_SHA,
            "physical_report_sha256": risk_saved.physical.REPORT_SHA,
            "policy": old["policy"], "model_hash": old["model_hash"], "goal": old["goal"],
            "action_mean": old["action_mean"], "action_std": old["action_std"],
            "latent_dim": old["latent_dim"], "roles": roles,
            "bank_order": ["toward_02", "away_02", "zero", "toward_05"],
            "scope": "known-root post-outcome development; continuation of sealed risk-selected physical branches",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "plan"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        planner.protocol = protocol
        planner.AUTHORITY = AUTHORITY
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(planner.plan(path, args.contract_sha, ROOT / "results" / f"{NAME}_PLAN_{args.tag}", args.authority))
