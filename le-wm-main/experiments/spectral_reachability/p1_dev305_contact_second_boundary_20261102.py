"""Compile a new fixed contact bank from real RGB after a full five-step packet."""
import argparse

import numpy as np

from . import p1_dev305_contact_model_select_20261102 as planner
from . import p1_dev305_contact_risk_projection_saved_20261102 as previous

base, need, run, ROOT = planner.base, planner.need, planner.run, planner.ROOT
NAME = "p1_dev305_contact_second_boundary_20261102"
AUTHORITY = "DEV305_CONTACT_SECOND_BOUNDARY_CUDA"
ORIGINAL_PROTOCOL = planner.protocol


def protocol():
    # The reused planner calls its module-level protocol; restore the original
    # while validating the old sealed source, then return to the new contract.
    current = planner.protocol
    planner.protocol = ORIGINAL_PROTOCOL
    try:
        checked = previous.analyze()
    finally:
        planner.protocol = current
    old = base.read(ROOT / "results" / f"{planner.NAME}_PREP_DEV01.json", previous.source.PREP_SHA)
    physical_spec = base.read(ROOT / "results" / f"{previous.real.source.NAME}_PREP_DEV01.json",
                              previous.real.PREP_SHA)
    physical = base.read(ROOT / "results" / f"{previous.real.source.NAME}_RUN_DEV01/report.json",
                         previous.real.REPORT_SHA)
    directory = ROOT / "results" / f"{previous.real.source.NAME}_RUN_DEV01"
    selected = [row for row in checked["rows"] if row["selected"]]
    need(len(selected) == len(old["roles"]) == 3, "selected roles")
    roles = []
    for prior in old["roles"]:
        choice = next(row for row in selected if row["arm"] == prior["arm"])
        key = choice["arm"] + "__" + choice["candidate"]
        source = next(row for row in physical_spec["roles"] if row["arm"] == key)
        actual = next(row for row in physical["rows"] if row["arm"] == key)
        need(actual["status"] == "fixed_horizon_completed" and len(actual["observations"]) == 6,
             "finished packet")
        actions = np.asarray([step["action"] for step in actual["transitions"]], dtype="<f4")
        need(actions.shape == (5, 2)
             and np.array_equal(actions, np.asarray(source["tail"], dtype="<f4")), "dispatched packet")
        frame = actual["observations"][-1]
        image = {"path": str((directory / frame["file"]).resolve()),
                 "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}
        pixels = run.coherent.base.frame(image)
        need(planner.original.visibility.inspect(pixels)["eligible"], "current RGB guard")
        direction = run.coherent.base.first.engine._geometry(pixels)["contact_direction_raw_xy"]
        history = np.concatenate((np.asarray(prior["actual_history"], dtype="<f4")[-5:], actions))
        need(np.array_equal(history[:5], np.asarray(source["history"], dtype="<f4")[-5:]),
             "previous packet history binding")
        roles.append({"arm": prior["arm"], "images": [*prior["images"][-2:], image],
                      "actual_history": history.tolist(), "actions": planner.packets(direction).tolist(),
                      "previous_candidate": choice["candidate"]})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "planner_sha256": base.sha(planner.__file__), "previous_risk_sha256": previous.REPORT_SHA,
            "physical_sha256": previous.real.REPORT_SHA, "policy": old["policy"],
            "model_hash": old["model_hash"], "goal": old["goal"], "action_mean": old["action_mean"],
            "action_std": old["action_std"], "latent_dim": old["latent_dim"], "roles": roles,
            "bank_order": ["toward_02", "away_02", "zero", "toward_05"],
            "scope": "known-root post-outcome development; real RGB and actions at five-step boundary; goal guard not certified",
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
