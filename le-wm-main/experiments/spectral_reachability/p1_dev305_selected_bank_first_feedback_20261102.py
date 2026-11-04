"""Replan after the first packet of a previously selected fixed-bank action."""
import argparse
import hashlib

import numpy as np

from . import p1_dev305_triprefix_midfeedback_20261102 as core
from . import p1_dev305_compass_midrgb_model_20261102 as mid
from . import p1_dev305_two_packet_model_bank_saved_20261102 as bank
from . import p1_rgb_visibility_20261101_main as visibility

base, need, run, ROOT = core.base, core.need, core.run, core.ROOT
NAME = "p1_dev305_selected_bank_first_feedback_20261102"
AUTHORITY = "DEV305_SELECTED_BANK_FIRST_FEEDBACK_CUDA"
MID_PREP_SHA = "cc8fa593bdf176cda58edcca63861d9ff7c223b2809960c45a8fe04e053bcb91"


def protocol():
    midpoint = base.read(ROOT / "results" / f"{mid.NAME}_PREP_DEV01.json", MID_PREP_SHA)
    bank_report = base.read(ROOT / "results" / f"{bank.model.NAME}_RUN_DEV01/report.json", bank.REPORT_SHA)
    bank_spec = base.read(ROOT / "results" / f"{bank.model.NAME}_PREP_DEV01.json", bank.CONTRACT_SHA)
    need(midpoint == mid.protocol() and bank_spec == bank.model.protocol()
         and bank_report["contract_sha256"] == bank.CONTRACT_SHA, "sources")
    roles = []
    for role in midpoint["roles"]:
        selected = next(x for x in bank_report["rows"] if x["arm"] == role["arm"])["selected_pair"]
        need(selected[0] == 12, "first packet no longer matches fixed midpoint")
        candidate = next(x for x in role["rows"] if x["name"] == "m10_d4")
        need(np.array_equal(np.asarray(candidate["history"][-5:], dtype="<f4"),
                            np.repeat(np.asarray(bank_spec["bank"][selected[0]]["action"], dtype="<f4")[None], 5, axis=0)),
             "actual first packet")
        frame = run.coherent.base.frame(candidate["images"][-1])
        screen = visibility.inspect(frame)
        need(screen["eligible"], "current RGB fails unchanged visibility guard")
        geometry = run.coherent.base.first.engine._geometry(frame)
        history = np.asarray(candidate["history"], dtype="<f4")
        roles.append({"arm": role["arm"], "images": candidate["images"], "actual_history": history.tolist(),
                      "prefix_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
                      "contact_direction_raw_xy": geometry["contact_direction_raw_xy"], "visibility": screen,
                      "old_selected_pair": selected})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "optimizer_code_sha256": base.sha(core.__file__), "source_mid_contract_sha256": MID_PREP_SHA,
            "source_bank_report_sha256": bank.REPORT_SHA, "policy": midpoint["policy"],
            "model_hash": midpoint["model_hash"], "action_mean": midpoint["action_mean"],
            "action_std": midpoint["action_std"], "latent_dim": midpoint["latent_dim"], "goal": midpoint["goal"],
            "roles": roles, "optimizer": "same CEM seed, two-packet terminal score; execute first packet",
            "per_arm": {"predict_calls": 10, "predict_samples": 3200},
            "selection_inputs": "RGB and actual action history only",
            "classification": "known_root_post_outcome_development",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


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
        # Reuse the existing identical 3-arm CEM implementation without mutating its file.
        core.protocol = protocol
        core.AUTHORITY = AUTHORITY
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(core.plan(path, args.contract_sha, ROOT / "results" / f"{NAME}_PLAN_{args.tag}", args.authority))
