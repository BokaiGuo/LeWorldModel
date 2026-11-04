"""Compare terminal visibility projections with every sealed physical RGB frame.

This is a post-outcome diagnostic, not an online safety policy or a new rollout.
"""
import argparse
from pathlib import Path

from . import p1_dev305_contact_risk_projection_saved_20261102 as saved

base, need, ROOT = saved.base, saved.need, saved.ROOT
NAME = "p1_dev305_contact_risk_trajectory_20261102"


def analyze():
    # This recomputes the prior selector and checks the contract, model, action
    # binding, and terminal image hashes before using its output.
    terminal = saved.analyze()
    physical = base.read(
        ROOT / "results" / f"{saved.real.source.NAME}_RUN_DEV01/report.json",
        saved.real.REPORT_SHA,
    )
    directory = ROOT / "results" / f"{saved.real.source.NAME}_RUN_DEV01"
    by_key = {item["arm"]: item for item in physical["rows"]}
    need(len(by_key) == 12, "unique physical arms")
    rows = []
    for item in terminal["rows"]:
        key = item["arm"] + "__" + item["candidate"]
        episode = by_key[key]
        observations = episode["observations"]
        need(len(observations) == 6 and len(episode["transitions"]) == 5,
             "complete five-step trajectory")
        timeline = []
        for step, frame in enumerate(observations):
            need(frame["step"] == step, "observation step binding")
            ref = {"path": str((directory / frame["file"]).resolve()),
                   "file_sha256": frame["file_sha256"],
                   "pixel_sha256": frame["pixel_sha256"]}
            screen = saved.source.visibility.inspect(saved.source.run.coherent.base.frame(ref))
            timeline.append({"step": step, "visible": screen["eligible"],
                             "object_box": screen["object"]["bbox"],
                             "pusher_box": screen["pusher"]["bbox"],
                             "reward": None if step == 0 else episode["transitions"][step - 1]["reward"]})
            if step:
                need(episode["transitions"][step - 1]["step"] == step, "transition step binding")
        need(timeline[-1]["visible"] == item["actual_visible"], "terminal binding")
        first_invalid = next((x["step"] for x in timeline if not x["visible"]), None)
        rows.append({"arm": item["arm"], "candidate": item["candidate"],
                     "selected": item["selected"],
                     "predicted_terminal_visible": item["predicted_visible"],
                     "actual_terminal_visible": item["actual_visible"],
                     "first_invalid_step": first_invalid,
                     "all_frames_visible": first_invalid is None,
                     "timeline": timeline})
    return {"schema": NAME + ".v1", "rows": rows,
            "scope": "post-outcome known-root replay; 8-pixel RGB mask guard is not a safety certificate",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
