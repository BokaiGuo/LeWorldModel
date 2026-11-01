"""Compare one-packet and two-packet replanning after verified identical midpoint."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_same_first5_feedback_compare_20261030_main as pair
from . import codex_p1_horizon1_physical_20261030_main as h1

NAME = "codex_p1_horizon1_compare_20261030_main"
DIR = h1.ROOT / "results/codex_p1_horizon1_physical_20261030_main_RUN01"
REPORT = "f8add9a6fc7eafc7af23553dd958dabcdb3957bd96e5941c74a04b081f6c0b11"
SAVED = h1.ROOT / "results/codex_p1_horizon1_physical_20261030_main_saved_RUN01.json"
SAVED_SHA = "ed4c53bb27d3d556ad4d9e6c8fd87ca510829b28efe2f23a399181d09b7abbff"


def midpoint(folder, report_sha, condition):
    report = pair.read(folder / "report.json", report_sha)
    frames = [f for f in report["row"]["trajectory"] if f["phase"] == "probe"
              and f["step"] == 5 and f["condition"] == condition
              and f["overlay_index"] == (1 if condition == "on" else 0)]
    pair.need(len(frames) == 1 and Path(frames[0]["png"]).name == frames[0]["png"], "midpoint frame")
    f = frames[0]
    path = folder / f["png"]
    pair.need(h1.physical.old.sha(path) == f["file_sha256"], "midpoint file hash")
    with Image.open(path) as image:
        pixels = np.asarray(image).copy()
    pair.need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
              and hashlib.sha256(pixels.tobytes()).hexdigest() == f["pixel_sha256"], "midpoint pixels")
    return f["pixel_sha256"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    pair.need(out.is_absolute() and out.parent == h1.ROOT / "results"
              and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(),
              "exclusive output")
    one, two = pair.read(SAVED, SAVED_SHA), pair.read(pair.FB_SAVED, pair.FB_SHA)
    one_report = pair.read(DIR / "run_report.json", REPORT)
    two_report = pair.read(pair.replay.FB_DIR / "run_report.json", pair.replay.FB_REPORT_SHA)
    plan = pair.read(h1.PLAN, h1.PLAN_SHA)

    def load(i, a, b):
        folder1 = DIR / f"row_{i:02d}"
        folder2 = pair.replay.FB_DIR / f"row_{i:02d}" / "physical"
        condition = one["rows"][i]["condition"]
        pair.need(midpoint(folder1, a["report_sha256"], condition)
                  == midpoint(folder2, b["report_sha256"], condition)
                  == plan["rows"][i]["midpoint_pixel_sha256"], "actual midpoint mismatch")
        return (pair.transitions(folder1, a["report_sha256"]),
                pair.transitions(folder2, b["report_sha256"]))

    result = pair.compare(one, two, one_report, two_report, load)
    result["schema"] = NAME
    result["summary"]["actual_midpoint_pixels_verified"] = 24
    result["interpretation"] = "feedback_minus_open means two-packet minus one-packet endpoint error"
    result["limitations"] = ["known roots and repeated trajectories; development only",
                             "same first five actions and midpoint RGB; different optimization horizons",
                             "same ten gradient updates, different forward costs; not equal total budget",
                             "one-packet counts are explicit code-path counts, not independent hook metering",
                             "neither result establishes P1 mechanism or independent performance"]
    result["sources"] = {str(SAVED): SAVED_SHA, str(pair.FB_SAVED): pair.FB_SHA,
                         str(DIR / "run_report.json"): REPORT,
                         str(pair.replay.FB_DIR / "run_report.json"): pair.replay.FB_REPORT_SHA,
                         str(h1.PLAN): h1.PLAN_SHA}
    result["code_sha256"] = h1.physical.old.sha(Path(__file__).resolve())
    print(json.dumps({"sha256": h1.physical.old.h10._write_new(out, result),
                      "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
