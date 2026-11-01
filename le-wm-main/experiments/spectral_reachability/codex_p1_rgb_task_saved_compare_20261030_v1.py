"""Fixed-panel RGB feature comparison with post-hoc physical metrics.

No model calls, no physical execution, no threshold tuning, no candidate
selection. Image scoring receives only current RGB and task goal RGB. This
process is not OS-isolated from source state, which is used only afterwards.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_rgb_task_score_20261030_v1 as rgb
from . import codex_p1_multistart20_physical_20261030_main as wrapper
from . import codex_p1_h10_paired_table_20261030_main as metrics
from . import p1_backtrack_physical_runner_20261030 as physical

ROOT = physical.ROOT
NAME = "codex_p1_rgb_task_saved_compare_20261030_v1"
DIR = ROOT/"results/codex_p1_multistart20_physical_20261030_main_RUN01"
SAVED = ROOT/"results/codex_p1_multistart20_physical_20261030_main_SAVED_RUN01.json"
SAVED_SHA = "c7cbdf2109d79cf29642659e2f751c8b20a75991cbb30e0427d2c4cdeea25454"
REPORT_SHA = "c2e3a49a61dc1fb6b2a32fe53d6c4977bafb1bd00d967693596e39b3d87ef469"
need = physical.require


def build():
    need(physical.sha(SAVED) == SAVED_SHA, "saved receipt SHA")
    existing = physical.native.read(SAVED, cap=64*1024**2)
    wrapper.install()
    wrapper.saved.DIR, wrapper.saved.REPORT_SHA = DIR, REPORT_SHA
    verified = wrapper.saved.consume()
    verified["schema"] = wrapper.NAME+".consume"
    verified["wrapper_sha256"] = physical.sha(Path(wrapper.__file__).resolve())
    verified["limitations"] += "; wrapper and sources not a complete dependency closure"
    need(verified == existing and len(existing["rows"]) == 24, "strict saved re-consumption")
    source_prepare, source_data = physical.material()
    anchors = wrapper.run.old.h10.native.read(wrapper.run.old.h10.INPUT, cap=64*1024**2)["anchors"]
    need(len(anchors) == 24, "fixed roster")
    images = []
    # Complete the RGB-only feature pass for the fixed panel before reading any
    # physical endpoint state for the post-hoc comparison below.
    for i, (row, anchor) in enumerate(zip(existing["rows"], anchors, strict=True)):
        need(row["index"] == anchor["index"] == i and row["root_slot"] == anchor["root_slot"]
             and row["condition"] == anchor["condition"] and row["role"] == anchor["role"], "row identity")
        count = row["returned_suffix"]
        need(row["status"] in ("probe_completed", "native_stopped") and 1 <= count <= 10
             and (count == 10 or row["status"] == "native_stopped"), "native stop and horizon")
        folder = DIR/f"row_{i:02d}"
        report = physical.native.read(folder/"report.json", cap=64*1024**2)
        frame = [x for x in report["row"]["trajectory"] if x["phase"] == "probe" and x["step"] == count]
        need(len(frame) == 1 and frame[0]["condition"] == row["condition"], "terminal fresh image")
        path = folder/frame[0]["png"]
        need(path.is_file() and not path.is_symlink() and physical.sha(path) == frame[0]["file_sha256"],
             "terminal frame SHA")
        with Image.open(path) as image:
            need(image.mode == "RGB" and image.format == "PNG", "native RGB")
            current = np.asarray(image).copy()
        goal = np.asarray(source_data["roots"][anchor["root_slot"]]["packets"][anchor["condition"]]["goal_rgb"])
        try:
            features = rgb.score(current, goal)
            need(features["current_pixel_sha256"] == frame[0]["pixel_sha256"], "terminal pixels")
            error = None
        except ValueError as exc:
            features, error = None, str(exc)
        images.append({"index": i, "identity": {k: anchor[k] for k in
                       ("root_slot", "condition", "arm", "role", "logical_pair")},
                       "status": row["status"], "returned_suffix": count, "full_horizon": count == 10,
                       "frame_sha256": frame[0]["file_sha256"],
                       "rgb_features": features, "rgb_rejection": error})
    # Retrospective truth is never passed into the RGB-only scorer.
    for item, anchor in zip(images, anchors, strict=True):
        report = physical.native.read(DIR/f"row_{item['index']:02d}/report.json", cap=64*1024**2)
        observations = report["row"]["responses"]
        need(len(observations) == item["returned_suffix"]
             and observations[-1]["step"] == item["returned_suffix"], "real last observation")
        item["posthoc_physical_metrics"] = metrics.physical_metrics(observations[-1]["state"], anchor["goal"])
        item["native_stop_at_horizon"] = item["status"] == "native_stopped" and item["full_horizon"]
        item["early_native_stop"] = item["status"] == "native_stopped" and not item["full_horizon"]
    comparisons = []
    for slot, condition in itertools.product((0, 1), ("off", "on")):
        group = [r for r in images if r["identity"]["root_slot"] == slot and r["identity"]["condition"] == condition]
        need(len(group) == 6, "six fixed rows per goal")
        for left, right in itertools.combinations(group, 2):
            if not left["full_horizon"] or not right["full_horizon"]:
                comparisons.append({"left": left["index"], "right": right["index"], "status": "early_stop"})
                continue
            if left["rgb_features"] is None or right["rgb_features"] is None:
                comparisons.append({"left": left["index"], "right": right["index"], "status": "rgb_rejected"})
                continue
            diff_rgb = (left["rgb_features"]["pusher_centroid_distance_pixels"]
                        - right["rgb_features"]["pusher_centroid_distance_pixels"])
            diff_truth = (left["posthoc_physical_metrics"]["pusher_position_error"]
                          - right["posthoc_physical_metrics"]["pusher_position_error"])
            order = "tie" if diff_rgb == 0 or diff_truth == 0 else ("agree" if diff_rgb*diff_truth > 0 else "disagree")
            comparisons.append({"left": left["index"], "right": right["index"], "status": order,
                                "pusher_rgb_difference_pixels": diff_rgb, "posthoc_pusher_difference": diff_truth})
    need(physical.sha(SAVED) == SAVED_SHA and physical.sha(DIR/"run_report.json") == REPORT_SHA
         and all(physical.sha(p) == h for p, h in source_prepare["bindings"].items()), "source drift")
    counts = {kind: sum(x["status"] == kind for x in comparisons)
              for kind in ("agree", "disagree", "tie", "rgb_rejected", "early_stop")}
    return {"schema": NAME, "rows": images, "fixed_rows": 24, "comparisons": comparisons,
            "comparison_counts": counts, "saved_sha256": SAVED_SHA, "run_sha256": REPORT_SHA,
            "scorer_sha256": physical.sha(Path(rgb.__file__).resolve()),
            "analyzer_sha256": physical.sha(Path(__file__).resolve()),
            "scope": "post-outcome fixed panel; RGB score uses only current/goal pixels; "
                     "truth is retrospective only, process isolation not certified",
            "calibrated_success_classifier": False, "known_roots": True, "independent_confirmation": False,
            "new_model_calls": 0, "new_physical_calls": 0, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT/"results" and out.name.startswith(NAME+"_"), "exclusive output")
    result = build()
    digest = wrapper.run.old.h10._write_new(out, result)
    print(json.dumps({"sha256": digest, "rows": len(result["rows"]),
                      "rgb_rejections": sum(r["rgb_rejection"] is not None for r in result["rows"]),
                      "comparison_counts": result["comparison_counts"]}))


if __name__ == "__main__":
    main()
