"""Saved-data audit of RGB pixel axes vs PushT native axes and action ABI.

Uses physical state only retrospectively for this audit; never for a planner.
No GPU or new physical execution. Sprite-centroid visibility is not certified.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_p1_two_stage_saved_20261030_v1 as saved
from . import codex_p1_rgb_task_score_20261030_v2 as rgb

ROOT = saved.run.ROOT
NAME = "codex_p1_rgb_action_mapping_audit_20261030_v1"
RUN = ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_run_CODEX01"
PREP = ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_prepare_CODEX01.json"
SAVED = ROOT/"results/codex_p1_two_stage_saved_20261030_v1_CODEX01.json"
PREP_SHA = "edea86663dd55caa9a7c5a6f48862030fb84248164df1691f38a82b2ca9672c5"
RUN_SHA = "2edad31c9315b6b2ab7cb9f8424c7b51a4f91ef283293066ccd784c91d3d2ded"
SAVED_SHA = "e5a1eebfbf12e14c9705b6020dcdb7dba5a6eb0e1c41e2168678f1da56a48c82"
need, sha = saved.need, saved.sha


def audit_pixel_pair(previous, current, pixel_floor=2.0):
    """Compare two observed locations; small pixel changes are not informative."""
    pix = np.asarray(current["pixel_xy"]) - np.asarray(previous["pixel_xy"])
    native = np.asarray(current["posthoc_native_xy"]) - np.asarray(previous["posthoc_native_xy"])
    return [{"pixel_delta": float(pix[k]), "native_delta": float(native[k]),
             "classification": ("below_pixel_floor" if abs(pix[k]) < pixel_floor else
                                ("same_sign" if pix[k]*native[k] > 0 else "opposite_or_zero"))}
            for k in (0, 1)]


def build():
    need(sha(SAVED) == SAVED_SHA, "saved hash")
    receipt = json.loads(SAVED.read_bytes())
    need(saved.consume(RUN, RUN_SHA, PREP, PREP_SHA) == receipt, "saved re-consumption")
    import pymunk.pygame_util as pygame_util
    need(pygame_util.positive_y_is_up is False, "pygame y axis configuration")
    rows, counts = [], {"same_sign": [0, 0], "opposite_or_zero": [0, 0],
                        "below_pixel_floor": [0, 0]}
    for i in range(24):
        report = json.loads((RUN/f"row_{i:02d}/physical/report.json").read_bytes())
        frames = [r for r in report["row"]["trajectory"] if
                  (r["phase"] == "plan" and r["step"] == 5) or r["phase"] == "probe"]
        need(len(frames) == 11 and receipt["rows"][i]["accepted_suffix_steps"] == 10, "full fixed frame roster")
        points, comparisons = [], []
        for rec in frames:
            path = RUN/f"row_{i:02d}/physical"/rec["png"]
            need(sha(path) == rec["file_sha256"], "frame hash")
            image = saved.run.episode.feedback.rgb_pixels(path.read_bytes())
            try:
                _, marker = rgb._mask(image, rgb.BLUE, 8, single=True)
                point = {"phase": rec["phase"], "step": rec["step"],
                         "pixel_xy": marker["centroid_xy"], "posthoc_native_xy": rec["state"][:2],
                         "pixel_sha256": rec["pixel_sha256"]}
            except ValueError as exc:
                point = {"phase": rec["phase"], "step": rec["step"], "rejected": str(exc)}
            points.append(point)
        for a, b in zip(points, points[1:]):
            if "pixel_xy" not in a or "pixel_xy" not in b:
                continue
            axes = audit_pixel_pair(a, b)
            for axis, value in enumerate(axes):
                counts[value["classification"]][axis] += 1
            comparisons.append({"from": [a["phase"], a["step"]], "to": [b["phase"], b["step"]],
                                "axes": axes})
        rows.append({"index": i, "points": points, "adjacent_comparisons": comparisons})
    need(sha(SAVED) == SAVED_SHA, "saved drift")
    return {"schema": NAME, "rows": rows, "axis_counts": counts, "saved_sha256": SAVED_SHA,
            "coordinate_contract": {"render_canvas": [512, 512], "output_rgb": [224, 224],
                                    "resize_native_per_pixel": 512/224,
                                    "pymunk_positive_y_is_up": False,
                                    "relative_action_target_scale_native": 100,
                                    "raw_0_2_target_offset_native": 20,
                                    "actual_displacement_not_equal_to_target_offset": True},
            "interpretation": "retrospective coordinate audit, not physical intervention or planner input",
            "os_isolation_certified": False, "new_physical_calls": 0, "new_model_calls": 0,
            "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    need(path.is_absolute() and path.parent == ROOT/"results" and path.name.startswith(NAME+"_"), "output scope")
    result = build()
    print(json.dumps({"sha256": saved.run.h10._write_new(path, result), "axis_counts": result["axis_counts"]}))


if __name__ == "__main__":
    main()
