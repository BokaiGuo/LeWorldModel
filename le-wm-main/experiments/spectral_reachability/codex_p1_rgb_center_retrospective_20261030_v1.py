"""Fixed-roster posthoc pixel error audit, never a planning input.

The saved state is used only after the image-only estimator has returned.
This is not an independent confirmation, deployment certificate or new run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from . import codex_p1_rgb_feedback_saved_20261030_v1 as saved
from . import codex_p1_rgb_visual_diagnostic_20261030_v1 as visual

ROOT = saved.run.ROOT
RUN = ROOT/"results/codex_p1_rgb_feedback_launcher_20261030_v1_run_CODEX01"
PREP = ROOT/"results/codex_p1_rgb_feedback_launcher_20261030_v1_prepare_CODEX01.json"
SAVED = ROOT/"results/codex_p1_rgb_feedback_saved_20261030_v1_CODEX01.json"
RUN_SHA = "a1c55ba949850607f979452dc36852b96cdf293cd9b994c6c2f92f24c981e536"
PREP_SHA = "c9d8f1dd4a27b51b68409fe6cd2e5bcfc5b923a5412cff0fbfe32f9675870316"
SAVED_SHA = "a5c96489369da4b12ce85fc4acbaee36b032161de718b6bfb8d66ec880cb8113"
NAME = "codex_p1_rgb_center_retrospective_20261030_v1"


def image_only_estimate(image):
    """Return image-only conservative result and untrusted diagnostic centroid."""
    decision = visual.assess(image)
    a = image.astype(np.int16)
    mask = (a[..., 2] >= 150) & (a[..., 2]-a[..., 0] >= 80) & (a[..., 2]-a[..., 1] >= 60)
    _, n = label(mask, structure=np.ones((3, 3)))
    if n != 1:
        tentative = None
    else:
        y, x = np.nonzero(mask)
        tentative = [float(x.mean()), float(y.mean())]
    return decision, tentative


def posthoc_error(estimate_xy, native_state):
    if estimate_xy is None:
        return None
    state = np.asarray(native_state, dtype=np.float64)
    xy = np.asarray(estimate_xy, dtype=np.float64)
    if state.shape != (7,) or xy.shape != (2,) or not np.isfinite(state).all() or not np.isfinite(xy).all():
        raise ValueError("audit coordinate ABI")
    # Ideal output-to-canvas scale; interpolation and rasterization add bias.
    return float(np.linalg.norm(xy - state[:2]*(224.0/512.0)))


def build():
    need = saved.need
    need(saved.sha(SAVED) == SAVED_SHA, "saved hash")
    checked = saved.consume(RUN, RUN_SHA, PREP, PREP_SHA)
    need(checked == json.loads(SAVED.read_bytes()), "saved re-consumption")
    rows = []
    for item in checked["rows"]:
        folder = RUN/f"row_{item['index']:02d}"/"physical"
        report = json.loads((folder/"report.json").read_bytes())
        recs = [r for r in report["row"]["trajectory"] if r["phase"] == "probe" and r["step"] == 5]
        need(len(recs) == 1, "fixed cp10")
        rec = recs[0]
        path = folder/rec["png"]
        need(path.is_file() and not path.is_symlink() and saved.sha(path) == rec["file_sha256"], "image")
        image = saved.run.episode.feedback.rgb_pixels(path.read_bytes())
        need(hashlib.sha256(image.tobytes()).hexdigest() == rec["pixel_sha256"], "pixels")
        # Deliberately finish image-only inference before touching saved state.
        visual_result, tentative = image_only_estimate(image)
        error = posthoc_error(tentative, rec["state"])
        rows.append({"index": item["index"], "old_decision": item["rgb_decision"],
                     "visual_decision": visual_result["decision"], "reason": visual_result["reason"],
                     "rgb_pixel_sha256": rec["pixel_sha256"],
                     "tentative_center_error_pixels": error,
                     "center_is_authorized_for_planning": False})
    need(len(rows) == 24 and saved.sha(SAVED) == SAVED_SHA, "roster or source drift")
    return {"schema": NAME, "rows": rows, "fixed_rows": 24,
            "run_sha256": RUN_SHA, "prepare_sha256": PREP_SHA, "saved_sha256": SAVED_SHA,
            "image_estimator_inputs": ["current_rgb"],
            "state_use": "posthoc_error_only_not_planning", "coordinate_scale": "224/512 ideal",
            "occlusion_free_certified": False, "new_model_calls": 0, "new_physical_calls": 0,
            "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    saved.need(path.is_absolute() and path.parent == ROOT/"results" and path.name.startswith(NAME+"_"),
               "exclusive output scope")
    result = build()
    print(json.dumps({"sha256": saved.run.h10._write_new(path, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
