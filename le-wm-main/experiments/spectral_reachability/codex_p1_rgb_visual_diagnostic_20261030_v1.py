"""Image-only blue-marker diagnostics; not an action selector or truth source.

Thresholds are exploratory, fixed-renderer assumptions and do not establish
occlusion-free perception. Ambiguous contact, clipped or multiple markers are
rejected; the output does not authorize physical execution.
"""
import hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt, label


def assess(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise ValueError("native RGB224 required")
    a = image.astype(np.int16)
    exact = np.all(image == (65, 105, 225), axis=-1)
    blue = (a[..., 2] >= 150) & (a[..., 2]-a[..., 0] >= 80) & (a[..., 2]-a[..., 1] >= 60)
    # The narrow gray range also detects dark object edges, not just the fill.
    gray = ((a[..., 0] >= 65) & (a[..., 0] <= 190)
            & (a[..., 1] >= 65) & (a[..., 1] <= 200)
            & (a[..., 2] >= 65) & (a[..., 2] <= 210)
            & (np.abs(a[..., 0]-a[..., 1]) <= 35)
            & (np.abs(a[..., 1]-a[..., 2]) <= 35))
    def components(mask):
        labels, n = label(mask, structure=np.ones((3, 3)))
        result = []
        for k in range(1, n+1):
            y, x = np.nonzero(labels == k)
            result.append({"count": int(len(x)), "bounds_xyxy": [int(x.min()), int(y.min()),
                           int(x.max()), int(y.max())]})
        return sorted(result, key=lambda c: -c["count"])
    exact_parts, blue_parts = components(exact), components(blue)
    result = {"pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "exact_components": exact_parts, "blue_components": blue_parts,
              "decision": "rejected", "reason": None, "centroid_xy": None,
              "gray_distance_pixels": None, "occlusion_free_certified": False}
    if len(blue_parts) != 1:
        result["reason"] = "blue marker absent or multiple components"
        return result
    box = blue_parts[0]["bounds_xyxy"]
    w, h = box[2]-box[0]+1, box[3]-box[1]+1
    if box[0] == 0 or box[1] == 0 or box[2] == 223 or box[3] == 223:
        result["reason"] = "marker clipped"
        return result
    fill = blue_parts[0]["count"] / (w*h)
    if not (8 <= w <= 20 and 8 <= h <= 20 and 0.65 <= w/h <= 1.55
            and blue_parts[0]["count"] >= 30 and 0.55 <= fill <= 0.90):
        result["reason"] = "marker shape ambiguous"
        return result
    distance = float(distance_transform_edt(~gray)[blue].min()) if gray.any() else None
    result["gray_distance_pixels"] = distance
    if distance is None or distance <= 2:
        result["reason"] = "object proximity or unavailable context"
        return result
    y, x = np.nonzero(blue)
    result["centroid_xy"] = [float(x.mean()), float(y.mean())]
    result["decision"] = "tentative_visual_marker"
    return result


def diagnostic():
    """Reconsume the previous fixed run, then inspect cp10/goal RGB only."""
    import io
    import json
    from PIL import Image
    from . import codex_p1_rgb_feedback_saved_20261030_v1 as saved
    run = saved.run
    root = run.ROOT
    directory = root/"results/codex_p1_rgb_feedback_launcher_20261030_v1_run_CODEX01"
    prepared = root/"results/codex_p1_rgb_feedback_launcher_20261030_v1_prepare_CODEX01.json"
    ps = "c9d8f1dd4a27b51b68409fe6cd2e5bcfc5b923a5412cff0fbfe32f9675870316"
    rs = "a1c55ba949850607f979452dc36852b96cdf293cd9b994c6c2f92f24c981e536"
    consumed = saved.consume(directory, rs, prepared, ps)
    anchors = run.h10.native.read(run.h10.INPUT, cap=64*1024**2)["anchors"]
    source_report = run.h10.native.read(run.h10.old.prep.DIRECTORY/"report.json", cap=64*1024**2)
    _, task = run.h10.runner.physical.verify(source_report["execution_binding"]["prepared"],
                                              run.h10.old.prep.PREP_SHA)
    rows = []
    for item, anchor in zip(consumed["rows"], anchors, strict=True):
        folder = directory/f"row_{item['index']:02d}"/"physical"
        report = json.loads((folder/"report.json").read_bytes())
        events = saved.journal_reader.events(folder/"journal.jsonl", report["journal_chain_tail"])
        decisions = [e for e in events if e["kind"] == "rgb_decision"]
        if not decisions:
            rows.append({"index": item["index"], "status": "no_cp10_decision"})
            continue
        receipt = decisions[0]["receipt"]
        selected = [e for e in events if e["kind"] == "fresh_observation"
                    and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5
                    and e["receipt"]["condition"] == anchor["condition"]
                    and e["receipt"]["overlay_index"] == (1 if anchor["condition"] == "on" else 0)]
        if len(selected) != 1:
            raise ValueError("cp10 image identity")
        image_path = folder/selected[0]["receipt"]["png"]
        png = image_path.read_bytes()
        if hashlib.sha256(png).hexdigest() != receipt["cp10_png_sha256"]:
            raise ValueError("cp10 image hash")
        current = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        goal = task["roots"][anchor["root_slot"]]["packets"][anchor["condition"]]["goal_rgb"]
        if hashlib.sha256(goal.tobytes()).hexdigest() != receipt["goal_pixel_sha256"]:
            raise ValueError("fixed goal hash")
        rows.append({"index": item["index"], "old_decision": receipt["decision"],
                     "current": assess(current), "goal": assess(goal)})
    return {"schema": "codex_p1_rgb_visual_diagnostic_20261030_v1", "rows": rows,
            "source_run_sha256": rs, "source_prepare_sha256": ps,
            "diagnostic_only": True, "physical_execution_authorized": False,
            "occlusion_free_certified": False, "P1_admission": False}


def main():
    import argparse
    import json
    from pathlib import Path
    from . import codex_p1_rgb_feedback_saved_20261030_v1 as saved
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    if not (path.is_absolute() and path.parent == saved.run.ROOT/"results"
            and path.name.startswith("codex_p1_rgb_visual_diagnostic_20261030_v1_")):
        raise ValueError("exclusive output scope")
    result = diagnostic()
    digest = saved.run.h10._write_new(path, result)
    print(json.dumps({"sha256": digest, "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
