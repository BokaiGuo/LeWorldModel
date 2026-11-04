"""Independent CPU audit of image bindings and stored JEPA score arithmetic."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_jepa_native_executed_packet_diag_20261104_codex import ROOT, SOURCE, sha

OUT = ROOT / "results/pc_jepa_native_static_factor_v3_20261104_codex"


def main():
    destination = OUT / "audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    render_path, score_path = OUT / "render.json", OUT / "scores.json"
    render, scores = json.loads(render_path.read_text()), json.loads(score_path.read_text())
    if render["status"] != "render_complete" or scores["status"] != "complete":
        raise ValueError("incomplete source")
    for path, expected in render["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"changed input {path}")
    for path, expected in ((OUT / "images.npz", render["images_sha256"]),
                           (OUT / "features.npz", scores["features_sha256"]),
                           (render_path, scores["render_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"hash mismatch: {path}")
    rows = []
    with np.load(OUT / "images.npz", allow_pickle=False) as images, \
            np.load(OUT / "features.npz", allow_pickle=False) as features:
        if set(images.files) != set(features.files) or len(images.files) != 66:
            raise ValueError("incomplete image/features set")
        for row in scores["rows"]:
            arm, frame, mode = row["arm"], row["frame"], row["mode"]
            prefix = f"{arm}_{frame}"
            source = np.load(SOURCE / f"{arm}_physical/frame_{frame:02d}/rgb.npy", allow_pickle=False)
            if not np.array_equal(images[f"{prefix}_base_native"], source):
                raise ValueError("base image mismatch")
            if not np.array_equal(images[f"{prefix}_both_no_contacts"], images["goal_no_contacts"]):
                raise ValueError("target image mismatch")
            goal = features[f"goal_{mode}"].astype(np.float64)
            calculated = {name: float(np.mean((features[f"{prefix}_{name}_{mode}"].astype(np.float64) - goal) ** 2))
                          for name in ("base", "agent", "object", "both")}
            score_error = max(abs(calculated[n] - row["mse"][n]) for n in calculated)
            effects = {"agent_effect": calculated["agent"] - calculated["base"],
                       "object_effect": calculated["object"] - calculated["base"],
                       "interaction": calculated["both"] - calculated["agent"] - calculated["object"] + calculated["base"]}
            effect_error = max(abs(row[n] - effects[n]) for n in effects)
            if max(score_error, effect_error) != 0:
                raise ValueError("score arithmetic mismatch")
            overlay = {name: int(np.count_nonzero(images[f"{prefix}_{name}_native"] !=
                                                   images[f"{prefix}_{name}_no_contacts"]))
                       for name in ("base", "agent", "object", "both")}
            rows.append({"arm": arm, "frame": frame, "mode": mode, "score_abs_error": score_error,
                         "effect_abs_error": effect_error, "overlay_channels": overlay})
    if len(rows) != 16:
        raise ValueError("missing rows")
    result = {"status": "saved_data_audit_passed", "render_sha256": sha(render_path),
              "scores_sha256": sha(score_path), "audit_script_sha256": sha(Path(__file__)), "rows": rows,
              "scope": "saved arrays and score arithmetic; no independent encoder or physical replay"}
    with destination.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"audited {len(rows)} scored rows, exact arithmetic and base RGB")


if __name__ == "__main__":
    main()
