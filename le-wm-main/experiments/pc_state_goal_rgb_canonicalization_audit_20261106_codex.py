"""Post-hoc image and physical label audit; selection record is read-only."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
from experiments.pc_input_permissions import file_hash
from experiments.pc_state_goal_rgb_canonicalization_dev_20261106_codex import OUT, SOURCE, PROTOCOL, OLD, digest


def main():
    output = OUT.with_name(OUT.stem + "_audit.json")
    if output.exists():
        raise FileExistsError(output)
    record = json.loads(OUT.read_text())
    if (record["protocol_hash"] != file_hash(PROTOCOL) or record["task_manifest_hash"] != file_hash(SOURCE)
            or record["old_result_sha256"] != file_hash(OLD)
            or record["transform_sha256"] != file_hash(Path(__file__).with_name("pc_goal_rgb_canonicalization.py"))
            or record["runner_sha256"] != file_hash(Path(__file__).with_name(
                "pc_state_goal_rgb_canonicalization_dev_20261106_codex.py"))):
        raise ValueError("source drift")
    rows, root_metrics = [], []
    with np.load(SOURCE, allow_pickle=False) as data:
        for i in range(4):
            images, masks, choices = [], [], {}
            for camera in range(2):
                row = next(r for r in record["rows"] if r["root"] == 1880+i and r["camera"] == camera)
                goal = data["images"][2*i+1, 0, camera, 0]
                current = data["images"][2*i, 0, camera, 0]
                shift = goal_shift(goal)
                corrected, valid = translate(current, shift["dx"], shift["dy"])
                corrected_goal, _ = translate(goal, shift["dx"], shift["dy"])
                arm = row["arms"]["goal_rgb_centered"]
                if digest(corrected) != arm["current_image_sha256"] or digest(corrected_goal) != arm["goal_image_sha256"]:
                    raise ValueError("corrected image drift")
                if digest(current) != row["arms"]["raw"]["current_image_sha256"]:
                    raise ValueError("raw image drift")
                target = data["traces"][2*i+1, 0, camera, 0, :2]
                distances = np.linalg.norm(data["traces"][2*i, :, camera, -1, :2] - target, axis=-1)
                decisions = {}
                for name in ("raw", "goal_rgb_centered"):
                    v = row["arms"][name]
                    if v["selected"] != int(np.argmin(v["scores"])):
                        raise ValueError("selection inconsistent")
                    decisions[name] = {"selected": v["selected"],
                                       "physical_regret": float(distances[v["selected"]] - distances.min())}
                    choices.setdefault(name, []).append(v["selected"])
                decisions["zero"] = {"physical_regret": float(distances[0] - distances.min())}
                rows.append({"root": 1880+i, "camera": camera, "decisions": decisions})
                images.append(corrected)
                masks.append(valid)
            overlap = masks[0] & masks[1]
            centered_mae = float(np.abs(images[0].astype(float) - images[1].astype(float))[overlap].mean())
            orig0, orig1 = (data["images"][2*i, 0, camera, 0] for camera in (0, 1))
            raw_mae = float(np.abs(orig0.astype(float) - orig1.astype(float))[overlap].mean())
            root_metrics.append({"root": 1880+i, "raw_mae_on_centered_overlap": raw_mae,
                                 "centered_mae_on_overlap": centered_mae,
                                 "inconsistent": {name: values[0] != values[1]
                                                  for name, values in choices.items()}})
    inconsistency = {name: sum(r["inconsistent"][name] for r in root_metrics)
                     for name in ("raw", "goal_rgb_centered")}
    regret = {name: float(np.mean([r["decisions"][name]["physical_regret"] for r in rows]))
              for name in ("raw", "goal_rgb_centered", "zero")}
    exit_met = inconsistency["goal_rgb_centered"] < inconsistency["raw"] and regret["goal_rgb_centered"] <= regret["raw"]
    output_value = {"source_sha256": file_hash(OUT), "audit_code_sha256": file_hash(__file__),
                    "rows": rows, "root_metrics": root_metrics, "inconsistency_roots": inconsistency,
                    "mean_physical_regret": regret, "development_exit_met": exit_met,
                    "scope": "state-assisted development; post-hoc evaluation, not independent confirmation"}
    with output.open("x") as stream:
        json.dump(output_value, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output_value[k] for k in ("inconsistency_roots", "mean_physical_regret",
                                                   "development_exit_met")}, indent=2))


if __name__ == "__main__":
    main()
