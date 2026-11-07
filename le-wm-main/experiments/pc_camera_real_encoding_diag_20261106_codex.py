"""Post-seal ground-truth image encoding; not a planning input."""
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from experiments.pc_camera_multiview_dev_20261106_codex import ROOT, PRIVATE, SELECT, choose
from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

OUT = ROOT / "results/pc_camera_real_encoding_diag_20261106_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_CAMERA_REAL_ENCODING_DIAG_PROTOCOL_20261106_CODEX.md"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sealed = json.loads(SELECT.read_text())
    print("Post-hoc true-image encoding vs sealed predictions; no new action selection.", flush=True)
    model, _, permission = load_model()
    fingerprint = model_fingerprint(model)
    rows = []
    with np.load(PRIVATE, allow_pickle=False) as data, torch.inference_mode():
        for row in sealed["rows"]:
            if "rejected" in row:
                rows.append({"root": row["root"], "camera": row["camera"], "rejected": row["rejected"]})
                continue
            i, camera = row["root"] - 1884, row["camera"]
            goal = data["images"][2*i+1, 0, camera, 0]
            shift = goal_shift(goal)
            target, _ = translate(goal, shift["dx"], shift["dy"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            goal_embedding = model.encode(rgb_tensor(target))[0]
            costs = []
            for action in range(3):
                image = data["images"][2*i, action, camera, -1]
                transformed, _ = translate(image, shift["dx"], shift["dy"])
                z = model.encode(rgb_tensor(transformed))[0]
                costs.append(float((z - goal_embedding).square().mean()))
            torch.cuda.synchronize()
            target_state = data["traces"][2*i+1, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][2*i, :, camera, -1, :2] - target_state, axis=-1)
            predicted = row["scores"][1]
            if choose(row["scores"][1:])["centered"] != int(np.argmin(predicted)):
                raise ValueError("sealed prediction mismatch")
            rows.append({"root": row["root"], "camera": camera, "predicted_cost": predicted,
                         "real_encoding_cost": costs, "physical_distance": distances.tolist(),
                         "predicted_selected": int(np.argmin(predicted)),
                         "real_encoding_selected": int(np.argmin(costs)),
                         "physical_selected": int(np.argmin(distances)),
                         "encoding_seconds": time.perf_counter() - started})
    if fingerprint != model_fingerprint(model):
        raise ValueError("weights changed")
    if len(rows) != 16 or any("rejected" in r for r in rows):
        raise ValueError("incomplete diagnostic; do not compute aggregate")
    summary = {"prediction_vs_encoding": sum(r["predicted_selected"] == r["real_encoding_selected"] for r in rows),
               "encoding_vs_physical": sum(r["real_encoding_selected"] == r["physical_selected"] for r in rows),
               "prediction_vs_physical": sum(r["predicted_selected"] == r["physical_selected"] for r in rows),
               "denominator": len(rows)}
    output = {"selection_sha256": file_hash(SELECT), "physical_sha256": file_hash(PRIVATE),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "checkpoint_audit": asdict(permission), "model_fingerprint": fingerprint, "summary": summary, "rows": rows,
              "scope": "post-hoc real encoding diagnostic, state-assisted exposed tasks"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    case = next(r for r in rows if r["root"] == 1887 and r["camera"] == 1)
    print(json.dumps({"summary": summary, "1887_camera1": case}, indent=2))


if __name__ == "__main__":
    main()
