"""Fixed-bank state-assisted latent-effect anchoring with separate physical audit."""
import json
import sys

import numpy as np

from experiments import pc_camera_multiview_dev_20261106_codex as prior
from experiments.pc_input_permissions import file_hash

ROOT = prior.ROOT
PREFIX = ROOT / "results/pc_pointmaze_latent_effect_anchor_20261107_codex"
SELECT = PREFIX.with_name(PREFIX.name + "_selection.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_LATENT_EFFECT_ANCHOR_DEV_PROTOCOL_20261107_CODEX.md"


def score():
    import torch
    from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
    from experiments.pc_input_permissions import StateAssistedPointMaze
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
    from experiments.pc_state_track_smoke_20261106_codex import load_model

    if SELECT.exists():
        raise FileExistsError(SELECT)
    previous = json.loads(prior.SELECT.read_text())
    model, dataset, permission = load_model()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_fingerprint"]:
        raise ValueError("model differs from prior")
    wrapper = StateAssistedPointMaze(model, permission)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    print("Compare original and current-anchored latent effect scores on the fixed input bank.", flush=True)
    with np.load(prior.PUBLIC, allow_pickle=False) as data, torch.inference_mode():
        for i in range(8):
            for camera in range(2):
                root = 1884 + i
                old = next(r for r in previous["rows"] if r["root"] == root and r["camera"] == camera)
                shift = goal_shift(data["goal"][i, camera])
                goal_image = translate(data["goal"][i, camera], shift["dx"], shift["dy"])[0]
                image = translate(data["current"][i, camera], shift["dx"], shift["dy"])[0]
                inputs = {"visual": rgb_tensor(image),
                          "proprio": torch.tensor(data["proprio"][i, camera], dtype=torch.float32,
                                                   device="cuda")[None, None]}
                goal = model.encode(rgb_tensor(goal_image))[0]
                current = model.encode(rgb_tensor(image))[0]
                predictions = []
                for x in (0., .1, -.1):
                    raw = torch.tensor([[x, 0.]] * 5, device="cuda")
                    packet = ((raw - mean) / std).reshape(1, 1, 10)
                    predictions.append(wrapper.predict(inputs, packet)["visual"][-1])
                baseline = predictions[0]
                original = [float((prediction - goal).square().mean()) for prediction in predictions]
                anchored = [float((current + prediction - baseline - goal).square().mean())
                            for prediction in predictions]
                np.testing.assert_allclose(original, old["scores"][1], atol=1e-6, rtol=1e-5)
                rows.append({"root": root, "camera": camera, "original": original, "anchored": anchored,
                             "original_choice": int(np.argmin(original)), "anchored_choice": int(np.argmin(anchored)),
                             "zero_prediction_current_mse": float((baseline-current).square().mean())})
    if fingerprint != model_fingerprint(model):
        raise ValueError("model drift")
    result = {"rows": rows, "track": "state_assisted", "prediction_calls": len(wrapper.receipts),
              "additional_current_encodings": len(rows), "model_sha256": fingerprint,
              "input_sha256": file_hash(prior.PUBLIC), "prior_sha256": file_hash(prior.SELECT),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "scope": "exposed fixed bank; no candidate outcomes read during selection"}
    with SELECT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Sealed {len(rows)} decisions and {len(wrapper.receipts)} predictions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    selected = json.loads(SELECT.read_text())
    if (selected["input_sha256"] != file_hash(prior.PUBLIC) or selected["prior_sha256"] != file_hash(prior.SELECT)
            or selected["protocol_sha256"] != file_hash(PROTOCOL) or selected["runner_sha256"] != file_hash(__file__)):
        raise ValueError("sealed dependencies changed")
    rows = []
    with np.load(prior.PRIVATE, allow_pickle=False) as data:
        for row in selected["rows"]:
            i, camera = row["root"] - 1884, row["camera"]
            target = data["traces"][2*i+1, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][2*i, :, camera, -1, :2] - target, axis=-1)
            rows.append({"root": row["root"], "camera": camera,
                         "original_regret": float(distances[row["original_choice"]] - distances.min()),
                         "anchored_regret": float(distances[row["anchored_choice"]] - distances.min())})
    if len(rows) != 16 or {(r["root"], r["camera"]) for r in rows} != {
            (root, camera) for root in range(1884, 1892) for camera in range(2)}:
        raise ValueError("incomplete or duplicate panel")
    roots = [{"root": root,
              "original": float(np.mean([r["original_regret"] for r in rows if r["root"] == root])),
              "anchored": float(np.mean([r["anchored_regret"] for r in rows if r["root"] == root]))}
             for root in range(1884, 1892)]
    means = {name: float(np.mean([r[name] for r in roots])) for name in ("original", "anchored")}
    wins = sum(r["anchored"] < r["original"] for r in roots)
    case = next(r for r in rows if r["root"] == 1887 and r["camera"] == 1)
    disagreements = {name: sum(
        next(r[name + "_choice"] for r in selected["rows"] if r["root"] == root and r["camera"] == 0)
        != next(r[name + "_choice"] for r in selected["rows"] if r["root"] == root and r["camera"] == 1)
        for root in range(1884, 1892)) for name in ("original", "anchored")}
    result = {"rows": rows, "per_root": roots, "mean_regret": means, "improved_roots": wins,
              "view_disagreements": disagreements, "case_1887_camera1": case,
              "development_exit_met": bool(wins >= 5 and means["anchored"] < means["original"]
                                           and case["anchored_regret"] < case["original_regret"]),
              "selection_sha256": file_hash(SELECT), "physical_sha256": file_hash(prior.PRIVATE),
              "scope": "state-assisted exposed fixed-candidate evaluation; not closed-loop"}
    with AUDIT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("mean_regret", "improved_roots", "view_disagreements",
                                              "case_1887_camera1", "development_exit_met")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("score", "audit"):
        raise SystemExit("expected score|audit")
    {"score": score, "audit": audit}[sys.argv[1]]()
