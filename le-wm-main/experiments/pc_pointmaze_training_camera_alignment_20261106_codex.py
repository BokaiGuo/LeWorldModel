"""Same-action comparison of legacy and archive-aligned goal-image centering."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.pc_camera_multiview_dev_20261106_codex import PUBLIC, PRIVATE, SELECT, ROOT
from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

NAME = "pc_pointmaze_training_camera_alignment_20261106_codex"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_TRAIN_CAMERA_ALIGNMENT_DEV_PROTOCOL_20261106_CODEX.md"
OUT = ROOT / "results" / (NAME + "_selection.json")
AUDIT = ROOT / "results" / (NAME + "_audit.json")


def training_shift(goal):
    old = goal_shift(goal)
    shift = {"dx": old["dx"], "dy": old["dy"] - 14}
    if max(abs(shift["dx"]), abs(shift["dy"])) > 20:
        raise ValueError("training-reference translation exceeds supported range")
    return shift


def transform(current, goal, name):
    if name == "raw":
        return current, goal
    shift = goal_shift(goal) if name == "old_center" else training_shift(goal)
    current, a = translate(current, shift["dx"], shift["dy"])
    goal, b = translate(goal, shift["dx"], shift["dy"])
    if not np.array_equal(a, b):
        raise ValueError("current/goal transform drift")
    return current, goal


def score():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads(SELECT.read_text())
    model, dataset, permission = load_model()
    before = model_fingerprint(model)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    print("Paired GPU scoring: archive-centered image alignment vs old and raw; no physical outcome inputs.", flush=True)
    with np.load(PUBLIC, allow_pickle=False) as data, torch.inference_mode():
        for i in range(8):
            for camera in range(2):
                current, goal, proprio = (data[key][i, camera].copy() for key in ("current", "goal", "proprio"))
                row = {"root": 1884 + i, "camera": camera, "arms": {}}
                try:
                    images = {name: transform(current, goal, name) for name in ("raw", "old_center", "train_center")}
                except ValueError as exc:
                    row["rejected"] = str(exc)
                    rows.append(row)
                    continue
                for name, (current_arm, goal_arm) in images.items():
                    obs = {"visual": rgb_tensor(current_arm),
                           "proprio": torch.tensor(proprio, dtype=torch.float32, device="cuda")[None, None]}
                    target = model.encode(rgb_tensor(goal_arm))[0]
                    z = model.encode(obs)
                    costs = []
                    for x in (0., .1, -.1):
                        raw = torch.tensor([[x, 0.]] * 5, device="cuda")
                        action = ((raw - mean) / std).reshape(1, 1, 10)
                        predicted = model.unroll(z, action)["visual"][-1]
                        costs.append(float((predicted - target).square().mean()))
                    if not np.isfinite(costs).all():
                        raise ValueError("nonfinite cost")
                    row["arms"][name] = {"scores": costs, "selected": int(np.argmin(costs))}
                old = next(r for r in previous["rows"] if r["root"] == row["root"] and r["camera"] == camera)
                np.testing.assert_allclose(row["arms"]["old_center"]["scores"], old["scores"][1], atol=1e-6, rtol=1e-5)
                rows.append(row)
                print(f"{row['root']}/{camera}: { {k: v['selected'] for k, v in row['arms'].items()} }", flush=True)
    if before != model_fingerprint(model):
        raise ValueError("model changed")
    output = {"rows": rows, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "input_sha256": file_hash(PUBLIC), "previous_selection_sha256": file_hash(SELECT),
              "model_fingerprint": before, "input_audit_hash": permission.fingerprint,
              "model_predictions": sum(3*len(r["arms"]) for r in rows),
              "scope": "state-assisted development; no physical outcomes used for selection"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Sealed results: {OUT}")


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    record = json.loads(OUT.read_text())
    if (record["protocol_sha256"] != file_hash(PROTOCOL) or record["runner_sha256"] != file_hash(__file__)
            or record["input_sha256"] != file_hash(PUBLIC)):
        raise ValueError("experiment drift")
    model, dataset, permission = load_model()
    if model_fingerprint(model) != record["model_fingerprint"] or permission.fingerprint != record["input_audit_hash"]:
        raise ValueError("model input contract drift")
    rows = []
    with np.load(PRIVATE, allow_pickle=False) as data, torch.inference_mode():
        for row in record["rows"]:
            if "rejected" in row:
                rows.append(row)
                continue
            i, camera = row["root"] - 1884, row["camera"]
            current = data["images"][2*i, 0, camera, 0]
            goal = data["images"][2*i+1, 0, camera, 0]
            archive = torch.load(dataset.data_path / f"obses/episode_{row['root']:03d}.pth",
                                 map_location="cpu", weights_only=True).numpy()
            goal_position = data["traces"][2*i+1, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][2*i, :, camera, -1, :2] - goal_position, axis=-1)
            arms = {}
            for name in ("raw", "old_center", "train_center"):
                c, g = transform(current, goal, name)
                goal_z = model.encode(rgb_tensor(g))[0]
                encoded_costs = []
                for action in range(3):
                    image = data["images"][2*i, action, camera, -1]
                    image, _ = transform(image, goal, name)
                    encoded_costs.append(float((model.encode(rgb_tensor(image))[0] - goal_z).square().mean()))
                choice = row["arms"][name]["selected"]
                if choice != int(np.argmin(row["arms"][name]["scores"])):
                    raise ValueError("sealed choice mismatch")
                arms[name] = {"selected": choice, "real_encoding_selected": int(np.argmin(encoded_costs)),
                              "real_encoding_scores": encoded_costs,
                              "physical_regret": float(distances[choice] - distances.min()),
                              "archive_initial_mae": float(np.abs(c.astype(float) - archive[0].astype(float)).mean()),
                              "archive_goal_mae": float(np.abs(g.astype(float) - archive[5].astype(float)).mean())}
            rows.append({"root": row["root"], "camera": camera, "arms": arms, "physical_distances": distances.tolist()})
    valid = [r for r in rows if "arms" in r]
    summary = {}
    for name in ("raw", "old_center", "train_center"):
        summary[name] = {"mean_regret": float(np.mean([r["arms"][name]["physical_regret"] for r in valid])),
                         "prediction_real_encoding_matches": sum(r["arms"][name]["selected"] ==
                                                                 r["arms"][name]["real_encoding_selected"] for r in valid),
                         "mean_initial_archive_mae": float(np.mean([r["arms"][name]["archive_initial_mae"] for r in valid]))}
    case = next((r for r in valid if r["root"] == 1887 and r["camera"] == 1), None)
    exit_met = bool(len(valid) == 16 and case and
                    summary["train_center"]["mean_regret"] < summary["old_center"]["mean_regret"] and
                    summary["train_center"]["prediction_real_encoding_matches"] >
                    summary["old_center"]["prediction_real_encoding_matches"] and
                    case["arms"]["train_center"]["physical_regret"] < case["arms"]["old_center"]["physical_regret"])
    result = {"rows": rows, "summary": summary, "selection_sha256": file_hash(OUT),
              "physical_sha256": file_hash(PRIVATE), "valid": len(valid), "total": 16,
              "development_exit_met": exit_met, "scope": "state-assisted exposed-root post-hoc evaluation"}
    with AUDIT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"summary": summary, "1887_camera1": case, "development_exit_met": exit_met}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("score", "audit"):
        raise SystemExit("usage: python -m experiments.pc_pointmaze_training_camera_alignment_20261106_codex score|audit")
    {"score": score, "audit": audit}[sys.argv[1]]()
