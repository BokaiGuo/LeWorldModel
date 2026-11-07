"""Paired DINO encoder backend test, with post-seal physical evaluation."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from experiments.pc_camera_multiview_dev_20261106_codex import PUBLIC, PRIVATE, SELECT, ROOT
from experiments.pc_dinov2_adapter_equivalence_dev_20261106_codex import convert
from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

NAME = "pc_pointmaze_dino_backend_dev_20261106_codex"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_DINO_BACKEND_PREDICTION_DEV_PROTOCOL_20261106_CODEX.md"
OUT = ROOT / "results" / (NAME + "_selection.json")
AUDIT = ROOT / "results" / (NAME + "_audit.json")
SOURCE = ROOT / "external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS = ROOT / ".cache/dinov2/dinov2-small/model.safetensors"


def setup():
    model, dataset, permission = load_model()
    shim = model.model.encoder.base_model
    mapped = convert(load_file(str(WEIGHTS)))
    source = torch.hub.load(str(SOURCE), "dinov2_vits14", source="local", pretrained=False)
    source.load_state_dict(mapped, strict=True)
    source = source.cuda().eval().requires_grad_(False)
    if source.interpolate_offset != 0.1:
        raise ValueError("unexpected snapshot interpolation")
    return model, dataset, permission, {"hf_shim": shim, "source_snapshot": source}


def inputs(data, i, camera):
    current, goal, proprio = (data[key][i, camera].copy() for key in ("current", "goal", "proprio"))
    shift = goal_shift(goal)
    current, _ = translate(current, shift["dx"], shift["dy"])
    goal, _ = translate(goal, shift["dx"], shift["dy"])
    return current, goal, proprio, shift


def score():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads(SELECT.read_text())
    model, dataset, permission, backends = setup()
    predictor_before = model_fingerprint(model.model.predictor)
    rows = []
    print("Paired GPU predictions: same observation, action and predictor; varying DINO implementation.", flush=True)
    with np.load(PUBLIC, allow_pickle=False) as data, torch.inference_mode():
        mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
        for i in range(8):
            for camera in range(2):
                current, goal, proprio, _ = inputs(data, i, camera)
                row = {"root": i + 1884, "camera": camera, "arms": {}}
                for name, encoder in backends.items():
                    model.model.encoder.base_model = encoder
                    obs = {"visual": rgb_tensor(current),
                           "proprio": torch.tensor(proprio, dtype=torch.float32, device="cuda")[None, None]}
                    goal_embedding = model.encode(rgb_tensor(goal))[0]
                    z = model.encode(obs)
                    costs = []
                    for action in (0., .1, -.1):
                        raw = torch.tensor([[action, 0.]] * 5, device="cuda")
                        packet = ((raw - mean) / std).reshape(1, 1, 10)
                        pred = model.unroll(z, packet)["visual"][-1]
                        costs.append(float((pred - goal_embedding).square().mean()))
                    row["arms"][name] = {"scores": costs, "selected": int(np.argmin(costs))}
                old = next(x for x in previous["rows"] if x["root"] == row["root"] and x["camera"] == camera)
                np.testing.assert_allclose(row["arms"]["hf_shim"]["scores"], old["scores"][1], atol=1e-6, rtol=1e-5)
                rows.append(row)
                print(f"{row['root']}/{camera}: " + str({k: v["selected"] for k, v in row["arms"].items()}), flush=True)
    if predictor_before != model_fingerprint(model.model.predictor):
        raise ValueError("predictor weights changed")
    output = {"rows": rows, "input_sha256": file_hash(PUBLIC), "prior_selection_sha256": file_hash(SELECT),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "hf_weights_sha256": file_hash(WEIGHTS), "source_hub_sha256": file_hash(SOURCE / "hubconf.py"),
              "predictor_fingerprint": predictor_before, "input_audit_hash": permission.fingerprint,
              "scope": "state-assisted dev adapter comparison; source snapshot not independently certified"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Sealed selection: {OUT}")


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    record = json.loads(OUT.read_text())
    if (record["protocol_sha256"] != file_hash(PROTOCOL) or record["input_sha256"] != file_hash(PUBLIC)
            or record["runner_sha256"] != file_hash(__file__) or record["hf_weights_sha256"] != file_hash(WEIGHTS)):
        raise ValueError("source changed after selection")
    model, _, permission, backends = setup()
    if permission.fingerprint != record["input_audit_hash"] or model_fingerprint(model.model.predictor) != record["predictor_fingerprint"]:
        raise ValueError("model changed after selection")
    rows = []
    print("Post-seal: evaluate real encoded successors and physical regret on GPU.", flush=True)
    with np.load(PRIVATE, allow_pickle=False) as data, torch.inference_mode():
        for row in record["rows"]:
            i, camera = row["root"] - 1884, row["camera"]
            goal = data["images"][2*i+1, 0, camera, 0]
            shift = goal_shift(goal)
            goal, _ = translate(goal, shift["dx"], shift["dy"])
            final = []
            for action in range(3):
                image = data["images"][2*i, action, camera, -1]
                image, _ = translate(image, shift["dx"], shift["dy"])
                final.append(image)
            target_state = data["traces"][2*i+1, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][2*i, :, camera, -1, :2] - target_state, axis=-1)
            evaluated = {}
            for name, encoder in backends.items():
                model.model.encoder.base_model = encoder
                goal_z = model.encode(rgb_tensor(goal))[0]
                real_scores = [float((model.encode(rgb_tensor(image))[0] - goal_z).square().mean()) for image in final]
                pred = row["arms"][name]
                if pred["selected"] != int(np.argmin(pred["scores"])):
                    raise ValueError("invalid sealed selection")
                evaluated[name] = {"selected": pred["selected"], "predicted_scores": pred["scores"],
                                   "real_encoding_scores": real_scores,
                                   "real_encoding_selected": int(np.argmin(real_scores)),
                                   "physical_regret": float(distances[pred["selected"]] - distances.min())}
            rows.append({"root": row["root"], "camera": camera, "physical_distances": distances.tolist(),
                         "arms": evaluated})
    summary = {}
    for name in backends:
        summary[name] = {"mean_regret": float(np.mean([r["arms"][name]["physical_regret"] for r in rows])),
                         "predict_real_encoding_matches": sum(r["arms"][name]["selected"] ==
                                                             r["arms"][name]["real_encoding_selected"] for r in rows),
                         "view_disagreement_roots": sum(
                             next(r for r in rows if r["root"] == root and r["camera"] == 0)["arms"][name]["selected"]
                             != next(r for r in rows if r["root"] == root and r["camera"] == 1)["arms"][name]["selected"]
                             for root in range(1884, 1892))}
    case = next(r for r in rows if r["root"] == 1887 and r["camera"] == 1)
    result = {"rows": rows, "summary": summary, "selection_sha256": file_hash(OUT),
              "physical_sha256": file_hash(PRIVATE),
              "development_exit_met": bool(
                  len(rows) == 16 and summary["source_snapshot"]["mean_regret"] <= summary["hf_shim"]["mean_regret"]
                  and summary["source_snapshot"]["predict_real_encoding_matches"] >
                  summary["hf_shim"]["predict_real_encoding_matches"]
                  and case["arms"]["source_snapshot"]["physical_regret"] < case["arms"]["hf_shim"]["physical_regret"]),
              "scope": "state-assisted post-hoc exposed-root diagnostic"}
    with AUDIT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"summary": summary, "1887_camera1": case, "development_exit_met": result["development_exit_met"]},
                     indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("score", "audit"):
        raise SystemExit("usage: python -m experiments.pc_pointmaze_dino_backend_dev_20261106_codex score|audit")
    {"score": score, "audit": audit}[sys.argv[1]]()
