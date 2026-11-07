"""Prior-observation residual correction; separate preparation, scoring, audit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model

NAME = "pc_pointmaze_causal_residual_dev_20261106_codex"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CAUSAL_RESIDUAL_DEV_PROTOCOL_20261106_CODEX.md"
REPLAY = ROOT / "results/pc_pointmaze_packet_clock_replay_20261106_codex.json"
BACKEND = ROOT / "experiments/pc_real_camera_effect_dev_20261106_codex.py"
PRIVATE = ROOT / "results" / (NAME + "_physical.npz")
PUBLIC = ROOT / "results" / (NAME + "_inputs.npz")
FEATURES = ROOT / "results" / (NAME + "_features.npz")
SELECTION = ROOT / "results" / (NAME + "_selection.json")
AUDIT = ROOT / "results" / (NAME + "_audit.json")


def correct(candidate, prior_prediction, current_encoding):
    if candidate.shape != prior_prediction.shape or candidate.shape != current_encoding.shape:
        raise ValueError("latent shapes do not agree")
    if not all(torch.isfinite(x).all().item() for x in (candidate, prior_prediction, current_encoding)):
        raise ValueError("nonfinite latent")
    return candidate + current_encoding - prior_prediction


def prepare():
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    for path in (PUBLIC, PRIVATE):
        if path.exists():
            raise FileExistsError(path)
    replay = json.loads(REPLAY.read_text())
    if (not replay["strict_replay_atol_1e_4"] or
            replay["source_states_sha256"] != file_hash(DATA / "states.pth") or
            replay["source_actions_sha256"] != file_hash(DATA / "actions.pth")):
        raise ValueError("prior action-state replay not bound")
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True)
    actions = torch.load(DATA / "actions.pth", map_location="cpu", weights_only=True)
    lengths = torch.load(DATA / "seq_lengths.pth", map_location="cpu", weights_only=True)
    if any(int(lengths[root]) < 11 for root in range(1884, 1892)):
        raise ValueError("episode too short")
    previous_actions = actions[1884:1892, :5].float().numpy()
    if not np.isfinite(previous_actions).all() or np.abs(previous_actions).max() > 1:
        raise ValueError("historical actions not valid")
    state_batch = np.stack([states[root, frame].numpy() for root in range(1884, 1892)
                            for frame in (0, 5, 10)])
    with tempfile.TemporaryDirectory(prefix="pc_causal_residual_") as tmp:
        path = Path(tmp)
        request, xml = path / "request.npz", path / "scene.xml"
        np.savez_compressed(request, states=state_batch)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        command = ("import sys; from experiments import pc_real_camera_effect_dev_20261106_codex as b; "
                   "b.CAMERAS=((3.,3.),(3.5,3.)); b.backend(*sys.argv[1:])")
        subprocess.run([str(LEGACY / "bin/python"), "-c", command, str(request), str(xml), str(PRIVATE)],
                       env=env, cwd=ROOT, check=True, timeout=240)
    with np.load(PRIVATE, allow_pickle=False) as data:
        if data["traces"].shape != (24, 3, 2, 6, 4):
            raise ValueError("unexpected trajectory shape")
        np.testing.assert_allclose(data["traces"][:, :, 0], data["traces"][:, :, 1], atol=1e-10, rtol=0)
        actual_initial = data["traces"][:, 0, 0, 0]
        np.testing.assert_allclose(actual_initial, state_batch, atol=1e-6, rtol=0)
        images = data["images"]
        np.savez_compressed(PUBLIC, previous=images[::3, 0, :, 0], current=images[1::3, 0, :, 0],
                            goal=images[2::3, 0, :, 0], previous_proprio=data["traces"][::3, 0, :, 0],
                            current_proprio=data["traces"][1::3, 0, :, 0], previous_actions=previous_actions)
    print(f"Prepared public planner inputs and separate physical payload: {PUBLIC}", flush=True)


def score():
    for path in (SELECTION, FEATURES):
        if path.exists():
            raise FileExistsError(path)
    print("Causal correction: previous prediction error only; no future physical file opened.", flush=True)
    model, dataset, permission = load_model()
    fingerprint = model_fingerprint(model)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    features, rows = {}, []
    with np.load(PUBLIC, allow_pickle=False) as data, torch.inference_mode():
        for i in range(8):
            for camera in range(2):
                def obs(image_key, state_key):
                    return {"visual": rgb_tensor(data[image_key][i, camera]),
                            "proprio": torch.tensor(data[state_key][i, camera], dtype=torch.float32,
                                                    device="cuda")[None, None]}
                previous, current = model.encode(obs("previous", "previous_proprio")), model.encode(
                    obs("current", "current_proprio"))
                goal = model.encode(rgb_tensor(data["goal"][i, camera]))[0]
                previous_raw = torch.tensor(data["previous_actions"][i], device="cuda", dtype=torch.float32)
                previous_packet = ((previous_raw - mean) / std).reshape(1, 1, 10)
                prior_prediction = model.unroll(previous, previous_packet)["visual"][-1]
                current_latent = current["visual"][:, -1]
                if current_latent.shape != prior_prediction.shape:
                    raise ValueError("previous prediction/current observation shape mismatch")
                raw_latents, corrected_latents = [], []
                for x in (0., .1, -.1):
                    raw = torch.tensor([[x, 0.]] * 5, device="cuda")
                    packet = ((raw - mean) / std).reshape(1, 1, 10)
                    predicted = model.unroll(current, packet)["visual"][-1]
                    raw_latents.append(predicted)
                    corrected_latents.append(correct(predicted, prior_prediction, current_latent))
                predicted_costs = {
                    "raw": [float((z-goal).square().mean()) for z in raw_latents],
                    "corrected": [float((z-goal).square().mean()) for z in corrected_latents],
                    "persistence": [float((current_latent-goal).square().mean())] * 3}
                choices = {name: int(np.argmin(costs)) for name, costs in predicted_costs.items()}
                key = f"r{1884+i}_c{camera}"
                features[key+"_raw"] = torch.stack(raw_latents).cpu().numpy()
                features[key+"_corrected"] = torch.stack(corrected_latents).cpu().numpy()
                rows.append({"root": 1884+i, "camera": camera, "scores": predicted_costs,
                             "choices": choices,
                             "historical_residual_rms": float((current_latent-prior_prediction).square().mean().sqrt()),
                             "predictor_calls": 4})
                print(f"{key}: {choices}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    np.savez_compressed(FEATURES, **features)
    output = {"rows": rows, "model_fingerprint": fingerprint, "input_audit_hash": permission.fingerprint,
              "public_sha256": file_hash(PUBLIC), "features_sha256": file_hash(FEATURES),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "backend_sha256": file_hash(BACKEND), "prior_replay_sha256": file_hash(REPLAY),
              "predictor_calls": sum(row["predictor_calls"] for row in rows),
              "scope": "state-assisted; past residual and RGB/current proprio only"}
    with SELECTION.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Sealed choices and predicted features: {SELECTION}", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    record = json.loads(SELECTION.read_text())
    checks = (("public_sha256", PUBLIC), ("features_sha256", FEATURES), ("protocol_sha256", PROTOCOL),
              ("runner_sha256", Path(__file__)), ("backend_sha256", BACKEND), ("prior_replay_sha256", REPLAY))
    if any(record[name] != file_hash(path) for name, path in checks):
        raise ValueError("sealed dependency drift")
    model, _, permission = load_model()
    if model_fingerprint(model) != record["model_fingerprint"] or permission.fingerprint != record["input_audit_hash"]:
        raise ValueError("model input contract drift")
    rows = []
    with np.load(PRIVATE, allow_pickle=False) as data, np.load(FEATURES, allow_pickle=False) as predictions, torch.inference_mode():
        for row in record["rows"]:
            i, camera = row["root"]-1884, row["camera"]
            target = data["traces"][3*i+2, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][3*i+1, :, camera, -1, :2] - target, axis=-1)
            real_z = []
            for action in range(3):
                z = model.encode(rgb_tensor(data["images"][3*i+1, action, camera, -1]))[0]
                real_z.append(z.cpu().numpy())
            true_features = np.stack(real_z)
            key = f"r{row['root']}_c{camera}"
            latent_mse = {name: float(np.mean((predictions[key+"_"+name] - true_features) ** 2))
                          for name in ("raw", "corrected")}
            regret = {}
            for name, selected in row["choices"].items():
                if selected != int(np.argmin(row["scores"][name])):
                    raise ValueError("selection drift")
                regret[name] = float(distances[selected] - distances.min())
            rows.append({"root": row["root"], "camera": camera, "latent_mse": latent_mse, "regret": regret,
                         "physical_distances": distances.tolist(), "choices": row["choices"]})
    per_root = []
    for root in range(1884, 1892):
        pair = [r for r in rows if r["root"] == root]
        if len(pair) != 2:
            raise ValueError("missing camera outcome")
        per_root.append({"root": root,
                         "latent_mse": {k: float(np.mean([r["latent_mse"][k] for r in pair]))
                                        for k in ("raw", "corrected")},
                         "regret": {k: float(np.mean([r["regret"][k] for r in pair]))
                                    for k in ("raw", "corrected", "persistence")}})
    mse = {k: float(np.mean([r["latent_mse"][k] for r in per_root])) for k in ("raw", "corrected")}
    regrets = {k: float(np.mean([r["regret"][k] for r in per_root]))
               for k in ("raw", "corrected", "persistence")}
    not_worse = sum(r["regret"]["corrected"] <= r["regret"]["raw"] for r in per_root)
    improved = sum(r["regret"]["corrected"] < r["regret"]["raw"] for r in per_root)
    passed = bool(len(rows) == 16 and mse["corrected"] < mse["raw"] and regrets["corrected"] <= regrets["raw"]
                  and not_worse >= 5 and improved >= 1)
    output = {"rows": rows, "per_root": per_root, "mean_mse": mse, "mean_regret": regrets,
              "roots_not_worse": not_worse, "roots_strictly_improved": improved, "development_exit_met": passed,
              "selection_sha256": file_hash(SELECTION), "physical_sha256": file_hash(PRIVATE),
              "scope": "state-assisted exposed-task post-hoc evaluation; not RGB-only or closed-loop"}
    with AUDIT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("mean_mse", "mean_regret", "roots_not_worse",
                                             "roots_strictly_improved", "development_exit_met")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "score", "audit"):
        raise SystemExit("usage: python -m experiments.pc_pointmaze_causal_residual_dev_20261106_codex prepare|score|audit")
    {"prepare": prepare, "score": score, "audit": audit}[sys.argv[1]]()
