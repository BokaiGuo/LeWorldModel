"""Previously exposed roots: seal model selections before physical evaluation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
from experiments.pc_input_permissions import StateAssistedPointMaze, file_hash, manifest
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model
from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME

NAME = "pc_camera_guard_external_dev_20261106_codex"
PROTOCOL = ROOT.parent / "docs/PC_CAMERA_DISAGREEMENT_GUARD_DEV_PROTOCOL_20261106_CODEX.md"
BACKEND = ROOT / "experiments/pc_real_camera_effect_dev_20261106_codex.py"
PAYLOAD = ROOT / "results" / (NAME + ".npz")
SELECTION = ROOT / "results" / (NAME + "_selection.json")
AUDIT = ROOT / "results" / (NAME + "_audit.json")


def main():
    for path in (PAYLOAD, SELECTION, AUDIT):
        if path.exists():
            raise FileExistsError(path)
    archive_path = DATA / "states.pth"
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    states = np.stack([archive[root, frame].numpy()
                       for root in range(1884, 1888) for frame in (0, 5)])
    print("Generate fixed exposed-root physical replay; seal predictions before reading outcomes.", flush=True)
    with tempfile.TemporaryDirectory(prefix="pc_guard_") as temp:
        folder = Path(temp)
        request, xml = folder / "request.npz", folder / "scene.xml"
        np.savez_compressed(request, states=states)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(BACKEND), "--backend",
                        str(request), str(xml), str(PAYLOAD)], env=env, check=True, timeout=180)
    with np.load(PAYLOAD, allow_pickle=False) as data:
        if data["traces"].shape != (8, 3, 2, 6, 4):
            raise ValueError("unexpected physical trajectory shape")
        np.testing.assert_allclose(data["traces"][:, :, 0], data["traces"][:, :, 1], atol=1e-10, rtol=0)
        inputs = [(1884 + i, camera, data["images"][2*i, 0, camera, 0].copy(),
                   data["images"][2*i+1, 0, camera, 0].copy(),
                   data["traces"][2*i, 0, camera, 0].copy())
                  for i in range(4) for camera in range(2)]
    model, dataset, permission = load_model()
    fingerprint = model_fingerprint(model)
    wrapper = StateAssistedPointMaze(model, permission)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    with torch.inference_mode():
        for root, camera, current, goal, proprio in inputs:
            try:
                shift = goal_shift(goal)
                centered_current, mask = translate(current, shift["dx"], shift["dy"])
                centered_goal, target_mask = translate(goal, shift["dx"], shift["dy"])
                if not np.array_equal(mask, target_mask):
                    raise ValueError("inconsistent masks")
            except ValueError as exc:
                rows.append({"root": root, "camera": camera, "rejected": str(exc)})
                continue
            arms = {}
            for name, image, target in (("raw", current, goal),
                                        ("centered", centered_current, centered_goal)):
                torch.cuda.synchronize()
                started = time.perf_counter()
                inputs_model = {"visual": rgb_tensor(image),
                                "proprio": torch.tensor(proprio, device="cuda", dtype=torch.float32)[None, None]}
                target_emb = model.encode(rgb_tensor(target))[0]
                scores = []
                for action in (0., .1, -.1):
                    raw = torch.tensor([[action, 0.]] * 5, device="cuda")
                    packet = ((raw - mean) / std).reshape(1, 1, 10)
                    pred = wrapper.predict(inputs_model, packet)["visual"][-1]
                    scores.append(float((pred - target_emb).square().mean()))
                if not np.isfinite(scores).all():
                    raise ValueError("nonfinite model scores")
                torch.cuda.synchronize()
                arms[name] = {"scores": scores, "selected": int(np.argmin(scores)),
                              "seconds": time.perf_counter() - started}
            raw_sel, center_sel = arms["raw"]["selected"], arms["centered"]["selected"]
            rows.append({"root": root, "camera": camera, "shift": shift,
                         "valid_pixels": int(mask.sum()), "arms": arms,
                         "guard": raw_sel if raw_sel == center_sel else 0, "zero": 0})
            print(f"root={root} camera={camera} raw={raw_sel} centered={center_sel} guard={rows[-1]['guard']}", flush=True)
    if fingerprint != model_fingerprint(model):
        raise ValueError("model changed")
    result = manifest(permission, algorithm_version="camera_disagreement_guard_dev_v1",
                      protocol_path=PROTOCOL, task_manifest_path=PAYLOAD, receipts=wrapper.receipts)
    result.update(rows=rows, model_fingerprint=fingerprint, archive_sha256=file_hash(archive_path),
                  runner_sha256=file_hash(__file__), backend_sha256=file_hash(BACKEND),
                  scope="state-assisted; previously exposed development roots; no RGB-only claim")
    with SELECTION.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Sealed selections: {SELECTION}", flush=True)

    # Only after sealing, evaluate against physical outcomes.
    names = ("raw", "centered", "guard", "zero")
    with np.load(PAYLOAD, allow_pickle=False) as data:
        audited = []
        for row in rows:
            if "rejected" in row:
                audited.append(row)
                continue
            i, camera = row["root"] - 1884, row["camera"]
            goal_state = data["traces"][2*i+1, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][2*i, :, camera, -1, :2] - goal_state, axis=-1)
            choices = {"raw": row["arms"]["raw"]["selected"], "centered": row["arms"]["centered"]["selected"],
                       "guard": row["guard"], "zero": 0}
            audited.append({"root": row["root"], "camera": camera, "choices": choices,
                            "regret": {k: float(distances[v] - distances.min()) for k, v in choices.items()}})
    valid = [row for row in audited if "regret" in row]
    means = {name: float(np.mean([row["regret"][name] for row in valid])) for name in names} if valid else {}
    all_valid = len(valid) == 8
    per_root = {}
    if all_valid:
        for root in range(1884, 1888):
            pair = [row for row in valid if row["root"] == root]
            per_root[str(root)] = {name: float(np.mean([row["regret"][name] for row in pair])) for name in names}
    exit_met = bool(all_valid and means["guard"] <= means["raw"] and means["guard"] <= means["centered"]
                    and all(not (p["guard"] > p["raw"] and p["guard"] > p["centered"])
                            for p in per_root.values()))
    output = {"selection_sha256": file_hash(SELECTION), "rows": audited, "mean_regret": means,
              "per_root": per_root, "geometry_complete": all_valid, "development_exit_met": exit_met,
              "scope": "post-hoc physical evaluation, state-assisted exposed development roots"}
    with AUDIT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
