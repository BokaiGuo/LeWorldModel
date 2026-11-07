"""Three-process state-assisted camera ensemble experiment on exposed roots."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
NAME = "pc_camera_multiview_dev_20261106_codex"
PRIVATE = ROOT / "results" / (NAME + "_physical.npz")
PUBLIC = ROOT / "results" / (NAME + "_inputs.npz")
SELECT = ROOT / "results" / (NAME + "_selection.json")
AUDIT = ROOT / "results" / (NAME + "_audit.json")
PROTOCOL = ROOT.parent / "docs/PC_CAMERA_MULTIVIEW_ENSEMBLE_DEV_PROTOCOL_20261106_CODEX.md"
BACKEND = ROOT / "experiments/pc_real_camera_effect_dev_20261106_codex.py"
OFFSETS = ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2))


def choose(scores):
    """Raw per-view costs, shape (5, 3)."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (5, 3) or not np.isfinite(values).all():
        raise ValueError("expected finite 5x3 scores")
    individual = np.argmin(values, axis=1)
    return {"centered": int(individual[0]), "multi": int(np.argmin(values.mean(axis=0))),
            "guard": int(individual[0]) if np.all(individual == individual[0]) else 0,
            "zero": 0}


def prepare():
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    for p in (PRIVATE, PUBLIC):
        if p.exists():
            raise FileExistsError(p)
    archive = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True)
    states = np.stack([archive[root, frame].numpy() for root in range(1884, 1892)
                       for frame in (0, 5)])
    with tempfile.TemporaryDirectory(prefix="pc_multiview_") as tmp:
        path = Path(tmp)
        request, xml = path / "request.npz", path / "scene.xml"
        np.savez_compressed(request, states=states)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(BACKEND), "--backend",
                        str(request), str(xml), str(PRIVATE)], env=env, check=True, timeout=240)
    with np.load(PRIVATE, allow_pickle=False) as data:
        if data["traces"].shape != (16, 3, 2, 6, 4):
            raise ValueError("unexpected trace shape")
        np.testing.assert_allclose(data["traces"][:, :, 0], data["traces"][:, :, 1], atol=1e-10, rtol=0)
        images = data["images"]
        np.savez_compressed(PUBLIC, current=images[::2, 0, :, 0], goal=images[1::2, 0, :, 0],
                            proprio=data["traces"][::2, 0, :, 0])
    print(f"Prepared separate input and physical payload: {PUBLIC}", flush=True)


def score():
    import torch
    from experiments.pc_input_permissions import StateAssistedPointMaze, manifest
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
    from experiments.pc_state_track_smoke_20261106_codex import load_model
    if SELECT.exists():
        raise FileExistsError(SELECT)
    print("GPU evaluation: fixed inputs only, no physical outcome file opened.", flush=True)
    model, dataset, permission = load_model()
    fingerprint = model_fingerprint(model)
    wrapper = StateAssistedPointMaze(model, permission)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    with np.load(PUBLIC, allow_pickle=False) as data, torch.inference_mode():
        for i in range(8):
            for camera in range(2):
                current, goal, proprio = (data[k][i, camera].copy() for k in ("current", "goal", "proprio"))
                row = {"root": 1884 + i, "camera": camera}
                try:
                    shift = goal_shift(goal)
                    shifts = [(shift["dx"] + dx, shift["dy"] + dy) for dx, dy in OFFSETS]
                    pairs = [(current, goal)] + [(translate(current, dx, dy)[0],
                                                   translate(goal, dx, dy)[0]) for dx, dy in shifts]
                except ValueError as exc:
                    row["rejected"] = str(exc)
                    rows.append(row)
                    continue
                costs, timings = [], []
                for image, target in pairs:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    model_inputs = {"visual": rgb_tensor(image),
                                    "proprio": torch.tensor(proprio, device="cuda", dtype=torch.float32)[None, None]}
                    goal_embedding = model.encode(rgb_tensor(target))[0]
                    values = []
                    for x in (0., .1, -.1):
                        raw = torch.tensor([[x, 0.]] * 5, device="cuda")
                        packet = ((raw - mean) / std).reshape(1, 1, 10)
                        pred = wrapper.predict(model_inputs, packet)["visual"][-1]
                        values.append(float((pred - goal_embedding).square().mean()))
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - started)
                    costs.append(values)
                selected = choose(costs[1:])
                selected["raw"] = int(np.argmin(costs[0]))
                row.update(shift=shift, scores=costs, choices=selected, per_view_seconds=timings)
                rows.append(row)
                print(f"root={row['root']} camera={camera} choices={selected}", flush=True)
    if fingerprint != model_fingerprint(model):
        raise ValueError("frozen model changed")
    output = manifest(permission, algorithm_version="fixed_multiview_cost_mean_v1",
                      protocol_path=PROTOCOL, task_manifest_path=PUBLIC, receipts=wrapper.receipts)
    output.update(rows=rows, model_fingerprint=fingerprint, runner_sha256=file_hash(__file__),
                  transform_sha256=file_hash(ROOT / "experiments/pc_goal_rgb_canonicalization.py"),
                  scope="state-assisted exposed development roots; selection input excludes physical outcomes")
    with SELECT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Sealed {len(rows)} decisions and {len(wrapper.receipts)} GPU predictions: {SELECT}")


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    record = json.loads(SELECT.read_text())
    if (record["protocol_hash"] != file_hash(PROTOCOL) or record["task_manifest_hash"] != file_hash(PUBLIC)
            or record["runner_sha256"] != file_hash(__file__)):
        raise ValueError("experiment binding changed")
    names = ("raw", "centered", "multi", "guard", "zero")
    rows = []
    with np.load(PRIVATE, allow_pickle=False) as data:
        for row in record["rows"]:
            if "rejected" in row:
                rows.append(row)
                continue
            idx, camera = row["root"] - 1884, row["camera"]
            target = data["traces"][2*idx+1, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][2*idx, :, camera, -1, :2] - target, axis=-1)
            check = choose(row["scores"][1:])
            check["raw"] = int(np.argmin(row["scores"][0]))
            if check != row["choices"]:
                raise ValueError("selection does not match sealed scores")
            rows.append({"root": row["root"], "camera": camera, "choices": check,
                         "regret": {k: float(distances[v] - distances.min()) for k, v in check.items()}})
    panels = {}
    for start in (1884, 1888):
        subset = [r for r in rows if start <= r["root"] < start + 4 and "regret" in r]
        complete = len(subset) == 8
        panel = {"valid": len(subset), "total": 8}
        if complete:
            panel["mean_regret"] = {k: float(np.mean([r["regret"][k] for r in subset])) for k in names}
            panel["view_disagreements"] = {
                k: sum(next(r for r in subset if r["root"] == root and r["camera"] == 0)["choices"][k]
                       != next(r for r in subset if r["root"] == root and r["camera"] == 1)["choices"][k]
                       for root in range(start, start + 4)) for k in names}
        panels[str(start)] = panel
    complete = all(p["valid"] == 8 for p in panels.values())
    case = next((r for r in rows if r["root"] == 1887 and r["camera"] == 1 and "regret" in r), None)
    passed = bool(complete and case and case["regret"]["multi"] < case["regret"]["centered"]
                  and all(p["mean_regret"]["multi"] <= p["mean_regret"]["centered"]
                          and p["view_disagreements"]["multi"] <= p["view_disagreements"]["centered"]
                          for p in panels.values()))
    result = {"selection_sha256": file_hash(SELECT), "physical_sha256": file_hash(PRIVATE),
              "rows": rows, "panels": panels, "development_exit_met": passed,
              "scope": "state-assisted; previously exposed roots; not independent or closed loop"}
    with AUDIT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"panels": panels, "1887_camera1": case, "development_exit_met": passed}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "score", "audit"):
        raise SystemExit("usage: python -m experiments.pc_camera_multiview_dev_20261106_codex prepare|score|audit")
    {"prepare": prepare, "score": score, "audit": audit}[sys.argv[1]]()
