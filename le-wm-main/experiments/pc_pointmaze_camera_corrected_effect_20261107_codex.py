"""Sealed fixed-bank state-assisted comparison after camera calibration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

from experiments import pc_pointmaze_camera_geometry_20261107_codex as geometry
from experiments import pc_pointmaze_packetwise_three_way_20261107_codex as prior
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CAMERA_CORRECTED_EFFECT_PROTOCOL_20261107_CODEX.md"
WORKER = ROOT / "experiments/pc_pointmaze_corrected_camera_render_20261107_codex.py"
BASE = ROOT / "results/pc_pointmaze_camera_corrected_effect_20261107_codex"
INPUT = Path(str(BASE) + "_input.npz")
PRED = Path(str(BASE) + "_prediction.npz")
PLAN = Path(str(BASE) + "_plan.json")
FRAMES = Path(str(BASE) + "_frames.npz")
RENDER = Path(str(BASE) + "_render.json")
RESULT = Path(str(BASE) + "_result.json")


def render_states(states):
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("invalid states")
    with tempfile.TemporaryDirectory(prefix="pc_corrected_camera_") as temp:
        folder = Path(temp)
        req, xml, res = folder / "req.npz", folder / "scene.xml", folder / "res.npz"
        with req.open("xb") as f:
            np.savez_compressed(f, states=states)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(req), str(xml), str(res)],
                       env=env, check=True, timeout=180)
        with np.load(res, allow_pickle=False) as f:
            images = f["frames"].copy()
    if images.shape != (len(states), 224, 224, 3):
        raise ValueError("invalid rendered images")
    return images


def prepare():
    if INPUT.exists():
        raise FileExistsError(INPUT)
    calibration = json.loads(geometry.RESULT.read_text())
    if calibration["frames"] != file_hash(geometry.FRAMES) or calibration["selected_camera"] != [3., 3.]:
        raise ValueError("camera source drift")
    with np.load(geometry.FRAMES, allow_pickle=False) as images, np.load(prior.prior.old.inputs.PUBLIC, allow_pickle=False) as public:
        current = images["frames"][:, calibration["selected_index"]].copy()
        np.testing.assert_array_equal(current, images["reference"])
        proprio, goal_proprio = public["proprio"][:, 0].copy(), public["goal_proprio"][:, 0].copy()
    goals = render_states(goal_proprio)
    with INPUT.open("xb") as f:
        np.savez_compressed(f, current=current, goal=goals, proprio=proprio, goal_proprio=goal_proprio,
                            calibration_sha256=np.asarray(file_hash(geometry.RESULT)),
                            source_public_sha256=np.asarray(file_hash(prior.prior.old.inputs.PUBLIC)),
                            worker_sha256=np.asarray(file_hash(WORKER)))
    print("Corrected current and goal observations prepared before physical outcomes.", flush=True)


def predict():
    if PRED.exists() or PLAN.exists():
        raise FileExistsError("prediction exists")
    prior.check_plan()
    model, _, audit1 = load_model()
    old_plan = json.loads(prior.prior.PLAN.read_text())
    fingerprint = model_fingerprint(model)
    if fingerprint != old_plan["model_sha256"] or audit1.fingerprint != old_plan["audit1"]:
        raise ValueError("model drift")
    audit2 = configure_context(model, audit1)
    if audit2.fingerprint != old_plan["audit2"]:
        raise ValueError("context drift")
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(prior.prior.old.inputs.CONFIG.read_text())
    arrays, rows = {}, []
    with (np.load(INPUT, allow_pickle=False) as public, np.load(prior.prior.old.POOL, allow_pickle=False) as pool,
          torch.inference_mode()):
        if (str(public["calibration_sha256"]) != file_hash(geometry.RESULT)
                or str(public["worker_sha256"]) != file_hash(WORKER)
                or str(public["source_public_sha256"]) != file_hash(prior.prior.old.inputs.PUBLIC)):
            raise ValueError("input drift")
        actions = pool["actions"]
        np.testing.assert_array_equal(actions, prior.prior.old.action_bank())
        for index in range(8):
            root = 1884 + index
            current = {"visual": rgb_tensor(public["current"][index]),
                       "proprio": torch.as_tensor(public["proprio"][index], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(public["goal"][index]),
                    "proprio": torch.as_tensor(public["goal_proprio"][index], dtype=torch.float32, device="cuda")[None, None]}
            objective = ReprTargetDistMPCObjective(cfg, model.encode(dict(goal)), alpha=.1)
            windows = {}
            for window, audit in ((1, audit1), (2, audit2)):
                model.ctxt_window = window
                boundary = StateAssistedPointMaze(model, audit)
                values, costs = {"visual": [], "proprio": []}, []
                for start in (0, 4):
                    raw = torch.as_tensor(actions[start:start + 4], dtype=torch.float32)
                    packets = model.preprocessor.normalize_actions(raw).reshape(4, 5, 10).transpose(0, 1).contiguous().cuda()
                    expanded = {k: v.expand(4, *v.shape[1:]).contiguous() for k, v in current.items()}
                    pred = boundary.predict(expanded, packets)
                    costs.extend(objective(pred, packets).cpu().tolist())
                    for key in values:
                        if pred[key].shape[:2] != (6, 4):
                            raise ValueError("clock mismatch")
                        values[key].append(pred[key].cpu().numpy())
                for key in values:
                    arrays[f"r{root}_w{window}_{key}"] = np.concatenate(values[key], axis=1)
                windows[str(window)] = {"cost": costs, "choice": int(np.argmin(costs))}
            rows.append({"root": root, "windows": windows})
            print(f"root={root} predicted", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "worker": file_hash(WORKER), "input": file_hash(INPUT), "prediction": file_hash(PRED),
                   "old_plan": file_hash(prior.prior.old.PLAN), "model": fingerprint, "rows": rows},
                  f, indent=2, allow_nan=False)


def check_plan():
    plan = json.loads(PLAN.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("worker", WORKER),
                      ("input", INPUT), ("prediction", PRED), ("old_plan", prior.prior.old.PLAN)):
        if plan[key] != file_hash(path):
            raise ValueError(f"sealed drift: {key}")
    return plan


def render():
    if FRAMES.exists() or RENDER.exists():
        raise FileExistsError("render exists")
    check_plan()
    physical = json.loads(prior.prior.old.PHYSICAL.read_text())
    if physical["plan_sha256"] != file_hash(prior.prior.old.PLAN):
        raise ValueError("physical trace drift")
    arrays = {}
    with np.load(INPUT, allow_pickle=False) as inputs:
        for i, row in enumerate(physical["rows"]):
            if row["root"] != 1884 + i:
                raise ValueError("root mismatch")
            trace = np.asarray(row["traces"])
            if trace.shape != (8, 26, 5):
                raise ValueError("trace shape")
            np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (8, 26)), atol=1e-10, rtol=0)
            images = render_states(trace[:, ::5, :4].reshape(-1, 4)).reshape(8, 6, 224, 224, 3)
            np.testing.assert_array_equal(images[:, 0], np.broadcast_to(inputs["current"][i], images[:, 0].shape))
            arrays[f"r{1884+i}"] = images
            print(f"root={1884+i} corrected physical images rendered", flush=True)
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with RENDER.open("x") as f:
        json.dump({"plan": file_hash(PLAN), "physical": file_hash(prior.prior.old.PHYSICAL),
                   "frames": file_hash(FRAMES)}, f, indent=2)


def rank(a, b):
    correct, count = prior.prior.old.agreement(a, b)
    return {"correct": correct, "pairs": count, "accuracy": correct / count if count else None}


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, render_meta = check_plan(), json.loads(RENDER.read_text())
    for key, path in (("plan", PLAN), ("physical", prior.prior.old.PHYSICAL), ("frames", FRAMES)):
        if render_meta[key] != file_hash(path):
            raise ValueError(f"render drift {key}")
    model, _, audit = load_model()
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model drift")
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(prior.prior.old.inputs.CONFIG.read_text())
    physical = json.loads(prior.prior.old.PHYSICAL.read_text())
    rows = []
    with (np.load(INPUT, allow_pickle=False) as inputs, np.load(PRED, allow_pickle=False) as pred,
          np.load(FRAMES, allow_pickle=False) as frames, torch.inference_mode()):
        for index, real in enumerate(physical["rows"]):
            root = 1884 + index
            if real["root"] != root or plan["rows"][index]["root"] != root:
                raise ValueError("root mismatch")
            goal = {"visual": rgb_tensor(inputs["goal"][index]),
                    "proprio": torch.as_tensor(inputs["goal_proprio"][index], dtype=torch.float32, device="cuda")[None, None]}
            objective = ReprTargetDistMPCObjective(cfg, model.encode(dict(goal)), alpha=.1)
            trace = np.asarray(real["traces"])
            actual = {"visual": [], "proprio": []}
            costs = []
            for candidate in range(8):
                observations = []
                for step in range(6):
                    state = torch.as_tensor(trace[candidate, step * 5, :4], dtype=torch.float32, device="cuda")[None, None]
                    observations.append(model.encode({"visual": rgb_tensor(frames[f"r{root}"][candidate, step]),
                                                      "proprio": state}))
                costs.append(float(objective(observations[-1], None).item()))
                for key in actual:
                    actual[key].append(np.stack([o[key][0, 0].cpu().numpy() for o in observations]))
            actual = {key: np.stack(value).swapaxes(0, 1) for key, value in actual.items()}
            steps = []
            for step in range(1, 6):
                metric = {"environment_step": step * 5}
                for key in actual:
                    metric[key] = {"persistence": prior.strict_mse(actual[key][0], actual[key][step])}
                    for window in (1, 2):
                        metric[key][f"window{window}"] = prior.strict_mse(pred[f"r{root}_w{window}_{key}"][step], actual[key][step])
                steps.append(metric)
            distance = np.asarray(real["final_distance"])
            windows = {}
            for window in ("1", "2"):
                case = plan["rows"][index]["windows"][window]
                windows[window] = {"pred_vs_encoded": rank(case["cost"], costs),
                                   "pred_vs_physical": rank(case["cost"], distance),
                                   "choice": case["choice"], "regret": float(distance[case["choice"]] - distance.min())}
            rows.append({"root": root, "initial_success": real["initial_success"], "steps": steps,
                         "windows": windows, "encoded_vs_physical": rank(costs, distance),
                         "encoded_cost": costs, "encoded_choice": int(np.argmin(costs))})
            print(f"root={root} corrected encoding evaluated", flush=True)
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model changed")
    summary = {"steps": [], "windows": {}}
    for i in range(5):
        item = {"environment_step": (i + 1) * 5}
        for key in ("visual", "proprio"):
            item[key] = {name: float(np.mean([r["steps"][i][key][name] for r in rows]))
                         for name in ("persistence", "window1", "window2")}
        summary["steps"].append(item)
    for window in ("1", "2"):
        summary["windows"][window] = {name: float(np.mean([r["windows"][window][name]["accuracy"] for r in rows]))
                                       for name in ("pred_vs_encoded", "pred_vs_physical")}
        summary["windows"][window]["mean_regret"] = float(np.mean([r["windows"][window]["regret"] for r in rows]))
    summary["encoded_vs_physical"] = float(np.mean([r["encoded_vs_physical"]["accuracy"] for r in rows]))
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "render": file_hash(RENDER),
                   "scope": "exposed calibrated-camera state-assisted diagnostic",
                   "summary": summary, "rows": rows}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "predict", "render", "evaluate"):
        raise SystemExit("expected prepare|predict|render|evaluate")
    {"prepare": prepare, "predict": predict, "render": render, "evaluate": evaluate}[sys.argv[1]]()
