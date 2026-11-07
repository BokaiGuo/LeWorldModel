"""Packetwise alignment on a previously sealed state-assisted action bank."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from experiments import pc_pointmaze_context_effect_three_way_20261107_codex as prior
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = prior.old.inputs.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PACKETWISE_THREE_WAY_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_packetwise_three_way_20261107_codex"
PRED = Path(str(BASE) + "_prediction.npz")
PLAN = Path(str(BASE) + "_plan.json")
FRAMES = Path(str(BASE) + "_frames.npz")
RENDER = Path(str(BASE) + "_render.json")
RESULT = Path(str(BASE) + "_result.json")
BACKEND = ROOT / "experiments/pc_pointmaze_packetwise_render_20261107_codex.py"


def strict_mse(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"invalid representation shapes: {left.shape} {right.shape}")
    return float(np.square(left - right).mean())


def predict():
    if PRED.exists() or PLAN.exists():
        raise FileExistsError("sealed prediction exists")
    sealed = json.loads(prior.PLAN.read_text())
    for field, path in (("protocol_sha256", prior.PROTOCOL), ("runner_sha256", prior.__file__),
                        ("prediction_sha256", prior.PRED), ("public_sha256", prior.old.inputs.PUBLIC),
                        ("pool_sha256", prior.old.POOL)):
        if sealed[field] != file_hash(path):
            raise ValueError(f"source drift: {field}")
    model, _, audit1 = load_model()
    if model_fingerprint(model) != sealed["model_sha256"] or audit1.fingerprint != sealed["audit1"]:
        raise ValueError("model drift")
    audit2 = configure_context(model, audit1)
    if audit2.fingerprint != sealed["audit2"]:
        raise ValueError("context drift")
    arrays, rows = {}, []
    with (np.load(prior.old.inputs.PUBLIC, allow_pickle=False) as public,
          np.load(prior.old.POOL, allow_pickle=False) as pool,
          np.load(prior.PRED, allow_pickle=False) as old, torch.inference_mode()):
        actions = pool["actions"]
        np.testing.assert_array_equal(actions, prior.old.action_bank())
        for index in range(8):
            root = 1884 + index
            inputs = {"visual": rgb_tensor(public["current"][index, 0]),
                      "proprio": torch.as_tensor(public["proprio"][index, 0],
                                                  dtype=torch.float32, device="cuda")[None, None]}
            errors = {}
            for window, audit in ((1, audit1), (2, audit2)):
                model.ctxt_window = window
                boundary = StateAssistedPointMaze(model, audit)
                values = {"visual": [], "proprio": []}
                for start in (0, 4):
                    raw = torch.as_tensor(actions[start:start + 4], dtype=torch.float32)
                    packets = model.preprocessor.normalize_actions(raw).reshape(4, 5, 10).transpose(0, 1).contiguous().cuda()
                    expanded = {k: v.expand(4, *v.shape[1:]).contiguous() for k, v in inputs.items()}
                    output = boundary.predict(expanded, packets)
                    for key in values:
                        if output[key].shape[:2] != (6, 4):
                            raise ValueError("prediction clock mismatch")
                        values[key].append(output[key].cpu().numpy())
                for key in values:
                    value = np.concatenate(values[key], axis=1)
                    reference = old[f"r{root}_w{window}_{key}"]
                    if value[-1].shape != reference.shape:
                        raise ValueError("endpoint shape mismatch")
                    error = float(np.max(np.abs(value[-1] - reference)))
                    if error > 1e-5:
                        raise ValueError(f"endpoint drift {root} {window} {key}: {error}")
                    errors[f"{window}_{key}"] = error
                    arrays[f"r{root}_w{window}_{key}"] = value
            rows.append({"root": root, "endpoint_max_abs": errors})
            print(f"root={root} packet predictions complete", flush=True)
    if model_fingerprint(model) != sealed["model_sha256"]:
        raise ValueError("model changed")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "source_plan": file_hash(prior.PLAN), "source_prediction": file_hash(prior.PRED),
                   "prediction": file_hash(PRED), "model": sealed["model_sha256"],
                   "rows": rows}, f, indent=2, allow_nan=False)


def check_plan():
    plan = json.loads(PLAN.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("source_plan", prior.PLAN),
                      ("source_prediction", prior.PRED), ("prediction", PRED)):
        if plan[key] != file_hash(path):
            raise ValueError(f"sealed drift: {key}")
    return plan


def render():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if FRAMES.exists() or RENDER.exists():
        raise FileExistsError("render exists")
    check_plan()
    physical = json.loads(prior.old.PHYSICAL.read_text())
    if physical["plan_sha256"] != file_hash(prior.old.PLAN):
        raise ValueError("physical drift")
    old_render = json.loads(prior.RENDER.read_text())
    if old_render["frames_sha256"] != file_hash(prior.FRAMES) or old_render["physical_sha256"] != file_hash(prior.old.PHYSICAL):
        raise ValueError("prior rendering drift")
    arrays, rows = {}, []
    with (np.load(prior.old.inputs.PUBLIC, allow_pickle=False) as public,
          np.load(prior.FRAMES, allow_pickle=False) as old,
          tempfile.TemporaryDirectory(prefix="pc_packet_render_") as temp):
        folder = Path(temp)
        xml = folder / "scene.xml"
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, row in enumerate(physical["rows"]):
            root = 1884 + index
            trace = np.asarray(row["traces"])
            if row["root"] != root or trace.shape != (8, 26, 5):
                raise ValueError("invalid physical trace")
            np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (8, 26)), atol=1e-10, rtol=0)
            req, out = folder / f"req_{root}.npz", folder / f"res_{root}.npz"
            with req.open("xb") as f:
                np.savez_compressed(f, states=trace[:, ::5, :4])
            subprocess.run([str(LEGACY / "bin/python"), str(BACKEND), str(req), str(xml), str(out)],
                           env=env, check=True, timeout=120)
            with np.load(out, allow_pickle=False) as f:
                images = f["frames"].copy()
            if images.shape != (8, 6, 224, 224, 3):
                raise ValueError("bad frames")
            np.testing.assert_array_equal(images[:, 0], np.broadcast_to(public["current"][index, 0], images[:, 0].shape))
            np.testing.assert_array_equal(images[:, -1], old[f"r{root}_frames"])
            arrays[f"r{root}"] = images
            rows.append({"root": root, "initial_and_final_pixel_max": 0})
            print(f"root={root} rendered and endpoints verified", flush=True)
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with RENDER.open("x") as f:
        json.dump({"plan": file_hash(PLAN), "physical": file_hash(prior.old.PHYSICAL),
                   "prior_render": file_hash(prior.RENDER), "frames": file_hash(FRAMES),
                   "backend": file_hash(BACKEND), "rows": rows}, f, indent=2)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = check_plan()
    metadata = json.loads(RENDER.read_text())
    for key, path in (("plan", PLAN), ("physical", prior.old.PHYSICAL),
                      ("prior_render", prior.RENDER), ("frames", FRAMES), ("backend", BACKEND)):
        if metadata[key] != file_hash(path):
            raise ValueError(f"render drift: {key}")
    model, _, audit = load_model()
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model drift")
    physical = json.loads(prior.old.PHYSICAL.read_text())
    rows = []
    with np.load(PRED, allow_pickle=False) as predicted, np.load(FRAMES, allow_pickle=False) as frames, torch.inference_mode():
        for index, physical_row in enumerate(physical["rows"]):
            root = 1884 + index
            trace = np.asarray(physical_row["traces"])
            actual = {"visual": [], "proprio": []}
            for candidate in range(8):
                observations = []
                for step in range(6):
                    state = torch.as_tensor(trace[candidate, step * 5, :4], dtype=torch.float32, device="cuda")[None, None]
                    observations.append(model.encode({"visual": rgb_tensor(frames[f"r{root}"][candidate, step]),
                                                      "proprio": state}))
                for key in actual:
                    actual[key].append(np.stack([v[key][0, 0].cpu().numpy() for v in observations]))
            actual = {key: np.stack(value).swapaxes(0, 1) for key, value in actual.items()}
            metrics = []
            for step in range(1, 6):
                item = {"environment_step": step * 5}
                for key in actual:
                    truth = actual[key][step]
                    values = {"persistence": strict_mse(actual[key][0], truth)}
                    for window in (1, 2):
                        p = predicted[f"r{root}_w{window}_{key}"][step]
                        values[f"window{window}"] = strict_mse(p, truth)
                        values[f"window{window}_previous_truth"] = strict_mse(p, actual[key][step - 1])
                        if step < 5:
                            values[f"window{window}_next_truth"] = strict_mse(p, actual[key][step + 1])
                    item[key] = values
                metrics.append(item)
            rows.append({"root": root, "initial_success": physical_row["initial_success"], "metrics": metrics})
            print(f"root={root} aligned", flush=True)
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model changed")
    summary = []
    for index in range(5):
        per = {"environment_step": (index + 1) * 5}
        for key in ("visual", "proprio"):
            values = [row["metrics"][index][key] for row in rows]
            per[key] = {name: float(np.mean([v[name] for v in values])) for name in values[0]}
        summary.append(per)
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "render": file_hash(RENDER),
                   "rows": rows, "summary": summary,
                   "scope": "exposed state-assisted fixed-bank evaluation; re-rendered successors"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "render", "evaluate"):
        raise SystemExit("expected predict|render|evaluate")
    {"predict": predict, "render": render, "evaluate": evaluate}[sys.argv[1]]()
