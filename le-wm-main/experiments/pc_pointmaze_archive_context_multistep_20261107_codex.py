"""Sealed same-trajectory context comparison, followed by physical/latent audit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from experiments import pc_fixed_prefix_state_physical_20261107_codex as physics
from experiments import pc_pointmaze_archive_render_domain_20261107_codex as prior
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_ARCHIVE_CONTEXT_MULTISTEP_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_archive_context_multistep_20261107_codex"
INPUT = Path(str(BASE) + "_input.npz")
PRED = Path(str(BASE) + "_prediction.npz")
PLAN = Path(str(BASE) + "_plan.json")
RESULT = Path(str(BASE) + "_result.json")
ROOTS = range(1884, 1892)
FOLDER = ROOT / "data/point_maze_tail1800_1899"


def mse(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid feature comparison")
    return float(np.square(a - b).mean())


def prepare():
    if INPUT.exists():
        raise FileExistsError(INPUT)
    states = torch.load(FOLDER / "states.pth", map_location="cpu", weights_only=True)
    actions = torch.load(FOLDER / "actions.pth", map_location="cpu", weights_only=True)
    frames = [torch.load(prior.archive_path(root), map_location="cpu", weights_only=True)[0].numpy() for root in ROOTS]
    with INPUT.open("xb") as f:
        np.savez_compressed(f, images=np.stack(frames), states=states[1884:1892, 0].float().numpy(),
                            actions=actions[1884:1892, :25].float().numpy(),
                            archive_hashes=np.asarray([file_hash(prior.archive_path(root)) for root in ROOTS]),
                            states_hash=np.asarray(file_hash(FOLDER / "states.pth")),
                            actions_hash=np.asarray(file_hash(FOLDER / "actions.pth")))
    print("Prepared initial observations and fixed actions; future frames not exported.")


def predict():
    if PRED.exists() or PLAN.exists():
        raise FileExistsError("sealed prediction exists")
    print("Predict both windows before reading future observations.", flush=True)
    model, dataset, audit1 = load_model()
    fingerprint = model_fingerprint(model)
    if model.ctxt_window != 1:
        raise ValueError("unexpected context-1 loader")
    audit2 = configure_context(model, audit1)
    arrays, rows = {}, []
    with np.load(INPUT, allow_pickle=False) as inputs, torch.inference_mode():
        for index, root in enumerate(ROOTS):
            current = {"visual": rgb_tensor(inputs["images"][index]),
                       "proprio": torch.as_tensor(inputs["states"][index], dtype=torch.float32,
                                                   device="cuda")[None, None]}
            raw = torch.as_tensor(inputs["actions"][index], dtype=torch.float32)
            packets = model.preprocessor.normalize_actions(raw).reshape(5, 1, 10).cuda()
            receipts = {}
            for window, audit in ((1, audit1), (2, audit2)):
                model.ctxt_window = window
                boundary = StateAssistedPointMaze(model, audit)
                output = boundary.predict(current, packets)
                for key in ("visual", "proprio"):
                    value = output[key].detach().cpu().numpy()
                    if value.shape[:2] != (6, 1) or not np.isfinite(value).all():
                        raise ValueError("bad output clock")
                    arrays[f"r{root}_w{window}_{key}"] = value
                receipts[str(window)] = boundary.receipts
            first_diff = max(float(np.max(np.abs(arrays[f"r{root}_w1_{k}"][1] - arrays[f"r{root}_w2_{k}"][1])))
                             for k in ("visual", "proprio"))
            if first_diff > 1e-5:
                raise ValueError("first packet mismatch")
            rows.append({"root": root, "first_packet_max_error": first_diff, "receipts": receipts})
            print(f"root={root} predicted", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "input_sha256": file_hash(INPUT), "pred_sha256": file_hash(PRED),
                   "model_sha256": fingerprint, "audit1": audit1.fingerprint, "audit2": audit2.fingerprint,
                   "rows": rows, "scope": "factual actions only, no future observations"}, f, indent=2)


def evaluate():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(PLAN.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("input_sha256", INPUT), ("pred_sha256", PRED)):
        if plan[key] != file_hash(path):
            raise ValueError(f"sealed drift: {key}")
    model, _, audit1 = load_model()
    if model_fingerprint(model) != plan["model_sha256"] or audit1.fingerprint != plan["audit1"]:
        raise ValueError("model or audit changed")
    audit2 = configure_context(model, audit1)
    if audit2.fingerprint != plan["audit2"]:
        raise ValueError("context2 audit changed")
    states = torch.load(FOLDER / "states.pth", map_location="cpu", weights_only=True)
    xml = official_xml()
    rows = []
    with (np.load(INPUT, allow_pickle=False) as inputs, np.load(PRED, allow_pickle=False) as pred,
          tempfile.TemporaryDirectory(prefix="pc_archive_context_") as temporary, torch.inference_mode()):
        if str(inputs["states_hash"]) != file_hash(FOLDER / "states.pth") or str(inputs["actions_hash"]) != file_hash(FOLDER / "actions.pth"):
            raise ValueError("archive drift")
        folder = Path(temporary)
        xml_path = folder / "scene.xml"
        xml_path.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, root in enumerate(ROOTS):
            archive = prior.archive_path(root)
            if str(inputs["archive_hashes"][index]) != file_hash(archive):
                raise ValueError("image archive drift")
            request, response = folder / f"req_{root}.npz", folder / f"res_{root}.npz"
            with request.open("xb") as f:
                np.savez_compressed(f, state=inputs["states"][index], actions=inputs["actions"][index][None])
            subprocess.run([str(LEGACY / "bin/python"), str(Path(physics.__file__).resolve()),
                            "--backend", str(request), str(xml_path), str(response)], env=env, check=True, timeout=120)
            with np.load(response, allow_pickle=False) as replay:
                trace = replay["trace"][0].copy()
            if trace.shape != (26, 5):
                raise ValueError("bad replay shape")
            np.testing.assert_allclose(trace[:, 4], np.arange(26) * .05, rtol=0, atol=1e-10)
            physical_error = float(np.max(np.abs(trace[:, :4] - states[root, :26].numpy())))
            if physical_error > 1e-4:
                raise ValueError(f"physical state mismatch root {root}: {physical_error}")
            images = torch.load(archive, map_location="cpu", weights_only=True)
            metrics = []
            for step in range(1, 6):
                t = step * 5
                observation = {"visual": rgb_tensor(images[t].numpy()),
                               "proprio": states[root, t].float().cuda()[None, None]}
                actual = model.encode(observation)
                values = {}
                for key in ("visual", "proprio"):
                    truth = actual[key][0, 0].cpu().numpy()
                    values[key] = {f"window{window}": mse(pred[f"r{root}_w{window}_{key}"][step, 0], truth)
                                   for window in (1, 2)}
                    values[key]["persistence"] = mse(pred[f"r{root}_w1_{key}"][0, 0], truth)
                metrics.append({"environment_step": t, **values})
            rows.append({"root": root, "physical_max_error": physical_error, "metrics": metrics})
            print(f"root={root} replayed and encoded", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    summary = []
    for step in range(5):
        per = {"environment_step": (step + 1) * 5}
        for key in ("visual", "proprio"):
            values = [r["metrics"][step][key] for r in rows]
            per[key] = {name: float(np.mean([v[name] for v in values]))
                        for name in ("window1", "window2", "persistence")}
            per[key]["window2_lower_roots"] = sum(v["window2"] < v["window1"] for v in values)
        summary.append(per)
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "plan_sha256": file_hash(PLAN), "rows": rows, "summary": summary,
                   "scope": "exposed archive factual trajectories; not decision or RGB evidence"}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "predict", "evaluate"):
        raise SystemExit("expected prepare|predict|evaluate")
    {"prepare": prepare, "predict": predict, "evaluate": evaluate}[sys.argv[1]]()
