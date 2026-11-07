"""Three-way candidate alignment with prediction sealed before physical access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

from experiments import pc_pointmaze_context_multistep_effect_20261107_codex as old
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

PROTOCOL = old.inputs.ROOT.parent / "docs/PC_POINTMAZE_CONTEXT_EFFECT_THREE_WAY_PROTOCOL_20261107_CODEX.md"
BASE = old.inputs.ROOT / "results/pc_pointmaze_context_effect_three_way_20261107_codex"
PRED = Path(str(BASE) + "_prediction.npz")
PLAN = Path(str(BASE) + "_plan.json")
FRAMES = Path(str(BASE) + "_frames.npz")
RENDER = Path(str(BASE) + "_render.json")
RESULT = Path(str(BASE) + "_result.json")
BACKEND = old.inputs.ROOT / "experiments/pc_pointmaze_render_state_bank_20261107_codex.py"


def source_plan():
    plan = json.loads(old.PLAN.read_text())
    for key, path in (("protocol_sha256", old.PROTOCOL), ("runner_sha256", old.__file__),
                      ("pool_sha256", old.POOL), ("public_sha256", old.inputs.PUBLIC),
                      ("config_sha256", old.inputs.CONFIG)):
        if plan[key] != file_hash(path):
            raise ValueError(f"source plan drift: {key}")
    return plan


def predict():
    if PRED.exists() or PLAN.exists():
        raise FileExistsError("prediction exists")
    print("Recompute predictions without accessing physical outcomes.", flush=True)
    source = source_plan()
    model, _, audit1 = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(old.inputs.CONFIG.read_text())
    fingerprint = model_fingerprint(model)
    if fingerprint != source["model_sha256"] or model.ctxt_window != 1:
        raise ValueError("model drift")
    audit2 = configure_context(model, audit1)
    arrays, rows = {}, []
    with (np.load(old.inputs.PUBLIC, allow_pickle=False) as public,
          np.load(old.POOL, allow_pickle=False) as pool, torch.inference_mode()):
        actions = pool["actions"]
        np.testing.assert_array_equal(actions, old.action_bank())
        for index, entry in enumerate(source["rows"]):
            root = entry["root"]
            current = {"visual": rgb_tensor(public["current"][index, 0]),
                       "proprio": torch.as_tensor(public["proprio"][index, 0], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(public["goal"][index, 0]),
                    "proprio": torch.as_tensor(public["goal_proprio"][index, 0], dtype=torch.float32, device="cuda")[None, None]}
            item = {"root": root, "max_score_error": {}, "receipts": {}}
            for window, audit in ((1, audit1), (2, audit2)):
                model.ctxt_window = window
                boundary = StateAssistedPointMaze(model, audit)
                boundary._record("encode_goal", goal)
                target = model.encode(dict(goal))
                objective = ReprTargetDistMPCObjective(cfg, target, alpha=.1)
                predictions = {"visual": [], "proprio": []}
                scores = []
                for start in (0, 4):
                    raw = torch.as_tensor(actions[start:start+4], dtype=torch.float32)
                    packets = model.preprocessor.normalize_actions(raw).reshape(4, 5, 10).transpose(0, 1).contiguous().cuda()
                    expanded = {key: value.expand(4, *value.shape[1:]).contiguous() for key, value in current.items()}
                    pred = boundary.predict(expanded, packets)
                    scores.extend(objective(pred, packets).cpu().tolist())
                    for key in predictions:
                        predictions[key].append(pred[key][-1].cpu().numpy())
                error = float(np.max(np.abs(np.asarray(scores) - np.asarray(entry["windows"][str(window)]["cost"]))))
                if error > 1e-5:
                    raise ValueError(f"source score mismatch {root} {window}: {error}")
                for key in predictions:
                    arrays[f"r{root}_w{window}_{key}"] = np.concatenate(predictions[key])
                item["max_score_error"][str(window)] = error
                item["receipts"][str(window)] = boundary.receipts
            rows.append(item)
            print(f"root={root} predictions sealed in memory", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weight drift")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_plan_sha256": file_hash(old.PLAN), "pool_sha256": file_hash(old.POOL),
                   "public_sha256": file_hash(old.inputs.PUBLIC), "prediction_sha256": file_hash(PRED),
                   "model_sha256": fingerprint, "audit1": audit1.fingerprint, "audit2": audit2.fingerprint,
                   "rows": rows}, f, indent=2, allow_nan=False)


def render():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if FRAMES.exists() or RENDER.exists():
        raise FileExistsError("render exists")
    plan = json.loads(PLAN.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("source_plan_sha256", old.PLAN), ("pool_sha256", old.POOL),
                      ("public_sha256", old.inputs.PUBLIC), ("prediction_sha256", PRED)):
        if plan[key] != file_hash(path):
            raise ValueError(f"prediction drift: {key}")
    physical = json.loads(old.PHYSICAL.read_text())
    if physical["plan_sha256"] != file_hash(old.PLAN):
        raise ValueError("physical source drift")
    frames, rows = {}, []
    with np.load(old.inputs.PUBLIC, allow_pickle=False) as public, tempfile.TemporaryDirectory(prefix="pc_render_bank_") as temp:
        folder = Path(temp)
        xml_path = folder / "scene.xml"
        xml_path.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, entry in enumerate(physical["rows"]):
            root = entry["root"]
            trace = np.asarray(entry["traces"])
            if root != 1884 + index or trace.shape != (8, 26, 5):
                raise ValueError("physical trace shape")
            np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (8, 26)), atol=1e-10, rtol=0)
            states = np.concatenate((trace[:1, 0, :4], trace[:, -1, :4]), axis=0)
            req, res = folder / f"req_{root}.npz", folder / f"res_{root}.npz"
            with req.open("xb") as f:
                np.savez_compressed(f, states=states)
            subprocess.run([str(LEGACY / "bin/python"), str(BACKEND), str(req), str(xml_path), str(res)],
                           env=env, check=True, timeout=120)
            with np.load(res, allow_pickle=False) as data:
                images = data["frames"].copy()
            if images.shape != (9, 224, 224, 3):
                raise ValueError("invalid frames")
            diff = np.abs(images[0].astype(np.int16) - public["current"][index, 0].astype(np.int16))
            frames[f"r{root}_frames"] = images[1:]
            rows.append({"root": root, "initial_pixel_max": int(diff.max()), "initial_pixel_mae": float(diff.mean())})
            print(f"root={root} rendered; initial max pixel difference={diff.max()}", flush=True)
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, **frames)
    with RENDER.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(old.PHYSICAL),
                   "frames_sha256": file_hash(FRAMES), "backend_sha256": file_hash(BACKEND), "rows": rows},
                  f, indent=2)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, render_meta = json.loads(PLAN.read_text()), json.loads(RENDER.read_text())
    for actual, path in ((plan["protocol_sha256"], PROTOCOL), (plan["runner_sha256"], __file__),
                         (plan["prediction_sha256"], PRED), (render_meta["plan_sha256"], PLAN),
                         (render_meta["physical_sha256"], old.PHYSICAL), (render_meta["frames_sha256"], FRAMES),
                         (render_meta["backend_sha256"], BACKEND)):
        if actual != file_hash(path):
            raise ValueError("evaluation input drift")
    source = source_plan()
    physical = json.loads(old.PHYSICAL.read_text())
    model, _, audit1 = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    if model_fingerprint(model) != plan["model_sha256"] or audit1.fingerprint != plan["audit1"]:
        raise ValueError("model drift")
    if configure_context(model, audit1).fingerprint != plan["audit2"]:
        raise ValueError("context audit drift")
    cfg = yaml.safe_load(old.inputs.CONFIG.read_text())
    rows = []
    with (np.load(old.inputs.PUBLIC, allow_pickle=False) as public, np.load(PRED, allow_pickle=False) as pred,
          np.load(FRAMES, allow_pickle=False) as frames, torch.inference_mode()):
        for index, entry in enumerate(source["rows"]):
            root = entry["root"]
            real = physical["rows"][index]
            if real["root"] != root:
                raise ValueError("physical root mismatch")
            goal = {"visual": rgb_tensor(public["goal"][index, 0]),
                    "proprio": torch.as_tensor(public["goal_proprio"][index, 0], dtype=torch.float32, device="cuda")[None, None]}
            objective = ReprTargetDistMPCObjective(cfg, model.encode(dict(goal)), alpha=.1)
            actual = {"visual": [], "proprio": []}
            encoded_costs = []
            for j in range(8):
                state = np.asarray(real["traces"][j][-1][:4], dtype=np.float32)
                truth = model.encode({"visual": rgb_tensor(frames[f"r{root}_frames"][j]),
                                      "proprio": torch.as_tensor(state, device="cuda")[None, None]})
                encoded_costs.append(float(objective(truth, None).item()))
                for key in actual:
                    actual[key].append(truth[key][0, 0].cpu().numpy())
            actual = {k: np.stack(v) for k, v in actual.items()}
            distance = np.asarray(real["final_distance"])
            encoded_choice = int(np.argmin(encoded_costs))
            def rank(a, b):
                correct, count = old.agreement(a, b)
                return {"correct": correct, "pairs": count, "accuracy": correct / count if count else None}
            cases = {}
            for window in ("1", "2"):
                scores = entry["windows"][window]["cost"]
                cases[window] = {
                    "pred_vs_encoded": rank(scores, encoded_costs), "pred_vs_physical": rank(scores, distance),
                    "feature_mse": {key: float(np.mean((pred[f"r{root}_w{window}_{key}"].astype(np.float64)
                                                        - actual[key].astype(np.float64)) ** 2))
                                    for key in ("visual", "proprio")}}
            rows.append({"root": root, "initial_success": real["initial_success"], "windows": cases,
                         "encoded_vs_physical": rank(encoded_costs, distance),
                         "encoded_cost": encoded_costs, "encoded_choice": encoded_choice,
                         "encoded_regret": float(distance[encoded_choice] - distance.min())})
            print(f"root={root} actual encodings complete", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    summary = {"encoded_vs_physical": float(np.mean([r["encoded_vs_physical"]["accuracy"] for r in rows])),
               "encoded_mean_regret": float(np.mean([r["encoded_regret"] for r in rows]))}
    for window in ("1", "2"):
        summary[window] = {
            "pred_vs_encoded": float(np.mean([r["windows"][window]["pred_vs_encoded"]["accuracy"] for r in rows])),
            "pred_vs_physical": float(np.mean([r["windows"][window]["pred_vs_physical"]["accuracy"] for r in rows])),
            "visual_mse": float(np.mean([r["windows"][window]["feature_mse"]["visual"] for r in rows])),
            "proprio_mse": float(np.mean([r["windows"][window]["feature_mse"]["proprio"] for r in rows]))}
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "plan_sha256": file_hash(PLAN), "render_sha256": file_hash(RENDER),
                   "rows": rows, "summary": summary, "scope": "exposed evaluation-only successor encodings"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "render", "evaluate"):
        raise SystemExit("expected predict|render|evaluate")
    {"predict": predict, "render": render, "evaluate": evaluate}[sys.argv[1]]()
