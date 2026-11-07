"""Exploratory aligned encoding and physical analysis of two sealed CEM choices."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from experiments import pc_pointmaze_goal25_packet_cem_20261107_codex as source
from experiments.pc_input_permissions import STATE, file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PACKET_CEM_SCORE_DECOMPOSITION_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_packet_cem_score_decomposition_20261107_codex"
FRAMES = Path(str(BASE) + "_frames.npz")
RENDER = Path(str(BASE) + "_render.json")
RESULT = Path(str(BASE) + "_result.json")


def relation(a, b, eps=1e-10):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("need two finite values")
    da, db = a[0] - a[1], b[0] - b[1]
    return None if abs(da) <= eps or abs(db) <= eps else bool(da * db > 0)


def render():
    if FRAMES.exists() or RENDER.exists():
        raise FileExistsError("render exists")
    plan, physical = source.check_plan(), json.loads(source.PHYSICAL.read_text())
    if physical["plan"] != file_hash(source.PLAN) or physical["witness"] != file_hash(source.prior.PRIVATE):
        raise ValueError("source drift")
    states, rows = [], []
    with np.load(source.prior.PRIVATE, allow_pickle=False) as witness, np.load(source.prior.PUBLIC, allow_pickle=False) as public:
        for index, row in enumerate(physical["rows"]):
            if row["root"] != 1884 + index or plan["rows"][index]["root"] != row["root"]:
                raise ValueError("root mismatch")
            traces = np.asarray(row["trace"])
            if traces.shape != (3, 26, 5):
                raise ValueError("trace shape")
            states.extend((traces[0, -1, :4], traces[1, -1, :4], witness["goal"][index]))
        frames = source.prior.retention.corrected.render_states(np.asarray(states)).reshape(8, 3, 224, 224, 3)
        for i in range(8):
            diff = np.abs(frames[i, 2].astype(np.int16) - public["goal"][i].astype(np.int16))
            rows.append({"root": 1884 + i, "goal_pixel_mae": float(diff.mean()), "goal_pixel_max": int(diff.max())})
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, frames=frames)
    with RENDER.open("x") as f:
        json.dump({"protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "source_plan": file_hash(source.PLAN), "source_physical": file_hash(source.PHYSICAL),
                   "frames": file_hash(FRAMES), "worker": file_hash(source.prior.retention.corrected.WORKER),
                   "rows": rows}, f, indent=2)
    print("Rendered sealed end states and recorded archive goal pixel differences.", flush=True)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    meta = json.loads(RENDER.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("source_plan", source.PLAN),
                      ("source_physical", source.PHYSICAL), ("frames", FRAMES),
                      ("worker", source.prior.retention.corrected.WORKER)):
        if meta[key] != file_hash(path):
            raise ValueError(f"render drift: {key}")
    plan, physical = source.check_plan(), json.loads(source.PHYSICAL.read_text())
    model, _, audit = load_model()
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model drift")
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(source.prior.retention.corrected.prior.prior.old.inputs.CONFIG.read_text())
    rows = []
    with (np.load(FRAMES, allow_pickle=False) as frames, np.load(source.prior.PUBLIC, allow_pickle=False) as public,
          torch.inference_mode()):
        for i, p in enumerate(plan["rows"]):
            root = 1884 + i
            trace = np.asarray(physical["rows"][i]["trace"])
            if p["root"] != root or physical["rows"][i]["root"] != root or trace.shape != (3, 26, 5):
                raise ValueError("root or clock mismatch")
            goal = {"visual": rgb_tensor(public["goal"][i]),
                    "proprio": torch.as_tensor(public["goal_proprio"][i], dtype=torch.float32, device="cuda")[None, None]}
            objective = ReprTargetDistMPCObjective(cfg, model.encode(dict(goal)), alpha=.1)
            encoded_cost = []
            for k in range(2):
                state = torch.as_tensor(trace[k, -1, :4], dtype=torch.float32, device="cuda")[None, None]
                encoded = model.encode({"visual": rgb_tensor(frames["frames"][i, k]), "proprio": state})
                encoded_cost.append(float(objective(encoded, None).item()))
            pred_cost = [p["selected"][x]["cost"] for x in ("48", "160")]
            distance = np.asarray(physical["rows"][i]["distance"])[:2, -1].tolist()
            identical = p["chosen"]["48"] == p["chosen"]["160"]
            rows.append({"root": root, "initial_success": physical["rows"][i]["initial_success"],
                         "same_chain": identical, "predicted_cost": pred_cost, "encoded_cost": encoded_cost,
                         "physical_distance": distance, "predicted_vs_encoded": relation(pred_cost, encoded_cost),
                         "encoded_vs_physical": relation(encoded_cost, distance),
                         "predicted_vs_physical": relation(pred_cost, distance),
                         "goal_pixel_mae": meta["rows"][i]["goal_pixel_mae"],
                         "goal_pixel_max": meta["rows"][i]["goal_pixel_max"]})
            print(f"root={root} predicted={pred_cost} encoded={encoded_cost} distance={distance}", flush=True)
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model changed")
    comparisons = {}
    for field in ("predicted_vs_encoded", "encoded_vs_physical", "predicted_vs_physical"):
        values = [r[field] for r in rows if r[field] is not None]
        comparisons[field] = {"agree": sum(values), "non_ties": len(values)}
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "render": file_hash(RENDER), "plan": file_hash(source.PLAN),
                   "rows": rows, "comparisons": comparisons,
                   "scope": "post hoc exposed state-assisted two-choice diagnostic"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("render", "evaluate"):
        raise SystemExit("expected render|evaluate")
    {"render": render, "evaluate": evaluate}[sys.argv[1]]()
