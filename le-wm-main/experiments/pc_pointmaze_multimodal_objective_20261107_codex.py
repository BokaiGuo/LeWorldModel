"""Separate goal-state export, state-assisted GPU selection, and physical audit."""
import json
import sys

import numpy as np

from experiments import pc_camera_multiview_dev_20261106_codex as prior
from experiments.pc_input_permissions import file_hash

ROOT = prior.ROOT
PREFIX = ROOT / "results/pc_pointmaze_multimodal_objective_20261107_codex"
PUBLIC = PREFIX.with_name(PREFIX.name + "_input.npz")
SELECT = PREFIX.with_name(PREFIX.name + "_selection.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_MULTIMODAL_OBJECTIVE_DEV_PROTOCOL_20261107_CODEX.md"
CONFIG = ROOT / ("external/jepa-wms/configs/evals/simu_env_planning/mz/jepa-wm/"
                 "mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml")
ALPHA = 0.1


def prepare():
    if PUBLIC.exists():
        raise FileExistsError(PUBLIC)
    with np.load(prior.PUBLIC, allow_pickle=False) as public, np.load(prior.PRIVATE, allow_pickle=False) as private:
        current = public["current"]
        goal = public["goal"]
        proprio = public["proprio"]
        goal_proprio = private["traces"][1::2, 0, :, 0].copy()
        if current.shape[:2] != (8, 2) or goal_proprio.shape != (8, 2, 4):
            raise ValueError("unexpected panel layout")
        np.testing.assert_allclose(private["traces"][::2, 0, :, 0], proprio, atol=0, rtol=0)
        with PUBLIC.open("xb") as stream:
            np.savez_compressed(stream, current=current, goal=goal, proprio=proprio, goal_proprio=goal_proprio)
    print("Exported current and goal observations only; no action outcomes.", flush=True)


def score():
    import torch
    import yaml
    from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
    from experiments.pc_input_permissions import StateAssistedPointMaze
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
    from experiments.pc_state_track_smoke_20261106_codex import load_model

    if SELECT.exists():
        raise FileExistsError(SELECT)
    config = yaml.safe_load(CONFIG.read_text())
    objective = config["planner"]["planning_objective"]
    if objective["objective_type"] != "L2" or objective["alpha"] != ALPHA or objective["sum_all_diffs"]:
        raise ValueError("official objective configuration changed")
    previous = json.loads(prior.SELECT.read_text())
    model, dataset, permission = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_fingerprint"]:
        raise ValueError("model differs from sealed prior")
    wrapper = StateAssistedPointMaze(model, permission)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    print("Score fixed candidates with visual-only and official-weight multimodal objectives; no outcomes read.", flush=True)
    with np.load(PUBLIC, allow_pickle=False) as data, torch.inference_mode():
        for index in range(8):
            for camera in range(2):
                root = 1884 + index
                old = next(r for r in previous["rows"] if r["root"] == root and r["camera"] == camera)
                shift = goal_shift(data["goal"][index, camera])
                target = translate(data["goal"][index, camera], shift["dx"], shift["dy"])[0]
                current = translate(data["current"][index, camera], shift["dx"], shift["dy"])[0]
                inp = {"visual": rgb_tensor(current),
                       "proprio": torch.tensor(data["proprio"][index, camera], device="cuda",
                                                dtype=torch.float32)[None, None]}
                target_in = {"visual": rgb_tensor(target),
                             "proprio": torch.tensor(data["goal_proprio"][index, camera], device="cuda",
                                                      dtype=torch.float32)[None, None]}
                target_enc = model.encode(target_in)
                goal_objective = ReprTargetDistMPCObjective(config, target_enc, alpha=ALPHA)
                visual, prop, combined = [], [], []
                for x in (0., .1, -.1):
                    raw = torch.tensor([[x, 0.]] * 5, device="cuda")
                    packet = ((raw - mean) / std).reshape(1, 1, 10)
                    pred = wrapper.predict(inp, packet)
                    v = (pred["visual"][-1] - target_enc["visual"]).square().mean()
                    p = (pred["proprio"][-1] - target_enc["proprio"]).square().mean()
                    official = goal_objective(pred, packet)
                    torch.testing.assert_close(official.squeeze(), v + ALPHA * p, atol=1e-6, rtol=1e-5)
                    visual.append(float(v))
                    prop.append(float(p))
                    combined.append(float(official.squeeze()))
                np.testing.assert_allclose(visual, old["scores"][1], atol=1e-6, rtol=1e-5)
                rows.append({"root": root, "camera": camera, "visual": visual, "proprio": prop,
                             "combined": combined, "visual_choice": int(np.argmin(visual)),
                             "combined_choice": int(np.argmin(combined))})
    if fingerprint != model_fingerprint(model):
        raise ValueError("model drift")
    record = {"rows": rows, "track": "state_assisted", "goal_permission": "full goal proprio",
              "model_sha256": fingerprint, "public_sha256": file_hash(PUBLIC),
              "protocol_sha256": file_hash(PROTOCOL), "config_sha256": file_hash(CONFIG),
              "prior_selection_sha256": file_hash(prior.SELECT), "runner_sha256": file_hash(__file__),
              "permission_fingerprint": permission.fingerprint, "prediction_calls": len(wrapper.receipts),
              "scope": "exposed fixed candidates; no physical outcomes read during selection"}
    with SELECT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(f"Sealed {len(rows)} decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    selection = json.loads(SELECT.read_text())
    if (selection["public_sha256"] != file_hash(PUBLIC) or selection["protocol_sha256"] != file_hash(PROTOCOL)
            or selection["prior_selection_sha256"] != file_hash(prior.SELECT)
            or selection["runner_sha256"] != file_hash(__file__) or selection["config_sha256"] != file_hash(CONFIG)):
        raise ValueError("selection dependencies changed")
    rows = []
    with np.load(prior.PRIVATE, allow_pickle=False) as data:
        for row in selection["rows"]:
            i, camera = row["root"] - 1884, row["camera"]
            target = data["traces"][2*i+1, 0, camera, 0, :2]
            distances = np.linalg.norm(data["traces"][2*i, :, camera, -1, :2] - target, axis=-1)
            rows.append({"root": row["root"], "camera": camera, "distances": distances.tolist(),
                         "visual_regret": float(distances[row["visual_choice"]] - distances.min()),
                         "combined_regret": float(distances[row["combined_choice"]] - distances.min())})
    roots = [{"root": root,
              "visual": float(np.mean([r["visual_regret"] for r in rows if r["root"] == root])),
              "combined": float(np.mean([r["combined_regret"] for r in rows if r["root"] == root]))}
             for root in range(1884, 1892)]
    case = next(r for r in rows if r["root"] == 1887 and r["camera"] == 1)
    wins = sum(r["combined"] < r["visual"] for r in roots)
    means = {name: float(np.mean([r[name] for r in roots])) for name in ("visual", "combined")}
    output = {"rows": rows, "per_root": roots, "mean_regret": means, "improved_roots": wins,
              "case_1887_camera1": case,
              "development_exit_met": bool(len(rows) == 16 and wins >= 5 and means["combined"] < means["visual"]
                                           and case["combined_regret"] <= case["visual_regret"]),
              "selection_sha256": file_hash(SELECT), "physical_sha256": file_hash(prior.PRIVATE),
              "scope": "state-assisted, higher goal-input permission, exposed fixed-candidate development"}
    with AUDIT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("mean_regret", "improved_roots", "case_1887_camera1",
                                              "development_exit_met")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "score", "audit"):
        raise SystemExit("expected prepare|score|audit")
    {"prepare": prepare, "score": score, "audit": audit}[sys.argv[1]]()
