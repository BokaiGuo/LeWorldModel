"""Post-hoc packet-aligned teacher forcing on sealed state-assisted CEM chains."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments import pc_pointmaze_goal25_packet_cem_20261107_codex as source
from experiments.pc_input_permissions import file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PACKET_TEACHER_FORCING_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_packet_teacher_forcing_20261107_codex"
FRAMES = Path(str(BASE) + "_frames.npz")
META = Path(str(BASE) + "_render.json")
RESULT = Path(str(BASE) + "_result.json")
STEPS = np.arange(0, 26, 5)


def check_trace(trace):
    trace = np.asarray(trace)
    if trace.shape != (3, 26, 5) or not np.isfinite(trace).all():
        raise ValueError("invalid physical trace")
    np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (3, 26)),
                               rtol=0, atol=1e-10)
    return trace


def render():
    if FRAMES.exists() or META.exists():
        raise FileExistsError("sealed render exists")
    plan = source.check_plan()
    physical = json.loads(source.PHYSICAL.read_text())
    if physical["plan"] != file_hash(source.PLAN) or physical["witness"] != file_hash(source.prior.PRIVATE):
        raise ValueError("source drift")
    states = []
    for i, row in enumerate(physical["rows"]):
        if row["root"] != 1884 + i or plan["rows"][i]["root"] != row["root"]:
            raise ValueError("root mismatch")
        states.extend(check_trace(row["trace"])[:2, STEPS, :4].reshape(-1, 4))
    frames = source.prior.retention.corrected.render_states(np.asarray(states)).reshape(8, 2, 6, 224, 224, 3)
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, frames=frames)
    with META.open("x") as stream:
        json.dump({"protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "plan": file_hash(source.PLAN), "physical": file_hash(source.PHYSICAL),
                   "frames": file_hash(FRAMES), "worker": file_hash(source.prior.retention.corrected.WORKER)},
                  stream, indent=2)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    meta = json.loads(META.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("plan", source.PLAN),
                      ("physical", source.PHYSICAL), ("frames", FRAMES),
                      ("worker", source.prior.retention.corrected.WORKER)):
        if meta[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    plan, physical = source.check_plan(), json.loads(source.PHYSICAL.read_text())
    model, dataset, audit = load_model()
    configure_context(model, audit)
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("model drift")
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    import yaml
    cfg = yaml.safe_load(source.prior.retention.corrected.prior.prior.old.inputs.CONFIG.read_text())
    rows = []
    with np.load(FRAMES, allow_pickle=False) as rendered, np.load(source.prior.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, row in enumerate(plan["rows"]):
            root = 1884 + i
            if row["root"] != root or physical["rows"][i]["root"] != root:
                raise ValueError("root mismatch")
            trace = check_trace(physical["rows"][i]["trace"])
            goal = {"visual": rgb_tensor(public["goal"][i]),
                    "proprio": torch.as_tensor(public["goal_proprio"][i], device="cuda")[None, None]}
            objective = ReprTargetDistMPCObjective(cfg, model.encode(goal), alpha=.1)
            for j, limit in enumerate(("48", "160")):
                actions = torch.as_tensor(row["selected"][limit]["actions"], dtype=torch.float32, device="cuda")
                if actions.shape != (25, 2) or not torch.isfinite(actions).all() or actions.abs().max() > 1:
                    raise ValueError("invalid action chain")
                packets = ((actions - dataset.action_mean.cuda()) / dataset.action_std.cuda()).reshape(5, 1, 10)
                images = rendered["frames"][i, j]
                observations = {"visual": torch.stack([rgb_tensor(x)[0, 0] for x in images])[None],
                                "proprio": torch.as_tensor(trace[j, STEPS, :4], dtype=torch.float32, device="cuda")[None]}
                actual = model.encode(observations)
                initial = {"visual": rgb_tensor(public["current"][i]),
                           "proprio": torch.as_tensor(public["proprio"][i], dtype=torch.float32, device="cuda")[None, None]}
                model.ctxt_window = 2
                automatic = model.unroll(model.encode(initial), packets)
                np.testing.assert_allclose(float(objective(automatic, packets).item()),
                                           row["selected"][limit]["cost"], rtol=2e-4, atol=2e-6)
                measures = []
                for t in range(5):
                    current = actual[:, t:t+1]
                    model.ctxt_window = 2
                    single = model.unroll(current, packets[t:t+1])
                    if t == 0:
                        history = single
                    else:
                        history = unroll_history(model.model, actual["visual"][:, t-1:t+1],
                                                 actual["proprio"][:, t-1:t+1],
                                                 packets[t-1:t].transpose(0, 1), packets[t:t+1], 2)
                    actual_next = actual[:, t+1:t+2]
                    methods = {}
                    for name, pred in (("autoregressive", automatic), ("single_observation", single),
                                       ("observed_history", history)):
                        pos = t + 1 if name == "autoregressive" else -1
                        visual = pred["visual"][pos:pos+1] if pos >= 0 else pred["visual"][-1:]
                        proprio = pred["proprio"][pos:pos+1] if pos >= 0 else pred["proprio"][-1:]
                        predicted_step = automatic[:1].clone()
                        predicted_step["visual"], predicted_step["proprio"] = visual, proprio
                        methods[name] = {"visual_mse": float((visual - actual_next["visual"]).square().mean()),
                                         "proprio_mse": float((proprio - actual_next["proprio"]).square().mean()),
                                         "goal_cost": float(objective(predicted_step, None).item())}
                    actual_cost = float(objective(actual_next, None).item())
                    measures.append({"environment_step": int(STEPS[t+1]), "actual_goal_cost": actual_cost,
                                     "methods": methods})
                difference = np.abs(images[0].astype(np.int16) - public["current"][i].astype(np.int16))
                rows.append({"root": root, "budget": int(limit), "same_chain": row["chosen"]["48"] == row["chosen"]["160"],
                             "initial_pixel_mae": float(difference.mean()), "initial_pixel_max": int(difference.max()),
                             "steps": measures})
                print(f"root={root} budget={limit} completed", flush=True)
    if model_fingerprint(model) != plan["model"]:
        raise ValueError("weights changed")
    with RESULT.open("x") as stream:
        json.dump({"track": "official_state_assisted", "render": file_hash(META), "rows": rows,
                   "scope": "post-hoc exposed roots; no feedback into planning"}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("render", "evaluate"):
        raise SystemExit("expected render|evaluate")
    {"render": render, "evaluate": evaluate}[sys.argv[1]]()
