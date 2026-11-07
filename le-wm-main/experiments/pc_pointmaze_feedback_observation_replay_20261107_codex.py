"""Replay observations and independently rescore sealed feedback candidates."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from experiments import pc_pointmaze_packet_cem_feedback_20261107_codex as source
from experiments.pc_input_permissions import StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FEEDBACK_OBSERVATION_REPLAY_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_feedback_observation_replay_20261107_codex"
OBS = Path(str(BASE) + "_observations.npz")
META = Path(str(BASE) + "_observations.json")
RESULT = Path(str(BASE) + "_scores.json")


def check_source():
    data = json.loads(source.OUTPUT.read_text())
    for key, path in (("protocol", source.PROTOCOL), ("runner", source.__file__),
                      ("bridge", Path(source.__file__).with_name("pc_pointmaze_state_feedback_bridge_20261107_codex.py")),
                      ("source_plan", source.source.PLAN), ("source_physical", source.source.PHYSICAL)):
        if data[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    if len(data["rows"]) != 8:
        raise ValueError("wrong number of roots")
    return data


def capture():
    if OBS.exists() or META.exists():
        raise FileExistsError("sealed observations exist")
    data = check_source()
    frames, states, rows = [], [], []
    with np.load(source.source.prior.PRIVATE, allow_pickle=False) as private, np.load(source.source.prior.PUBLIC, allow_pickle=False) as public:
        for i, row in enumerate(data["rows"]):
            if row["root"] != 1884 + i:
                raise ValueError("root mismatch")
            actions = np.asarray(row["executed_actions"], dtype=np.float32)
            if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                raise ValueError("invalid actions")
            rgb_list, state_list = [], []
            with source.NativeRGBBridge(private["initial"][i]) as env:
                if not np.array_equal(env.initial_rgb, public["current"][i]):
                    raise ValueError("initial image mismatch")
                for k in range(5):
                    rgb, proprio = env.step(actions[5*k:5*(k+1)])
                    if k < 4:
                        np.testing.assert_allclose(proprio, row["decisions"][k]["observation_proprio_before"],
                                                   rtol=0, atol=1e-6)
                        rgb_list.append(rgb)
                        state_list.append(proprio)
                np.testing.assert_allclose(proprio, row["final_proprio"], rtol=0, atol=1e-6)
                if env.step_count != 25:
                    raise ValueError("clock mismatch")
            frames.append(np.stack(rgb_list))
            states.append(np.stack(state_list))
            rows.append({"root": row["root"],
                         "rgb_sha256": [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in rgb_list]})
            print(f"captured root {row['root']}", flush=True)
    with OBS.open("xb") as stream:
        np.savez_compressed(stream, rgb=np.stack(frames), proprio=np.stack(states))
    with META.open("x") as stream:
        json.dump({"protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "feedback": file_hash(source.OUTPUT), "observations": file_hash(OBS),
                   "bridge": file_hash(Path(source.__file__).with_name("pc_pointmaze_state_feedback_bridge_20261107_codex.py")),
                   "rows": rows}, stream, indent=2)


def score_comparison(old, new):
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != (3, 16) or new.shape != old.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("invalid score arrays")
    return {"max_abs_difference": float(np.max(np.abs(old-new))),
            "winner_matches": bool(np.argmin(old) == np.argmin(new)),
            "elite_set_matches": [bool(set(np.argsort(a, kind="stable")[:4]) == set(np.argsort(b, kind="stable")[:4]))
                                  for a, b in zip(old, new, strict=True)],
            "old_winner_margin": float(np.partition(old.ravel(), 1)[1]-np.min(old))}


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    meta = json.loads(META.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("feedback", source.OUTPUT),
                      ("observations", OBS),
                      ("bridge", Path(source.__file__).with_name("pc_pointmaze_state_feedback_bridge_20261107_codex.py"))):
        if meta[key] != file_hash(path):
            raise ValueError(f"observation drift: {key}")
    data = check_source()
    model, dataset, audit = load_model()
    audit = configure_context(model, audit)
    if model_fingerprint(model) != data["model"]:
        raise ValueError("model drift")
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(source.source.prior.retention.corrected.prior.prior.old.inputs.CONFIG.read_text())
    rows = []
    with np.load(OBS, allow_pickle=False) as obs, np.load(source.source.prior.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        if obs["rgb"].shape != (8, 4, 224, 224, 3) or obs["proprio"].shape != (8, 4, 4):
            raise ValueError("observation shape mismatch")
        for i, row in enumerate(data["rows"]):
            target = model.encode({"visual": rgb_tensor(public["goal"][i]),
                                   "proprio": torch.as_tensor(public["goal_proprio"][i], dtype=torch.float32, device="cuda")[None, None]})
            objective = ReprTargetDistMPCObjective(cfg, target, alpha=.1)
            decisions = []
            for k, decision in enumerate(row["decisions"]):
                observation = {"visual": rgb_tensor(obs["rgb"][i, k]),
                               "proprio": torch.as_tensor(obs["proprio"][i, k], dtype=torch.float32, device="cuda")[None, None]}
                boundary = StateAssistedPointMaze(model, audit)
                new, old = [], []
                n = decision["horizon_packets"]
                for round_ in decision["rounds"]:
                    candidates = np.asarray(round_["candidates"], dtype=np.float32)
                    if candidates.shape != (16, n, 2):
                        raise ValueError("candidate shape mismatch")
                    costs = []
                    for j in range(0, 16, 4):
                        raw = torch.as_tensor(np.repeat(candidates[j:j+4], 5, axis=1), device="cuda")
                        packed = ((raw - dataset.action_mean.cuda()) / dataset.action_std.cuda()).reshape(4, n, 10).transpose(0, 1).contiguous()
                        expanded = {key: value.expand(4, *value.shape[1:]).contiguous() for key, value in observation.items()}
                        costs.extend(objective(boundary.predict(expanded, packed), packed).cpu().tolist())
                    new.append(costs)
                    old.append(round_["costs"])
                decisions.append({"step": decision["step"], **score_comparison(old, new), "replayed_costs": new})
            rows.append({"root": row["root"], "decisions": decisions})
            print(f"rescored root {row['root']}", flush=True)
    if model_fingerprint(model) != data["model"]:
        raise ValueError("model changed")
    comparisons = [d for row in rows for d in row["decisions"]]
    with RESULT.open("x") as stream:
        json.dump({"track": "official_state_assisted", "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "observations": file_hash(OBS), "feedback": file_hash(source.OUTPUT), "rows": rows,
                   "summary": {"max_abs_difference": max(x["max_abs_difference"] for x in comparisons),
                               "winner_matches": sum(x["winner_matches"] for x in comparisons),
                               "decisions": len(comparisons),
                               "elite_set_matches": sum(sum(x["elite_set_matches"]) for x in comparisons),
                               "elite_sets": 3 * len(comparisons)},
                   "scope": "replayed images; original frames not saved; not exact input proof"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("capture", "evaluate"):
        raise SystemExit("expected capture|evaluate")
    {"capture": capture, "evaluate": evaluate}[sys.argv[1]]()
