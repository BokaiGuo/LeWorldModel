"""Frozen 52-arm replay protocol and pure contract helpers; no simulator or model execution."""
import json
import math
from pathlib import Path

import numpy as np

from . import reacher_p1_52_replay_roster_20261102_codexexclusive as roster
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "results/reacher_p1_52_replay_roster_20261102_codexexclusive_01.json"
ROSTER_SHA = "31b26ef23f90b2a85161b5ffee4ba6170265d54ee84fd52c4ec6899cc992cf48"
AUTHORITY = "P1_52_FIXED_ACTION_REPLAY_DEVELOPMENT"


def sha(path):
    return host.bridge.sha_file(path)


def prepare():
    host.require(sha(ROSTER) == ROSTER_SHA, "roster changed")
    entries = json.loads(ROSTER.read_text())
    regenerated = json.loads(json.dumps(roster.make_roster(), allow_nan=False))
    host.require(entries == regenerated, "roster/source authentication failed")
    source = json.loads((ROOT / "results/reacher_response_svd_cp5_PLAN_20261102_v1.json").read_text())
    q_source = json.loads((ROOT / "results/reacher_response_svd_qcp5_PLAN_20261102_v1.json").read_text())
    host.require(source["weights"] == q_source["weights"] and source["stats"] == q_source["stats"]
                 and source["model_parameter_hash"] == q_source["model_parameter_hash"], "model mismatch")
    files = {**source["files"], **q_source["files"]}
    own = Path(__file__).parent
    for name in ("reacher_p1_52_replay_contract_20261102_codexexclusive.py",
                 "reacher_p1_52_replay_host_20261102_codexexclusive.py",
                 "reacher_p1_52_replay_consumer_20261102_codexexclusive.py",
                 "reacher_p1_52_replay_roster_20261102_codexexclusive.py"):
        path = own / name
        files[str(path)] = sha(path)
    files[str(ROSTER)] = ROSTER_SHA
    for path, expected in files.items():
        host.require(sha(path) == expected, f"source drift: {path}")
    return {"schema": "p1_52_replay_contract_v1", "roster": str(ROSTER),
            "roster_sha256": ROSTER_SHA, "files": files, "weights": source["weights"],
            "stats": source["stats"], "model_parameter_hash": source["model_parameter_hash"],
            "roots": source["roots"], "authority": AUTHORITY, "gpu_uuid": host.UUID,
            "lock": str(host.LOCK), "arms": 52, "distance_tolerance": 1e-8,
            "prediction_horizon_blocks": 2, "action_block": 5,
            "metric": "unweighted L2 in role-specific planning coordinates",
            "physical_outcome_usage": "replay consistency only, never action selection",
            "source_closure_complete": False, "independent_outcome": False,
            "equal_total_compute": False, "p1_admission": False}


def bind_plan(path, expected):
    host.require(sha(path) == expected, "plan hash")
    saved = json.loads(Path(path).read_text())
    host.require(saved == prepare(), "plan or source drift")
    return saved


def normalized_suffix(actions, mean, std, low, high):
    actions = np.asarray(actions, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    host.require(actions.shape == (15, 2) and mean.shape == std.shape == (10,)
                 and np.isfinite(actions).all() and np.isfinite(mean).all()
                 and np.isfinite(std).all() and np.all(std >= 1e-6), "action/stats shape")
    normalized = (actions[5:].reshape(2, 10) - mean) / std
    host.require(np.isfinite(normalized).all()
                 and np.all(normalized >= np.asarray(low))
                 and np.all(normalized <= np.asarray(high)), "suffix outside chart")
    reconstructed = (normalized * std + mean).reshape(10, 2)
    host.require(np.allclose(reconstructed, actions[5:], rtol=0, atol=1e-6), "suffix roundtrip")
    return normalized


def metrics(predicted, actual, goal, physical):
    values = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (predicted, actual, goal)]
    host.require(values[0].size > 0 and all(x.shape == values[0].shape and np.isfinite(x).all()
                                            for x in values), "latent shape/finite")
    host.require(isinstance(physical, (int, float)) and math.isfinite(physical) and physical >= 0,
                 "physical distance")
    p, a, g = values
    return {"predicted_goal_distance": float(np.linalg.norm(p-g)),
            "actual_rgb_goal_distance": float(np.linalg.norm(a-g)),
            "prediction_error": float(np.linalg.norm(p-a)), "physical_distance": float(physical)}
