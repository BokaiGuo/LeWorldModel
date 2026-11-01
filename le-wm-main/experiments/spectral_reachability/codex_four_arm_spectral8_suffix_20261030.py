"""Paid spectral8 H3 initializer plus common raw suffix; no model loader or launcher.

Only explicit adapter calls are metered; no complete total-cost or fairness claim.
The caller must durably persist the supplied events or raise before proceeding.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as raw
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as suffix
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay

RADII = (float(np.sqrt(5) * .25), float(np.sqrt(5) * .75))
LABEL = "spectral8_h3_extension_not_original_R_or_fairness"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy(mean, std, history_size=3):
    return {"schema": replay.POLICY_SCHEMA, "candidate_ids": [f"spectral_{i}" for i in range(8)],
            "horizon": 3, "history_size": history_size, "metric": "terminal_latent_mean_squared_error",
            "tie": "first_declared_candidate", "replay_batch": 1,
            "action_chart": "raw_packet_5x2_box_minus1_to1", "mean": mean, "std": std}


def required_caps(config):
    spec.require(type(config) is dict and set(config) == {"steps", "learning_rate", "history_size"}
                 and type(config["steps"]) is int and 0 < config["steps"] <= 100
                 and type(config["history_size"]) is int and config["history_size"] > 0
                 and type(config["learning_rate"]) in (float, int)
                 and math.isfinite(config["learning_rate"]) and config["learning_rate"] > 0,
                 "suffix config")
    out = dict.fromkeys(spec.AXES, 0)
    calls = 3 + 8 * config["steps"] * 3 + 8 * 3
    out.update(action_encode_calls=calls, predict_calls=calls, predict_samples=calls,
               backward_calls=192+8*config["steps"], vjp_rows=192+8*config["steps"],
               jacobian_calls=1, svd_calls=1, replay_calls=8, selection_evaluations=8)
    return out


def spectral_bank(jac, projection):
    """Numerical kernel; production caller must invoke only inside SVD dispatch."""
    spec.require(type(jac) is np.ndarray and jac.shape == (192, 30) and np.isfinite(jac).all()
                 and type(projection) is np.ndarray and projection.shape == (192, 192)
                 and np.isfinite(projection).all(), "finite Jacobian/projection")
    projected = projection.astype(np.float64) @ jac.astype(np.float64)
    spec.require(np.isfinite(projected).all(), "nonfinite projected Jacobian")
    _, singular, right = np.linalg.svd(projected, full_matrices=False)
    spec.require(np.isfinite(singular).all() and np.isfinite(right).all(), "nonfinite SVD")
    # Refuse unstable mode identity rather than silently choosing a basis.
    spec.require(singular[1] > 0 and singular[0]-singular[1] > 1e-10*singular[0]
                 and singular[1]-singular[2] > 1e-10*singular[0], "degenerate leading modes")
    for row in right:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    params = np.stack([sign * radius * right[mode]
                       for radius in RADII for mode in range(2) for sign in (-1, 1)])
    return np.tanh(params).astype(np.float32).reshape(8, 3, 5, 2), singular


def run(leaf, states, history, goal, mean, std, projection, config, replay_policy, contract, *, emit):
    """Run one structured arm; return eight initialized/optimized candidates and selection."""
    spec.require(type(leaf) is leafmod.LeafAdapter, "LeafAdapter required")
    try:
        spec.require(callable(emit) and spec.digest(contract) == leaf.contract_sha
                     and spec.digest(config) == leaf.config_sha
                     and contract["identity"]["optimizer"] == file_sha(suffix.__file__)
                     and contract["arms"]["structured"]["initializer_sha256"] == file_sha(__file__)
                     and spec.digest(replay_policy) == contract["phase_policy_sha256"],
                     "contract/source binding")
        with leaf.driver.path.open("rb") as stream:
            header = json.loads(stream.readline(1024*1024))["payload"]
        spec.require(header["arm"] == "structured", "structured ledger required")
        needed = required_caps(config)
        spec.require(all(leaf.driver.charged[k] + needed[k] <= leaf.driver.caps[k] for k in spec.AXES),
                     "full path reserve unavailable")
        for t in (states, history, goal, mean, std, projection):
            leaf._tensor(t)
            spec.require(t.is_floating_point() and leafmod._finite(t)
                         and t.dtype == states.dtype and t.device == states.device, "input tensor")
        h = config["history_size"]
        expected_policy = policy(mean.detach().cpu().tolist(), std.detach().cpu().tolist(), h)
        spec.require(states.dtype == torch.float32 and replay_policy == expected_policy and states.shape == (1,h,192)
                     and history.shape == (1,h-1,10) and goal.shape == (1,192)
                     and mean.shape == std.shape == (2,) and projection.shape == (192,192)
                     and torch.equal(projection, torch.eye(192, device=states.device, dtype=states.dtype))
                     and bool((std > 0).all().item()), "input/policy dimensions")
        leaf._dispatch("jacobian", "jacobian", {"jacobian_calls": 1}, lambda: None)
        nominal_raw = torch.zeros((1,3,5,2), device=states.device, dtype=states.dtype, requires_grad=True)
        prediction = raw._normal(leaf, "jacobian", states.detach(), history.detach(), nominal_raw, mean, std)
        rows = []
        for i in range(192):
            grad = leaf._dispatch("jacobian", "backward", {"backward_calls": 1, "vjp_rows": 1},
                lambda i=i: torch.autograd.grad(prediction[0,i], nominal_raw, retain_graph=i < 191)[0])
            spec.require(grad.shape == nominal_raw.shape and leafmod._finite(grad), "invalid Jacobian row")
            rows.append(grad.detach().reshape(-1))
        jac = torch.stack(rows).cpu().numpy()
        q = projection.detach().cpu().numpy()
        emit("spectral_signal", {"nominal": prediction.detach().cpu().tolist(),
                                 "jacobian": jac.tolist(), "projection": q.tolist()})
        bank, singular = leaf._dispatch("initialization", "svd", {"svd_calls": 1},
                                        lambda: spectral_bank(jac, q))
        emit("spectral_bank", {"raw": bank.tolist(), "singular_values": singular.tolist(),
                               "radii": list(RADII), "order": "radius,mode,sign(-1,+1)", "label": LABEL})
        initial = torch.from_numpy(bank).to(device=states.device, dtype=states.dtype)
        optimized = torch.cat([suffix.optimize(leaf, states, history, initial[i:i+1], goal, mean, std, config)
                               for i in range(8)], dim=0)
        emit("optimized_candidates", {"raw": optimized.detach().cpu().tolist()})
        final = replay.replay_and_select(leaf, states, history, optimized, goal, mean, std, replay_policy, contract)
        emit("final_selection", {"scores": final["scores"], "index": final["selected_index"],
                                 "id": final["selected_id"]})
        return {"initial_raw": initial.detach(), "optimized_raw": optimized.detach(),
                "scores": final["scores"], "selected_index": final["selected_index"],
                "selected_id": final["selected_id"], "label": LABEL, "fairness_proven": False,
                "physical_execution_authorized": False}
    except BaseException:
        leaf._abort()
        raise
