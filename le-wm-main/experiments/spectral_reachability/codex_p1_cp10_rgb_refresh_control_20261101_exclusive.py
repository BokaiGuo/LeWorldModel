"""Fixed-root development contract and pure causal decision helpers.

Not an execution entry point: physical dispatch and durable CUDA accounting have
not yet been implemented. The contract deliberately never reads future frames
or outcome fields as selector inputs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_p1_cp10_rgb_refresh_control_20261101_exclusive"
ROLES = ("selected", "zero", "same_bank_greedy")
POLICIES = ("open_loop_shadow", "rgb_refresh")
IDENTITY = ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")


def actions(value, shape):
    x = np.asarray(value, dtype="<f4")
    if x.shape != shape or not np.isfinite(x).all() or np.any(np.abs(x) > 1):
        raise ValueError("invalid raw action shape, bounds, or finiteness")
    return x


def digest_actions(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def choose(scores):
    """Both boundaries use the same fixed tie order: hold, optimized."""
    a = np.asarray(scores, dtype=np.float64)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("two finite scores required")
    return ("hold", "optimized")[int(a[1] < a[0])]


def choose_last(policy, cp5_choice, cp10_scores):
    if policy not in POLICIES or cp5_choice not in ("hold", "optimized"):
        raise ValueError("unknown policy or prior choice")
    fresh_choice = choose(cp10_scores)
    return cp5_choice if policy == "open_loop_shadow" else fresh_choice


def cp10_context(cp5_context, prefix, committed_first, cp10_latent, mean, std):
    """Append actual cp10 RGB latent and two actual five-step action packets."""
    prefix = actions(prefix, (5, 2))
    committed_first = actions(committed_first, (5, 2))
    z = np.asarray(cp10_latent, dtype=np.float32)
    m, s = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if (z.shape != (192,) or not np.isfinite(z).all() or m.shape != (2,)
            or s.shape != (2,) or not np.isfinite(m).all() or not np.isfinite(s).all()
            or not (s > 0).all()):
        raise ValueError("invalid latent or normalizer")
    context = deepcopy(cp5_context)
    old = np.asarray(context["latent_history"], dtype=np.float32)
    if old.shape != (3, 192) or not np.isfinite(old).all():
        raise ValueError("invalid cp5 latent history")
    context["latent_history"] = np.concatenate((old[1:], z[None])).tolist()
    context["action_history"] = ((np.stack((prefix, committed_first)) - m) / s).reshape(2, 10).tolist()
    return context


def contract():
    # Revalidates pinned report, journal, frames, full source closure and action seals.
    from . import codex_p1_diag32_cp5_hold_verify_20261101_exclusive as source
    from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical
    source.verify()
    report = json.loads((source.DIRECTORY / "report.json").read_bytes())
    original = json.loads(source.CONTRACT.read_bytes())
    preseal = json.loads(source.live.PRESEAL.read_bytes())
    if tuple(r["role"] for r in report["rows"]) != ROLES:
        raise ValueError("role roster")
    candidates = {}
    for row in report["rows"]:
        role = row["role"]
        hold = actions(original["hold_tails"][role]["raw"], (2, 5, 2))
        optimized = actions(row["comparison"]["optimized_winner"]["actions"], (2, 5, 2))
        prefix = actions(preseal["prefixes"][role], (5, 2))
        point = row["observations"][5]
        candidates[role] = {
            "prefix": prefix.tolist(), "prefix_float32_sha256": digest_actions(prefix),
            "hold": hold.tolist(), "hold_float32_sha256": digest_actions(hold),
            "optimized": optimized.tolist(), "optimized_float32_sha256": digest_actions(optimized),
            "required_cp5": {key: point[key] for key in IDENTITY},
        }
    return {
        "schema": NAME + ".contract.v1",
        "source_report": str(source.DIRECTORY / "report.json"),
        "source_report_sha256": source.REPORT_SHA,
        "source_contract": str(source.CONTRACT), "source_contract_sha256": source.CONTRACT_SHA,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "roles": list(ROLES), "policies": list(POLICIES), "candidates": candidates,
        "selection": {
            "cp5": "encode actual cp5 RGB; score both complete two-packet tails with model q2; hold on tie",
            "first_packet": "commit first packet of cp5 winner, identically for both policies",
            "cp10": "encode actual cp10 RGB; score both remaining packets with one-step model MSE; hold on tie",
            "open_loop_shadow": "compute cp10 scores but dispatch second packet of cp5 winner",
            "rgb_refresh": "dispatch second packet of cp10 score winner",
            "inputs_exclude": ["physical state", "physical metrics", "reward", "prior outcomes", "future frames"],
        },
        "causal_context": "latent history root/cp5/cp10; action history actual prefix and committed first packet, normalized",
        "per_episode_neural_caps": {"actual_rgb_encode": 2, "action_encoder": 6, "predict": 6, "reverse": 0},
        "budget_condition": "complete episodes only; native early stop/censor preserves actual charges; no padding",
        "physical_caps_total": {k: v * 6 for k, v in physical.CAPS.items()},
        "wall_seconds": 1800, "cuda_allocated_cap_bytes": 4 * 1024**3,
        "cuda_reserved_cap_bytes": 6 * 1024**3,
        "cooperative_gpu_lock": str(ROOT / "results/e3_p2_host_cuda_cooperative.lock"),
        "stop": "native termination/truncation or original RGB/canvas guard, preserve censored steps",
        "development_only": True, "known_outcomes_before_design": True,
        "cached_candidates_not_general_end_to_end": True,
        "previous_optimizer_cost_inherited_not_equalized": True,
        "physical_future_used_for_selection": False,
        "independent_confirmation": False, "P1_admission": False,
        "execution_status": "blocked_until_verified_durable_same_episode_runner_exists",
    }


def seal(path):
    value = contract()
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    target = ROOT / "results" / (NAME + "_PRESEAL_CODEX01.json")
    print(json.dumps({"contract": str(target), "sha256": seal(target),
                      "execution": "not implemented; no GPU or physical calls"}))
