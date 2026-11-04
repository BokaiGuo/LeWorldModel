"""Known-root same-bank conditional-suffix development bridge, no launch CLI.

The caller supplies a frozen transition and owns device, lock, source sealing,
durable ledger, model hooks and postchecks. CPU fixtures are not model evidence.
"""
import hashlib
import json

import numpy as np

from . import p2_same_bank_native_objective_20261102_exclusive as native
from . import pusht_full_packet_compiler_20261027 as compiler

NAME = "p2_same_bank_conditional_suffix_ablation_20261102_exclusive"
DELTA_CAP = 0.5
DAMPING = 0.002


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def conditional_suffix(fixed, raw, transition, ledger, device, candidate):
    """Compile suffix from this candidate's predicted full prefix context.

    Linearize terminal embedding around its existing raw suffix, solve a
    damped L2-ball least-squares update, then project to the raw action box.
    This differs from the historical zero-logit/atanh conditional compiler.
    """
    import torch

    action = native.optimizer.raw_actions(raw)
    if type(candidate) is not int or candidate not in range(16):
        raise ValueError("candidate index")
    context = json.loads(fixed["context_json"])
    goal = json.loads(fixed["goal_json"])
    state = torch.as_tensor(context["latent_history"], dtype=torch.float32, device=device).reshape(1, 3, 192)
    history = torch.as_tensor(context["action_history"], dtype=torch.float32, device=device).reshape(1, 2, 10)
    target = np.asarray(goal, dtype=np.float64)
    if target.shape != (192,) or not np.isfinite(target).all():
        raise ValueError("target shape")
    if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(history).all()):
        raise ValueError("nonfinite context")
    prefix = torch.as_tensor(action[:5], dtype=torch.float32, device=device)
    suffix = torch.as_tensor(action[5:].reshape(10), dtype=torch.float32, device=device)
    with ledger.phase("composition_inherited"):
        ledger.record("conditional_compile_start", {"candidate": candidate,
                      "input_action_sha256": _digest(action.tobytes()),
                      "context_sha256": fixed["context_sha256"],
                      "goal_sha256": fixed["goal_sha256"]})
        with torch.no_grad():
            mid, mid_history = ledger.call("conditional_prefix_transition",
                                           lambda: transition(state, history, prefix))
        if (mid.shape != (1, 3, 192) or mid_history.shape != (1, 2, 10)
                or not bool(torch.isfinite(mid).all()) or not bool(torch.isfinite(mid_history).all())):
            raise ValueError("invalid predicted prefix context")
        mid, mid_history = mid.detach(), mid_history.detach()

        def terminal(x):
            end, _ = ledger.call("conditional_suffix_transition",
                                 lambda: transition(mid, mid_history, x.reshape(5, 2)))
            if end.shape != (1, 3, 192):
                raise ValueError("invalid terminal context")
            return end[0, -1]

        with torch.no_grad():
            nominal = terminal(suffix).detach().cpu().numpy().astype(np.float64)
        jac = ledger.call("conditional_jacobian_192_output",
                          lambda: torch.autograd.functional.jacobian(
                              terminal, suffix.detach().requires_grad_(True), vectorize=False))
        jacobian = jac.detach().cpu().numpy().astype(np.float64)
        if nominal.shape != (192,) or jacobian.shape != (192, 10) or not np.isfinite(jacobian).all():
            raise ValueError("suffix Jacobian dimensions")
        solve = ledger.call("conditional_bounded_ridge_svd_solve",
                            lambda: compiler.bounded_ridge(jacobian, target - nominal,
                                                           damping=DAMPING, cap=DELTA_CAP))
        delta = np.asarray(solve["params"], dtype=np.float64)
        if delta.shape != (10,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > DELTA_CAP + 1e-8:
            raise ValueError("invalid bounded suffix solution")
        composed = action.copy()
        composed[5:] = np.clip(action[5:].astype(np.float64) + delta.reshape(5, 2), -1, 1).astype(np.float32)
        composed = native.optimizer.raw_actions(composed)
        mid_sha = _digest(mid.cpu().numpy().astype("<f4").tobytes() +
                          mid_history.cpu().numpy().astype("<f4").tobytes())
        ledger.record("conditional_compile_return", {
            "candidate": candidate, "predicted_full_prefix_context_sha256": mid_sha,
            "source_action_sha256": _digest(action.tobytes()),
            "compiled_action_sha256": _digest(composed.tobytes()),
            "suffix_delta_norm": float(np.linalg.norm(delta)),
            "changed_float32_entries": int(np.count_nonzero(composed[5:] != action[5:])),
            "jacobian_shape": [192, 10], "requested_output_vjps": 192,
        })
    return composed


def run_bridge(fixed, transition, ledger, device):
    """Run paired initialization and common optimizer through injected transition.

    Enforces the existing real bank and shared context/goal. The ledger must
    persist before dispatch. Caller must provide all native trust boundaries.
    """
    if fixed != native.verified_input() or not callable(transition):
        raise ValueError("bound bank/context/goal or transition")
    meter = native.CappedMeter(ledger)
    evaluate = native.make_objective(fixed, transition, meter, device)
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    rows = []
    for index, raw in enumerate(fixed["bank"]):
        candidate = native.optimizer.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
        compiled = conditional_suffix(fixed, candidate, transition, ledger, device, index)
        initial = {native.optimizer.ARMS[0]: compiled, native.optimizer.ARMS[1]: candidate}

        def callback(arm, action, gradient):
            return evaluate(arm, context, goal, action, gradient)

        def emit(event):
            ledger.record("optimizer_" + event["kind"], {"candidate_index": index, **event})

        pair = native.optimizer.optimize_pair(initial, callback, emit)
        rows.append({"candidate_index": index, "source_candidate_sha256": _digest(raw),
                     "composition_initial_sha256": _digest(compiled.tobytes()),
                     "spectral_only_initial_sha256": _digest(candidate.tobytes()),
                     "composition_initial_hex": compiled.tobytes().hex(),
                     "arms": pair["rows"]})
    expected = {"score": 144, "gradient": 128}
    if any(evaluate.attempted[arm] != expected for arm in native.optimizer.ARMS):
        raise ValueError("common optimizer callback census")
    return {"schema": NAME + ".development_v1", "rows": rows,
            "context_sha256": fixed["context_sha256"], "goal_sha256": fixed["goal_sha256"],
            "handoff_sha256": fixed["handoff_sha256"], "pair_sha256": fixed["pair_sha256"],
            "common_callback_attempts": evaluate.attempted,
            "common_instrumented_attempts": meter.counts,
            "composition_compiles": 16, "production_authorized": False,
            "independent_confirmation": False, "equal_total_compute": False,
            "P1_admission": False, "P2_admission": False}
