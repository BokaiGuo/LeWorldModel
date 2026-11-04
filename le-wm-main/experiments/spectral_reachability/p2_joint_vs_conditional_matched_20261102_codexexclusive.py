"""Matched-operation conditional vs joint initializer wired to real P2 optimizer.

The caller owns CUDA/model provenance, locking and the durable ledger. This
module cannot authorize or execute a host session by itself.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import threading

import numpy as np

from . import p2_same_bank_native_objective_20261102_exclusive as native
from . import p2_sra_common_optimizer_bridge_20261102_exclusive as optimizer
from . import p2_same_bank_conditional_suffix_ablation_20261102_exclusive as conditional
from . import p2_conditional_hook_meter_20261102_exclusive as old_meter

ARMS = optimizer.ARMS
HOOKS = old_meter.HOOKS
# sra_initialization is a historical ledger category. Here it explicitly
# denotes the joint Gauss-Newton control; it is NOT historical SRA.
INIT_CATEGORY = {ARMS[0]: "composition_inherited", ARMS[1]: "sra_initialization"}
INIT_CAPS = {"init_transition": 32, "init_jacobian": 16, "init_bounded_solve": 16,
             **{axis: 32 for axis in HOOKS}}
COMMON = {"transition_packet": 288, "autograd_grad": 128, **{axis: 288 for axis in HOOKS}}
CAPS = {**{c: dict(INIT_CAPS) for c in INIT_CATEGORY.values()},
        **{"common_" + arm: dict(COMMON) for arm in ARMS}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


class LedgerView:
    def __init__(self, meter):
        self.meter = meter

    @property
    def category(self):
        return self.meter.raw.category

    @contextmanager
    def phase(self, name):
        self.meter.check_thread()
        try:
            with self.meter.raw.phase(name):
                yield
        except BaseException:
            self.meter.failed = True
            raise

    def record(self, kind, payload):
        self.meter.check_thread()
        return self.meter.raw.record(kind, payload)

    def call(self, axis, operation):
        return self.meter.dispatch(axis, operation, hook=False)


class MatchedMeter:
    """Fail-stop, pre-dispatch attempts across both initializer and optimizer."""
    def __init__(self, raw):
        require(all(callable(getattr(raw, x, None)) for x in ("record", "call", "phase"))
                and raw.category is None, "new durable ledger required")
        self.raw, self.owner = raw, threading.get_ident()
        self.failed = self.finished = False
        self.attempted = {key: dict.fromkeys(axes, 0) for key, axes in CAPS.items()}
        self.returned = deepcopy(self.attempted)
        self.ledger = LedgerView(self)
        self.raw.record("matched_meter_open", {"caps": CAPS,
            "sra_initialization_role": "joint_gauss_newton_not_historical_sra"})

    def check_thread(self):
        if threading.get_ident() != self.owner:
            self.failed = True
            raise RuntimeError("single owner thread required")

    def phase(self, category):
        return self.ledger.phase(category)

    def dispatch(self, axis, operation, *, hook):
        self.check_thread()
        category = self.raw.category
        allowed = CAPS.get(category, {})
        if (self.failed or self.finished or axis not in allowed or not callable(operation)
                or (axis in HOOKS) != hook):
            self.failed = True
            self.raw.record("matched_meter_reject", {"category": category, "axis": str(axis)})
            raise ValueError("invalid or failed meter route")
        if self.attempted[category][axis] >= allowed[axis]:
            self.failed = True
            self.raw.record("matched_meter_reject", {"category": category, "axis": axis, "reason": "cap"})
            raise ValueError("attempt cap")
        self.attempted[category][axis] += 1
        try:
            value = self.raw.call(axis, operation)
            if self.failed:
                raise RuntimeError("nested metered operation failed")
        except BaseException:
            self.failed = True
            raise
        self.returned[category][axis] += 1
        return value

    def call(self, axis, operation):
        return self.dispatch(axis, operation, hook=axis in HOOKS)

    def finish(self):
        self.check_thread()
        if (self.failed or self.finished or self.raw.category is not None or
                self.attempted != CAPS or self.returned != CAPS):
            self.failed = True
            self.raw.record("matched_meter_incomplete", {"attempted": self.attempted, "returned": self.returned})
            raise ValueError("incomplete matched schedule")
        self.finished = True
        self.raw.record("matched_meter_complete", {"attempted": self.attempted, "returned": self.returned})
        return {"attempted": deepcopy(self.attempted), "returned": deepcopy(self.returned),
                "equal_total_compute": False}


def optimize_schedule(bank, initialize, evaluate, emit):
    """Shared real optimizer, same bank and arm order; callbacks retain attempts."""
    require(callable(initialize) and callable(evaluate) and callable(emit), "callbacks required")
    rows = []
    for index, raw in enumerate(bank):
        require(isinstance(raw, bytes) and len(raw) == 80, "bank bytes")
        action = optimizer.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
        initial = {arm: optimizer.raw_actions(initialize(arm, action.copy(), index)) for arm in ARMS}
        result = optimizer.optimize_pair(initial, evaluate,
                 lambda event: emit({"candidate_index": index, **event}))
        rows.append({"candidate_index": index, "source_sha256": hashlib.sha256(raw).hexdigest(),
                     "initializer_float32_hex": {arm: initial[arm].tobytes().hex() for arm in ARMS},
                     "initializer_sha256": {arm: hashlib.sha256(initial[arm].tobytes()).hexdigest()
                                            for arm in ARMS}, "arms": result["rows"]})
    return rows


def initialize_native(fixed, raw, transition, meter, device, arm, index):
    """Both routes consume two forward transitions and one 192-output Jacobian."""
    import torch
    require(torch.device(device).type == "cuda", "CUDA required; CPU neural inference forbidden")
    return _initialize_with_transition(fixed, raw, transition, meter, device, arm, index)


def _initialize_with_transition(fixed, raw, transition, meter, device, arm, index):
    """Private mathematical seam; CPU tests supply only non-neural analytic functions."""
    import torch
    from . import pusht_full_packet_compiler_20261027 as compiler
    require(arm in ARMS and type(index) is int and 0 <= index < 16, "initializer route")
    action = optimizer.raw_actions(raw)
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    state = torch.as_tensor(context["latent_history"], dtype=torch.float32, device=device).reshape(1, 3, 192)
    history = torch.as_tensor(context["action_history"], dtype=torch.float32, device=device).reshape(1, 2, 10)
    target = np.asarray(goal, dtype=np.float64)
    require(target.shape == (192,) and np.isfinite(target).all()
            and bool(torch.isfinite(state).all()) and bool(torch.isfinite(history).all()), "invalid context")
    action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
    view = meter.ledger
    with view.phase(INIT_CATEGORY[arm]):
        view.record("init_start", {"arm": arm, "candidate": index, "action_sha256": hashlib.sha256(
            action.tobytes()).hexdigest(), "role": "conditional_suffix" if arm == ARMS[0] else "joint_no_composition"})
        if arm == ARMS[0]:
            with torch.no_grad():
                mid, middle_history = view.call("init_transition",
                    lambda: transition(state, history, action_tensor[:5]))
            mid, middle_history = mid.detach(), middle_history.detach()
            x0 = action_tensor[5:].reshape(10)
        else:
            x0 = action_tensor.reshape(20)
        nominal = []
        def terminal(x):
            if arm == ARMS[0]:
                end, _ = view.call("init_transition",
                                   lambda: transition(mid, middle_history, x.reshape(5, 2)))
            else:
                a = x.reshape(10, 2)
                first, past = view.call("init_transition", lambda: transition(state, history, a[:5]))
                end, _ = view.call("init_transition", lambda: transition(first, past, a[5:]))
            require(end.shape == (1, 3, 192), "invalid terminal shape")
            value = end[0, -1]
            nominal.append(value.detach().cpu().numpy().astype(np.float64))
            return value
        jac = view.call("init_jacobian", lambda: torch.autograd.functional.jacobian(
            terminal, x0.detach().requires_grad_(True), vectorize=False))
        matrix = jac.detach().cpu().numpy().astype(np.float64)
        width = 10 if arm == ARMS[0] else 20
        require(len(nominal) == 1 and matrix.shape == (192, width)
                and np.isfinite(matrix).all() and np.isfinite(nominal[0]).all(), "Jacobian output")
        solution = view.call("init_bounded_solve", lambda: compiler.bounded_ridge(
            matrix, target - nominal[0], damping=conditional.DAMPING, cap=conditional.DELTA_CAP))
        delta = np.asarray(solution["params"], dtype=np.float64)
        require(delta.shape == (width,) and np.isfinite(delta).all()
                and np.linalg.norm(delta) <= conditional.DELTA_CAP + 1e-8, "bounded solution")
        changed = action.copy()
        slot = slice(5, 10) if arm == ARMS[0] else slice(0, 10)
        changed[slot] = np.clip(changed[slot].astype(np.float64) + delta.reshape(-1, 2), -1, 1).astype(np.float32)
        changed = optimizer.raw_actions(changed)
        view.record("init_returned", {"arm": arm, "candidate": index, "jacobian_shape": [192, width],
                    "requested_output_vjps": 192, "changed_entries": int(np.count_nonzero(changed != action)),
                    "output_sha256": hashlib.sha256(changed.tobytes()).hexdigest()})
        return changed


def run_native(fixed, transition, meter, device):
    """Actual planner wiring; outer caller must install hooks and authenticate CUDA."""
    import torch
    require(torch.device(device).type == "cuda", "CUDA required; CPU neural inference forbidden")
    require(fixed == native.verified_input() and callable(transition) and isinstance(meter, MatchedMeter),
            "source, transition or meter")
    require(len(fixed["bank"]) == 16, "fixed bank census")
    evaluate = native.make_objective(fixed, transition, meter, device)
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    rows = optimize_schedule(
        fixed["bank"], lambda arm, raw, index: initialize_native(fixed, raw, transition, meter, device, arm, index),
        lambda arm, raw, grad: evaluate(arm, context, goal, raw, grad),
        lambda event: meter.ledger.record("optimizer_" + event["kind"], event))
    require(all(v == {"score": 144, "gradient": 128} for v in evaluate.attempted.values()),
            "common optimizer census")
    winners = {}
    for arm in ARMS:
        options = [(entry["model_score"], row["candidate_index"], entry["winner_iteration"], entry)
                   for row in rows for entry in row["arms"] if entry["arm"] == arm]
        require(len(options) == 16, "winner candidate census")
        score, index, iteration, entry = min(options, key=lambda x: (x[0], x[1], x[2]))
        winners[arm] = {"candidate_index": index, "iteration": iteration, "model_score": score,
                        "raw_float32_hex": entry["raw_float32_hex"]}
    return {"schema": "p2_conditional_vs_joint_matched_development_v1", "rows": rows,
            "model_selected_winners": winners,
            "context_sha256": fixed["context_sha256"], "goal_sha256": fixed["goal_sha256"],
            "legacy_control_label": ARMS[1], "control_method": "joint_gauss_newton_not_historical_sra",
            "jacobian_shapes": [[192, 10], [192, 20]], "equal_total_compute": False,
            "production_authorized": False, "independent_confirmation": False, "P2_admission": False}
