"""Reusable D2P local compiler over a frozen, explicitly metered backend.

Neural backends must enforce CUDA, model immutability and actual call budgets.
The operation counters here are not neural compute accounting. Linear reach
values are unconstrained ridge objectives, not physical capability certificates.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .d2p_composition_witness_v2 import _context, digest


def _array(value, shape=None):
    a = np.asarray(value)
    if a.dtype.kind not in "fi" or not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError("finite real array with declared shape required")
    return a.astype(np.float64)


def local_geometry(jacobian, residual, sqrt_q, inverse_sqrt_r, damping):
    """K = sqrt(Q) G inverse_sqrt(R), with rank-truncated unreachable residual."""
    g, r, q, ri = map(_array, (jacobian, residual, sqrt_q, inverse_sqrt_r))
    if (g.ndim != 2 or min(g.shape) < 1 or r.shape != (g.shape[0],)
            or q.shape != (len(r), len(r)) or ri.shape != (g.shape[1], g.shape[1])
            or not np.isfinite(damping) or damping <= 0):
        raise ValueError("aligned metric/response and positive damping required")
    k, weighted_r = q @ g @ ri, q @ r
    u, singular, vh = np.linalg.svd(k, full_matrices=False)
    tolerance = singular[0] * max(k.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    basis = u[:, :rank]
    perpendicular = weighted_r - basis @ (basis.T @ weighted_r)
    whitened_delta = vh.T @ ((singular / (singular ** 2 + damping)) * (u.T @ weighted_r))
    remaining = weighted_r - k @ whitened_delta
    return {
        "rank": rank, "singular_values": singular.tolist(),
        "unreachable_fraction": float(perpendicular @ perpendicular / (weighted_r @ weighted_r + 1e-12)),
        "unreachable_residual": perpendicular.tolist(),
        "raw_delta": (ri @ whitened_delta).tolist(),
        "raw_spectral_directions": (ri @ vh[:rank].T).T.tolist(),
        "linear_reach_value": float((remaining @ remaining + damping * (whitened_delta @ whitened_delta))
                                   / len(r)),
        "linear_reach_scope": "unconstrained_local_ridge_not_physical_capability",
    }


@dataclass(frozen=True)
class CompilerConfig:
    packet_steps: int
    suffix_packets: int = 2
    spectral_modes: int = 2
    prefix_radius: float = .5
    damping: float = .01
    beta: float = 1.
    line_scales: tuple = (0., .25, .5, 1.)
    max_rollout_operations: int = 512
    max_jacobian_operations: int = 64

    def __post_init__(self):
        for name in ("packet_steps", "suffix_packets", "spectral_modes",
                     "max_rollout_operations", "max_jacobian_operations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("positive integer compiler bounds required")
        if (not np.isfinite([self.prefix_radius, self.damping, self.beta]).all()
                or self.prefix_radius <= 0 or self.damping <= 0 or self.beta < 0):
            raise ValueError("finite valid compiler coefficients")
        if (type(self.line_scales) is not tuple or not self.line_scales or self.line_scales[0] != 0
                or any(type(x) not in (int, float) or not np.isfinite(x) or not 0 <= x <= 1
                       for x in self.line_scales)):
            raise ValueError("fixed bounded line search must include zero incumbent")


class LocalCompiler:
    """Compile complete prefix/suffix witnesses without rewards or physical state.

    backend.rollout(context, raw_actions) returns a full model context.
    backend.jacobian(context, raw_actions) differentiates its terminal latent
    against flattened raw action coordinates in the same ordering.
    """
    def __init__(self, backend, *, goal, sqrt_q, action_scale, low, high, config, emit=None):
        self.backend, self.config = backend, config
        self.goal, self.q = _array(goal), _array(sqrt_q)
        self.scale, self.low, self.high = map(_array, (action_scale, low, high))
        if (self.goal.ndim != 1 or self.q.shape != (len(self.goal), len(self.goal))
                or self.scale.ndim != 1 or not self.scale.size
                or self.low.shape != self.scale.shape or self.high.shape != self.scale.shape
                or np.any(self.scale <= 0) or np.any(self.low >= self.high)
                or np.any(self.low > 0) or np.any(self.high < 0)):
            raise ValueError("compatible goal/metric and zero-containing raw action domain required")
        if not isinstance(config, CompilerConfig):
            raise ValueError("frozen compiler configuration required")
        self.counts = {"rollout_attempted": 0, "rollout_returned": 0,
                       "jacobian_attempted": 0, "jacobian_returned": 0}
        self.emit = emit if emit is not None else lambda event: None

    def _actions(self, actions):
        a = _array(actions)
        if (a.ndim != 2 or a.shape[1] != len(self.scale) or not len(a)
                or len(a) % self.config.packet_steps
                or np.any(a < self.low) or np.any(a > self.high)):
            raise ValueError("complete bounded raw action packets required")
        return a

    def _call(self, kind, context, actions):
        _context(context)
        actions = self._actions(actions)
        if context["latent_dim"] != len(self.goal):
            raise ValueError("goal/context dimension mismatch")
        cap = getattr(self.config, "max_" + kind + "_operations")
        if self.counts[kind + "_attempted"] >= cap:
            raise RuntimeError("compiler operation cap exhausted")
        self.counts[kind + "_attempted"] += 1
        self.emit({"kind": kind + "_attempt", "counts": dict(self.counts),
                   "context_sha256": digest(context), "actions": actions.tolist()})
        try:
            result = getattr(self.backend, kind)(deepcopy(context), actions.copy())
            if kind == "rollout":
                _context(result)
                if any(result[k] != context[k] for k in ("layout", "context_length", "latent_dim", "action_dim")):
                    raise ValueError("model context layout changed")
            else:
                result = _array(result, (len(self.goal), actions.size))
        except Exception as exc:
            self.emit({"kind": kind + "_failed", "error": str(exc), "counts": dict(self.counts)})
            raise
        self.counts[kind + "_returned"] += 1
        self.emit({"kind": kind + "_return", "counts": dict(self.counts)})
        return result

    def _cost(self, context):
        r = self.q @ (self.goal - _array(context["latent_history"][-1], self.goal.shape))
        return float(np.mean(r ** 2))

    def geometry(self, context, actions):
        nominal = self._call("rollout", context, actions)
        jacobian = self._call("jacobian", context, actions)
        ri = np.diag(np.tile(self.scale, len(actions)))
        result = local_geometry(jacobian, self.goal - np.asarray(nominal["latent_history"][-1]),
                                self.q, ri, self.config.damping)
        return nominal, result

    def progress(self, context, packets=None):
        packets = self.config.suffix_packets if packets is None else packets
        if type(packets) is not int or packets < 1:
            raise ValueError("positive packet horizon required")
        nominal_actions = np.zeros((packets * self.config.packet_steps, len(self.scale)))
        nominal, geometry = self.geometry(context, nominal_actions)
        delta = np.asarray(geometry["raw_delta"]).reshape(nominal_actions.shape)
        bound = 1.
        positive, negative = delta > 0, delta < 0
        upper = np.broadcast_to(self.high, delta.shape)
        lower = np.broadcast_to(self.low, delta.shape)
        if positive.any():
            bound = min(bound, float(np.min(upper[positive] / delta[positive])))
        if negative.any():
            bound = min(bound, float(np.min(lower[negative] / delta[negative])))
        best = (self._cost(nominal), nominal_actions, nominal, 0.)
        trials = []
        for scale in self.config.line_scales:
            actions = nominal_actions + scale * bound * delta
            successor = nominal if scale == 0 else self._call("rollout", context, actions)
            cost = self._cost(successor)
            trials.append({"scale": scale, "cost": cost})
            if cost < best[0]:
                best = cost, actions, successor, scale
        return {"kind": "progress", "input_context": deepcopy(context), "output_context": best[2],
                "actions": best[1].tolist(), "nonlinear_cost": best[0], "accepted_scale": best[3],
                "geometry": geometry, "trials": trials}

    def compile(self, context):
        root_progress = self.progress(context, packets=1)
        geometry = root_progress["geometry"]
        bank = [("zero", np.zeros((self.config.packet_steps, len(self.scale)))),
                ("progress", np.asarray(root_progress["actions"]))]
        for i, raw in enumerate(geometry["raw_spectral_directions"][:self.config.spectral_modes]):
            direction = np.asarray(raw).reshape(bank[0][1].shape)
            direction /= max(float(np.max(np.abs(direction))), 1e-12)
            for sign in (-1, 1):
                proposal = sign * self.config.prefix_radius * direction
                if np.all(proposal >= self.low) and np.all(proposal <= self.high):
                    bank.append((f"spectrum_{i}_{sign:+d}", proposal))
        programs = []
        for name, actions in bank:
            successor = self._call("rollout", context, actions)
            suffix = self.progress(successor)
            stage_cost = self._cost(successor)
            objective = stage_cost + self.config.beta * suffix["geometry"]["linear_reach_value"]
            full_actions = actions.tolist() + suffix["actions"]
            direct = self._call("rollout", context, full_actions)
            if digest(direct) != digest(suffix["output_context"]):
                raise ValueError("full action witness differs from split composition")
            programs.append({"candidate": name, "input_context": deepcopy(context),
                             "output_context": direct, "actions": full_actions,
                             "prefix": {"kind": "reconfiguration_candidate", "actions": actions.tolist(),
                                        "output_context": successor, "stage_cost": stage_cost},
                             "suffix": suffix, "objective": objective,
                             "nonlinear_cost": self._cost(direct), "witness_replayed": True})
        selected = min(range(len(programs)), key=lambda i: (programs[i]["objective"], i))
        greedy = min(range(len(programs)), key=lambda i: (programs[i]["prefix"]["stage_cost"], i))
        return {"schema": "d2p_local_compiler.v1", "programs": programs, "selected_index": selected,
                "same_bank_greedy_index": greedy, "root_geometry": geometry,
                "selection": "stage_plus_beta_local_ridge_reach_value",
                "operation_counts": dict(self.counts), "actual_neural_cost_certified": False,
                "backend_counters": deepcopy(getattr(self.backend, "counters", {})),
                "P1_admission": False, "physical_mechanism_verified": False}
