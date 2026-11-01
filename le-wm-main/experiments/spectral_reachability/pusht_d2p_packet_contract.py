"""Bounded PushT packet primitives; no checkpoint, runner, or simulator imports.

One token is five *relative* live commands, flattened in time-major order.
Repeating a command is legal, but still requires five separate physical steps.
These contracts do not establish training statistics, P0 admission, or P1 cause.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import uuid

import numpy as np

from experiments.pusht_protocol_contracts import root_fingerprint


VERSION = "pusht_d2p_packet_contract.v1"
ACTION_BLOCK = 5
HISTORY_SIZE = 3
REQUIRED_GATES = (
    "strict_packet_execution", "contiguous_boundary_history",
    "repeated_packet_equivalence", "goal_metadata_and_render_purity",
    "first_stored_transition_replay",
)


def contract_spec():
    return {
        "version": VERSION,
        "action_block": ACTION_BLOCK,
        "command_width": 2,
        "layout": "time_major_5x2",
        "execution": "five_separate_relative_env_steps",
        "normalization": "explicit_supplied_stats_no_extra_div100",
        "history": "actual_contiguous_boundaries_root_anchored_v1",
        "observed_suffix": "no_second_bootstrap_no_extra_action_shift",
        "suffix_projection": "identical_nonempty_content_identity_required",
    }


def _frozen(value, *, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def packet_commands(packet):
    """Validate the *entire* live packet before stepping; never silently clip."""
    values = np.asarray(packet, dtype=np.float64)
    if values.shape == (10,):
        values = values.reshape(ACTION_BLOCK, 2)
    if values.shape != (ACTION_BLOCK, 2):
        raise ValueError("packet must be exactly time-major [5,2] or [10]")
    if not np.isfinite(values).all():
        raise ValueError("packet commands must be finite")
    if np.any(np.abs(values) > 1):
        raise ValueError("live command bounds are [-1,1]; clipping is forbidden")
    return _frozen(values)


def repeat_command(command):
    command = np.asarray(command, dtype=np.float64)
    if command.shape != (2,):
        raise ValueError("a repeated live command must have shape [2]")
    return packet_commands(np.tile(command, (ACTION_BLOCK, 1)))


class PacketTerminated(ValueError):
    """A partial packet cannot be completed with synthetic/padded boundaries."""


def execute_packet(env, packet, *, on_step=None):
    commands = packet_commands(packet)
    results = []
    for command in commands:
        result = env.step(command.copy())
        if len(result) != 5:
            raise ValueError("expected Gymnasium five-field step result")
        results.append(result)
        if on_step is not None:
            on_step(result)
        if result[2] or result[3]:
            raise PacketTerminated(f"packet terminated/truncated at command {len(results)}/5")
    return results


def normalize_live_packet(commands, *, mean, std):
    """Differentiable live [..,5,2] -> model [..,10], using supplied stats only.

    Stats may be per-command [2] (tiled, never repeat-interleaved) or per-packet
    [10]. The caller owns provenance; public constants are not checkpoint truth.
    No tensor detachment occurs on the normalized/gradient path.
    """
    import torch  # optional tensor arithmetic only; never loads a neural model

    if not isinstance(commands, torch.Tensor) or not commands.is_floating_point():
        raise ValueError("live commands must be a floating torch tensor")
    if commands.ndim < 2 or tuple(commands.shape[-2:]) != (ACTION_BLOCK, 2):
        raise ValueError("live tensor must have trailing shape [5,2]")
    if not bool(torch.isfinite(commands).all()) or bool((commands.abs() > 1).any()):
        raise ValueError("live tensor commands must be finite and in bounds")
    stats = []
    for name, value in (("mean", mean), ("std", std)):
        stat = torch.as_tensor(value, dtype=commands.dtype, device=commands.device)
        if tuple(stat.shape) == (2,):
            stat = stat.repeat(ACTION_BLOCK)
        if tuple(stat.shape) != (10,) or not bool(torch.isfinite(stat).all()):
            raise ValueError(f"explicit {name} must be finite [2] or [10]")
        if name == "std" and bool((stat <= 0).any()):
            raise ValueError("explicit std must be strictly positive")
        stats.append(stat)
    return (commands.reshape(*commands.shape[:-2], 10) - stats[0]) / stats[1]


def state_of(env):
    state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("official physical state must be a finite seven-vector")
    return _frozen(state)


def render_without_advancing(env):
    """Record RGB only if repeated official renders leave physical state intact."""
    before = state_of(env)
    first = np.array(env.render(), copy=True)
    second = np.array(env.render(), copy=True)
    after = state_of(env)
    if not np.array_equal(before, after) or not np.array_equal(first, second):
        raise ValueError("render advanced state or changed repeated RGB")
    if first.shape != (224, 224, 3) or first.dtype != np.uint8:
        raise ValueError("expected official uint8 HWC 224x224 RGB")
    return _frozen(first)


@dataclass(frozen=True)
class ObservedContext:
    root_id: str
    history_id: str
    kind: str
    decision_step: int
    frame_steps: tuple[int, ...]
    frames: tuple[np.ndarray, ...]
    action_packets: np.ndarray


class BoundaryHistory:
    """Actual runtime history, not interpolated/skipped frames or reset replay.

    After three warmup packets: frames F0,F1,F2, actions A0,A1,A2, live F3.
    After one prefix A3: observed suffix frames F2,F3,F4, carry A1,A2,A3.
    A suffix appender can then append A4 and take the last three actions. Do not
    bootstrap this observed context again or shift its carried actions again.
    """

    def __init__(self, env, *, root_state, goal):
        for value in (root_state, goal):
            if np.asarray(value).shape != (7,) or not np.isfinite(value).all():
                raise ValueError("root and goal must be finite seven-vectors")
        self.env = env
        self.root_id = root_fingerprint({"state": root_state, "goal": goal})
        self.history_id = uuid.uuid4().hex
        self.step = 0
        self._partial = False
        self._packets = []
        self._states = [state_of(env)]
        self._frames = [render_without_advancing(env)]
        self._step_states = []

    @property
    def packets(self):
        return _frozen(np.asarray(self._packets).reshape(-1, ACTION_BLOCK, 2))

    @property
    def frames(self):
        return tuple(self._frames)

    @property
    def boundary_steps(self):
        return tuple(range(0, len(self._frames) * ACTION_BLOCK, ACTION_BLOCK))

    @property
    def step_states(self):
        return tuple(self._step_states)

    def execute(self, packet):
        if self._partial:
            raise ValueError("history has a partial/failed packet; reset is required")
        commands = packet_commands(packet)
        if (not np.array_equal(state_of(self.env), self._states[-1]) or
                not np.array_equal(render_without_advancing(self.env), self._frames[-1])):
            raise ValueError("environment advanced outside this boundary history")

        def record(result):
            self.step += 1
            actual = state_of(self.env)
            if not np.array_equal(np.asarray(result[0]["state"]), actual):
                raise ValueError("step observation is not the actual physical state")
            self._step_states.append(actual)

        try:
            execute_packet(self.env, commands, on_step=record)
            frame = render_without_advancing(self.env)
            state = state_of(self.env)
        except Exception:
            self._partial = True
            raise
        self._packets.append(commands)
        self._frames.append(frame)
        self._states.append(state)
        return frame

    def _context_at(self, decision_step, kind):
        if decision_step % ACTION_BLOCK or decision_step < 15 or decision_step > self.step:
            raise ValueError("context must be anchored at a recorded decision boundary")
        n = decision_step // ACTION_BLOCK
        if kind == "warmup_outgoing" and n == HISTORY_SIZE:
            indices = range(n - HISTORY_SIZE, n)
        elif kind == "observed_carry":
            indices = range(n - HISTORY_SIZE + 1, n + 1)
        else:
            raise ValueError("invalid context role/bootstrap alignment")
        return ObservedContext(
            self.root_id, self.history_id, kind, decision_step,
            tuple(i * ACTION_BLOCK for i in indices),
            tuple(self._frames[i] for i in indices),
            _frozen(self._packets[n - HISTORY_SIZE:n]),
        )

    def warmup_context(self):
        if self.step != 15 or self._partial:
            raise ValueError("warmup requires exactly three completed physical packets")
        return self._context_at(self.step, "warmup_outgoing")


def update_observed_context(previous, history):
    """Advance exactly one packet using the same actual root/runtime recorder."""
    if history._partial or history.step != previous.decision_step + ACTION_BLOCK:
        raise ValueError("observed update requires exactly one contiguous new boundary")
    expected = history._context_at(previous.decision_step, previous.kind)
    scalar_fields = ("root_id", "history_id", "kind", "decision_step", "frame_steps")
    if any(getattr(previous, key) != getattr(expected, key) for key in scalar_fields):
        raise ValueError("context root/runtime/boundary identity mismatch")
    if (len(previous.frames) != HISTORY_SIZE or
            any(not np.array_equal(a, b) for a, b in zip(previous.frames, expected.frames)) or
            not np.array_equal(previous.action_packets, expected.action_packets)):
        raise ValueError("context frames/actions do not match actual recorded history")
    return history._context_at(history.step, "observed_carry")


def projection_identity(projection):
    value = np.asarray(projection)
    if value.dtype.kind != "f" or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("projection must be a nonempty finite floating array")
    digest = hashlib.sha256(json.dumps([value.dtype.str, value.shape]).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _sha_identity(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def require_shared_suffix_projection(selected_identity, zero_identity):
    if (not _sha_identity(selected_identity) or not _sha_identity(zero_identity) or
            selected_identity != zero_identity):
        raise ValueError("selected and zero suffixes must share the same projection identity")
    return selected_identity


def validate_receipt(receipt):
    """Fail closed on old P0/P1 receipts, absent gates, and unbound reports.

    Passing this validator admits only this sidecar's physical contract evidence.
    It does not authorize historical prediction reuse or any neural/P1 claim.
    """
    if (receipt.get("schema") != VERSION or receipt.get("contract") != contract_spec() or
            receipt.get("scope") != "physical_contract_only" or receipt.get("status") != "passed"):
        raise ValueError("receipt is old, failed, or has a different explicit version contract")
    gates = receipt.get("gates", {})
    if any(gates.get(key) is not True for key in REQUIRED_GATES):
        raise ValueError("receipt lacks complete physical contract gates")
    if any(not _sha_identity(receipt.get(key)) for key in ("protocol_sha256", "implementation_sha256")):
        raise ValueError("receipt must bind immutable protocol and implementation")
    return receipt
