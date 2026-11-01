"""Non-dispatching paired physical protocol state machine.

All receipts are caller-supplied; no runtime, model, environment, or GPU is loaded.
An acknowledgement is not proof of durable storage or of actual execution.
"""
import copy
import hashlib
import json
import math

ARMS = ("full_d2p", "sra_no_composition")
IDENTITY = ("root_sha256", "task_sha256", "goal_sha256", "history_sha256",
            "model_sha256", "normalizer_sha256", "reset_sha256", "controller_sha256",
            "renderer_sha256", "guard_sha256")
AXES = ("predict_calls", "predict_samples", "action_encoder", "visual_encoder",
        "jacobian", "vjp_directions", "svd", "solve", "physical_steps",
        "setup_steps", "io_bytes", "wall_ms")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def packet(commands, size):
    require(isinstance(commands, list) and len(commands) == size, "packet length")
    for command in commands:
        require(isinstance(command, list) and len(command) == 2 and all(
            type(x) in (int, float) and math.isfinite(x) and -1 <= x <= 1 for x in command), "invalid command")
    return copy.deepcopy(commands)


class PairedProtocol:
    """One shared protocol, independent per-arm histories, no direct dispatch."""

    def __init__(self, identities, initial_frames, initial_actions, horizon, packet_steps, caps, persist):
        require(set(identities) == set(ARMS), "both arm identities required")
        require(all(set(identities[a]) == set(IDENTITY) and all(sha(identities[a][k]) for k in IDENTITY) for a in ARMS),
                "complete source identities required")
        require(identities[ARMS[0]] == identities[ARMS[1]], "identity mismatch")
        require(type(horizon) is int and type(packet_steps) is int and 0 < packet_steps <= horizon
                and horizon % packet_steps == 0, "invalid schedule")
        require(set(initial_frames) == set(ARMS) and set(initial_actions) == set(ARMS), "missing initial histories")
        require(initial_frames[ARMS[0]] == initial_frames[ARMS[1]]
                and initial_actions[ARMS[0]] == initial_actions[ARMS[1]], "initial history mismatch")
        frames = initial_frames[ARMS[0]]
        require(isinstance(frames, list) and len(frames) == 3 and all(sha(x) for x in frames), "initial RGB hashes")
        hist = initial_actions[ARMS[0]]
        require(isinstance(hist, list) and len(hist) == 2, "two initial packets required")
        for p in hist:
            packet(p, packet_steps)
        require(set(caps) == set(ARMS) and all(set(caps[a]) == set(AXES) for a in ARMS), "explicit per-axis caps required")
        for arm in ARMS:
            require(all(type(caps[arm][x]) is int and caps[arm][x] >= 0 for x in AXES), "invalid cap")
        require(callable(persist), "persistence acknowledgement required")
        self.identity = copy.deepcopy(identities[ARMS[0]])
        self.horizon, self.packet_steps = horizon, packet_steps
        self.caps, self.persist = copy.deepcopy(caps), persist
        self.states = {}
        for arm in ARMS:
            self.states[arm] = {"phase": "observe", "step": 0, "frames": copy.deepcopy(frames),
                                "actions": copy.deepcopy(hist), "selection": None, "commands": None,
                                "pending_step": None, "terminal_reason": None,
                                "observations": {0: {"pixel_sha256": frames[-1], "physical_label": None}}, "attempted": {x: 0 for x in AXES},
                                "completed": {x: 0 for x in AXES}, "pending_resources": {}}

    def state(self, arm):
        require(arm in ARMS, "unknown arm")
        return self.states[arm]

    def observe(self, arm, step, frames, actions):
        s = self.state(arm)
        require(s["phase"] == "observe" and step == s["step"] and step < self.horizon, "wrong observation boundary")
        require(frames == s["frames"] and actions == s["actions"], "not own observed history")
        s["phase"] = "select"
        return {"arm": arm, "step": step, "identity": copy.deepcopy(self.identity),
                "frames": copy.deepcopy(frames), "actions": copy.deepcopy(actions)}

    def record_resource_attempt(self, arm, operation_id, amounts):
        s = self.state(arm)
        require(s["phase"] not in ("stopped", "complete") and isinstance(operation_id, str)
                and operation_id and operation_id not in s["pending_resources"], "invalid operation")
        require(isinstance(amounts, dict) and amounts and set(amounts) <= set(AXES) - {"physical_steps"}
                and all(type(v) is int and v > 0 for v in amounts.values()), "invalid resource amounts")
        require(all(s["attempted"][k] + v <= self.caps[arm][k] for k, v in amounts.items()), "resource cap")
        for k, v in amounts.items():
            s["attempted"][k] += v
        s["pending_resources"][operation_id] = copy.deepcopy(amounts)

    def record_resource_return(self, arm, operation_id):
        s = self.state(arm)
        require(operation_id in s["pending_resources"], "unknown resource attempt")
        amounts = s["pending_resources"].pop(operation_id)
        for k, v in amounts.items():
            s["completed"][k] += v

    def select(self, arm, request, commands, planner_receipt_sha256):
        s = self.state(arm)
        expected = {"arm": arm, "step": s["step"], "identity": self.identity,
                    "frames": s["frames"], "actions": s["actions"]}
        require(s["phase"] == "select" and request == expected, "request identity/history mismatch")
        require(sha(planner_receipt_sha256) and not s["pending_resources"], "missing planner receipt or pending work")
        commands = packet(commands, self.packet_steps)
        envelope = {"request": copy.deepcopy(request), "commands": commands,
                    "planner_receipt_sha256": planner_receipt_sha256}
        token = digest(envelope)
        try:
            ack = self.persist(copy.deepcopy(envelope))
            require(ack == token, "selection persistence acknowledgement")
        except Exception:
            s["phase"], s["terminal_reason"] = "stopped", "selection_persistence_failure"
            raise
        s["selection"], s["commands"], s["phase"] = token, commands, "execute"
        return token

    def begin_step(self, arm, token, guard_authorized):
        s = self.state(arm)
        require(s["phase"] == "execute" and s["selection"] == token and s["pending_step"] is None, "invalid selection/step")
        require(guard_authorized is True, "guard refused")
        require(s["attempted"]["physical_steps"] + 1 <= self.caps[arm]["physical_steps"], "physical cap")
        offset = s["step"] % self.packet_steps
        event = {"arm": arm, "step": s["step"] + 1, "selection_sha256": token,
                 "command": copy.deepcopy(s["commands"][offset])}
        try:
            require(self.persist(copy.deepcopy(event)) == digest(event), "step persistence acknowledgement")
        except Exception:
            s["phase"], s["terminal_reason"] = "stopped", "step_intent_persistence_failure"
            raise
        s["attempted"]["physical_steps"] += 1
        s["pending_step"] = event
        return copy.deepcopy(event)

    def finish_step(self, arm, event, pixel_sha256, physical_label, terminated=False, truncated=False):
        s = self.state(arm)
        require(s["phase"] == "execute" and s["pending_step"] == event and event is not None, "step return mismatch")
        require(sha(pixel_sha256) and type(terminated) is bool and type(truncated) is bool, "bad observation")
        require(isinstance(physical_label, dict) and set(physical_label) == {"state7", "reward"}
                and isinstance(physical_label["state7"], list) and len(physical_label["state7"]) == 7
                and all(type(x) in (int, float) and math.isfinite(x) for x in physical_label["state7"])
                and type(physical_label["reward"]) in (int, float) and math.isfinite(physical_label["reward"]),
                "invalid physical label")
        s["completed"]["physical_steps"] += 1
        s["step"] += 1
        s["pending_step"] = None
        s["observations"][s["step"]] = {"pixel_sha256": pixel_sha256, "physical_label": copy.deepcopy(physical_label),
                                         "terminated": terminated, "truncated": truncated}
        if terminated or truncated:
            s["phase"] = "stopped"
            s["terminal_reason"] = {"terminated": terminated, "truncated": truncated}
        elif s["step"] % self.packet_steps == 0:
            s["frames"] = s["frames"][1:] + [pixel_sha256]
            s["actions"] = s["actions"][1:] + [s["commands"]]
            s["commands"], s["selection"] = None, None
            s["phase"] = "complete" if s["step"] == self.horizon else "observe"

    def stop(self, arm, reason):
        s = self.state(arm)
        require(s["phase"] not in ("complete", "stopped") and isinstance(reason, str) and reason, "invalid stop")
        s["phase"], s["terminal_reason"] = "stopped", reason

    def snapshot(self):
        rows = []
        for arm in ARMS:
            s = self.states[arm]
            rows.append({"arm": arm, "status": s["phase"], "step": s["step"],
                         "terminal_reason": copy.deepcopy(s["terminal_reason"]),
                         "physical_step_pending": s["pending_step"] is not None,
                         "resources_attempted": copy.deepcopy(s["attempted"]),
                         "resources_completed_lower_bound": copy.deepcopy(s["completed"]),
                         "pending_resources": copy.deepcopy(s["pending_resources"]),
                         "checkpoints": {str(t): s["observations"].get(t) for t in range(self.horizon + 1)}})
        return {"identity": copy.deepcopy(self.identity), "rows": rows, "fixed_arm_denominator": 2,
                "fixed_time_denominator": 2 * (self.horizon + 1), "equal_total_compute": False,
                "flops": None, "durable_storage_verified": False, "actual_execution_verified": False,
                "scientific_admission": False, "dispatch_authorized": False}
