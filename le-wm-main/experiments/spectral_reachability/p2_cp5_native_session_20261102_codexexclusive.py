"""Split native session mechanics; no launcher or authorization is provided.

Use of this class alone does not authenticate the caller-supplied source data,
environment, storage, cooperative lock, or model winner. A future launcher must
verify those before construction and recheck them after execution.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from . import p3_goal_patch_consistent_compiler_20261027_v2 as normalization


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class NativeSession:
    """Native API split at plan step 15, then one public step at a time.

    The injected `physical` API is the historic physical module. Its Store and
    Meter must be used by a real launcher; fake equivalents are test-only.
    This class does not accept a model winner and cannot authorize a suffix.
    """
    def __init__(self, physical, env, root, source_plan, plan, row, meter, store, prepared, hooks):
        self.physical, self.env, self.root = physical, env, root
        self.source_plan, self.plan = source_plan, plan
        self.row, self.meter, self.store, self.prepared = row, meter, store, prepared
        self.hooks = hooks
        self.state = "new"
        self.plan_step = 0
        self.entered = False
        commands = np.asarray(plan["commands"], dtype="<f4")
        need(commands.shape == (25, 2) and np.isfinite(commands).all()
             and (np.abs(commands) <= 1).all()
             and commands.astype(float).tolist() == plan["commands"], "source plan float32")
        self.commands = commands

    def _emit(self, kind, **fields):
        self.store.emit(kind, **self.meter.where, **fields)

    def _advance(self, command, phase, step):
        p, body = self.physical, self.env.unwrapped
        need(not body.eval_state(self.goal, p.source.vector(self.obs["state"]))[0],
             "already task terminal")
        self._emit("segment_step_intent", phase=phase, step=step,
                   raw_float32_hex=command.tobytes().hex())
        try:
            obs, info, terminated, truncated, reward, actual = p.dispatch(
                self.env, command, phase, step, self.row, self.meter, self.store, self.goal)
            image = p.observe(self.env, obs, phase, step, self.row, self.meter, self.store, self.root["warm_trace"])
            success = p.paired.check_return(self.env, command, actual, info, self.goal, reward, terminated)
            if phase == "plan":
                bound = p.invariant(body)
                need(p.moving.exact(np.asarray(info["goal"]), self.root["packets"]["off"]["goal_rgb"])
                     and {k: v for k, v in bound.items() if k != "snapshot"} ==
                     {k: v for k, v in self.bound.items() if k != "snapshot"}, "goal or renderer changed")
                calc = p.native.metrics(self.goal, p.source.vector(obs["state"]))
                need(calc["native_success"] is bool(success) and calc["native7_reward"] == reward,
                     "native metric mismatch")
            self.obs = obs
            self._emit("segment_step_returned", phase=phase, step=step,
                       terminated=terminated, truncated=truncated)
            if terminated or truncated:
                self._emit("segment_native_stop", phase=phase, step=step,
                           terminated=terminated, truncated=truncated)
                self.state = "stopped"
                raise p.native.allmode.NativeStop("native stop without padding")
            return image
        except BaseException as exc:
            if self.state != "stopped":
                self.state = "failed"
            self._emit("segment_step_failed", phase=phase, step=step,
                       error_type=type(exc).__name__, message=str(exc)[:500])
            raise

    def reconstruct(self):
        need(self.state == "new", "session already used")
        p, body, packet = self.physical, self.env.unwrapped, self.root["packets"]["off"]
        try:
            self.hooks.__enter__()
            self.entered = True
            warm = p.moving.warmup_commands()
            need(p.moving.exact(np.asarray(packet["raw_action_history"]), warm), "warmup action identity")
            self.obs, self.goal = p.place(self.env, self.source_plan, self.root["slot"], self.meter, self.store)
            images = [p.observe(self.env, self.obs, "warmup", 0, self.row, self.meter, self.store,
                                self.root["warm_trace"])]
            for step, command in enumerate(warm, 1):
                image = self._advance(command, "warmup", step)
                if step in (5, 10):
                    images.append(image)
            need(p.moving.exact(np.asarray(images), packet["rgb_history"]), "warmup RGB identity")
            state, goal = p.source.vector(self.obs["state"]), p.source.vector(self.root["goal"])
            need(p.source.root_hash(state, goal) == self.root["root_sha256"], "native root identity")
            self.goal = goal
            p.bind_goal(body, goal, packet["goal_rgb"], self.meter, self.store)
            self.bound = p.invariant(body)
            self._emit("root_guard_accepted", root_sha256=self.root["root_sha256"],
                       task_id=self.plan["task_id"], snapshot=deepcopy(self.bound["snapshot"]))
            p.observe(self.env, self.obs, "plan", 0, self.row, self.meter, self.store, self.root["warm_trace"])
            frames = []
            for step, command in enumerate(self.commands[:15], 1):
                image = self._advance(command, "plan", step)
                if step in (10, 15):
                    frames.append(image)
            need(self.store.prefix_events.get(self.row["index"], []) == self.prepared["prefix_reference"],
                 "exact original warmup/shared prefix event identity")
            self._emit("shared_prefix_guard_accepted", root_sha256=self.root["root_sha256"],
                       task_id=self.plan["task_id"], prefix_float32_hex=self.commands[:15].tobytes().hex())
            snapshot = deepcopy(p.invariant(body)["snapshot"])
            guard = hashlib.sha256(canonical({"root": self.root["root_sha256"], "task": self.plan["task_id"],
                "prefix": self.commands[:15].tobytes().hex(), "snapshot": snapshot,
                "reference": self.prepared["prefix_reference"]})).hexdigest()
            self.plan_step, self.state = 15, "ready"
            return {"root_sha256": self.root["root_sha256"], "task_id": self.plan["task_id"],
                    "rgb": np.stack(frames), "history": normalization.normalized_history(self.commands[5:15]),
                    "guard": guard}
        except BaseException as exc:
            if self.state != "stopped":
                self.state = "failed"
            self._emit("reconstruction_failed", error_type=type(exc).__name__, message=str(exc)[:500])
            raise

    def step(self, command):
        need(self.state == "ready" and self.plan_step < 20, "session not ready or cp5 reached")
        a = np.asarray(command)
        need(a.dtype == np.dtype("<f4") and a.shape == (2,) and np.isfinite(a).all()
             and (np.abs(a) <= 1).all(), "raw action")
        step = self.plan_step + 1
        image = self._advance(a.copy(), "plan", step)
        self.plan_step = step
        snapshot = deepcopy(self.physical.invariant(self.env.unwrapped)["snapshot"])
        if step == 20:
            self._emit("cp5_capture", step=step, snapshot=snapshot,
                       rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        return {"rgb": image, "snapshot": snapshot, "terminated": False, "truncated": False}

    def close(self):
        if self.entered:
            self.entered = False
            self.hooks.__exit__(None, None, None)
