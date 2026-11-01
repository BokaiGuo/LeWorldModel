"""dev303 CEM host bridge. Importing this module does not run CUDA or PushT.

This is a wiring component, not an authorized experiment launcher.  Its caller
must authenticate the model, reset, source frames and durable event sink.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from . import dev303_full_cem_cp5_wiring_20261102_exclusive as cem


def require(value, message):
    if not value:
        raise ValueError(message)


class Host:
    """Causal CEM host; capture, encoder, actuator and sink are injected."""
    def __init__(self, model, encode, capture, step, initial_latents, goal,
                 raw_history, mean, std, emit, initial_rgb_sha256):
        self.model, self.encode, self.capture, self.step, self.emit = (
            model, encode, capture, step, emit)
        self.latents = list(initial_latents)
        self.goal, self.mean, self.std = goal, mean, std
        self.raw_history = np.asarray(raw_history, dtype=np.float32).copy()
        require(isinstance(initial_rgb_sha256, str) and len(initial_rgb_sha256) == 64
                and all(c in "0123456789abcdef" for c in initial_rgb_sha256),
                "initial RGB identity")
        self.initial_rgb_sha256 = initial_rgb_sha256
        require(len(self.latents) == 3 and self.raw_history.shape == (10, 2),
                "initial history")
        require(all(isinstance(x, torch.Tensor) and x.shape == goal.shape
                    and x.device == goal.device and torch.isfinite(x).all()
                    for x in self.latents) and goal.ndim == 1, "latent history")
        require(mean.shape == std.shape == (2,) and mean.device == goal.device
                and std.device == goal.device and bool(torch.isfinite(mean).all())
                and bool(torch.isfinite(std).all()) and bool((std > 0).all())
                and np.isfinite(self.raw_history).all(), "normalization")
        self.cost = {f"{kind}_{state}": 0 for kind in
                     ("rgb_encode", "render", "control", "action_encode", "predict")
                     for state in ("attempted", "returned")}
        self.actions = []

    def _call(self, kind, callback):
        self.cost[kind + "_attempted"] += 1
        self.emit({"type": "cost_attempt", "kind": kind})
        start = time.monotonic()
        try:
            value = callback()
        except BaseException:
            self.emit({"type": "cost_failure", "kind": kind,
                       "elapsed_seconds": time.monotonic() - start})
            raise
        self.cost[kind + "_returned"] += 1
        self.emit({"type": "cost_return", "kind": kind,
                   "elapsed_seconds": time.monotonic() - start})
        return value

    def action_encode(self, actions, *, phase):
        return self._call("action_encode", lambda: self.model.action_encoder(actions))

    def predict(self, states, actions, *, phase):
        return self._call("predict", lambda: self.model.predict(states, actions))

    def observe(self, step, actual):
        require(step == len(self.actions) and np.array_equal(
            actual, np.asarray(self.actions, np.float32).reshape(-1, 2)), "action history")
        if step == 5:
            pixels = self._call("render", lambda: self.capture(step))
            require(isinstance(pixels, np.ndarray) and pixels.ndim == 3
                    and pixels.shape[-1] == 3 and pixels.dtype == np.uint8, "RGB")
            sha = hashlib.sha256(pixels.tobytes()).hexdigest()
            latent = self._call("rgb_encode", lambda: self.encode(pixels))
            require(isinstance(latent, torch.Tensor) and latent.shape == self.goal.shape
                    and latent.device == self.goal.device and bool(torch.isfinite(latent).all()),
                    "feedback latent")
            self.latents = self.latents[-2:] + [latent]
            self.raw_history = np.vstack((self.raw_history[-5:], actual))
            self.emit({"type": "feedback", "step": step, "rgb_sha256": sha,
                       "actual_actions": actual.tolist()})
        else:
            require(step == 0, "feedback step")
            sha = self.initial_rgb_sha256
        device = self.goal.device
        history = torch.as_tensor(self.raw_history, dtype=self.goal.dtype, device=device)
        context = {"states": torch.stack(self.latents).unsqueeze(0),
                   "history": ((history - self.mean) / self.std).reshape(1, 2, 10),
                   "goal": self.goal.unsqueeze(0), "mean": self.mean, "std": self.std,
                   "matrix": torch.eye(self.goal.numel(), dtype=self.goal.dtype, device=device)}
        return {"step": step, "actual_actions": actual, "rgb_sha256": sha, "context": context}

    def issue(self, start, commands):
        require(start == len(self.actions), "dispatch position")
        returned = []
        stopped = False
        for action in commands:
            action = np.asarray(action, dtype=np.float32)
            n = len(self.actions) + 1
            self.emit({"type": "control_dispatch", "step": n, "action": action.tolist()})
            def call():
                actual, done = self.step(action.copy())
                require(type(done) is bool and np.array_equal(np.asarray(actual), action),
                        "action mismatch or invalid stop")
                return np.asarray(actual, np.float32).copy(), done
            actual, stopped = self._call("control", call)
            self.actions.append(actual)
            returned.append(actual)
            self.emit({"type": "control_return", "step": n, "action": actual.tolist(),
                       "stopped": stopped})
            if stopped:
                break
        return {"executed_actions": np.asarray(returned, np.float32).reshape(-1, 2),
                "stopped": stopped}

    def run(self, *, seed):
        started = time.monotonic()
        try:
            return cem.run_model_callbacks(self, self.observe, self.issue,
                                           seed=seed, emit=self.emit)
        finally:
            self.emit({"type": "host_cost", "counts": dict(self.cost),
                       "wall_seconds": time.monotonic() - started,
                       "flops": None, "gpu_energy": None,
                       "initial_encoding_and_warmup": "caller_must_account"})


def native_callbacks(env, store, model, device, goal_state, goal_rgb, emit):
    """Bind the host to the existing native capture, transform and step ABI.

    The caller must first verify reset/warmup with the sealed dev303 receipts.
    """
    from . import p1_dev301_blind_plan_physical_20261101_codex as base
    from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as native
    from . import preflight_pusht_d2p_p0 as p0

    transform = p0.image_transform()

    def capture(step):
        store.check()
        row = base.capture(store, env, "cem", "control", step)
        pixels = base.image(store.path / row["file"])
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"],
                "saved feedback pixels")
        emit({"type": "native_capture", "step": step, "receipt": row})
        return pixels

    def encode(pixels):
        image = p0._image_tensor(pixels, transform=transform, device=device)
        with torch.inference_mode():
            return model.encode({"pixels": image})["emb"][0, 0].detach()

    def step(action):
        obs, reward, terminated, truncated, info = env.step(action.copy())
        actual = native._check_return(env, obs, info, goal_state, goal_rgb, action)
        emit({"type": "native_step", "action": np.asarray(actual).tolist(),
              "reward": float(reward), "terminated": bool(terminated),
              "truncated": bool(truncated)})
        return actual, bool(terminated or truncated)

    return capture, encode, step
