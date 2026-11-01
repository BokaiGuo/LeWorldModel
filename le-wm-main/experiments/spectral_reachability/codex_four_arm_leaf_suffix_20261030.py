"""Opt-in LeWM outer-call meter and fixed projected-gradient suffix.

Analytic non-module fixtures may run on CPU. Real torch modules require frozen
eval CUDA. This does not intercept internal modules or prove hook coverage.
"""
import json
import math

import torch

from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec


def _require(condition, message):
    spec.require(condition, message)


def _finite(t):
    return bool(torch.isfinite(t).all().item())


class LeafAdapter:
    """Exact LeWM encode/action_encoder/predict ABI, explicit single-thread use."""
    def __init__(self, model, driver, contract, contract_sha, *, analytic_fixture=False):
        spec.preflight(contract, contract_sha)
        _require(type(driver) is durable.Driver and not driver.closed and not driver.failed,
                 "active durable driver")
        with driver.path.open("rb") as f:
            row = json.loads(f.readline(1024 * 1024))
        body = {k: row[k] for k in ("index", "previous", "kind", "payload")}
        _require(row["index"] == 0 and row["previous"] is None and row["kind"] == "header"
                 and row["sha256"] == spec.digest(body) and row["payload"]["contract"] == contract
                 and row["payload"]["contract_sha256"] == contract_sha, "driver/contract binding")
        _require(all(callable(getattr(model, k, None)) for k in ("encode", "action_encoder", "predict")),
                 "LeWM callable interface")
        _require(type(analytic_fixture) is bool, "fixture flag")
        if analytic_fixture:
            _require(not isinstance(model, torch.nn.Module), "analytic fixture must not be a Module")
        else:
            _require(isinstance(model, torch.nn.Module) and all(not m.training for m in model.modules())
                     and all(not p.requires_grad and p.device.type == "cuda" for p in model.parameters())
                     and all(b.device.type == "cuda" for b in model.buffers()), "frozen eval CUDA model required")
        self.model, self.driver = model, driver
        self.contract_sha = contract_sha
        self.config_sha = contract["identity"]["optimizer_config"]
        self.analytic_fixture = analytic_fixture

    def _abort(self):
        # Existing driver has no abort API. Invalid operation is a durable fail-stop
        # marker; never overwrite an earlier failure or conceal journal I/O errors.
        if not self.driver.failed and not self.driver.closed:
            try:
                self.driver.dispatch("setup", "__abort__", {}, lambda: None)
            except ValueError:
                pass

    def _dispatch(self, phase, kind, cost, callback):
        vector = dict.fromkeys(spec.AXES, 0)
        vector.update(cost)
        return self.driver.dispatch(phase, kind, vector, callback)

    def _tensor(self, value):
        _require(isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > 0,
                 "nonempty batch tensor required")
        if not self.analytic_fixture:
            _require(value.device.type == "cuda", "CUDA inputs required")
        return value

    def encode(self, info, *, phase="setup"):
        try:
            _require(type(info) is dict and set(info) == {"pixels"}, "encode requires pixels only")
            pixels = self._tensor(info["pixels"])
            return self._dispatch(phase, "encode", {"encode_calls": 1, "encode_samples": pixels.shape[0]},
                                  lambda: self.model.encode(dict(info)))
        except BaseException:
            self._abort()
            raise

    def action_encode(self, actions, *, phase):
        try:
            self._tensor(actions)
            return self._dispatch(phase, "action_encode", {"action_encode_calls": 1},
                                  lambda: self.model.action_encoder(actions))
        except BaseException:
            self._abort()
            raise

    def predict(self, states, action_embeddings, *, phase):
        try:
            self._tensor(states)
            self._tensor(action_embeddings)
            _require(states.shape[0] == action_embeddings.shape[0], "batch mismatch")
            return self._dispatch(phase, "predict",
                                  {"predict_calls": 1, "predict_samples": states.shape[0]},
                                  lambda: self.model.predict(states, action_embeddings))
        except BaseException:
            self._abort()
            raise

    def gradient(self, loss, actions, *, phase):
        try:
            _require(isinstance(loss, torch.Tensor) and loss.ndim == 0 and _finite(loss), "finite scalar loss")
            self._tensor(actions)
            return self._dispatch(phase, "backward", {"backward_calls": 1, "vjp_rows": 1},
                                  lambda: torch.autograd.grad(loss, actions)[0])
        except BaseException:
            self._abort()
            raise


def optimize(leaf, states, history_actions, initial_actions, goal, config):
    """Fixed-step projected gradient descent, with no selection or verification.

    Inputs: states [B,H,D], past actions [B,H-1,A], initialization [B,T,A],
    goal [B,D]. Every prediction encodes the aligned past+current action context.
    Returned actions are post-update and have not been evaluated.
    """
    _require(type(leaf) is LeafAdapter, "leaf adapter required")
    try:
        _require(type(config) is dict and set(config) ==
                 {"steps", "learning_rate", "history_size", "lower", "upper"}, "suffix config fields")
        _require(spec.digest(config) == leaf.config_sha, "suffix config hash mismatch")
        _require(type(config["steps"]) is int and config["steps"] > 0
                 and type(config["history_size"]) is int and config["history_size"] > 0, "steps/history")
        for k in ("learning_rate", "lower", "upper"):
            _require(type(config[k]) in (int, float) and math.isfinite(config[k]), "finite config")
        _require(config["learning_rate"] > 0 and config["lower"] < config["upper"], "optimizer bounds")
        for t in (states, history_actions, initial_actions, goal):
            leaf._tensor(t)
            _require(t.is_floating_point() and _finite(t), "finite floating inputs")
        _require(states.ndim == 3 and history_actions.ndim == 3 and initial_actions.ndim == 3
                 and goal.ndim == 2, "input ranks")
        b, h, dim = states.shape
        _require(h == config["history_size"] and goal.shape == (b, dim)
                 and history_actions.shape[:2] == (b, h - 1)
                 and initial_actions.shape[0] == b and initial_actions.shape[1] > 0
                 and initial_actions.shape[2] > 0 and history_actions.shape[2] == initial_actions.shape[2],
                 "history/action/goal shapes")
        _require(all(t.device == states.device and t.dtype == states.dtype
                     for t in (history_actions, initial_actions, goal)), "device/dtype mismatch")
        _require(bool(((initial_actions >= config["lower"]) & (initial_actions <= config["upper"])).all().item()),
                 "initial action out of bounds")
        base = states.detach()
        past = history_actions.detach()
        target = goal.detach()
        actions = initial_actions.detach().clone()
        for _ in range(config["steps"]):
            actions = actions.detach().requires_grad_(True)
            z, a_hist = base, past
            for step in range(actions.shape[1]):
                context = torch.cat((a_hist, actions[:, step:step+1]), dim=1)
                encoded = leaf.action_encode(context, phase="common_suffix")
                pred = leaf.predict(z, encoded, phase="common_suffix")
                _require(isinstance(pred, torch.Tensor) and pred.shape == z.shape and _finite(pred),
                         "predict shape/finite")
                z = torch.cat((z, pred[:, -1:]), dim=1)[:, -h:]
                a_hist = context[:, -(h - 1):] if h > 1 else context[:, :0]
            loss = (z[:, -1] - target).square().mean()
            grad = leaf.gradient(loss, actions, phase="common_suffix")
            _require(grad.shape == actions.shape and _finite(grad), "gradient shape/finite")
            with torch.no_grad():
                actions = (actions - config["learning_rate"] * grad).clamp(config["lower"], config["upper"])
            _require(_finite(actions), "nonfinite optimizer update")
        return actions.detach()
    except BaseException:
        leaf._abort()
        raise
