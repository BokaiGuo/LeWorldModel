"""Prospective vector reservation guard and opt-in LeWM module bypass sentinels.

This guards only explicitly routed calls and installed module boundaries. It
does not authenticate arbitrary Python, CUDA kernels, or source-time consent.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

NAME = "p1_prospective_total_guard_20261101_leibniz"
AXES = (
    "model_load", "source_encode_calls", "source_encode_samples", "action_encode_calls",
    "predict_calls", "predict_samples", "backward_calls", "vjp_rows",
    "jacobian_calls", "svd_calls", "solve_calls", "candidate_evaluations",
    "replay_calls", "feedback_calls", "physical_dispatch_calls", "physics_steps",
    "render_calls", "reset_calls",
)
PRIMARY = {
    "model_load": "model_load", "source_encode": "source_encode_calls",
    "action_encode": "action_encode_calls", "predict": "predict_calls",
    "backward": "backward_calls", "jacobian": "jacobian_calls", "svd": "svd_calls",
    "solve": "solve_calls", "candidate": "candidate_evaluations", "replay": "replay_calls",
    "feedback": "feedback_calls", "physical_dispatch": "physical_dispatch_calls",
    "physics_step": "physics_steps", "render": "render_calls", "reset": "reset_calls",
}
EXTRA = {"source_encode": "source_encode_samples", "predict": "predict_samples", "backward": "vjp_rows"}
PHASES = ("shared_setup", "arm_setup", "planning", "feedback", "physical", "verification")
IDENTITY = ("root", "task", "goal", "model", "action_normalizer", "hardware", "method", "controller")
HOOKS = {"source_encode": ["encoder", "projector"], "action_encode": ["action_encoder"],
         "predict": ["predictor", "pred_proj"]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def vector(value):
    require(type(value) is dict and set(value) == set(AXES)
            and all(type(v) is int and v >= 0 for v in value.values()), "complete integer vector required")
    return value


def validate(contract, expected):
    require(type(contract) is dict and set(contract) == {"schema", "run_id", "arm", "caps", "source_sha256", "identity",
            "shared_initialization", "hook_modules", "scope"}, "contract fields")
    require(contract["schema"] == NAME + ".contract.v1"
            and type(contract["run_id"]) is str and bool(contract["run_id"])
            and contract["arm"] in ("structured", "cem"), "run/arm")
    vector(contract["caps"])
    require(contract["shared_initialization"] == "full_standalone_charge_each_arm"
            and contract["scope"] == "per_root_per_arm_start_before_load"
            and contract["hook_modules"] == HOOKS, "scope/shared allocation/hooks")
    require(type(contract["source_sha256"]) is dict and bool(contract["source_sha256"])
            and all(type(p) is str and Path(p).is_absolute() and type(h) is str
                    and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
                    for p, h in contract["source_sha256"].items()), "source bindings")
    require(str(Path(__file__).resolve()) in contract["source_sha256"], "guard source not bound")
    require(type(contract["identity"]) is dict and set(contract["identity"]) == set(IDENTITY)
            and all(type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
                    for h in contract["identity"].values()), "complete identity hashes")
    require(type(expected) is str and digest(contract) == expected, "external contract hash")
    return True


def validate_pair(structured, cem):
    validate(structured, digest(structured))
    validate(cem, digest(cem))
    require(structured["arm"] == "structured" and cem["arm"] == "cem"
            and structured["run_id"] != cem["run_id"] and structured["caps"] == cem["caps"],
            "paired arms/caps")
    common = ("root", "task", "goal", "model", "action_normalizer", "hardware")
    require(all(structured["identity"][k] == cem["identity"][k] for k in common), "paired identity")
    return {"pair_compatible": True, "precommitment_time_verified": False,
            "coverage_verified": False, "fairness_proven": False}


def check_sources(contract):
    for name, sha in contract["source_sha256"].items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                "source drift: " + name)


class Guard:
    """Durable pre-dispatch reservation, nested same-thread calls, fail-stop."""
    def __init__(self, path, contract, expected):
        validate(contract, expected)
        check_sources(contract)
        path = Path(path)
        require(path.is_absolute() and path.parent.is_dir() and not path.exists() and not path.is_symlink(),
                "exclusive journal")
        self.path, self.contract = path, deepcopy(contract)
        self.charged = dict.fromkeys(AXES, 0)
        self.stack, self.handles = [], []
        self.lock = threading.RLock()
        self.failed = self.closed = False
        self.index, self.previous = 0, None
        self.stream = path.open("xb")
        self.emit("header", {"contract": self.contract, "contract_sha256": expected})

    def emit(self, kind, payload):
        event = {"index": self.index, "previous": self.previous, "kind": kind, "payload": payload}
        event["sha256"] = digest(event)
        data = canonical(event) + b"\n"
        require(self.stream.write(data) == len(data), "journal short write")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.index += 1
        self.previous = event["sha256"]

    def fail(self, reason):
        self.failed = True
        self.emit("violation", {"reason": reason})
        raise ValueError(reason)

    def install_model_sentinels(self, model):
        require(not self.closed and not self.failed and not self.handles, "sentinel lifecycle")
        modules = {name: getattr(model, name, None) for names in HOOKS.values() for name in names}
        require(all(callable(getattr(m, "register_forward_pre_hook", None)) for m in modules.values())
                and len({id(m) for m in modules.values()}) == len(modules), "distinct module hooks required")
        def register(name, module):
            def hook(_module, _args):
                if not self.stack or name not in self.stack[-1]["expected_hooks"]:
                    self.fail("unguarded module call: " + name)
                seen = self.stack[-1]["observed_hooks"]
                if name in seen:
                    self.fail("duplicate module call: " + name)
                seen.append(name)
            return module.register_forward_pre_hook(hook)
        try:
            for name, module in modules.items():
                self.handles.append(register(name, module))
        except BaseException:
            for handle in self.handles:
                handle.remove()
            self.failed = True
            raise
        self.emit("hooks_installed", {"modules": sorted(modules)})

    def dispatch(self, phase, kind, units, fn):
        require(self.lock.acquire(blocking=False), "concurrent dispatch forbidden")
        try:
            require(not self.failed and not self.closed, "guard fail-stopped")
            require(phase in PHASES and kind in PRIMARY and callable(fn)
                    and type(units) is int and units > 0, "dispatch input")
            if kind in HOOKS:
                require(len(self.handles) == sum(map(len, HOOKS.values())), "model hooks required")
            delta = dict.fromkeys(AXES, 0)
            delta[PRIMARY[kind]] = 1 if kind in EXTRA else units
            if kind in EXTRA:
                delta[EXTRA[kind]] = units
            payload = {"id": self.index, "phase": phase, "kind": kind, "delta": delta}
            if any(self.charged[k] + delta[k] > self.contract["caps"][k] for k in AXES):
                self.failed = True
                self.emit("blocked", payload)
                raise ValueError("budget exceeded before dispatch")
            self.emit("attempt", payload)
            for k in AXES:
                self.charged[k] += delta[k]
            entry = {"id": payload["id"], "expected_hooks": list(HOOKS.get(kind, ())), "observed_hooks": []}
            self.stack.append(entry)
            try:
                result = fn()
                require(not self.failed, "guard violation during callback")
                require(sorted(entry["observed_hooks"]) == sorted(entry["expected_hooks"]), "missing model hook")
            except BaseException as exc:
                self.failed = True
                self.emit("failure", {"id": payload["id"], "type": type(exc).__name__, "message": str(exc)[:500]})
                raise
            finally:
                self.stack.pop()
            self.emit("returned", {"id": payload["id"], "observed_hooks": entry["observed_hooks"]})
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def close(self):
        require(not self.closed and not self.stack, "close outside callback")
        for handle in self.handles:
            handle.remove()
        try:
            check_sources(self.contract)
        except BaseException:
            self.failed = True
        self.emit("terminal", {"status": "failed_or_blocked" if self.failed else "closed_unverified",
                               "charged": self.charged, "coverage_verified": False,
                               "full_cost_certified": False, "fairness_proven": False})
        self.closed = True
        self.stream.close()
        return {"journal": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "status": "failed_or_blocked" if self.failed else "closed_unverified",
                "coverage_verified": False, "full_cost_certified": False}


class LeWMFacade:
    """New-caller integration seam for real LeWM CUDA model, not sealed hosts."""
    def __init__(self, model, guard):
        import torch
        require(isinstance(model, torch.nn.Module) and all(not m.training for m in model.modules())
                and all(p.device.type == "cuda" and not p.requires_grad for p in model.parameters())
                and all(b.device.type == "cuda" for b in model.buffers()), "frozen eval CUDA model required")
        self.model, self.guard = model, guard
        guard.install_model_sentinels(model)

    @staticmethod
    def _cuda(tensor):
        require(getattr(getattr(tensor, "device", None), "type", None) == "cuda"
                and len(tensor.shape) >= 1 and tensor.shape[0] > 0, "CUDA tensor required")

    def encode(self, pixels, phase):
        self._cuda(pixels)
        require(len(pixels.shape) >= 2 and pixels.shape[1] > 0, "time dimension required")
        return self.guard.dispatch(phase, "source_encode", pixels.shape[0] * pixels.shape[1],
                                   lambda: self.model.encode({"pixels": pixels}))

    def action_encode(self, actions, phase):
        self._cuda(actions)
        return self.guard.dispatch(phase, "action_encode", 1, lambda: self.model.action_encoder(actions))

    def predict(self, states, actions, phase):
        self._cuda(states)
        self._cuda(actions)
        require(states.shape[0] == actions.shape[0], "predict batch mismatch")
        return self.guard.dispatch(phase, "predict", states.shape[0], lambda: self.model.predict(states, actions))

    def gradient(self, loss, actions, phase, *, directions=1):
        self._cuda(actions)
        require(getattr(getattr(loss, "device", None), "type", None) == "cuda"
                and directions == 1 and len(loss.shape) == 0, "only one CUDA scalar gradient per dispatch")
        import torch
        return self.guard.dispatch(phase, "backward", 1, lambda: torch.autograd.grad(loss, actions)[0])


def verify_journal(path, expected_sha, contract_sha):
    """Recompute reservations and lifecycle from externally anchored bytes."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "journal path")
    raw = path.read_bytes()
    require(raw.endswith(b"\n") and hashlib.sha256(raw).hexdigest() == expected_sha, "journal hash")
    events = [json.loads(line) for line in raw.splitlines()]
    previous = None
    for i, event in enumerate(events):
        body = {k: event[k] for k in ("index", "previous", "kind", "payload")}
        require(event["index"] == i and event["previous"] == previous and event["sha256"] == digest(body),
                "journal chain")
        previous = event["sha256"]
    require(len(events) >= 2 and events[0]["kind"] == "header" and events[-1]["kind"] == "terminal",
            "journal lifecycle")
    header = events[0]["payload"]
    require(header["contract_sha256"] == contract_sha, "contract anchor")
    validate(header["contract"], contract_sha)
    check_sources(header["contract"])
    charged = dict.fromkeys(AXES, 0)
    stack, failed, hooks = [], False, False
    for event in events[1:-1]:
        kind, p = event["kind"], event["payload"]
        if kind == "hooks_installed":
            require(not hooks and p["modules"] == sorted(x for names in HOOKS.values() for x in names), "hooks")
            hooks = True
        elif kind in ("attempt", "blocked"):
            require(not failed and p["id"] == event["index"] and p["kind"] in PRIMARY
                    and p["phase"] in PHASES, "attempt identity")
            delta = vector(p["delta"])
            allowed = {PRIMARY[p["kind"]]}
            if p["kind"] in EXTRA:
                allowed.add(EXTRA[p["kind"]])
                require(delta[PRIMARY[p["kind"]]] == 1 and delta[EXTRA[p["kind"]]] > 0, "sample units")
            else:
                require(delta[PRIMARY[p["kind"]]] > 0, "operation units")
            require(all(delta[k] == 0 for k in AXES if k not in allowed), "unexpected units")
            if p["kind"] in HOOKS:
                require(hooks, "missing model hooks")
            over = any(charged[k] + delta[k] > header["contract"]["caps"][k] for k in AXES)
            require(over == (kind == "blocked"), "budget decision")
            if over:
                failed = True
            else:
                for k in AXES:
                    charged[k] += delta[k]
                stack.append(p)
        elif kind == "violation":
            failed = True
        elif kind in ("returned", "failure"):
            require(stack and stack[-1]["id"] == p["id"], "attempt return pairing")
            attempt = stack.pop()
            if kind == "failure":
                failed = True
            else:
                require(not failed and sorted(p["observed_hooks"]) == sorted(HOOKS.get(attempt["kind"], ())),
                        "hook return coverage")
        else:
            raise ValueError("unknown journal event")
    terminal = events[-1]["payload"]
    require(not stack and terminal["charged"] == charged and not terminal["coverage_verified"]
            and not terminal["full_cost_certified"] and not terminal["fairness_proven"]
            and (terminal["status"] == "failed_or_blocked" if failed
                 else terminal["status"] in ("closed_unverified", "failed_or_blocked")), "terminal accounting")
    return {"status": terminal["status"], "charged": charged, "coverage_verified": False,
            "full_cost_certified": False, "journal_sha256": expected_sha}
