"""Development-only common objective and durable accounting primitives.

Not a CUDA runner or an authorization to run one. Caller must authenticate
both initializers, context, target, checkpoint and the host cooperative lock.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import p2_sra_common_optimizer_bridge_20261102_exclusive as bridge
from . import codex_p2_program_model_scope_20261030_v1 as instrumentation

CATEGORIES = ("shared_setup", "composition_inherited", "sra_initialization",
              "common_composition", "common_no_composition_sra", "final_replay")
B0_REPORT_SHA = "d39f8e4f5f292034331b8a60b3fe0682b04d81206fd904200db2295a38e3ef2d"
B0_RUN = "p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_20261102_p2continuation_host01"


def verified_b0():
    """Read-only authentication of the inherited capture, context and winner."""
    from . import p2_shared_capture_launch_20261102_exclusive as shared
    folder = shared.RESULTS / B0_RUN
    verification = shared.audit(folder, B0_REPORT_SHA)
    report = json.loads((folder / "report.json").read_bytes())
    pair_bytes = (folder / "pair.json").read_bytes()
    if (hashlib.sha256(pair_bytes).hexdigest() != report["pair_sha256"]
            or shared.wire.checkpoint.sha(folder / "report.json") != B0_REPORT_SHA):
        raise ValueError("inherited source drift")
    pair = json.loads(pair_bytes)
    rows = [b for b in pair["branches"] if b["arm"] == "prefix_chain"]
    if len(rows) != 1:
        raise ValueError("unique inherited composition branch required")
    raw = bytes.fromhex(shared.wire.reuse.winner_action_hex(rows[0]))
    actions = bridge.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
    return {"verification": verification, "context": pair["context"], "goal": pair["goal"],
            "composition_raw": actions, "composition_sha256": hashlib.sha256(raw).hexdigest(),
            "report_sha256": B0_REPORT_SHA}


class Ledger:
    """Append-only, hash-chained events; each record is fsynced before return."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("xb", buffering=0)
        self.previous = "0" * 64
        self.index = 0
        self.category = None
        self.serial = 0
        self.record("ledger_open", {"unknown": ["FLOPs", "energy", "io_bytes",
                                                  "autograd_engine_internal_calls"]})
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def record(self, kind, payload):
        row = {"index": self.index, "previous_sha256": self.previous,
               "category": self.category, "kind": kind, "payload": payload}
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        line = json.dumps({"record": row, "sha256": digest}, sort_keys=True,
                          separators=(",", ":")).encode() + b"\n"
        if self.stream.write(line) != len(line):
            raise OSError("short ledger write")
        os.fsync(self.stream.fileno())
        self.index += 1
        self.previous = digest
        return digest

    @contextmanager
    def phase(self, category):
        if category not in CATEGORIES or self.category is not None:
            raise ValueError("invalid or nested resource category")
        self.category = category
        try:
            yield
        finally:
            self.category = None

    def call(self, axis, operation):
        if self.category is None or not axis:
            raise ValueError("resource call needs category and axis")
        self.serial += 1
        ident = self.serial
        self.record("attempt", {"id": ident, "axis": axis})
        try:
            result = operation()
        except BaseException as exc:
            self.record("failed", {"id": ident, "axis": axis,
                                   "error_type": type(exc).__name__})
            raise
        self.record("returned", {"id": ident, "axis": axis})
        return result

    def close(self):
        self.stream.close()


def verify_ledger(path):
    previous = "0" * 64
    outstanding = {}
    seen = set()
    counts = {}
    length = 0
    data = Path(path).read_bytes()
    if not data or not data.endswith(b"\n"):
        raise ValueError("empty or truncated ledger")
    for index, line in enumerate(data.splitlines()):
        length += 1
        item = json.loads(line)
        row = item["record"]
        if index == 0 and (row["kind"] != "ledger_open" or row["category"] is not None):
            raise ValueError("missing ledger opening")
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        if row["index"] != index or row["previous_sha256"] != previous or item["sha256"] != digest:
            raise ValueError("ledger chain mismatch")
        previous = digest
        if row["kind"] in ("attempt", "returned", "failed"):
            category, payload = row["category"], row["payload"]
            if category not in CATEGORIES:
                raise ValueError("invalid category")
            key = (category, payload["axis"])
            counts.setdefault(key, {"attempt": 0, "returned": 0, "failed": 0})
            counts[key][row["kind"]] += 1
            ident = payload["id"]
            if row["kind"] == "attempt":
                if ident in seen:
                    raise ValueError("duplicate attempt")
                seen.add(ident)
                outstanding[ident] = key
            elif outstanding.pop(ident, None) != key:
                raise ValueError("unmatched return or failure")
    if not length:
        raise ValueError("empty ledger")
    return {"tail_sha256": previous, "counts": counts, "outstanding": outstanding}


def objective(transition, context, goal, ledger, device):
    """Build shared terminal MSE/gradient callback. Fixture transitions are not evidence."""
    import torch
    state = torch.as_tensor(context["latent_history"], dtype=torch.float32, device=device).reshape(1, 3, 192)
    history = torch.as_tensor(context["action_history"], dtype=torch.float32, device=device).reshape(1, 2, 10)
    target = torch.as_tensor(goal, dtype=torch.float32, device=device).reshape(192)
    if not all(bool(torch.isfinite(t).all()) for t in (state, history, target)):
        raise ValueError("nonfinite context or goal")

    def evaluate(arm, raw, gradient_required):
        if arm not in bridge.ARMS:
            raise ValueError("unknown arm")
        raw = bridge.raw_actions(raw)
        with ledger.phase("common_" + arm):
            action = torch.tensor(raw, dtype=torch.float32, device=device,
                                  requires_grad=gradient_required)
            current, between = state, history
            for packet in range(2):
                current, between = ledger.call(
                    "transition_packet", lambda: transition(current, between, action[packet*5:(packet+1)*5]))
            loss = (current[0, -1] - target).square().mean()
            gradient = ledger.call("autograd_grad", lambda: torch.autograd.grad(loss, action)[0]) if gradient_required else None
            score = float(loss.detach().cpu())
            grad = None if gradient is None else gradient.detach().cpu().numpy().astype(np.float32)
            return score, grad
    return evaluate


def native_objective(model, device, context, goal, ledger):
    """Create CUDA adapter; caller wraps execution in native_model_scope and lock."""
    from . import pusht_full_packet_compiler_20261027 as compiler
    return objective(compiler.make_transition(model, device), context, goal, ledger, device)


def native_model_scope(model, ledger):
    return instrumentation.model_scope(model, ledger)
