"""Paired multi-start cp5-feedback planning kernel; not a physical launcher.

The caller must authenticate and persist real RGB captures, CUDA model identity,
device/lock and native physical costs. The kernel never imports an environment.
"""
from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from . import p2_sra_common_optimizer_bridge_20261102_exclusive as optimizer
from . import p3_goal_patch_consistent_compiler_20261027_v2 as normalization

NAME = "p2_prefix_feedback_common_budget_20261102_codexexclusive"
ARMS = optimizer.ARMS
N = 16
LIMITS = {"start_score": 288, "start_gradient": 256, "start_transition": 576,
          "feedback_capture": 32, "feedback_encode": 32,
          "suffix_score": 288, "suffix_gradient": 256, "suffix_transition": 288}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def contract():
    return {"schema": NAME + ".contract_v1", "candidates": N, "arms": list(ARMS),
            "bank": "same 16 float32 10x2 inputs in fixed order for both arms",
            "first_stage": "same two-packet objective and projected optimizer; conditional first packet frozen, joint both packets variable",
            "feedback": "fresh physical cp5 RGB/history for every candidate and arm, CUDA encoding from RGB only",
            "second_stage": "same one-packet objective and optimizer from each own actual cp5; first packet frozen in both arms",
            "selection": "minimum returned cp5 model terminal score, ties candidate then iterate; no physical scalar metrics",
            "caps": LIMITS, "score_steps": 9, "gradient_steps": 8, "stop_policy":
            "any early native stop or capture/encode failure aborts pair; retain all attempts, no imputation",
            "deployment": "reconstruct selected prefix, require exact RGB/history binding before suffix; separate adapter",
            "historical_sra": False, "equal_total_compute": False, "independent_confirmation": False,
            "physical_adapter_ready": False, "P2_admission": False}


class Meter:
    """Per-session attempted and returned calls with fail-stop nested accounting."""
    def __init__(self, emit, sync=lambda: None, clock=time.monotonic_ns):
        need(callable(emit) and callable(sync) and callable(clock), "meter callbacks")
        self.emit, self.sync, self.clock = emit, sync, clock
        self.attempted = dict.fromkeys(LIMITS, 0)
        self.returned = dict.fromkeys(LIMITS, 0)
        self.failed = False
        self.serial = 0

    def call(self, kind, fn, **context):
        need(not self.failed and kind in LIMITS and callable(fn), "meter route")
        if self.attempted[kind] >= LIMITS[kind]:
            self.failed = True
            raise ValueError("resource cap: " + kind)
        self.serial += 1
        ident = self.serial
        self.attempted[kind] += 1
        self.emit({"kind": "resource_attempt", "resource": kind, "id": ident, **context})
        status = "failed"
        elapsed = None
        try:
            self.sync()
            begin = self.clock()
            try:
                value = fn()
                self.sync()
            finally:
                elapsed = self.clock() - begin
                need(type(elapsed) is int and elapsed >= 0, "monotonic timing")
            if self.failed:
                raise RuntimeError("nested resource failed")
            self.returned[kind] += 1
            status = "returned"
            return value
        except BaseException:
            self.failed = True
            raise
        finally:
            self.emit({"kind": "resource_finish", "resource": kind, "id": ident,
                       "status": status, "elapsed_ns": elapsed, "inclusive": True, **context})

    def finish(self):
        need(not self.failed and self.attempted == LIMITS and self.returned == LIMITS, "incomplete paired budget")
        return {"attempted": deepcopy(self.attempted), "returned": deepcopy(self.returned),
                "flops": None, "equal_total_compute": False}


def validate_capture(value, arm, index, prefix, root_sha, task_id, base_history, base_rgb_sha256):
    """Only RGB and public action history flow to the encoder, never native state."""
    need(type(value) is dict and set(value) == {
        "arm", "candidate_index", "root_sha256", "task_id", "prefix_hex",
        "rgb_history", "action_history", "capture_receipt_sha256", "terminated", "truncated"
    }, "capture ABI; privileged state forbidden")
    need(value["arm"] == arm and value["candidate_index"] == index and
         value["root_sha256"] == root_sha and value["task_id"] == task_id and
         value["prefix_hex"] == prefix.tobytes().hex(), "root/action/candidate binding")
    need(type(value["terminated"]) is bool and type(value["truncated"]) is bool,
         "stop flags")
    need(not value["terminated"] and not value["truncated"], "native stop before feedback")
    pixels = np.asarray(value["rgb_history"])
    actions = np.asarray(value["action_history"])
    need(pixels.shape == (3, 224, 224, 3) and pixels.dtype == np.uint8
         and actions.shape == (2, 10) and actions.dtype == np.float32 and np.isfinite(actions).all(),
         "actual RGB/history ABI")
    base = np.asarray(base_history, dtype="<f4")
    need(base.shape == (2, 10) and np.isfinite(base).all(), "initial action history")
    expected = np.stack((base[1], normalization.normalized_history(prefix)[0])).astype("<f4")
    need(actions.astype("<f4").tobytes() == expected.tobytes(),
         "actual history does not match prefix and initial context")
    rgb_sha = [sha(x.tobytes()) for x in pixels]
    need(isinstance(base_rgb_sha256, (list, tuple)) and len(base_rgb_sha256) == 2
         and rgb_sha[:2] == list(base_rgb_sha256), "initial RGB history mismatch")
    receipt = value["capture_receipt_sha256"]
    need(isinstance(receipt, str) and len(receipt) == 64 and all(c in "0123456789abcdef" for c in receipt),
         "capture receipt digest")
    return {"rgb_history": pixels.copy(), "action_history": actions.copy(),
            "rgb_sha256": rgb_sha,
            "action_history_sha256": sha(actions.astype("<f4").tobytes()),
            "capture_receipt_sha256": receipt}


def validate_context(context, expected_history):
    need(type(context) is dict and set(context) == {"layout", "latent_history", "action_history"},
         "encoded context ABI")
    z = np.asarray(context["latent_history"])
    actions = np.asarray(context["action_history"])
    need(context["layout"] == "states_with_between_state_actions_v1" and z.shape == (3, 192)
         and np.isfinite(z).all() and actions.shape == (2, 10) and np.isfinite(actions).all()
         and np.asarray(actions, dtype="<f4").tobytes() == expected_history.astype("<f4").tobytes(),
         "RGB encoded context/history")


def run_case(bank, root_sha, task_id, base_action_history, base_rgb_sha256, *, evaluate_start, capture, encode,
             evaluate_suffix, emit, sync=lambda: None):
    """Two optimizer stages; callbacks must provide authenticated native backends.

    Capture trial prefixes are exploratory paid physical operations, not winner
    execution. No physical outcome/reward/state enters an optimizer callback.
    Failures stop the whole root; the caller must persist emitted events.
    """
    need(isinstance(bank, (tuple, list)) and len(bank) == N and callable(emit), "fixed bank")
    need(all(callable(f) for f in (evaluate_start, capture, encode, evaluate_suffix)), "backend ABI")
    meter = Meter(emit, sync)
    try:
        result = _run_case(bank, root_sha, task_id, base_action_history, base_rgb_sha256, evaluate_start=evaluate_start,
                           capture=capture, encode=encode, evaluate_suffix=evaluate_suffix, emit=emit, meter=meter)
    except BaseException as exc:
        emit({"kind": "root_aborted", "error_type": type(exc).__name__, "message": str(exc)[:500],
              "attempted": deepcopy(meter.attempted), "returned": deepcopy(meter.returned),
              "candidate_denominator": N, "winners": None})
        raise
    emit({"kind": "root_complete", "candidate_denominator": N, "costs": result["costs"]})
    return result


def _run_case(bank, root_sha, task_id, base_action_history, base_rgb_sha256, *, evaluate_start, capture, encode,
              evaluate_suffix, emit, meter):
    rows = []
    for i, raw in enumerate(bank):
        need(isinstance(raw, bytes) and len(raw) == 80, "bank bytes")
        action = optimizer.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
        def start_eval(arm, x, gradient):
            def evaluate():
                return evaluate_start(arm, x.copy(), gradient, meter)
            score, grad = meter.call("start_score", evaluate, arm=arm, candidate=i)
            if gradient and arm == ARMS[0]:
                grad = grad.copy()
                grad[:5] = 0
            return score, grad
        start = optimizer.optimize_pair({arm: action.copy() for arm in ARMS}, start_eval,
            lambda e: emit({"stage": "start", "candidate": i, **e}))
        observed, contexts, initial = {}, {}, {}
        for arm, entry in zip(ARMS, start["rows"], strict=True):
            chosen = optimizer.raw_actions(np.frombuffer(bytes.fromhex(entry["raw_float32_hex"]),
                                                         dtype="<f4").reshape(10, 2))
            prefix = chosen[:5].copy()
            if arm == ARMS[0]:
                need(prefix.tobytes() == action[:5].tobytes(), "conditional prefix changed")
            captured = meter.call("feedback_capture", lambda: capture(arm, i, prefix.copy()),
                                  arm=arm, candidate=i)
            obs = validate_capture(captured, arm, i, prefix, root_sha, task_id, base_action_history, base_rgb_sha256)
            context = meter.call("feedback_encode", lambda: encode(
                obs["rgb_history"].copy(), obs["action_history"].copy()), arm=arm, candidate=i)
            validate_context(context, obs["action_history"])
            contexts[arm], initial[arm] = context, chosen
            observed[arm] = {"prefix_sha256": sha(prefix.tobytes()), "rgb_sha256": obs["rgb_sha256"],
                             "action_history_sha256": obs["action_history_sha256"],
                             "capture_receipt_sha256": obs["capture_receipt_sha256"],
                             "context_sha256": sha(canonical(context))}
            emit({"kind": "feedback_bound", "candidate": i, "arm": arm, **observed[arm]})
        def suffix_eval(arm, x, gradient):
            need(x[:5].tobytes() == initial[arm][:5].tobytes(), "post-feedback prefix changed")
            score, grad = meter.call("suffix_score", lambda: evaluate_suffix(
                arm, contexts[arm], x.copy(), gradient, meter), arm=arm, candidate=i)
            if gradient:
                grad = grad.copy()
                grad[:5] = 0
            return score, grad
        suffix = optimizer.optimize_pair(initial, suffix_eval,
            lambda e: emit({"stage": "suffix", "candidate": i, **e}))
        rows.append({"candidate_index": i, "bank_sha256": sha(raw), "start": start["rows"],
                     "feedback": observed, "suffix": suffix["rows"]})
    counts = meter.finish()
    winners = {}
    for arm in ARMS:
        score, index, iteration, entry = min(
            ((e["model_score"], row["candidate_index"], e["winner_iteration"], e)
             for row in rows for e in row["suffix"] if e["arm"] == arm), key=lambda x: x[:3])
        winners[arm] = {"candidate_index": index, "iteration": iteration, "model_score": score,
                        "raw_float32_hex": entry["raw_float32_hex"],
                        "feedback": rows[index]["feedback"][arm]}
    return {"schema": NAME + ".result_v1", "rows": rows, "winners": winners,
            "costs": counts, "candidate_denominator": N, "physical_adapter_authenticated": False,
            "equal_total_compute": False, "independent_confirmation": False, "P2_admission": False}


def native_evaluators(root_context, goal, transition, device):
    """Real differentiable transition adapter. Caller must authenticate frozen CUDA."""
    import torch
    need(torch.device(device).type == "cuda", "CUDA required; no CPU neural inference")
    need(callable(transition), "native transition")
    def evaluate(stage, context, x, gradient, meter):
        state = torch.as_tensor(context["latent_history"], dtype=torch.float32, device=device).reshape(1, 3, 192)
        history = torch.as_tensor(context["action_history"], dtype=torch.float32, device=device).reshape(1, 2, 10)
        target = torch.as_tensor(goal, dtype=torch.float32, device=device).reshape(192)
        actions = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=gradient)
        if stage == "start":
            state, history = meter.call("start_transition", lambda: transition(state, history, actions[:5]))
        state, history = meter.call(stage + "_transition",
                                    lambda: transition(state, history, actions[5:]))
        loss = (state[0, -1] - target).square().mean()
        grad = meter.call(stage + "_gradient", lambda: torch.autograd.grad(loss, actions)[0]) if gradient else None
        return float(loss.detach().cpu()), None if grad is None else grad.detach().cpu().numpy().astype(np.float32)
    return (lambda arm, x, grad, meter: evaluate("start", root_context, x, grad, meter),
            lambda arm, context, x, grad, meter: evaluate("suffix", context, x, grad, meter))
