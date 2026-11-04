"""Development bridge to one frozen native objective; no automatic launch.

CPU fixtures may use make_objective. native_session acquires the established
host lock before loading CUDA, but is not itself a production authorization,
source seal, full resource cap, or composition-mechanism implementation.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from . import p2_same_bank_common_optimizer_20261102_exclusive as bank
from . import p2_common_objective_ledger_20261102_exclusive as objective
from . import p2_sra_common_optimizer_bridge_20261102_exclusive as optimizer
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

LOCK = bank.ROOT / "results/e3_p2_host_cuda_cooperative.lock"
PROOF = bank.ROOT / "results/p2_b0_isolated_consume_20261102_exclusive_run01.json"
PROOF_SHA = "9f0b42299b897905677a6a5f33082a3e228a846cbd794a7b6a3ff30937a2faaa"
CAPS = {"transition_packet": 288, "autograd_grad": 128,
        "action_encoder": 288, "predict": 288, "predictor": 288, "pred_proj": 288}
CALLBACK_CAPS = {"score": 144, "gradient": 128}


def check_normalizer(metadata, mean, std):
    """Require the runtime normalizer to equal the sealed B0 normalizer."""
    expected_mean = np.asarray(metadata["mean"], dtype=np.float32)
    expected_std = np.asarray(metadata["std"], dtype=np.float32)
    actual_mean, actual_std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if (expected_mean.shape != (2,) or expected_std.shape != (2,) or
            not np.isfinite(expected_mean).all() or not np.isfinite(expected_std).all() or
            not (expected_std > 0).all() or
            not np.array_equal(expected_mean, actual_mean) or
            not np.array_equal(expected_std, actual_std)):
        raise ValueError("runtime action normalizer mismatch")


def verified_input():
    fixed = bank.load_real()
    raw = PROOF.read_bytes()
    if bank.sha(raw) != PROOF_SHA:
        raise ValueError("isolated B0 proof SHA")
    proof = json.loads(raw)
    if (proof["status"] != "consumer_returned_success" or
            proof["consumer_result"]["status"] != "verified_saved_pair_known_root_not_admitted" or
            proof["consumer_result"]["report_sha256"] !=
            "d39f8e4f5f292034331b8a60b3fe0682b04d81206fd904200db2295a38e3ef2d"):
        raise ValueError("isolated B0 proof")
    return fixed


class CappedMeter:
    """Hard attempt caps on explicitly instrumented calls, not FLOPs or walltime."""
    def __init__(self, ledger):
        self.ledger = ledger
        self.counts = {a: {k: 0 for k in CAPS} for a in optimizer.ARMS}

    def phase(self, category):
        return self.ledger.phase(category)

    def call(self, axis, operation):
        category = self.ledger.category
        arm = category.removeprefix("common_") if isinstance(category, str) else None
        if arm not in optimizer.ARMS or axis not in CAPS:
            raise ValueError("unrecognized operation category or axis")
        count = self.counts[arm][axis]
        if count >= CAPS[axis]:
            raise ValueError("instrumented attempt cap exceeded")
        self.counts[arm][axis] = count + 1
        return self.ledger.call(axis, operation)


def make_objective(fixed, transition, meter, device):
    """Same context, goal, raw action interface and MSE for either arm."""
    if fixed != verified_input():
        raise ValueError("fixed bank/context/source changed")
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    # The existing core enforces (3,192), (2,10), (10,2) and computes autograd.
    evaluate = objective.objective(transition, context, goal, meter, device)
    attempted = {arm: {"score": 0, "gradient": 0} for arm in optimizer.ARMS}

    def callback(arm, supplied_context, supplied_goal, raw, gradient_required):
        if arm not in optimizer.ARMS or type(gradient_required) is not bool:
            raise ValueError("arm or gradient flag")
        if (bank.canonical(supplied_context) != fixed["context_json"] or
                bank.canonical(supplied_goal) != fixed["goal_json"]):
            raise ValueError("context or goal mismatch")
        optimizer.raw_actions(raw)
        counts = attempted[arm]
        if counts["score"] >= CALLBACK_CAPS["score"] or (
                gradient_required and counts["gradient"] >= CALLBACK_CAPS["gradient"]):
            raise ValueError("callback attempt cap exceeded")
        counts["score"] += 1
        counts["gradient"] += int(gradient_required)
        with meter.phase("common_" + arm):
            meter.ledger.record("objective_attempt", {"arm": arm, "gradient": gradient_required})
        try:
            score, grad = evaluate(arm, raw, gradient_required)
            if (type(score) is not float or not np.isfinite(score) or
                    (gradient_required and (not isinstance(grad, np.ndarray) or
                     grad.shape != (10, 2) or grad.dtype != np.float32 or not np.isfinite(grad).all())) or
                    (not gradient_required and grad is not None)):
                raise ValueError("invalid native objective return")
        except BaseException as exc:
            with meter.phase("common_" + arm):
                meter.ledger.record("objective_failed", {"arm": arm, "error_type": type(exc).__name__})
            raise
        with meter.phase("common_" + arm):
            meter.ledger.record("objective_returned", {"arm": arm, "gradient": gradient_required})
        return score, grad
    callback.attempted = attempted
    return callback


@contextmanager
def native_session(fixed, ledger):
    """Host-only adapter. Must be embedded in a separately authorized runner."""
    if fixed != verified_input():
        raise ValueError("fixed source changed")
    with locking.gpu_lock(LOCK):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from . import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint
        from . import preflight_pusht_d2p_p0 as p0
        from . import pusht_full_packet_compiler_20261027 as compiler

        check_normalizer(checkpoint.native_metadata(), p0.PUBLIC_ACTION_MEAN, p0.PUBLIC_ACTION_STD)
        for path, expected in checkpoint.MODELS.items():
            if checkpoint.sha(path) != expected:
                raise ValueError("checkpoint file drift")
        device = torch.device("cuda:0")
        model = load_compatible_pretrained(str(checkpoint.POLICY), device=device).eval().requires_grad_(False)
        checkpoint.frozen(model)
        if checkpoint.tensor_hash(model) != checkpoint.MODEL_TENSOR_SHA:
            raise ValueError("model tensor hash")
        meter = CappedMeter(ledger)
        transition = compiler.make_transition(model, device)
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
                with objective.native_model_scope(model, meter):
                    yield make_objective(fixed, transition, meter, device), meter
        finally:
            try:
                torch.cuda.synchronize(device)
                checkpoint.frozen(model)
                if checkpoint.tensor_hash(model) != checkpoint.MODEL_TENSOR_SHA:
                    raise ValueError("model tensor drift")
                for path, expected in checkpoint.MODELS.items():
                    if checkpoint.sha(path) != expected:
                        raise ValueError("checkpoint file drift")
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def run_schedule(fixed, callback, emit):
    """Ordered all-candidate schedule; explicit caller owns persistent events."""
    if fixed != verified_input() or not callable(callback) or not callable(emit):
        raise ValueError("source or callback")
    rows = []
    for i, raw in enumerate(fixed["bank"]):
        action = optimizer.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
        def evaluate(arm, x, gradient):
            return callback(arm, json.loads(fixed["context_json"]),
                            json.loads(fixed["goal_json"]), x, gradient)
        result = optimizer.optimize_pair({arm: action.copy() for arm in optimizer.ARMS},
                                         evaluate, lambda event: emit({"candidate_index": i, **event}))
        rows.append({"index": i, "initial_sha256": bank.sha(raw), "arms": result["rows"]})
    return {"rows": rows, "fixture_or_development_only": True,
            "equal_total_compute": False, "composition_mechanism_authenticated": False,
            "production_execution_authorized": False, "P1_admission": False, "P2_admission": False}
