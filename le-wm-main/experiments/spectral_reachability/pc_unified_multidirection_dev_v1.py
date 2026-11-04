"""Single development entry for fixed-terminal multidirection v1 decisions."""
import hashlib
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_multidirection_feedback_v1 as feedback
from experiments.spectral_reachability import pc_multidirection_successor_compiler_v1 as initial
from experiments.spectral_reachability import pc_response_diverse_compiler_v1 as response
from experiments.spectral_reachability import pc_cem_scale_floor_v1 as suffix

METHOD = "multidirection_v1_development"
CAP, PACKET = 480, 5


def _source_hashes():
    modules = (initial, feedback, response, suffix, initial.prior, initial.prior.policy)
    files = {Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules)}
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


class _Meter:
    def __init__(self, predictor):
        self.predictor = predictor
        self.token_samples = self.calls = self.samples = 0
        self.trace = []

    def predict(self, actions):
        requested = np.asarray(actions, dtype=float)
        if (requested.ndim != 3 or requested.shape[0] < 1 or requested.shape[1] not in (5, 10, 15)
                or requested.shape[2] != 2 or not np.isfinite(requested).all() or np.any(np.abs(requested) > 1)):
            raise ValueError("invalid normalized action batch")
        tokens = len(requested) * requested.shape[1] // PACKET
        if self.token_samples + tokens > CAP:
            raise ValueError("prediction token cap exceeded")
        positions, scores = self.predictor(requested)
        positions, scores = np.asarray(positions), np.asarray(scores)
        if positions.shape != (len(requested), 2) or scores.shape != (len(requested),):
            raise ValueError("invalid prediction response shape")
        if not np.issubdtype(positions.dtype, np.floating) or not np.issubdtype(scores.dtype, np.floating):
            raise ValueError("model predictions must be floating point")
        if not np.isfinite(positions).all() or not np.isfinite(scores).all():
            raise ValueError("nonfinite model prediction")
        self.token_samples += tokens
        self.calls += 1
        self.samples += len(requested)
        self.trace.append({"requested_actions": requested.tolist(),
                           "model_raw_float32_actions": requested.astype(np.float32).tolist(),
                           "positions": positions.tolist(), "scores": scores.tolist(), "tokens": tokens})
        return positions, scores


def compile_decision(*, root, step, target, predict, context_id, carry=None):
    """Compile one bounded decision; provenance validation is caller-owned."""
    if not isinstance(root, int) or isinstance(root, bool) or root < 0:
        raise ValueError("root must be a nonnegative integer")
    if step not in (0, 1, 2) or isinstance(step, bool):
        raise ValueError("step must be 0, 1 or 2")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target must be a finite 2-vector")
    if not isinstance(context_id, str) or not context_id.strip() or not callable(predict):
        raise ValueError("context_id and prediction callback required")
    horizon = 15 - PACKET * step
    if step == 0:
        if carry is not None:
            raise ValueError("initial decision cannot accept carry")
    else:
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (horizon, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
            raise ValueError("invalid carry for remaining horizon")
    meter = _Meter(predict)
    hashes = _source_hashes()
    if step == 0:
        result = initial.decide(root, target, meter.predict, meter)
    else:
        result = feedback.decide(root, step, target, meter.predict, meter, carry)
    chain = np.asarray(result["selected"]["chain"], dtype=float)
    if chain.shape != (horizon, 2) or not np.isfinite(chain).all() or np.any(np.abs(chain) > 1):
        raise ValueError("invalid selected chain")
    if hashes != _source_hashes():
        raise RuntimeError("compiler sources changed during planning")
    return {"method": METHOD, "root": root, "step": step, "horizon": horizon, "context_id": context_id,
            "target": target.tolist(), "input_carry": None if carry is None else carry.tolist(),
            "core_result": result, "dispatched_actions": chain[:PACKET].tolist(),
            "remaining_tail": chain[PACKET:].tolist(),
            "counts": {"prediction_calls": meter.calls, "trajectory_samples": meter.samples,
                       "prediction_token_samples": meter.token_samples, "token_cap": CAP},
            "prediction_trace": meter.trace, "compiler_source_sha256": hashes}
