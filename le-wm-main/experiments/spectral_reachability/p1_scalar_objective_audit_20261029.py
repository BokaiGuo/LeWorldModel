"""Independent numerical check of the two-packet native AD scalar objective.

This complements, but does not replace, the native IO and gradient audit.  It
recomputes the intended identity-Q mean squared error from the saved endpoint
and goal, without running a model or changing any sealed producer artifact.
"""
from __future__ import annotations

import math
import numpy as np


def audit_arm(task, result):
    """Check both AD losses for all eight iterations; reject missing data.

    Float32 subtraction and squaring are reconstructed, while reduction uses
    float64; the tolerance allows the order-dependent float32 CUDA reduction.
    It is not an independent proof of the gradient or model execution.
    """
    goal = np.asarray(task['goal'], dtype=np.float32)
    if goal.shape != (192,) or not np.isfinite(goal).all():
        raise ValueError('finite 192-element goal required')
    gradients = result['gradients']
    if not isinstance(gradients, list) or len(gradients) != 8:
        raise ValueError('complete eight-iteration gradient ledger required')
    largest_error = 0.0
    for iteration, row in enumerate(gradients, 1):
        if row['iteration'] != iteration:
            raise ValueError('gradient iteration order mismatch')
        for name in ('stage', 'suffix'):
            entry = row[name]
            endpoint = np.asarray(entry['context']['latent_history'][-1], dtype=np.float32)
            if endpoint.shape != goal.shape or not np.isfinite(endpoint).all():
                raise ValueError('finite 192-element AD endpoint required')
            loss = entry['native_float32_loss']
            if isinstance(loss, bool) or not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0:
                raise ValueError('finite nonnegative native scalar required')
            with np.errstate(over='ignore', invalid='ignore'):
                delta = np.subtract(endpoint, goal, dtype=np.float32)
                squares = np.multiply(delta, delta, dtype=np.float32)
            if not np.isfinite(squares).all():
                raise ValueError('float32 objective overflow')
            expected = float(np.mean(squares, dtype=np.float64))
            error = abs(float(loss) - expected)
            # 192 nonnegative terms: this exceeds the usual gamma_192 bound
            # for an IEEE float32 reduction, without masking material errors.
            bound = 3e-5 * expected + float(np.finfo(np.float32).tiny)
            if error > bound:
                raise ValueError(f'{name} iteration {iteration}: native scalar does not match goal/endpoint MSE')
            largest_error = max(largest_error, error)
    return {'checked_scalar_losses': 16, 'max_absolute_error': largest_error,
            'scope': 'saved endpoint/goal identity-Q MSE only', 'P1_admission': False}
