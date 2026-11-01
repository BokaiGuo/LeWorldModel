"""Temporary Q-weighted root action bank from saved Jacobian; no outcome inputs.

Fixed atanh-coordinate radii are prospective calibration choices, not guarantees
of trust-region validity or physical observability. No training or neural work.
"""
import numpy as np


def synthesize(*, jacobian, nominal_terminal, goal, projection):
    arrays = []
    for value in (jacobian, nominal_terminal, goal, projection):
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fi' or not np.isfinite(raw).all():
            raise ValueError('finite real arrays required')
        arrays.append(raw.astype(np.float64))
    jac, terminal, target, q = arrays
    if (jac.ndim != 2 or jac.shape[1] != 2 or terminal.shape != (jac.shape[0],)
            or target.shape != terminal.shape or q.shape != (len(target), len(target))):
        raise ValueError('single two-coordinate prefix Jacobian and aligned Q/goal required')
    response = q @ jac
    left, singular, right = np.linalg.svd(response, full_matrices=False)
    # Remove SVD sign ambiguity without conditioning on observed outcomes.
    for i in range(len(singular)):
        if right[i, np.argmax(np.abs(right[i]))] < 0:
            right[i] *= -1
            left[:, i] *= -1
    residual = q @ (target-terminal)
    progress = right.T @ ((singular/(singular**2+.01))*(left.T@residual))
    original_norm = float(np.linalg.norm(progress))
    if original_norm > .75:
        progress *= .75/original_norm
    params = [np.zeros(2), progress]
    kinds = ['zero', 'root_dls_progress']
    for radius in (.25, .75):
        for index, direction in enumerate(right):
            for sign in (-1, 1):
                params.append(sign*radius*direction)
                kinds.append(f'spectral_radius_{radius}_mode_{index}_sign_{sign}')
    bank = np.stack(params)[:, None, :]
    return dict(schema='pusht_task_spectral_bank.v1', params=bank.tolist(), kinds=kinds,
        singular_values=singular.tolist(), right_vectors=right.tolist(),
        unbounded_progress_norm=original_norm, progress_norm_cap=.75, radii=[.25,.75],
        damping=.01, action_coordinates='atanh command; five identical raw primitive steps',
        additional_training=False, physical_labels_used=False,
        mechanism_verified=False, P1_admission=False)
