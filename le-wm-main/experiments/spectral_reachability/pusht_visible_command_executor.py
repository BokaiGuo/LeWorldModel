"""Actual per-primitive RGB guard integration, not a recovery controller."""
import numpy as np
from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard, ObservationUnavailable


def execute(env, commands, *, root_id, agent_rgb, on_observation=None):
    if isinstance(commands, list) and any(isinstance(x, bool) for row in commands for x in row):
        raise ValueError('boolean commands forbidden')
    commands = np.asarray(commands)
    if (commands.dtype.kind not in 'fi' or commands.ndim != 2 or commands.shape[1] != 2
            or not 1 <= len(commands) <= 10 or not np.isfinite(commands).all() or np.any(np.abs(commands) > 1)):
        raise ValueError('bounded nonempty ten-step raw program required')
    commands = commands.astype(np.float64)
    guard = RgbObservabilityGuard(root_id=root_id, agent_rgb=agent_rgb)
    completed = attempted = 0
    rgb = np.asarray(env.render()).copy()
    status, error = 'completed', None
    try:
        guard.observe(rgb, primitive_step=0)
        for command in commands:
            authorized = guard.dispatch(command)
            attempted += 1
            result = env.step(authorized)
            if len(result) != 5:
                raise ValueError('Gymnasium five-field transition required')
            completed += 1
            rgb = np.asarray(env.render()).copy()
            # Callback is evaluator bookkeeping only, never a planning input.
            if on_observation is not None:
                on_observation(completed, command.copy(), result, rgb.copy())
            guard.observe(rgb, primitive_step=completed)
            if result[2] or result[3]:
                status = 'environment_terminated'
                break
    except ObservationUnavailable as exc:
        status, error = 'observation_unavailable', str(exc)
    except Exception as exc:
        status, error = 'execution_failed', f'{type(exc).__name__}: {exc}'
    return {'status': status, 'error': error, 'attempted_commands': attempted,
            'completed_commands': completed, 'planned_commands': len(commands),
            'guard': guard.receipt(), 'terminal_rgb': rgb,
            'remaining_commands_not_executed': len(commands)-completed}
