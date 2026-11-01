"""cp5 setup with native initial-success stop; no production authority."""
from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v1 as draft


def setup(*args, **kwargs):
    """Run source setup, and never dispatch an action after initial success.

    The predicate is the native evaluator.  Success at control5 is recorded
    without fabricating an environment step; no planner is called.
    """
    row=kwargs['row'];store=kwargs['store'];env=args[0] if args else kwargs['env']
    result=draft.setup(*args,**kwargs)
    if result is None:return None
    state=env.unwrapped._get_obs()
    success,distance=env.unwrapped.eval_state(result['goal'],state)
    draft.need(type(success) in (bool, draft.np.bool_) and draft.np.isfinite(distance),
               'finite native initial success/distance')
    row.update(initial_success=bool(success),initial_success_control=5 if success else None,
               initial_native_distance=float(distance))
    store.emit('initial_native_evaluation',episode_id=row['episode_id'],boundary=5,
               success=bool(success),distance=float(distance))
    if success:
        row.update(status='native_initial_success_control5',native_success=True,
                   terminated=None,truncated=None)
        return None
    return result
