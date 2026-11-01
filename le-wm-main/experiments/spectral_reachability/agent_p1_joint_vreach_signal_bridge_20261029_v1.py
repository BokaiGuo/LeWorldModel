"""Saved H1 native-signal adapter; this is not a production runner."""
from copy import deepcopy

from experiments.spectral_reachability import agent_p3_parent_native_adapter_20261029_v1 as parent
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native

core = parent.core


def from_transcript(root_task, parent_context, transcript):
    """Audit a complete native call and return the joint-core signal ABI.

    The caller remains responsible for authenticating the root task and the
    prefix which produced parent_context.  This function does not run a model.
    """
    root = core.validate_task(deepcopy(root_task))
    local = parent.parent_task(root, deepcopy(parent_context))
    core.require(type(transcript) is list and len(transcript) >= 2, 'complete signal transcript')
    start = transcript[0]
    core.require(start.get('method') == 'signal' and start.get('task') == local,
                 'exact root-derived parent task')
    req = core.request(local, dict(context=local['root_context']),
                       operation='compile_finite_horizon_J', horizon=1)
    core.require(start.get('request') == req, 'exact H1 parent request')
    core.require(native.audit_native(transcript) is True, 'complete native H1 audit')
    boundaries = [e for e in transcript[1:-1] if e['kind'] == 'boundary' and e.get('phase') == 'nominal']
    core.require(len(boundaries) == 1 and boundaries[0]['step'] == 0, 'one H1 nominal boundary')
    b = boundaries[0]
    core.keys(b['arrays'], ('latent', 'history'))
    latent, history = (native.signal.unblob(b['arrays'][k]) for k in ('latent', 'history'))
    core.require(latent.shape == (1, 3, 192) and history.shape == (1, 2, 10), 'full boundary shape')
    context = dict(local['root_context'], latent_history=latent[0].tolist(), action_history=history[0].tolist())
    core.checked_path(local, local['root_context'], [[0., 0.]] * 5, [context])
    response = transcript[-1]['result']
    core.require(context['latent_history'][-1] == response['nominal'], 'nominal response matches boundary')
    return dict(nominal_context=context, jacobian=deepcopy(response['jacobian']))
