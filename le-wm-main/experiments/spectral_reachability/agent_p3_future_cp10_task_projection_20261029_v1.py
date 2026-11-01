"""Future-owned task projection; caller must independently authenticate sources."""
from copy import deepcopy
import hashlib
import numpy as np

from experiments.spectral_reachability import agent_p3_same_time_capability_probe_cuda_20261029_v1 as old

NAME = 'agent_p3_future_cp10_task_projection_20261029_v1'
ORDER = old.ORDER
digest = old.digest  # Original task schema uses the producer's compact JSON codec.


def require(ok, message):
    if not ok: raise ValueError(message)


def project(specs, latents, latent_descriptors):
    """Project five supplied cp10s; no source/receipt or runtime authorization implied."""
    require(len(specs) == len(latents) == len(latent_descriptors) == 5, 'fixed five')
    cp5, endpoints = {}, []
    for index, ((root, role), spec, latent, desc) in enumerate(zip(ORDER, specs, latents, latent_descriptors, strict=True)):
        require(spec['index'] == index and spec['root_slot'] == root and spec['role'] == role, 'fixed order')
        old_task = old.core.validate_task(deepcopy(spec['cp5_task']))
        require(spec['absolute_steps'] == [10, 15, 20] and
                [x['absolute_step'] for x in spec['history_sources']] == [10, 15, 20], 'history chronology')
        require(spec['source']['state'] == spec['source']['snapshot']['state'], 'state snapshot')
        raw = old.core.finite(spec['causal_raw10'], (10, 2), True).astype('<f4')
        require(raw.tobytes().hex() == spec['causal_raw10_bits'], 'raw10 bits')
        mean = np.asarray(old_task['action_mean'], dtype=np.float32)
        std = np.asarray(old_task['action_std'], dtype=np.float32)
        require(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), 'normalizer')
        tokens = ((raw - mean) / std).reshape(2, 10)
        require(tokens.tobytes() == np.asarray(spec['normalized_action_history'], dtype='<f4').tobytes(), 'tokens')
        z = np.asarray(latent)
        require(z.shape == (192,) and z.dtype == np.dtype('float32') and np.isfinite(z).all(), 'latent192 float32')
        require(isinstance(desc, dict) and desc.get('shape') == [192] and desc.get('dtype') == 'float32', 'latent descriptor')
        old.core.hash_value(desc['sha256'])
        identity = dict(namespace=NAME, index=index, root_slot=root, role=role, absolute_steps=[10,15,20],
                        history_sources=deepcopy(spec['history_sources']), raw10_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                        old_cp5_task_sha256=digest(old_task), source_report=deepcopy(spec['source_report']),
                        source_receipt=deepcopy(spec['source_receipt']), encoded_latent=deepcopy(desc))
        context = deepcopy(old_task['root_context'])
        context['latent_history'] = deepcopy(context['latent_history'][1:]) + [z.tolist()]
        context['action_history'] = tokens.tolist()
        task = deepcopy(old_task); task['root_context'] = context
        task['binding'].update(context_sha256=digest(context), task_id=digest([old_task['binding']['task_id'], identity]),
            task_sha256=digest([old_task, context, identity]), task_contract_sha256=digest([NAME, identity]))
        task = old.core.validate_task(task)
        if root in cp5:
            require(old.raw(cp5[root]) == old.raw(old_task), 'same-root cp5 task identity')
        else: cp5[root] = old_task
        endpoints.append(dict(index=index, root=root, role=role, task=task, task_sha256=digest(task),
                              identity=identity, source=deepcopy(spec['source']),
                              base_task_id=spec['base_task_id'], causal_raw10=raw.tolist(),
                              causal_raw10_bits=raw.tobytes().hex()))
    signals = [dict(index=r, time=5, root=r, role='reference', task=cp5[r], task_sha256=digest(cp5[r])) for r in (0,1)]
    signals += [dict(index=i+2, time=10, **{k:e[k] for k in ('root','role','task','task_sha256')}) for i,e in enumerate(endpoints)]
    return dict(schema=NAME+'.projection', signals=signals, endpoints=endpoints,
                source_authority_verified=False, execution_authorized=False, fresh_roots_certified=False,
                scientific_admission=False)
