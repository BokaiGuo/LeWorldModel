"""Independent arithmetic rebuild of saved observed-context chain.

Caller must separately authenticate frame, encoder tensor, physical returns and
timeline. This verifier cannot issue a physical or scientific receipt.
"""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import p3_matched_order_feedback_check_codex_20261029 as structural

seam = structural.seam
need = seam.need


def audit(root, warm_raw10, result, numeric_events, observed_latents):
    structural.check(result,numeric_events)
    root = seam.core.validate_task(root)
    need(result['root_sha256'] == seam.core.digest(root), 'root task binding')
    need(type(observed_latents) is list and len(observed_latents) == 4, 'four observed encoder outputs')
    context = deepcopy(root['root_context'])
    latent = np.asarray(context['latent_history'],dtype=np.float32)
    action = np.asarray(context['action_history'],dtype=np.float32)
    warm = np.asarray(warm_raw10,dtype=np.float32)
    mean = np.asarray(root['action_mean'],dtype=np.float32); scale = np.asarray(root['action_std'],dtype=np.float32)
    need(latent.shape == (3,192) and action.shape == (2,10) and warm.shape == (10,2) and
         np.isfinite(warm).all() and (np.abs(warm)<=1).all() and (scale>0).all(), 'initial actual history ABI')
    initial = ((warm-mean)/scale).reshape(2,10).astype(np.float32)
    need(initial.tobytes() == action.tobytes(), 'full warm10 matches original model history')
    contexts = [e for e in numeric_events if e['kind'] == 'actual_context']
    for i,(packet,encoded,event) in enumerate(zip(result['packets'],observed_latents,contexts,strict=True)):
        need(type(encoded) is dict and set(encoded) == {'boundary','latent','rgb_sha256'} and
             encoded['boundary'] == packet['end'] == event['end'] and
             encoded['rgb_sha256'] == event['rgb_sha256'], 'same observed boundary/RGB hash')
        z = np.asarray(encoded['latent'])
        need(z.dtype == np.float32 and z.shape == (192,) and np.isfinite(z).all(), 'observed projected latent')
        raw = np.asarray(packet['raw'],dtype=np.float32)
        need(raw.shape == (5,2) and np.isfinite(raw).all() and (np.abs(raw)<=1).all(), 'actual returned raw5')
        token = ((raw-mean)/scale).reshape(10).astype(np.float32)
        latent = np.concatenate((latent[1:],z[None]),axis=0)
        action = np.stack((action[-1],token),axis=0)
        context['latent_history'] = latent.tolist(); context['action_history'] = action.tolist()
        need(seam.core.digest(seam.core.context(context)) == event['context_sha256'], 'independent observed context')
        if packet['end'] == 5*result['blocks'][0]['horizon']:
            need(event['context_sha256'] == result['blocks'][1]['context_sha256'], 'second block actual context')
    return dict(status='observed_context_arithmetic_only',boundaries=4,
                physical_authenticated=False,model_authenticated=False,scientific_pass=False)
