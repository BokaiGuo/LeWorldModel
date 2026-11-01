"""Saved-source-bound integer proof; no model, physics or production dispatch."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'experiments/spectral_reachability/agent_p3_c3_common_cap_cuda_20261029_v1.py': '7419b72f12ea81f90703d2c075143aa46f07c8fbe47ba0313c488239797520a7',
    'experiments/spectral_reachability/agent_p3_c3_k2_host_engine_20261029_v1.py': 'c4bf1b3f4b7ae06ebeb6490337aed6d3b0ce4a7e3bec6a06005766e4aa9415c6',
    'innerdocs/agent_p3_c3_common_cap_scheduler_protocol_20261029_v1.md': '0ea4114c0bc88f780f2a788644ddc24851e7ba32e129029350cc5c98bad2255d',
}
AXES = ('predict_API','predict_samples','predict_context_tokens','action_encoder_calls',
 'action_encoder_samples','action_encoder_context_tokens','predictor_calls','predictor_samples',
 'predictor_context_tokens','pred_proj_calls','pred_proj_rows','encoder_calls','encoder_samples',
 'projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
 'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls',
 'signal_compile_calls','program_compile_calls','replay_calls','metric_calls','schedule_decisions')

def vector(**values):
    if set(values)-set(AXES) or any(type(v) is not int or v < 0 for v in values.values()):
        raise ValueError('invalid nonnegative integer resource vector')
    return {k: values.get(k, 0) for k in AXES}

def add(*vectors):
    if any(set(v) != set(AXES) or any(type(x) is not int or x < 0 for x in v.values()) for v in vectors):
        raise ValueError('complete integer vectors required')
    return {k: sum(v[k] for v in vectors) for k in AXES}

def predict(n):
    return vector(predict_API=n,predict_samples=n,predict_context_tokens=3*n,
      action_encoder_calls=n,action_encoder_samples=n,action_encoder_context_tokens=3*n,
      predictor_calls=n,predictor_samples=n,predictor_context_tokens=3*n,
      pred_proj_calls=n,pred_proj_rows=3*n)

def signal():
    return add(predict(8),vector(jacobian_API=2,signal_compile_calls=2,VJP_directions=384,
      grad_API=384,raw_backward_callbacks=384,array_SVD_calls=2,schedule_decisions=1))

def progress(h,w):
    if h not in (1,3) or w not in (4,8): raise ValueError('original queue shape')
    return add(predict(h*w),vector(program_compile_calls=w,metric_calls=w,solve_calls=1,array_SVD_calls=1))

def reconfiguration(h):
    if h not in (1,3): raise ValueError('original horizon')
    return add(predict(9*(h+3)),vector(jacobian_API=9,signal_compile_calls=9,
      VJP_directions=1728,grad_API=1728,raw_backward_callbacks=1728,
      array_SVD_calls=10,solve_calls=9,program_compile_calls=10,replay_calls=18,metric_calls=18))

def verification(tokens):
    if type(tokens) is not int or not 1 <= tokens <= 9: raise ValueError('chain length')
    return add(predict(2*tokens),vector(replay_calls=2))

def cap():
    per = add(signal(),reconfiguration(3))
    return add({k:17*v for k,v in per.items()},verification(9))

def fits(used, queued, reserve, ceiling):
    total = add(used,queued,reserve)
    if set(ceiling) != set(AXES) or any(type(x) is not int or x < 0 for x in ceiling.values()):
        raise ValueError('complete ceiling required')
    return {k: total[k]-ceiling[k] for k in AXES if total[k] > ceiling[k]}

def proof(root=ROOT):
    for name, expected in PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected: raise ValueError('source drift: '+name)
    s, worst, final, c = signal(), reconfiguration(3), verification(9), cap()
    for h,w in itertools.product((1,3),(4,8)):
        if fits(vector(),progress(h,w),vector(),worst): raise ValueError('progress cost exceeds bound')
    if fits(vector(),reconfiguration(1),vector(),worst): raise ValueError('R1 exceeds bound')
    # Before every possible next signal/queue, previous parents may each cost S+R3.
    # This dominates every actual mixture, including budget already spent on the signal.
    for prior in range(17):
        used = {k: prior*(s[k]+worst[k]) for k in AXES}
        if fits(used,s,vector(),c): raise ValueError('signal preflight fails')
        used = add(used,s)
        for h,w in itertools.product((1,3),(4,8)):
            for q in (progress(h,w), reconfiguration(h)):
                if fits(used,q,final,c): raise ValueError('atomic queue plus final reserve fails')
    if 1+8+8 != 17 or 17*8 > 144: raise ValueError('original tree structural bound')
    return {'schema':'agent_p3_c3_delivery_budget_exploratory_20261029_v1.proof',
      'selection':'post-K2 exploratory; identical to existing K17, not a new independent experiment',
      'slots':[{'slot':i,'root':r,'mode':m} for i,(r,m) in enumerate(
          itertools.product((0,1),('fixedH1','staticH3_progress','adaptive')))],
      'parent_depth_counts':[1,8,8], 'candidate_identity_max':136,'identity_denominator_per_slot':144,
      'signal':s,'worst_R3':worst,'final_max':final,'common_cap_per_slot':c,
      'panel_cap':{k:6*v for k,v in c.items()},'atomic_queue_budget_proved':True,
      'conditional_only':'requires original policy bounds, bounded actual implementations, successful execution and eligible witness',
      'guarantees_model_witness':False,'guarantees_runtime_storage_or_wall':False,
      'scientific_admission':False,'physical_release':False,'same_flops':False,'sources':PINS}

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('proof',)); p.parse_args()
    print(json.dumps(proof(),sort_keys=True,allow_nan=False))

if __name__ == '__main__': main()
