"""Two-packet native transcript adapter and saved single-arm audit; no runner."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_two_packet_uphill_generator_20261029_v1 as method

F,core=method.F,method.core
require,raw=F.require,F.raw
CONTROL={'score_intent','score_return','gradient_intent','gradient_return',
         'winner_replay_intent','winner_replay_return','arm_return'}


class LabelAdapter:
    """Preserve native audited gradient prefix without changing the method."""
    def __init__(self,backend):self.backend=backend
    def step(self,c,a,label):return self.backend.step(c,a,label)
    def gradient(self,c,a,label):return self.backend.gradient(c,a,'gradient_'+label)


def schedule(q,result):
    arm=result['arm'];rows=result['iterates'];root=q['task']['root_context']
    require(arm in method.ARMS and len(rows)==9,'full arm shape')
    out=[(f'{arm}_prefix_0',root,rows[0]['prefix']),(f'{arm}_suffix_0',rows[0]['mid'],rows[0]['suffix'])]
    for i in range(1,9):
        prev,cur=rows[i-1],rows[i]
        out.extend([(f'gradient_{arm}_stage_gradient_{i}',root,prev['prefix']),
                    (f'gradient_{arm}_suffix_gradient_{i}',prev['mid'],prev['suffix']),
                    (f'{arm}_prefix_{i}',root,cur['prefix']),(f'{arm}_suffix_{i}',cur['mid'],cur['suffix'])])
    out.extend([(f'{arm}_winner_prefix_replay',root,rows[0]['prefix']),
                (f'{arm}_winner_suffix_replay',rows[0]['mid'],rows[0]['suffix'])])
    return out


def audit(q,result,receipt,events):
    method.request(q);plan=schedule(q,result)
    native=[e for e in events if e['kind'] not in CONTROL]
    starts,ends,linked=F.audit_native(q['task'],receipt,native,plan)
    require(len(starts)==len(ends)==36 and receipt['caps']==F.caps(36,16),'whole same-arm native ledger')
    grad={}
    for i,(label,_,a) in enumerate(plan):
        if not label.startswith('gradient_'):continue
        phase=f'f{i:03}'
        op=next(x for x in receipt['events'] if x['resource']=='grad' and x['scope']['forward']==phase)
        cbop=next(x for x in receipt['events'] if x['resource']=='raw_backward' and x['scope']['forward']==phase)
        ai,ao=linked[op['index']];cb=linked[cbop['index']][0]
        core.keys(ai['arrays'],('grad:loss','grad:raw'));core.keys(ao['arrays'],('grad:output',));core.keys(cb['arrays'],('value',))
        loss,leaf,g,b=(F.signal.unblob(x) for x in (ai['arrays']['grad:loss'],ai['arrays']['grad:raw'],ao['arrays']['grad:output'],cb['arrays']['value']))
        require(loss.shape==(1,) and leaf.shape==g.shape==b.shape==(5,2)
                and leaf.tobytes()==np.asarray(a,np.float32).tobytes() and g.tobytes()==b.tobytes(),'exact scalar/raw/gradient/callback native buffers')
        grad[i]=dict(context=deepcopy(ends[i]['context']),native_float32_loss=float(loss[0]),raw_gradient=g.tolist())
    class Replay:
        def __init__(self):self.i=0
        def pull(self,c,a,label,is_gradient):
            i=self.i;self.i+=1;expected=plan[i]
            require(expected[0]==('gradient_'+label if is_gradient else label)
                    and raw(c)==raw(expected[1]) and F.bits(a)==F.bits(expected[2]),'literal saved callback identity')
            return deepcopy(grad[i] if is_gradient else ends[i]['context'])
        def step(self,c,a,label):return self.pull(c,a,label,False)
        def gradient(self,c,a,label):return self.pull(c,a,label,True)
    replay=Replay();control=[];rebuilt=method.run_arm(q,result['arm'],replay,control.append)
    require(replay.i==36 and raw(rebuilt)==raw(result) and raw(control)==raw([e for e in events if e['kind'] in CONTROL]),
            'all original method math/control/decisions rebuilt from native saved IO')
    positions={id(e):i for i,e in enumerate(events)}
    # Every model call must be bracketed by the corresponding control pair;
    # gradient pairs bracket two calls, score pairs bracket two calls.
    for c in [e for e in events if e['kind'] in ('score_intent','gradient_intent','winner_replay_intent')]:
        k=c['kind'].replace('_intent','_return');after=[e for e in events if e['kind']==k and positions[id(e)]>positions[id(c)]]
        require(after,'matching control return');end=after[0]
        enclosed=[e for e in starts if positions[id(c)]<positions[id(e)]<positions[id(end)]]
        require(len(enclosed)==2,'each intent encloses its exact two paid calls')
    return dict(arm=result['arm'],selected_iteration=result['selected_iteration'],status=result['status'],native_forwards=36,scalar_reverse=16)
