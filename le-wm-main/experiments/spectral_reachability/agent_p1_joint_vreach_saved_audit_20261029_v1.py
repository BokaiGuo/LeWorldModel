"""Saved-only one-arm joint-Vreach native and method reconstruction."""
from copy import deepcopy
import numpy as np

from experiments.spectral_reachability import agent_p1_joint_vreach_multistart_20261029_v1 as method
from experiments.spectral_reachability import agent_p1_joint_vreach_signal_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as component

F, core = method.F, method.core
require, raw = F.require, F.raw
CONTROL = {'resource_attempt', 'resource_return', 'resource_failure', 'candidate_return', 'stage_gradient'}
SIGNAL = {'joint_signal_attempt', 'joint_signal_event', 'joint_signal_return', 'joint_signal_failure'}


def audit(q, arm, result, scalar_receipt, events):
    """Require completed native IO, rerun method using saved outputs only."""
    method.original.request(q)
    require(arm in method.ARMS and result['arm'] == arm, 'arm identity')
    # Scalar ledger resource events have an `entry`; method envelope events
    # instead have label/charge and cannot be conflated with native counters.
    def is_control(e):
        return e['kind'] in CONTROL and not (e['kind'].startswith('resource_') and 'entry' in e)
    scalar_events = [e for e in events if e['kind'] not in SIGNAL and not is_control(e)]
    starts = [e for e in scalar_events if e['kind'] == 'forward_started']
    schedule = [(e['label'], e['input'], e['commands']) for e in starts]
    require(len(schedule) == 98 and sum(x[0].startswith('gradient_') for x in schedule) == 16,
        '98 native forwards and 16 scalar reverses')
    starts, ends, linked = F.audit_native(q['task'], scalar_receipt, scalar_events, schedule)
    gradients = {}
    goal = np.asarray(q['task']['goal'], np.float32)
    for i, (label, _, actions) in enumerate(schedule):
        if not label.startswith('gradient_'): continue
        phase = f'f{i:03}'
        op = next(x for x in scalar_receipt['events'] if x['resource']=='grad' and x['scope']['forward']==phase)
        cb = next(x for x in scalar_receipt['events'] if x['resource']=='raw_backward' and x['scope']['forward']==phase)
        ai, ao = linked[op['index']]; back = linked[cb['index']][0]
        core.keys(ai['arrays'], ('grad:loss','grad:raw')); core.keys(ao['arrays'], ('grad:output',))
        core.keys(back['arrays'], ('value',))
        loss, leaf, grad, callback = (F.signal.unblob(x) for x in (ai['arrays']['grad:loss'],
            ai['arrays']['grad:raw'], ao['arrays']['grad:output'], back['arrays']['value']))
        require(loss.shape == (1,) and leaf.shape == grad.shape == callback.shape == (5,2) and
            leaf.tobytes()==np.asarray(actions,np.float32).tobytes() and grad.tobytes()==callback.tobytes(),
            'scalar native loss/raw/gradient/callback')
        end = np.asarray(ends[i]['context']['latent_history'][-1], np.float32)
        squares = np.square(np.subtract(end, goal, dtype=np.float32), dtype=np.float32)
        expected = float(np.mean(squares, dtype=np.float64)); actual = float(loss[0])
        require(np.isfinite(squares).all() and np.isfinite(actual) and actual >= 0 and
            abs(actual-expected) <= 3e-5*expected+float(np.finfo(np.float32).tiny), 'identity-Q scalar objective')
        gradients[i] = dict(context=deepcopy(ends[i]['context']), native_float32_loss=actual, raw_gradient=grad.tolist())
    calls = []; active = None
    for event in events:
        kind = event['kind']
        if kind == 'joint_signal_attempt':
            require(active is None, 'no nested signal'); active = [event, []]
        elif kind == 'joint_signal_event':
            require(active is not None and event['label']==active[0]['label'], 'ordered signal event')
            active[1].append(event['payload'])
        elif kind == 'joint_signal_return':
            require(active is not None and event['label']==active[0]['label'], 'ordered signal return')
            receipt=event['receipt']; require(receipt['status']=='complete' and receipt['resources']==component.expected('postcontext_H1'), 'full signal resources')
            require(raw(active[1])==raw(receipt['events']), 'complete signal events match receipt')
            entries=active[1]
            a=[i for i,x in enumerate(entries) if x['channel']=='native' and x['payload']['kind']=='native_call_attempt']
            b=[i for i,x in enumerate(entries) if x['channel']=='native' and x['payload']['kind']=='native_call_return']
            require(len(a)==len(b)==1 and a[0]<b[0], 'one native invocation span')
            native_events=[x['payload'] for x in entries[a[0]:b[0]+1]]
            require(raw(native_events)==raw(receipt['native_transcript']), 'signal transcript is actual emitted native stream')
            local=receipt['local_task']; parent=local['root_context']
            require(active[0]['root_task_sha256']==core.digest(q['task']) and
                active[0]['parent_context_sha256']==core.digest(parent) and active[0]['local_task_sha256']==core.digest(local), 'root/parent signal binding')
            out=bridge.from_transcript(q['task'], parent, receipt['native_transcript'])
            require(out['nominal_context']==receipt['result']['nominal_path'][0] and out['jacobian']==receipt['result']['jacobian'], 'actual signal output')
            calls.append((event['label'], parent, out)); active=None
        elif kind == 'joint_signal_failure': raise ValueError('failed signal cannot certify complete arm')
    require(active is None and len(calls)==20, 'all 20 H1 signals')
    order=[('scalar', e['label']) if e['kind']=='forward_started' else ('signal', e['label'])
        for e in events if e['kind'] in ('forward_started','joint_signal_attempt')]
    class Replay:
        def __init__(self): self.i=0; self.j=0; self.k=0
        def ordered(self,kind,label):
            require(self.k<len(order) and order[self.k]==(kind,label), 'actual mixed native dispatch ordering')
            self.k+=1
        def take(self, c, a, label, gradient):
            i=self.i; self.i+=1
            require(i<len(schedule), 'scalar replay count')
            s=schedule[i]
            self.ordered('scalar', 'gradient_'+label if gradient else label)
            require(s[0]==('gradient_'+label if gradient else label) and raw(s[1])==raw(c) and F.bits(s[2])==F.bits(a), 'scalar call context/action identity')
            return deepcopy(gradients[i] if gradient else ends[i]['context'])
        def step(self,c,a,label): return self.take(c,a,label,False)
        def gradient(self,c,a,label): return self.take(c,a,label,True)
        def signal(self,c,label):
            require(self.j<len(calls), 'signal replay count')
            self.ordered('signal',label)
            name, parent, out=calls[self.j]; self.j+=1
            require(name==label and raw(parent)==raw(c), 'signal exact candidate parent'); return deepcopy(out)
    replay=Replay(); control=[]
    rebuilt=method.run_arm(q,arm,replay,control.append)
    require(replay.i==98 and replay.j==20 and replay.k==len(order) and raw(rebuilt)==raw(result) and
        raw(control)==raw([e for e in events if is_control(e)]), 'full method, baseline, candidates, envelope replay')
    return dict(native_forwards=138, scalar_reverse=16, signal_calls=20, jacobian_VJP=3840,
        scalar_VJP=16, candidates=20, selected=result['selected'], P1_admission=False)
