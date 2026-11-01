"""Fixed multi-start alternating prefix and conditional Vreach suffix; RAM callback core."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_two_packet_uphill_generator_20261029_v1 as original
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler

NAME='agent_p1_joint_vreach_multistart_20261029_v1'
F,core=original.F,original.core
require,raw=F.require,F.raw
STARTS=('winner','zero','positive_e0','negative_e0')
ARMS=('projected_ascent','outward_free_tangent')
ITERATIONS=4


def contract():
    return dict(schema=NAME+'.contract',slots=4,known_roots=2,arms=list(ARMS),starts=list(STARTS),updates=4,
        candidates_per_start=5,candidates_per_arm=20,candidates_per_slot=40,candidate_denominator=160,
        initializer='winner prefix, zero, +/- 0.25 coordinate0 in original float64 projected packet ball',
        update='eta1 ascent or outward-free tangent ascent then ball projection; fresh conditional zero-H1 J and original bounded ridge suffix',
        tangent='for x!=0 remove positive radial component of ascent at every radius, then project; no boundary proximity threshold',
        reach='original bounded_ridge own_reach_value, identity192, damping .002, original cap',
        eligibility='strict stage_Q > root initial_Q AND own_reach_value < winner-prefix own_reach_value AND terminal_Q < original-winner terminal_Q',
        selection='model-only min(terminal_Q,start_index,iteration) over eligible; null if none',
        baseline='original same-slot winner raw10 independent replay and winner-prefix conditional Vreach',
        callback_cost_per_arm=dict(predict=138,scalar_reverse=16,jacobian=20,jacobian_vjp_directions=3840,svd=20,solve=20),
        callback_cost_total=dict(predict=1104,scalar_reverse=128,jacobian=160,jacobian_vjp_directions=30720,svd=160,solve=160),
        accounting='per arm baseline2 + 20*(prefix1, conditional signal2, suffix1, replay2) + 16 gradient forwards; signal assumes 192 VJP',
        ledger_scope='callback-level charges; native nested callbacks and GPU time are not certified',
        encoder=0,physics=0,training=0,FLOPs=None,equal_FLOPs=False,stop_on_first_failure=True,
        production_ready=False,source_authority_verified=False,P1_admission=False,freshness=False,physical_release=False)


def starts(winner):
    x=compiler.finite(winner,(10,))
    require(np.linalg.norm(x)<=F.CAP,'winner already in original prefix ball')
    e=np.zeros(10);e[0]=0.25
    return [x.copy(),np.zeros(10),e,-e]


class Ledger:
    """Charges fixed call-envelope before callback; not a native resource audit."""
    CAPS=contract()['callback_cost_per_arm']
    def __init__(self,sink):
        self.sink=sink;self.attempted={k:0 for k in self.CAPS};self.completed={k:0 for k in self.CAPS};self.failed=False
    def call(self,label,charges,callback):
        require(not self.failed and all(k in self.CAPS and type(v) is int and v>=0 for k,v in charges.items()),'valid callback charge')
        require(all(self.attempted[k]+charges.get(k,0)<=self.CAPS[k] for k in self.CAPS),'atomic callback cap')
        for k,v in charges.items():self.attempted[k]+=v
        self.sink(dict(kind='resource_attempt',label=label,charge=deepcopy(charges),attempted=deepcopy(self.attempted)))
        try:v=callback()
        except BaseException as exc:
            self.failed=True;self.sink(dict(kind='resource_failure',label=label,error=repr(exc),
                attempted=deepcopy(self.attempted),completed=deepcopy(self.completed)));raise
        for k,n in charges.items():self.completed[k]+=n
        self.sink(dict(kind='resource_return',label=label,completed=deepcopy(self.completed)))
        return v
    def receipt(self):return dict(caps=deepcopy(self.CAPS),attempted=deepcopy(self.attempted),completed=deepcopy(self.completed),
        failed=self.failed,scope='callback charge envelope only; not actual native IO',FLOPs=None)


def tangent_ascent(x, ascent):
    x=compiler.finite(x,(10,));direction=compiler.finite(ascent,(10,))
    norm2=float(x@x); radial=float(x@direction)
    if norm2>0 and radial>0:direction=direction-(radial/norm2)*x
    return direction


def run_arm(q,arm,backend,sink):
    """backend.step, gradient and signal(mid,label): signal returns nominal_context and J[192,10]."""
    require(arm in ARMS,'fixed comparison arm');task,wx,wy=original.request(q);core.validate_task(task)
    require(np.linalg.norm(wy)<=F.CAP,'winner suffix ball');goal=compiler.finite(task['goal'],(192,))
    root=task['root_context'];initial=F.cost(goal,root);ledger=Ledger(sink);rows=[]
    def step(context,commands,label):
        output=ledger.call(label,dict(predict=1),lambda:backend.step(deepcopy(context),commands,label))
        core.checked_path(task,context,commands.tolist(),[output]);return output
    def signal(mid,label):
        result=ledger.call(label,dict(predict=2,jacobian=1,jacobian_vjp_directions=192),lambda:backend.signal(deepcopy(mid),label))
        require(type(result) is dict and set(result)=={'nominal_context','jacobian'},'explicit conditional signal ABI')
        zero=np.zeros((5,2),np.float32);core.checked_path(task,mid,zero.tolist(),[result['nominal_context']])
        return result,compiler.finite(result['nominal_context']['latent_history'][-1],(192,)),compiler.finite(result['jacobian'],(192,10))
    a,b=F.chart(wx),F.chart(wy)
    bmid=step(root,a,'baseline_prefix');bend=step(bmid,b,'baseline_suffix');baseline=F.cost(goal,bend)
    baseline_reach=None
    for start,x0 in enumerate(starts(wx)):
        x=x0.copy()
        for iteration in range(ITERATIONS+1):
            prefix=F.chart(x);mid=step(root,prefix,f's{start}_i{iteration}_prefix')
            sig,nominal,j=signal(mid,f's{start}_i{iteration}_signal')
            solve=ledger.call(f's{start}_i{iteration}_ridge',dict(svd=1,solve=1),lambda:compiler.bounded_ridge(j,goal-nominal))
            suffix=F.chart(solve['params']);end=step(mid,suffix,f's{start}_i{iteration}_suffix')
            replay_mid=step(root,prefix,f's{start}_i{iteration}_replay_prefix')
            replay_end=step(replay_mid,suffix,f's{start}_i{iteration}_replay_suffix')
            require(raw(replay_mid)==raw(mid) and raw(replay_end)==raw(end),'independent action replay exact')
            if start==0 and iteration==0:
                require(F.bits(prefix)==F.bits(a) and raw(mid)==raw(bmid),'same original winner prefix/context')
                baseline_reach=solve['own_reach_value']
            stage=F.cost(goal,mid);terminal=F.cost(goal,end);reach=solve['own_reach_value']
            flags=dict(stage_uphill=stage>initial,reach_better=reach<baseline_reach,terminal_better=terminal<baseline)
            witness=all(flags.values());commands=np.concatenate((prefix,suffix))
            row=dict(start=start,start_kind=STARTS[start],iteration=iteration,prefix_logits=x.tolist(),prefix=prefix.tolist(),
                mid=deepcopy(mid),signal=deepcopy(sig),suffix_solve=solve,suffix=suffix.tolist(),end=deepcopy(end),
                commands=commands.tolist(),raw_float32_bits=F.bits(commands),stage_Q=stage,own_reach_value=reach,
                terminal_Q=terminal,**flags,witness=witness)
            rows.append(row);sink(dict(kind='candidate_return',slot=q['slot'],candidate=deepcopy(row)))
            if iteration<ITERATIONS:
                grad=ledger.call(f's{start}_i{iteration}_stage_gradient',dict(predict=1,scalar_reverse=1),
                    lambda:backend.gradient(deepcopy(root),prefix,f's{start}_i{iteration}_stage_gradient'))
                require(type(grad) is dict and set(grad)=={'context','native_float32_loss','raw_gradient'}
                    and raw(grad['context'])==raw(mid) and type(grad['native_float32_loss']) in (int,float)
                    and np.isfinite(grad['native_float32_loss']) and grad['native_float32_loss']>=0,'same current prefix gradient')
                gx=core.finite(grad['raw_gradient'],(5,2),True).astype(float)*(1-prefix.astype(float)**2)
                sink(dict(kind='stage_gradient',slot=q['slot'],start=start,iteration=iteration,gradient=deepcopy(grad)))
                ascent=gx.reshape(10)
                if arm=='outward_free_tangent':ascent=tangent_ascent(x,ascent)
                x=F.proposal(x,-ascent);require(np.linalg.norm(x)<=F.CAP,'projected prefix ball')
    require(len(rows)==20 and ledger.attempted==ledger.completed==ledger.CAPS,'all20 candidate callback envelopes')
    eligible=[v for v in rows if v['witness']]
    chosen=min(eligible,key=lambda r:(r['terminal_Q'],r['start'],r['iteration'])) if eligible else None
    return dict(slot=q['slot'],arm=arm,status='model_witness' if chosen else 'no_witness',initial_Q=initial,
        baseline_terminal_Q=baseline,baseline_reach_value=baseline_reach,baseline_commands=q['winner_commands'],
        candidates=rows,selected=None if chosen is None else dict(start=chosen['start'],iteration=chosen['iteration'],
            commands=deepcopy(chosen['commands']),raw_float32_bits=chosen['raw_float32_bits']),resources=ledger.receipt(),
        candidate_denominator=20,P1_admission=False,freshness=False,physical_release=False)


def panel(requests,factory,sink):
    require(type(requests) is list and len(requests)==4,'fixed four tasks')
    for i,q in enumerate(requests):original.request(q);require(q['slot']==i,'fixed order')
    rows=[dict(slot=i,arms=[dict(arm=a,status='not_started',result=None) for a in ARMS]) for i in range(4)]
    for q,row in zip(requests,rows,strict=True):
        for a,entry in zip(ARMS,row['arms'],strict=True):
            entry['status']='attempted';resource_events=[]
            def forward(e):
                if e['kind'].startswith('resource_'):resource_events.append(deepcopy(e))
                sink(dict(slot=q['slot'],arm=a,event=e))
            try:
                entry['result']=run_arm(q,a,factory(q,a),forward)
                if a==ARMS[1]:
                    first=row['arms'][0]['result'];second=entry['result']
                    require(first['initial_Q']==second['initial_Q'] and first['baseline_terminal_Q']==second['baseline_terminal_Q']
                        and first['baseline_reach_value']==second['baseline_reach_value']
                        and raw(first['baseline_commands'])==raw(second['baseline_commands']),'same paid baseline across arms')
                entry['status']='sealed'
            except Exception as exc:
                entry.update(status='failed_partial',result=None,error=repr(exc),resource_events=resource_events,
                    failed_resource_snapshot=next((e for e in reversed(resource_events) if e['kind']=='resource_failure'),None))
                sink(dict(kind='arm_failure',slot=q['slot'],arm=a,error=repr(exc)))
                return dict(schema=NAME+'.panel',status='partial_failclosed',rows=rows,slot_denominator=4,arm_denominator=8,
                    candidate_denominator=160,P1_admission=False,freshness=False,physical_release=False)
    return dict(schema=NAME+'.panel',status='complete_model_only',rows=rows,slot_denominator=4,arm_denominator=8,
        candidate_denominator=160,P1_admission=False,freshness=False,physical_release=False)
