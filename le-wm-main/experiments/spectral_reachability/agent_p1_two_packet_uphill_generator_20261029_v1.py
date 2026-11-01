"""Paired equal-call prefix ascent/descent and suffix descent method core.

Callback-only engineering core: no source authority, CUDA launcher or physical
release. Input must come from an independently authenticated original winner.
"""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_nonlinear_suffix_library_20261028_v2 as F

NAME='agent_p1_two_packet_uphill_generator_20261029_v1'
core=F.core
require=F.require
ARMS=('stage_descent','stage_ascent')


def contract():
    return dict(schema=NAME+'.contract',slots=4,known_roots=2,arms=list(ARMS),iterations=8,
        iterates_per_arm=9,candidate_denominator=72,arm_denominator=8,slot_denominator=4,
        direction='simultaneous prefix ascent or descent of stage Q; suffix descent of terminal Q at the same current prefix',
        update='original float64 chart chain and original projected_logits proposal, eta=1; independent original cap per packet',
        witness='strict stage_Q > initial_root_Q and terminal_Q < same-action original_winner_terminal_Q',
        selection='minimum terminal_Q then iteration among witnesses, otherwise null',
        per_arm=dict(predict=36,scalar_reverse=16),total=dict(predict=288,scalar_reverse=128),
        cost_scope='per arm initial2 + 8*(2 AD forward + 2 scoring forward) + independent original-winner replay2',
        equal_budget='equal scheduled forward and reverse counts only; not equal FLOPs or wall',
        encoder=0,physics=0,training=0,retries=0,stop_on_first_failure=True,
        source_authority_verified=False,production_ready=False,P1_admission=False,physical_release=False)


def request(q):
    require(type(q) is dict and set(q)=={'slot','root_slot','condition','task','winner_prefix_logits','winner_suffix_logits',
        'winner_commands','winner_raw_bits','winner_authority'},'strict model-only request whitelist')
    require(type(q['slot']) is int and 0<=q['slot']<4 and (q['root_slot'],q['condition'])==((0,'off'),(0,'on'),(1,'off'),(1,'on'))[q['slot']], 'fixed slot order')
    require(type(q['winner_authority']) is str and len(q['winner_authority'])==64 and all(x in '0123456789abcdef' for x in q['winner_authority']), 'explicit authority digest, not certification')
    task=core.validate_task(q['task']);x=core.finite(q['winner_prefix_logits'],(10,));y=core.finite(q['winner_suffix_logits'],(10,))
    require(len(task['goal'])==192 and np.array_equal(np.asarray(task['projection']),np.eye(192)), 'original identity192 Q only')
    a,b=F.chart(x),F.chart(y);commands=np.concatenate((a,b))
    require(F.bits(commands)==q['winner_raw_bits'] and F.bits(q['winner_commands'])==q['winner_raw_bits']
            and np.asarray(q['winner_commands']).shape==(10,2), 'exact two-packet winner chart/float32')
    # Initial endpoints are the source solver's witness, not freshly projected.
    # Host must certify both original domains before invoking this core.
    return task,x.astype(float),y.astype(float)


def run_arm(q,arm,backend,sink):
    task,x,y=request(q);require(arm in ARMS,'fixed arm')
    root=task['root_context'];goal=task['goal'];initial=F.cost(goal,root);rows=[];gradients=[]
    def score(i):
        a,b=F.chart(x),F.chart(y)
        sink(dict(kind='score_intent',arm=arm,iteration=i,prefix_bits=F.bits(a),suffix_bits=F.bits(b)))
        mid=backend.step(deepcopy(root),a,f'{arm}_prefix_{i}');core.checked_path(task,root,a.tolist(),[mid])
        end=backend.step(deepcopy(mid),b,f'{arm}_suffix_{i}');core.checked_path(task,mid,b.tolist(),[end])
        row=dict(iteration=i,prefix_logits=x.tolist(),suffix_logits=y.tolist(),prefix=a.tolist(),suffix=b.tolist(),
                 raw_bits=F.bits(np.concatenate((a,b))),mid=deepcopy(mid),end=deepcopy(end),
                 stage_Q=F.cost(goal,mid),terminal_Q=F.cost(goal,end))
        sink(dict(kind='score_return',arm=arm,row=deepcopy(row)));return row
    rows.append(score(0));winner_Q=rows[0]['terminal_Q']
    for i in range(1,9):
        prior=rows[-1];a,b=F.chart(x),F.chart(y)
        sink(dict(kind='gradient_intent',arm=arm,iteration=i,raw_bits=prior['raw_bits']))
        dg=backend.gradient(deepcopy(root),a,f'{arm}_stage_gradient_{i}')
        sg=backend.gradient(deepcopy(prior['mid']),b,f'{arm}_suffix_gradient_{i}')
        for d in (dg,sg):
            require(type(d) is dict and set(d)=={'context','native_float32_loss','raw_gradient'}
                    and type(d['native_float32_loss']) in (int,float) and np.isfinite(d['native_float32_loss'])
                    and d['native_float32_loss']>=0,'literal finite native scalar AD witness')
        require(dg['context']==prior['mid'] and sg['context']==prior['end'],'same current context/action AD replay')
        gx=core.finite(dg['raw_gradient'],(5,2),True).astype(float)*(1-a.astype(float)**2)
        gy=core.finite(sg['raw_gradient'],(5,2),True).astype(float)*(1-b.astype(float)**2)
        grads=dict(iteration=i,stage=deepcopy(dg),suffix=deepcopy(sg),prefix_logit_gradient=gx.reshape(10).tolist(),suffix_logit_gradient=gy.reshape(10).tolist())
        gradients.append(grads);sink(dict(kind='gradient_return',arm=arm,gradient=deepcopy(grads)))
        x=F.proposal(x,(-1 if arm=='stage_ascent' else 1)*gx.reshape(10));y=F.proposal(y,gy.reshape(10))
        require(np.linalg.norm(x)<=F.CAP and np.linalg.norm(y)<=F.CAP,'both updated original projected packet balls')
        rows.append(score(i))
    # A separately paid baseline replay is not the selected candidate's output.
    first=rows[0];a=np.asarray(first['prefix'],np.float32);b=np.asarray(first['suffix'],np.float32)
    sink(dict(kind='winner_replay_intent',arm=arm,raw_bits=first['raw_bits']))
    mid=backend.step(deepcopy(root),a,f'{arm}_winner_prefix_replay')
    end=backend.step(deepcopy(mid),b,f'{arm}_winner_suffix_replay')
    require(mid==first['mid'] and end==first['end'],'independent same-action original winner replay')
    sink(dict(kind='winner_replay_return',arm=arm,mid=deepcopy(mid),end=deepcopy(end)))
    for r in rows:
        r['uphill_from_initial']=r['stage_Q']>initial
        r['better_terminal_than_winner']=r['terminal_Q']<winner_Q
        r['witness']=r['uphill_from_initial'] and r['better_terminal_than_winner']
    eligible=[r for r in rows if r['witness']]
    chosen=min(eligible,key=lambda v:(v['terminal_Q'],v['iteration']))['iteration'] if eligible else None
    result=dict(arm=arm,slot=q['slot'],initial_Q=initial,original_winner_terminal_Q=winner_Q,
                iterates=rows,gradients=gradients,selected_iteration=chosen,
                status='model_witness' if chosen is not None else 'no_witness',physical_release=False,P1_admission=False)
    sink(dict(kind='arm_return',arm=arm,result=deepcopy(result)));return result


def panel(requests,backend_factory,sink):
    require(type(requests) is list and len(requests)==4,'fixed four-slot denominator')
    for i,q in enumerate(requests):request(q);require(q['slot']==i,'exact fixed order')
    rows=[dict(slot=i,arms=[dict(arm=a,status='not_started',result=None) for a in ARMS]) for i in range(4)]
    for q,row in zip(requests,rows,strict=True):
        for a,entry in zip(ARMS,row['arms'],strict=True):
            entry['status']='attempted';sink(dict(kind='arm_intent',slot=q['slot'],arm=a))
            try:
                entry['result']=run_arm(q,a,backend_factory(q,a),lambda e:sink(dict(slot=q['slot'],event=e)))
                if a=='stage_ascent':
                    first=row['arms'][0]['result'];second=entry['result']
                    require(first['initial_Q']==second['initial_Q'] and first['original_winner_terminal_Q']==second['original_winner_terminal_Q']
                            and first['iterates'][0]==second['iterates'][0], 'same actual root, raw winner and baseline model outputs')
                entry['status']='sealed'
            except Exception as exc:
                entry['status']='failed_partial';entry['result']=None;entry['error']=type(exc).__name__+': '+str(exc)
                sink(dict(kind='arm_failure',slot=q['slot'],arm=a,error=entry['error']))
                return dict(schema=NAME+'.panel',status='partial_failclosed',rows=rows,fixed_denominator=4,
                            candidate_denominator=72,P1_admission=False,physical_release=False)
    return dict(schema=NAME+'.panel',status='complete_model_only',rows=rows,fixed_denominator=4,
                candidate_denominator=72,P1_admission=False,physical_release=False)
