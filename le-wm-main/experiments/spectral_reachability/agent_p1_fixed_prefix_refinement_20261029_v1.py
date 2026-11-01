"""Fixed 64-prefix paired suffix refinement; callback core, not a host."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_two_packet_uphill_generator_20261029_v1 as parent

NAME='agent_p1_fixed_prefix_refinement_20261029_v1'
F,core=parent.F,parent.core
require,raw=F.require,F.raw


def contract():
    return dict(schema=NAME+'.contract',slots=4,source_arms=list(parent.ARMS),source_iterations=list(range(1,9)),
        groups=64,suffix_iterations=8,iterates_per_group=9,candidate_denominator=576,
        anchor='original same-slot winner terminal Q; strict stage Q > root initial Q AND terminal Q < anchor',
        start='each source arm iteration 1..8 exact prefix and its own suffix, no outcome selection',
        update='fixed prefix; original chart chain and eta=1 projected suffix step, simultaneous stale prefix gradient removed',
        per_group=dict(predict=19,scalar_reverse=8),new_total=dict(predict=1216,scalar_reverse=512),
        inherited_parent=dict(predict=288,scalar_reverse=128),combined=dict(predict=1504,scalar_reverse=640),
        accounting='prefix replay1 + suffix scores9 + suffix AD forward8 + selected suffix replay1 per group',
        equal_budget='both source arms 8 groups/slot and same scheduled counts; no equal FLOPs claim',
        stops_on_failure=True,encoder=0,physics=0,training=0,production_ready=False,P1_admission=False,physical_release=False)


def requests(prepared,report,receipt,report_sha):
    """Pure join; caller must first independently reconstruct full host consume."""
    from experiments.spectral_reachability import agent_p1_two_packet_host_cuda_20261029_v1 as host
    require(prepared['schema']==host.NAME+'.prepare' and prepared['contract']==host.contract()
        and report['schema']==host.NAME+'.report' and report['status']=='complete_model_only' and report['errors']==[]
        and receipt['schema']==host.NAME+'.consume' and receipt['status']=='verified_model_only' and receipt['complete'] is True
        and receipt['report_sha256']==report_sha and receipt['fixed_denominator']==4 and receipt['arm_denominator']==8
        and receipt['candidate_denominator']==72 and raw(receipt['rows'])==raw(report['rows'])
        and receipt['actual_new_compute']=={k:F.caps(288,128) for k in ('attempted','completed')},'actual complete original fixed72')
    require(all(x[k] is False for x in (report,receipt) for k in ('P1_admission','freshness','physical_release')), 'no changed science gate')
    rows=[]
    for slot,(spec,saved) in enumerate(zip(prepared['authority']['rows'],receipt['rows'],strict=True)):
        base=spec['request'];parent.request(base);require(slot==saved['index']==base['slot'] and len(saved['arms'])==2,'fixed slot')
        for name,entry in zip(parent.ARMS,saved['arms'],strict=True):
            r=entry['result'];require(entry['arm']==name and entry['status']=='sealed' and r['arm']==name
                and len(r['iterates'])==9 and [x['iteration'] for x in r['iterates']]==list(range(9)), 'all original paid iterates')
            for v in r['iterates'][1:]:
                rows.append(dict(index=len(rows),slot=slot,source_arm=name,source_iteration=v['iteration'],
                    task=deepcopy(base['task']),prefix_logits=deepcopy(v['prefix_logits']),suffix_logits=deepcopy(v['suffix_logits']),
                    expected_raw_bits=v['raw_bits'],expected_mid=deepcopy(v['mid']),expected_end=deepcopy(v['end']),
                    initial_Q=r['initial_Q'],baseline_Q=r['original_winner_terminal_Q'],source_report_sha256=report_sha))
    require(len(rows)==64,'fixed all64, no filtering');return rows


def run_group(q,backend,sink):
    """19 real callback forwards including eight reverse; no model loading."""
    required={'index','slot','source_arm','source_iteration','task','prefix_logits','suffix_logits','expected_raw_bits',
              'expected_mid','expected_end','initial_Q','baseline_Q','source_report_sha256'}
    require(type(q) is dict and set(q)==required,'explicit no-outcome request')
    require(type(q['index']) is int and 0<=q['index']<64 and type(q['slot']) is int and q['slot']==q['index']//16
        and q['source_arm']==parent.ARMS[(q['index']%16)//8] and q['source_iteration']==q['index']%8+1,'fixed group identities')
    task=core.validate_task(q['task']);require(len(task['goal'])==192 and np.array_equal(task['projection'],np.eye(192)),'original Q')
    core.hash_value(q['source_report_sha256']);root=task['root_context'];goal=task['goal']
    require(F.cost(goal,root)==q['initial_Q'] and type(q['baseline_Q']) in (int,float)
            and np.isfinite(q['baseline_Q']) and q['baseline_Q']>=0,'original goal/anchor')
    x=core.finite(q['prefix_logits'],(10,));y=core.finite(q['suffix_logits'],(10,)).astype(float)
    require(np.linalg.norm(x)<=F.CAP and np.linalg.norm(y)<=F.CAP,'original projected packet domains')
    prefix=F.chart(x);suffix=F.chart(y)
    require(F.bits(np.concatenate((prefix,suffix)))==q['expected_raw_bits'],'exact source action bytes')
    sink(dict(kind='prefix_intent',group=q['index'],raw_bits=F.bits(prefix)))
    mid=backend.step(deepcopy(root),prefix,'prefix_replay');core.checked_path(task,root,prefix.tolist(),[mid])
    require(raw(mid)==raw(q['expected_mid']),'literal source prefix model context')
    stage=F.cost(goal,mid);sink(dict(kind='prefix_return',context=deepcopy(mid),stage_Q=stage))
    rows=[];grads=[]
    def score(i):
        a=F.chart(y);sink(dict(kind='score_intent',iteration=i,raw_bits=F.bits(a)))
        end=backend.step(deepcopy(mid),a,f'suffix_{i}');core.checked_path(task,mid,a.tolist(),[end])
        row=dict(iteration=i,logits=y.tolist(),commands=a.tolist(),raw_bits=F.bits(a),context=deepcopy(end),terminal_Q=F.cost(goal,end))
        sink(dict(kind='score_return',row=deepcopy(row)));return row
    rows.append(score(0));require(raw(rows[0]['context'])==raw(q['expected_end']),'source same-action terminal replay')
    for i in range(1,9):
        a=F.chart(y);sink(dict(kind='gradient_intent',iteration=i,raw_bits=F.bits(a)))
        d=backend.gradient(deepcopy(mid),a,f'gradient_{i}')
        require(type(d) is dict and set(d)=={'context','native_float32_loss','raw_gradient'} and raw(d['context'])==raw(rows[-1]['context'])
            and type(d['native_float32_loss']) in (int,float) and np.isfinite(d['native_float32_loss']) and d['native_float32_loss']>=0,'actual scalar reverse and same-context forward')
        g=core.finite(d['raw_gradient'],(5,2),True).astype(float)*(1-a.astype(float)**2)
        grads.append(deepcopy(d));sink(dict(kind='gradient_return',iteration=i,gradient=deepcopy(d)))
        y=F.proposal(y,g.reshape(10));require(np.linalg.norm(y)<=F.CAP,'original projected suffix ball')
        rows.append(score(i))
    chosen=min(rows,key=lambda r:(r['terminal_Q'],r['iteration']))
    sink(dict(kind='winner_replay_intent',iteration=chosen['iteration'],raw_bits=chosen['raw_bits']))
    replay=backend.step(deepcopy(mid),np.asarray(chosen['commands'],np.float32),'winner_replay')
    require(raw(replay)==raw(chosen['context']),'independent selected suffix replay')
    sink(dict(kind='winner_replay_return',context=deepcopy(replay)))
    for row in rows:
        row['uphill_from_initial']=stage>q['initial_Q'];row['better_terminal_than_original']=row['terminal_Q']<q['baseline_Q']
        row['witness']=row['uphill_from_initial'] and row['better_terminal_than_original']
    eligible=[v for v in rows if v['witness']];w=min(eligible,key=lambda v:(v['terminal_Q'],v['iteration'])) if eligible else None
    result=dict(index=q['index'],slot=q['slot'],source_arm=q['source_arm'],source_iteration=q['source_iteration'],stage_Q=stage,
        initial_Q=q['initial_Q'],baseline_Q=q['baseline_Q'],iterates=rows,gradients=grads,selected_iteration=None if w is None else w['iteration'],
        status='no_witness' if w is None else 'model_witness',P1_admission=False,physical_release=False)
    sink(dict(kind='group_return',result=deepcopy(result)));return result


def panel(groups,backend_factory,sink):
    require(type(groups) is list and len(groups)==64,'fixed64 group denominator')
    for i,q in enumerate(groups):
        require(q['index']==i and q['slot']==i//16 and q['source_arm']==parent.ARMS[(i%16)//8]
            and q['source_iteration']==i%8+1,'all group identities before dispatch')
    rows=[dict(index=i,status='not_started',result=None) for i in range(64)]
    for q,row in zip(groups,rows,strict=True):
        row['status']='attempted';sink(dict(kind='group_started',index=q['index']))
        try:
            row['result']=run_group(q,backend_factory(q),lambda e:sink(dict(index=q['index'],event=e)))
            row['status']='sealed'
        except Exception as exc:
            row['status']='failed_partial';row['error']=type(exc).__name__+': '+str(exc)
            sink(dict(kind='group_failure',index=q['index'],error=row['error']))
            return dict(schema=NAME+'.panel',status='partial_failclosed',rows=rows,group_denominator=64,
                        candidate_denominator=576,P1_admission=False,physical_release=False)
    return dict(schema=NAME+'.panel',status='complete_model_only',rows=rows,group_denominator=64,
                candidate_denominator=576,P1_admission=False,physical_release=False)
