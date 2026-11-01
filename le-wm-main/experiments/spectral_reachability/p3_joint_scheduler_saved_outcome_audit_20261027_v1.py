"""Independent bounded saved-only P3 metric/scheduler arithmetic audit.

No scheduler imports, models, simulator, neural/physical calls or network.
Rebuilds banks/own-Q scores/history and consumes every durable journal event.
"""
from __future__ import annotations
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal

NAME='p3_joint_scheduler_saved_outcome_audit_20261027_v1'
SCHEDULER='p3_joint_horizon_goal_metric_scheduler_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
INPUT='results/'+SCHEDULER+'_cuda_main01'
REPORT_SHA='23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950'
PREPARE_SHA='7886406284604cf17ba97671030ef32fde9d32f6f4c18b8293db0c2129654a47'
METRICS=('old_Q','goal_patch_Q')
ARMS=tuple(q+'__'+s for q in METRICS for s in ('uniform','dynamic'))
META=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10)
require=io.require
def digest(v):return io.sha(io.canonical(v).encode())
def same(a,b):return io.canonical(a)==io.canonical(b)


def array(v,shape):
    a=np.asarray(v)
    require(a.dtype.kind in 'fi' and a.shape==shape and np.isfinite(a).all(),'finite aligned real array')
    return a.astype(float)


def near(a,b,message='saved arithmetic',atol=1e-10):
    aa,bb=np.asarray(a),np.asarray(b)
    require(aa.shape==bb.shape and aa.dtype.kind in 'fi' and bb.dtype.kind in 'fi' and
        np.isfinite(aa).all() and np.isfinite(bb).all() and np.allclose(aa,bb,atol=atol,rtol=1e-10),message)


def projection(q):
    q=array(q,(192,192));require(np.allclose(q,q.T,atol=1e-8,rtol=0) and
        np.allclose(q@q,q,atol=1e-8,rtol=0) and np.trace(q)>.5,'nonzero orthogonal native Q')
    return q


def array_id(value):
    a=np.ascontiguousarray(value,dtype='<f8')
    require(np.isfinite(a).all(),'finite array identity')
    return dict(shape=list(a.shape),dtype='<f8',bytes_sha256=io.sha(a.tobytes()))


def own_cost(z,goal,q):
    r=np.asarray(q)@(array(z,(192,))-array(goal,(192,)))
    return float(np.mean(r*r))


def allocation(schedule,c2,c3):
    require(schedule in ('uniform','dynamic') and all(type(c) in (int,float) and math.isfinite(c) and 0<=c<=1 for c in (c2,c3)),'fixed scheduler inputs')
    gain=max(c2-c3,0)/max(c2,1e-12)
    b3=5 if schedule=='uniform' else math.floor(10*(.25+.5*gain)+.5)
    require(1<=b3<=9,'both H groups retained')
    return dict(blocks={'2':10-b3,'3':b3},candidates={'2':3*(10-b3),'3':2*b3},search_predicts=60,gain=gain)


def bank(seed,h,n,params):
    require(type(seed) is int and 0<=seed<4 and h in (2,3) and type(n) is int and 2<=n<=27,'bounded bank')
    params=array(params,(10*h,));require(np.linalg.norm(params)<=.75*np.sqrt(5*h)+1e-10,'ridge joint ball')
    rng=np.random.Generator(np.random.PCG64(20261027+100*seed+h))
    values=[np.zeros((5*h,2))]
    for _ in range(n-1):
        d=rng.normal(size=10*h)
        u=d/np.linalg.norm(d)*rng.random()**(1/(10*h))*np.sqrt(5*h)*.75
        values.append(np.tanh(u).reshape(5*h,2))
    values[1]=np.tanh(params).reshape(5*h,2)
    return values


def context(c):
    require(all(type(c.get(k)) is type(v) and c[k]==v for k,v in META.items()),'native context metadata')
    return array(c['latent_history'],(3,192)),array(c['action_history'],(2,10))


def history(before,after,actions,normalization):
    lat,tokens=context(before);_,actual=context(after)
    require(same({k:v for k,v in before.items() if k not in ('latent_history','action_history')},
        {k:v for k,v in after.items() if k not in ('latent_history','action_history')}),'unchanged full metadata')
    raw=np.asarray(actions);require(raw.shape in ((5,2),(10,2),(15,2)) and raw.dtype.kind in 'fi' and
        np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'bounded raw time-major controls')
    mean,std=(np.asarray(normalization[k],dtype=np.float32) for k in ('mean','std'))
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0),'normalizer')
    new=((raw.astype(np.float32)-mean)/std).reshape(-1,10)
    expected=np.concatenate((tokens.astype(np.float32),new),axis=0)[-2:]
    require(np.array_equal(actual,expected.astype(float)),'literal float32 normalized history CONTENT/order')
    h=len(new)
    if h<3:require(np.array_equal(np.asarray(after['latent_history'])[:3-h],lat.astype(np.float32)[h:].astype(float)),'retained latent order')


def check_compilation(record,c,normalization):
    require(set(c)=={'raw','metrics'} and set(c['raw'])=={'2','3'} and set(c['metrics'])==set(METRICS),'full derivative/metric census')
    packet=record['packet'];goal=array(packet['goal'],(192,));context(packet['root_context'])
    signals={q:{} for q in METRICS}
    for h in (2,3):
        raw=c['raw'][str(h)];j=array(raw['jacobian'],(192,10*h));n=array(raw['nominal'],(192,))
        for phase in ('nominal','jacobian_forward'):
            require(len(raw[phase+'_segments'])==h,'all completed derivative forward boundaries')
            before=packet['root_context']
            for after in raw[phase+'_segments']:history(before,after,np.zeros((5,2)),normalization);before=after
        require(same(raw['nominal'],raw['nominal_segments'][-1]['latent_history'][-1]),'nominal terminal content')
        require(same(raw['nominal_segments'],raw['jacobian_forward_segments']),'identical zero nominal/AD forward contexts')
        for qname in METRICS:
            q=projection(record['metrics'][qname]['projection']);own=c['metrics'][qname][str(h)]
            k=q@j;r=q@(goal-n)
            near(own['response'],k,'own-Q action response');near(own['residual'],r,'own-Q goal residual')
            u,s,_=np.linalg.svd(k,full_matrices=False)
            rank=int(np.count_nonzero(s>s[0]*1e-6)) if s[0]>0 else 0
            norm=float(r@r);chi=0. if norm==0 else float(np.clip(np.sum((r-u[:,:rank]@(u[:,:rank].T@r))**2)/norm,0,1))
            signal=own['signal'];require(signal['horizon']==h and signal['rank']==rank,'signal H/rank')
            near(signal['chi'],chi,'residual chi');near(signal['residual_squared_norm'],norm);near(signal['singular_values'],s)
            sol=own['solve'];d=array(sol['params'],(10*h,));mult=sol['ball_multiplier'];radius=.75*np.sqrt(5*h)
            require(type(mult) in (int,float) and math.isfinite(mult) and mult>=0,'nonnegative ball multiplier')
            require(np.linalg.norm(d)<=radius+1e-9,'bounded ridge feasibility')
            near(k.T@(k@d-r)+(.002+mult)*d,np.zeros(10*h),'ridge KKT stationarity',atol=1e-8)
            near(mult*(float(d@d)-radius**2),0.,'ridge complementary slackness',atol=1e-8)
            remaining=r-k@d
            for key,value in dict(radius=radius,norm=float(np.linalg.norm(d)),linear_terminal_cost=float(remaining@remaining/192),
                ridge_penalty=float(.002*(d@d)/192),own_reach_value=float((remaining@remaining+.002*(d@d))/192)).items():near(sol[key],value,'ridge '+key)
            near(sol['singular_values'],s);require(same(own['proposal_logits'],sol['params']),'ridge logits identity')
            near(own['raw_proposal'],np.tanh(d).reshape(5*h,2),'ridge raw proposal')
            signals[qname][str(h)]=chi
    return signals


def check_arm(record,arm,source,compiled,normalization):
    require(arm in ARMS and source['arm']==arm and source['status']=='completed','complete known arm')
    qname,schedule=arm.split('__');packet=record['packet'];q=projection(record['metrics'][qname]['projection'])
    require(source['active_metric_id']==record['metrics'][qname]['active_metric_id'] and
        source['source_input_sha256']==packet['input_sha256'],'same root/input/goal/active metric binding')
    sig=compiled['metrics'][qname];alloc=allocation(schedule,sig['2']['signal']['chi'],sig['3']['signal']['chi'])
    require(same(alloc,source['allocation']),'frozen chi allocation')
    counts=alloc['candidates'];require(len(source['candidates'])==sum(counts.values()),'complete candidate count no deduplication')
    expected_keys=[(h,i) for h in (2,3) for i in range(counts[str(h)])]
    require([(c['horizon'],c['index']) for c in source['candidates']]==expected_keys and
        all(type(c['horizon']) is int and type(c['index']) is int for c in source['candidates']),'candidate H/index coverage/order')
    rebuilt_costs=[];pos=0
    for h in (2,3):
        for i,actions in enumerate(bank(record['seed_index'],h,counts[str(h)],sig[str(h)]['proposal_logits'])):
            c=source['candidates'][pos];pos+=1
            require(c['completed'] is True and same(c['actions'],actions.tolist()),'fresh own-Q ridge/unchanged random pool index')
            logits=sig[str(h)]['proposal_logits'] if i==1 else np.arctanh(actions).reshape(-1).tolist()
            require(same(c['logits'],logits),'saved logits and coordinates')
            history(packet['root_context'],c['context'],actions,normalization)
            cost=own_cost(c['context']['latent_history'][-1],packet['goal'],q);near(c['cost'],cost,'own-Q nonlinear candidate cost')
            rebuilt_costs.append(dict(horizon=h,index=i,cost=cost))
    winner=min(source['candidates'],key=lambda c:(c['cost'],c['horizon'],c['index']))
    require(same(source['winner'],winner),'stable own-Q minimum with original candidate copy')
    require(source['whole_and_segmented_exact'] is True and source['verification_predicts']==2*winner['horizon'] and
        same(source['whole_replay_context'],winner['context']) and same(source['segmented_replay_context'],winner['context']),'winner independent whole+segmented full-context exact')
    history(packet['root_context'],source['whole_replay_context'],winner['actions'],normalization)
    require(len(source['segments'])==winner['horizon'],'all replay segments')
    before=packet['root_context']
    for i,after in enumerate(source['segments']):history(before,after,winner['actions'][5*i:5*i+5],normalization);before=after
    require(same(before,winner['context']),'final segmented endpoint')
    return dict(arm=arm,metric=qname,allocation=alloc,winner_horizon=winner['horizon'],winner_index=winner['index'],
        own_Q_cost=own_cost(winner['context']['latent_history'][-1],packet['goal'],q),
        candidate_count=len(rebuilt_costs),candidate_costs=rebuilt_costs,
        duplicate_action_records_retained=len(source['candidates'])-len({digest(c['actions']) for c in source['candidates']}),
        search_predicts=sum(c['horizon'] for c in source['candidates']),verification_predicts=2*winner['horizon'])


def check_journal(record,row,events,arm_order):
    """Consume exact expected durable sequence, including all neural call pairs."""
    require(len(events)<=2500,'bounded root journal events')
    position=0;calls=[];counts=Counter();roles=Counter()
    def take(expected):
        nonlocal position
        require(position<len(events) and same(events[position],expected),'journal order/content at '+str(position))
        position+=1
    def forwards(n,role):
        for _ in range(n):
            for kind in ('action_encoder','predict'):
                e=dict(kind=kind,role=role,index=len(calls),context_tokens=3,completed=False)
                take(e);e=dict(e,completed=True);take(e);calls.append(e);roles[role,kind]+=1;counts[kind]+=1
    take(dict(kind='root_context_arrays',root_sha256=row['root_sha256'],context=record['packet']['root_context']))
    c=row['compilation']
    for h in (2,3):
        raw=c['raw'][str(h)]
        for i,ctx in enumerate(raw['nominal_segments']):
            forwards(1,'shared');take(dict(kind='compile_boundary_arrays',phase='nominal',horizon=h,segment=i+1,context=ctx))
        take(dict(kind='nominal_array',horizon=h,nominal=raw['nominal']))
        jac=dict(kind='jacobian',horizon=h,vjp_directions=192,completed=False);take(jac)
        for i,ctx in enumerate(raw['jacobian_forward_segments']):
            forwards(1,'shared');take(dict(kind='compile_boundary_arrays',phase='jacobian_forward',horizon=h,segment=i+1,context=ctx))
        take(dict(kind='jacobian_array',horizon=h,jacobian=raw['jacobian']));take(dict(jac,completed=True));counts['jacobians']+=1
        for q in METRICS:
            own=c['metrics'][q][str(h)];snapshot={k:own[k] for k in ('response','residual')}
            take(dict(metric=q,kind='response_arrays',horizon=h,arrays=dict(snapshot)))
            for role in ('signal','ridge'):
                svd=dict(metric=q,kind='svd',role=role,horizon=h,completed=False);take(svd)
                for key in (('signal',) if role=='signal' else ('solve','proposal_logits','raw_proposal')):snapshot[key]=own[key]
                take(dict(metric=q,kind='response_arrays',horizon=h,arrays=dict(snapshot)))
                take(dict(svd,completed=True));counts['execution_svds']+=1
    require(set(arm_order)==set(ARMS) and len(arm_order)==4,'all arm execution order')
    for arm in arm_order:
        source=row['arms'][arm]
        for candidate in source['candidates']:
            initial={k:candidate[k] for k in ('horizon','index','actions','logits')}
            take(dict(kind='candidate',**initial,completed=False));forwards(candidate['horizon'],arm)
            take(dict(kind='candidate_context_arrays',horizon=candidate['horizon'],index=candidate['index'],context=candidate['context']))
            take(dict(kind='candidate',**candidate));counts['candidate_records']+=1
        forwards(source['winner']['horizon'],arm)
        take(dict(kind='whole_replay_context',arm=arm,context=source['whole_replay_context']))
        for i,ctx in enumerate(source['segments']):
            forwards(1,arm);take(dict(kind='segmented_replay_context',arm=arm,segment=i+1,context=ctx))
    require(position==len(events),'no missing, extra or uncharged journal operations')
    receipt=row['neural_receipt']
    require(same(receipt['events'],calls) and receipt['blocked']==0 and receipt['cleanup_errors']==receipt['journal_errors']==[],'actual module receipt equality')
    for kind in ('predict','action_encoder'):
        require(receipt['attempted'][kind]==receipt['completed'][kind]==counts[kind]<=274,'root independent module caps')
        require(roles['shared',kind]==10,'shared nominal+AD forward counted once')
        for arm in ARMS:require(roles[arm,kind]==60+row['arms'][arm]['verification_predicts']<=66,'matched search plus actual H verification')
    require(counts['jacobians']==2 and counts['execution_svds']==8,'all fresh AD/SVD operations')
    return dict(counts)


def load_bundle(root):
    root=Path(root).resolve();e=io.Evidence(root);directory=root/INPUT
    report=e.json(directory/'report.json',REPORT_SHA);binding=e.json(directory/'execution_protocol.json')
    require(binding['prepare_sha256']==PREPARE_SHA,'fixed actual preparation')
    prepared=Path(binding['prepared']);receipt=e.json(prepared/'prepare.json',PREPARE_SHA)
    p=e.json(prepared/'protocol.json',receipt['protocol_sha256'])
    require(p['schema']==SCHEDULER+'.protocol' and receipt['status']=='prepared_not_executed' and
        digest(p)==receipt['protocol_id']==report['protocol_id'],'frozen protocol identity')
    for seal in receipt['source_seals']:e.seal(seal)
    for path,h in {**p['sources_sha256'],**p['bindings']}.items():e.raw(Path(path),h)
    events={}
    require(len(report['roots'])<=4,'bounded root count')
    for i,row in enumerate(report['roots']):
        require(same(row,e.json(directory/f'root_{i}.json')),'root file/report equality')
        path=directory/f'root_{i}.events.jsonl';require(row['journal_path']==str(path),'fixed journal own path')
        events[row['root_sha256']]=[io.decode(line) for line in e.raw(path,row['journal_sha256']).splitlines()]
    e.recheck();return dict(protocol=p,report=report,events=events),e


def compare(root,metric,arms):
    u,d=(arms[metric+'__'+s] for s in ('uniform','dynamic'))
    require(u['active_metric_id']==d['active_metric_id'] and u['source_input_sha256']==d['source_input_sha256'],'scheduler comparison same own Q/input/root')
    coverage=[]
    for h in (2,3):
        us={c['index']:c for c in u['candidates'] if c['horizon']==h};ds={c['index']:c for c in d['candidates'] if c['horizon']==h}
        shared=sorted(us.keys()&ds.keys())
        require(all(same(us[k],ds[k]) for k in shared),'common candidate action/context/cost identical')
        coverage.append(dict(horizon=h,uniform_indices=sorted(us),dynamic_indices=sorted(ds),shared_indices=shared,
            uniform_only_indices=sorted(us.keys()-ds.keys()),dynamic_only_indices=sorted(ds.keys()-us.keys()),
            uniform_best_own_Q_cost=min(c['cost'] for c in us.values()),dynamic_best_own_Q_cost=min(c['cost'] for c in ds.values())))
    hu,hd=u['winner']['horizon'],d['winner']['horizon']
    return dict(root_sha256=root,metric=metric,active_metric_id=u['active_metric_id'],stratum='same_H' if hu==hd else 'mixed_H',
        uniform_winner_H=hu,dynamic_winner_H=hd,uniform_own_Q_cost=u['winner']['cost'],dynamic_own_Q_cost=d['winner']['cost'],
        dynamic_minus_uniform_same_Q_cost=d['winner']['cost']-u['winner']['cost'],
        candidate_coverage=coverage,search_predicts_each=60,
        uniform_verification_predicts=2*hu,dynamic_verification_predicts=2*hd,
        interpretation='model-only own-Q comparison; mixed-H not an equal-length effect' if hu!=hd else 'model-only own-Q same-H selected comparison; candidate sets still differ')


def audit(b):
    p,r=b['protocol'],b['report']
    require(r['schema']==SCHEDULER+'.report' and r['status']=='cuda_completed' and r['errors']==[] and r['model_unchanged'] is True,'successful saved CUDA report')
    require(r['contract']==p['contract'] and r['new_physical_steps']==0 and
        all(r[k] is False for k in ('P1_admission','P2_admission','P3_admission','P4_admission')),'no physical/admission claims')
    require(p['contract']['total_resource_matched'] is False and p['contract']['search_predicts_per_arm']==60 and
        p['contract']['damping']==.002 and p['contract']['horizons']==[2,3],'fixed bounded contract')
    records={x['packet']['root_sha256']:x for x in p['records']}
    require(len(records)==len(p['records'])==4,'four distinct original roots')
    rows=io.index(r['roots'],('root_sha256',));require(set(rows)=={(x,) for x in records},'all original roots exactly once')
    totals=Counter();arm_rows=[];pairs=[]
    for i,record in enumerate(p['records']):
        packet=record['packet'];root=packet['root_sha256'];row=rows[root,]
        require(record['seed_index']==i and row['status']=='completed' and set(row['arms'])==set(ARMS),'four complete arms per root')
        require(packet['input_sha256']==digest({k:packet[k] for k in ('root_sha256','root_context','goal','projection')}),'source native root/context/goal hash')
        require(set(record['metrics'])==set(METRICS) and same(record['metrics']['old_Q']['projection'],packet['projection']),'original old Q and complete metric set')
        for q in METRICS:
            identity=dict(root_sha256=root,source_input_sha256=packet['input_sha256'],label=q,
                projection=array_id(record['metrics'][q]['projection']),goal=array_id(packet['goal']),provenance=record['provenance'])
            require(digest(identity)==record['metrics'][q]['active_metric_id'],'source metric identity')
        signals=check_compilation(record,row['compilation'],p['normalization'])
        require(p['arm_order'][i]==list(ARMS[i:]+ARMS[:i]),'balanced frozen arm order')
        counts=check_journal(record,row,b['events'][root],p['arm_order'][i]);totals.update(counts)
        for arm in ARMS:
            result=check_arm(record,arm,row['arms'][arm],row['compilation'],p['normalization'])
            q,schedule=arm.split('__')
            rebuilt_alloc=allocation(schedule,signals[q]['2'],signals[q]['3'])
            require(rebuilt_alloc['blocks']==result['allocation']['blocks'],'recomputed SVD chi same schedule blocks')
            require(result['search_predicts']==60,'exact search cost H-weighted candidate coverage')
            arm_rows.append(dict(root_sha256=root,seed_index=i,**result))
        pairs.extend(compare(root,q,row['arms']) for q in METRICS)
    verification=sum(x['verification_predicts'] for x in arm_rows)
    require(totals['predict']==totals['action_encoder']==40+960+verification<=1096,'all forward costs counted exactly once')
    require(r['resources']['attempted']==dict(predict=totals['predict'],action_encoder=totals['action_encoder']) and
        r['resources']['incomplete_counts_are_lower_bound'] is False and r['resources']['backward_engine_api_calls'] is None,'global resource receipt')
    require(r['source_validation_calls']==dict(attempted=2,completed=2) and
        r['resources']['completed_source_validation_svds']==32 and r['resources']['validation_saved_array_svds_per_source_material_call']==16,'separate source validation SVDs')
    require(same(r['resources']['inherited'],p['inherited_resources']),'inherited resource accounting not zero')
    strata=[]
    for q in METRICS:
        for stratum in ('same_H','mixed_H'):
            group=[x for x in pairs if x['metric']==q and x['stratum']==stratum]
            deltas=[x['dynamic_minus_uniform_same_Q_cost'] for x in group]
            strata.append(dict(metric=q,stratum=stratum,count=len(group),
                negative=sum(x<0 for x in deltas),zero=sum(x==0 for x in deltas),positive=sum(x>0 for x in deltas),
                pooled_cost_delta=None,reason='per-root own-Q differences only; no pooled magnitude across root-specific Qs'))
    return dict(schema=NAME+'.report',status='saved_arithmetic_verified',all_checks=True,
        denominator=dict(roots=4,arms=16,scheduler_pairs=8,new_independent_roots=0),rows=arm_rows,pairs=pairs,strata=strata,
        budget=dict(predict=totals['predict'],action_encoder=totals['action_encoder'],shared_predict_included=40,
            nominal_predict_included=20,ad_forward_predict_included=20,search_predict_included=960,
            verification_predict_included=verification,jacobians=totals['jacobians'],execution_svds=totals['execution_svds'],
            source_validation_svds=32,mathematical_vjp_directions=1536,jacobian_api_calls=8,backward_engine_api_calls=None,
            candidate_records=totals['candidate_records'],four_root_predict_cap=1096),
        inherited_resources=p['inherited_resources'],audit_saved_array_svds=16,new_neural_calls=0,new_physical_steps=0,
        physical_improvement_claim=False,total_flops_matched=False,P1_admission=False,P3_admission=False,
        gaps=['Four previously exposed calibration roots only; not independent P3 admission',
            'No physics/endpoint encoder report: all costs are model-predicted own-Q only',
            'Mixed-H comparisons change path length; same-H winner comparisons still have different candidate coverage',
            'Never subtract old-Q from goal-patch-Q costs or pool numerical deltas across different Qs',
            'Saved latent/Jacobian truth and actual AD connectivity not independently regenerated; model unchanged is inherited executor claim',
            'Cached input history cost unknown; historical metric/planner costs inherited separately, not full-pipeline or total-FLOPs equivalence',
            'Successful pinned report required; malformed/partial inputs rejected, never silently omitted or charged zero'])


def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(io.canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())


def run(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW output')
    b,e=load_bundle(root);result=audit(b);e.recheck()
    sources=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md'),Path(io.__file__).resolve()]
    output.mkdir(exist_ok=False)
    seal=create_seal(files=sources,args_json=io.canonical(dict(report_sha256=REPORT_SHA)),output_dir=output/'source_seal')
    result.update(input_report_sha256=REPORT_SHA,prepare_sha256=PREPARE_SHA,inputs_sha256=dict(sorted(e.hashes.items())),source_seal=seal)
    write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    r=run(args.project_root,args.output)
    print(io.canonical({k:r[k] for k in ('status','all_checks','denominator','budget','strata')}))


if __name__=='__main__':main()
