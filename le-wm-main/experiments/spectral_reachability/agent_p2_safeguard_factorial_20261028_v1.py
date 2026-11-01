"""Saved-only safeguard factorial inventory and exact missing-forward prepare.

No neural/physics runner: MAIN must bind a reviewed durable native dispatcher.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v2 as v2

NAME='agent_p2_safeguard_factorial_20261028_v1'
ROOT=v2.ROOT
v1=v2.old
require,sha,raw=v2.require,v2.sha,v2.raw
REPORTS=('8146692ccc9a3c2c6c09ff3c99419274e2e8a8ff86bb3ac0465af01dc6401eea',
         '363925ff3714c6c9a5fbd2ec6c59e86dbcb63fe685cde1945d292ec1a9196117')
SOURCES=(v2.OLD_SHA,'e459cdc5de507353873346d6f86bc223afed291982904c4c4c6ce9bdfb7eaeab')
PHYSICAL=('c191b8076469f78fb7e02c793e0b061bcf6706266b1af6e90f3944748db2e11d',
          '7e71cf504691cb1b132c437267319f3291a62b0f6c9f839f0652b42a974f8de8')
ANCHORS=('zero','incumbent')


def contract():
    return dict(roots=2,branches=8,cells=32,anchors=list(ANCHORS),guard=[False,True],
        alphas=list(v2.ALPHAS),proposal_slots=64,incumbents=8,
        factor_A='anchor only; same half-radius gradient proposal formula and exact float64 tanh chart',
        factor_B='stable argmin four proposals only vs incumbent(index0)+same four proposals',
        no_guard='best fixed-alpha proposal; NOT unconditional alpha1 and NOT original ridge algorithm',
        chart='NumPy tanh(float64 logits).astype(float32); no inverse/rounding repair/clip',
        domain='original fulljoint20 ball plus exact remaining suffix radius and strict raw(-1,1)',
        Q='original identity192 terminal MSE',goal='unchanged',
        zero_J='original tanh(float32 logits) AD at literal zero; local tanh derivative=1, equals raw J there',
        new_jacobian=0,new_encoder=0,new_physics=0,max_new_predict=64,
        accounting='saved inherited counts separate; inclusive API and module axes not added',
        total_FLOPs=None,equal_FLOPs=False,wall_seconds=900,output_cap_bytes=64*1024**2,
        final_reserve_bytes=2*1024**2,disk_floor_bytes=200*1024**2,
        run_available=False,physical_release=False,P1_admission=False,P2_admission=False,fresh=False,
        legacy_comparison_is_factorial=False,scientific_pass=False)


def cost(b,context):
    z=np.asarray(context['latent_history'][-1],np.float64);g=np.asarray(b['goal'],np.float64)
    require(z.shape==g.shape==(192,) and np.isfinite(z).all() and np.isfinite(g).all()
        and np.array_equal(b['q'],np.eye(192)),'original finite identity192 goal')
    return float(np.mean((z-g)**2))


def candidate(b,commands,context,origin):
    bits=v2.suffix_bits(commands)
    v2.F.planner.advance(b,b['observed_context'],commands,context)
    return dict(commands=deepcopy(commands),raw_bits=bits,context=deepcopy(context),
        context_sha256=v2.F.core.digest(b['observed_context']),goal=deepcopy(b['goal']),
        cost=cost(b,context),origin=deepcopy(origin))


def proposals(b,anchor,z,jac):
    """Same v2 formula; only the linearization anchor differs, no outcomes."""
    require(anchor in ANCHORS,'fixed anchor')
    local=deepcopy(b)
    if anchor=='zero':local['incumbent_logits']=np.zeros(10).tolist()
    p=v2.proposals(local,z,jac)
    return dict(anchor=anchor,gradient=p['gradient'],direction=p['direction'],slope=p['slope'],
        descent_used=p['descent_used'],logits=p['logits'][1:])


def find_prediction(b,commands,catalog):
    bits=v2.suffix_bits(commands);matches=[x for x in catalog if x['raw_bits']==bits]
    if not matches:return None
    for m in matches:
        require(m['context_sha256']==v2.F.core.digest(b['observed_context']) and m['goal']==b['goal']
            and m['cost']==cost(b,m['context']),'exact same causal prediction and goal')
        require(m['context']==matches[0]['context'] and m['cost']==matches[0]['cost'],'duplicate raw predictions agree exactly')
    return dict(context=deepcopy(matches[0]['context']),cost=matches[0]['cost'],
        authorities=[deepcopy(m['origin']) for m in matches])


def choose(incumbent,items,guard):
    """Only model cost and literal action inputs; no native labels accepted."""
    require(type(guard) is bool and len(items)==4,'fixed four alpha slots')
    exact={'index','commands','raw_bits','prediction'}
    require(all(set(x)==exact and x['index']==i+1 for i,x in enumerate(items)),'exact model-only proposal ABI')
    require(set(incumbent)=={'commands','raw_bits','prediction'} and incumbent['prediction'] is not None,'sealed incumbent prediction')
    if any(x['prediction'] is None for x in items):return None
    pool=([dict(incumbent,index=0)] if guard else [])+items
    for x in pool:
        require(v2.suffix_bits(x['commands'])==x['raw_bits'] and np.isfinite(x['prediction']['cost']),'literal finite model candidate')
    best=min(pool,key=lambda x:(x['prediction']['cost'],x['index']))
    return dict(selected_index=best['index'],commands=deepcopy(best['commands']),raw_bits=best['raw_bits'],
        model_Q=best['prediction']['cost'],guard=guard,incumbent_Q=incumbent['prediction']['cost'])


def physical_matches(commands,plans):
    """Identity only; deliberately no outcome field is read or copied."""
    bits=v2.physical.bits(commands)
    return [deepcopy(x['authority']) for x in plans if v2.physical.bits(x['commands'])==bits]


def branch_table(i,b,r1,r2,plan,load1,load2,executed):
    catalog=[]
    for version,row in ((1,r1),(2,r2)):
        for j,e in enumerate(row['evaluations']):
            c=candidate(b,e['commands'],e['context'],dict(version=version,report_sha256=REPORTS[version-1],evaluation=j))
            require(e['cost']==c['cost'],'saved native model Q exactly recomputed');catalog.append(c)
    incumbent=dict(commands=deepcopy(b['old_suffix_commands']),raw_bits=v2.suffix_bits(b['old_suffix_commands']),
        prediction=find_prediction(b,b['old_suffix_commands'],catalog))
    anchor_data=dict(zero=(load1(r1['arrays']['nominal']),load1(r1['arrays']['jacobian'])),
        incumbent=(r2['evaluations'][0]['context']['latent_history'][-1],load2(r2['arrays']['jacobian'])))
    axes=[];requests=[]
    for anchor in ANCHORS:
        p=proposals(b,anchor,*anchor_data[anchor]);items=[]
        if anchor=='incumbent':
            require(p['logits']==r2['proposal']['logits'][1:] and p['gradient']==r2['proposal']['gradient'],
                'exact frozen v2 incumbent proposal replay')
        for j,u in enumerate(p['logits'],1):
            require(v2.legality(b,u)['legal'],'all raw/logit/joint caps; no tolerance/clip')
            commands=v2.chart(u).astype(float).tolist();prediction=find_prediction(b,commands,catalog)
            item=dict(index=j,commands=commands,raw_bits=v2.suffix_bits(commands),prediction=prediction);items.append(item)
            if prediction is None:
                requests.append(dict(branch=i,anchor=anchor,alpha_index=j,alpha=v2.ALPHAS[j-1],
                    logits=u,domain=v2.legality(b,u),commands=commands,raw_bits=item['raw_bits'],
                    input_context=deepcopy(b['observed_context']),context_sha256=v2.F.core.digest(b['observed_context']),
                    raw_history10=deepcopy(b['observed_raw_history']),goal=deepcopy(b['goal']),
                    action_mean=deepcopy(b['action_mean']),action_std=deepcopy(b['action_std']),
                    native_input=v2.F.planner.incoming(b,b['observed_context'],commands),
                    dispatch=dict(backend='frozen F.d2p.TorchBackend.step',arguments='deepcopy(input_context), rawfloat32[5,2]',
                        modules=['action_encoder','predict','pred_proj'],CUDA_only=True,expected_modules_each=1,predict_API=1)))
        cells=[]
        for guarded in (False,True):
            selected=choose(incumbent,items,guarded)
            refs=[] if selected is None else physical_matches(b['prefix_commands']+selected['commands'],executed)
            cells.append(dict(guard=guarded,selection=selected,model_status='missing_predictions' if selected is None else 'saved_complete',
                physical_status='unknown_model_selection' if selected is None else 'sealed_receipt_exact_full10_reference' if refs else 'missing_exact_branch',
                physical_references=refs,native_metrics=None))
        axes.append(dict(anchor=anchor,proposal=p,candidates=items,cells=cells))
    old=r1['evaluations'][0];new=r1['evaluations'][1]
    legacy=[]
    for label,guarded,index in (('zero_ridge',False,1),('zero_ridge',True,0 if old['cost']<=new['cost'] else 1),
        ('incumbent_gradient',False,min(range(1,5),key=lambda j:(r2['evaluations'][j]['cost'],j))),
        ('incumbent_gradient',True,r2['selected_index'])):
        row=r1 if label=='zero_ridge' else r2;e=row['evaluations'][index]
        legacy.append(dict(algorithm=label,guard=guarded,evaluation=index,model_Q=e['cost'],
            raw_bits=v2.suffix_bits(e['commands']),physical_references=physical_matches(b['prefix_commands']+e['commands'],executed)))
    return dict(index=i,root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'],task_id=plan['task_id'],
        context_sha256=v2.F.core.digest(b['observed_context']),goal_sha256=v2.F.core.digest(b['goal']),
        prefix_raw_bits=v2.suffix_bits(b['prefix_commands']),incumbent=incumbent,
        legacy_confounded_reconstruction=legacy,controlled_axes=axes,missing_requests=requests)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def authority(directory,version):
    m=(v1,v2)[version-1];p=Path(directory).resolve()
    require(sha(m.__file__)==SOURCES[version-1] and sha(p/'report.json')==REPORTS[version-1],'sealed fixed model/compiler authority')
    prepared,plans=m.consume_compile(p,REPORTS[version-1]);return prepared,plans,v2.physical.read(p/'report.json')


def build(zero_output,incumbent_output,zero_physical_receipt,incumbent_physical_receipt):
    p1,plans1,r1=authority(zero_output,1);p2,plans2,r2=authority(incumbent_output,2)
    require(p1['expected_model_hash']==p2['expected_model_hash'] and p1['plans']==p2['plans']
        and len(p1['branches'])==len(p2['branches'])==8,'same frozen model and original plans, fixed8')
    receipts=[];bindings=dict(p1['bindings']);v2.physical.merge(bindings,p2['bindings'])
    for i,(output,r,receipt) in enumerate(zip((zero_output,incumbent_output),(r1,r2),
        (zero_physical_receipt,incumbent_physical_receipt),strict=True)):
        path=Path(receipt).resolve();require(sha(path)==PHYSICAL[i],'actual MAIN physical strict receipt SHA')
        x=v2.physical.read(path)
        require(x['feedback_compile_sha256']==REPORTS[i] and x['fixed_denominator']==8 and len(x['rows'])==8
            and x['P1_admission'] is x['P2_admission'] is x['physical_release'] is False,'native receipt fixed8 identity')
        receipts.append(x);v2.physical.merge(bindings,{str(path):PHYSICAL[i],str(Path(output).resolve()/'report.json'):REPORTS[i],
            r['prepared']:r['prepare_sha256']})
        v2.physical.merge(bindings,{str(Path(output).resolve()/name):h for name,h in r['artifact_sha256'].items()})
    rows=[]
    for i,(b1,b2,plan) in enumerate(zip(p1['branches'],p2['branches'],p2['plans'],strict=True)):
        require(all(b1[k]==b2[k] for k in b1 if k!='branch_id'),'same original actual-mid/history/goal/prefix/domain')
        executed=[]
        for j,plans in enumerate((plans1,plans2)):
            receipt=receipts[j]['rows'][i]
            require(receipt['index']==i and receipt['intervention']['selection_digest']==plans[i]['selection_digest']
                and receipt['intervention']['compiler_selection_digest']==plans[i]['compiler_selection_digest']
                and receipt['control']['selection_digest']==plan['selection_digest'],'exact selected native receipt plan join')
            for role,current in (('intervention',plans[i]),('control',plan)):
                require(all(receipt[role][k]==current[k] for k in ('root_slot','condition','arm')),'native task slot identity')
                executed.append(dict(commands=current['commands'],authority=dict(kind=role,receipt_sha256=PHYSICAL[j],
                    index=i,selection_digest=current['selection_digest'],physical_report_sha256=receipts[j]['physical_report_sha256'])))
        rows.append(branch_table(i,b2,r1['rows'][i],r2['rows'][i],plan,
            lambda e:v1.load_npy(zero_output,e),lambda e:v1.load_npy(incumbent_output,e),executed))
    requests=[q for row in rows for q in row['missing_requests']]
    for i,q in enumerate(requests):q['request_index']=i
    v2.physical.merge(bindings,{str(f):sha(f) for f in own_files()});v2.bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',status='blocked_missing_model_candidates' if requests else 'saved_model_factorial_complete_physical_not_released',
        arguments=dict(zero_output=str(Path(zero_output).resolve()),incumbent_output=str(Path(incumbent_output).resolve()),
            zero_physical_receipt=str(Path(zero_physical_receipt).resolve()),incumbent_physical_receipt=str(Path(incumbent_physical_receipt).resolve())),
        contract=contract(),rows=rows,requests=requests,expected_model_hash=p2['expected_model_hash'],policy=p2['policy'],bindings=bindings,
        new_compute=dict(phase='planned_not_executed',predict=len(requests),action_encoder=len(requests),pred_proj=len(requests),predict_API_inclusive=len(requests),
            jacobian=0,AD=0,visual_encoder=0,physics=0,total_FLOPs=None),
        actual_new_compute=None,worker_neural_calls=0,
        inherited_compute=[dict(report_sha256=h,resources=[x['resources'] for x in r['rows']],
            AD_API=[x['AD_API_resources'] for x in r['rows']],predict_API=[x['predict_API_resources'] for x in r['rows']])
            for h,r in zip(REPORTS,(r1,r2),strict=True)],
        blockers=['new fixed-alpha zero-anchor candidates require actual frozen-model forward and durable native trace'] if requests else [],
        remaining_execution_dependencies=['reviewed durable CUDA dispatch/entry/return adapter for these exact requests',
            'all candidate outputs strict validation before stable model-only selections',
            'any selected full10 without exact sealed native receipt needs separate MAIN physics authorization'],
        physical_metrics_consumed=False,physical_receipt_verification='pinned MAIN strict receipt reference, not rerun native PNG audit',
        P1_admission=False,P2_admission=False,physical_release=False)


def verify(path,expected):
    require(sha(path)==expected,'actual own prepare SHA');p=v2.physical.read(path)
    require(raw(p)==raw(build(**p['arguments'])),'complete saved inventory and missing-request reconstruction');return p


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink()
        and len(payload)<=32*1024**2 and shutil.disk_usage(path.parent).free>=200*1024**2+len(payload),'new bounded own metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'complete metadata');f.flush();os.fsync(f.fileno())
    v2.kernel.paired.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value['status'])


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify'))
    for k in ('zero-output','incumbent-output','zero-physical-receipt','incumbent-physical-receipt','output','prepared','prepare-sha'):q.add_argument('--'+k)
    a=q.parse_args()
    if a.command=='prepare':result=write(a.output,build(a.zero_output,a.incumbent_output,a.zero_physical_receipt,a.incumbent_physical_receipt))
    else:p=verify(a.prepared,a.prepare_sha);result=dict(status=p['status'],branches=len(p['rows']),missing_forward=len(p['requests']))
    print(raw(result).decode(),end='',flush=True)


if __name__=='__main__':main()
