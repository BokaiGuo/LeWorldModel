"""Independent saved-data audit of actual GN fixed-segment 32-call diagnostic.

No producer execution/checker, neural model, image encoding, or simulator.
Reconstruct every operation, four frozen trials, same-Q cost and full H2 path.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME='p1_gn_step_diagnostic_saved_audit_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
ACTUAL=ROOT/'results/p1_gn_step_diagnostic_cuda_20261027_v1_cuda_main01'
REPORT_SHA='2636b3a89cbe96b97d64dce853f506a837e4191e05b90dd5b754146fde658848'
PREPARE_SHA='76a4248cd61d1d6e2a0056e6eb6cb9c41ef1b2b7fd5bd6ba07770a9ecbfa01ec'
SCALES=(.5,.25,.125,.0625)
FLOOR=200*1024**2
LIMIT=100*1024


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular nonsymlink input')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


class Reader:
    def __init__(self):self.bindings={}
    def bind(self,path,expected=None):
        p=Path(path).absolute();h=sha(p)
        require(expected is None or expected==h,'SHA mismatch: '+str(p))
        require(str(p) not in self.bindings or self.bindings[str(p)]==h,'input changed while reading')
        self.bindings[str(p)]=h
        return p
    def read(self,path,expected=None):
        p=self.bind(path,expected);require(p.stat().st_size<32*1024**2,'bounded file')
        b=p.read_bytes()
        if p.name.endswith('.gz'):b=gzip.decompress(b)
        require(len(b)<32*1024**2,'bounded expanded JSON')
        value=json.loads(b);require(sha(p)==self.bindings[str(p)],'input changed during JSON read')
        return value
    def recheck(self):
        for p,h in self.bindings.items():require(sha(p)==h,'input changed during audit')


def array(value,shape):
    a=np.asarray(value);require(a.shape==shape and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite aligned array')
    return a.astype(float)


def cost(task,terminal):
    residual=array(task['projection'],(192,192))@(array(terminal,(192,))-array(task['goal'],(192,)))
    return float(np.mean(residual*residual))


def segment(x,y,s,cap):
    require(s in SCALES,'fixed scale')
    x,y=array(x,(2,10)),array(y,(2,10))
    require(all(np.array_equal(a,a.astype(np.float32)) and (np.linalg.norm(a,axis=1)<=cap).all() for a in (x,y)),
            'original feasible float32 endpoints')
    z=(x+s*(y-x)).astype(np.float32)
    require((np.linalg.norm(z.astype(float),axis=1)<=cap).all(),'float32 segment remains in original balls')
    return z.astype(float).tolist()


def path(task,logits,commands,contexts):
    x=array(logits,(2,10));raw=array(commands,(10,2))
    require(np.array_equal(x,x.astype(np.float32)) and (np.linalg.norm(x,axis=1)<=task['packet_logit_cap']).all(),'float32 original action domain')
    require(np.array_equal(raw,np.tanh(x.astype(np.float32)).reshape(10,2)),'exact tanh raw commands')
    require(len(contexts)==2,'complete full H2, not partial')
    tokens=(raw.astype(np.float32).reshape(2,10)-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(np.asarray(task['action_std'],np.float32),5)
    before=task['root_context']
    for k,c in enumerate(contexts):
        core.context(c)
        require(c['latent_history'][:2]==before['latent_history'][1:],'full latent history chronology')
        require(np.array_equal(c['action_history'],np.concatenate((np.asarray(before['action_history']),tokens[k:k+1]))[-2:]),
                'exact normalized incoming actions and order')
        before=c
    return cost(task,before['latent_history'][-1])


def check_arrays(p,report,events,evaluations,selections,results,tasks,components):
    """Pure in-memory checker; tests mutate copies, never disk fixtures."""
    require(p['scales']==list(SCALES) and report['scales']==list(SCALES),'all four frozen scales')
    require(p['predict_cap']==p['action_encoder_cap']==32 and p['per_root_cap']==16 and p['full_H2_evaluations']==16,'frozen 32-call protocol')
    require(all(p[k]==0 for k in ('new_AD_cap','new_image_cap','new_physics_cap')),'no new AD/image/physics budget')
    require(len(p['roots'])==2 and len(report['rows'])==2 and len(evaluations)==16 and len(events)==192,'complete two-root census')
    cursor=0;entries=[];rows=[];seen=set()
    def event(want):
        nonlocal cursor
        require(cursor<len(events) and events[cursor]==want,'exact ordered event '+str(cursor)+' '+want['kind'])
        cursor+=1
    for i,root in enumerate(p['roots']):
        require(root['root_index']==i and root['root_sha256'] not in seen,'stable distinct root order');seen.add(root['root_sha256'])
        encoded=tasks[i];t=encoded['task'];component=components[i]
        core.validate_task(t)
        require(encoded['task_id']==root['task_id']==core.task_id(t) and encoded['root_sha256']==root['root_sha256'],'exact task/root identity')
        require(component['task_id']==root['task_id'] and component['configuration']==dict(method='full_H2_GN',B1=4,B2=8,
            initial_logits=[np.zeros((2,10)).tolist()],step_scale=1.,damping=.002,extra_line_search_trials=0),'original B8 GN endpoint source')
        run=component['direct_runs'];require(len(run)==1 and len(run[0])==9,'complete saved B8 trajectory')
        incumbent=run[0][7]['incumbent'];step=run[0][8];rejected=step['proposal']
        core.check_prediction(t,incumbent);core.check_prediction(t,rejected);core.check_derivative(t,step['derivative'],incumbent)
        require(step['accepted'] is False and step['incumbent']==incumbent and rejected['cost']>=incumbent['cost'],'original iter8 rejected step')
        require(root['incumbent_logits']==incumbent['logits'] and root['rejected_logits']==rejected['logits'] and
                root['saved_incumbent_cost']==incumbent['cost'] and root['saved_rejected_cost']==rejected['cost'],'same original endpoint arrays/costs')
        xs=[incumbent['logits']]
        require(len(root['candidates'])==4,'all fixed candidates')
        for s,c in zip(SCALES,root['candidates']):
            x=segment(incumbent['logits'],rejected['logits'],s,t['packet_logit_cap'])
            require(c==dict(scale=s,logits=x,commands=np.tanh(np.asarray(x,np.float32)).reshape(10,2).astype(float).tolist()),'independent segment arithmetic')
            xs.append(x)
        vals=[];winner=None;selection=None
        for j in range(8):
            order=8*i+j;v=evaluations[order]
            label='incumbent' if j==0 else 'scale_'+str(j-1) if j<5 else 'selected'
            mode='whole_replay' if j<5 else ('nominal','whole_replay','segmented_replay')[j-5]
            x=xs[j] if j<5 else winner['logits']
            require(v['logits']==x,'all candidate and winner logits bound')
            request=dict(root_index=i,task_id=root['task_id'],label=label,mode=mode,logits=x,commands=v['commands'])
            require(set(v)==set(request)|{'contexts','cost'} and all(v[k]==z for k,z in request.items()),'exact evaluation identity')
            require(v['cost']==path(t,x,v['commands'],v['contexts']),'independent same-Q nonlinear cost')
            event(dict(kind='evaluation_started',order=order,root_index=i,label=label,mode=mode))
            event(dict(kind='evaluation_request',order=order,**request))
            for k,c in enumerate(v['contexts']):
                for resource in ('action_encoder','predict'):
                    e=dict(index=len(entries),resource=resource,samples=1,completed=False,root_index=i,label=label,
                           mode=mode,packet=k,evaluation_order=order)
                    event(dict(kind='module_attempt',entry=e));e=dict(e,completed=True)
                    event(dict(kind='module_return',entry=e));entries.append(e)
                event(dict(kind='native_packet',order=order,root_index=i,label=label,mode=mode,packet=k,context=c))
            if j==0:require(v['contexts']==incumbent['contexts'] and v['cost']==incumbent['cost'],'fresh incumbent exact reproduction')
            if j<5:vals.append(v)
            if j==4:
                idx=min(range(5),key=lambda q:(vals[q]['cost'],q));winner=vals[idx]
                selection=dict(root_sha256=root['root_sha256'],root_index=i,evaluations=vals,winner_index=idx,
                    selected_scale=None if idx==0 else SCALES[idx-1],strict_improvement=winner['cost']<vals[0]['cost'])
                require(selections[i]==selection,'independent stable minimum including incumbent and all four trials')
            if j>=5:require(all(v[f]==winner[f] for f in ('logits','commands','contexts','cost')),'three paid winner H2 replays exact')
        prediction=dict(task_id=root['task_id'],logits=winner['logits'],commands=winner['commands'],contexts=winner['contexts'],
            whole_replay=winner['contexts'][-1],segmented_replay=winner['contexts'],cost=winner['cost'])
        costs=[dict(scale=s,cost=v['cost'],improvement=vals[0]['cost']-v['cost']) for s,v in zip(SCALES,vals[1:])]
        want=dict(root_sha256=root['root_sha256'],root_index=i,selected_scale=selection['selected_scale'],incumbent_cost=vals[0]['cost'],
            scale_costs=costs,selected_cost=winner['cost'],strict_improvement=selection['strict_improvement'],verified_prediction=prediction,full32_required=True)
        require(results[i]==report['rows'][i]==want,'complete root result and top report exact')
        rows.append({k:deepcopy(want[k]) for k in ('root_sha256','root_index','incumbent_cost','scale_costs','selected_scale','selected_cost','strict_improvement')})
    require(cursor==len(events) and len(entries)==64 and entries==report['module_entries'],'all actual module attempts/returns, no padding or missing packets')
    work={r:dict(attempted=sum(e['resource']==r for e in entries),completed=sum(e['resource']==r and e['completed'] for e in entries)) for r in ('predict','action_encoder')}
    require(work==report['resources']=={r:dict(attempted=32,completed=32) for r in work},'exact actual 32 calls each')
    require(all(report[k]==0 for k in ('new_AD_calls','new_image_calls','new_physical_steps')) and report['new_Q'] is False and
            report['training'] is False and report['P1_admission'] is False and report['scientific_pass_threshold'] is None,'scope/admission not broadened')
    return dict(rows=rows,resources=work,event_count=cursor,full_H2_evaluations=16,complete_H2_packet_arrays=32,
                selected_replay_H2_evaluations=6,P1_admission=False)


def audit(directory=ACTUAL,report_sha=REPORT_SHA):
    b=Path(directory).resolve();reader=Reader();r=reader.read(b/'report.json',report_sha)
    require(r['schema']=='p1_gn_step_diagnostic_cuda_20261027_v1.report' and r['status']=='cuda_fixed_segment_diagnostic_completed' and
            r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
            r['model_hash_before'] is not None and r['model_hash_before']==r['model_hash_after'],'complete actual source/model status')
    binding=reader.read(b/'execution_binding.json')
    require(binding['prepare_sha256']==r['prepare_sha256']==PREPARE_SHA,'original external prepare hash')
    p=reader.read(binding['prepared'],PREPARE_SHA)
    require(p['schema']=='p1_gn_step_diagnostic_cuda_20261027_v1.prepare' and r['model_hash_before']==p['expected_model_hash'],'original prepare/model identity')
    for f,h in p['bindings_sha256'].items():reader.bind(f,h)
    names={f'eval_{i:02d}.json' for i in range(16)}|{'events.jsonl','execution_binding.json',
        'root_0.result.json','root_0.selection.json','root_1.result.json','root_1.selection.json'}
    require(set(r['artifacts_sha256'])==names and {f.name for f in b.iterdir()}==names|{'report.json'},'complete exclusive artifact census')
    for n,h in r['artifacts_sha256'].items():reader.bind(b/n,h)
    events=[json.loads(line) for line in (b/'events.jsonl').read_bytes().splitlines()]
    evaluations=[reader.read(b/f'eval_{i:02d}.json') for i in range(16)]
    selections=[reader.read(b/f'root_{i}.selection.json') for i in range(2)]
    results=[reader.read(b/f'root_{i}.result.json') for i in range(2)]
    tasks=[reader.read(root['task_path']) for root in p['roots']]
    components=[reader.read(root['component_path']) for root in p['roots']]
    receipt=check_arrays(p,r,events,evaluations,selections,results,tasks,components)
    reader.recheck()
    return dict(schema=NAME,status='saved_arrays_audited_not_admitted',**receipt,bindings_sha256=reader.bindings,
        actual_report_sha256=report_sha,model_unchanged=True,new_neural_calls=0,new_physics_steps=0,
        limitations=['actual model-Q finite-step improvement, not physical task gain',
            'same two calibration roots, no fresh independent confirmation',
            'no new derivative/finite-difference correctness or stationarity proof',
            'fixed-step GN rejection does not establish absence of a feasible improving step',
            'not P1 admission; no fitted threshold or retuned scales'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=ACTUAL)
    parser.add_argument('--report-sha',default=REPORT_SHA);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    try:r=audit(a.directory,a.report_sha)
    except Exception as exc:r=dict(schema=NAME,status='audit_failed',error=dict(type=type(exc).__name__,message=str(exc)),P1_admission=False)
    files=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    r['audit_assets_sha256']={str(p.resolve()):sha(p) for p in files}
    raw=(json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(f.stat().st_size for f in files)+len(raw)<=LIMIT,'all new source/tests/doc/report <=100KiB')
    require(shutil.disk_usage(a.output.parent).free>=FLOOR+len(raw),'200MiB reserve')
    with a.output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in r.items() if k not in ('bindings_sha256','audit_assets_sha256')},sort_keys=True))
    if r['status']=='audit_failed':raise SystemExit(2)


if __name__=='__main__':main()
