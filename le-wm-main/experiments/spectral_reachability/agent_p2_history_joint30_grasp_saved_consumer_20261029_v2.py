"""Saved-only v2 repair: exact parameter*0 for EMPTY Adam snapshots.

No neural executor. Frozen v1 production/prepare/artifacts remain untouched.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from fractions import Fraction
import sysconfig
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_cuda_20261029_v1 as v1
from experiments.spectral_reachability.agent_p2_history_joint30_grasp_cuda_20261029_v1 import (
    need,inputs,arr,pack,unpack,phases,seed,close_arithmetic,equal,Saved,
    loss_check,project,chart,budget,LRX,LRS,LRV,SYNCSTEPS,
    filehash,readjson,verify,contract,WALL,CAP,Cursor,journal_lines,tasks,
    digest,expected_census,checkbindings,raw,ROOT,upstream,BALL)

NAME='agent_p2_history_joint30_grasp_saved_consumer_20261029_v2'
V1_PINS={
 'experiments/spectral_reachability/agent_p2_history_joint30_grasp_cuda_20261029_v1.py':'6c44630f75259ce107b5b001a8dfc7f50b60c4f7b6ab920371e2994cde1ca9a1',
 'tests/test_agent_p2_history_joint30_grasp_cuda_20261029_v1.py':'2ae0aae3363500f089f392ed131a976aa32e560df9c9e792890397ca96a9c1fb',
 'innerdocs/agent_p2_history_joint30_grasp_cuda_20261029_v1.md':'22f2a0e69561b92acb9dbe8b817a6136a8e50670919136b7afc1719b58d054c2'}
ACTUAL_OUTPUT=ROOT/'results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01'
ACTUAL_REPORT_SHA='d92e24d9807a20f296811d4321b64cc87f2c8c73c04677a350f136debc1ce14a'
SITE=Path(sysconfig.get_paths()['purelib'])
ARITHMETIC_PINS={
 str(SITE/'torch/optim/adam.py'):'7efa850aab7342ed1e8675cf0f7a77df038ab532496434b0a7f2aa6866cb3a5c',
 str(SITE/'torch/include/ATen/native/Lerp.h'):'74cfc5466a751ad53c847c89cdcb0f8580318bd4a574c6963109cfcef372681d'}


def consumer_bindings():
    out={}
    for name,h in V1_PINS.items():
        need(filehash(ROOT/name)==h,'sealed v1 consumer/source/test/doc');out[str(ROOT/name)]=h
    for p,h in ARITHMETIC_PINS.items():
        need(filehash(p)==h,'exact installed Adam/lerp reference source');out[p]=h
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        out[str(p)]=filehash(p)
    return out


def initial_memory(parameter):
    """Producer optimizer_snapshot's EMPTY-state branch, exact signs and dtype.

    This does NOT canonicalize actual initialized moments or journal arrays.
    Called only when the frozen algorithm constructs a fresh optimizer.
    """
    a=np.asarray(parameter)
    need(a.dtype in (np.dtype('float32'),np.dtype('float64')) and np.isfinite(a).all(),'finite native parameter')
    zero=pack(a*np.asarray(0,dtype=a.dtype))
    return dict(step=0,m=zero,q=zero.copy())


def native_lerp(moment,gradient):
    """CUDA lerp small-weight branch: rounded difference then RN fused add.

    Exact rational scalar arithmetic; no torch, AD, model or widened tolerance.
    float32 gets a nearest-even check to avoid float64 double-rounding ties.
    """
    dtype=moment.dtype;weight=dtype.type(1-.9);delta=(gradient-moment).astype(dtype)
    out=np.empty_like(moment)
    for i in range(moment.size):
        exact=Fraction(float(weight))*Fraction(float(delta.flat[i]))+Fraction(float(moment.flat[i]))
        candidate=dtype.type(float(exact))
        if dtype==np.float32:
            candidates=[candidate,np.nextafter(candidate,dtype.type(-np.inf)),np.nextafter(candidate,dtype.type(np.inf))]
            candidate=min(candidates,key=lambda a:(abs(Fraction(float(a))-exact),int(np.asarray(a).view(np.uint32))&1))
        out.flat[i]=candidate
    return out


def adam_check(before,after,param,grad,proposal,lr):
    need(set(before)==set(after)=={'step','m','q'} and type(after['step']) is int and after['step']==before['step']+1,'installed Adam step chronology')
    dtype=param.dtype;eps=np.finfo(dtype).eps
    m=unpack(before['m'],param.shape,dtype);q=unpack(before['q'],param.shape,dtype)
    # Fused=False/foreach=False torch uses lerp for first moment; allow only
    # operation-count rounding, not a tolerance on model predictions or gradients.
    want_m=native_lerp(m,grad);want_q=.999*q+(1-.999)*grad*grad
    am=unpack(after['m'],param.shape,dtype);aq=unpack(after['q'],param.shape,dtype)
    close_arithmetic(am,want_m,eps=eps,ops=64);close_arithmetic(aq,want_q,eps=eps,ops=64)
    t=after['step'];update=lr/(1-.9**t)*am/(np.sqrt(aq)/np.sqrt(1-.999**t)+1e-8)
    # Bound absolute cancellation by the magnitudes before subtraction.
    want=param-update;error=128*eps*(abs(param)+abs(update)+1e-20)
    need(np.isfinite(proposal).all() and np.all(abs(proposal-want)<=error),'Adam proposal vs saved moments/gradient')


def validate_root(task,slot,c,result):
    c.root=slot;b=Saved(task,c);root=arr(task['root_context']['latent_history'],(3,192));goal=arr(task['goal'],(192,))[None]
    # Preserve the actual torch.linspace fractions; verify mathematical thirds
    # separately, without forcing NumPy's different endpoint rounding.
    init=c.get('initialized');need(set(init)=={'kind','seed','fraction','x','v'} and init['seed']==seed(slot),'fixed seed/init')
    frac=unpack(init['fraction'],(1,2,1),np.float32)
    close_arithmetic(frac,np.array([1/3,2/3]).reshape(1,2,1),ops=8)
    x=unpack(init['x'],(30,),np.float64);v=unpack(init['v'],(1,2,192),np.float32)
    need(not x.any() and np.array_equal(v,root[-1,None,None]+frac*(goal[:,None]-root[-1,None,None])),'zero logits/exact saved-fraction linear interpolation')
    mx=initial_memory(x);mv=initial_memory(v)
    for kind,k,j in phases():
        if kind=='sync' and j==0:mx=initial_memory(x)
        c.expect(dict(kind='optimization_attempt',phase=kind,iteration=k,sync_step=j,x=pack(x),v=pack(v)))
        preds=b.soft(v,x) if kind=='joint' else [b.serial(x,'sync')[0]]
        loss=c.get('backward_attempt');need(set(loss)=={'kind','phase','iteration','sync_step','loss','dyn','dense'} and
            (loss['phase'],loss['iteration'],loss['sync_step'])==(kind,k,j),'one scalar backward after3 forwards')
        loss_check(loss,preds,v,goal,kind=='joint')
        rb=c.get('chart_backward');need(set(rb)=={'kind','raw_gradient'},'one raw chart gradient')
        gr=unpack(rb['raw_gradient'],(3,5,2),np.float32)
        grads=c.get('backward_return');need(set(grads)=={'kind','gx','gv'},'gradient closure')
        gx=unpack(grads['gx'],(30,),np.float64)
        need(np.array_equal(gx,gr.reshape(30).astype(np.float64)*(1-np.tanh(x)**2)),'exact float64 tanh chain once')
        gv=None
        if kind=='joint':
            gv=unpack(grads['gv'],(1,2,192),np.float32)
            expected=np.stack([2/192*(v[:,kk]-preds[kk]) for kk in range(2)],1)
            close_arithmetic(gv,expected,ops=16)
        else:need(grads['gv'] is None,'no sync virtual optimization')
        c.expect(dict(kind='optimizer_attempt',phase=kind,iteration=k,sync_step=j))
        op=c.get('optimizer_return');need(set(op)=={'kind','x_before','v_before','x_after','v_after','proposal','x','v'},'full optimizer evidence')
        equal(op['x_before'],mx,'actual persisted action Adam memory');proposal=unpack(op['proposal'],(30,),np.float64)
        adam_check(mx,op['x_after'],x,gx,proposal,LRX if kind=='joint' else LRS);mx=op['x_after']
        xx=unpack(op['x'],(30,),np.float64);need(np.array_equal(xx,project(proposal)) and np.linalg.norm(xx)<=BALL,'one global30 projection')
        vv=unpack(op['v'],(1,2,192),np.float32)
        if kind=='joint':
            equal(op['v_before'],mv,'persisted virtual Adam memory');adam_check(mv,op['v_after'],v,gv,vv,LRV);mv=op['v_after']
        else:need(op['v_before'] is op['v_after'] is None and np.array_equal(vv,v),'sync does not refresh virtual')
        x,v=xx,vv
        if kind=='joint':
            noise=c.get('noise');need(set(noise)=={'kind','noise','v'},'only virtual noise')
            nn=unpack(noise['noise'],(1,2,192),np.float32);nv=unpack(noise['v'],(1,2,192),np.float32)
            need(np.array_equal(nv,v+nn),'literal saved virtual noise addition');v=nv
        if kind=='sync' and j==SYNCSTEPS-1:
            c.expect(dict(kind='joint_optimizer_reset',after=k+1,x=pack(x),v=pack(v)));mx=initial_memory(x);mv=initial_memory(v)
    commands=chart(x).reshape(15,2);c.expect(dict(kind='actions_frozen',x=pack(x),commands=pack(commands)))
    _,whole=b.serial(x,'whole');_,segmented=b.serial(x,'segmented',True)
    equal(whole,segmented,'fresh whole and segmented all3 exact');need(b.n==2706,'all2706 forwards/root')
    rebuilt=dict(root_slot=slot,task_id=inputs.core.task_id(task),logits=x.tolist(),commands=commands.tolist(),
        virtual=v.tolist(),whole=whole,segmented=segmented,
        terminal_Q=float(np.mean((np.asarray(whole[-1]['latent_history'][-1])-np.asarray(task['goal']))**2)),
        resources=budget(),physical_release=False,P1_admission=False,P2_admission=False,official_reproduction=False)
    equal(result,rebuilt,'final last action, no best/outcome selection');c.expect(dict(kind='root_complete',result_sha256=digest(rebuilt)))
    c.root=None;return rebuilt


def consume(output,expected):
    own=consumer_bindings();output=Path(output).absolute();need(filehash(output/'report.json')==expected,'external actual report SHA');r=readjson(output/'report.json')
    need(r['schema']==v1.NAME+'.report' and r['status']=='normalized_model_variant_complete_not_admitted' and r['errors']==[] and
        r['sources_verified'] is True and r['model_before'] is not None and r['model_before']==r['model_after'], 'complete real variant not failed partial')
    p=verify(r['prepared'],r['prepare_sha256']);equal(r['contract'],contract(),'no protocol drift')
    need(r['model_before']['original_state_dict']==p['expected_model_hash'] and r['actual_new_physics']==r['actual_new_encoder']==r['actual_new_projector']==0 and
        all(r[k] is False for k in ('physical_release','P1_admission','P2_admission','official_reproduction','equal_budget_to_1902')),'original model and evidence boundaries')
    expected_guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    equal(r['precision_before'],expected_guard,'precision at entry');equal(r['precision_after'],expected_guard,'precision at exit')
    need(0<=r['wall_seconds']<WALL and r['CUDA_peak_bytes']['allocated']<=4*1024**3 and r['CUDA_peak_bytes']['reserved']<=6*1024**3,'bounded native wall/peak')
    names={'binding.json','loader.json','events.jsonl.gz','root0.json','root1.json'}
    need(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'exact bounded artifact census')
    for n,h in r['artifact_sha256'].items():need(filehash(output/n)==h,'whole actual artifact SHA')
    equal(readjson(output/'binding.json'),dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'real execution binding')
    load=readjson(output/'loader.json');upstream.loader.validate_loader(load,dict(p,bindings_sha256=p['bindings']),r['prepare_sha256'])
    c=Cursor(journal_lines(output/'events.jsonl.gz'))
    for f in load['files']:
        c.expect(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']))
        c.expect(dict(kind='consumed_buffer_frozen',receipt=f))
    need(len(r['rows'])==2,'complete two-root denominator');plans=[]
    for i,(task,row,s) in enumerate(zip(tasks(p),r['rows'],p['slots'],strict=True)):
        equal(row,dict(root_slot=i,condition='off',task_id=s['task_id'],status='complete',result_file=f'root{i}.json'),'fixed source slots/order')
        result=validate_root(task,i,c,readjson(output/row['result_file']))
        plan=dict(schema=NAME+'.model_plan',root_slot=i,condition='off',task_id=s['task_id'],
            task_file=s['task_file'],task_sha256=s['task_sha256'],source_report_sha256=inputs.SOURCE,
            blind_pointer=s['blind_pointer'],original_context=deepcopy(task['root_context']),goal=deepcopy(task['goal']),
            Q=deepcopy(task['projection']),normalization=inputs.normalization(),model_sha256=p['expected_model_hash'],
            raw_steps=15,commands=result['commands'],raw_float32_sha256=inputs.sha(np.asarray(result['commands'],'<f4').tobytes()),
            logits=result['logits'],terminal=result['whole'][-1],model_terminal_Q=result['terminal_Q'],
            planner_report_sha256=expected,physical_release=False,official_reproduction=False,P1_admission=False,P2_admission=False)
        plans.append(dict(plan,selection_digest=digest(plan)))
    c.done();need(c.n==r['journal_events'] and c.tail==r['journal_tail'],'complete journal head/tail')
    equal(c.counts,r['event_counts'],'full physical-free actual census');equal(c.counts,expected_census(),'all resources/modules/no extra dispatch')
    checkbindings(p['bindings'])
    for n,h in r['artifact_sha256'].items():need(filehash(output/n)==h,'final actual file closure')
    need(filehash(output/'report.json')==expected,'final report unchanged')
    equal(consumer_bindings(),own,'unchanged v2 consumer bundle');return dict(schema=NAME+'.consume',consumer_bindings=own,fix_policy='exact empty parameter-times-zero; installed beta complements and RN-FMA lerp; unchanged bounds',status='history_joint30_variant_verified_not_admitted',plans=plans,
        planner_report_sha256=expected,prepare_sha256=r['prepare_sha256'],resources=contract()['total'],
        new_neural_calls=0,new_physics=0,physical_release=False,P1_admission=False,P2_admission=False,
        official_reproduction=False,equal_budget_to_1902=False,errors=[])


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--output',required=True);q.add_argument('--report-sha',required=True);q.add_argument('--receipt-output')
    a=q.parse_args(argv);v=consume(a.output,a.report_sha)
    if a.receipt_output:
        # Reuse exclusive bounded writer only; no producer authority relabeling.
        p=Path(a.receipt_output).absolute();b=raw(v)
        import os,shutil
        need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=2*1024**2 and
            shutil.disk_usage(p.parent).free>=v1.FLOOR+len(b),'new bounded own v2 receipt')
        with p.open('xb') as f:need(f.write(b)==len(b),'short receipt');f.flush();os.fsync(f.fileno())
        v1.upstream.fsync_dir(p.parent);v=dict(path=str(p),sha256=filehash(p))
    else:v={k:val for k,val in v.items() if k!='plans'}|dict(plans_count=len(v['plans']))
    print(raw(v).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())

