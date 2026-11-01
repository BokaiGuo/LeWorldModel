"""Actual selected prefix5 -> fixed H1 suffix correction pilot; MAIN CUDA only.

Prepare/verify are saved-data only. No encoder, physical execution, task sourcepack,
certificate, witness or terminal10 latent inputs to suffix selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as base
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v1 as rgb

NAME='p1_constructive_observed_suffix_cuda_20261027_v1'
ROOT=base.ROOT
CAP,FLOOR,FINAL=2*1024**2,200*1024**2,256*1024
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
KINDS=('zero_nominal','original_imagined_suffix','fresh_observed_ridge')
RGB_SOURCE_SHA='3508316e56515db38eca74c8516c4f434c50d273f1927830445767ef81400dbe'


def raw5(value):
    a=np.asarray(value)
    require(a.shape==(5,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and (abs(a)<=1).all() and
        np.array_equal(a,a.astype(np.float32)),'literal bounded float32 raw5x2, no round/clip/rescale')
    return a.astype(np.float32)


def observed_context(task,prefix,commands):
    rgb.enc.native(prefix); raw=raw5(commands); initial=task['root_context']
    token=base.encode.core.normalized_tokens(task,raw)[0].astype(float).tolist()
    context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=deepcopy(initial['latent_history'][1:])+[prefix.astype(float).tolist()],
        action_history=[deepcopy(initial['action_history'][-1]),token])
    base.d2p.validate_context(context); return context


def advance(context,commands,terminal):
    base.d2p.validate_context(context); token=base.d2p.normalized_history(raw5(commands))[0].astype(float).tolist()
    z=base.d2p.compiler.finite(terminal,(192,))
    result=dict(context,latent_history=deepcopy(context['latent_history'][1:])+[z.tolist()],
        action_history=[deepcopy(context['action_history'][-1]),token])
    base.d2p.validate_advance(context,commands,result); return result


def budget(): return dict(predict=5,action_encoder=5,jacobian=1,svd=1)


def contract():
    return dict(branches=4,order=[dict(root_slot=i,arm=a) for i in range(2) for a in base.ARMS],
        candidates=list(KINDS),selection='same original Q nonlinear H1 terminal cost, stable candidate-index ties',
        per_branch=budget(),total={k:4*v for k,v in budget().items()},AD_API_per_branch=dict(grad=1,jacobian=1),
        mathematical_vjp_per_branch=192,ADforward_in_predict_per_branch=1,
        schedule='zero nominal1 + fresh zero-point ADforward1 + original suffix1 + ridge suffix1 + winner independent replay1',
        damping=.002,packet_logit_cap=float(.75*np.sqrt(5)),no_dedup=True,no_early_exit=True,
        observed_history='old root latent_history[1:] + actual prefix5; actions=[old last zero token, normalized own raw prefix5]',
        new_encoder=0,new_projector=0,new_physics=0,training=False,new_Q=False,
        P1_admission=False,P2_admission=False,non_greedy_claim=False,fair_budget_claim=False,
        scope='known-task saved-prefix feedback counterfactual replay pilot, not online control, full-H2 replanning or mechanism admission',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        output_cap_bytes=CAP,reserve_bytes=FLOOR,no_retries=True)


def compile_suffix(task,context,original_suffix,backend,operation,journal):
    """Pure kernel with explicit callback backend; tests use arrays, not models."""
    base.encode.core.validate_task(task); base.d2p.validate_context(context)
    original=raw5(original_suffix); q=np.asarray(task['projection']); goal=np.asarray(task['goal'])
    require(np.array_equal(np.asarray(task['action_mean'],np.float32),base.d2p.ACTION_MEAN) and
        np.array_equal(np.asarray(task['action_std'],np.float32),base.d2p.ACTION_STD),'same original normalizer as backend')
    journal(dict(kind='suffix_problem_frozen',context=deepcopy(context),parent_task_id=base.encode.core.task_id(task),original_suffix=original.tolist()))
    nominal=base.d2p.compiler.finite(backend.nominal(deepcopy(context)),(192,))
    journal(dict(kind='zero_nominal_saved',terminal=nominal.tolist()))
    j=base.d2p.compiler.finite(backend.jacobian(deepcopy(context),operation),(192,10))
    journal(dict(kind='fresh_J_saved',jacobian=j.tolist(),nominal=nominal.tolist()))
    solve=base.d2p.with_svd_receipts(lambda:base.d2p.compiler.bounded_ridge(q@j,q@(goal-nominal),
        damping=.002,cap=task['packet_logit_cap']),operation)
    ridge=np.tanh(np.asarray(solve['params'],np.float32)).reshape(5,2)
    journal(dict(kind='ridge_frozen_before_prediction',solve=solve,commands=ridge.astype(float).tolist()))
    candidates=[]
    for index,(kind,commands) in enumerate(zip(KINDS,(np.zeros((5,2),np.float32),original,ridge),strict=True)):
        # Nominal's model call was already paid; no second zero evaluation.
        c=advance(context,commands,nominal) if index==0 else backend.step(deepcopy(context),commands.copy())
        base.d2p.validate_advance(context,commands,c)
        require(c['latent_history'][:2]==context['latent_history'][1:],'H1 complete latent chronology')
        cost=base.d2p.cost(c['latent_history'][-1],goal,q)
        row=dict(index=index,kind=kind,commands=commands.astype(float).tolist(),context=c,terminal_cost=cost)
        candidates.append(deepcopy(row)); journal(dict(kind='candidate_saved',candidate=deepcopy(row)))
    winner=deepcopy(min(candidates,key=lambda c:(c['terminal_cost'],c['index'])))
    journal(dict(kind='winner_frozen_before_replay',winner=winner))
    replay=backend.replay(deepcopy(context),raw5(winner['commands']))
    require(replay==winner['context'],'independent H1 winner replay must reproduce full context exactly')
    journal(dict(kind='winner_replay_saved',context=replay))
    return dict(status='observed_suffix_selected_not_admitted',context=context,fresh_jacobian=j.tolist(),
        zero_nominal=nominal.tolist(),ridge_solve=solve,candidates=candidates,winner=winner,independent_replay=replay,
        selected_index=winner['index'],new_physics=0,P1_admission=False,P2_admission=False)


class Ledger(base.D2PLedger):
    def begin(self,resource):
        require(sum(e['resource']==resource for e in self.neural)<5,'suffix predispatch five-forward cap')
        return super().begin(resource)
    def journal(self,e):
        if e['kind'] in ('jacobian','svd') and not e['completed'] and 'error' not in e:
            require(sum(x['kind']==e['kind'] for x in self.math)<1,'suffix one J/one SVD cap')
        return super().journal(e)


class OneAD(base.ADTrace):
    def dispatch(self,kind,fn,args,kwargs):
        require(sum(e['resource']==kind for e in self.events)<1,'suffix one batched-grad/one functional-J API cap')
        return super().dispatch(kind,fn,args,kwargs)


def execute_branch(data,backend,ledger,sink):
    def operation(kind,callback):
        e=dict(kind=kind,index=len(ledger.math),completed=False)
        if kind=='jacobian': e.update(shape=[192,10],mathematical_vjp_directions=192)
        ledger.journal(e)
        try:
            value=callback(); ledger.sync(); e['completed']=True; return value
        except BaseException as exc: e['error']=str(exc); raise
        finally: base.d2p.finish_event(ledger.journal,e)
    return compile_suffix(data['task'],data['observed_context'],data['original_suffix'],backend,operation,sink)


def descriptor(plan,item,row,report_sha):
    return dict(branch=item['branch'],root_slot=item['root_slot'],arm=item['arm'],parent_task_id=plan['task_id'],
        task_export=plan['task_export'],task_sha256=plan['task_sha256'],parent_selection_sha256=plan['selection_sha256'],
        prefix_image_index=item['index'],prefix_latent_file=row['latent']['file'],prefix_latent_sha256=row['latent']['sha256'],
        prefix_png_sha256=item['checkpoint']['file_sha256'],physical_selection_digest=item['selection_digest'],
        RGB_report_sha256=report_sha)


def material(rgb_report,rgb_sha,planner_output,planner_sha):
    """All inputs saved, external-SHA anchored. Do not call rgb.build/verify."""
    for h in (rgb_sha,planner_sha): base.encode.core.hash_value(h)
    require(sha(rgb.__file__)==RGB_SOURCE_SHA,'approved frozen Meitner source SHA')
    rp=Path(rgb_report).absolute(); output=Path(planner_output).absolute()
    require(sha(rp)==rgb_sha,'external actual Meitner RGB report SHA'); r=read(rp)
    require(r['schema']==rgb.NAME+'.report' and r['status']=='selected_rgb_encoded_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==base.encode.MODEL_TENSOR_SHA and r['P1_admission'] is False and
        r['new_predict']==r['new_AD']==r['new_physics']==0,'complete actual selected RGB encoding receipt')
    base.exact_counts(r['resources'],dict(encoder=8,projector=8))
    require(Path(r['inputs']['planner_report']).absolute()==output/'report.json' and r['inputs']['planner_sha']==planner_sha,'same external parent planner report')
    bindings={str(rp):rgb_sha,str(output/'report.json'):planner_sha}
    require(set(r['artifact_sha256'])|{'report.json'}=={p.name for p in rp.parent.iterdir() if p.is_file()},'exact Meitner output census')
    for filename,h in r['artifact_sha256'].items():
        require(Path(filename).name==filename and filename not in ('','.','..'),'flat encoded artifact')
        path=rp.parent/filename; require(sha(path)==h,'RGB artifact SHA'); bindings[str(path)]=h
    binding=read(rp.parent/'binding.json'); pp=Path(binding['prepared']).absolute()
    require(binding['prepare_sha256']==r['prepare_sha256'] and sha(pp)==r['prepare_sha256'],'Meitner prepare chain')
    p=read(pp); bindings[str(pp)]=r['prepare_sha256']
    require(p['schema']==rgb.NAME+'.prepare' and p['arguments']==r['inputs'] and p['contract']==r['contract'] and
        p['expected_model_hash']==base.encode.MODEL_TENSOR_SHA,'same encoded images/protocol/model')
    for path,h in p['bindings'].items():
        path=Path(path).absolute()
        require(path.name!='sourcepack.json' and 'certificate' not in path.name and not path.name.endswith('_blind_inputs.npz'),'no privileged source or raw input reads')
        require(sha(path)==h,'upstream physical/PNG/task/model/source byte seal'); bindings[str(path)]=h
    require(str(Path(rgb.__file__).resolve()) in p['bindings'],'Meitner source must be in actual run seal')
    consumed=base.consume(output,planner_sha); plans=consumed['plans']
    require([(x['order'],x['arm']) for x in plans]==[(i,a) for i in range(2) for a in base.ARMS],'all four original selected branches')
    require(len(r['rows'])==len(p['images'])==8 and all(x['status']=='encoded' for x in r['rows']),'all eight images complete; no partial substitution')
    expected=[(b,role) for b in range(4) for role in ('prefix5','terminal10')]
    require([(x['branch'],x['role']) for x in r['rows']]==expected and
        [(x['branch'],x['role']) for x in p['images']]==expected,'fixed8 image role/order')
    records=[];data=[]
    for b,plan in enumerate(plans):
        item=p['images'][2*b]; row=r['rows'][2*b]
        i=base.load_export(plan['task_export'],plan['task_sha256']); task=i['gn_task']
        export=read(plan['task_export'])
        require(type(export['source_slot']) is int and export['source_slot']==plan['order'] and
            export['blind_sha256']==export['blind_source_sha256'],'approved opaque slot/blind identity')
        require(item==row['provenance'] and item['index']==row['index']==2*b and item['branch']==b and
            row['task_export']==plan['task_export'] and row['task_sha256']==plan['task_sha256'] and
            item['root_slot']==row['root_slot']==plan['order'] and item['arm']==row['arm']==plan['arm'] and
            item['task_id']==row['task_id']==plan['task_id']==i['task_id'] and item['checkpoint']['step']==5 and
            item['checkpoint']['phase']=='plan' and item['root']==task['root_context']['latent_history'][-1] and
            item['goal']==task['goal'] and item['predicted']==plan['prediction']['latent_history'][-2] and item['Q']=='identity192','own prefix5/root/task/arm/goal and original selected model prediction')
        latent=row['latent']; require(latent['file']==f'image_{2*b:02d}.native192.npy' and latent['dtype']=='float32' and latent['shape']==[192] and
            r['artifact_sha256'][latent['file']]==latent['sha256'],'only own durable prefix NPY')
        path=rp.parent/latent['file']; require(path.stat().st_size<4096,'small native192 NPY')
        with path.open('rb') as f: z=np.load(f,allow_pickle=False)
        rgb.enc.native(z)  # Only prefix NPY is loaded; terminal10 is never an optimization input.
        commands=np.asarray(plan['commands']); c=observed_context(task,z,commands[:5])
        rec=descriptor(plan,item,row,rgb_sha)
        rec.update(source_slot=export['source_slot'],blind_sha256=export['blind_sha256'])
        rec['observed_context_sha256']=base.encode.core.digest(c)
        rec['observed_problem_id']=base.encode.core.digest(dict(parent_task_id=plan['task_id'],context=c,
            goal=task['goal'],projection=task['projection'],prefix_commands=commands[:5].tolist(),receipt=rec))
        records.append(rec); data.append(dict(record=rec,task=task,observed_context=c,
            original_prefix=raw5(commands[:5]),original_suffix=raw5(commands[5:])))
        bindings[str(Path(plan['task_export']).absolute())]=plan['task_sha256']
    parent=read(output/'report.json'); parent_prepare=Path(parent['prepared']).absolute()
    original=base.verify(parent_prepare,parent['prepare_sha256']); bindings[str(parent_prepare)]=parent['prepare_sha256']
    bindings.update(original['bindings_sha256'])
    for key,h in parent['artifact_sha256'].items(): bindings[str(output/key)]=h
    files=base.source_files()+[Path(__file__),Path(rgb.__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files})
    for path,h in bindings.items(): require(sha(path)==h,'final saved input/source TOCTOU')
    arguments=dict(rgb_report=str(rp),rgb_sha=rgb_sha,planner_output=str(output),planner_sha=planner_sha)
    return dict(schema=NAME+'.prepare',arguments=arguments,branches=records,contract=contract(),bindings=bindings,
        policy=str(base.encode.POLICY),expected_model_hash=base.encode.MODEL_TENSOR_SHA),data


def prepare(output,**arguments):
    p,_=material(**arguments); raw=canonical(p); path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare +200MiB')
    with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(prepared,expected):
    require(sha(prepared)==expected,'external suffix prepare SHA'); p=read(prepared)
    rebuilt,data=material(**p['arguments']); require(p==rebuilt,'full saved parent/context/encoder/source reconstruction'); return p,data


class Store(base.Store):
    def __init__(self,output):
        import zlib
        self.output=Path(output).absolute(); require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW2MiB +200MiB')
        self.output.mkdir(); self.comp=zlib.compressobj(6,zlib.DEFLATED,31); self.closed=self.broken=False; self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL),'suffix total2MiB cap incl partial')
        return super()._write(name,raw,append=append,final=final)


def seal_plan(data,result):
    record=data['record']; suffix=raw5(result['winner']['commands'])
    commands=np.concatenate((raw5(data['original_prefix']),suffix)).astype(float).tolist()
    plan=dict(schema=NAME+'.suffix_plan',status='selected_pending_global_closure',record=record,
        source_slot=record['source_slot'],arm=record['arm'],blind_sha256=record['blind_sha256'],commands=commands,
        parent_task_id=record['parent_task_id'],observed_problem_id=record['observed_problem_id'],
        observed_context=data['observed_context'],original_prefix_commands=data['original_prefix'].astype(float).tolist(),
        suffix_commands=suffix.astype(float).tolist(),selected_index=result['selected_index'],
        selected_kind=result['winner']['kind'],predicted_context=result['winner']['context'],
        nonlinear_terminal_cost=result['winner']['terminal_cost'],new_physics=0,P1_admission=False,P2_admission=False,
        scope=contract()['scope'])
    plan['selection_sha256']=base.encode.core.digest(plan)
    return plan


def run(prepared,expected,output):
    p,data=verify(prepared,expected); store=Store(output); errors=[]; model=None; before=after=host=None; source_ok=False
    rows=[dict(record=d['record'],source_slot=d['record']['source_slot'],arm=d['record']['arm'],
        blind_sha256=d['record']['blind_sha256'],status='not_started',resources={},AD_API_resources={}) for d in data]
    store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    try:
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        base.encode.frozen(model); before=base.encode.tensor_hash(model); require(before==p['expected_model_hash'],'original frozen model tensor SHA')
        with sdpa_kernel(SDPBackend.MATH):
            for d,row in zip(data,rows,strict=True):
                row['status']='attempted'; record=d['record']
                def sink(e): store.event(dict(branch=record['branch'],observed_problem_id=record['observed_problem_id'],event=e))
                sync=lambda:torch.cuda.synchronize(device); ledger=Ledger(sink,sync)
                ad=OneAD(torch,base.ARMS[1],sink,sync)
                try:
                    backend=base.d2p.TorchBackend(model,device)
                    with ad,base.module_guard(model,ledger): result=execute_branch(d,backend,ledger,sink)
                    base.exact_counts(ledger.counts(),budget()); base.exact_counts(ad.counts(),dict(grad=1,jacobian=1))
                    stem=f'branch_{record["branch"]}'; store.put(stem+'.result.json.gz',result)
                    plan=seal_plan(d,result); store.put(stem+'.plan.json',plan)
                    sink(dict(kind='suffix_selection_sealed',plan=plan)); row.update(status='suffix_plan_sealed',plan_file=stem+'.plan.json',
                        commands=plan['commands'],selection_sha256=plan['selection_sha256'])
                finally: row['resources']=ledger.counts(); row['AD_API_resources']=ad.counts()
    except BaseException as exc:
        errors.append(base.encode.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e: errors.append(base.encode.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted': row['status']='failed_partial'
        if model is not None:
            try: base.encode.frozen(model); after=base.encode.tensor_hash(model)
            except BaseException as e: errors.append(base.encode.error('final_model',e))
        try: verify(prepared,expected); source_ok=True
        except BaseException as e: errors.append(base.encode.error('final_sources',e))
        try: store.close()
        except BaseException as e: errors.append(base.encode.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='suffix_plan_sealed' for r in rows)
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as e: errors.append(base.encode.error('artifact_hash',e)); complete=False
        report=dict(schema=NAME+'.report',status='observed_suffix_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            source_verified=source_ok,host=host,contract=p['contract'],suffix_commands_released=complete,
            resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),
                completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()},
            new_encoder=0,new_projector=0,new_physics=0,training=False,P1_admission=False,P2_admission=False)
        try: store.put('report.json',report,final=True)
        except BaseException as e: complete=False; errors.append(base.encode.error('final_report',e)); print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial suffix output preserved; no retry: '+str(errors)); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','run'))
    for k in ('rgb-report','rgb-sha','planner-output','planner-sha','prepared','prepare-sha','output'): p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare': result=prepare(a.output,rgb_report=a.rgb_report,rgb_sha=a.rgb_sha,planner_output=a.planner_output,planner_sha=a.planner_sha)
    elif a.command=='verify': result=dict(status='verified_no_execution',branches=len(verify(a.prepared,a.prepare_sha)[0]['branches']))
    else: result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__': main()
