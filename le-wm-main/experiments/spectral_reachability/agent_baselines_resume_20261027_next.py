"""Fixed observed-prefix CEM baseline. Prepare/saved consumption; MAIN CUDA only.

Not official GRASP/Hi-LeWM/PlanWAM, not a complete CEM-MPC policy or P1 pass.
Four pre-existing prefixes; no physical labels enter proposal/selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import gzip
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zlib

import numpy as np

from experiments.spectral_reachability import moving_history_observed_suffix_cuda_20261027_v1 as prior

NAME='agent_baselines_resume_20261027_next'
ARM='observed_prefix_CEM64'
ROOT=prior.ROOT
require,sha,canonical,core=prior.require,prior.sha,prior.canonical,prior.core
CAP,FLOOR,FINAL,PREP_CAP=16*1024**2,200*1024**2,4*1024**2,2*1024**2
PRIOR_SHA='3e72cef9375c5fa5b010593e2285e8eeb8d6b51b71310b70663b7f7406a7eb56'
SAVED=ROOT/'results/agent_baselines_resume_20261027_sampling_v2_actual01.json'
SAVED_SHA='c72cb6d96db7ae5cb6527e610145145d66b76c23d038b301efad047fc87ce5ee'
AUTH='MAIN_OBSERVED_PREFIX_CEM64_CUDA_ONLY'
ORDER=[(s,a) for s in (0,1) for a in prior.ARMS]


def seed(slot,prefix_arm):
    require(type(slot) is int and (slot,prefix_arm) in ORDER,'fixed known prefix identity')
    return 20261027+777+slot*101+prior.ARMS.index(prefix_arm)*10007


def contract():
    return dict(arm=ARM,branches=4,roots=2,order=[list(x) for x in ORDER],
        feedback='same four actual prefix5 observations as frozen bounded-ridge suffix calibration',
        scope='controlled suffix solver comparison, NOT complete policy CEM-MPC',
        horizon=1,packet_steps=5,iterations=4,samples=16,elite=4,candidates=64,
        initial_mean=0.,initial_std=1.,std_clip=[.05,1.],ddof=0,smoothing=0.,
        seeds=[seed(s,a) for s,a in ORDER],rng='NumPy default_rng PCG64',
        proposals='Gaussian in float64 logits, radially project each10-vector onto original packet ball',
        projection_density_correction=False,update='mean/std of projected top4, stable(cost,index); no incumbent injection',
        selection='global stable argmin over ALL64 evaluated candidates; independent replay must exactly match',
        per_branch=dict(predict=65,action_encoder=65,pred_proj=65,predict_API=65,predictor_samples=65),
        total=dict(predict=260,action_encoder=260,pred_proj=260,predict_API=260,predictor_samples=260),
        replay_per_branch=1,old_suffix_evaluations=0,nominal_evaluations=0,padding=0,
        AD=0,jacobian=0,VJP=0,SVD=0,solve=0,new_encoder=0,new_projector=0,new_physics=0,
        goal='original task goal; no new goal/labels',Q='original identity192 mean192 squared error; no fit',
        history='latent_history[1:]+actual prefix5; two tokens from warmup last5+selected prefix5, float32 normalization',
        action='raw=tanh(projected float32 logits), original native units/no second /100; dispatch never clipped',
        packet_logit_cap=float(prior.compiler.CAP),training=False,new_Q=False,no_retries=True,
        CUDA_only=True,CPU_neural_fallback=False,SDP='MATH',TF32=False,autocast=False,
        independent_confirmation=False,P1_admission=False,P3_admission=False,official_reproduction=False,
        equal_total_compute=False,equal_FLOPs=False,
        inherited='old source/task/planner/physical/RGB paid; do not add per-branch duplicated costs',
        inherited_RGB=dict(encoder=8,projector=8),
        comparison='same observed context/action/goal/feedback as ridge; 65 vs5 predicts, not matched compute',
        output_cap_bytes=CAP,floor_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=PREP_CAP)


def coverage():
    """New protocol/availability view, not another transcription of all238 cells."""
    require(sha(SAVED)==SAVED_SHA,'previous actual saved comparison seal')
    r=prior.blind.read(SAVED,2*1024**2);m=r['evidence_matrix']
    require(r['status']=='saved_sampling_comparison_not_admitted' and r['P1_admission'] is False and
        r['new_neural_calls']==r['new_physics_calls']==0 and len(r['rows'])==4,'real prior delivery, not a plan')
    actual={
        'GRASP':dict(kind='local_bounded_variant',arms=['GRASP_H3_bounded'],
            gap='not full author DINO-WM lifted/swept protocol; native moving roots differ'),
        'Hi-LeWM':dict(kind='missing',arms=[],
            gap='high object fetch terminal failed_preserved; no complete checkpoint, native normalizer/loader unexecuted'),
        'LeWM':dict(kind='local_world_model_and_solver_variants',arms=['CEM','kernel_faithful_mppi_nominal','moving_gradient_open_loop'],
            gap='Table10 solver/checkpoint/split/statistics/full budgets not matched; current sampling MPPI dispatches sampled winner'),
        'PlanWAM':dict(kind='missing_in_manipulation_protocol',arms=[],
            gap='driving observations/actions/scorer/trained modules incompatible; no PushT author score')}
    rows=[]
    for paper in ('GRASP','Hi-LeWM','LeWM','PlanWAM'):
        cells=[(i,x) for i,x in enumerate(m['author_reported_cells']) if x['paper']==paper]
        local=[(i,x) for i,x in enumerate(m['prior_local_actual']) if x['arm'] in actual[paper]['arms']]
        rows.append(dict(paper=paper,author_reported=[dict(id=x['id'],value=x['author_reported'],version=x['version'],
            table=x['provenance']['table_id'],pdf_page=x['provenance']['pdf_page_1based'],
            source_pointer='/evidence_matrix/author_reported_cells/'+str(i)) for i,x in cells],
            local_actual=[dict(root_slot=x['root_slot'],arm=x['arm'],values=x['local_actual'],
                source_pointer='/evidence_matrix/prior_local_actual/'+str(i)) for i,x in local],
            local_status=actual[paper],protocol=cells[0][1]['protocol']['supplement'],
            all_protocol_records_retained_in_source=True,official_reproduction=False,
            protocol_match=False,improvement=None))
    require(sum(len(x['author_reported']) for x in rows)==238,'full four-paper denominator')
    probe=ROOT/'results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json'
    require(sha(probe)=='669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90','real asset blocker report')
    failure=prior.blind.read(probe)
    require(failure['status']=='failed_preserved' and failure['first_error']['message']=='Truncated range','do not upgrade failed asset fetch')
    return dict(schema=NAME+'.coverage',source=dict(file=str(SAVED),sha256=SAVED_SHA),papers=rows,
        asset_blocker=dict(file=str(probe),sha256=sha(probe),status=failure['status'],first_error=failure['first_error']),
        next_implementation=dict(arm=ARM,status='implemented_not_CUDA_executed',contract=contract()),
        scientific_admission=False,network_calls=0,neural_calls=0,physics_calls=0)


def build():
    require(sha(prior.__file__)==PRIOR_SHA,'frozen actual-prefix input provider')
    p=prior.build(str(prior.RGB_REPORT),prior.RGB_REPORT_SHA,prior.RGB_SOURCE_SHA)
    require([(b['source_slot'],b['arm']) for b in p['branches']]==ORDER,'all four original prefixes, no subset')
    p['schema']=NAME+'.prepare';p['contract']=contract();p['arguments']={}
    for b in p['branches']:
        require(np.array_equal(b['q'],np.eye(192)),'unchanged original Q; no metric adaptation')
        require(b['packet_logit_cap']==p['contract']['packet_logit_cap'],'same legal packet support')
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        p['bindings'][str(f.resolve())]=sha(f)
    p['bindings'][str(SAVED)]=SAVED_SHA
    for path,h in p['bindings'].items():require(sha(path)==h,'source/input/model closure before prepare')
    require(len(canonical(p))<=PREP_CAP,'explicit four-Q prepare must fit2MiB, never silently increase')
    return p


def write_new(path,raw,cap=PREP_CAP):
    path=Path(path).absolute();require(path.parent.is_dir() and not path.exists() and not path.is_symlink() and
        len(raw)<=cap and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive bounded new output +200MiB floor')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
    return dict(file=str(path),sha256=sha(path),bytes=len(raw))


def prepare(path):
    p=build();receipt=write_new(path,canonical(p));return dict(prepared=receipt['file'],prepare_sha256=receipt['sha256'],bytes=receipt['bytes'],contract=p['contract'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external own prepare SHA')
    p=prior.blind.read(path,PREP_CAP);require(p==build(),'exact source-bound prepare reconstruction');return p


class Store(prior.blind.Store):
    def __init__(self,output):
        self.output=Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW16MiB output +200MiB floor')
        self.output.mkdir();self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat saved artifact')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'real output cap/final reserve/floor')
        with (self.output/name).open('ab' if append else 'xb') as f:
            require(f.write(raw)==len(raw),'short artifact write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


def project(bank,cap):
    a=np.asarray(bank,np.float64)
    require(a.shape==(16,10) and np.isfinite(a).all() and cap==contract()['packet_logit_cap'],'fixed Gaussian proposal bank')
    norms=np.linalg.norm(a,axis=1,keepdims=True)
    return a*np.minimum(1.,cap/np.maximum(norms,np.finfo(np.float64).tiny))


def stage_input(branch,raw):
    return prior.planner.incoming(branch,branch['observed_context'],raw)


class Modules(prior.Modules):
    """Reuse immutable full native array capture; own65-call stage contract."""
    def pre(self,kind):
        def enter(module,args):
            require(not self.failed and not self.pending and len(self.entries)<195 and
                kind==self.ORDER[len(self.entries)%3] and self.meter.stage==len(self.entries)//3,
                'exact65 H1 native forwards and original module ordering')
            e=dict(index=len(self.entries),resource=kind,stage=self.meter.stage,completed=False,inputs=[])
            self.entries.append(e);self.pending[module]=e
            self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            for n,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':require(a.shape==(1,3,10) and np.array_equal(a[0],self.meter.expected_input['action_history']),'actual nonzero incoming3 tokens')
                elif kind=='predict':
                    require(a.shape[:2]==(1,3) and (n!=0 or (a.shape==(1,3,192) and np.array_equal(a[0],self.meter.expected_input['latent_history']))),'actual observed3 context')
                    if n==1:require(np.array_equal(a,self.action_embedding),'actual AE->predictor')
                else:require(a.shape==(3,192) and np.array_equal(a,self.predictor_output.reshape(3,192)),'actual predictor->projector')
                e['inputs'].append(prior.save_array(self.store,f'{self.stem}.module_{e["index"]:03d}.input_{n}.npy',a))
            self.store.event(dict(kind='module_inputs_saved',entry=deepcopy(e)))
        return enter


def evaluate(branch,backend,meter,store,row):
    """No labels/old terminal are accepted. RAM mock backend is not neural evidence."""
    row.update(status='attempted',iterations=[],evaluations=[])
    mean=np.zeros(10,np.float64);std=np.ones(10,np.float64)
    rng=np.random.default_rng(seed(branch['source_slot'],branch['arm']))
    context=branch['observed_context'];q=np.asarray(branch['q']);goal=np.asarray(branch['goal'])
    def forward(raw,phase,index):
        meter.stage=index;meter.last_native=None;meter.expected_input=stage_input(branch,raw)
        entry=dict(index=index,phase=phase,commands=raw.astype(float).tolist(),status='attempted')
        row['evaluations'].append(entry)
        store.event(dict(kind='candidate_attempt',branch_id=branch['branch_id'],receipt=deepcopy(entry)))
        after=backend.step(deepcopy(context),raw)
        prior.planner.advance(branch,context,raw,after);z=np.asarray(after['latent_history'][-1])
        require(z.shape==(192,) and np.isfinite(z).all(),'finite true native192 H1 terminal')
        if meter.require_native:require(np.array_equal(z,meter.last_native),'actual full native module output bridge')
        entry.update(status='completed',context=after,cost=float(np.mean((q@(z-goal))**2)))
        store.event(dict(kind='candidate_return',branch_id=branch['branch_id'],receipt=deepcopy(entry)))
        return entry
    for iteration in range(4):
        requested=rng.normal(mean,std,(16,10));params=project(requested,branch['packet_logit_cap'])
        it=dict(iteration=iteration,mean_before=mean.tolist(),std_before=std.tolist(),requested=requested.tolist(),params=params.tolist())
        row['iterations'].append(it)
        store.event(dict(kind='proposal_bank_frozen',branch_id=branch['branch_id'],receipt=deepcopy(it)))
        costs=[]
        for j,v in enumerate(params):
            raw=np.tanh(v.astype(np.float32)).reshape(5,2)
            costs.append(forward(raw,'candidate',16*iteration+j)['cost'])
        elites=np.argsort(np.asarray(costs),kind='stable')[:4]
        mean=params[elites].mean(axis=0);std=np.clip(params[elites].std(axis=0,ddof=0),.05,1.)
        it.update(costs=costs,elite=elites.tolist(),mean_after=mean.tolist(),std_after=std.tolist())
        store.event(dict(kind='proposal_update',branch_id=branch['branch_id'],receipt=deepcopy(it)))
    selected=int(np.argmin([x['cost'] for x in row['evaluations']]))
    row['selected_index']=selected
    chosen=row['evaluations'][selected]
    store.event(dict(kind='selection_frozen_before_replay',branch_id=branch['branch_id'],selected_index=selected))
    replay=forward(np.asarray(chosen['commands'],np.float32),'independent_winner_replay',64)
    require(replay['context']==chosen['context'] and replay['cost']==chosen['cost'],'MATH exact fresh winner replay, no fallback/reselection')
    row.update(status='suffix_sealed',suffix_commands=deepcopy(chosen['commands']),model_Q=chosen['cost'],replay_exact=True)
    return row


def validate_row(branch,row):
    require(row['branch_id']==branch['branch_id'] and row['status']=='suffix_sealed' and len(row['iterations'])==4 and
        len(row['evaluations'])==65 and row['replay_exact'] is True,'complete fixed candidate/replay denominator')
    rng=np.random.default_rng(seed(branch['source_slot'],branch['arm']));mean=np.zeros(10);std=np.ones(10)
    for it in range(4):
        bank=row['iterations'][it];requested=rng.normal(mean,std,(16,10));params=project(requested,branch['packet_logit_cap'])
        require(bank['iteration']==it and np.array_equal(bank['requested'],requested) and np.array_equal(bank['params'],params) and
            np.array_equal(bank['mean_before'],mean) and np.array_equal(bank['std_before'],std),'deterministic CEM RNG/proposals/projected domain')
        costs=[]
        for j,v in enumerate(params):
            e=row['evaluations'][16*it+j];raw=np.tanh(v.astype(np.float32)).reshape(5,2)
            require(e['index']==16*it+j and e['phase']=='candidate' and e['status']=='completed' and np.array_equal(e['commands'],raw),'all literal64 proposals evaluated')
            prior.planner.advance(branch,branch['observed_context'],raw,e['context'])
            cost=float(np.mean((np.asarray(branch['q'])@(np.asarray(e['context']['latent_history'][-1])-np.asarray(branch['goal'])))**2))
            require(e['cost']==cost and np.isfinite(cost),'original goal/Q scoring no outcomes')
            costs.append(cost)
        elite=np.argsort(costs,kind='stable')[:4];mean=params[elite].mean(axis=0);std=np.clip(params[elite].std(axis=0,ddof=0),.05,1.)
        require(bank['costs']==costs and bank['elite']==elite.tolist() and np.array_equal(bank['mean_after'],mean) and
            np.array_equal(bank['std_after'],std),'exact top4 update, no post-hoc alteration')
    selected=int(np.argmin([e['cost'] for e in row['evaluations'][:64]]));e=row['evaluations'][selected];replay=row['evaluations'][64]
    require(type(row['selected_index']) is int and row['selected_index']==selected and row['suffix_commands']==e['commands'] and
        row['model_Q']==e['cost'] and replay==dict(e,index=64,phase='independent_winner_replay'),
        'stable global winner AND exact independently executed replay')


def model_plan(branch,row):
    p=dict(schema=NAME+'.model_plan',source_slot=branch['source_slot'],arm=ARM,prefix_arm=branch['arm'],
        branch_id=branch['branch_id'],parent_task_id=branch['parent_task_id'],blind_sha256=branch['blind_sha256'],
        parent_selection_sha256=branch['planner_selection_sha256'],physical_prefix_selection_digest=branch['physical_selection_digest'],
        observed_context_sha256=core.digest(branch['observed_context']),prefix_npy=branch['prefix_npy'],
        prefix_commands=branch['prefix_commands'],suffix_commands=row['suffix_commands'],
        commands=branch['prefix_commands']+row['suffix_commands'],selected_index=row['selected_index'],model_Q=row['model_Q'],
        horizon=2,reoptimized_horizon=1,raw_steps=10,selection='model_only_stable64_argmin',P1_admission=False,
        scientific_admission=False,independent_confirmation=False,equal_total_compute=False)
    p['selection_sha256']=core.digest(p);return p


def native_arrays(branch,row,load):
    entries=row['module_ledger'];require(len(entries)==195,'all65 x3 native modules')
    for n,evaluation in enumerate(row['evaluations']):
        ins=[];outs=[];incoming=stage_input(branch,evaluation['commands'])
        for j,kind in enumerate(Modules.ORDER):
            e=entries[3*n+j]
            require(e['index']==3*n+j and e['stage']==n and e['resource']==kind and e['completed'] is True,'native actual stage/order/return')
            ins.append([load(a) for a in e['inputs']]);outs.append(load(e['output']))
        require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],incoming['action_history']),'native exact normalized three action packets')
        require(len(ins[1])==2 and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],incoming['latent_history']) and
            np.array_equal(ins[1][1],outs[0]) and outs[1].shape==(1,3,192),'native context and AE-output->predictor')
        require(len(ins[2])==1 and np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192) and
            np.array_equal(outs[2][-1],evaluation['context']['latent_history'][-1]),'full native predictor/projector successor bridge')


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA execution authorization')
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(index=i,source_slot=b['source_slot'],prefix_arm=b['arm'],branch_id=b['branch_id'],status='not_started') for i,b in enumerate(p['branches'])]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        prior.observed.enc.frozen(model);before=prior.observed.enc.tensor_hash(model)
        require(before==p['expected_model_hash'],'same frozen native checkpoint tensor hash')
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for branch,row in zip(p['branches'],rows,strict=True):
                row['status']='attempted';store.event(dict(kind='branch_started',branch=deepcopy(branch)))
                meter=SimpleNamespace(stage=None,expected_input=None,last_native=None,require_native=True)
                modules=Modules(model,store,f'branch_{row["index"]:02d}',meter,lambda:torch.cuda.synchronize(device))
                try:
                    with modules:evaluate(branch,prior.d2p.TorchBackend(model,device),meter,store,row)
                finally:
                    row['module_ledger']=deepcopy(modules.entries);row['resources']=modules.counts()
                validate_row(branch,row)
                prior.blind.exact_counts(row['resources'],dict(predict=65,action_encoder=65,pred_proj=65))
                row['plan']=model_plan(branch,row)
                store.put(f'branch_{row["index"]:02d}.json.gz',row)
                store.event(dict(kind='branch_sealed',receipt=deepcopy(row)))
    except BaseException as e:
        errors.append(prior.observed.enc.error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(prior.observed.enc.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:prior.observed.enc.frozen(model);after=prior.observed.enc.tensor_hash(model)
            except BaseException as e:errors.append(prior.observed.enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(prior.observed.enc.error('final_source',e))
        try:store.close()
        except BaseException as e:errors.append(prior.observed.enc.error('journal_finalize',e))
        hashes={}
        try:
            for path in store.output.iterdir():
                require(path.is_file() and not path.is_symlink(),'flat actual output');hashes[path.name]=sha(path)
        except BaseException as e:errors.append(prior.observed.enc.error('final_hashes',e))
        complete=not errors and source_ok and before is not None and before==after and all('plan' in r for r in rows)
        report=dict(schema=NAME+'.report',status='observed_CEM_commands_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],rows=rows,errors=errors,
            first_error=errors[0] if errors else None,model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,commands_released=complete,host=host,artifact_sha256=hashes,
            new_images=0,new_physics=0,training=False,P1_admission=False,P3_admission=False,
            AD_exclusion='torch.no_grad + no derivative APIs in this runner; no independent engine AD hook',
            resources={k:dict(attempted=sum(r.get('resources',{}).get(k,{}).get('attempted',0) for r in rows),
                completed=sum(r.get('resources',{}).get(k,{}).get('completed',0) for r in rows)) for k in Modules.ORDER})
        try:store.put('report.json',report,final=True)
        except BaseException as e:
            errors.append(prior.observed.enc.error('final_report',e));complete=False
            print(canonical(dict(first_error=errors[0] if errors else None,errors=errors)).decode(),flush=True)
    require(complete,'partial evidence retained, no retry: '+str(errors))
    return dict(status=report['status'],report=str(store.output/'report.json'),report_sha256=sha(store.output/'report.json'),resources=report['resources'])


def consume(output,expected):
    core.hash_value(expected);output=Path(output).absolute();require(sha(output/'report.json')==expected,'external actual report SHA')
    r=prior.blind.read(output/'report.json',CAP)
    require(r['schema']==NAME+'.report' and r['status']=='observed_CEM_commands_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is r['model_unchanged'] is r['source_verified'] is True and
        r['P1_admission'] is r['P3_admission'] is False,'successful frozen own report only')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['contract']==p['contract'] and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'frozen model and protocol')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'full bounded artifact census')
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..') and not (output/name).is_symlink() and sha(output/name)==h,'all actual artifact seals')
    require(prior.blind.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution prepare binding')
    raw,eof=prior.blind.inflate((output/'events.jsonl.gz').read_bytes(),CAP*16)
    require(eof and raw.endswith(b'\n'),'complete bounded journal')
    events=[prior.blind.decode(x) for x in raw.splitlines()]
    require([e['branch'] for e in events if e['kind']=='branch_started']==p['branches'] and
        [e['receipt'] for e in events if e['kind']=='branch_sealed']==r['rows'] and len(r['rows'])==4,'all four denominator/start/seal')
    def load(a):
        require(a['file'] in r['artifact_sha256'] and a['sha256']==r['artifact_sha256'][a['file']],'native array closure')
        stream=BytesIO((output/a['file']).read_bytes());value=np.load(stream,allow_pickle=False)
        require(stream.tell()==(output/a['file']).stat().st_size and str(value.dtype)==a['dtype'] and list(value.shape)==a['shape'] and
            value.dtype==np.float32 and np.isfinite(value).all(),'finite exact original float32 arrays')
        return value
    plans=[]
    for i,(branch,row) in enumerate(zip(p['branches'],r['rows'],strict=True)):
        require(row['index']==i and row['source_slot']==branch['source_slot'] and row['prefix_arm']==branch['arm'],'stable observed prefix branch')
        validate_row(branch,row);native_arrays(branch,row,load)
        prior.blind.exact_counts(row['resources'],dict(predict=65,action_encoder=65,pred_proj=65))
        require(row['plan']==model_plan(branch,row) and prior.blind.read(output/f'branch_{i:02d}.json.gz',CAP)==row,'own model selection seal')
        scoped=[e for e in events if e.get('branch_id')==branch['branch_id']]
        require([e['receipt'] for e in scoped if e['kind']=='candidate_return']==row['evaluations'] and
            [e['receipt'] for e in scoped if e['kind']=='candidate_attempt']==[
                {k:v for k,v in dict(e,status='attempted').items() if k not in ('context','cost')} for e in row['evaluations']],
            'all65 exact attempted/returned candidates')
        require([e['receipt'] for e in scoped if e['kind']=='proposal_update']==row['iterations'] and
            [e['receipt'] for e in scoped if e['kind']=='proposal_bank_frozen']==[
                {k:v for k,v in it.items() if k not in ('costs','elite','mean_after','std_after')} for it in row['iterations']],
            'all fixed proposal banks persisted before their evaluations')
        releases=[n for n,e in enumerate(scoped) if e['kind']=='selection_frozen_before_replay']
        require(len(releases)==1 and scoped[releases[0]]['selected_index']==row['selected_index'] and
            sum(e['kind']=='candidate_return' for e in scoped[:releases[0]])==64 and
            scoped[releases[0]+1]['kind']=='candidate_attempt' and scoped[releases[0]+1]['receipt']['index']==64,
            'selection frozen before the only fresh replay; never select replay/outcome')
        # Native journal is scoped by stable branch start/seal boundaries, not guessed array index.
        left=next(n for n,e in enumerate(events) if e['kind']=='branch_started' and e['branch']['branch_id']==branch['branch_id'])
        right=next(n for n,e in enumerate(events) if e['kind']=='branch_sealed' and e['receipt']['branch_id']==branch['branch_id'])
        native=events[left:right+1];ledger=row['module_ledger']
        require([e['entry'] for e in native if e['kind']=='module_return']==ledger and
            [e['entry'] for e in native if e['kind']=='module_attempt']==[
                {k:v for k,v in dict(e,completed=False,inputs=[]).items() if k!='output'} for e in ledger],
            'every native attempted and returned module persisted')
        plans.append(dict(row['plan'],whole_run_report_sha256=expected))
    require(r['resources']=={k:dict(attempted=260,completed=260) for k in Modules.ORDER} and
        sha(output/'report.json')==expected,'actual complete260 module counts and final report seal')
    return dict(schema=NAME+'.consume',status='observed_CEM_commands_verified_not_admitted',errors=[],plans=plans,
        P1_admission=False,P3_admission=False,equal_total_compute=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('coverage','prepare','verify','run','consume'))
    for name in ('output','prepared','prepare-sha','report-sha','authorization'):parser.add_argument('--'+name)
    a=parser.parse_args(argv)
    if a.command=='coverage':
        value=coverage();out=write_new(a.output,canonical(value)) if a.output else value
    elif a.command=='prepare':out=prepare(a.output)
    elif a.command=='verify':out=dict(status='verified_no_execution',branches=len(verify(a.prepared,a.prepare_sha)['branches']))
    elif a.command=='run':out=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:out=consume(a.output,a.report_sha)
    print(canonical(out).decode(),end='')
    return 0


if __name__=='__main__':raise SystemExit(main())
