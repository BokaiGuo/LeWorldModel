"""Fixed eight teacher_full anchors CEM64. MAIN CUDA execution only.

Not official GRASP/Hi-LeWM/PlanWAM, not a complete CEM-MPC policy or P1 pass.
Eight fixed teacher_full anchors; no physical outcome enters proposal/selection.
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

NAME='agent_moving_CEM8_20261028_v1'
ARM='teacher_full_CEM64'
ROOT=prior.ROOT
require,sha,canonical,core=prior.require,prior.sha,prior.canonical,prior.core
CAP,FLOOR,FINAL,PREP_CAP=32*1024**2,200*1024**2,12*1024**2,2*1024**2
from experiments.spectral_reachability import agent_baselines_resume_20261027_next as frozen
from experiments.spectral_reachability import capability_suffix_factorial_cuda_MAIN_20261027_v2 as factorial
from experiments.spectral_reachability import consume_suffix_factorial_MAIN_20261027_v2 as strict

AUTH='MAIN_FIXED8_TEACHER_CEM64_CUDA_ONLY'
ARMS=('line_search_GN','full10_conditional_D2P')
ORDER=[(s,c,a) for s in (0,1) for c in ('off','on') for a in ARMS]
FACTORIAL_PREP=ROOT/'results/capability_suffix_factorial_cuda_MAIN_20261027_v2_prepare_MAIN01.json'
FACTORIAL_PREP_SHA='330412c675844d7edff23ade45012cf4ef4920f64bd47a31a0c49e6872f673f5'
FACTORIAL_OUTPUT=ROOT/'results/capability_suffix_factorial_cuda_MAIN_20261027_v2_MAIN01'
FACTORIAL_REPORT_SHA='fc620c9fdcbb5046c1d076c79cd0703da999275ec921de799f3747c0204d69ab'
STRICT_FILE=ROOT/'results/consume_suffix_factorial_MAIN_20261027_v2_MAIN01.json'
STRICT_SHA='3b17d6b6a7323545d0fb81034dac5e0c1d479b5a76de307bb2ca1a1ae1d8ad58'
FROZEN_SHA='bc8e588bc11a4ab50572ace8a99ef397f05f0ee8b6edfab5ae5a42f022528e10'


def seed(slot,condition,arm):
    require(type(slot) is int and (slot,condition,arm) in ORDER,'fixed8 teacher identity')
    return 20261028+800000+104729*ORDER.index((slot,condition,arm))


def contract():
    c=deepcopy(frozen.contract())
    c.update(arm=ARM,branches=8,order=[list(k) for k in ORDER],
        feedback='original eight factorial v2 teacher_full actual prefix5 contexts; same-condition goal',
        scope='fixed known-root eight-anchor suffix CEM calibration; NOT old observed4 cohort',
        seeds=[seed(*k) for k in ORDER],seed_rule='20261028+800000+104729*anchor_ordinal',
        total={k:520 for k in c['total']},
        history='exact teacher_full latent3; float32 tokens of warmup last5 + own original prefix5',
        action='NumPy tanh once to literal float32 raw5 BEFORE model; same raw model/dispatch; no CUDA tanh',
        saved_validation_per_build=dict(SVD=48,neural=0,physics=0,
            scope='strict parent saved bank validation, separate from zero new planning SVD; repeated by each verify/build'),
        inherited='transitive factorial prepare closure; shared encoding/planning/physics paid once, not duplicated per branch',
        inherited_RGB=None,
        shared_resource_axes=dict(source='factorial v2 prepare/strict receipt, full historical upstream costs NOT zero',
            fixed_direction=dict(encoder=40,projector=40,predict=80,action_encoder=80,pred_proj=80),
            factorial_all24=dict(predict=312,action_encoder=312,pred_proj=312,jacobians=24,VJP_directions=4608,SVD=48),
            factorial_teacher8_subset=dict(predict=104,action_encoder=104,pred_proj=104,jacobians=8,VJP_directions=1536,SVD=16),
            double_count_warning='teacher8 subset is included in all24; earlier source/overlay/planner/physical costs additional, not a total'),
        comparison='65 predict vs13 predict+1J+192VJP+2SVD per anchor; same support/context/Q, NOT equal total compute',
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,prepare_cap_bytes=PREP_CAP)
    return c


def assemble(cases,tasks):
    require(len(cases)==24,'fixed24 parent, no outcome subset')
    require([(c['anchor'],c['variant']) for c in cases]==[(i,v) for i in range(8) for v in factorial.VARIANTS],
        'all original fixed factorial identities/order')
    out=[]
    for ordinal,key in enumerate(ORDER):
        case=cases[3*ordinal+1];plan=case['original_plan'];task=tasks[key]
        require((case['root_slot'],case['condition'],case['arm'])==key and case['scale']==1. and
            (plan['root_slot'],plan['condition'],plan['arm'])==key and case['task_id']==plan['task_id'] and
            case['selection_digest']==plan['selection_digest'] and case['prefix']==plan['commands'][:5] and
            plan['selection_uses_physics'] is False,'teacher own root/condition/arm/prefix/task seal')
        require(case['goal']==task['goal'] and task['packet_logit_cap']==contract()['packet_logit_cap'],'original same-condition goal and domain')
        raw=np.asarray(plan['commands']);warm=np.asarray(task['root_raw_action_history'])
        require(raw.shape==(10,2) and np.isfinite(raw).all() and np.max(abs(raw))<=1 and
            np.array_equal(raw,raw.astype(np.float32)) and np.array_equal(warm,prior.planner.fixed_raw()),
            'literal original commands and nonzero warmup, no zero-only interface')
        context=deepcopy(task['root_context']);context.update(deepcopy(case['context']))
        core.context(context)
        require(context['latent_history'][:2]==task['root_context']['latent_history'][1:],'actual teacher chronology')
        past=np.concatenate([warm[5:].astype(np.float32),raw[:5].astype(np.float32)])
        require(np.array_equal(context['action_history'],core.normalized_tokens(task,past)),
            'pure actual raw warmup/prefix float32 normalization, no /100')
        b=dict(anchor=ordinal,source_slot=key[0],condition=key[1],arm=key[2],
            parent_task_id=case['task_id'],blind_sha256=plan['blind_sha256'],
            planner_selection_sha256=case['selection_digest'],prefix_commands=deepcopy(case['prefix']),
            original_plan=deepcopy(plan),observed_context=context,observed_raw_history=past.astype(float).tolist(),
            goal=deepcopy(case['goal']),q=np.eye(192).tolist(),action_mean=deepcopy(task['action_mean']),
            action_std=deepcopy(task['action_std']),packet_logit_cap=task['packet_logit_cap'])
        b['branch_id']=core.digest(b);out.append(b)
    require(len({b['branch_id'] for b in out})==8,'eight unique fixed branches')
    return out


def build():
    require(sha(frozen.__file__)==FROZEN_SHA,'unchanged frozen CEM64 algorithm source')
    require(sha(STRICT_FILE)==STRICT_SHA,'MAIN actual strict consumer receipt')
    parent=factorial.verify(FACTORIAL_PREP,FACTORIAL_PREP_SHA)
    verified=strict.consume(FACTORIAL_OUTPUT,FACTORIAL_REPORT_SHA)
    require(verified==strict.read(STRICT_FILE) and verified['prepare_sha256']==FACTORIAL_PREP_SHA,
        'strict saved native inputs/raw modules and actual prepare binding')
    tasks={};task_bindings={}
    for c in parent['cases'][1::3]:
        plan=c['original_plan'];name=plan['task_file']
        require(Path(name).name==name,'flat original task artifact')
        path=factorial.source.PLANNER/name
        require(str(path) in parent['bindings'] and sha(path)==parent['bindings'][str(path)],'original task in inherited source closure')
        tasks[c['root_slot'],c['condition'],c['arm']]=prior.blind.read(path,PREP_CAP)['task']
        task_bindings[str(path)]=sha(path)
    branches=assemble(parent['cases'],tasks)
    # Parent closure is verified transitively, not copied 6162 times into own prepare.
    bindings={str(FACTORIAL_PREP):FACTORIAL_PREP_SHA,str(FACTORIAL_OUTPUT/'report.json'):FACTORIAL_REPORT_SHA,
        str(STRICT_FILE):STRICT_SHA,**task_bindings}
    files=prior.blind.source_files()+[Path(frozen.__file__),Path(prior.__file__),Path(prior.planner.__file__),
        Path(factorial.__file__),Path(strict.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for f in files:bindings[str(Path(f).resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'final source/input/model closure')
    p=dict(schema=NAME+'.prepare',contract=contract(),branches=branches,bindings=bindings,
        policy=parent['policy'],expected_model_hash=parent['model_hash'],
        inherited_prepare_sha256=FACTORIAL_PREP_SHA,strict_consumer_sha256=STRICT_SHA)
    require(len(canonical(p))<=PREP_CAP,'explicit eight Q prepare must fit2MiB; report if not')
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
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW32MiB output +200MiB floor')
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
    rng=np.random.default_rng(seed(branch['source_slot'],branch['condition'],branch['arm']))
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
    rng=np.random.default_rng(seed(branch['source_slot'],branch['condition'],branch['arm']));mean=np.zeros(10);std=np.ones(10)
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
    p=dict(schema=NAME+'.model_plan',source_slot=branch['source_slot'],root_slot=branch['source_slot'],
        condition=branch['condition'],arm=ARM,prefix_arm=branch['arm'],anchor=branch['anchor'],
        branch_id=branch['branch_id'],task_id=branch['parent_task_id'],blind_sha256=branch['blind_sha256'],
        original_selection_digest=branch['planner_selection_sha256'],
        teacher_context_sha256=core.digest(branch['observed_context']),
        prefix_commands=branch['prefix_commands'],suffix_commands=row['suffix_commands'],
        commands=branch['prefix_commands']+row['suffix_commands'],selected_index=row['selected_index'],model_Q=row['model_Q'],
        horizon=2,reoptimized_horizon=1,raw_steps=10,raw_authority='literal_float32_same_model_and_dispatch',
        selection='model_only_stable64_argmin',P1_admission=False,P3_admission=False,
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
    rows=[dict(index=i,source_slot=b['source_slot'],prefix_arm=b['arm'],condition=b['condition'],anchor=b['anchor'],branch_id=b['branch_id'],status='not_started') for i,b in enumerate(p['branches'])]
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
        report=dict(schema=NAME+'.report',status='teacher8_CEM_commands_sealed_not_admitted' if complete else 'partial_or_failed',
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
    require(r['schema']==NAME+'.report' and r['status']=='teacher8_CEM_commands_sealed_not_admitted' and
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
        [e['receipt'] for e in events if e['kind']=='branch_sealed']==r['rows'] and len(r['rows'])==8,'all eight denominator/start/seal')
    def load(a):
        require(a['file'] in r['artifact_sha256'] and a['sha256']==r['artifact_sha256'][a['file']],'native array closure')
        stream=BytesIO((output/a['file']).read_bytes());value=np.load(stream,allow_pickle=False)
        require(stream.tell()==(output/a['file']).stat().st_size and str(value.dtype)==a['dtype'] and list(value.shape)==a['shape'] and
            value.dtype==np.float32 and np.isfinite(value).all(),'finite exact original float32 arrays')
        return value
    plans=[]
    for i,(branch,row) in enumerate(zip(p['branches'],r['rows'],strict=True)):
        require(row['index']==i and row['source_slot']==branch['source_slot'] and row['prefix_arm']==branch['arm'] and row['condition']==branch['condition'] and row['anchor']==branch['anchor'],'stable eight teacher prefix branch')
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
    require(r['resources']=={k:dict(attempted=520,completed=520) for k in Modules.ORDER} and
        sha(output/'report.json')==expected,'actual complete520 module counts and final report seal')
    return dict(schema=NAME+'.consume',status='teacher8_CEM_commands_verified_not_admitted',errors=[],plans=plans,
        P1_admission=False,P3_admission=False,equal_total_compute=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for name in ('output','prepared','prepare-sha','report-sha','authorization'):parser.add_argument('--'+name)
    a=parser.parse_args(argv)
    if a.command=='prepare':out=prepare(a.output)
    elif a.command=='verify':out=dict(status='verified_no_execution',branches=len(verify(a.prepared,a.prepare_sha)['branches']))
    elif a.command=='run':out=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:out=consume(a.output,a.report_sha)
    print(canonical(out).decode(),end='')
    return 0


if __name__=='__main__':raise SystemExit(main())
