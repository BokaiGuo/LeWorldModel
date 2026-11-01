"""Fixed common-prefix physical interventions; MAIN explicit physics only.

Prepare/mock use saved arrays. No neural imports, optimization, new targets,
outcome-selected actions, independent auditor or P1 admission.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as old
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME='p1_constructive_common_prefix_probe_physical_20261027_v1'
ROOT=old.ROOT; builder=old.builder; source=old.source
require,sha=old.require,old.sha
AUTH='MAIN_CONSTRUCTIVE_COMMON_PREFIX_PROBE_PHYSICS_ONLY'
ROLES=('Z','G','D'); RHO=.25; LOGIT_CAP=float(.75*np.sqrt(5))
CAP,FLOOR,FINAL,PNG_CAP=256*1024**2,200*1024**2,2*1024**2,1024**2
PLANNER_SHA='4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d'
PHYSICAL_SHA='d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300'
D2P='full10_conditional_D2P'
DESIGN=ROOT/'innerdocs/p1_constructive_common_prefix_capability_probe_design_20261027_v1.md'


def raw(v): return (source.canonical(v)+'\n').encode()
def digest(v): return source.digest(v)
def read(path,limit=8*1024**2):
    p=Path(path); require(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'bounded regular saved file')
    b=p.read_bytes()
    if p.name.endswith('.gz'):
        d=zlib.decompressobj(31); b=d.decompress(b,32*1024**2+1)
        require(len(b)<=32*1024**2 and d.eof and not d.unused_data and not d.unconsumed_tail,'single bounded complete gzip')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key'); out[k]=v
        return out
    v=json.loads(b,object_pairs_hook=pairs); raw(v); return v


def bits(v,dtype):
    a=np.asarray(v,dtype=dtype); require(np.isfinite(a).all(),'finite bit identity')
    return np.ascontiguousarray(a).tobytes().hex()


def raw5(v):
    a=np.asarray(v)
    require(a.shape==(5,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
        (abs(a)<=1).all() and np.array_equal(a,a.astype(np.float32)),'literal bounded float32 raw5')
    return a.astype(np.float32)


def probes():
    bank=[]
    for coordinate,sign in [(None,0)]+[(j,s) for j in range(10) for s in (1,-1)]:
        x=np.zeros(10,np.float32)
        if coordinate is not None: x[coordinate]=sign*RHO
        require(np.linalg.norm(x.astype(float))<=LOGIT_CAP,'probe original packet ball')
        u=np.tanh(x).reshape(5,2)
        bank.append(dict(index=len(bank),coordinate=coordinate,sign=sign,rho=RHO,
            logits=x.astype(float).tolist(),commands=u.astype(float).tolist(),raw_float32_bits=bits(u,'<f4')))
    require(len(bank)==21,'fixed21 probes'); return bank


def choose_roles(task,result):
    """Only frozen model scores; no physical endpoint participates."""
    core.validate_task(task)
    require(np.array_equal(task['projection'],np.eye(192)) and task['packet_logit_cap']==LOGIT_CAP,'original identity Q/cap')
    cs=result['candidates']; bank=result['bank']
    require(result['status']=='completed' and len(cs)==len(bank['params'])==len(bank['kinds'])==10,'all ten original candidates')
    for i,c in enumerate(cs):
        require(c['index']==i and c['completed'] is True and c['independent_model_replay_exact'] is True and
            c['prefix_params']==bank['params'][i] and c['proposal_kind']==bank['kinds'][i],'original candidate identity')
        x=core.array(c['prefix_params'],(10,))
        require(np.linalg.norm(x)<=LOGIT_CAP+1e-12,'original packet ball; arithmetic roundoff only')
        u=np.tanh(x.astype(np.float32)).reshape(5,2)
        require(np.array_equal(u,c['prefix_commands']) and np.array_equal(u,np.asarray(c['commands'])[:5]),'exact time-major tanh prefix')
        r=np.asarray(task['projection'])@(core.array(c['predicted_prefix'],(192,))-np.asarray(task['goal']))
        require(c['stage_cost']==float(np.mean(r*r)) and c['objective']==c['stage_cost']+c['own_reach_value'],'saved same-Q model selection')
    g=min(range(10),key=lambda i:(cs[i]['stage_cost'],i))
    d=min(range(10),key=lambda i:(cs[i]['objective'],i))
    require(result['winner']==cs[d],'stable original D winner')
    out=[dict(role='Z',candidate_index=None,prefix_logits=np.zeros(10).tolist(),prefix_commands=np.zeros((5,2)).tolist(),stage_Q=None)]
    for role,i in (('G',g),('D',d)):
        out.append(dict(role=role,candidate_index=i,prefix_logits=cs[i]['prefix_params'],prefix_commands=raw5(cs[i]['prefix_commands']).astype(float).tolist(),stage_Q=cs[i]['stage_cost']))
    # Current protocol explicitly requires the already-recorded G=D case.
    require(bits(out[1]['prefix_commands'],'<f4')==bits(out[2]['prefix_commands'],'<f4'),
        'current G=D prerequisite contradicted; do not silently launch six positions')
    return out


def deduplicate(root_records):
    positions=[]; aliases=[]
    require(len(root_records)==2 and [r['root_slot'] for r in root_records]==[0,1] and
        len({r['root_sha256'] for r in root_records})==2,'fixed two distinct source roots')
    for r in root_records:
        require([a['role'] for a in r['roles']]==list(ROLES),'six logical roles fixed')
        seen={}
        for role in r['roles']:
            prefix=raw5(role['prefix_commands'])
            key=dict(root_sha256=r['root_sha256'],goal_state_bits=bits(r['goal'],'<f8'),
                recipe_contract=r['recipe_contract'],action_history_contract=r['action_history_contract'],
                prefix_raw_bits=bits(prefix,'<f4'))
            h=digest(key)
            if h not in seen:
                pid=len(positions); seen[h]=pid
                positions.append(dict(position_id=pid,root_slot=r['root_slot'],root_sha256=r['root_sha256'],
                    dedup_key=key,position_sha256=h,prefix_commands=prefix.astype(float).tolist(),role_aliases=[]))
            pid=seen[h]; alias=dict(root_slot=r['root_slot'],position_id=pid,**deepcopy(role))
            alias['prefix_logits_sha256']=digest(role['prefix_logits']); positions[pid]['role_aliases'].append(alias); aliases.append(alias)
    require(2<=len(positions)<=4,'current G=D unique-position bound')
    for k in (0,1):
        a=[x for x in aliases if x['root_slot']==k]
        require(a[1]['position_id']==a[2]['position_id'],'G/D must alias, never duplicate observations')
    return positions,aliases


def direction(root,goal):
    r=source.vector(root); g=source.vector(goal); d=g[:4]-r[:4]; norm=float(np.linalg.norm(d))
    chord=np.array([np.cos(g[4])-np.cos(r[4]),np.sin(g[4])-np.sin(r[4])]); cn=float(np.linalg.norm(chord))
    return dict(root_state=r.tolist(),goal=g.tolist(),position_delta=d.tolist(),position_unit=(d/norm).tolist() if norm else None,
        angle_chord_unit=(chord/cn).tolist() if cn else None,position_units='pixels',angle_units='separate unit-circle chord')


def contract(m,wall=600):
    require(type(wall) is int and 0<wall<=600 and 2<=m<=4,'predeclared current-position/wall bounds')
    n=21*m
    return dict(roots=2,logical_roles=6,positions=m,probes_per_position=21,branches=n,rho=RHO,
        warmup=10,prefix_steps=5,suffix_steps=5,with_target=False,wall_seconds=wall,
        caps=dict(reset=2*n,public_step=20*n,physics_step=204*n,explicit_render=22*n,goal_render=2*n,set_state=4*n,goal_bind=n),
        PNG_cap=22*n,PNG_bytes_cap=PNG_CAP,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        order='root slot, unique position first appearance Z/G/D, zero then coordinate0..9 plus/minus',
        signed_zero='preserve float32 bits; +0/-0 are NOT merged',
        pairing='per-position probe0 step5 exact state64/RGB/raw-prefix32/fullbody/action-history; G/D also old physical prefix',
        solver_cache='not serialized; equality only exported body fields plus recipe/version/order',
        new_model_loads=0,new_encoder=0,new_projector=0,new_predict=0,new_action_encoder=0,new_AD=0,new_J=0,new_SVD=0,new_solve=0,new_GPU=0,
        selection='frozen model-only G/D + fixed interventions, no outcome feedback to selector',
        P1_admission=False,P2_admission=False,independent_roots=2,new_independent_samples=0,equal_compute_claim=False)


def source_files():
    pending=[Path(__file__),Path(old.__file__),Path(core.__file__),DESIGN,
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]; seen=set()
    while pending:
        f=pending.pop().resolve()
        if f in seen: continue
        seen.add(f)
        if f.suffix!='.py' or not f.is_relative_to(ROOT/'experiments'): continue
        for node in ast.walk(ast.parse(f.read_text())):
            names=[]
            if isinstance(node,ast.Import): names=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments'):
                names=[node.module]+[node.module+'.'+v.name for v in node.names]
            for name in names:
                child=ROOT/Path(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and child.is_file(): pending.append(child)
    return sorted(seen)


def collect_saved(sourcepack,source_sha,source_prepared,source_prepare_sha,planner_report,planner_sha,physical_report,physical_sha,wall_seconds=600):
    require(planner_sha==PLANNER_SHA and physical_sha==PHYSICAL_SHA,'this fixed known-root protocol requires approved external receipts')
    arguments=dict(sourcepack=str(Path(sourcepack).resolve()),source_sha=source_sha,source_prepared=str(Path(source_prepared).resolve()),
        source_prepare_sha=source_prepare_sha,planner_report=str(Path(planner_report).resolve()),planner_sha=planner_sha,
        physical_report=str(Path(physical_report).resolve()),physical_sha=physical_sha,wall_seconds=wall_seconds)
    data=old.load_source(sourcepack,source_sha,source_prepared,source_prepare_sha)
    bindings=deepcopy(data['bindings'])
    def bound(path,h):
        f=Path(path).resolve(); require(sha(f)==h,'explicit saved input SHA '+str(f)); bindings[str(f)]=h; return read(f)
    pr=bound(planner_report,planner_sha); ph=bound(physical_report,physical_sha)
    require(pr['schema']=='p1_constructive_blind_planners_cuda_20261027_v1.report' and pr['status']=='model_plans_sealed_not_admitted' and
        pr['errors']==[] and pr['commands_released'] is True and pr['source_verified'] is pr['model_unchanged'] is True,'actual frozen complete planner')
    pp=bound(pr['prepared'],pr['prepare_sha256'])
    require(len(pp['slots'])==2 and [s['order'] for s in pp['slots']]==[0,1] and
        pr['model_hash_before']==pr['model_hash_after']==pp['expected_model_hash'],'original two task/model identity')
    require(ph['schema']==old.NAME+'.report' and ph['status']=='physical_calibration_recorded' and ph['errors']==[] and
        ph['sourcepack_sha256']==source_sha and len(ph['rows'])==4 and
        [(x['root_slot'],x['arm']) for x in ph['rows']]==[(i,a) for i in (0,1) for a in ('line_search_GN',D2P)],'original physical root/arm census')
    pb=read(Path(physical_report).parent/'execution_binding.json')
    bindings[str((Path(physical_report).parent/'execution_binding.json').resolve())]=sha(Path(physical_report).parent/'execution_binding.json')
    phys_p=bound(pb['prepared'],pb['prepare_sha256']); manifest=bound(phys_p['manifest'],phys_p['manifest_sha256'])
    require((str(Path(manifest['sourcepack']).resolve()),manifest['sourcepack_sha256'],str(Path(manifest['source_prepared']).resolve()),manifest['source_prepare_sha256'])==
        (arguments['sourcepack'],source_sha,arguments['source_prepared'],source_prepare_sha),'same source recipe/goal contract')
    jpath=Path(physical_report).parent/'journal.jsonl'; require(sha(jpath)==ph['journal_sha256'],'actual physical journal hash')
    lines=jpath.read_bytes().splitlines(); require(lines,'physical journal present')
    tail=json.loads(lines[-1])['event_sha256']; je=old.journal(jpath,tail); bindings[str(jpath.resolve())]=ph['journal_sha256']
    for name,h in pp['bindings_sha256'].items():
        f=Path(name); require(name in {s['task_export'] for s in pp['slots']} or
            (f.suffix in ('.py','.md','.pt','.json') and ('results' not in f.parts or f.name.startswith('task_'))),'public planner closure only')
        require(sha(f)==h,'original source/model/export seal'); bindings[name]=h
    records=[]; contexts=[]; old_prefix={}
    for k,(slot,ref) in enumerate(zip(pp['slots'],data['references'],strict=True)):
        require(slot['order']==k and ref['slot']['index']==k,'stable source slot')
        export=bound(slot['task_export'],slot['task_sha256']); task=export['task']; core.validate_task(task)
        require(export['task_id']==slot['task_id']==core.task_id(task) and export['source_slot']==k and
            export['blind_sha256']==export['blind_source_sha256']==ref['blind_sha256'] and export['with_target'] is False,
            'same approved opaque task/root/goal render binding')
        require(not np.asarray(task['root_raw_action_history']).any() and
            np.array_equal(core.normalized_tokens(task,np.zeros((10,2))),task['root_context']['action_history']),'exact actual warmup action history')
        result_name=f'root_{k}_{D2P}.result.json.gz'; plan_name=f'root_{k}_{D2P}.plan.json'
        result=bound(Path(planner_report).parent/result_name,pr['artifact_sha256'][result_name])
        plan=bound(Path(planner_report).parent/plan_name,pr['artifact_sha256'][plan_name])
        roles=choose_roles(task,result); winner=result['winner']
        selected=deepcopy(plan); identity=selected.pop('selection_sha256')
        require(digest(selected)==identity and plan['task_id']==slot['task_id'] and plan['task_sha256']==slot['task_sha256'] and
            plan['commands']==winner['commands'] and plan['task_contract_sha256']==task['task_contract_sha256'],'original selected model plan')
        row=ph['rows'][2*k+1]; selected_physical=manifest['plans'][2*k+1]
        require(row['initial_exact'] is True and row['full_horizon'] is True and row['first_error'] is None and
            row['status'] in ('horizon_completed','native_success') and row['goal']==ref['goal'].tolist() and
            row['root_state']==ref['states'][-1].tolist() and selected_physical['commands']==plan['commands'] and
            selected_physical['selection_digest']==row['selection_digest'],'old actual D2P prefix/goal/root')
        returns=[e for e in je if e['kind']=='returned_transition' and e['branch']==2*k+1 and e['phase']=='plan' and e['step']<=5]
        require([e['step'] for e in returns]==[1,2,3,4,5] and
            bits([e['command'] for e in returns],'<f4')==bits(winner['commands'][:5],'<f4'),'actual original five commands')
        cp=next(c for c in row['checkpoints'] if c['phase']=='plan' and c['step']==5)
        f=Path(physical_report).parent/cp['png']; require(f.name==cp['png'] and sha(f)==cp['file_sha256'],'original prefix PNG provenance')
        bindings[str(f.resolve())]=cp['file_sha256']
        from PIL import Image
        with Image.open(f) as im: pixels=np.asarray(im)
        require(pixels.shape==(224,224,3) and pixels.dtype==np.uint8 and builder.image_sha(pixels)==cp['pixel_sha256'],'old prefix full RGB bytes')
        old_prefix[k]=cp
        recipe=dict(slot=ref['slot'],recipe=data['plan']['recipe'],environment=builder.ENV_SHA,with_target=False,
            sourcepack_sha=source_sha,source_prepare_sha=source_prepare_sha,warmup=10,render_size=224,relative=True,action_scale=100)
        history=dict(raw_zero10_bits=bits(np.zeros((10,2),np.float32),'<f4'),action_mean=task['action_mean'],action_std=task['action_std'],
            normalized_history=task['root_context']['action_history'])
        records.append(dict(root_slot=k,root_sha256=ref['root_sha256'],goal=ref['goal'].tolist(),recipe_contract=recipe,
            action_history_contract=history,roles=roles,task_id=slot['task_id'],task_sha256=slot['task_sha256'],
            direction=direction(ref['states'][-1],ref['goal'])))
        contexts.append(task)
    positions,aliases=deduplicate(records); plans=[]
    for pos in positions:
        for probe in probes():
            u=np.concatenate((raw5(pos['prefix_commands']),raw5(probe['commands'])))
            plans.append(dict(index=len(plans),root_slot=pos['root_slot'],position_id=pos['position_id'],probe_index=probe['index'],
                arm='position_'+str(pos['position_id'])+'_probe_'+str(probe['index']),commands=u.astype(float).tolist(),
                selection_digest=digest(dict(position=pos['position_sha256'],probe=probe))))
    bindings.update({str(f):sha(f) for f in source_files()})
    for path,h in bindings.items(): require(sha(path)==h,'final prepare seal '+path)
    p=dict(schema=NAME+'.prepare',arguments=arguments,roots=records,positions=positions,role_aliases=aliases,probes=probes(),plans=plans,
        contract=contract(len(positions),wall_seconds),bindings=bindings,
        inherited_costs=dict(source=data['pack']['counts'],planner=pr['resources'],physical=ph['counts'],
            other_encoding_and_feedback='separate existing receipts, not rerun or charged as zero'),P1_admission=False)
    data.update(tasks=contexts,old_prefix=old_prefix)
    return p,data


def prepare(output,**arguments):
    p,_=collect_saved(**arguments); b=raw(p); f=Path(output).absolute()
    require(not f.exists() and len(b)<=256*1024 and shutil.disk_usage(f.parent).free>=FLOOR+len(b)+4096,'NEW prepare<=256KiB +200MiB floor')
    with f.open('xb') as out:
        require(out.write(b)==len(b),'short prepare write'); out.flush(); os.fsync(out.fileno())
    syncdir(f.parent); return dict(prepared=str(f),prepare_sha256=sha(f),bytes=len(b),contract=p['contract'])


def verify(path,expected):
    source.hash_value(expected); require(sha(path)==expected,'external prepare SHA')
    p=read(path,256*1024); now,data=collect_saved(**p['arguments'])
    require(p==now,'complete source/role/probe/budget reconstruction'); return p,data


def syncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


class Store(builder.Store):
    def __init__(self,path):
        self.path=Path(path).absolute(); require(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'NEW256MiB +200MiB floor')
        self.path.mkdir(); syncdir(self.path.parent); self.previous=None; self.index=0; self.broken=False
    def check(self,size=4096,final=False):
        used=sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+size<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size,'whole-output cap /200MiB floor')
    def write(self,name,value,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.path/name).is_symlink(),'flat exclusive artifact')
        self.check(len(value),final)
        with (self.path/name).open('ab' if append else 'xb') as f:
            require(f.write(value)==len(value),'short output write'); f.flush(); os.fsync(f.fileno())
        syncdir(self.path)
        return None if append else sha(self.path/name)
    def emit(self,kind,**fields):
        require(not self.broken,'no journal retry after partial write')
        try: return super().emit(kind,**fields)
        except BaseException:
            self.broken=True; raise
    def frame(self,name,value):
        from PIL import Image
        b=io.BytesIO(); Image.fromarray(value).save(b,format='PNG'); content=b.getvalue()
        require(len(content)<=PNG_CAP,'1MiB per PNG cap'); return self.write(name,content)


class FatalBudget(ValueError):
    """Stop whole run, keep all remaining planned rows as not_started."""


class Ops(old.Ops):
    def __init__(self,p,store,clock=time.monotonic):
        self.p,self.store,self.clock=p,store,clock; self.started=clock(); self.where={}; self.failed=False; self.index=0
        self.counts={k:dict(attempted=0,completed=0) for k in p['contract']['caps']}
    def call(self,kind,fn,**fields):
        require(not self.failed,'no dispatch after failed nested operation')
        try:
            if self.clock()-self.started>=self.p['contract']['wall_seconds']: raise FatalBudget('wall cap')
            c=self.counts[kind]
            if c['attempted']>=self.p['contract']['caps'][kind]: raise FatalBudget('predispatch cap '+kind)
            self.store.check(4096); c['attempted']+=1; index=self.index; self.index+=1
            self.store.emit('operation_attempted',operation_index=index,resource=kind,**self.where,**fields)
            v=fn(); c['completed']+=1
            self.store.emit('operation_completed',operation_index=index,resource=kind,**self.where,**fields); return v
        except BaseException:
            self.failed=True; raise


def body_dynamics(body):
    result={}
    for name in ('agent','block'):
        b=getattr(body,name)
        a=np.asarray([*b.position,*b.velocity,b.angle,b.angular_velocity,*b.force,b.torque],float)
        require(a.shape==(9,) and np.isfinite(a).all(),'finite complete exported body dynamics')
        result[name]=dict(position=a[:2].tolist(),velocity=a[2:4].tolist(),unwrapped_angle=float(a[4]),angular_velocity=float(a[5]),
            force=a[6:8].tolist(),torque=float(a[8]),is_sleeping=bool(b.is_sleeping),float64_bits=bits(a,'<f8'))
    return result


def prefix_signature(receipt,prefix,history):
    return dict(state_float64_bits=bits(receipt['state'],'<f8'),RGB_bytes_sha256=receipt['pixel_sha256'],
        body_dynamics=receipt['body_dynamics'],prefix_raw_float32_bits=bits(prefix,'<f4'),
        raw_history_float32_bits=bits(np.concatenate((np.zeros((10,2),np.float32),raw5(prefix))),'<f4'),
        action_history_contract=history)


class PrefixMismatch(ValueError):
    """An exact prefix failed; no suffix dispatch, retain denominator."""


class BranchSink:
    def __init__(self,store,p,data,plan,row,body,anchors):
        self.store,self.p,self.data,self.plan,self.row,self.body,self.anchors=store,p,data,plan,row,body,anchors
        self.latest=None; self.returned=[]
    def __getattr__(self,name): return getattr(self.store,name)
    def emit(self,kind,**fields):
        if kind=='returned_transition' and fields['phase']=='plan' and fields['step']<=5:
            self.returned.append(fields['command'])
        if kind=='fresh_observation':
            receipt=fields['receipt']; receipt['body_dynamics']=body_dynamics(self.body); self.latest=receipt
        self.store.emit(kind,**fields)
        if kind=='guard_receipt' and fields['phase']=='plan' and fields['step']==5 and fields.get('accepted') is True:
            plan=self.plan; pos=self.p['positions'][plan['position_id']]; k=plan['root_slot']; receipt=self.latest
            require(len(self.returned)==5 and bits(self.returned,'<f4')==bits(pos['prefix_commands'],'<f4'),'five actual prefix returns')
            sig=prefix_signature(receipt,self.returned,self.p['roots'][k]['action_history_contract'])
            ok=True; reasons=[]
            if any(a['role'] in ('G','D') for a in pos['role_aliases']):
                old_cp=self.data['old_prefix'][k]
                if (bits(receipt['state'],'<f8')!=bits(old_cp['state'],'<f8') or receipt['pixel_sha256']!=old_cp['pixel_sha256'] or
                    digest(receipt['snapshot'])!=digest(old_cp['snapshot'])):
                    ok=False; reasons.append('original actual G/D prefix state/RGB/snapshot mismatch')
            pid=plan['position_id']
            if plan['probe_index']==0:
                require(pid not in self.anchors,'no replacement zero prefix')
                self.anchors[pid]=deepcopy(sig)
            elif pid not in self.anchors or digest(sig)!=digest(self.anchors[pid]):
                ok=False; reasons.append('within-position zero-prefix exact mismatch/unavailable')
            self.row.update(prefix_exact=ok,prefix_signature=sig,prefix_checkpoint=deepcopy(receipt))
            self.store.emit('prefix_pairing',branch=plan['index'],position_id=pid,probe_index=plan['probe_index'],accepted=ok,reasons=reasons,signature=sig)
            if not ok: raise PrefixMismatch('; '.join(reasons))


def retain(row,exc,stage):
    e=dict(stage=stage,type=type(exc).__name__,message=str(exc)[:2000])
    if row['first_error'] is None: row['first_error']=e
    else: row['secondary_errors'].append(e)
    row['status']='prefix_not_reproduced' if isinstance(exc,PrefixMismatch) else 'failed_partial'


def install_hooks(env,ops):
    """Hook actual nested reset setter/goal-render/space steps, with cleanup."""
    body=env.unwrapped; originals=[]; spaces=[]; resetting=False
    def install_space():
        space=body.space
        if any(s[0] is space for s in spaces): return
        owned='step' in space.__dict__; fn=space.step; spaces.append((space,owned,fn))
        space.step=lambda *a,**kw:ops.call('physics_step',lambda:fn(*a,**kw))
    old_set,old_render,old_reset=body._set_state,body.render,env.reset
    def setter(value): install_space(); return ops.call('set_state',lambda:old_set(value))
    def render(): return ops.call('goal_render',old_render) if resetting else old_render()
    def reset(*a,**kw):
        nonlocal resetting
        resetting=True
        try: return old_reset(*a,**kw)
        finally: resetting=False
    try:
        for obj,key,new in ((body,'_set_state',setter),(body,'render',render),(env,'reset',reset)):
            originals.append((obj,key,key in obj.__dict__,getattr(obj,key))); setattr(obj,key,new)
    except BaseException:
        for obj,key,owned,fn in reversed(originals):
            if owned: setattr(obj,key,fn)
            else: delattr(obj,key)
        raise
    def restore(row):
        for obj,key,owned,fn in [(s,'step',owned,fn) for s,owned,fn in reversed(spaces)]+list(reversed(originals)):
            try:
                if owned: setattr(obj,key,fn)
                else: delattr(obj,key)
            except BaseException as exc: retain(row,exc,'restore_hook')
    return restore


def full_endpoint(row):
    return row.get('prefix_exact') is True and row.get('full_horizon') is True and row['status'] in ('horizon_completed','native_success') and row['first_error'] is None


def analyze(p,rows):
    require(len(rows)==len(p['plans']) and [(r['index'],r['position_id'],r['probe_index']) for r in rows]==
        [(x['index'],x['position_id'],x['probe_index']) for x in p['plans']],'fixed21M rows, no filtering')
    positions=[]; response={}
    for pos in p['positions']:
        group=[r for r in rows if r['position_id']==pos['position_id']]; require(len(group)==21,'all21 denominator')
        zero=group[0] if full_endpoint(group[0]) else None; root=p['roots'][pos['root_slot']]; u=root['direction']['position_unit']; au=root['direction']['angle_chord_unit']
        values=[]; columns=[]
        for r in group:
            available=zero is not None and full_endpoint(r); delta=T=angle_delta=angleT=None
            if available:
                a=source.vector(r['final_state']); z=source.vector(zero['final_state']); delta=(a[:4]-z[:4]).tolist()
                T=float(np.asarray(u)@delta) if u is not None else None
                angle_delta=[float(np.cos(a[4])-np.cos(z[4])),float(np.sin(a[4])-np.sin(z[4]))]
                angleT=float(np.asarray(au)@angle_delta) if au is not None else None
            v=dict(probe_index=r['probe_index'],full_endpoint=full_endpoint(r),available=available,delta_position4=delta,
                agent_delta=None if delta is None else delta[:2],object_delta=None if delta is None else delta[2:],
                common_direction_T=T,angle_chord_delta=angle_delta,common_angle_chord_T=angleT)
            values.append(v); response[(pos['position_id'],r['probe_index'])]=v
        for j in range(10):
            a,b=group[1+2*j],group[2+2*j]; ok=zero is not None and full_endpoint(a) and full_endpoint(b)
            B=E=along=None
            if ok:
                ap=source.vector(a['final_state'])[:4]; bp=source.vector(b['final_state'])[:4]; zp=source.vector(zero['final_state'])[:4]
                B=((ap-bp)/(2*RHO)).tolist(); E=(ap+bp-2*zp).tolist(); along=float(np.asarray(u)@B) if u is not None else None
            columns.append(dict(coordinate=j,available=ok,B=B,E=E,a=along))
        avec=[c['a'] for c in columns]; drift=None
        if zero is not None: drift=(np.asarray(zero['final_state'])[:4]-np.asarray(zero['prefix_checkpoint']['state'])[:4]).tolist()
        positions.append(dict(position_id=pos['position_id'],root_slot=pos['root_slot'],role_aliases=pos['role_aliases'],planned=21,
            complete=sum(full_endpoint(r) for r in group),missing_probe_indices=[r['probe_index'] for r in group if not full_endpoint(r)],
            zero_endpoint_available=zero is not None,zero_suffix_drift_position4=drift,responses=values,columns=columns,
            a_vector=avec,a_L2=float(np.linalg.norm(avec)) if all(v is not None for v in avec) else None))
    comparisons=[]
    for k in (0,1):
        ids=[pos['position_id'] for pos in p['positions'] if pos['root_slot']==k]
        for a in ids:
            for b in ids:
                if a>=b: continue
                comparisons.append(dict(root_slot=k,position_a=a,position_b=b,by_probe=[dict(probe_index=i,
                    T_b_minus_a=None if response[(a,i)]['common_direction_T'] is None or response[(b,i)]['common_direction_T'] is None else
                    response[(b,i)]['common_direction_T']-response[(a,i)]['common_direction_T']) for i in range(21)]))
    return dict(positions=positions,position_comparisons=comparisons,G_minus_D='structural zero: identical intervention aliases, not two observations',
        fitted_controller=False,posthoc_action_selection=False,P1_admission=False)


def execute(p,data,store,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN physics authorization')
    rows=[dict(index=x['index'],root_slot=x['root_slot'],position_id=x['position_id'],probe_index=x['probe_index'],arm=x['arm'],
        selection_digest=x['selection_digest'],status='not_started',first_error=None,secondary_errors=[],warmup_returned=0,
        own_attempted=0,own_returned=0,initial_exact=False,full_horizon=False,prefix_exact=False,native_success=False,
        terminated=False,truncated=False,PNG=0,checkpoints=[]) for x in p['plans']]
    result=dict(schema=NAME+'.report',status='running',contract=p['contract'],rows=rows,errors=[],first_error=None,
        logical_roles=p['role_aliases'],positions=p['positions'],roots=p['roots'],inherited_costs=p['inherited_costs'],
        new_neural=0,new_physics='actual hook counts below',P1_admission=False,P2_admission=False,new_independent_samples=0,source_verified=False)
    ops=Ops(p,store); anchors={}
    def problem(stage,exc):
        e=dict(stage=stage,type=type(exc).__name__,message=str(exc)[:2000]); result['errors'].append(e)
        if result['first_error'] is None: result['first_error']=e
    try:
        store.emit('run_started',contract=p['contract'],positions=p['positions'],roles=p['role_aliases'],plans_sha256=digest(p['plans']))
        for plan,row in zip(p['plans'],rows,strict=True):
            env=None; restore=None
            try:
                store.check(262144)
                if ops.clock()-ops.started>=p['contract']['wall_seconds']: raise FatalBudget('wall before factory')
                store.emit('environment_create_attempt',branch=plan['index']); env=factory(); store.emit('environment_create_completed',branch=plan['index'])
                body=env.unwrapped; builder.validate_env(body)
                require(body.render_action is False and body.render_size==224 and body.render_mode=='rgb_array','same target-free native RGB contract')
                sink=BranchSink(store,p,data,plan,row,body,anchors); ops.store=sink
                restore=install_hooks(env,ops); row['status']='running'
                old.branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,sink,plan['index'])
            except BaseException as exc:
                retain(row,exc,'branch')
                if result['first_error'] is None: result['first_error']=deepcopy(row['first_error'])
                if isinstance(exc,(FatalBudget,KeyboardInterrupt,SystemExit)) or ops.failed or store.broken:
                    problem('execution_stop',exc); break
            finally:
                if restore is not None: restore(row)
                if env is not None:
                    try: store.emit('environment_close_attempt',branch=plan['index'])
                    except BaseException as exc: retain(row,exc,'close_attempt_journal')
                    try:
                        env.close()
                    except BaseException as exc: retain(row,exc,'close')
                    else:
                        try: store.emit('environment_close_completed',branch=plan['index'])
                        except BaseException as exc: retain(row,exc,'close_completion_journal')
                ops.store=store
                try: store.emit('branch_closed',receipt=row)
                except BaseException as exc: problem('branch_journal',exc)
            if store.broken: break
    except BaseException as exc: problem('execution',exc)
    try:
        for path,h in p['bindings'].items(): require(sha(path)==h,'final source/input drift '+path)
        result['source_verified']=True
    except BaseException as exc: problem('final_source',exc)
    try: result['analysis']=analyze(p,rows)
    except BaseException as exc: problem('final_arithmetic',exc)
    result['counts']=ops.counts; result['wall_seconds']=ops.clock()-ops.started
    all_full=all(full_endpoint(r) for r in rows)
    try:
        c=ops.counts
        require(all(v['attempted']==v['completed'] for v in c.values()),'unreturned nested calls')
        if all_full:
            require(c['reset']['completed']==2*len(rows),'reset census')
            require(c=={k:dict(attempted=v,completed=v) for k,v in p['contract']['caps'].items()},'complete actual hook counts, not budget formula alone')
        require(c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'render/PNG actual census')
    except BaseException as exc: problem('final_counts',exc)
    result.update(planned=len(rows),completed=sum(full_endpoint(r) for r in rows),not_started=sum(r['status']=='not_started' for r in rows),
        native_successes=sum(r['native_success'] for r in rows),status='common_prefix_probes_completed' if all_full and not result['errors'] else 'partial_or_failed')
    if result['first_error'] is None:
        result['first_error']=next((r['first_error'] for r in rows if r['first_error'] is not None),None)
    try: store.emit('run_finished',status=result['status'],counts=ops.counts,planned=len(rows),completed=result['completed'],errors=result['errors'])
    except BaseException as exc: problem('final_journal',exc); result['status']='partial_or_failed'
    result['journal_chain_tail']=store.previous
    try:
        hashes={f.name:sha(f) for f in store.path.iterdir() if f.is_file()}
        store.put('manifest.json',dict(artifact_sha256=hashes,journal_chain_tail=store.previous),final=True)
        result['manifest_sha256']=sha(store.path/'manifest.json')
        store.put('report.json',result,final=True)
    except BaseException as exc:
        problem('final_report',exc); result['status']='partial_or_failed'; print(source.canonical(dict(first_error=result['first_error'],errors=result['errors'])),flush=True)
    return result


def run(prepared,expected,output,*,authorization,factory=None):
    require(authorization==AUTH,'explicit MAIN physics authorization before verify/factory')
    p,data=verify(prepared,expected); store=Store(output)
    store.put('execution_binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    if factory is None:
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
    return execute(p,data,store,factory,authorization=authorization)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('sourcepack','source-sha','source-prepared','source-prepare-sha','planner-report','planner-sha','physical-report','physical-sha','prepared','prepare-sha','output'):
        parser.add_argument('--'+key)
    parser.add_argument('--wall-seconds',type=int,default=600); parser.add_argument('--authorize-physics',action='store_true'); a=parser.parse_args()
    if a.command=='prepare':
        keys='sourcepack source_sha source_prepared source_prepare_sha planner_report planner_sha physical_report physical_sha wall_seconds'.split()
        r=prepare(a.output,**{k:getattr(a,k) for k in keys})
    elif a.command=='verify':
        p,_=verify(a.prepared,a.prepare_sha); r=dict(status='saved_protocol_verified_no_execution',contract=p['contract'])
    else:
        require(a.authorize_physics,'MAIN --authorize-physics required'); r=run(a.prepared,a.prepare_sha,a.output,authorization=AUTH)
    print(source.canonical({k:v for k,v in r.items() if k not in ('rows','analysis','roots','positions','logical_roles')}))
    if a.command=='run' and r['status']!='common_prefix_probes_completed': raise SystemExit(2)


if __name__=='__main__': main()
