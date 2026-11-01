"""Fixed common-prefix native192 fidelity experiment; MAIN HOST CUDA only.

Actual endpoints are labels, never predictor inputs or action selectors.
Prepare/verify and callback tests are saved-data/NumPy only.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_common_prefix_probe_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v1 as rgb

NAME='p1_constructive_common_prefix_probe_cuda_20261027_v1'
ROOT=physical.ROOT; base=rgb.planner; enc=base.encode; core=enc.core
require,sha,read=physical.require,physical.sha,physical.read
canonical=base.canonical
PREPARE_SHA='70616f76f08f560301cd0b5445cad8fcb41a9e53415bcac7a9d0efbf34cf560e'
REPORT_SHA='450b182e4ed4b02959f74593099cf85c728619bfc5ddc3678ea1940c846e1210'
PHYSICAL_SOURCE_SHA='d48047562096a52a077b00e9ab16e109b81974495ce023bd1e1d6d5bef1564dd'
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,256*1024


def contract():
    return dict(roots=2,positions=4,probes_per_position=21,planned=84,images=92,
        image_order='root0 warm5/warm10, root1 warm5/warm10; position0..3 prefix5; branch0..83 terminal10',
        caps=dict(encoder=92,projector=92,predict=84,action_encoder=84),
        context='actual [warm5,warm10,prefix5]; normalized [raw zero5,own raw prefix5]',
        probes=physical.probes(),horizon=1,representation='native_projected_cls192',Q='identity192',
        direction='approved blindtask goal - original root latent, shared within each root; unit and unnormalized signed projections',
        secant='(z_plus-z_minus)/(2*.25), logits coordinate not raw derivative; no AD',
        labels='full guarded terminal10 only; never predictor input; no selection or fitting',
        partial='all84 rows and92 image slots retained; fixed missing slots skipped, never replaced/zero-filled; resource counts actual',
        zero_norm='undefined direction -> null signed projections, not zero',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False,batch=1,dtype='float32'),
        AD=0,J=0,VJP=0,SVD=0,solve=0,physics=0,training=False,selection=False,
        no_cpu_neural_fallback=True,no_retries=True,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,
        P1_admission=False,P2_admission=False,causal_root_cause_claim=False)


def load_journal(path,tail):
    """Stream full hash chain; retain only input/label provenance, not futures for selection."""
    require(Path(path).stat().st_size<=physical.CAP,'bounded physical journal')
    previous=None; count=0; kept=[]; census={k:dict(attempted=0,completed=0) for k in physical.contract(4)['caps']}; active={}; first=last=None; next_operation=0
    with Path(path).open('rb') as stream:
        for line in stream:
            require(len(line)<=2*1024**2 and line.endswith(b'\n'),'untorn bounded event')
            e=base.decode(line)
            require(e['index']==count and e['previous_sha256']==previous and
                physical.digest({k:v for k,v in e.items() if k!='event_sha256'})==e['event_sha256'],'physical full hashchain')
            previous=e['event_sha256']; count+=1; last=e
            if first is None: first=e
            kind=e['kind']
            if kind in ('operation_attempted','operation_completed'):
                resource=e['resource']; c=census.setdefault(resource,dict(attempted=0,completed=0)); key=e['operation_index']
                if kind=='operation_attempted':
                    require(key==next_operation,'unique ordered actual operation'); next_operation+=1; active[key]=resource; c['attempted']+=1
                else:
                    require(active.pop(key,None)==resource,'operation return identity'); c['completed']+=1
            if kind=='fresh_observation':
                r=e['receipt']
                if (r['phase'],r['step']) in (('warmup',5),('warmup',10),('plan',5),('plan',10)): kept.append(e)
            elif kind in ('branch_started','branch_closed','prefix_pairing','returned_transition','guard_receipt'): kept.append(e)
    require(first is not None and first['kind']=='run_started' and last['kind']=='run_finished' and previous==tail,'physical terminal closure')
    return dict(first=first,last=last,events=kept,counts=census,unreturned=active)


def assemble(p,r,j):
    """Pure identity/provenance join. Fixed slot indices, including partials."""
    require(p['schema']==physical.NAME+'.prepare' and r['schema']==physical.NAME+'.report' and
        r['status'] in ('common_prefix_probes_completed','partial_or_failed') and r['P1_admission'] is False,
        'terminal physical protocol, no scientific admission')
    require(p['contract']==r['contract']==physical.contract(4) and len(p['positions'])==4 and len(p['plans'])==len(r['rows'])==84,
        'fixed4 positions /84 denominator /original600s contract')
    require(r['roots']==p['roots'] and r['positions']==p['positions'] and r['logical_roles']==p['role_aliases'] and
        r['source_verified'] is True,'same roots/roles/recipes')
    require(j['first']['positions']==p['positions'] and j['first']['roles']==p['role_aliases'] and
        j['first']['plans_sha256']==physical.digest(p['plans']) and j['last']['counts']==r['counts']==j['counts'] and
        j['last']['status']==r['status'] and j['last']['errors']==r['errors'],'journal/report/plan census')
    require([x['position_id'] for x in p['positions']]==list(range(4)) and p['probes']==physical.probes(),'original fixed probe bank')
    if r['status']=='common_prefix_probes_completed':
        require(not j['unreturned'] and r['errors']==[] and all(physical.full_endpoint(x) for x in r['rows']) and
            r['counts']=={k:dict(attempted=v,completed=v) for k,v in physical.contract(4)['caps'].items()},'full status requires all actual hooks and84 endpoints')
    events=j['events']; images=[]; rows=[]
    def receipt(branch,phase,step):
        es=[e for e in events if e['kind']=='fresh_observation' and e.get('branch')==branch and
            e['receipt']['phase']==phase and e['receipt']['step']==step]
        if not es: return None
        require(len(es)==1,'unique physical receipt'); c=deepcopy(es[0]['receipt'])
        gs=[e for e in events if e['kind']=='guard_receipt' and e.get('branch')==branch and e['phase']==phase and e['step']==step]
        require(len(gs)==1 and gs[0]['png']==c['png'] and gs[0]['file_sha256']==c['file_sha256'],'fresh guard correspondence')
        require(c['state']==c['observation_state']==c['snapshot']['state'],'state/RGB provenance')
        return c if gs[0]['accepted'] is True else None
    def image(role,root,position,branch,cp):
        x=dict(index=len(images),role=role,root_slot=root,position_id=position,branch=branch,checkpoint=cp,
            available=cp is not None); images.append(x); return x['index']
    # Never choose a later successful branch as a replacement for the fixed zero branch.
    for root in (0,1):
        pos=next(x for x in p['positions'] if x['root_slot']==root); b=21*pos['position_id']
        for step in (5,10): image('warm'+str(step),root,None,b,receipt(b,'warmup',step))
    for pos in p['positions']:
        pid=pos['position_id']; b=21*pid; row=r['rows'][b]
        cp=receipt(b,'plan',5) if row['prefix_exact'] is True else None
        if cp is not None: require(cp==row['prefix_checkpoint'],'prefix checkpoint binding')
        image('prefix5',pos['root_slot'],pid,b,cp)
    for i,(plan,row) in enumerate(zip(p['plans'],r['rows'],strict=True)):
        pid=i//21; k=i%21; pos=p['positions'][pid]
        require(plan['index']==row['index']==i and plan['position_id']==row['position_id']==pid and
            plan['probe_index']==row['probe_index']==k and plan['root_slot']==row['root_slot']==pos['root_slot'] and
            plan['selection_digest']==row['selection_digest'] and plan['commands']==pos['prefix_commands']+p['probes'][k]['commands'],
            'exact all84 branch/root/position/probe/action identity')
        closed=[e for e in events if e['kind']=='branch_closed' and e['receipt'].get('index')==i]
        require((len(closed)==1 and closed[0]['receipt']==row) or (not closed and row['status']=='not_started'),'durable row or explicit not_started')
        if row['status']!='not_started':
            starts=[e for e in events if e['kind']=='branch_started' and e.get('branch')==i]
            require(len(starts)==1 and starts[0]['plan']==plan,'branch precommit')
        if row['initial_exact']:
            require(row['root_state']==p['roots'][pos['root_slot']]['direction']['root_state'] and
                row['goal']==p['roots'][pos['root_slot']]['goal'],'original native root/goal identity')
        ts=[e for e in events if e['kind']=='returned_transition' and e.get('branch')==i and e['phase']=='plan']
        require([e['step'] for e in ts]==list(range(1,row['own_returned']+1)) and
            [e['command'] for e in ts]==plan['commands'][:row['own_returned']],'all actual returned commands')
        require(all(e['returned_goal_state']==p['roots'][pos['root_slot']]['goal'] for e in ts),'same approved physical task goal throughout')
        if row['prefix_exact']:
            pairs=[e for e in events if e['kind']=='prefix_pairing' and e.get('branch')==i]
            require(len(pairs)==1 and pairs[0]['accepted'] is True and pairs[0]['signature']==row['prefix_signature']==r['rows'][21*pid]['prefix_signature'],
                'all within-position prefix signatures exact')
            prefix=receipt(i,'plan',5)
            require(prefix is not None and prefix==row['prefix_checkpoint'] and physical.prefix_signature(prefix,
                [e['command'] for e in ts[:5]],p['roots'][pos['root_slot']]['action_history_contract'])==row['prefix_signature'],
                'recomputed actual state64/RGB/raw-prefix32/history/fullbody signature')
        cp=receipt(i,'plan',10) if physical.full_endpoint(row) else None
        if cp is not None:
            require(cp in row['checkpoints'] and cp['state']==row['final_state'] and ts[-1]['state']==cp['state'],'full terminal label only')
        ix=image('terminal10',pos['root_slot'],pid,i,cp)
        rows.append(dict(index=i,root_slot=pos['root_slot'],position_id=pid,probe_index=k,terminal_image=ix,
            physical_status=row['status'],full_endpoint=cp is not None,selection_digest=row['selection_digest']))
    require(len(images)==92,'all92 fixed image slots'); return images,rows


def material(physical_report,physical_sha):
    core.hash_value(physical_sha); path=Path(physical_report).resolve()
    require(physical_sha==REPORT_SHA,'this version pins approved actual84 report; partial semantics tested without resealing upstream')
    require(sha(physical.__file__)==PHYSICAL_SOURCE_SHA and sha(path)==physical_sha,'external actual report / frozen physical source')
    r=read(path); binding=read(path.parent/'execution_binding.json')
    require(binding['prepare_sha256']==PREPARE_SHA and sha(binding['prepared'])==PREPARE_SHA,'approved physical prepare')
    p=read(binding['prepared']); bindings={str(path):physical_sha,str(path.parent/'execution_binding.json'):sha(path.parent/'execution_binding.json'),
        str(Path(binding['prepared']).resolve()):PREPARE_SHA}
    for f,h in p['bindings'].items(): require(sha(f)==h,'upstream seal '+f); bindings[f]=h
    manifest_path=path.parent/'manifest.json'; manifest=read(manifest_path)
    require(sha(manifest_path)==r['manifest_sha256'] and manifest['journal_chain_tail']==r['journal_chain_tail'],'manifest/tail binding')
    bindings[str(manifest_path)]=r['manifest_sha256']
    jp=path.parent/'journal.jsonl'; require(sha(jp)==manifest['artifact_sha256']['journal.jsonl'],'manifest journal SHA')
    j=load_journal(jp,r['journal_chain_tail']); bindings[str(jp)]=sha(jp)
    images,rows=assemble(p,r,j)
    for item in images:
        if not item['available']: continue
        c=item['checkpoint']; f=rgb.local(path.parent,c['png'])
        require(manifest['artifact_sha256'][c['png']]==c['file_sha256'],'PNG manifest provenance')
        rgb.png(f,c['file_sha256'],c['pixel_sha256']); bindings[str(f)]=c['file_sha256']; item['png_path']=str(f)
    # Approved blind task only: never load sourcepack, witness/certificate or candidate outcomes.
    parent=read(p['arguments']['planner_report']); require(sha(p['arguments']['planner_report'])==physical.PLANNER_SHA,'approved planner')
    parent_p=read(parent['prepared']); require(sha(parent['prepared'])==parent['prepare_sha256'],'parent prepare')
    bindings[str(Path(parent['prepared']).resolve())]=parent['prepare_sha256']; tasks=[]
    for k,slot in enumerate(parent_p['slots']):
        require(slot['order']==k and sha(slot['task_export'])==slot['task_sha256'],'fixed blind task seal')
        export=read(slot['task_export']); t=export['task']; core.validate_task(t)
        require(export['source_slot']==k and core.task_id(t)==slot['task_id']==p['roots'][k]['task_id'] and
            slot['task_sha256']==p['roots'][k]['task_sha256'] and np.array_equal(t['projection'],np.eye(192)),'same original task/identity Q')
        require(not np.asarray(t['root_raw_action_history']).any(),'original real zero history')
        require(np.array_equal(np.asarray(t['action_mean'],np.float32),base.d2p.ACTION_MEAN) and
            np.array_equal(np.asarray(t['action_std'],np.float32),base.d2p.ACTION_STD),'original native normalizer')
        tasks.append(t); bindings[str(Path(slot['task_export']).resolve())]=slot['task_sha256']
    require(len(tasks)==2,'exact two approved tasks'); enc.native_metadata()
    files=physical.source_files()+base.source_files()+[Path(rgb.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files}); bindings.update(enc.MODELS)
    for f,h in bindings.items(): require(sha(f)==h,'final input/source readback')
    return dict(schema=NAME+'.prepare',arguments=dict(physical_report=str(path),physical_sha=physical_sha),
        bindings=bindings,images=images,rows=rows,positions=p['positions'],tasks=tasks,contract=contract(),
        policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,
        inherited_costs=dict(physical=r['counts'],upstream=p['inherited_costs'])),None


def prepare(output,**arguments):
    p,_=material(**arguments); value=canonical(p); f=Path(output).absolute()
    require(not f.exists() and len(value)<=2*1024**2 and shutil.disk_usage(f.parent).free>=FLOOR+len(value),'NEW small prepare/200MiB floor')
    with f.open('xb') as stream: require(stream.write(value)==len(value),'short prepare'); stream.flush(); os.fsync(stream.fileno())
    physical.syncdir(f.parent); return dict(prepared=str(f),prepare_sha256=sha(f),bytes=len(value))


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA'); p=read(path,2*1024**2); now,_=material(**p['arguments'])
    require(p==now,'full source/frame/action/task reconstruction'); return p


def observed_context(task,warm5,warm10,prefix,raw_prefix):
    for a in (warm5,warm10,prefix): enc.native(a)
    actions=np.concatenate((np.zeros((5,2),np.float32),physical.raw5(raw_prefix)))
    c=dict(task['root_context'],latent_history=[a.astype(float).tolist() for a in (warm5,warm10,prefix)],
        action_history=core.normalized_tokens(task,actions).astype(float).tolist())
    core.context(c); return c


def vector(value):
    a=np.asarray(value,dtype=float); require(a.shape==(192,) and np.isfinite(a).all(),'finite native192'); return a


def arithmetic(p,predicted,actual):
    require(len(predicted)==len(actual)==84,'full84 denominator')
    result=[]
    for pos in p['positions']:
        pid=pos['position_id']; task=p['tasks'][pos['root_slot']]; g=vector(task['goal']); root=vector(task['root_context']['latent_history'][-1])
        d=g-root; norm=float(np.linalg.norm(d)); u=d/norm if norm else None; offset=21*pid
        ps=predicted[offset:offset+21]; zs=actual[offset:offset+21]; rows=[]; columns=[]
        def signed(a): return None if a is None or u is None else dict(raw=float(d@a),unit=float(u@a))
        for k,(pred,real) in enumerate(zip(ps,zs,strict=True)):
            dp=vector(pred)-vector(ps[0]) if pred is not None and ps[0] is not None else None
            dz=vector(real)-vector(zs[0]) if real is not None and zs[0] is not None else None
            rows.append(dict(probe_index=k,predicted_available=pred is not None,actual_available=real is not None,
                predicted_vs_actual_mse=None if pred is None or real is None else float(np.mean((vector(pred)-vector(real))**2)),
                predicted_Q=None if pred is None else float(np.mean((vector(pred)-g)**2)),
                actual_Q=None if real is None else float(np.mean((vector(real)-g)**2)),
                predicted_delta=None if dp is None else dp.tolist(),actual_delta=None if dz is None else dz.tolist(),
                predicted_signed=signed(dp),actual_signed=signed(dz),
                delta_mse=None if dp is None or dz is None else float(np.mean((dp-dz)**2))))
        for coordinate in range(10):
            a,b=1+2*coordinate,2+2*coordinate; col=dict(coordinate=coordinate)
            for label,bank in (('predicted',ps),('actual',zs)):
                B=(vector(bank[a])-vector(bank[b]))/(2*physical.RHO) if bank[a] is not None and bank[b] is not None else None
                E=vector(bank[a])+vector(bank[b])-2*vector(bank[0]) if B is not None and bank[0] is not None else None
                col[label]=dict(B=None if B is None else B.tolist(),signed=signed(B),E=None if E is None else E.tolist())
            bp,ba=col['predicted']['B'],col['actual']['B']
            col['secant_mse']=None if bp is None or ba is None else float(np.mean((np.asarray(bp)-ba)**2)); columns.append(col)
        result.append(dict(position_id=pid,root_slot=pos['root_slot'],planned=21,rows=rows,columns=columns,
            common_direction=d.tolist(),direction_norm=norm,direction_defined=u is not None,
            predicted_B_192x10=None if any(c['predicted']['B'] is None for c in columns) else np.asarray([c['predicted']['B'] for c in columns]).T.tolist(),
            actual_B_192x10=None if any(c['actual']['B'] is None for c in columns) else np.asarray([c['actual']['B'] for c in columns]).T.tolist()))
    return dict(planned=84,positions=result,units='native192 latent only; not physical position/angle u',P1_admission=False,P2_admission=False)


class Ledger:
    def __init__(self,sink,sync=lambda:None): self.sink,self.sync=sink,sync; self.entries=[]; self.scope=None; self.failed=False
    def call(self,resource,fn):
        require(not self.failed and self.scope is not None,'scoped inference/no retry')
        expected=('encoder','projector') if self.scope[0]=='image' else ('action_encoder','predict')
        current=[e['resource'] for e in self.entries if e['scope']==list(self.scope)]
        require(len(current)<2 and resource==expected[len(current)],'exact per-slot module order/no duplicate')
        require(sum(e['resource']==resource for e in self.entries)<contract()['caps'][resource],'predispatch cap')
        e=dict(index=len(self.entries),scope=list(self.scope),resource=resource,completed=False); self.entries.append(e)
        try:
            self.sink(dict(kind='module_attempt',entry=deepcopy(e))); value=fn(); self.sync(); e['completed']=True
            self.sink(dict(kind='module_completed',entry=deepcopy(e))); return value
        except BaseException as exc:
            self.failed=True
            try: self.sink(dict(kind='module_error',entry=deepcopy(e),error=str(exc)))
            except BaseException as secondary: exc.add_note('journal error '+str(secondary))
            raise
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in contract()['caps']}


@contextmanager
def guard(model,torch,ledger):
    undo=[]; primary=None
    def patch(obj,name,value): undo.append((obj,name,getattr(obj,name))); setattr(obj,name,value)
    def forbidden(*a,**kw): ledger.failed=True; raise ValueError('AD/training forbidden')
    try:
        for resource,module,shape in (('encoder',model.encoder,(1,3,224,224)),('projector',model.projector,(1,192)),
            ('action_encoder',model.action_encoder,(1,3,10)),('predict',model.predictor,(1,3,192))):
            original=module.forward
            def forward(*a,_r=resource,_fn=original,_shape=shape,**kw):
                require(not torch.is_grad_enabled() and a[0].device.type=='cuda' and tuple(a[0].shape)==_shape,'native single CUDA inference')
                return ledger.call(_r,lambda:_fn(*a,**kw))
            patch(module,'forward',forward)
        for obj,names in ((torch.autograd,('grad','backward')),(torch.Tensor,('backward',)),
            (torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
            (torch.func,('grad','grad_and_value','jacrev','jacfwd','hessian','jvp','vjp'))):
            for name in names: patch(obj,name,forbidden)
        yield
    except BaseException as exc: primary=exc; raise
    finally:
        failures=[]
        for obj,name,original in reversed(undo):
            try: setattr(obj,name,original)
            except BaseException as exc: failures.append(exc)
        if failures:
            if primary is not None: primary.add_note('restore '+repr(failures))
            else: raise failures[0]


def execute(p,encode,predict,ledger,store,image_rows,rows):
    """RGB-only encoding callback; predictor receives context and fixed raw probe ONLY."""
    zs={}; predictions=[None]*84; labels=[None]*84
    for item,row in zip(p['images'],image_rows,strict=True):
        if not item['available']: row['status']='missing'; continue
        row['status']='attempted'; ledger.scope=('image',item['index'])
        z=encode(item); enc.native(z); zs[item['index']]=z.copy()
        row.update(status='encoded',native192=z.astype(float).tolist()); store.event(dict(kind='image_saved',receipt=deepcopy(row)))
    # All terminal labels are stored above but are not passed through this input interface.
    for pos in p['positions']:
        pid=pos['position_id']; root=pos['root_slot']; task=p['tasks'][root]; ids=(2*root,2*root+1,4+pid)
        if not all(i in zs for i in ids):
            for row in rows[21*pid:21*pid+21]: row['status']='missing_observed_context'
            continue
        require(np.array_equal(zs[ids[0]],np.asarray(task['root_context']['latent_history'][1],np.float32)) and
            np.array_equal(zs[ids[1]],np.asarray(task['root_context']['latent_history'][2],np.float32)),'reencoded observed warm5/10 must exactly anchor original blindtask')
        context=observed_context(task,*(zs[i] for i in ids),pos['prefix_commands'])
        store.event(dict(kind='observed_context_frozen',position_id=pid,context=context))
        for k in range(21):
            index=21*pid+k; row=rows[index]; ledger.scope=('predict',index); row['status']='attempted'
            commands=physical.raw5(p['contract']['probes'][k]['commands'])
            after=predict(deepcopy(context),commands.copy()); base.d2p.validate_advance(context,commands,after)
            z=np.asarray(after['latent_history'][-1],np.float32); enc.native(z); predictions[index]=z.astype(float).tolist()
            row.update(status='predicted',input_context=deepcopy(context),commands=commands.astype(float).tolist(),prediction_context=after)
            store.event(dict(kind='H1_saved',receipt=deepcopy(row)))
    for i,row in enumerate(p['rows']):
        ix=row['terminal_image']
        if ix in zs: labels[i]=zs[ix].astype(float).tolist()
    ledger.scope=None
    return arithmetic(p,predictions,labels)


def run(prepared,expected,output):
    p=verify(prepared,expected); store=base.Store(output); errors=[]; model=None; before=after=host=None; source_ok=False; analysis=None
    ledger=Ledger(store.event)
    image_rows=[dict(index=x['index'],role=x['role'],root_slot=x['root_slot'],position_id=x['position_id'],branch=x['branch'],status='not_started',provenance=x) for x in p['images']]
    rows=[dict(x,status='not_started') for x in p['rows']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model); before=enc.tensor_hash(model); require(before==p['expected_model_hash'],'unchanged original model')
        transform=p0.image_transform(); backend=base.d2p.TorchBackend(model,device); ledger.sync=lambda:torch.cuda.synchronize(device)
        def encode(item):
            c=item['checkpoint']; pixels=rgb.png(item['png_path'],c['file_sha256'],c['pixel_sha256'])
            x=p0._image_tensor(pixels,transform=transform,device=device); z=model.encode({'pixels':x})['emb']
            require(z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'original projected CLS192')
            return z[0,0].detach().cpu().numpy().copy()
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH),guard(model,torch,ledger):
            analysis=execute(p,encode,backend.step,ledger,store,image_rows,rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(enc.error('error_journal',secondary))
    finally:
        ledger.scope=None
        for row in image_rows+rows:
            if row['status']=='attempted': row['status']='failed_partial'
        if model is not None:
            try: enc.frozen(model); after=enc.tensor_hash(model)
            except BaseException as exc: errors.append(enc.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(enc.error('final_source',exc))
        try: store.close()
        except BaseException as exc: errors.append(enc.error('final_journal',exc))
        all_full=all(x['status']=='encoded' for x in image_rows) and all(x['status']=='predicted' for x in rows)
        expected_counts={k:dict(attempted=v,completed=v) for k,v in contract()['caps'].items()}
        if all_full and ledger.counts()!=expected_counts: errors.append(dict(stage='final_counts',message='actual full counters !=92/84'))
        closed=not errors and not ledger.failed and source_ok and before is not None and before==after and analysis is not None
        result=dict(schema=NAME+'.report',status=('common_prefix_native_fidelity_completed_not_admitted' if all_full else 'partial_saved_data_evaluated_not_admitted') if closed else 'partial_or_failed',
            inputs=p['arguments'],prepare_sha256=expected,rows=rows,images=image_rows,analysis=analysis,resources=ledger.counts(),errors=errors,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            source_verified=source_ok,host=host,contract=p['contract'],inherited_costs=p['inherited_costs'],
            P1_admission=False,P2_admission=False,new_AD=0,new_physics=0,training=False)
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as exc: errors.append(enc.error('artifact_hash',exc)); closed=False
        result['artifact_sha256']=hashes
        if not closed: result['status']='partial_or_failed'
        try: store.put('report.json',result,final=True)
        except BaseException as exc: errors.append(enc.error('final_report',exc)); print(canonical(dict(errors=errors)).decode(),flush=True); closed=False
    require(closed,'failed partial retained; no retry '+str(errors)); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('physical-report','physical-sha','prepared','prepare-sha','output'): parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare': result=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha)
    elif a.command=='verify': result=dict(status='verified_no_execution',contract=verify(a.prepared,a.prepare_sha)['contract'])
    else:
        r=run(a.prepared,a.prepare_sha,a.output); result={k:v for k,v in r.items() if k not in ('rows','images','analysis')}
    print(canonical(result).decode(),end='')


if __name__=='__main__': main()
