"""Fixed 24x3 moving CEM/nominal-MPPI; MAIN-only CUDA, no final replay."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import os
from pathlib import Path
import shutil
import sysconfig
import time
import zlib
import numpy as np
from experiments.spectral_reachability import p1_moving_sampling_baselines_cuda_20261027_v1 as old

NAME = 'moving_prediction_matched_sampling_cuda_20261027_v1'
ROOT, core, gn, enc, base = old.ROOT, old.core, old.gn, old.enc, old.base
require, sha, canonical, read = old.require, old.sha, old.canonical, old.read
ARMS = old.ARMS
RESOURCES = ('predict_API', 'predictor', 'action_encoder', 'pred_proj')
CAP, FLOOR, FINAL = 64*1024**2, 200*1024**2, 512*1024
WALL, EXPANDED = 900, 256*1024**2
OLD_SHA = 'f30a1c5ac5ef618cfe3add37326aae29513278fe7c4319b5a743d9f6422300e6'
OFFICIAL = Path(sysconfig.get_paths()['purelib'])/'stable_worldmodel/wm/lewm/lewm.py'
OFFICIAL_SHA = 'ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2'


def budget(n=144):
    return {k: dict(attempted=n, completed=n) for k in RESOURCES}


def contract():
    return dict(roots=2, arms=list(ARMS), groups=4, iterations=3, samples=24,
        candidates_per_group=72, horizon=2, raw_steps=10, batch_size=1,
        per_group=budget(), total=budget(576),
        predictor_samples_per_group=144, total_predictor_samples=576,
        CEM=dict(elite=4, initial_mean=0., initial_std=1., std_clip=[.05, 1.],
            update='projected logits mean/std ddof0; no smoothing/incumbent injection'),
        nominal_MPPI=dict(initial_nominal=0., temperature=1., noise_std=.5,
            parameter_box='[-cap,+cap]^20', update='softmax-cost weighted BOX PARAMETERS, then boxclip',
            scored_nominals=0, importance_correction=False, official=False),
        selection='stable global argmin(cost,index) of ALL72 already evaluated candidates; both arms',
        MPPI_difference_from_old='24x3 proposals only; no initial/updated nominal scoring; sampled winner, not final nominal',
        seeds={a:[old.seed(s,a) for s in (0,1)] for a in ARMS}, rng='NumPy default_rng PCG64',
        cap=float(.75*np.sqrt(5)), action_map='float32 tanh(packet-ball-projected logits); raw10x2',
        Q='original identity192 terminal mean192 squared error',
        history='original moving three states/TWO actual normalized incoming packets; full rolling H2',
        encoder_report_sha256=old.REPORT_SHA, source_authority_sha256=old.SOURCE_AUTHORITY_SHA,
        inherited_encoding_once=old.contract()['inherited_encoding_once'],
        count_scope='actual API dispatch and module entries, charged before call; failures no refund; batch1',
        action_encoder_context_rows_per_group=432, pred_proj_context_rows_per_group=432,
        extra_replay=0, padding=0, nominal_evaluations=0, AD=0, J=0, VJP=0, SVD=0,
        new_encoder=0, new_image_projector=0, new_physics=0, training=False, Qfit=False,
        MATH=True, TF32=False, autocast=False, CPU_neural_fallback=False, retries=0,
        output_cap_bytes=CAP, floor_bytes=FLOOR, final_reserve_bytes=FINAL, wall_seconds=WALL,
        wall_scope='load/search and dispatch/write checkpoints; cannot preempt a blocked native call',
        matched_claim='CEM vs local nominal-MPPI API calls AND predictor samples only',
        equal_FLOPs=False, equal_total_compute=False, equal_AD_compute=False,
        with_target=False, independent_confirmation=False, P1_admission=False)


def build():
    require(sha(old.__file__)==OLD_SHA and sha(OFFICIAL)==OFFICIAL_SHA, 'frozen old loader/kernel and official model API')
    p=deepcopy(old.build())
    p.update(schema=NAME+'.prepare', arguments={}, contract=contract())
    for f in (Path(__file__), OFFICIAL, ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        p['bindings_sha256'][str(f.resolve())]=sha(f)
    require(len(p['slots'])==2, 'fixed original two moving tasks')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h, 'prepare source closure TOCTOU')
    return p


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<=256*1024 and not path.exists() and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(raw), 'exclusive small prepare / floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    require(sha(path)==expected, 'external prepare SHA')
    p=read(path);require(p==build(), 'entire own prepare/source/model/task closure');return p


class Store(base.Store):
    """Own caps/time; inherited event/put/close use this override, no globals patched."""
    def __init__(self,output):
        self.started=time.monotonic();self.output=Path(output).absolute()
        require(not self.output.exists() and not self.output.is_symlink() and
            shutil.disk_usage(self.output.parent).free>=FLOOR+CAP, 'NEW64MiB +200MiB floor')
        self.output.mkdir();self.comp=zlib.compressobj(6,zlib.DEFLATED,31)
        self.closed=self.broken=False;self._write('events.jsonl.gz',b'')

    def check_time(self):
        require(time.monotonic()-self.started<=WALL, '900s dispatch/write wall limit')

    def _write(self,name,raw,append=False,final=False):
        if not final:self.check_time()
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(), 'flat regular artifact')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and
            shutil.disk_usage(self.output).free>=FLOOR+len(raw), 'all output bytes/cap/floor')
        with (self.output/name).open('ab' if append else 'xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


class Meter:
    """Four resource counters per fixed root/arm. Actual callbacks, not simulated padding."""
    def __init__(self,sink,sync=lambda:None,check=lambda:None):
        self.sink,self.sync,self.check=sink,sync,check
        self.group=None;self.packet=None;self.ops=[];self.fault=False

    def emit(self,kind,**values):
        require(not self.fault, 'failed journal/dispatch cannot resume')
        try:self.sink(dict(kind=kind,evaluation_id=self.group,**deepcopy(values)))
        except BaseException:self.fault=True;raise

    def counts(self):
        return {r:dict(attempted=sum(e['resource']==r for e in self.ops),
            completed=sum(e['resource']==r and e['status']=='returned' for e in self.ops)) for r in RESOURCES}

    def predictor_samples(self):
        return dict(attempted=sum(e['samples'] for e in self.ops if e['resource']=='predictor'),
            completed=sum(e['samples'] for e in self.ops if e['resource']=='predictor' and e['status']=='returned'))

    def available(self):
        self.check();require(not self.fault and all(v['attempted']<144 for v in self.counts().values()), 'packet predispatch 144 cap/no retry')

    def begin(self,resource,inputs):
        self.check();require(not self.fault and resource in RESOURCES and self.counts()[resource]['attempted']<144,
            'resource predispatch cap/no retry')
        pending=[e for e in self.ops if e['status']=='pending']
        require((not pending if resource in ('action_encoder','predict_API') else
            len(pending)==1 and pending[0]['resource']=='predict_API'), 'only one API with sequential module children')
        e=dict(index=len(self.ops),resource=resource,candidate=self.group,packet=self.packet,
            batch_size=1,samples=1,context_rows=3,inputs=deepcopy(inputs),status='pending')
        self.ops.append(e);self.emit('dispatch_attempted',receipt=e);return e

    def end(self,e,output):
        require(any(e is x for x in self.ops) and e['status']=='pending', 'issued live ticket')
        require(not any(x is not e and x['status']=='pending' for x in self.ops) or e['resource']!='predict_API', 'children must return first')
        self.sync();e.update(status='returned',output=deepcopy(output));self.emit('dispatch_returned',receipt=e)

    def fail(self,e,exc):
        e.update(status='failed',error_type=type(exc).__name__,message=str(exc))
        try:self.emit('dispatch_failed',receipt=e)
        finally:self.fault=True

    def call(self,resource,inputs,fn,pack=lambda x:x):
        e=self.begin(resource,inputs)
        try:
            value=fn();self.end(e,pack(value));return value
        except BaseException as exc:
            try:self.fail(e,exc)
            except BaseException as secondary:exc.add_note('failure receipt: '+repr(secondary))
            raise


def tensor_receipt(value):
    a=value.detach().cpu().numpy()
    require(a.dtype==np.float32 and np.isfinite(a).all(), 'finite native float32 module receipt')
    return dict(shape=list(a.shape),dtype='float32',values=a.astype(float).tolist())


def module_shapes(resource,shapes):
    if resource=='action_encoder':valid=shapes==[(1,3,10)]
    elif resource=='predictor':valid=len(shapes)==2 and shapes[0]==(1,3,192) and len(shapes[1])==3 and shapes[1][:2]==(1,3)
    else:valid=resource=='pred_proj' and len(shapes)==1 and len(shapes[0])==2 and shapes[0][0]==3
    require(valid, 'actual batch1/context3 module predispatch shape')


class NativeCuda(gn.screen.NativeCuda):
    """Same full-H2 math, own actual API and module hooks (no method monkeypatch)."""
    @contextmanager
    def guard(self,meter):
        hooks=[];pending={};primary=None
        require(not hasattr(self,'_active_meter'), 'one active backend')
        self._active_meter=meter
        def pre(resource):
            def enter(module,args):
                require(module not in pending, 'no recursive module')
                shapes=[tuple(x.shape) for x in args]
                module_shapes(resource,shapes)
                pending[module]=meter.begin(resource,[tensor_receipt(x) for x in args])
            return enter
        def post(module,args,output):
            ticket=pending.pop(module,None)
            if ticket is not None:
                if output is None:meter.fail(ticket,RuntimeError('module failed/no output'))
                else:meter.end(ticket,tensor_receipt(output))
        def forbid(module,args):raise ValueError('new image encoder/projector forbidden')
        try:
            for module,r in ((self.model.predictor,'predictor'),(self.model.action_encoder,'action_encoder'),(self.model.pred_proj,'pred_proj')):
                hooks.append(module.register_forward_pre_hook(pre(r)))
                hooks.append(module.register_forward_hook(post,always_call=True))
            for module in (self.model.encoder,self.model.projector):hooks.append(module.register_forward_pre_hook(forbid))
            yield
        except BaseException as exc:primary=exc;raise
        finally:
            cleanup=[]
            for h in reversed(hooks):
                try:h.remove()
                except BaseException as exc:cleanup.append(repr(exc))
            del self._active_meter
            if cleanup:
                if primary is not None:primary.add_note('hook cleanup: '+repr(cleanup))
                else:raise RuntimeError('hook cleanup: '+repr(cleanup))

    def _step(self,task,z,h,raw):
        m=self._active_meter;m.available();m.packet=0 if m.packet is None else m.packet
        token=(raw.reshape(10)-self._tensor(np.tile(task['action_mean'],5)))/self._tensor(np.tile(task['action_std'],5))
        require(np.array_equal(token.detach().cpu().numpy(),core.normalized_tokens(task,raw.detach().cpu().numpy())[0]), 'normalization once exact')
        outgoing=self.torch.cat((h,token[None,None]),1)
        before=m.counts();a=self.model.action_encoder(outgoing)
        output=m.call('predict_API',[tensor_receipt(z),tensor_receipt(a)],lambda:self.model.predict(z,a),tensor_receipt)
        require(all(m.counts()[r]['attempted']==before[r]['attempted']+1 and
            m.counts()[r]['completed']==before[r]['completed']+1 for r in RESOURCES), 'actual API and each module exactly once')
        successor=output[:,-1:];m.packet+=1
        return self.torch.cat((z[:,1:],successor),1),outgoing[:,1:]


def proposals(rng,arm,mean,std,cap):
    if arm==ARMS[0]:parameters=rng.normal(mean,std,size=(24,2,10))
    else:parameters=np.clip(mean+rng.normal(size=(24,2,10))*.5,-cap,cap)
    return parameters,[gn.screen.projected_logits(x,cap) for x in parameters]


def update(arm,chunk,cap):
    require(len(chunk)==24 and all(np.isfinite(c['cost']) and c['cost']>=0 for c in chunk), '24 complete finite costs')
    if arm==ARMS[0]:
        elite=sorted(range(24),key=lambda k:(chunk[k]['cost'],chunk[k]['index']))[:4]
        x=np.asarray([chunk[i]['logits'] for i in elite]);mean=x.mean(axis=0);std=np.clip(x.std(axis=0,ddof=0),.05,1.)
        detail=dict(elite_indices=[chunk[i]['index'] for i in elite])
    else:
        w=old.nominal.kernels.softmax_weights(np.asarray([c['cost'] for c in chunk]),temperature=1.)
        mean=np.clip(np.tensordot(w,np.asarray([c['parameters'] for c in chunk]),axes=(0,0)),-cap,cap)
        std=np.full((2,10),.5);detail=dict(weights=w.tolist())
    return mean,std,detail


def optimize(task,slot,arm,backend,meter):
    core.validate_task(task);require(type(slot) is int and slot in (0,1) and arm in ARMS, 'fixed group')
    rng=np.random.default_rng(old.seed(slot,arm));cap=task['packet_logit_cap']
    mean=np.zeros((2,10));std=np.ones((2,10)) if arm==ARMS[0] else np.full((2,10),.5)
    bank=[];updates=[]
    for it in range(3):
        params,xs=proposals(rng,arm,mean,std,cap)
        meter.emit('bank_frozen',iteration=it,mean=mean.tolist(),std=std.tolist(),parameters=params.tolist(),logits=[x.tolist() for x in xs])
        chunk=[]
        for j,x in enumerate(xs):
            meter.group=f'candidate_{len(bank)}';meter.packet=0
            p=gn.single(task,x,backend,meter,'whole_replay')
            c=dict(index=len(bank),iteration=it,parameters=params[j].tolist(),prediction=p)
            c.update(old.cem.compact(p));bank.append(c);chunk.append(c);meter.emit('candidate_saved',candidate=c)
        mean,std,detail=update(arm,chunk,cap)
        u=dict(iteration=it,mean=mean.tolist(),std=std.tolist(),**detail)
        updates.append(u);meter.emit('distribution_update',update=u)
    require(len(bank)==72, 'ALL72 no initial/nominal/replay')
    winner=min(range(72),key=lambda i:(bank[i]['cost'],i))
    meter.emit('winner_sealed_no_replay',index=winner,commands=bank[winner]['commands'],cost=bank[winner]['cost'])
    return dict(task_id=core.task_id(task),source_slot=slot,arm=arm,seed=old.seed(slot,arm),
        candidates=bank,updates=updates,winner_index=winner)


def choice(task,result,slot,arm):
    """Pure-array exact RNG/update/path/cost reconstruction, no inference."""
    require(result['task_id']==core.task_id(task) and result['source_slot']==slot and result['arm']==arm and
        result['seed']==old.seed(slot,arm) and len(result['candidates'])==72, 'group/RNG/full candidate denominator')
    bank=result['candidates'];events=[]
    class Saved:
        def rollout(self,t,raw,m,mode):
            i=int(m.group.split('_')[-1]);p=bank[i]['prediction'];old.nominal.check_single(t,p)
            require(p['commands']==raw.astype(float).tolist(), 'saved exact raw10')
            for k,c in enumerate(p['contexts']):m.emit('native_packet',mode=mode,packet=k,context=c)
            return deepcopy(p['contexts'])
    rebuilt=optimize(task,slot,arm,Saved(),Meter(events.append))
    require(rebuilt==result, 'all72 deterministic RNG/projection/H2/history/Q/update/winner')
    c=bank[result['winner_index']];p=c['prediction']
    return dict(task_id=core.task_id(task),commands=c['commands'],logits=c['logits'],prediction=p['contexts'][-1],
        model_terminal_cost=c['cost'],selected_index=c['index']),events


def literal(r,shape=None):
    require(set(r)=={'shape','dtype','values'} and r['dtype']=='float32', 'literal module receipt schema')
    a=np.asarray(r['values']);require(list(a.shape)==r['shape'] and (shape is None or a.shape==shape) and
        np.isfinite(a).all() and np.array_equal(a,a.astype(np.float32)), 'literal finite float32 shape')
    return a


def journal_check(task,result,slot,arm,events):
    selected,expected=choice(task,result,slot,arm)
    algorithm=[e for e in events if not e['kind'].startswith('dispatch_')]
    require(algorithm==expected, 'complete causal candidate banks/H2/results/updates/winner journal')
    operations=[];live={};packet_ops={};active=None;packet=0
    for e in events:
        if e['kind']=='evaluation_request':
            require(active is None and not live, 'no overlapping candidate')
            active=e['evaluation_id'];packet=0
        elif e['kind']=='native_packet':
            require(not live and active==e['evaluation_id'] and e['packet']==packet and
                len(packet_ops.get((active,packet),[]))==4, 'all packet dispatches returned before durable context')
            packet+=1
        elif e['kind']=='evaluation_complete':
            require(not live and active==e['evaluation_id'] and packet==2, 'H2 complete before cost')
            active=None
        if not e['kind'].startswith('dispatch_'):continue
        r=e['receipt'];i=r['index'];resource=r['resource']
        require(active==e['evaluation_id']==r['candidate'] and r['packet']==packet and resource in RESOURCES and r['batch_size']==r['samples']==1 and r['context_rows']==3, 'scoped batch1 receipt')
        if e['kind']=='dispatch_attempted':
            require(i==len(operations) and r['status']=='pending', 'pre-dispatch sequential actual ticket')
            require(not live if resource in ('action_encoder','predict_API') else len(live)==1 and next(iter(live.values()))['resource']=='predict_API', 'module under actual API')
            operations.append(r);live[i]=r;packet_ops.setdefault((r['candidate'],r['packet']),[]).append(i)
        else:
            require(e['kind']=='dispatch_returned' and i in live and r==dict(live[i],status='returned',output=r['output']), 'actual matching successful return')
            if resource=='predict_API':require(len(live)==1, 'module returns before API')
            operations[i]=r;del live[i]
    require(active is None and not live and len(operations)==576, 'per-group 144x4 real dispatches returned')
    counts={k:dict(attempted=sum(r['resource']==k for r in operations),completed=sum(r['resource']==k for r in operations)) for k in RESOURCES}
    require(counts==budget(), 'exact per-group actual budget')
    require(set(packet_ops)=={(f'candidate_{i}',k) for i in range(72) for k in (0,1)}, 'all144 packets unique')
    for c in result['candidates']:
        previous=task['root_context'];tokens=core.normalized_tokens(task,c['commands'])
        for k,context in enumerate(c['prediction']['contexts']):
            ops=[operations[i] for i in packet_ops[(f'candidate_{c["index"]}',k)]]
            require([r['resource'] for r in ops]==['action_encoder','predict_API','predictor','pred_proj'], 'one each per actual packet')
            ae,api,pred,proj=ops;outgoing=np.concatenate((previous['action_history'],tokens[k:k+1]))[None]
            require(len(ae['inputs'])==1 and np.array_equal(literal(ae['inputs'][0],(1,3,10)),outgoing), 'actual AE full incoming history plus current token')
            a=literal(ae['output']);require(a.ndim==3 and a.shape[:2]==(1,3), 'AE output logical batch/context')
            z=np.asarray(previous['latent_history'])[None]
            require(len(api['inputs'])==len(pred['inputs'])==2 and api['inputs']==pred['inputs'] and
                np.array_equal(literal(api['inputs'][0],(1,3,192)),z) and np.array_equal(literal(api['inputs'][1]),a), 'actual API/predictor input bridge')
            predout=literal(pred['output']);require(predout.ndim==3 and predout.shape[:2]==(1,3), 'predictor output context')
            require(len(proj['inputs'])==1 and np.array_equal(literal(proj['inputs'][0]),predout.reshape(3,-1)), 'actual predictor to projector')
            out=literal(proj['output'],(3,192));require(np.array_equal(literal(api['output'],(1,3,192)),out[None]) and
                np.array_equal(out[-1],context['latent_history'][-1]), 'actual API/projector output to rolled successor')
            previous=context
    return selected


def model_plan(slot,arm,selected):
    return dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=arm,
        blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
        task_contract_sha256=slot['task_contract_sha256'],with_target=False,physical_execution=False,
        selection_uses_physics=False,P1_admission=False,official_MPPI=False,**selected)


def initial_rows(p):
    return [dict(source_slot=s['source_slot'],arm=a,task_id=s['task_id'],blind_sha256=s['blind_sha256'],
        status='not_started',commands=None,resources={},predictor_samples=dict(attempted=0,completed=0)) for s in p['slots'] for a in ARMS]


def execute(task,slot,arm,model,device,torch,sink,row,check):
    m=Meter(sink,lambda:torch.cuda.synchronize(device),check)
    try:
        b=NativeCuda(model,device,task)
        require(not torch.is_autocast_enabled('cuda'), 'no CUDA autocast')
        with torch.inference_mode(),b.guard(m):r=optimize(task,slot,arm,b,m)
        require(m.counts()==budget(), 'actual per-group API/modules returned144')
        return r
    finally:row['resources']=m.counts();row['predictor_samples']=m.predictor_samples()


def plan_all(p,store,runner,rows):
    for row in rows:
        s=row['source_slot'];a=row['arm'];slot=p['slots'][s];row['status']='attempted'
        task=old.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        def sink(e):store.event(dict(source_slot=s,arm=a,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',seed=old.seed(s,a),task_sha256=slot['task_sha256']))
        r=runner(task,s,a,sink,row);selected,_=choice(task,r,s,a)
        require(row['resources']==budget() and row['predictor_samples']==dict(attempted=144,completed=144), 'no missing/extra actual dispatches/samples')
        stem=f'root_{s}_{a}';store.put(stem+'.result.json.gz',r)
        plan=model_plan(slot,a,selected);plan['selection_sha256']=core.digest(plan)
        store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],selection_sha256=plan['selection_sha256'],
            result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json')


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);rows=initial_rows(p)
    errors=[];model=None;before=after=host=None;source_ok=False;complete=False
    try:
        print('Two fixed known moving roots; CEM vs local nominal-MPPI 144 API/samples each. Not FLOPs or P1.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'], 'original frozen model')
        with sdpa_kernel(SDPBackend.MATH):
            plan_all(p,store,lambda t,s,a,k,r:execute(t,s,a,model,device,torch,k,r,store.check_time),rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(enc.error('error_journal',secondary))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as exc:errors.append(enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(enc.error('final_source',exc))
        try:store.close()
        except BaseException as exc:errors.append(enc.error('journal_finalize',exc))
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as exc:errors.append(enc.error('artifact_listing',exc));files=[]
        for f in files:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as exc:errors.append(enc.error('artifact_hash',exc))
        resources={k:{phase:sum(r['resources'].get(k,{}).get(phase,0) for r in rows) for phase in ('attempted','completed')} for k in RESOURCES}
        samples={phase:sum(r['predictor_samples'][phase] for r in rows) for phase in ('attempted','completed')}
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows) and resources==budget(576) and samples==dict(attempted=576,completed=576)
        report=dict(schema=NAME+'.report',status='matched_sampling_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],rows=rows,
            resources=resources,predictor_samples=samples,errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,commands_released=complete,inherited_encoding_once=p['contract']['inherited_encoding_once'],
            elapsed_seconds=time.monotonic()-store.started,training=False,new_physics=0,P1_admission=False,equal_total_compute=False)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:complete=False;print(canonical(dict(report_write_error=enc.error('final_report',exc),errors=errors)).decode(),flush=True)
    require(complete, 'partial evidence preserved; no retry: '+str(errors));return report


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected, 'external actual report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='matched_sampling_plans_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is True and
        r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['contract']==p['contract'] and r['inherited_encoding_once']==p['contract']['inherited_encoding_once'] and
        r['resources']==budget(576) and r['predictor_samples']==dict(attempted=576,completed=576), 'whole-run complete actual model/resources/source contract')
    order=[(s,a) for s in (0,1) for a in ARMS]
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{a}.{x}' for s,a in order for x in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'}, 'all own artifacts only')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h, 'artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']), 'execution prepare binding')
    raw,eof=base.inflate((output/'events.jsonl.gz').read_bytes(),EXPANDED)
    require(eof and raw.endswith(b'\n'), 'single complete bounded durable journal')
    events=[base.decode(line) for line in raw.splitlines()];groups=[]
    for e in events:
        require(set(e)=={'source_slot','arm','task_id','event'}, 'only scoped successful events')
        key=(e['source_slot'],e['arm'])
        if not groups or groups[-1]!=key:groups.append(key)
    require(groups==order and len(r['rows'])==4, 'fixed four denominators/order/no resumption')
    plans=[]
    for row,(s,a) in zip(r['rows'],order,strict=True):
        slot=p['slots'][s];stem=f'root_{s}_{a}'
        require(row['source_slot']==s and row['arm']==a and row['task_id']==slot['task_id'] and row['blind_sha256']==slot['blind_sha256'] and
            row['status']=='model_plan_sealed' and row['resources']==budget() and row['predictor_samples']==dict(attempted=144,completed=144) and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json', 'strict root/arm/actual denominator join')
        task=old.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=read(output/row['result_file'])
        group=[e for e in events if (e['source_slot'],e['arm'])==(s,a)]
        require(all(e['task_id']==slot['task_id'] for e in group), 'journal actual task join')
        scope=[e['event'] for e in group]
        require(scope[0]==dict(kind='arm_started',seed=old.seed(s,a),task_sha256=slot['task_sha256']), 'start seal')
        selected=journal_check(task,result,s,a,scope[1:-1]);plan=model_plan(slot,a,selected)
        ident=core.digest(plan);plan['selection_sha256']=ident
        require(read(output/row['plan_file'])==plan and row['selection_sha256']==ident and row['commands']==selected['commands'] and
            scope[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=selected['commands']), 'already evaluated winner raw10 seal/no replay')
        plans.append(dict(plan,whole_run_report_sha256=expected))
    for path,h in p['bindings_sha256'].items():require(sha(path)==h, 'final source/model/input TOCTOU')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h, 'final artifact TOCTOU')
    require(sha(output/'report.json')==expected, 'final report SHA')
    return dict(schema=NAME+'.consume',rows=plans,resources=r['resources'],predictor_samples=r['predictor_samples'],inherited_encoding_once=r['inherited_encoding_once'],
        prediction_calls_and_samples_matched_between_sampling_arms=True,equal_total_compute=False,official_MPPI=False,P1_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(status='verified_no_execution',contract=verify(a.prepared,a.prepare_sha)['contract'])
    elif a.command=='run':r=run(a.prepared,a.prepare_sha,a.output)
    else:r=consume(a.output,a.report_sha)
    print(canonical(r).decode().strip())


if __name__=='__main__':main()
