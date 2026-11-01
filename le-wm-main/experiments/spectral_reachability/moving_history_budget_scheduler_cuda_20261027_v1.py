"""Preregistered four-policy moving/off fixed-H2 scheduler; MAIN CUDA only.

prepare/verify/consume are saved-array operations. No physics, training, visual
encoding, outcome feedback, baseline optimizer or zero-history gate is called.
"""
from __future__ import annotations

import argparse
import ast
import base64
from contextlib import contextmanager
from copy import deepcopy
import gzip
import os
from pathlib import Path
import shutil
import zlib
import numpy as np

from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as moving
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as authority
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v2 as fidelity

NAME = 'moving_history_budget_scheduler_cuda_20261027_v1'
ROOT = moving.ROOT
core, blind, enc = moving.core, moving.blind, moving.blind.encode
require, sha, read, canonical = moving.require, moving.sha, moving.read, moving.canonical
METHODS = ('U_uniform', 'D_geometry', 'S_swapped', 'N_no_signal_uniform')
CAP, FINAL, FLOOR = 16*1024**2, 1024**2, 200*1024**2
PREREG = ROOT.parent/'docs/D2P_MOVING_BUDGET_SCHEDULER_PREREG_20261027.md'
PREREG_SHA = '8e8cd4091dbaed5991dc119638ffdc00c8aed0852d0ed4c54cc6b1b3a82009bf'
HANDOFF = ROOT/'results/moving_history_planner_handoff_main01.json'
HANDOFF_SHA = '52c475ed499b077f261517bc592252b208926919909674cebf68fa4029cc8310'
ANCHOR_REPORT = ROOT/'results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json'
ANCHOR_SHA = 'cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0'
ENC_REPORT = ROOT/'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json'
ENC_SHA = 'e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6'
SOURCEPACK = ROOT/'results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json'
SOURCEPACK_SHA = 'c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
SOURCE_PREPARE = ROOT/'results/p1_constructive_moving_history_source_20261027_v1_prepare01.json'
SOURCE_PREPARE_SHA = 'c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9'
BALL = float(.75*np.sqrt(5))
NEURAL = ('predict', 'action_encoder', 'predictor', 'pred_proj')
MAXIMA = dict.fromkeys(NEURAL, 520) | dict(jacobian=2, vjp_directions=384, grad_api=384,
    raw_backward_hook=384, svd=2)
FLAGS = dict(P1_admission=False, P3_admission=False, independent_confirmation=False,
    compute_advantage=False, walltime_advantage=False)


def finite(a, shape):
    a = np.asarray(a)
    require(a.shape == shape and np.isfinite(a).all(), 'finite array shape '+str(shape))
    return a


def blob(value):
    a = np.ascontiguousarray(value)
    require(a.dtype in (np.dtype('float32'), np.dtype('float64')) and np.isfinite(a).all(), 'full native finite arrays')
    raw = a.tobytes()
    return dict(shape=list(a.shape), dtype=str(a.dtype), raw_sha256=core.sha(raw), bytes_b64=base64.b64encode(raw).decode())


def unblob(b):
    require(set(b) == {'shape', 'dtype', 'raw_sha256', 'bytes_b64'} and b['dtype'] in ('float32', 'float64'), 'array receipt')
    raw = base64.b64decode(b['bytes_b64'], validate=True)
    require(core.sha(raw) == b['raw_sha256'], 'raw array SHA')
    a = np.frombuffer(raw, dtype=b['dtype']).reshape(b['shape']).copy()
    require(np.isfinite(a).all(), 'finite saved array'); return a


def legal(raw, logits, anchor=False):
    raw = finite(raw, (10, 2)); logits = finite(logits, (2, 10))
    require(np.array_equal(raw, raw.astype(np.float32)) and (anchor or np.array_equal(logits, logits.astype(np.float32))), 'literal float32 raw; original anchor chart precision')
    # Original core.check_prediction domain tolerance, NOT a newly tuned cap.
    require(np.all(np.linalg.norm(logits.astype(np.float64), axis=1) <= BALL+(1e-12 if anchor else 0)), 'no anchor repair or proposal projection')
    require(np.array_equal(raw, np.tanh(logits.astype(np.float32)).reshape(10, 2)), 'literal float32 tanh binding')


def bank(slot, anchors):
    require(type(slot) is int and slot in (0, 1) and len(anchors) == 2, 'two fixed RNG slots and both anchors')
    rows = [dict(index=0, kind='zero', logits=np.zeros((2,10),np.float32).tolist(), commands=np.zeros((10,2),np.float32).tolist())]
    for k, a in enumerate(anchors, 1):
        legal(a['commands'], a['logits'], anchor=True)
        rows.append(dict(index=k, kind=moving.ARMS[k-1], logits=deepcopy(a['logits']), commands=deepcopy(a['commands'])))
    rng = np.random.Generator(np.random.PCG64(20261027+101*slot))
    for index in range(3,45):
        packets = []
        for _ in range(2):
            v = rng.standard_normal(10); norm = np.linalg.norm(v)
            require(norm > 0 and np.isfinite(norm), 'zero direction: reject, never redraw')
            packets.append(v/norm*BALL*rng.random()**(1/10))
        logits = np.asarray(packets, np.float32)
        raw = np.tanh(logits).reshape(10,2)
        legal(raw, logits)
        rows.append(dict(index=index, kind='PCG64_ball', logits=logits.astype(float).tolist(), commands=raw.astype(float).tolist()))
    return rows


def signal(task, nominal, jacobian, svd=np.linalg.svd):
    q = finite(task['projection'], (192,192)).astype(np.float64)
    r = q@(finite(task['goal'], (192,))-finite(nominal, (192,)))
    k = q@finite(jacobian, (192,20)); u,s,vt = svd(k, full_matrices=False)
    finite(u,(192,20)); finite(s,(20,)); finite(vt,(20,20))
    rank = int(np.count_nonzero(s > (s[0]*1e-6 if s.size else 0)))
    energy = float(r@r)
    outside = r-u[:,:rank]@(u[:,:rank].T@r)
    chi = 0. if energy == 0 else (1. if rank == 0 else float(np.clip((outside@outside)/energy,0,1)))
    return dict(rank=rank, singular_values=s.tolist(), residual=r.tolist(), energy=energy, chi=chi, w=(1-chi)*energy)


def allocate(method, weights=None):
    require(method in METHODS, 'fixed four strategies')
    if method in (METHODS[0], METHODS[3]):
        require(weights is None, 'uniform, especially N, receives NO signal'); return [10,10]
    w = finite(weights,(2,)).astype(float)
    require(np.all(w >= 0), 'nonnegative geometry weights')
    if method == METHODS[2]: w = w[::-1]
    b = 10 if w.sum() == 0 else int(np.floor(20*(.25+.5*w[0]/w.sum())+.5))
    require(5 <= b <= 15, '5..15 blocks per root'); return [b,20-b]


def schedule(blocks):
    require(len(blocks)==2 and sum(blocks)==20 and all(type(b) is int and 5<=b<=15 for b in blocks), 'fixed total20 blocks')
    out = [s for _ in range(5) for s in (0,1)]; used = [5,5]
    while len(out)<20:
        for s in (0,1):
            if used[s]<blocks[s]:out.append(s); used[s]+=1
    return out


def protocol():
    return dict(methods=list(METHODS), slots=[0,1], horizon=2, latent_dim=192, ball=BALL, seed='20261027+101*slot',
        casting='normal/radius float64 -> logits float32 once -> NumPy float32 tanh; no repair',
        blocks=20, candidates_per_block=3, maximum_bank=45, search_predict_per_method=120, replay_predict_per_method=8,
        shared_predict=8, actual_maxima=MAXIMA, processed_action_context_tokens=1560,
        standalone_logical_predict=dict(zip(METHODS,[136,136,136,128])), shared_actual_once=True,
        historical_encoding=dict(encoder=8,projector=8), historical_anchor_predict=288,
        historical_GN_per_root=dict(predict=78,vjp_directions=768), historical_D2P_per_root=dict(predict=66,vjp_directions=2112),
        historical_other_axes='see original full receipts; never infer unknown as zero',
        new_encoder=0,new_projector=0,new_solve=0,new_physics=0, training=False, Q='original off identity192 residual transform',
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=600,
        allocated_bytes=3*1024**3,reserved_bytes=4*1024**3, known_root_calibration=True, **FLAGS)


def closure():
    files=set(blind.source_files()+moving.moving.source_files()+[Path(__file__),Path(moving.__file__),
        Path(authority.__file__), Path(fidelity.__file__), PREREG, ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')])
    todo=list(files);seen=set()
    while todo:
        p=Path(todo.pop()).resolve()
        if p in seen or p.suffix!='.py' or not p.is_relative_to(ROOT/'experiments'):continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]+[node.module+'.'+x.name for x in node.names]
            for name in names:
                child=ROOT/Path(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and child.is_file() and child not in files: files.add(child);todo.append(child)
    return {str(p.resolve()):sha(p) for p in files}


def material(handoff_sha, anchor_report_sha, encoder_report_sha):
    """Saved-only binding. Privileged source proof stays in evaluator sidecar."""
    require([handoff_sha,anchor_report_sha,encoder_report_sha]==[HANDOFF_SHA,ANCHOR_SHA,ENC_SHA], 'fixed approved external inputs')
    bindings=closure()
    def bind(path,h):
        path=Path(path).resolve();require(sha(path)==h,'input/source hash '+str(path));bindings[str(path)]=h;return path
    require(sha(PREREG)==PREREG_SHA,'approved prereg unchanged')
    hand=read(bind(HANDOFF,handoff_sha));old=read(bind(ANCHOR_REPORT,anchor_report_sha))
    require(hand['schema']=='moving_history_planner_handoff.v1' and hand['errors']==[] and
        hand['projection']['source_report_sha256']==anchor_report_sha and hand['projection']['omitted_plan_fields']==['prediction'], 'handoff is not compiler report')
    require(old['schema']==moving.NAME+'.report' and old['errors']==[] and old['model_unchanged'] is True and old['commands_released'] is True,'original complete anchor authority')
    oldprepare=read(bind(old['prepared'],old['prepare_sha256']))
    for path,h in oldprepare['bindings_sha256'].items():bind(path,h)
    source=authority.load_source(SOURCEPACK,SOURCEPACK_SHA,SOURCE_PREPARE,SOURCE_PREPARE_SHA)
    bindings.update(source['bindings'])
    er=read(bind(ENC_REPORT,encoder_report_sha))
    require(er['status']=='moving_blind_tasks_encoded_and_sealed' and er['errors']==[] and er['model_unchanged'] is True and er['prepare_sha256']==moving.ENCODER_PREPARE_SHA,'original8 visual acquisition receipt')
    require(er['model_hash_before']==er['model_hash_after']==old['model_hash_before']==old['model_hash_after']==enc.MODEL_TENSOR_SHA,'same original encoding/anchor model tensors')
    for resource in ('encoder','projector'):
        require(er['resources'][resource]['attempted']==er['resources'][resource]['completed']==8,'actual historical8/8 visual samples')
    eb=read(bind(ENC_REPORT.parent/'binding.json',er['artifact_sha256']['binding.json']))
    # Encoder uses a different receipt spelling from planner; preserve exact file.
    ep=eb.get('prepared',eb.get('prepare'))
    require(ep is not None,'encoder prepare path receipt')
    bind(ep,er['prepare_sha256'])
    slots=[];tasks=[];banks=[]
    plans=hand['data']['plans'];require(len(plans)==4,'all GN/D2P anchors, no dropping arms')
    for s in (0,1):
        i=moving.load_export(moving.EXPORTS[s],moving.EXPORT_SHAS[s]);tasks.append(i['gn_task'])
        export=read(bind(moving.EXPORTS[s],moving.EXPORT_SHAS[s]));ref=source['references'][s]
        require(export['source_slot']==s and export['blind_sha256']==ref['blind_sha256'] and np.array_equal(ref['raw_history'],moving.fixed_raw()),'stable source/blind/history join')
        erow=[r for r in er['rows'] if r['task_id']==i['task_id']]
        require(len(erow)==1 and erow[0]['source_slot']==s and erow[0]['task_file_sha256']==moving.EXPORT_SHAS[s], 'encoder task identity join')
        image_arrays=[]
        for im in erow[0]['images']:
            path=bind(im['path'],im['sha256']);a=np.load(path,allow_pickle=False);finite(a,(192,));require(a.dtype==np.float32,'native original encoded arrays');image_arrays.append(a)
        require(len(image_arrays)==4 and np.array_equal(image_arrays[:3],i['gn_task']['root_context']['latent_history']) and np.array_equal(image_arrays[3],i['gn_task']['goal']), 'actual original context3+goal,8/8 acquisition')
        anchors=[];anchor_refs=[]
        for arm in moving.ARMS:
            pp=[p for p in plans if p['source_slot']==s and p['arm']==arm]
            rr=[r for r in old['rows'] if r['source_slot']==s and r['arm']==arm]
            require(len(pp)==len(rr)==1,'unique stable anchor identity')
            pp,rr=pp[0],rr[0];filename=rr['plan_file'];require(Path(filename).name==filename,'flat original plan')
            full=read(bind(ANCHOR_REPORT.parent/filename,old['artifact_sha256'][filename]))
            seal=full['selection_sha256'];unsealed={k:v for k,v in full.items() if k!='selection_sha256'}
            require(core.digest(unsealed)==seal==pp['selection_sha256']==rr['selection_sha256'],'original full selection proof, not projected pseudo-seal')
            projected={k:v for k,v in full.items() if k!='prediction'}
            projected['whole_run_report_sha256']=anchor_report_sha
            require(projected==pp and full['task_id']==i['task_id'] and full['blind_sha256']==ref['blind_sha256'] and full['task_sha256']==moving.EXPORT_SHAS[s], 'full/handoff/task/source provenance')
            anchors.append({k:deepcopy(full[k]) for k in ('commands','logits')})
            anchor_refs.append(dict(arm=arm,selection_sha256=seal,plan_file=str(ANCHOR_REPORT.parent/filename),handoff_pointer=f'/data/plans/{plans.index(pp)}'))
        b=bank(s,anchors);banks.append(b)
        slots.append(dict(source_slot=s,root_sha256=ref['root_sha256'],task_id=i['task_id'],blind_sha256=ref['blind_sha256'],
            task_export=moving.EXPORTS[s],task_sha256=moving.EXPORT_SHAS[s],task_contract_sha256=i['gn_task']['task_contract_sha256'],
            bank_sha256=core.digest(b),anchors=anchor_refs))
    for k in ('root_sha256','task_id','blind_sha256'):require(len({s[k] for s in slots})==2,'distinct stable roots')
    bindings.update(enc.MODELS)
    for path,h in bindings.items():require(sha(path)==h,'material final TOCTOU')
    p=dict(schema=NAME+'.prepare',arguments=dict(handoff_sha=handoff_sha,anchor_report_sha=anchor_report_sha,encoder_report_sha=encoder_report_sha),
        protocol=protocol(),slots=slots,bindings_sha256=bindings,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,
        historical_anchor_receipts=old['rows'],encoder_receipt=dict(report_sha256=encoder_report_sha,prepare_sha256=er['prepare_sha256'],resources=er['resources']))
    return p,tasks,banks


def prepare(output,**args):
    p,_,_=material(**args);raw=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(raw)<=256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw), 'NEW <=256KiB prepare +200MiB floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path,256*1024)
    actual,tasks,banks=material(**p['arguments']);require(p==actual,'exact current source/input closure')
    return p,tasks,banks


class Store(blind.Store):
    """Independent fixed16MiB writer; preserves every successful fsync."""
    def __init__(self,output):
        self.output=Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW16MiB output +200MiB floor')
        self.output.mkdir();self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat exclusive artifact')
        used=sum(p.stat().st_size for p in self.output.iterdir())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'16MiB cap/1MiB final reserve/200MiB floor')
        with (self.output/name).open('ab' if append else 'xb') as f:require(f.write(raw)==len(raw),'short durable write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


class Meter:
    def __init__(self,sink,limits,sync=lambda:None):
        self.sink,self.limits,self.sync=sink,limits,sync;self.events=[];self.scope=None;self.failed=False;self.forbidden_visual=[]
    def emit(self,kind,**fields):self.sink(dict(kind=kind,scope=deepcopy(self.scope),**fields))
    def begin(self,resource):
        self.limits.check();require(not self.failed and resource in MAXIMA and self.scope is not None,'scoped non-retrying dispatch')
        group,slot=self.scope
        require(group in ('shared',)+METHODS and type(slot) is int and slot in (0,1),'literal method/root scope')
        require(sum(e['resource']==resource for e in self.events)<MAXIMA[resource],'global predispatch cap '+resource)
        if resource in NEURAL:
            cap=4 if group=='shared' else (64 if group in (METHODS[0],METHODS[3]) else 94)
            require(sum(e['resource']==resource and e['scope']==self.scope for e in self.events)<cap,'root cap')
            if group!='shared':require(sum(e['resource']==resource and e['scope'][0]==group for e in self.events)<128,'method cap128')
        else:
            require(group=='shared','N/search never uses AD or SVD')
            cap=192 if resource in ('vjp_directions','grad_api','raw_backward_hook') else 1
            require(sum(e['resource']==resource and e['scope']==self.scope for e in self.events)<cap,'shared per-root math cap')
        e=dict(index=len(self.events),resource=resource,scope=deepcopy(self.scope),samples=1,completed=False)
        self.events.append(e);self.emit('resource_attempt',entry=deepcopy(e));return e
    def end(self,e):
        require(not e['completed'],'unique return');self.sync();self.limits.check();e['completed']=True;self.emit('resource_return',entry=deepcopy(e))
    def call(self,resource,fn):
        e=self.begin(resource)
        try:value=fn();self.end(e);return value
        except BaseException:self.failed=True;raise
    def counts(self,group=None,slot=None):
        es=[e for e in self.events if (group is None or e['scope'][0]==group) and (slot is None or e['scope'][1]==slot)]
        return {r:dict(attempted=sum(e['resource']==r for e in es),completed=sum(e['resource']==r and e['completed'] for e in es)) for r in MAXIMA}


class Native(moving.blind.gn.screen.NativeCuda):
    """Original full-H2 transition, own explicit predict API ledger; no globals."""
    @contextmanager
    def guard(self,meter):
        hooks=[];pending={};primary=None;self._active_meter=meter
        def arrays(x):
            if isinstance(x,(tuple,list)):return [arrays(v) for v in x]
            require(getattr(getattr(x,'device',None),'type',None)=='cuda' and x.dtype==self.torch.float32,'actual CUDA native float32 modules')
            return blob(x.detach().cpu().numpy())
        def pre(resource):
            def f(module,args):
                require(module not in pending,'no recursive module');e=meter.begin(resource);pending[module]=e
                meter.emit('module_input',entry=e['index'],resource=resource,arrays=arrays(args))
            return f
        def post(module,args,out):
            e=pending.pop(module,None)
            if e is not None and out is not None:
                meter.emit('module_output',entry=e['index'],resource=e['resource'],arrays=arrays(out));meter.end(e)
        def denied(module,args):
            meter.forbidden_visual.append(dict(scope=deepcopy(meter.scope),resource='encoder' if module is self.model.encoder else 'projector'))
            meter.emit('forbidden_visual_attempt',entry=meter.forbidden_visual[-1])
            raise ValueError('new visual encoder/projector forbidden')
        try:
            for m,r in ((self.model.action_encoder,'action_encoder'),(self.model.predictor,'predictor'),(self.model.pred_proj,'pred_proj')):
                hooks.append(m.register_forward_pre_hook(pre(r)));hooks.append(m.register_forward_hook(post,always_call=True))
            for m in (self.model.encoder,self.model.projector):hooks.append(m.register_forward_pre_hook(denied))
            yield
            require(not pending,'all native module returns complete')
        except BaseException as e:primary=e;meter.failed=True;raise
        finally:
            errors=[]
            for h in reversed(hooks):
                try:h.remove()
                except BaseException as e:errors.append(repr(e))
            del self._active_meter
            if errors:
                if primary is not None:primary.add_note('hook cleanup: '+repr(errors))
                else:raise RuntimeError('hook cleanup: '+repr(errors))

    def _step(self,task,z,h,raw):
        torch=self.torch;meter=self._active_meter
        token=(raw.reshape(10)-self._tensor(np.tile(task['action_mean'],5)))/self._tensor(np.tile(task['action_std'],5))
        require(np.array_equal(token.detach().cpu().numpy(),core.normalized_tokens(task,raw.detach().cpu().numpy())[0]),'actual float32 token arithmetic')
        outgoing=torch.cat((h,token[None,None]),1)
        meter.emit('context_input',context=dict(latent_history=z[0].detach().cpu().tolist(),action_history=outgoing[0].detach().cpu().tolist()))
        embedded=self.model.action_encoder(outgoing)
        prediction=meter.call('predict',lambda:self.model.predict(z,embedded))
        require(bool(torch.isfinite(prediction).all()) and tuple(prediction.shape)==(1,3,192),'finite full native prediction')
        meter.emit('predict_output',array=blob(prediction.detach().cpu().numpy()))
        return torch.cat((z[:,1:],prediction[:,-1:]),1),outgoing[:,1:]

    def jacobian(self,task,incumbent,meter):
        torch=self.torch
        with torch.enable_grad():
            raw=self._tensor(incumbent['commands']).requires_grad_(True);z,h=self._start(task)
            # Observable engine callback is separately named, not claimed kernel/FLOP count.
            hook=raw.register_hook(lambda g:(meter.call('raw_backward_hook',lambda:meter.emit('raw_backward',array=blob(g.detach().cpu().numpy()))),g)[1])
            primary=None
            try:
                paths=[]
                for k in range(2):
                    z,h=self._step(task,z,h,raw[5*k:5*k+5]);paths.append(self._saved(z,h));meter.emit('jacobian_forward_packet',packet=k,context=paths[-1])
                require(paths==incumbent['contexts'],'AD full path exact nominal equality MATH')
                terminal=z[0,-1];rows=[]
                for index in range(192):
                    def backward():return meter.call('grad_api',lambda:torch.autograd.grad(terminal[index],raw,retain_graph=index<191,create_graph=False)[0])
                    row=meter.call('vjp_directions',backward).detach().cpu().numpy().reshape(20).astype(np.float64)
                    rows.append(row);meter.emit('jacobian_row',row=index,array=blob(row))
                j=np.stack(rows)*(1-np.asarray(incumbent['commands']).reshape(20)**2)[None]
                return terminal.detach().cpu().numpy().astype(np.float64),j
            except BaseException as e:primary=e;raise
            finally:
                try:hook.remove()
                except BaseException as e:
                    if primary is not None:primary.add_note('raw hook removal: '+repr(e))
                    else:raise


def cost(task,path):
    x=finite(path[-1]['latent_history'][-1],(192,));q=np.asarray(task['projection'],np.float64)
    return float(np.mean((q@(x-np.asarray(task['goal'],np.float64)))**2))


def check_path(task,raw,path):
    require(len(path)==2,'complete H2 context census');before=task['root_context']
    for k,after in enumerate(path):moving.advance(task,before,np.asarray(raw,np.float32)[5*k:5*k+5],after);before=after


def model_plan(slot,method,winner):
    p=dict(schema=NAME+'.model_plan',**deepcopy(slot),arm=method,commands=deepcopy(winner['commands']),
        candidate_index=winner['index'],model_cost=winner['cost'],physical_execution=False,selection_uses_physics=False,**FLAGS)
    p['selection_sha256']=core.digest(p);return p


def panel(p,tasks,banks,backends,meter,result):
    """Algorithm inputs are blind tasks/banks only; no evaluator/outcome sidecar."""
    signals=result['signals'];rows=result['rows']
    for s in (0,1):
        meter.scope=['shared',s];b=backends[s];t=tasks[s];entry=signals[s];entry['status']='attempted'
        zero=banks[s][0];meter.emit('nominal_request',candidate=zero)
        with b.guard(meter):
            path=b.rollout(t,zero['commands'],meter,'nominal');entry['nominal_contexts']=path
            check_path(t,zero['commands'],path);meter.emit('nominal_complete',contexts=path)
            terminal,j=meter.call('jacobian',lambda:b.jacobian(t,dict(zero,contexts=path),meter))
        entry.update(terminal=terminal.tolist(),jacobian=blob(j));meter.emit('jacobian_complete',terminal=entry['terminal'],jacobian=entry['jacobian'])
        require(np.array_equal(terminal,path[-1]['latent_history'][-1]),'AD terminal nominal equality')
        entry['signal']=meter.call('svd',lambda:signal(t,terminal,j));entry['status']='complete';meter.emit('signal_complete',signal=entry['signal'])
    weights=[x['signal']['w'] for x in signals]
    for method in METHODS:
        # In particular N is constructed without passing weights/J to allocation or search.
        blocks=allocate(method) if method in (METHODS[0],METHODS[3]) else allocate(method,weights)
        result['allocations'][method]=blocks;order=schedule(blocks);seen=[0,0];curves=[]
        meter.scope=[method,None];meter.emit('allocation_frozen',blocks=blocks,schedule=order)
        for block,s in enumerate(order,1):
            meter.scope=[method,s];row=next(r for r in rows if r['arm']==method and r['source_slot']==s);row['status']='attempted'
            b,t=backends[s],tasks[s]
            with b.guard(meter):
                for index in range(seen[s]*3,seen[s]*3+3):
                    c=deepcopy(banks[s][index]);c['status']='attempted';row['candidates'].append(c);meter.emit('candidate_request',block=block,candidate=c)
                    path=b.rollout(t,c['commands'],meter,'search');c['contexts']=path;check_path(t,c['commands'],path)
                    c.update(cost=cost(t,path),status='complete');meter.emit('candidate_complete',candidate=c)
            seen[s]+=1
            if block in (10,20):
                best=[min(r['candidates'],key=lambda c:(c['cost'],c['index']))['cost'] for r in rows if r['arm']==method]
                curve=dict(block=block,best_costs=best,mean_cost=float(np.mean(best)),counts=meter.counts(method),elapsed_seconds=meter.limits.receipt()['elapsed_seconds'])
                curves.append(curve);meter.emit('paid_block_curve',curve=curve)
        result['curves'][method]=curves
        for s in (0,1):
            meter.scope=[method,s];row=next(r for r in rows if r['arm']==method and r['source_slot']==s);t,b=tasks[s],backends[s]
            winner=min(row['candidates'],key=lambda c:(c['cost'],c['index']));row['winner_index']=winner['index'];row['replays']={}
            meter.emit('winner_frozen',index=winner['index'],cost=winner['cost'],commands=winner['commands'])
            with b.guard(meter):
                for mode in ('whole_replay','segmented_replay'):
                    meter.emit('replay_request',mode=mode,commands=winner['commands'])
                    path=b.rollout(t,winner['commands'],meter,mode);row['replays'][mode]=path
                    check_path(t,winner['commands'],path);require(path==winner['contexts'],'search/whole/segmented full context exact equality')
                    meter.emit('replay_complete',mode=mode,contexts=path)
            row['plan']=model_plan(p['slots'][s],method,winner);row['status']='model_plan_sealed';meter.emit('model_selection_sealed',plan=row['plan'])


def census(p):
    return dict(signals=[dict(source_slot=s,status='not_started') for s in (0,1)],allocations={},curves={},
        rows=[dict(source_slot=s,task_id=p['slots'][s]['task_id'],arm=m,status='not_started',candidates=[]) for m in METHODS for s in (0,1)])


def summary(result):
    means={m:float(np.mean([r['plan']['model_cost'] for r in result['rows'] if r['arm']==m])) for m in METHODS}
    differences={m:[next(r['plan']['model_cost'] for r in result['rows'] if r['arm']==METHODS[1] and r['source_slot']==s)-
        next(r['plan']['model_cost'] for r in result['rows'] if r['arm']==m and r['source_slot']==s) for s in (0,1)] for m in (METHODS[0],METHODS[2],METHODS[3])}
    return dict(status='complete_two_known_roots_model_only',means=means,primary_D_minus_U=means[METHODS[1]]-means[METHODS[0]],
        secondary_D_minus_S=means[METHODS[1]]-means[METHODS[2]],secondary_D_minus_N=means[METHODS[1]]-means[METHODS[3]],root_differences=differences)


def run(prepared,prepare_sha,output):
    limits=fidelity.Limits(free=lambda:shutil.disk_usage(Path(output).parent).free)
    p,tasks,banks=verify(prepared,prepare_sha);limits.check();store=Store(output)
    meter=Meter(store.event,limits);result=census(p);errors=[];model=None;before=after=host=None;source_ok=False;complete=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha))
        store.put('banks.json.gz',banks);store.event(dict(kind='panel_census',rows=deepcopy(result['rows']),signals=deepcopy(result['signals'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device));limits.check()
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'frozen original model hash')
        meter.sync=lambda:torch.cuda.synchronize(device)
        with sdpa_kernel(SDPBackend.MATH):panel(p,tasks,banks,[Native(model,device,t) for t in tasks],meter,result)
        require(meter.counts()=={k:dict(attempted=v,completed=v) for k,v in MAXIMA.items()},'complete actual four-strategy resource vector')
        result['paired_summary']=summary(result)
    except BaseException as e:
        errors.append(enc.error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(enc.error('error_journal',e))
    finally:
        for row in result['rows']+result['signals']:
            if row['status']=='attempted':row['status']='failed_partial'
            elif errors and row['status']=='not_started':row['status']='blocked_not_dispatched'
            for c in row.get('candidates',[]):
                if c['status']=='attempted':c['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model);require(before==after,'final model unchanged')
            except BaseException as e:errors.append(enc.error('final_model',e))
        try:verify(prepared,prepare_sha);source_ok=True
        except BaseException as e:errors.append(enc.error('final_source',e))
        try:limits.check()
        except BaseException as e:errors.append(enc.error('final_limits',e))
        try:store.put('panel.json.gz',result,final=True)
        except BaseException as e:errors.append(enc.error('partial_panel',e))
        try:store.close()
        except BaseException as e:errors.append(enc.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in result['rows'])
        hashes={}
        try:
            for f in store.output.iterdir():hashes[f.name]=sha(f)
        except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        try:limits.check()
        except BaseException as e:errors.append(enc.error('artifact_final_limits',e));complete=False
        report=dict(schema=NAME+'.report',status='four_policy_commands_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha,artifact_sha256=hashes,
            errors=errors,first_error=errors[0] if errors else None,resources=meter.counts(),
            scope_resources={g:{str(s):meter.counts(g,s) for s in (0,1)} for g in ('shared',)+METHODS},
            actual_backward_observation='raw leaf tensor gradient hook callbacks; NOT kernel count; engine internals unknown',
            functional_jacobian_api_calls=0,jvp_api_calls=0,new_visual_entries=len(meter.forbidden_visual),new_visual_attempts=meter.forbidden_visual,
            new_physics=0,commands_released=complete,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            limits=limits.receipt(),protocol=protocol(),host=host,paired_summary=result.get('paired_summary',dict(status='not_available')),**FLAGS)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(enc.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial preserved, no retry: '+repr(errors));return report


def saved_check(p,tasks,banks,result,events):
    """Independent saved reconstruction, including real module-input histories."""
    require([(r['arm'],r['source_slot']) for r in result['rows']]==[(m,s) for m in METHODS for s in (0,1)],'all8 fixed slots')
    attempts=[e['entry'] for e in events if e['kind']=='resource_attempt'];returns=[e['entry'] for e in events if e['kind']=='resource_return']
    require([e['index'] for e in attempts]==list(range(len(attempts))) and len(returns)==len(attempts),'complete ordered ledger census')
    returned={e['index']:e for e in returns};require(len(returned)==len(returns),'unique returns')
    for a in attempts:require(returned[a['index']]==dict(a,completed=True),'exact attempt/return identity')
    def counts(group=None,s=None):
        es=[e for e in attempts if (group is None or e['scope'][0]==group) and (s is None or e['scope'][1]==s)]
        return {r:sum(e['resource']==r for e in es) for r in MAXIMA}
    require(counts()==MAXIMA,'520actual/2J384directions2SVD full vector')
    expected_inputs=[];expected_outputs=[];actual_inputs=[];expected_modules=[];semantic=[];module_arrays={}
    def expect(kind,scope,**fields):semantic.append(dict(kind=kind,scope=scope,**deepcopy(fields)))
    def pathcheck(s,raw,path,group):
        check_path(tasks[s],raw,path);before=tasks[s]['root_context']
        for k,after in enumerate(path):
            expected_inputs.append(([group,s],moving.incoming(tasks[s],before,np.asarray(raw,np.float32)[5*k:5*k+5])))
            expected_outputs.append(([group,s],after['latent_history'][-1]));before=after
    for s,entry in enumerate(result['signals']):
        require(entry['source_slot']==s and entry['status']=='complete','both shared signals')
        j=unblob(entry['jacobian']);require(signal(tasks[s],entry['terminal'],j)==entry['signal'],'original Q SVD signal recomputed')
        require(entry['terminal']==entry['nominal_contexts'][-1]['latent_history'][-1],'fresh AD nominal')
        pathcheck(s,banks[s][0]['commands'],entry['nominal_contexts'],'shared');pathcheck(s,banks[s][0]['commands'],entry['nominal_contexts'],'shared')
        expect('nominal_request',['shared',s],candidate=banks[s][0])
        expect('nominal_complete',['shared',s],contexts=entry['nominal_contexts'])
        expect('jacobian_complete',['shared',s],terminal=entry['terminal'],jacobian=entry['jacobian'])
        expect('signal_complete',['shared',s],signal=entry['signal'])
        require(all(counts('shared',s)[r]==4 for r in NEURAL) and all(counts('shared',s)[r]==192 for r in ('grad_api','vjp_directions','raw_backward_hook')) and counts('shared',s)['jacobian']==counts('shared',s)['svd']==1,'per-root shared vector')
        jr=[e for e in events if e['kind']=='jacobian_row' and e['scope']==['shared',s]]
        require([e['row'] for e in jr]==list(range(192)) and np.array_equal(np.stack([unblob(e['array']) for e in jr]),j),'all192 original zero-logit derivative rows')
        br=[e for e in events if e['kind']=='raw_backward' and e['scope']==['shared',s]]
        require(len(br)==192 and np.array_equal(np.stack([unblob(e['array']).reshape(20) for e in br]),j),'actual raw leaf backward callback arrays, zero tanh chain')
    weights=[e['signal']['w'] for e in result['signals']]
    for method in METHODS:
        blocks=allocate(method) if method in (METHODS[0],METHODS[3]) else allocate(method,weights)
        require(result['allocations'][method]==blocks,'predeclared allocation')
        expect('allocation_frozen',[method,None],blocks=blocks,schedule=schedule(blocks))
        selected_rows=[r for r in result['rows'] if r['arm']==method];progress=[0,0];curves=[]
        for block,s in enumerate(schedule(blocks),1):
            row=selected_rows[s];require(row['status']=='model_plan_sealed' and row['task_id']==p['slots'][s]['task_id'],'full stable branch')
            for index in range(progress[s]*3,progress[s]*3+3):
                c=row['candidates'][index];require({k:c[k] for k in banks[s][index]}==banks[s][index] and c['status']=='complete','fixed entire same-root candidate prefix')
                expect('candidate_request',[method,s],block=block,candidate=dict(banks[s][index],status='attempted'))
                expect('candidate_complete',[method,s],candidate=c)
                pathcheck(s,c['commands'],c['contexts'],method);require(c['cost']==cost(tasks[s],c['contexts']),'all sameQ candidate costs')
            progress[s]+=1
            if block in (10,20):curves.append([min(c['cost'] for c in selected_rows[q]['candidates'][:progress[q]*3]) for q in (0,1)])
        require([x['best_costs'] for x in result['curves'][method]]==curves and [x['block'] for x in result['curves'][method]]==[10,20],'paid boundaries only')
        for s,row in enumerate(selected_rows):
            require(len(row['candidates'])==3*blocks[s],'complete candidate denominator')
            w=min(row['candidates'],key=lambda c:(c['cost'],c['index']))
            require(row['winner_index']==w['index'] and row['plan']==model_plan(p['slots'][s],method,w),'stable argmin and selection seal')
            expect('winner_frozen',[method,s],index=w['index'],cost=w['cost'],commands=w['commands'])
            require(set(row['replays'])=={'whole_replay','segmented_replay'},'two independent full replays')
            for mode in ('whole_replay','segmented_replay'):
                pathcheck(s,w['commands'],row['replays'][mode],method);require(row['replays'][mode]==w['contexts'],'all full replay contexts equal')
                expect('replay_request',[method,s],mode=mode,commands=w['commands'])
                expect('replay_complete',[method,s],mode=mode,contexts=row['replays'][mode])
            expect('model_selection_sealed',[method,s],plan=row['plan'])
            require(all(counts(method,s)[r]==6*blocks[s]+4 for r in NEURAL),'per-root actual allocation+replay')
        require(all(counts(method)[r]==128 for r in NEURAL) and all(counts(method)[r]==0 for r in MAXIMA if r not in NEURAL),'exclusive128, no signal inside search/N')
    for e in events:
        if e['kind']=='context_input':actual_inputs.append((e['scope'],e['context']))
        if e['kind'] in ('module_input','module_output'):
            def decode_arrays(x):return [decode_arrays(v) for v in x] if isinstance(x,list) else unblob(x)
            a=decode_arrays(e['arrays']);resource=e['resource'];entry=attempts[e['entry']]
            require(entry['resource']==resource and entry['scope']==e['scope'],'actual module trace entry identity')
            expected_modules.append((e['entry'],e['kind']))
            module_arrays[e['entry'],e['kind']]=a
            if e['kind']=='module_input' and resource=='action_encoder':
                # Context input immediately precedes each actual AE request.
                n=sum(1 for x in expected_modules if x[1]=='module_input' and attempts[x[0]]['resource']=='action_encoder')-1
                require(np.array_equal(a[0][0],actual_inputs[n][1]['action_history']),'actual AE three tokens exact')
            if e['kind']=='module_input' and resource=='predictor':
                n=sum(1 for x in expected_modules if x[1]=='module_input' and attempts[x[0]]['resource']=='predictor')-1
                require(np.array_equal(a[0][0],actual_inputs[n][1]['latent_history']),'actual full predictor latent history')
    require(actual_inputs==expected_inputs,'ALL520 real nonzero context paths including fresh joint AD/replays')
    outputs=[e for e in events if e['kind']=='predict_output']
    require(len(outputs)==520 and [(e['scope'],unblob(e['array'])[0,-1].astype(float).tolist()) for e in outputs]==expected_outputs,
        'every actual native successor binds candidate/nominal/AD/replay terminal, including last packet')
    semantic_kinds={e['kind'] for e in semantic}
    require([e for e in events if e['kind'] in semantic_kinds]==semantic,'all frozen allocations/candidate requests/winner-before-replays/command seals in exact order')
    wanted={(e['index'],k) for e in attempts if e['resource'] in ('action_encoder','predictor','pred_proj') for k in ('module_input','module_output')}
    require(len(expected_modules)==len(wanted) and set(expected_modules)==wanted,'all native module arrays preserved')
    neural=[e for e in attempts if e['resource'] in NEURAL]
    require([e['resource'] for e in neural]==['action_encoder','predict','predictor','pred_proj']*520,'actual AE/API/predictor/projection request order')
    for request in range(520):
        ae,api,pred,proj=neural[request*4:request*4+4]
        require(ae['scope']==api['scope']==pred['scope']==proj['scope']==outputs[request]['scope'],'native request scope')
        get=lambda entry,kind:module_arrays[entry['index'],'module_'+kind]
        aeout=get(ae,'output');predin=get(pred,'input');predout=get(pred,'output');projin=get(proj,'input');projout=get(proj,'output')
        native=unblob(outputs[request]['array'])
        require(native.dtype==aeout.dtype==predout.dtype==projout.dtype==np.float32 and native.shape==aeout.shape==predout.shape==(1,3,192), 'full native precision and batch/context shapes')
        require(len(predin)==2 and len(projin)==1 and np.array_equal(predin[1],aeout) and np.array_equal(projin[0],predout.reshape(3,192)) and np.array_equal(projout,native.reshape(3,192)), 'actual AE -> predictor -> pred_proj -> API full array chain')
    require(summary(result)==result['paired_summary'],'fixed complete paired model-only summaries')
    return [r['plan'] for r in result['rows']]


def consume(output,report_sha):
    output=Path(output).absolute();require(sha(output/'report.json')==report_sha,'external actual scheduler report SHA')
    r=read(output/'report.json');require(r['schema']==NAME+'.report' and r['status']=='four_policy_commands_sealed_not_admitted' and r['errors']==[] and r['first_error'] is None and r['commands_released'] is True and r['model_unchanged'] is True and r['source_verified'] is True,'no partial release')
    p,tasks,banks=verify(r['prepared'],r['prepare_sha256']);require(r['protocol']==protocol() and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'same model/contract')
    names={'binding.json','banks.json.gz','panel.json.gz','events.jsonl.gz'}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'actual artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and read(output/'banks.json.gz')==banks,'prepare/bank freeze')
    plain,eof=blind.inflate((output/'events.jsonl.gz').read_bytes(),128*1024**2);require(eof and plain.endswith(b'\n'),'complete bounded journal')
    events=[blind.decode(line) for line in plain.splitlines()];result=read(output/'panel.json.gz',32*1024**2)
    plans=saved_check(p,tasks,banks,result,events)
    require(r['resources']=={k:dict(attempted=v,completed=v) for k,v in MAXIMA.items()} and r['paired_summary']==result['paired_summary'],'actual resource+summary receipts')
    entries=[e['entry'] for e in events if e['kind']=='resource_attempt']
    scopes={g:{str(s):{k:dict(attempted=sum(e['resource']==k and e['scope']==[g,s] for e in entries),
        completed=sum(e['resource']==k and e['scope']==[g,s] for e in entries)) for k in MAXIMA} for s in (0,1)} for g in ('shared',)+METHODS}
    require(r['scope_resources']==scopes,'reported method/root counts reconstructed from actual ledger')
    require(all(r[k] is False for k in FLAGS) and r['new_visual_entries']==0 and r['new_visual_attempts']==[] and r['new_physics']==0,'no admission/visual/physics claims')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final closure TOCTOU')
    require(sha(output/'report.json')==report_sha,'final report TOCTOU')
    return dict(schema=NAME+'.consume',status='four_policy_commands_verified_not_admitted',errors=[],plans=[dict(x,whole_run_report_sha256=report_sha) for x in plans],**FLAGS)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('handoff-sha','anchor-report-sha','encoder-report-sha','prepared','prepare-sha','output','report-sha'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':value=prepare(a.output,handoff_sha=a.handoff_sha,anchor_report_sha=a.anchor_report_sha,encoder_report_sha=a.encoder_report_sha)
    elif a.command=='verify':value=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)[0]['slots']))
    elif a.command=='consume':value=consume(a.output,a.report_sha)
    else:value=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(value).decode(),end='')


if __name__=='__main__':main()
