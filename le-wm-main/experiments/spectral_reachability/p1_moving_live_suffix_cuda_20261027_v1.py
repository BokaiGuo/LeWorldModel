"""Narrow same-process moving live suffix core; MAIN alone runs CUDA/physics.

No loader, environment, image encoder, selector fallback or CLI execution job.
The caller owns physical causality, episode admission and the durable journal.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
from pathlib import Path
import re
import shutil
import threading
import numpy as np
from experiments.spectral_reachability import moving_history_observed_suffix_cuda_20261027_v1 as frozen

NAME = 'p1_moving_live_suffix_cuda_20261027_v1'
ROOT = frozen.ROOT
CAP, FLOOR, FINAL = 8*1024**2, 200*1024**2, 256*1024
require, canonical, sha = frozen.require, frozen.canonical, frozen.sha
core, planner = frozen.core, frozen.planner
MODEL_SHA = frozen.observed.enc.MODEL_TENSOR_SHA
_LOCK = threading.Lock()  # AD instrumentation changes process-wide torch APIs.


def digest(value):
    """JSON binding, NOT an NPY/PNG file SHA; canonical includes final newline."""
    return hashlib.sha256(canonical(value)).hexdigest()


def fields(value, names, label):
    require(type(value) is dict and set(value) == set(names.split()), label+' exact fields')


def array(value, shape):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all()
            and np.array_equal(a, a.astype(np.float32)), 'literal native float32 '+str(shape))
    return a.astype(np.float32)


def source_bindings():
    """Read-only static local source closure; no task/material/model loading."""
    files = set(frozen.blind.source_files()) | {
        Path(__file__).resolve(), Path(frozen.__file__).resolve(),
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')}
    pending = list(files)
    seen = set()
    while pending:
        p = Path(pending.pop()).resolve()
        if p in seen or not p.is_relative_to(ROOT/'experiments') or p.suffix != '.py':
            continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [v.name for v in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]+[node.module+'.'+v.name for v in node.names]
            for name in names:
                if name.startswith('experiments.'):
                    child = ROOT/Path(*name.split('.')).with_suffix('.py')
                    if child.is_file() and child not in files:
                        files.add(child); pending.append(child)
    return {str(Path(p).resolve()): sha(p) for p in sorted(files)}


def modelbinding():
    """Pure source/constants helper; driver seals this before running episodes."""
    return dict(tensor_sha256=MODEL_SHA, model_files=deepcopy(frozen.observed.enc.MODELS),
        action_mean=frozen.d2p.ACTION_MEAN.astype(float).tolist(),
        action_std=frozen.d2p.ACTION_STD.astype(float).tolist(),
        training_provenance='unknown', source_closure=source_bindings())


def contract():
    return dict(schema=NAME+'.contract', modes=['I','R'], stages=list(frozen.STAGES),
        per_compile=frozen.budget(), predict_API=5, predictor_module=5,
        AD_API=dict(jacobian=1, grad=1, backward=0), mathematical_VJP_directions=192,
        AD_forward_included_in_predict=1, damping=.002, packet_logit_cap=float(.75*np.sqrt(5)),
        dispatch='new_suffix_unconditionally', new_encoder=0, new_projector=0, physics=0,
        CPU_neural=False, retries=0, output_cap_bytes=CAP, final_reserve_bytes=FINAL,
        disk_floor_bytes=FLOOR, O_compiles=False, P1_admission=False, P4_admission=False,
        independent_confirmation=False, goal_Q_normalizer_unchanged=True)


def validate_packet(packet):
    """Strict deployment whitelist. A self-seal does not prove physical origin."""
    fields(packet, 'schema mode context3 actualrawhistory10 oldrawsuffix5 goal identityQ modelbinding episode_observation_seal', 'packet')
    p = deepcopy(packet)
    require(p['schema'] == NAME+'.packet' and p['mode'] in ('I','R'), 'own I/R schema; O never compiles')
    c = p['context3']; core.context(c)
    fields(c, 'layout context_length latent_dim action_dim latent_history action_history', 'context3')
    array(c['latent_history'], (3,192)); array(c['action_history'], (2,10))
    raw = array(p['actualrawhistory10'], (10,2)); old = array(p['oldrawsuffix5'], (5,2))
    array(p['goal'], (192,))
    require((np.abs(raw)<=1).all() and (np.abs(old)<1).all(), 'legal literal raw history/suffix, no clipping')
    require(np.array_equal(raw[:5], planner.fixed_raw()[5:]), 'real moving warmup final5, never zero loader')
    cap = float(.75*np.sqrt(5))
    require(np.linalg.norm(np.arctanh(old.astype(float))) <= cap+1e-6, 'old suffix original logit ball')
    fields(p['identityQ'], 'kind dimension', 'Q')
    require(p['identityQ'] == dict(kind='identity', dimension=192) and type(p['identityQ']['dimension']) is int, 'unchanged identity192 Q')
    mb = p['modelbinding']
    fields(mb, 'tensor_sha256 model_files action_mean action_std training_provenance source_closure', 'modelbinding')
    require(mb['tensor_sha256'] == MODEL_SHA and mb['model_files'] == frozen.observed.enc.MODELS
            and mb['action_mean'] == frozen.d2p.ACTION_MEAN.astype(float).tolist()
            and mb['action_std'] == frozen.d2p.ACTION_STD.astype(float).tolist()
            and mb['training_provenance'] == 'unknown', 'original model/public normalizer, provenance unknown')
    require(type(mb['source_closure']) is dict and mb['source_closure'], 'nonempty externally sealed source closure')
    for path,h in mb['source_closure'].items():
        require(type(path) is str and Path(path).is_absolute(), 'absolute source paths')
        core.hash_value(h)
    require(np.array_equal(c['action_history'], core.normalized_tokens(mb, raw)), 'exact two real normalized packets, no /100/tanh/zero fill')
    s = p['episode_observation_seal']
    fields(s, 'episode_id source_slot prefix_arm mode parent_task_id task_export_sha256 parent_selection_sha256 step remaining_public_steps context_kind context_sha256 rawhistory_sha256 oldsuffix_sha256 goal_sha256 Q_sha256 modelbinding_sha256 rgb_file_sha256 rgb_pixels_sha256 encoded_npy_sha256 encoded_latent_sha256 encoder_receipt_sha256 imagined_context_sha256 observed_event_index encoded_event_index seal_sha256', 'episode observation seal')
    require(type(s['episode_id']) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', s['episode_id']), 'bounded literal episode id')
    require(type(s['source_slot']) is int and s['source_slot'] in (0,1) and s['prefix_arm'] in planner.ARMS, 'fixed two roots and GN/D2P prefixes')
    require(s['task_export_sha256'] == planner.EXPORT_SHAS[s['source_slot']], 'original moving export identity')
    require(s['mode'] == p['mode'] and type(s['step']) is int and s['step']==5
            and type(s['remaining_public_steps']) is int and s['remaining_public_steps']==5, 'own mode/step5/remaining5')
    require(s['context_kind'] == ('observed' if p['mode']=='R' else 'imagined'), 'explicit observed versus imagined context')
    for key in ('parent_task_id','task_export_sha256','parent_selection_sha256','context_sha256','rawhistory_sha256',
                'oldsuffix_sha256','goal_sha256','Q_sha256','modelbinding_sha256','rgb_file_sha256','rgb_pixels_sha256',
                'encoded_npy_sha256','encoded_latent_sha256','encoder_receipt_sha256','seal_sha256'):
        core.hash_value(s[key])
    for key,value in (('context',c),('rawhistory',p['actualrawhistory10']),('oldsuffix',p['oldrawsuffix5']),
                      ('goal',p['goal']),('Q',p['identityQ']),('modelbinding',mb)):
        require(s[key+'_sha256'] == digest(value), 'episode literal binding '+key)
    if p['mode']=='R':
        require(s['imagined_context_sha256'] is None and s['encoded_latent_sha256']==digest(c['latent_history'][-1]), 'R consumes own newly encoded latent')
    else:
        require(s['imagined_context_sha256']==digest(c), 'I consumes original selected imagined context, not actual latent')
    require(type(s['observed_event_index']) is int and type(s['encoded_event_index']) is int
            and 0<=s['observed_event_index']<s['encoded_event_index'], 'driver observation-before-encode event binding')
    require(s['seal_sha256']==digest({k:v for k,v in s.items() if k!='seal_sha256'}), 'literal episode observation seal')
    return p


def branch_input(p):
    """Internal bridge only. Never invokes an old task loader/build/verify."""
    mb=p['modelbinding']
    b=dict(observed_context=deepcopy(p['context3']), observed_raw_history=deepcopy(p['actualrawhistory10']),
        prefix_commands=deepcopy(p['actualrawhistory10'][5:]), old_suffix_commands=deepcopy(p['oldrawsuffix5']),
        goal=deepcopy(p['goal']), q=np.eye(192).tolist(), action_mean=mb['action_mean'], action_std=mb['action_std'],
        packet_logit_cap=float(.75*np.sqrt(5)), branch_id=digest(p))
    return b


class SuffixFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt=receipt
        super().__init__('suffix failed closed: '+str(receipt['first_error']))


class Sink:
    """Per-compile bounded durable bridge; does not finalize shared journal."""
    def __init__(self, store, check_limits):
        self.store,self.check_limits=store,check_limits
        self.used=0; self.files={}; self.broken=False
    def check(self, n=0, final=False):
        self.check_limits()
        require(self.used+n <= CAP-(0 if final else FINAL), 'suffix8MiB cap and final reserve')
        require(shutil.disk_usage(self.store.output).free >= FLOOR+n, 'actual200MiB disk floor')
    def _write(self, name, raw, final=False):
        require(not self.broken and Path(name).name==name and name not in self.files, 'exclusive flat suffix files, no retry')
        self.check(len(raw),final); self.used+=len(raw)
        try:self.store._write(name,raw)
        except BaseException:self.broken=True;raise
        self.files[name]=raw
    def event(self, value):
        raw=canonical(value); self.check(len(raw)); self.used+=len(raw)
        require(not self.broken,'no journal retry')
        try:self.store.event(value)
        except BaseException:self.broken=True;raise
    def load(self, entry):
        raw=self.files[entry['file']]
        require(hashlib.sha256(raw).hexdigest()==entry['sha256'],'saved array bytes binding')
        stream=BytesIO(raw);a=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(raw) and str(a.dtype)==entry['dtype'] and list(a.shape)==entry['shape'], 'single exact NPY')
        return a


def _execute(packet, *, store, check_limits, runner, check_model, check_source):
    """Non-neural orchestration; private callbacks enable RAM failure tests."""
    sink=Sink(store,check_limits)
    r=dict(schema=NAME+'.receipt',status='failed_partial',commands_released=False,
        contract=contract(),errors=[],first_error=None,row=None,model_hash_before=None,model_hash_after=None,
        source_verified=False, P1_admission=False, P4_admission=False)
    def error(phase, exc):
        r['errors'].append(frozen.observed.enc.error(phase,exc));r['first_error']=r['errors'][0]
    try:
        sink.check();p=validate_packet(packet);b=branch_input(p)
        r.update(packet_sha256=digest(p),episode_observation_seal=deepcopy(p['episode_observation_seal']))
        check_source(p['modelbinding']['source_closure'])
        r['model_hash_before']=check_model();require(r['model_hash_before']==MODEL_SHA,'original frozen model before')
        sink._write('packet.json',canonical(p));sink.event(dict(kind='suffix_started',packet_sha256=digest(p)))
        row=dict(index=0,branch_id=b['branch_id']);r['row']=row
        runner(b,sink,row)
        frozen.check_result(b,row);frozen.blind.exact_counts(row['resources'],frozen.budget())
        frozen.blind.exact_counts(row['AD_API_resources'],dict(grad=1,jacobian=1))
        frozen.blind.exact_counts(row['predict_API_resources'],dict(predict=5))
        require([e['stage'] for e in row['predict_API_ledger']]==list(frozen.STAGES)
                and all(e['completed'] for e in row['predict_API_ledger']), 'five actual model.predict API returns')
        frozen.saved_arrays(b,row,sink.load)
    except BaseException as exc:error('execution',exc)
    finally:
        if r['model_hash_before'] is not None:
            try:
                r['model_hash_after']=check_model()
                require(r['model_hash_before']==r['model_hash_after'],'unchanged model after')
            except BaseException as exc:error('final_model',exc)
        if r.get('packet_sha256'):
            try:check_source(packet['modelbinding']['source_closure']);r['source_verified']=True
            except BaseException as exc:error('final_source',exc)
    if not r['errors']:
        try:
            sink.check()
            r.update(status='suffix_sealed_not_admitted',commands_released=True,
                dispatch_raw5=deepcopy(r['row']['new_suffix_commands']), dispatch='new_suffix_unconditionally',
                model_unchanged=True, artifact_sha256={n:hashlib.sha256(v).hexdigest() for n,v in sink.files.items()})
            r['release_sha256']=digest(r)
            sink.event(dict(kind='suffix_release_seal',release_sha256=r['release_sha256'],packet_sha256=r['packet_sha256']))
            sink._write('receipt.json.gz',gzip.compress(canonical(r),mtime=0),final=True)
            return r
        except BaseException as exc:error('release',exc)
    r.update(status='failed_partial',commands_released=False)
    if r['row'] is not None and r['row'].get('status')=='attempted':
        r['row']['status']='failed_partial'
    r['artifact_sha256']={n:hashlib.sha256(v).hexdigest() for n,v in sink.files.items()}
    for key in ('dispatch_raw5','dispatch','release_sha256'):r.pop(key,None)
    try:
        # A broken sink is never retried; the caller receives the complete partial.
        if not sink.broken:
            sink._write('failure.json.gz',gzip.compress(canonical(r),mtime=0),final=True)
    except BaseException as exc:error('failure_persist',exc)
    raise SuffixFailure(r)


class PredictAPI:
    """Count model.predict wrapper entry, separately from predictor module hooks."""
    def __init__(self, model, meter, sink, sync):
        self.model,self.meter,self.sink,self.sync=model,meter,sink,sync;self.entries=[]
    def __enter__(self):
        self.had='predict' in self.model.__dict__;self.original_local=self.model.__dict__.get('predict')
        self.original=self.model.predict
        def call(*args,**kwargs):
            require(len(self.entries)<5 and all(e['completed'] for e in self.entries)
                and self.meter.stage==frozen.STAGES[len(self.entries)], 'five nonretry model.predict entries')
            e=dict(index=len(self.entries),stage=self.meter.stage,completed=False);self.entries.append(e)
            self.sink.event(dict(kind='predict_API_attempt',entry=deepcopy(e)))
            value=self.original(*args,**kwargs);self.sync();e['completed']=True
            self.sink.event(dict(kind='predict_API_return',entry=deepcopy(e)));return value
        self.model.predict=call;return self
    def __exit__(self, typ, exc, tb):
        try:
            if self.had:self.model.predict=self.original_local
            else:delattr(self.model,'predict')
        except BaseException as secondary:
            if exc is None:raise
            exc.add_note('predict API restore: '+repr(secondary))
    def counts(self):
        return dict(predict=dict(attempted=len(self.entries),completed=sum(e['completed'] for e in self.entries)))


def verify_suffix_seal(request, response):
    """Return raw5 only for the exact successful in-process release receipt.

    This is a structural handoff gate, not independent file/physical admission.
    Driver still checks own episode, journal causality and durable artifact SHA.
    """
    p=validate_packet(request);r=deepcopy(response)
    require(r['schema']==NAME+'.receipt' and r['status']=='suffix_sealed_not_admitted'
        and r['commands_released'] is True and r['errors']==[] and r['first_error'] is None
        and r['contract']==contract() and r['P1_admission'] is False and r['P4_admission'] is False,
        'complete own release only')
    h=r.pop('release_sha256');require(h==digest(r),'literal returned release SHA')
    require(r['packet_sha256']==digest(p) and r['episode_observation_seal']==p['episode_observation_seal']
        and r['model_hash_before']==r['model_hash_after']==MODEL_SHA and r['model_unchanged'] is True
        and r['source_verified'] is True,'own episode/packet and unchanged model/source')
    frozen.check_result(branch_input(p),r['row'])
    frozen.blind.exact_counts(r['row']['resources'],frozen.budget())
    frozen.blind.exact_counts(r['row']['AD_API_resources'],dict(grad=1,jacobian=1))
    frozen.blind.exact_counts(r['row']['predict_API_resources'],dict(predict=5))
    require(r['dispatch']=='new_suffix_unconditionally' and r['dispatch_raw5']==r['row']['new_suffix_commands'], 'new ridge fixed dispatch only')
    return deepcopy(r['dispatch_raw5'])


def compile_suffix(request, artifact_sink, *, model, device, check_limits):
    """MAIN host CUDA only; explicit already-loaded model, no disk/model loader.

    artifact_sink.output must be an existing dedicated compilation directory;
    artifact_sink._write/event must be exclusive/durable fsync operations.
    Shared driver journal is caller-owned. On failure NEVER dispatch commands
    from disk/partial receipt; catch SuffixFailure and retain .receipt.
    """
    require(_LOCK.acquire(blocking=False),'nonreentrant process-wide AD instrumentation')
    try:
        require(getattr(device,'type',None)=='cuda','explicit HOST CUDA; no CPU fallback')
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        require(torch.cuda.is_available() and not torch.is_inference_mode_enabled(), 'CUDA available, autograd enabled scope required')
        old_matmul=torch.backends.cuda.matmul.allow_tf32;old_cudnn=torch.backends.cudnn.allow_tf32
        def check_model():
            check_limits();frozen.observed.enc.frozen(model)
            require(all(v.device==device for v in list(model.parameters())+list(model.buffers())), 'same explicit CUDA device')
            return frozen.observed.enc.tensor_hash(model)
        def check_source(expected):require(expected==source_bindings(),'complete frozen source closure before/after')
        def runner(b,sink,row):
            sync=lambda:torch.cuda.synchronize(device)
            meter=frozen.Meter(sink.event,sync);meter.require_native=True
            modules=frozen.Modules(model,sink,'branch_00',meter,sync)
            ad=frozen.SuffixAD(torch,frozen.blind.ARMS[1],sink.event,sync);ad.stage=lambda:meter.stage
            api=PredictAPI(model,meter,sink,sync)
            backend=frozen.d2p.TorchBackend(model,device)
            try:
                with ad,modules,api:frozen.evaluate(b,backend,meter,sink,row)
            finally:
                row.update(resources={**modules.counts(),**meter.counts()},AD_API_resources=ad.counts(),
                    predict_API_resources=api.counts(),module_ledger=deepcopy(modules.entries),
                    math_ledger=deepcopy(meter.entries),AD_API_ledger=deepcopy(ad.events),predict_API_ledger=deepcopy(api.entries))
        try:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
                result=_execute(request,store=artifact_sink,check_limits=check_limits,runner=runner,check_model=check_model,check_source=check_source)
                verify_suffix_seal(request,result)
                return result
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_matmul;torch.backends.cudnn.allow_tf32=old_cudnn
    finally:_LOCK.release()
