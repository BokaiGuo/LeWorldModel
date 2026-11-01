"""External-SHA approved latent exports -> two model-only plans, MAIN CUDA only.

Never call old material/verify, encode, sourcepack, certificate or witness loaders.
Unequal fixed algorithm budgets; no retry, physics, training or P1 admission.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sysconfig
import zlib
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as encode
from experiments.spectral_reachability import p1_line_search_gn_cuda_20261027_v1 as gn
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v2 as d2p

NAME='p1_constructive_blind_planners_cuda_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
ARMS=('line_search_GN','full10_conditional_D2P')
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,256*1024
ENCODE_SOURCE_SHA='c3b071de1ad0b322e1fff77a716550f42014f0303a6b5fa09271fd4e55609b4a'
require=encode.require
canonical,sha=encode.canonical,encode.sha


def decode(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key'); out[k]=v
        return out
    value=json.loads(raw,object_pairs_hook=pairs); canonical(value); return value


def read(path,limit=8*1024**2):
    path=Path(path); require(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'bounded regular JSON input')
    raw=path.read_bytes()
    if path.name.endswith('.gz'):
        raw,eof=inflate(raw,limit); require(eof,'complete gzip input required')
    require(len(raw)<=limit,'bounded expanded JSON'); return decode(raw)


def inflate(raw,limit=32*1024**2):
    d=zlib.decompressobj(31); plain=d.decompress(raw,limit+1)
    require(len(plain)<=limit and not d.unused_data and not d.unconsumed_tail,'single bounded gzip, no trailing stream')
    return plain,d.eof


def interface(value):
    # The FINAL encoder loader owns its export/firewall schema. Consume only
    # the four public algorithm fields, never native/source identity metadata.
    result={k:deepcopy(value[k]) for k in ('gn_task','d2p_packet','q','task_id')}
    t=result['gn_task']; encode.core.validate_task(t)
    require(t['environment']=='swm/PushT-v1' and np.array_equal(t['projection'],np.eye(192)) and
        t['packet_logit_cap']==float(.75*np.sqrt(5)) and not np.asarray(t['root_raw_action_history']).any(),'identity Q, original cap, actual zero warmup')
    require(np.array_equal(np.asarray(t['action_mean'],np.float32),d2p.ACTION_MEAN) and
        np.array_equal(np.asarray(t['action_std'],np.float32),d2p.ACTION_STD),'TorchBackend hardcoded public normalizer must match task')
    require(t['model_sha256']==encode.MODELS[str(encode.POLICY/'weights.pt')] and
        t['model_config_sha256']==encode.MODELS[str(encode.POLICY/'config.json')],'same original checkpoint files')
    require(result['task_id']==encode.core.task_id(t) and
        result['d2p_packet']==dict(root_context=t['root_context'],goal=t['goal']) and result['q']==t['projection'],'both arms exact same task/context/goal/Q')
    for a in (t['goal'],t['root_context']['latent_history']):
        a=np.asarray(a); require(np.array_equal(a,a.astype(np.float32)),'native float32 latent export, not substituted precision')
    return result


def validate_export_envelope(export):
    # Finalized saved envelope is strict; algorithm arrays STILL come only from
    # encode.load_task/task_interfaces. source_slot is checked, never used as a
    # native-root lookup or passed to either planner.
    encode.core.fields(export,'schema task task_id task_contract blind_source_sha256 native_root_sha256 source_slot with_target blind_sha256 history_semantics source_admission global_freshness P1_admission new_independent_confirmation','approved final task export')
    c=export['task_contract']
    encode.core.fields(c,'schema prepare_sha256 whitelist_sha with_target source_slot blind_source_sha256 observed_steps raw_between_steps normalization model_bindings metric','approved final task contract')
    require(type(export['source_slot']) is int and 0<=export['source_slot']<4 and
        type(c['source_slot']) is int and c['source_slot']==export['source_slot'],'strict source_slot envelope identity')
    require(export['native_root_sha256'] is None and export['with_target'] is False and c['with_target'] is False and
        export['P1_admission'] is False and export['new_independent_confirmation'] is False and
        c['observed_steps']==[0,5,10] and type(c['raw_between_steps']) is int and c['raw_between_steps']==10 and
        c['metric']=='identity_native192_no_fit' and c['model_bindings']==encode.MODELS,'strict no-target/history/model contract, no native-root mapping')


def load_export(path,expected):
    encode.core.hash_value(expected); path=Path(path).absolute()
    require(path.name.startswith('task_') and path.name.endswith(('.json','.json.gz')),'explicit task export filename only')
    require(sha(path)==expected,'externally approved export SHA')
    validate_export_envelope(read(path,2*1024**2))
    result=interface(encode.load_task(path,expected)); require(sha(path)==expected,'export TOCTOU')
    return result


def budgets():
    return {ARMS[0]:dict(gn.budget()),ARMS[1]:dict(predict=66,action_encoder=66,jacobian=11,svd=12)}


def contract(n):
    return dict(arms=list(ARMS),roots=n,execution='fixed export order, GN then D2P; no replacement/retry/early exit',
        per_root_budget=budgets(),total_budget={a:{k:n*v for k,v in b.items()} for a,b in budgets().items()},
        AD_API_per_root={ARMS[0]:dict(grad=768,jacobian=0),ARMS[1]:dict(grad=11,jacobian=11)},
        mathematical_vjp_per_root={ARMS[0]:768,ARMS[1]:2112},
        ADforward_in_predict_per_root={ARMS[0]:8,ARMS[1]:11},
        GN=dict(updates=4,damping=.002,scales=list(gn.SCALES),initial_logits='zero2x10'),
        D2P=dict(candidates=10,conditional_suffix='fresh H1 per prefix',beta=1,damping=.002,
            selection='stage_cost+own_reach_value, stable bank-index ties',shared_derivatives=False),
        new_images=0,new_physics=0,training=False,new_Q=False,with_target=False,
        equal_budget_claim=False,P1_admission=False,independent_confirmation=False,
        metric='same approved identity192 task for both arms',visual_cost='inherited encoding is separate, not zero total-pipeline cost',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        no_cpu_neural_fallback=True,output_cap_bytes=CAP,reserve_bytes=FLOOR,
        source_access='only explicit approved latent task exports; no discovery/read of blind NPZ/sourcepack/certificate/witness',
        release='only complete final report + external SHA can release model-selected commands; no physics here')


def source_files():
    # Static local import closure: text/hash only, never call imported material.
    files=set(encode.source_files()+[Path(__file__),Path(gn.__file__),Path(d2p.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        ROOT/'tests/test_p1_line_search_gn_cuda_20261027_v1.py',ROOT/'tests/test_p3_goal_patch_consistent_compiler_20261027_v2.py'])
    todo=list(files); seen=set()
    while todo:
        p=Path(todo.pop()).resolve()
        if p in seen or not p.is_relative_to(ROOT/'experiments') or p.suffix!='.py': continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(node,ast.Import): names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments'):
                names=[node.module]+[node.module+'.'+x.name for x in node.names]
            for name in names:
                if not name.startswith('experiments.'): continue
                child=ROOT/Path(*name.split('.')).with_suffix('.py')
                if child.is_file() and child not in files: files.add(child); todo.append(child)
    site=Path(sysconfig.get_paths()['purelib'])
    files.update(site/x for x in ('torch/autograd/functional.py','torch/autograd/__init__.py','torch/_vmap_internals.py'))
    return sorted({p.resolve() for p in files})


def build(task_exports,task_shas):
    require(sha(Path(encode.__file__))==ENCODE_SOURCE_SHA,'MAIN approved final encoder interface source')
    require(len(task_exports)==len(task_shas) and 1<=len(task_exports)<=4,'explicit ordered1--4 approved exports')
    slots=[]
    for order,(path,h) in enumerate(zip(task_exports,task_shas,strict=True)):
        path=Path(path).absolute(); i=load_export(path,h)
        slots.append(dict(order=order,task_export=str(path),task_sha256=h,task_id=i['task_id']))
    for k in ('task_export','task_sha256','task_id'):
        require(len({s[k] for s in slots})==len(slots),'no duplicated export/source identity')
    norm=encode.native_metadata()  # model/config and public normalizer only, NO input discovery.
    for path,h in encode.MODELS.items(): require(sha(path)==h,'frozen local model file')
    for s in slots:
        t=load_export(s['task_export'],s['task_sha256'])['gn_task']
        require(t['action_mean']==norm['mean'] and t['action_std']==norm['std'],'bound public normalization')
    files=source_files(); bindings={str(p):sha(p) for p in files}
    bindings.update(encode.MODELS); bindings.update({s['task_export']:s['task_sha256'] for s in slots})
    return dict(schema=NAME+'.prepare',arguments=dict(task_exports=[s['task_export'] for s in slots],task_shas=list(task_shas)),
        slots=slots,contract=contract(len(slots)),policy=str(encode.POLICY),expected_model_hash=encode.MODEL_TENSOR_SHA,
        bindings_sha256=bindings,interface_source_sha256=sha(Path(encode.__file__)),global_freshness='unknown_calibration_only')


def prepare(output,**arguments):
    p=build(**arguments); raw=canonical(p); output=Path(output).absolute()
    require(len(raw)<256*1024 and not output.exists() and shutil.disk_usage(output.parent).free>=FLOOR+len(raw),'NEW small prepare +200MiB reserve')
    with output.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(output),prepare_sha256=sha(output),bytes=len(raw),contract=p['contract'])


def verify(prepared,expected):
    require(sha(prepared)==expected,'external prepare SHA'); p=read(prepared)
    require(p==build(**p['arguments']),'complete final interface/export/source/model reconstruction'); return p


class Store:
    def __init__(self,output):
        self.output=Path(output).absolute(); require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW8MiB +200MiB floor')
        self.output.mkdir(); self.comp=zlib.compressobj(6,zlib.DEFLATED,31); self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat nonsymlink output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'all bytes incl partial/cap/reserve')
        with (self.output/name).open('ab' if append else 'xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    def event(self,e):
        require(not self.closed and not self.broken,'no retry on failed/final journal')
        try: self._write('events.jsonl.gz',self.comp.compress(canonical(e))+self.comp.flush(zlib.Z_SYNC_FLUSH),append=True)
        except BaseException: self.broken=True; raise
    def put(self,name,value,final=False):
        raw=canonical(value); self._write(name,gzip.compress(raw,compresslevel=6,mtime=0) if name.endswith('.gz') else raw,final=final)
    def close(self):
        require(not self.closed and not self.broken,'never finalize broken journal'); self.closed=True
        self._write('events.jsonl.gz',self.comp.flush(zlib.Z_FINISH),append=True,final=True)


class D2PLedger:
    def __init__(self,sink,sync=lambda:None): self.sink,self.sync=sink,sync; self.neural=[]; self.math=[]; self.failed=False
    def begin(self,resource):
        require(not self.failed and resource in ('predict','action_encoder') and all(e['completed'] for e in self.neural),'no unscoped/recursive/retried neural')
        require(sum(e['resource']==resource for e in self.neural)<66,'D2P predispatch66 cap')
        e=dict(index=len(self.neural),resource=resource,completed=False); self.neural.append(e)
        self.sink(dict(kind='native_module_attempt',entry=deepcopy(e))); return e
    def end(self,e):
        require(not e['completed'],'unique module return'); self.sync(); e['completed']=True
        self.sink(dict(kind='native_module_completed',entry=deepcopy(e)))
    def journal(self,e):
        e=deepcopy(e)
        if e['kind'] in ('jacobian','svd'):
            if not e['completed'] and 'error' not in e:
                require(not self.failed and len(self.math)==e['index'] and all(x['completed'] for x in self.math),'ordered nonretry math')
                cap={'jacobian':11,'svd':12}[e['kind']]
                require(sum(x['kind']==e['kind'] for x in self.math)<cap,'predispatch math cap')
                self.math.append(deepcopy(e))
            else:
                require(e['index']<len(self.math) and self.math[e['index']]['kind']==e['kind'] and not self.math[e['index']]['completed'],'math completion identity')
                self.sync(); self.math[e['index']]=deepcopy(e)
                if not e['completed']: self.failed=True
        self.sink(e)
    def counts(self):
        return {k:dict(attempted=sum(e.get('resource',e.get('kind'))==k for e in self.neural+self.math),
            completed=sum(e.get('resource',e.get('kind'))==k and e['completed'] for e in self.neural+self.math)) for k in budgets()[ARMS[1]]}


@contextmanager
def module_guard(model,ledger):
    hooks=[]; pending={}; primary=None
    def pre(resource):
        def enter(module,args):
            x=args[0]; shape=(1,3,192) if resource=='predict' else (1,3,10)
            require(getattr(getattr(x,'device',None),'type',None)=='cuda' and tuple(x.shape)==shape,'single CUDA full-context input')
            require(module not in pending,'no recursive neural entry'); pending[module]=ledger.begin(resource)
        return enter
    def post(module,args,out):
        e=pending.pop(module,None)
        if e is not None:
            if out is None: ledger.failed=True
            else: ledger.end(e)
    def forbidden(*args):
        ledger.failed=True; ledger.sink(dict(kind='forbidden_visual_entry')); raise ValueError('no encoder/projector during latent planning')
    try:
        for m,r in ((model.predictor,'predict'),(model.action_encoder,'action_encoder')):
            hooks.append(m.register_forward_pre_hook(pre(r))); hooks.append(m.register_forward_hook(post,always_call=True))
        for m in (model.encoder,model.projector): hooks.append(m.register_forward_pre_hook(forbidden))
        yield
        require(not pending,'all native module attempts returned')
    except BaseException as exc:
        primary=exc; raise
    finally:
        errors=[]
        for hook in reversed(hooks):
            try: hook.remove()
            except BaseException as e: errors.append(e)
        if errors:
            if primary is not None: primary.add_note('module hook cleanup: '+repr(errors))
            else: raise errors[0]


class ADTrace:
    """Actual engine API entries separate from mathematical output directions."""
    def __init__(self,torch,arm,sink,sync):
        self.torch,self.arm,self.sink,self.sync=torch,arm,sink,sync; self.events=[]; self.depth=0; self.undo=[]; self.failed=False
    def dispatch(self,kind,fn,args,kwargs):
        caps=contract(1)['AD_API_per_root'][self.arm]
        require(not self.failed and sum(e['resource']==kind for e in self.events)<caps[kind],'AD API cap/no retry')
        require(kind!='grad' or self.arm==ARMS[0] or self.depth==1,'D2P grad only inside declared vectorized Jacobian')
        e=dict(index=len(self.events),resource=kind,completed=False,is_grads_batched=bool(kwargs.get('is_grads_batched',False)))
        self.events.append(e); self.sink(dict(kind='AD_API_attempt',entry=deepcopy(e)))
        if kind=='jacobian': self.depth+=1
        try:
            result=fn(*args,**kwargs); self.sync(); e['completed']=True
            self.sink(dict(kind='AD_API_completed',entry=deepcopy(e))); return result
        except BaseException: self.failed=True; raise
        finally:
            if kind=='jacobian': self.depth-=1
    def __enter__(self):
        def patch(obj,k,value): self.undo.append((obj,k,getattr(obj,k))); setattr(obj,k,value)
        def denied(*a,**k): self.failed=True; self.sink(dict(kind='forbidden_AD_API')); raise ValueError('unbudgeted AD API')
        try:
            for obj,k,resource in ((self.torch.autograd,'grad','grad'),(self.torch.autograd.functional,'jacobian','jacobian')):
                original=getattr(obj,k)
                patch(obj,k,lambda *a,_fn=original,_kind=resource,**kw:self.dispatch(_kind,_fn,a,kw))
            for obj,keys in ((self.torch.autograd,('backward',)),(self.torch.Tensor,('backward',)),
                (self.torch.autograd.functional,('hessian','jvp','vjp','hvp','vhp')),
                (self.torch.func,('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'))):
                for k in keys: patch(obj,k,denied)
            return self
        except BaseException: self.__exit__(None,None,None); raise
    def __exit__(self,typ,exc,tb):
        errors=[]
        for obj,k,value in reversed(self.undo):
            try: setattr(obj,k,value)
            except BaseException as e: errors.append(e)
        if errors:
            if exc is not None: exc.add_note('AD restore errors: '+repr(errors))
            else: raise errors[0]
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.events),completed=sum(e['resource']==k and e['completed'] for e in self.events)) for k in ('grad','jacobian')}


def selected(arm,i,result):
    """Array-only validation; no extra SVDs or model calls in result checks."""
    t=i['gn_task']; ident=i['task_id']
    if arm==ARMS[0]:
        require(result['task_id']==ident and len(result['steps'])==4,'four real GN updates')
        exact_counts(result['resources'],budgets()[arm])
        p=result['final_prediction']; encode.core.check_prediction(t,p)
    else:
        require(arm==ARMS[1] and result['status']=='completed' and len(result['candidates'])==10 and
            result['q_sha256']==d2p.digest(i['q']) and result['whole_and_segmented_exact'] is True,'complete fixed10 same-Q D2P')
        ops=result['operations']; kinds=['jacobian','svd','svd']+['jacobian','svd']*10
        require(len(ops)==23 and all(e['index']==k and e['kind']==kind and e['completed'] is True
            and e['q_sha256']==result['q_sha256'] for k,(e,kind) in enumerate(zip(ops,kinds,strict=True))),'all23 actual D2P math receipts')
        require(all(e['shape']==[192,10] and e['mathematical_vjp_directions']==192 for e in ops if e['kind']=='jacobian'),'actual Jacobian output-direction contract')
        for index,c in enumerate(result['candidates']):
            require(c['index']==index and c['completed'] is True and c['independent_model_replay_exact'] is True and
                c['stage_cost']==d2p.cost(c['predicted_prefix'],t['goal'],i['q']) and
                c['objective']==c['stage_cost']+c['own_reach_value'],'all10 fixed-beta model-only candidate scores')
        w=min(result['candidates'],key=lambda c:(c['objective'],c['index']))
        require(result['winner']==w and result['whole_replay_context']==result['segmented_replay_context']==w['context'],'stable winner exact replay')
        logits=[w['prefix_params'],w['suffix_params']]; tokens=encode.core.normalized_tokens(t,np.asarray(w['commands']))
        before=t['root_context']; prefix=dict(before,latent_history=before['latent_history'][1:]+[w['predicted_prefix']],
            action_history=[before['action_history'][-1],tokens[0].tolist()])
        p=dict(task_id=ident,logits=logits,commands=w['commands'],contexts=[prefix,w['context']],whole_replay=w['context'],
            segmented_replay=[prefix,w['context']],cost=w['nonlinear_terminal_cost'])
        encode.core.check_prediction(t,p)
    return dict(task_id=ident,logits=p['logits'],commands=p['commands'],prediction=p['contexts'][-1],model_terminal_cost=p['cost'])


def exact_counts(actual,expected):
    require(canonical(actual)==canonical({k:dict(attempted=v,completed=v) for k,v in expected.items()}),'all actual resources complete and exact, including ADforward')


def execute_arm(arm,i,model,device,torch,sink,receipt):
    sync=lambda:torch.cuda.synchronize(device)
    ad=ADTrace(torch,arm,sink,sync); ledger=None
    try:
        if arm==ARMS[0]:
            ledger=gn.Meter(sink,sync); backend=gn.screen.NativeCuda(model,device,i['gn_task'])
            with ad,backend.guard(ledger): result=gn.optimize(i['gn_task'],backend,ledger)
        else:
            ledger=D2PLedger(sink,sync); backend=d2p.TorchBackend(model,device)
            with ad,module_guard(model,ledger): result=d2p.compile_arm(i['d2p_packet'],i['q'],backend,ledger.journal)
        exact_counts(ledger.counts(),budgets()[arm]); exact_counts(ad.counts(),contract(1)['AD_API_per_root'][arm])
        selected(arm,i,result); return result
    finally:
        receipt['resources']=ledger.counts() if ledger is not None else {}
        receipt['AD_API_resources']=ad.counts()


def plan_all(p,store,runner,rows):
    for slot in p['slots']:
        i=load_export(slot['task_export'],slot['task_sha256'])
        require(i['task_id']==slot['task_id'],'approved opaque slot unchanged')
        for arm in ARMS:
            row=next(r for r in rows if r['order']==slot['order'] and r['arm']==arm); row['status']='attempted'
            def sink(e): store.event(dict(root_order=slot['order'],task_id=i['task_id'],arm=arm,event=e))
            sink(dict(kind='arm_started',task_sha256=slot['task_sha256']))
            result=runner(arm,i,sink,row); choice=selected(arm,i,result)
            exact_counts(row['resources'],budgets()[arm]); exact_counts(row['AD_API_resources'],contract(1)['AD_API_per_root'][arm])
            stem=f'root_{slot["order"]}_{arm}'; store.put(stem+'.result.json.gz',result)
            plan=dict(schema=NAME+'.model_plan',status='selected_pending_global_closure',arm=arm,order=slot['order'],
                task_export=slot['task_export'],task_sha256=slot['task_sha256'],
                task_contract_sha256=i['gn_task']['task_contract_sha256'],with_target=False,
                physical_execution=False,P1_admission=False,selection_uses_physics=False,**choice)
            plan['selection_sha256']=encode.core.digest(plan)
            store.put(stem+'.plan.json',plan); sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
            row.update(status='model_plan_sealed',result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected); store=Store(output); errors=[]; model=None; before=after=host=None; source_ok=False
    rows=[dict(order=s['order'],task_id=s['task_id'],arm=a,status='not_started',resources={},AD_API_resources={}) for s in p['slots'] for a in ARMS]
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
        encode.frozen(model); before=encode.tensor_hash(model); require(before==p['expected_model_hash'],'same frozen native model tensor SHA')
        with sdpa_kernel(SDPBackend.MATH):
            plan_all(p,store,lambda arm,i,sink,row:execute_arm(arm,i,model,device,torch,sink,row),rows)
    except BaseException as exc:
        errors.append(encode.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e: errors.append(encode.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted': row['status']='failed_partial'
        if model is not None:
            try: encode.frozen(model); after=encode.tensor_hash(model)
            except BaseException as e: errors.append(encode.error('final_model',e))
        try: verify(prepared,expected); source_ok=True
        except BaseException as e: errors.append(encode.error('final_source',e))
        try: store.close()
        except BaseException as e: errors.append(encode.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as e: errors.append(encode.error('artifact_hash',e)); complete=False
        result=dict(schema=NAME+'.report',status='model_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,
            resources={a:{k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows if r['arm']==a),
                completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows if r['arm']==a)) for k in budgets()[a]} for a in ARMS},
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,contract=p['contract'],
            commands_released=complete,all_plans_before_physics=True,new_images=0,new_physics=0,training=False,P1_admission=False,
            equal_budget_claim=False,global_freshness=p['global_freshness'])
        try: store.put('report.json',result,final=True)
        except BaseException as e:
            complete=False; errors.append(encode.error('final_report',e)); print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial artifacts preserved, no retry: '+str(errors)); return result


def consume(output,report_sha):
    """Read-only handoff. Does not execute commands or open any sourcepack."""
    output=Path(output).absolute(); require(sha(output/'report.json')==report_sha,'MAIN external actual report SHA')
    r=read(output/'report.json'); require(r['schema']==NAME+'.report' and r['status']=='model_plans_sealed_not_admitted' and
        r['commands_released'] is True and r['errors']==[] and r['model_unchanged'] is True and r['source_verified'] is True,'whole run release, not a partial plan')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['contract']==p['contract'] and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'same model/protocol')
    census={f.name for f in output.iterdir() if f.is_file()}
    require(census==set(r['artifact_sha256'])|{'report.json'},'exact saved artifact census')
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..') and sha(output/name)==h,'artifact external report binding')
    plain,eof=inflate((output/'events.jsonl.gz').read_bytes()); require(eof and plain.endswith(b'\n'),'final complete journal, not partial gzip')
    for line in plain.splitlines(): decode(line)
    expected=[(s['order'],a) for s in p['slots'] for a in ARMS]
    require([(row['order'],row['arm']) for row in r['rows']]==expected,'ALL approved roots x two arms')
    plans=[]
    for row in r['rows']:
        slot=p['slots'][row['order']]; arm=row['arm']; stem=f'root_{row["order"]}_{arm}'
        require(row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','canonical own result paths')
        exact_counts(row['resources'],budgets()[arm]); exact_counts(row['AD_API_resources'],contract(1)['AD_API_per_root'][arm])
        i=load_export(slot['task_export'],slot['task_sha256']); choice=selected(arm,i,read(output/row['result_file']))
        plan=read(output/row['plan_file']); identity=plan.pop('selection_sha256')
        require(encode.core.digest(plan)==identity==row['selection_sha256'] and all(plan[k]==v for k,v in choice.items()) and
            plan['task_sha256']==slot['task_sha256'],'same-task commands selection seal')
        plans.append(dict(plan,selection_sha256=identity,whole_run_report_sha256=report_sha))
    return dict(status='model_commands_verified_for_separate_MAIN_executor',plans=plans,new_physics=0,P1_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run','consume'))
    parser.add_argument('--task-exports',nargs='+'); parser.add_argument('--task-shas',nargs='+')
    for k in ('prepared','prepare-sha','output','report-sha'): parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare': result=prepare(a.output,task_exports=a.task_exports,task_shas=a.task_shas)
    elif a.command=='verify': result=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume': result=consume(a.output,a.report_sha)
    else: result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__': main()
