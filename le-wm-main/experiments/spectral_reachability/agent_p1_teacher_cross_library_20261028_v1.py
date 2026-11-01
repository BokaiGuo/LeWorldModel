"""Eight real teacher15 anchors, fixed common symmetric library; MAIN CUDA only.

Saved acquisition is separate from the blind compiler. No probe endpoint, reward,
native state or response score is accepted by the compiler's input interface.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import stat
import time
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_zero_signed_comparison_20261028_v1 as acquisition
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v2 as kernel
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as overlay
from experiments.spectral_reachability import moving_history_observed_suffix_cuda_20261027_v1 as suffix

NAME='agent_p1_teacher_cross_library_20261028_v1'
AUTH='MAIN_TEACHER8_COMMON_LIBRARY96_CUDA_ONLY'
ROOT=acquisition.ROOT
probe=acquisition.probe
enc=overlay.blind.encode
compiler=kernel.compiler
require,sha,canonical,digest=probe.require,probe.sha,probe.canonical,probe.digest
CAP,FINAL,FLOOR=128*1024**2,32*1024**2,200*1024**2
PREP_CAP,JOURNAL_CAP,REPORT_CAP,WALL=64*1024**2,64*1024**2,16*1024**2,600
ROLES=('Z','S')
LIBRARY=('zero','uZ','-uZ','uS','-uS')
PREREG=ROOT/'innerdocs/agent_p1_next_mechanism_test_20261028_v1.md'
PREREG_SHA='622bc124a5b08f787acac569f7b1875ed1424508d87e9a35bc0670b819d06423'
INPUTS={
 'probe_prepare':('results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json',acquisition.PROBE_SHA),
 'zero_prepare':('results/agent_mixedmode_zero_time_control_20261028_v1_prepare_MAIN01.json','1fbd2607d64ba75aa17ae42833e20ab353c017fb4fec92aef6d6663c24c2c076'),
 'zero_report':('results/agent_mixedmode_zero_time_control_20261028_v1_MAIN01/report.json',acquisition.ZERO_REPORT_SHA),
 'zero_receipt':('results/consume_zero_and_rolling_MAIN_20261028_v1_zero_MAIN01.json',acquisition.ZERO_RECEIPT_SHA),
 'selected_report':('results/agent_mixedmode_capability_physical_20261028_v1_MAIN01/report.json',acquisition.SELECTED_REPORT_SHA),
 'selected_receipt':('results/consume_mixedmode_capability_MAIN_20261028_v1_MAIN01.json',acquisition.SELECTED_RECEIPT_SHA)}


def contract():
    return dict(schema=NAME+'.contract',tasks=[list(k) for k in probe.ORDER],roles=list(ROLES),
        anchors=8,library=list(LIBRARY),future_physical_denominator=40,
        total=dict(encoder=8,projector=8,predict=96,predictor_samples=96,action_encoder=96,pred_proj=96,
                   jacobian=8,AD_forward_in_predict=8,mathematical_VJP=1536,svd=8,bounded_ridge=8),
        per_anchor_predict=12,AD_API_per_anchor=dict(jacobian=1,grad=1),
        damping=compiler.DAMPING,packet_logit_cap=compiler.CAP,beta=1.,Q='original identity192',
        acquisition='each task/role fixed zero-probe START15, same five independent start snapshots/pixels',
        context='same-condition original z5,z10 + own actual z15; warm raw5:10 + own prefix5',
        normalization='original float32 mean/std once, raw units, no second /100',
        raw_authority='once NumPy tanh(float32 logits), inverse is float32 sign flip, NOT time reversal',
        evaluation='all five fresh evaluations then all five independent fresh replays; no winner',
        reusable_encoding='no strictly bound complete eight-anchor encoding found; eight own calls, no dedup',
        failed_denominator='all8 anchors/40 library slots retained; whole run release only',
        physics=0,training=False,Qfit=False,retries=0,selection_uses_probe_response=False,
        P1_admission=False,C2_non_greedy_premise=False,known_roots=2,fresh_roots=0,
        independent_confirmation=False,equal_FLOPs=False,equal_total_compute=False,
        numerics=dict(CUDA_only=True,MATH=True,TF32=False,autocast=False,batch=1),
        inherited_encoding_and_search='paid separately; original history/goal and mixed selection not free',
        prepare_cap_bytes=PREP_CAP,output_cap_bytes=CAP,final_reserve_bytes=FINAL,
        journal_uncompressed_cap_bytes=JOURNAL_CAP,report_cap_bytes=REPORT_CAP,
        wall_seconds=WALL,disk_floor_bytes=FLOOR)


def flat(base,name):
    require(isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'flat path')
    p=Path(base)/name
    require(not p.is_symlink(),'no symlink artifact')
    return p


def read(path,expected=None,cap=PREP_CAP):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'bounded regular JSON')
    raw=p.read_bytes()
    if expected is not None:require(hashlib.sha256(raw).hexdigest()==probe.hash_value(expected),'external file SHA')
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    out=json.loads(raw,object_pairs_hook=pairs);canonical(out);return out


def rgb(path,file_sha,pixel_sha):
    from PIL import Image
    require(sha(path)==file_sha and Path(path).stat().st_size<=1024**2,'actual PNG file binding')
    with Image.open(path) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'official uint8 RGB224')
        a=np.asarray(im).copy()
    require(a.dtype==np.uint8 and hashlib.sha256(a.tobytes()).hexdigest()==pixel_sha,'actual PNG pixels')
    return a


def anchors_from_pairs(p,pairs,bases,bindings):
    """Acquisition only: native snapshots authenticate identity, never model input."""
    anchors=[]
    for ti,task in enumerate(p['tasks']):
        require((task['slot']['root_slot'],task['slot']['condition'])==probe.ORDER[ti],'fixed four task order')
        own=[x for x in pairs if x[0]['task']==ti]
        require([x[0]['probe_id'] for x in own]==[k for k,_ in probe.PROBES],'all five start replicates')
        for role,col,base in (('Z',1,bases[0]),('S',2,bases[1])):
            rows=[x[col] for x in own];f=rows[0]['frames'][0]
            require(rows[0]['probe_id']=='zero' and f['step']==0,'fixed zero-probe START not endpoint20')
            for r in rows:
                q=r['frames'][0]
                require(q['step']==0 and q['pre']==q['post']==f['pre'] and q['state']==f['state'] and
                        q['pixel_sha256']==f['pixel_sha256'],'five identical independent true start15 snapshots')
                path=flat(base,q['file']).resolve();rgb(path,q['sha256'],q['pixel_sha256']);bindings[str(path)]=q['sha256']
            prefix=own[0][0]['intervention_commands'] if role=='Z' else rows[0]['causal_prefix_commands']
            a=probe.raw(prefix,5);require(np.max(abs(a))<=1,'original raw action domain')
            if role=='Z':require(a.tobytes()==np.zeros((5,2),'<f4').tobytes(),'literal zero5 intervention')
            slot=deepcopy(task['slot']);kernel.validate_context(slot['packet']['root_context'])
            q=kernel.validate_q(slot['q']);require(np.array_equal(q,np.eye(192)),'original identity Q unchanged')
            path=Path(slot['task_file']);require(sha(path)==slot['task_sha256'],'original task encoding receipt')
            exported=overlay.read(path)
            require(exported['task_id']==slot['task_id'] and exported['task']['root_context']==slot['packet']['root_context'] and
                    exported['task']['goal']==slot['packet']['goal'],'same encoded task/context/goal authority')
            t=exported['task'];warm=np.concatenate((np.tile(np.asarray([.1,0.],'<f4'),(5,1)),np.tile(np.asarray([0.,.1],'<f4'),(5,1))))
            require(probe.raw(t['root_raw_action_history'],10).tobytes()==warm.tobytes() and
                    np.array_equal(t['root_context']['action_history'],kernel.normalized_history(warm)) and
                    np.array_equal(np.asarray(t['action_mean'],np.float32),kernel.ACTION_MEAN) and
                    np.array_equal(np.asarray(t['action_std'],np.float32),kernel.ACTION_STD) and
                    t['packet_logit_cap']==compiler.CAP and np.array_equal(t['projection'],q),
                    'original true nonzero raw10/float32 tokens/normalizer/ball/Q')
            bindings[str(path.resolve())]=slot['task_sha256']
            # This whitelist is the ONLY input accepted by blind_anchor()/compiler.
            a=dict(index=len(anchors),task=ti,root_slot=slot['root_slot'],condition=slot['condition'],role=role,
                   task_id=slot['task_id'],blind_sha256=slot['blind_sha256'],task_file=slot['task_file'],task_sha256=slot['task_sha256'],
                   root_context=slot['packet']['root_context'],goal=slot['packet']['goal'],q=q.tolist(),prefix_commands=prefix,
                   png=str(flat(base,f['file']).resolve()),png_sha256=f['sha256'],pixel_sha256=f['pixel_sha256'])
            a['anchor_id']=digest(a);anchors.append(a)
    return anchors


def own_files():
    return [Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),PREREG,
            Path(acquisition.__file__),Path(acquisition.math.__file__),Path(kernel.__file__),Path(suffix.__file__),*overlay.source_files()]


def build():
    values={};bindings={}
    for k,(name,h) in INPUTS.items():
        path=ROOT/name;values[k]=read(path,h);bindings[str(path)]=h
    p,zp,zr,zw,sr,sw=[values[k] for k in INPUTS]
    acquisition.accepted_receipts(zr,zw,sr,sw,INPUTS['zero_report'][1],INPUTS['selected_report'][1])
    pairings=acquisition.matched_rows(p,zp,zr,sr)
    # No physical consumer/response analysis/model call. Trust only external accepted receipts.
    bindings.update(p['bindings'])
    anchors=anchors_from_pairs(p,pairings,[(ROOT/INPUTS['zero_report'][0]).parent,(ROOT/INPUTS['selected_report'][0]).parent],bindings)
    require(sha(PREREG)==PREREG_SHA and enc.native_metadata()==overlay.moving.normalization(),'frozen prereg/model/normalizer')
    config=overlay.read(enc.POLICY/'config.json')
    require(overlay.encoder_output_shape(config)==[1,257,192] and config['action_encoder']['emb_dim']==192,'native dense capacity dimensions')
    kernel.validate_normalizer_source(ROOT)
    for f in own_files():bindings[str(Path(f).resolve())]=sha(f)
    bindings.update(enc.MODELS)
    for path,h in bindings.items():require(sha(path)==h,'complete saved source/artifact/model closure')
    result=dict(schema=NAME+'.prepare',contract=contract(),anchors=anchors,bindings=bindings,
                policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,
                ancestry=dict(probe_digest=probe.digest(p),tasks=[dict(index=t['index'],context_digest=t['context_digest'],
                    plan=t['plan'],native=t['native']) for t in p['tasks']]))
    require(len(canonical(result))<=PREP_CAP,'complete bounded64MiB prepare, no trimmed closure')
    return result


def write_new(path,raw):
    p=Path(path).absolute();require(p.name.startswith(NAME+'_') and len(raw)<=PREP_CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(raw),'own prefix,64MiB cap,200MiB floor')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
    sync_dir(p.parent)


def prepare(output):
    raw=canonical(build());write_new(output,raw)
    return dict(prepared=str(Path(output).absolute()),prepare_sha256=sha(output),bytes=len(raw))


def verify(path,expected):
    p=read(path,expected);require(p==build(),'reconstruct full own prepare, not borrowed schema');return p


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Store:
    """Exclusive O(1) durable bytes; partial writes charged, terminal full census."""
    def __init__(self,path):
        self.output=Path(path).absolute()
        require(self.output.name.startswith(NAME+'_') and not self.output.exists() and
                shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'exclusive128MiB output +200MiB floor')
        self.output.mkdir();sync_dir(self.output.parent)
        self.used=0;self.files={};self.index=0;self.previous=None;self.broken=False
    def check(self,size=0,final=False):
        require(self.used+size<=CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.output).free>=FLOOR+size,'cap/final reserve/floor')
    def write(self,name,raw,append=False,final=False):
        p=flat(self.output,name);old=self.files.get(name)
        require(type(raw) is bytes and (old is None or append) and (not append or old is not None or name=='journal.jsonl'),'exclusive/owned append')
        self.check(len(raw),final)
        if name=='journal.jsonl':require((old[2] if old else 0)+len(raw)<=JOURNAL_CAP,'uncompressed journal cap')
        fd=os.open(p,os.O_WRONLY|os.O_NOFOLLOW|(os.O_APPEND if old else os.O_CREAT|os.O_EXCL),0o600)
        try:
            s=os.fstat(fd);require(stat.S_ISREG(s.st_mode) and (old is None or (s.st_dev,s.st_ino,s.st_size)==old),'owned inode/size')
            with os.fdopen(fd,'ab' if old else 'wb',closefd=False) as f:
                require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
        finally:
            s=os.fstat(fd);os.close(fd);self.used+=s.st_size-(old[2] if old else 0);self.files[name]=(s.st_dev,s.st_ino,s.st_size)
        if old is None:sync_dir(self.output)
    def event(self,e):
        require(not self.broken,'no retry after broken journal')
        event=dict(index=self.index,previous_sha256=self.previous,payload=deepcopy(e));event['event_sha256']=digest(event)
        try:self.write('journal.jsonl',canonical(event)+b'\n',append=True)
        except BaseException:self.broken=True;raise
        self.index+=1;self.previous=event['event_sha256']
    def put(self,name,value,final=False):
        raw=canonical(value)
        if name=='report.json':require(len(raw)<=REPORT_CAP,'bounded final report')
        self.write(name,raw,final=final)
    def array(self,name,a):
        a=np.ascontiguousarray(a);require(a.dtype in (np.dtype('uint8'),np.dtype('float32'),np.dtype('float64')) and np.isfinite(a).all(),'original finite precision')
        b=BytesIO();np.save(b,a,allow_pickle=False);npy=b.getvalue();raw=gzip.compress(npy,mtime=0);name=name+'.npy.gz'
        self.write(name,raw)
        return dict(file=name,file_sha256=hashlib.sha256(raw).hexdigest(),npy_sha256=hashlib.sha256(npy).hexdigest(),
                    array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),dtype=str(a.dtype),shape=list(a.shape))
    def census(self):
        actual={}
        for p in self.output.iterdir():
            s=p.lstat();require(stat.S_ISREG(s.st_mode),'no unknown symlink/directory');actual[p.name]=(s.st_dev,s.st_ino,s.st_size)
        require(actual==self.files and sum(s[2] for s in actual.values())==self.used,'terminal full inode/size/byte census')
        self.check(0,final=True);return {n:sha(self.output/n) for n in sorted(actual)}


def blind_anchor(anchor,z):
    enc.native(z)
    root=deepcopy(anchor['root_context']);kernel.validate_context(root)
    prefix=probe.raw(anchor['prefix_commands'],5)
    raw=np.concatenate((np.tile(np.asarray([0.,.1],'<f4'),(5,1)),prefix))
    root.update(latent_history=root['latent_history'][1:]+[z.astype(float).tolist()],
                action_history=kernel.normalized_history(raw).astype(float).tolist())
    kernel.validate_context(root)
    # Explicitly excludes evaluator identity/state/snapshot/reward/probe response fields.
    return dict(context=root,goal=deepcopy(anchor['goal']),q=deepcopy(anchor['q']),raw_history=raw.tolist())


def raw_suffix(params):
    p=compiler.finite(params,(10,));require(np.linalg.norm(p)<=compiler.CAP+1e-12,'original logit ball')
    return np.tanh(p.astype(np.float32)).reshape(5,2)


def common_bank(z,s):
    z,s=probe.raw(z,5),probe.raw(s,5)
    return [np.zeros((5,2),'<f4'),z.copy(),np.negative(z),s.copy(),np.negative(s)]


def stages():return ['nominal','jacobian']+[f'evaluate_{i}' for i in range(5)]+[f'replay_{i}' for i in range(5)]


def incoming(context,raw):
    return dict(latent_history=deepcopy(context['latent_history']),
        action_history=np.concatenate((np.asarray(context['action_history'],np.float32),kernel.normalized_history(raw))).tolist())


def compile_library(anchors,zs,backend_factory,store,rows):
    """Pure fixed orchestration; backend may be RAM analytic fixture, never evidence."""
    require(len(anchors)==len(zs)==len(rows)==8,'complete fixed8 contexts')
    for a,z,row in zip(anchors,zs,rows,strict=True):
        require(a['index']==row['index'] and a['task']==row['index']//2 and a['role']==ROLES[row['index']%2],'fixed task/role order')
        row.update(status='attempted',anchor_id=a['anchor_id'],arrays={},evaluations=[],forwards=[],math=[])
        b=blind_anchor(a,z);row['blind_input']=deepcopy(b)
        for key,value in [('latents',b['context']['latent_history']),('tokens',b['context']['action_history']),('raw_history',b['raw_history']),('goal',b['goal'])]:
            row['arrays'][key]=store.array(f'a{a["index"]}_{key}',np.asarray(value,np.float32))
        be=backend_factory(row,b)
        for name in ('nominal','jacobian'):
            store.event(dict(kind='compiler_stage',anchor=a['index'],stage=name))
            value=be(name,b['context'],np.zeros((5,2),'<f4'))
            row['arrays'][name]=store.array(f'a{a["index"]}_{name}',np.asarray(value))
            store.event(dict(kind='compiler_array_saved',anchor=a['index'],stage=name,receipt=row['arrays'][name]))
            if name=='nominal':nominal=compiler.finite(value,(192,))
            else:jac=compiler.finite(value,(192,10))
        row['math'].append(dict(resource='svd',completed=False))
        store.event(dict(kind='solve_attempt',anchor=a['index'],resource='svd',helper_SVD=1))
        solve=compiler.bounded_ridge(np.asarray(b['q'])@jac,np.asarray(b['q'])@(np.asarray(b['goal'])-nominal),damping=compiler.DAMPING,cap=compiler.CAP)
        row['math'][-1]['completed']=True;row['solve']=solve
        row['suffix_commands']=raw_suffix(solve['params']).astype(float).tolist()
        row['arrays']['logits']=store.array(f'a{a["index"]}_logits',np.asarray(solve['params'],np.float64))
        row['arrays']['raw_suffix']=store.array(f'a{a["index"]}_raw_suffix',probe.raw(row['suffix_commands'],5))
        row['status']='suffix_frozen';store.event(dict(kind='suffix_frozen',anchor=a['index'],solve=solve,raw=row['suffix_commands']))
    # ALL eight solutions are frozen before any common-library cost is requested.
    for task in range(4):
        bank=common_bank(rows[2*task]['suffix_commands'],rows[2*task+1]['suffix_commands'])
        store.event(dict(kind='common_bank_frozen',task=task,library=[x.tolist() for x in bank]))
        for row in rows[2*task:2*task+2]:
            b=row['blind_input'];be=backend_factory(row,b)
            for pass_name in ('evaluate','replay'):
                for j,raw in enumerate(bank):
                    stage=f'{pass_name}_{j}';e=dict(stage=stage,library_index=j,raw_commands=raw.astype(float).tolist(),status='attempted')
                    row['evaluations'].append(e);store.event(dict(kind='library_attempt',anchor=row['index'],entry=deepcopy(e)))
                    after=be(stage,deepcopy(b['context']),raw.copy());kernel.validate_advance(b['context'],raw,after)
                    require(np.array_equal(np.asarray(after['latent_history'])[:2],np.asarray(b['context']['latent_history'])[1:]),'actual chronology shift')
                    value=float(np.mean((np.asarray(b['q'])@(np.asarray(after['latent_history'][-1])-np.asarray(b['goal'])))**2))
                    e.update(status='completed',cost=value,context=after,
                        latents=store.array(f'a{row["index"]}_{stage}_latents',np.asarray(after['latent_history'],np.float32)),
                        tokens=store.array(f'a{row["index"]}_{stage}_tokens',np.asarray(after['action_history'],np.float32)))
                    store.event(dict(kind='library_return',anchor=row['index'],entry=deepcopy(e)))
            require(all(row['evaluations'][i]['context']==row['evaluations'][i+5]['context'] for i in range(5)),'five exact independent replays')
            row['status']='all_five_replayed';store.event(dict(kind='anchor_completed',anchor=row['index']))
    return rows


class CudaBackend:
    """Frozen backend reused directly; no old task/global loader or gate patched."""
    def __init__(self,model,device,torch,trace,store,row,check):
        self.native=kernel.TorchBackend(model,device);self.torch=torch;self.device=device
        self.trace,self.store,self.row,self.check=trace,store,row,check
    def __call__(self,stage,context,raw):
        self.check();i=len(self.row['forwards']);require(i<12 and stage==stages()[i],'nonretry12 calls per anchor')
        f=dict(index=i,stage=stage,raw_commands=raw.tolist(),expected=incoming(context,raw),completed=False)
        self.row['forwards'].append(f);scope=dict(anchor=self.row['index'],stage=stage)
        self.store.event(dict(kind='forward_attempt',anchor=self.row['index'],entry=deepcopy(f)))
        with overlay.traced_scope(self.trace,scope,f):
            if stage=='jacobian':
                # Scoped, restored AD API instrumentation; no task/module global substitution.
                ad=suffix.SuffixAD(self.torch,'teacher',lambda e:self.store.event(dict(anchor=self.row['index'],**e)),lambda:self.torch.cuda.synchronize(self.device))
                ad.stage=lambda:stage
                try:
                    with ad:value=self.native.jacobian(deepcopy(context),lambda kind,fn:fn())
                finally:f['AD_API']=deepcopy(ad.events)
            elif stage=='nominal':value=self.native.nominal(deepcopy(context))
            else:value=self.native.step(deepcopy(context),raw.copy())
        require([m['resource'] for m in f['modules']]==['action_encoder','predictor','pred_proj'],'one full native forward')
        f['completed']=True;self.store.event(dict(kind='forward_return',anchor=self.row['index'],entry=deepcopy(f)))
        return value


class NativeTrace(overlay.NativeTrace):
    """Same original-precision capture utility, OWN hard predispatch8/96 limits."""
    def __enter__(self):
        def pre(kind):
            def hook(module,args):
                self.check();require(self.scope is not None and module not in self.pending,'scoped nonrecursive native input')
                image='image_index' in self.scope
                require((kind in ('encoder','projector'))==image,'visual/compiler scopes separated')
                require(sum(e['resource']==kind for e in self.entries)<(8 if image else 96),'own actual native predispatch8/96 cap')
                require(not any(e['scope']==self.scope and e['resource']==kind for e in self.entries),'no duplicate native invocation in same forward')
                e=dict(index=len(self.entries),resource=kind,scope=deepcopy(self.scope),completed=False)
                self.entries.append(e);self.pending[module]=e
                self.store.event(dict(kind='tensor_module_attempt',entry=deepcopy(e)))
                shape={'encoder':(1,3,224,224),'projector':(1,192),'action_encoder':(1,3,10),
                       'predictor':(1,3,192),'pred_proj':(3,192)}[kind]
                require(len(args)==(2 if kind=='predictor' else 1) and tuple(args[0].shape)==shape,
                        'full native batch1 dimensions, no hidden batching')
                e['inputs']=[self.save(x,f'module_{e["index"]:04}_input{j}') for j,x in enumerate(args)]
            return hook
        def post(module,args,out):
            e=self.pending.pop(module);require(out is not None,'actual native module returned')
            if hasattr(out,'last_hidden_state'):out=out.last_hidden_state
            if e['resource']=='encoder':require(tuple(out.shape)==self.encoder_shape,'full257 patch14 encoder output')
            e['output']=self.save(out,f'module_{e["index"]:04}_output')
            self.torch.cuda.synchronize();self.check();e['completed']=True
            self.store.event(dict(kind='tensor_module_completed',entry=deepcopy(e)))
        try:
            for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
                module=getattr(self.model,kind)
                self.hooks.append(module.register_forward_pre_hook(pre(kind)))
                self.hooks.append(module.register_forward_hook(post))
            return self
        except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise


def plans_for(p,rows):
    plans=[]
    for a,row in zip(p['anchors'],rows,strict=True):
        require(row['status']=='all_five_replayed','whole complete library release')
        for j,e in enumerate(row['evaluations'][:5]):
            plan=dict(schema=NAME+'.model_plan',index=len(plans),task=a['task'],root_slot=a['root_slot'],condition=a['condition'],role=a['role'],
                library_index=j,library_label=LIBRARY[j],anchor_id=a['anchor_id'],blind_sha256=a['blind_sha256'],task_id=a['task_id'],
                prefix_commands=a['prefix_commands'],suffix_commands=e['raw_commands'],commands=a['prefix_commands']+e['raw_commands'],
                raw_authority='literal_float32',teacher_context_sha256=digest(row['blind_input']['context']),model_Q=e['cost'],
                selection_uses_probe_response=False,selection_uses_physics=False,P1_admission=False)
            plan['plan_digest']=digest(plan);plans.append(plan)
    require(len(plans)==40,'fixed40 including duplicate commands');return plans


def actual_resources(modules,rows):
    def count(entries):return dict(attempted=len(entries),completed=sum(bool(e['completed']) for e in entries))
    result={kind:count([e for e in modules if e['resource']==native]) for kind,native in
            [('encoder','encoder'),('projector','projector'),('predict','predictor'),
             ('action_encoder','action_encoder'),('pred_proj','pred_proj')]}
    result['predictor_samples']=deepcopy(result['predict'])
    ad=[e for row in rows for f in row.get('forwards',[]) for e in f.get('AD_API',[])]
    result['jacobian']=count([e for e in ad if e['resource']=='jacobian'])
    result['AD_grad_API']=count([e for e in ad if e['resource']=='grad'])
    result['AD_forward_in_predict']=count([e for e in modules if e['resource']=='predictor' and e['scope'].get('stage')=='jacobian'])
    result['mathematical_VJP']={k:192*v for k,v in result['jacobian'].items()}
    math=[e for row in rows for e in row.get('math',[])]
    result['bounded_ridge']=count(math)
    # Successful helper source has exactly one SVD; failure before/inside it is not an API receipt.
    result['SVD_successful_helper_calls']=sum(bool(e['completed']) for e in math)
    result['SVD_failed_helper_internal_attempts']='unknown' if any(not e['completed'] for e in math) else 0
    return result


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA authorization')
    p=verify(prepared,expected);start=time.monotonic();store=Store(output);errors=[];model=None;before=after=None;source_ok=False;host=None
    rows=[dict(index=i,status='not_started') for i in range(8)];images=[];trace=None
    def check():require(time.monotonic()-start<=WALL,'600s predispatch wall');store.check()
    def err(phase,e):errors.append(dict(phase=phase,type=type(e).__name__,message=str(e)))
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen tensor hash')
        trace=NativeTrace(model,torch,store,check,[1,257,192]);zs=[]
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False),trace:
            transform=p0.image_transform()
            for a in p['anchors']:
                check();image=dict(index=a['index'],anchor_id=a['anchor_id'],status='attempted');images.append(image)
                pixels=rgb(a['png'],a['png_sha256'],a['pixel_sha256']);image['pixels']=store.array(f'image{a["index"]}_pixels',pixels)
                store.event(dict(kind='image_attempt',entry=deepcopy(image)))
                with overlay.traced_scope(trace,dict(image_index=a['index']),image),torch.inference_mode():
                    x=p0._image_tensor(pixels,transform=transform,device=device)
                    image['preprocessed']=trace.save(x,f'image{a["index"]}_preprocessed')
                    z=model.encode({'pixels':x})['emb'][0,0].detach().cpu().numpy().copy()
                enc.native(z);image['latent']=store.array(f'image{a["index"]}_latent',z)
                image['status']='completed';zs.append(z);store.event(dict(kind='image_return',entry=deepcopy(image)))
            factory=lambda row,b:CudaBackend(model,device,torch,trace,store,row,check)
            compile_library(p['anchors'],zs,factory,store,rows)
    except BaseException as e:
        err('execution',e)
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:err('error_journal',e)
    finally:
        for row in rows:
            if row['status'] not in ('not_started','all_five_replayed'):row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:err('final_model',e)
        try:verify(prepared,expected);check();source_ok=True
        except BaseException as e:err('final_source_or_wall',e)
        hashes={}
        try:hashes=store.census()
        except BaseException as e:err('final_census',e)
        good=not errors and source_ok and before==after==p['expected_model_hash'] and all(r['status']=='all_five_replayed' for r in rows)
        report=dict(schema=NAME+'.report',status='common_library_sealed_not_admitted' if good else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),rows=rows,images=images,
            native_module_attempts=deepcopy(trace.entries) if trace is not None else [],
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            commands_released=good,errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,
            resources=actual_resources(trace.entries if trace is not None else [],rows),
            journal=dict(events=store.index,last_sha256=store.previous),elapsed_seconds=time.monotonic()-start,host=host,
            P1_admission=False,new_physics=0,fixed_denominator=8,future_library_denominator=40,
            library_slots=[dict(anchor=i,library_index=j,status=next((e['status'] for e in rows[i].get('evaluations',[]) if e['stage']==f'evaluate_{j}'),'not_started')) for i in range(8) for j in range(5)])
        try:store.put('report.json',report,final=True);store.census()
        except BaseException as e:
            err('final_report',e);good=False
            try:store.put('failure.json',dict(schema=NAME+'.failure',errors=errors,fixed_denominator=8,commands_released=False),final=True)
            except BaseException as secondary:err('failure_receipt',secondary)
    require(good,'partial preserved, no retry/command release: '+str(errors));return report


def load_array(base,r):
    path=flat(base,r['file']);require(sha(path)==r['file_sha256'] and path.stat().st_size<=2*1024**2,'array compressed binding')
    with gzip.GzipFile(fileobj=BytesIO(path.read_bytes())) as f:npy=f.read(2*1024**2+1)
    require(len(npy)<=2*1024**2 and hashlib.sha256(npy).hexdigest()==r['npy_sha256'],'array NPY binding')
    stream=BytesIO(npy);a=np.load(stream,allow_pickle=False)
    require(a.dtype in (np.dtype('uint8'),np.dtype('float32'),np.dtype('float64')) and
        stream.tell()==len(npy) and list(a.shape)==r['shape'] and str(a.dtype)==r['dtype'] and
        np.isfinite(a).all() and hashlib.sha256(a.tobytes()).hexdigest()==r['array_sha256'],'array original dtype/shape/bytes')
    return a


def check_forward(f,b,load):
    require(f['completed'] is True and f['expected']==incoming(b['context'],probe.raw(f['raw_commands'],5)),'full literal input/token arithmetic')
    mods=f['modules'];require([m['resource'] for m in mods]==['action_encoder','predictor','pred_proj'] and all(m['completed'] for m in mods),'complete native module triple')
    ae,pr,pp=mods;ain=load(ae['inputs'][0]);aout=load(ae['output']);pin=[load(x) for x in pr['inputs']];pout=load(pr['output']);ppin=load(pp['inputs'][0]);ppout=load(pp['output'])
    require(all(x.dtype==np.float32 for x in (ain,aout,*pin,pout,ppin,ppout)) and
        ain.shape==(1,3,10) and np.array_equal(ain[0],f['expected']['action_history']) and
        aout.shape[:2]==(1,3) and len(pin)==2 and pin[0].shape==(1,3,192) and
        np.array_equal(pin[0][0],f['expected']['latent_history']) and np.array_equal(pin[1],aout) and
        pout.shape==(1,3,192) and np.array_equal(ppin,pout.reshape(3,192)) and ppout.shape==(3,192),
        'complete AE/predictor/projection bridge, not repeated receipt equality')
    if f['stage']=='jacobian':
        ad=f['AD_API'];require([(e['resource'],e['completed']) for e in ad]==[('jacobian',True),('grad',True)] and ad[1]['is_grads_batched'] is True,'actual vectorized AD API trace')
    else:require('AD_API' not in f,'no extra AD stages')
    return ppout[-1].copy()


def validate_saved(p,r,load):
    require(len(r['rows'])==len(r['images'])==8,'all8 teacher anchors, no replacement')
    for a,row,image in zip(p['anchors'],r['rows'],r['images'],strict=True):
        i=a['index'];require(digest({k:v for k,v in a.items() if k!='anchor_id'})==a['anchor_id'] and
            i==2*a['task']+ROLES.index(a['role']) and row['index']==image['index']==i and
            image['anchor_id']==row['anchor_id']==a['anchor_id'] and image['status']=='completed','anchor identity')
        pixels=load(image['pixels']);require(pixels.dtype==np.uint8 and np.array_equal(pixels,rgb(a['png'],a['png_sha256'],a['pixel_sha256'])),'actual15 pixels not terminal20')
        require([m['resource'] for m in image['modules']]==['encoder','projector'] and all(m['completed'] for m in image['modules']),'eight actual independent visual calls')
        en,proj=image['modules'];z=load(image['latent']);pre=load(image['preprocessed'])
        require(all(load(x).dtype==np.float32 for mod in image['modules'] for x in mod['inputs']+[mod['output']]) and
                z.dtype==pre.dtype==np.float32,'original native visual precision')
        overlay.visual_bridge(pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z,[1,257,192])
        b=blind_anchor(a,z);require(row['blind_input']==b,'teacher chronology exact reconstructed from raw')
        for key,value in [('latents',b['context']['latent_history']),('tokens',b['context']['action_history']),('raw_history',b['raw_history']),('goal',b['goal'])]:
            require(np.array_equal(load(row['arrays'][key]),np.asarray(value,np.float32)),'original input arrays')
        require(row['status']=='all_five_replayed' and [f['stage'] for f in row['forwards']]==stages() and
                row['math']==[dict(resource='svd',completed=True)],'12 exact forwards/one solve each')
        native=[check_forward(f,b,load) for f in row['forwards']]
        nominal=load(row['arrays']['nominal']);jac=load(row['arrays']['jacobian'])
        compiler.finite(nominal,(192,));compiler.finite(jac,(192,10))
        require(np.array_equal(native[0],nominal) and np.array_equal(native[1],nominal),'AD primal equals fresh nominal')
        solve=compiler.bounded_ridge(np.asarray(b['q'])@jac,np.asarray(b['q'])@(np.asarray(b['goal'])-nominal),damping=compiler.DAMPING,cap=compiler.CAP)
        require(row['solve']==solve and np.array_equal(load(row['arrays']['logits']),solve['params']),'independent saved ridge arithmetic')
        expected=raw_suffix(solve['params']);raw=probe.raw(row['suffix_commands'],5)
        require(raw.tobytes()==expected.tobytes()==load(row['arrays']['raw_suffix']).tobytes(),'once NumPy tanh raw authority')
        bank=common_bank(r['rows'][2*a['task']]['suffix_commands'],r['rows'][2*a['task']+1]['suffix_commands'])
        require(len(row['evaluations'])==10,'all5 evaluate+all5 replay')
        for j,e in enumerate(row['evaluations']):
            command=bank[j%5];after=e['context'];kernel.validate_advance(b['context'],command,after)
            require(e['status']=='completed' and e['library_index']==j%5 and e['stage']==stages()[j+2] and
                    probe.raw(e['raw_commands'],5).tobytes()==command.tobytes()==probe.raw(row['forwards'][j+2]['raw_commands'],5).tobytes() and
                    np.array_equal(after['latent_history'][:2],np.asarray(b['context']['latent_history'])[1:]) and
                    np.array_equal(after['latent_history'][-1],native[j+2]) and
                    np.array_equal(load(e['latents']),after['latent_history']) and np.array_equal(load(e['tokens']),after['action_history']),
                    'common raw buffer/actual full context/native output identity')
            cost=float(np.mean((np.asarray(b['q'])@(native[j+2].astype(float)-np.asarray(b['goal'])))**2))
            require(e['cost']==cost and after==row['evaluations'][j%5]['context'],'same Q actual native model score and independent replay')
    return plans_for(p,r['rows'])


def consume(output,report_sha):
    output=Path(output).absolute();r=read(output/'report.json',report_sha,REPORT_CAP)
    require(r['schema']==NAME+'.report' and r['status']=='common_library_sealed_not_admitted' and r['errors']==[] and
            r['first_error'] is None and r['commands_released'] is r['source_verified'] is r['model_unchanged'] is True,'complete own report only')
    require(r['fixed_denominator']==8 and r['future_library_denominator']==40 and
            r['library_slots']==[dict(anchor=i,library_index=j,status='completed') for i in range(8) for j in range(5)],'full40 release census')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==contract() and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['elapsed_seconds']<=WALL,'original model/protocol/wall')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'exact artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(flat(output,name))==h,'whole saved artifact closure')
    path=output/'journal.jsonl';require(path.stat().st_size<=JOURNAL_CAP,'bounded whole journal');events=[];prev=None
    with path.open('rb') as f:
        for line in f:
            require(line.endswith(b'\n'),'complete durable event');e=json.loads(line);h=e.pop('event_sha256')
            require(e['index']==len(events) and e['previous_sha256']==prev and digest(e)==h,'actual hashchain order')
            prev=h;events.append(e['payload'])
    require(r['journal']==dict(events=len(events),last_sha256=prev),'report journal terminal binding')
    # All actual attempt/return arrays must agree with the external report, not just its counters.
    attempted=[e['entry'] for e in events if e['kind']=='tensor_module_attempt']
    completed=[e['entry'] for e in events if e['kind']=='tensor_module_completed']
    modules=r['native_module_attempts']
    require(len(modules)==304 and [e['index'] for e in modules]==list(range(304)) and all(e['completed'] for e in modules),'16 visual +288 prediction actual native calls')
    require(len(attempted)==len(completed)==304 and completed==modules,'native module attempts/returns durable full denominator')
    for a,c in zip(attempted,completed,strict=True):
        require(a=={k:c[k] for k in ('index','resource','scope','completed')}|{'completed':False},'native predispatch attempt before input arrays')
    require([e['anchor'] for e in events if e['kind']=='suffix_frozen']==list(range(8)) and
            max(i for i,e in enumerate(events) if e['kind']=='suffix_frozen')<min(i for i,e in enumerate(events) if e['kind']=='common_bank_frozen'),
            'all eight suffixes freeze before shared bank evaluation')
    plans=validate_saved(p,r,lambda entry:load_array(output,entry))
    resources=actual_resources(modules,r['rows']);require(r['resources']==resources,'actual native/AD/helper count reconstruction')
    for kind,cap in contract()['total'].items():
        if kind=='svd':require(resources['SVD_successful_helper_calls']==cap and resources['SVD_failed_helper_internal_attempts']==0,'eight source-bound successful SVDs')
        else:require(resources[kind]==dict(attempted=cap,completed=cap),'all fixed resource axes actual exact '+kind)
    require(resources['AD_grad_API']==dict(attempted=8,completed=8),'eight actual batched grad engine calls')
    nested=[m for im in r['images'] for m in im['modules']]+[m for row in r['rows'] for f in row['forwards'] for m in f['modules']]
    require(sorted(nested,key=lambda x:x['index'])==modules,'no extra/unassociated native module calls')
    for kind,field in [('forward_return','forwards'),('library_return','evaluations')]:
        saved=[e for e in events if e['kind']==kind]
        for row in r['rows']:require([e['entry'] for e in saved if e['anchor']==row['index']]==row[field],'durable returned records equal saved rows')
    for row in r['rows']:
        i=row['index'];scope=[e for e in events if e.get('anchor')==i]
        require([e['entry']['stage'] for e in scope if e['kind']=='forward_attempt']==stages() and
                [e['entry']['stage'] for e in scope if e['kind']=='library_attempt']==stages()[2:],'all96/80 attempt lower bounds before dispatch')
        attempt=[e['entry'] for e in scope if e['kind']=='AD_API_attempt']
        done=[e['entry'] for e in scope if e['kind']=='AD_API_completed']
        actual=row['forwards'][1]['AD_API']
        require(done==[actual[1],actual[0]] and
                attempt==[dict(actual[0],completed=False),dict(actual[1],completed=False)],'nested actual AD attempts and returned engine calls')
        require(len([e for e in scope if e['kind']=='solve_attempt'])==1,'one bounded-ridge/SVD helper attempt each')
    require([e['entry'] for e in events if e['kind']=='image_return']==r['images'] and
            len([e for e in events if e['kind']=='image_attempt'])==8,'all8 actual image attempts/returns')
    return dict(schema=NAME+'.consume',status='teacher_cross_library_commands_verified_not_admitted',report_sha256=report_sha,
        prepare_sha256=r['prepare_sha256'],plans=plans,resources=resources,saved_arithmetic_SVD=8,
        P1_admission=False,C2_non_greedy_premise=False,new_neural_calls=0,new_physics_calls=0)


def dense_capacity():
    """Conservative gzip worst-case bound for original module arrays, no neural calls.

    Native192 AE dimension is bounded at192 by frozen config; verify at build via
    config receipts. Raw byte bound includes every duplicate in saved stage context.
    """
    image=224*224*3+2*3*224*224*4+257*192*4+2*192*4+192*4
    forward=(30+3*192+3*192+3*192+3*192+3*192+3*192)*4
    arrays=8*image+96*forward+8*(192*10*8+192*8+10*8)+80*(3*192+20)*4
    # Per-array NPY/header/gzip overhead, journal all dense receipts, terminal duplication.
    return dict(original_array_bytes_bound=arrays,array_file_count_bound=1200,
        storage_bound_bytes=arrays+1200*1024+JOURNAL_CAP+REPORT_CAP,
        output_cap_bytes=CAP,fits=arrays+1200*1024+JOURNAL_CAP+REPORT_CAP<CAP)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume','build-memory','capacity'))
    for name in ('output','prepared','prepare-sha','report-sha','authorization'):ap.add_argument('--'+name)
    a=ap.parse_args()
    if a.command=='prepare':out=prepare(a.output)
    elif a.command=='verify':out=dict(status='verified_no_execution',anchors=len(verify(a.prepared,a.prepare_sha)['anchors']))
    elif a.command=='build-memory':p=build();out=dict(status='saved_only_no_prepare_written',canonical_bytes=len(canonical(p)),anchors=len(p['anchors']),bindings=len(p['bindings']),capacity=dense_capacity())
    elif a.command=='capacity':out=dense_capacity()
    elif a.command=='run':out=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:out=consume(a.output,a.report_sha)
    print(canonical(out).decode())


if __name__=='__main__':main()
