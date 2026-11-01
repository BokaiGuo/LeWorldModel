"""Approved blind NPZ only -> real native tasks for GN/conditional full10 D2P.

MAIN owns sourcepack admission and CUDA launch. No certificate/sourcepack path
is accepted by this deployment interface. Prepare/tests never import torch.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sysconfig
import zipfile

import numpy as np
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME='p1_constructive_blind_encode_cuda_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
POLICY=Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
MODELS={str(POLICY/'config.json'):'2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09',
        str(POLICY/'weights.pt'):'48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'}
MODEL_TENSOR_SHA='4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546'
NORMALIZER_SHA='0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34'
APPROVED_INPUTS={str(ROOT/'results/p1_constructive_reachable_source_20261027_v1_main01/s0_blind_inputs.npz'):'d12ef7b66b69e53444ae6a4bde51c3830bacd6ad449d79845030ea70e2908da3',
                 str(ROOT/'results/p1_constructive_reachable_source_20261027_v1_main01/s1_blind_inputs.npz'):'9c0f6451de986bbb48c8251a8cfaab2b3fed60ce15698a9c6454c6206e228957'}
CAP,FLOOR,FINAL=2*1024**2,200*1024**2,128*1024
require=core.require


def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(raw): return hashlib.sha256(raw).hexdigest()


def sha(path):
    p=Path(path); require(p.is_file() and not p.is_symlink(),'regular nonsymlink input')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def read(path):
    require(Path(path).stat().st_size<2*1024**2,'small JSON')
    data=Path(path).read_bytes()
    if str(path).endswith('.gz'):
        with gzip.GzipFile(fileobj=BytesIO(data)) as stream: data=stream.read(2*1024**2+1)
    require(len(data)<2*1024**2,'bounded expanded JSON')
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key'); result[k]=v
        return result
    value=json.loads(data,object_pairs_hook=unique); canonical(value); return value


def load_blind(path,expected):
    """No constructive module import, no np.load directory discovery/side files."""
    path=Path(path).absolute(); core.hash_value(expected)
    require(path.name.endswith('_blind_inputs.npz') and path.resolve()==path and path.stat().st_size<=2*1024**2,'own regular blind NPZ only')
    require(sha(path)==expected,'external blind file SHA')
    raw=path.read_bytes(); require(digest(raw)==expected,'blind read TOCTOU')
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        entries=archive.infolist(); names={'rgb_history.npy','raw_action_history.npy','goal_rgb.npy'}
        require(len(entries)==3 and {e.filename for e in entries}==names,'exact blind-only archive fields; no privileged extra members')
        require(all(e.file_size<=512*1024 and not e.flag_bits&1 and e.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) for e in entries), 'bounded unencrypted NPZ members')
        require(sum(e.file_size for e in entries)<1024**2,'blind expanded cap')
    with np.load(BytesIO(raw),allow_pickle=False) as archive: values={k:archive[k].copy() for k in archive.files}
    require(values['rgb_history'].shape==(3,224,224,3) and values['rgb_history'].dtype==np.uint8,'three actual observed RGB224')
    require(values['goal_rgb'].shape==(224,224,3) and values['goal_rgb'].dtype==np.uint8,'goal-only RGB224')
    a=values['raw_action_history']; require(a.dtype==np.float32 and a.shape==(10,2) and not a.any(),'ten real zero raw actions')
    return values


def whitelist(path,expected):
    path=Path(path).absolute(); require(sha(path)==expected,'MAIN whitelist external SHA'); p=read(path)
    core.fields(p,'schema approved_by purpose source_admission admission_sha256 with_target history_steps raw_action_steps slots','approved blind whitelist')
    require(p['schema']==NAME+'.approved_blind_whitelist' and p['approved_by']=='MAIN' and p['purpose']=='calibration' and
            p['with_target'] is False and p['history_steps']==[0,5,10] and p['raw_action_steps']==10,'new no-target observed source contract')
    require(p['source_admission']=='MAIN_independently_admitted','MAIN full-source audit must precede task preparation')
    core.hash_value(p['admission_sha256']); require(1<=len(p['slots'])<=4,'one to four MAIN-approved slots, no automatic replacements')
    rows=[]
    for i,slot in enumerate(p['slots']):
        core.fields(slot,'order source_slot blind_npz blind_sha256','blind slot')
        require(type(slot['order']) is int and slot['order']==i and type(slot['source_slot']) is int and 0<=slot['source_slot']<4,'stable approved/source slot identity')
        core.hash_value(slot['blind_sha256'])
        blind=Path(slot['blind_npz']); blind=blind if blind.is_absolute() else path.parent/blind
        blind=blind.absolute(); packet=load_blind(blind,slot['blind_sha256'])
        rows.append(dict(slot,blind_npz=str(blind),pixel_sha256=[digest(rgb.tobytes()) for rgb in [*packet['rgb_history'],packet['goal_rgb']]],
                         raw_action_sha256=digest(packet['raw_action_history'].tobytes())))
    require(all(len({s[k] for s in rows})==len(rows) for k in ('source_slot','blind_sha256','blind_npz')),'no duplicated source/identity/path; repeated image pixels retained')
    require(sha(path)==expected,'whitelist postread TOCTOU')
    return p,rows


def native_metadata():
    for path,h in MODELS.items(): require(sha(path)==h,'original frozen checkpoint')
    config=read(POLICY/'config.json')
    require(config['encoder'].get('pretrained') is False,'offline checkpoint only')
    source=ROOT/'experiments/run_pusht_counterfactual.py'; require(sha(source)==NORMALIZER_SHA,'raw normalizer source seal')
    constants={}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'):
            require(isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='np.asarray','literal native normalizer')
            constants[node.targets[0].id]=np.asarray(ast.literal_eval(node.value.args[0]),np.float32).tolist()
    require(set(constants)=={'PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'},'original mean/std')
    return dict(mean=constants['PUBLIC_ACTION_MEAN'],std=constants['PUBLIC_ACTION_STD'],source_sha256=NORMALIZER_SHA,
                training_provenance_verified=False,raw_semantics='relative raw already action_scale/100; no second division')


def contract(n):
    return dict(with_target=False,roots=n,images_per_root=4,encoder_cap=4*n,projector_cap=4*n,
        image_order=['observed0','observed5','observed10','goal'],history='states3/between_actions2; actual zero warmup, not padding',
        horizon=2,packet_steps=5,raw_action_steps=10,representation='native_projected_cls192',
        metric='identity_native192, no fit',batch=1,predict_cap=0,actionencoder_cap=0,AD_cap=0,physics_cap=0,
        no_cpu_neural_fallback=True,no_retries=True,no_dedup=True,training=False,P1_admission=False,
        source_admission='MAIN whitelist attestation only; no reachability/certificate/freshness verification by deployment',
        task_interface='core.validate_task/core.task_id + screen.NativeCuda; conditional compile_arm(packet,q,backend,journal)',
        output_cap_bytes=CAP,reserve_bytes=FLOOR)


def source_files():
    site=Path(sysconfig.get_paths()['purelib'])
    relative=['stable_worldmodel/wm/lewm/lewm.py','stable_worldmodel/wm/lewm/module.py','stable_pretraining/backbone/utils.py',
        'stable_pretraining/data/dataset_stats.py','transformers/models/vit/modeling_vit.py','transformers/models/vit/configuration_vit.py',
        'torchvision/transforms/v2/_container.py','torchvision/transforms/v2/_misc.py','torchvision/transforms/v2/_geometry.py',
        'torchvision/transforms/v2/functional/_misc.py','torchvision/transforms/v2/functional/_geometry.py','torchvision/transforms/v2/functional/_type_conversion.py']
    local=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),Path(core.__file__),
        ROOT/'experiments/checkpoint_compat.py',ROOT/'experiments/gpu_preflight.py',ROOT/'experiments/run_pusht_counterfactual.py']
    local += [ROOT/'experiments/spectral_reachability'/f for f in ('preflight_pusht_d2p_p0.py','p1_conditional_direct_screen_20261027_v1.py',
        'pusht_d2p_packet_contract.py','p3_goal_patch_consistent_compiler_20261027_v1.py','pusht_full_packet_compiler_20261027.py',
        'pusht_full_packet_chart_20261027.py','pusht_composition_cuda_replay_v2.py')]
    return sorted(set(local+[site/r for r in relative]))


def build(approved_whitelist,whitelist_sha):
    approved_whitelist=str(Path(approved_whitelist).absolute()); manifest,slots=whitelist(approved_whitelist,whitelist_sha)
    require({s['blind_npz']:s['blind_sha256'] for s in slots}==APPROVED_INPUTS,'this version only MAIN explicitly approved two blind payloads')
    require([(s['source_slot'],s['blind_npz']) for s in slots]==list(enumerate(APPROVED_INPUTS)), 'fixed approved source-slot order s0 then s1')
    normalization=native_metadata()
    bindings={approved_whitelist:whitelist_sha,**MODELS,**{s['blind_npz']:s['blind_sha256'] for s in slots}}
    bindings.update({str(f.resolve()):sha(f) for f in source_files()})
    for path,h in bindings.items(): require(sha(path)==h,'prepare source/input TOCTOU')
    return dict(schema=NAME+'.prepare',arguments=dict(approved_whitelist=approved_whitelist,whitelist_sha=whitelist_sha),
        authority=dict(approved_by='MAIN',admission_sha256=manifest['admission_sha256'],sourcepack_path_not_provided=True),
        slots=slots,contract=contract(len(slots)),normalization=normalization,policy=str(POLICY),model_bindings=MODELS,
        expected_model_hash=MODEL_TENSOR_SHA,bindings_sha256=bindings,
        scientific_pass_threshold=None,global_freshness='unknown',sourcepack_admission='external MAIN attestation only; not independently verified by deployment')


def prepare(output,**arguments):
    p=build(**arguments); raw=canonical(p); output=Path(output).absolute()
    require(len(raw)<256*1024 and not output.exists(),'exclusive small prepare')
    require(shutil.disk_usage(output.parent).free>=FLOOR+len(raw),'prepare floor')
    with output.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(output),prepare_sha256=sha(output),bytes=len(raw),contract=p['contract'])


def verify(prepared,expected):
    require(sha(prepared)==expected,'external prepare SHA'); p=read(prepared)
    require(p==build(**p['arguments']),'complete whitelist/blind/model/source reconstruction'); return p


def task_export(p,prepare_sha,slot,packet,values):
    require(len(values)==4,'all three history plus goal, no duplicate elision')
    for z in values: native(z)
    require([digest(rgb.tobytes()) for rgb in [*packet['rgb_history'],packet['goal_rgb']]]==slot['pixel_sha256'] and
            digest(packet['raw_action_history'].tobytes())==slot['raw_action_sha256'],'exact source frame/action alignment')
    raw=packet['raw_action_history']; norm=p['normalization']
    tokens=(raw.reshape(2,10)-np.tile(np.asarray(norm['mean'],np.float32),5))/np.tile(np.asarray(norm['std'],np.float32),5)
    task_contract=dict(schema=NAME+'.task_contract',prepare_sha256=prepare_sha,whitelist_sha=p['arguments']['whitelist_sha'],
        with_target=False,source_slot=slot['source_slot'],blind_source_sha256=slot['blind_sha256'],
        observed_steps=[0,5,10],raw_between_steps=10,normalization=norm,model_bindings=p['model_bindings'],metric='identity_native192_no_fit')
    metric_id=core.digest(dict(metric='identity_native192',rank=192,no_fit=True,with_target=False))
    task=dict(environment='swm/PushT-v1',task_contract_sha256=core.digest(task_contract),
        model_sha256=MODELS[str(POLICY/'weights.pt')],model_config_sha256=MODELS[str(POLICY/'config.json')],
        metric_source_sha256=metric_id,goal_source_sha256=slot['pixel_sha256'][3],
        representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=norm['mean'],action_std=norm['std'],packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only',
        projection=np.eye(192).tolist(),goal=values[3].astype(float).tolist(),root_raw_action_history=raw.astype(float).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                          latent_history=[z.astype(float).tolist() for z in values[:3]],action_history=tokens.astype(float).tolist()))
    core.validate_task(task); ident=core.task_id(task)  # True hash of actual new arrays/contract, not an old task ID.
    return dict(schema=NAME+'.task_export',task=task,task_id=ident,task_contract=task_contract,
        blind_source_sha256=slot['blind_sha256'],native_root_sha256=None,source_slot=slot['source_slot'],with_target=False,blind_sha256=slot['blind_sha256'],
        history_semantics='observed_not_padding_even_when_numerically_identical',source_admission='MAIN attested, not reverified',
        global_freshness='unknown',P1_admission=False,new_independent_confirmation=False)


def task_interfaces(export):
    """Direct usable arrays; caller must separately meter/authorize fresh CUDA search."""
    require(export['schema']==NAME+'.task_export' and export['with_target'] is False and export['task_contract']['with_target'] is False,'new no-target task only')
    task=export['task']; core.validate_task(task)
    require(core.task_id(task)==export['task_id'] and core.digest(export['task_contract'])==task['task_contract_sha256'],'actual task seal')
    require(export['task_contract']['blind_source_sha256']==export['blind_source_sha256']==export['blind_sha256'] and export['native_root_sha256'] is None,'blind identity is NOT a physical root hash')
    return dict(gn_task=deepcopy(task),d2p_packet=dict(root_context=deepcopy(task['root_context']),goal=deepcopy(task['goal'])),
                q=deepcopy(task['projection']),task_id=export['task_id'],blind_source_sha256=export['blind_source_sha256'],with_target=False)


def load_task(path,expected):
    require(sha(path)==expected,'export task external SHA'); return task_interfaces(read(path))


def native(z):
    require(isinstance(z,np.ndarray) and z.dtype==np.float32 and z.shape==(192,) and np.isfinite(z).all(),'raw native float32 CLS192')


class Store:
    def __init__(self,output):
        self.output=Path(output).absolute(); require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'exclusive2MiB +200MiB reserve')
        self.output.mkdir()
    def put(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..'),'flat exclusive output')
        used=sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'global byte cap/floor incl partial writes')
        with (self.output/name).open('ab' if append else 'xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    def event(self,e): self.put('image.events.jsonl',canonical(e),append=True)
    def json(self,name,value,final=False):
        raw=canonical(value); self.put(name,gzip.compress(raw,mtime=0) if name.endswith('.gz') else raw,final=final)
    def latent(self,index,z):
        native(z); stream=BytesIO(); np.save(stream,z,allow_pickle=False); name=f'image_{index:02d}.native192.npy'; self.put(name,stream.getvalue())
        path=self.output/name
        with path.open('rb') as f: require(np.array_equal(np.load(f,allow_pickle=False),z),'NPY exact readback')
        return dict(path=str(path),sha256=sha(path),dtype='float32',shape=[192])


class Ledger:
    def __init__(self,n,sink,sync=lambda:None,no_grad=lambda:True):
        self.n,self.sink,self.sync,self.no_grad=n,sink,sync,no_grad; self.image=None; self.entries=[]; self.blocked=[]
    def dispatch(self,kind,tensor,callback,*args,**kwargs):
        expected=[(i,k) for i in range(4*self.n) for k in ('encoder','projector')]
        require(len(self.entries)<len(expected) and (self.image,kind)==expected[len(self.entries)],'fixed ALL4n image order/cap')
        require(self.no_grad() and all(e['completed'] for e in self.entries),'inference only/no retry after failure')
        require(getattr(getattr(tensor,'device',None),'type',None)=='cuda' and tuple(tensor.shape)==((1,3,224,224) if kind=='encoder' else (1,192)),'CUDA batch1 native module input')
        e=dict(index=len(self.entries),image=self.image,resource=kind,completed=False); self.entries.append(e)
        self.sink(dict(kind='module_attempt',entry=deepcopy(e)))
        try:
            result=callback(*args,**kwargs); self.sync(); e['completed']=True
            self.sink(dict(kind='module_return',entry=deepcopy(e))); return result
        except BaseException as exc:
            e['error']=error('module',exc)
            try: self.sink(dict(kind='module_error',entry=deepcopy(e)))
            except BaseException as secondary: exc.add_note('error journal: '+str(secondary))
            raise
    def forbid(self,name):
        def denied(*args,**kwargs):
            self.blocked.append(name); self.sink(dict(kind='forbidden',name=name)); raise ValueError('forbidden '+name)
        return denied
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in ('encoder','projector')}


@contextmanager
def guard(model,torch,ledger,errors):
    undo=[]
    def patch(obj,name,value):
        old=getattr(obj,name); undo.append((obj,name,old)); setattr(obj,name,value)
    try:
        for kind,module in (('encoder',model.encoder),('projector',model.projector)):
            old=module.forward
            def forward(*args,_kind=kind,_old=old,**kwargs): return ledger.dispatch(_kind,args[0],_old,*args,**kwargs)
            patch(module,'forward',forward)
        for obj,name in ((model,'predict'),(model.predictor,'forward'),(model.action_encoder,'forward'),(model.pred_proj,'forward'),
                         (torch.autograd,'grad'),(torch.autograd,'backward'),(torch.Tensor,'backward')): patch(obj,name,ledger.forbid(name))
        for obj,names in ((torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
                          (torch.func,('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'))):
            for name in names: patch(obj,name,ledger.forbid(name))
        yield
    finally:
        for obj,name,old in reversed(undo):
            try: setattr(obj,name,old)
            except BaseException as exc: errors.append(error('restore_'+name,exc))


def encode_all(p,prepare_sha,encode,ledger,store,rows,exports):
    """Callback receives only ONE RGB ndarray. Not packet/slot/state/witness."""
    for slot in p['slots']:
        row=dict(order=slot['order'],blind_source_sha256=slot['blind_sha256'],status='attempted',images=[]); rows.append(row)
        packet=load_blind(slot['blind_npz'],slot['blind_sha256']); values=[]
        for offset,rgb in enumerate([*packet['rgb_history'],packet['goal_rgb']]):
            image_index=4*slot['order']+offset; ledger.image=image_index
            store.event(dict(kind='image_attempt',index=image_index,root_order=slot['order'],role=['observed0','observed5','observed10','goal'][offset],pixel_sha256=slot['pixel_sha256'][offset]))
            z=encode(rgb.copy()); native(z)
            require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==image_index]==[('encoder',True),('projector',True)],'actual encoder/projector return per image')
            latent=store.latent(image_index,z); row['images'].append(latent); values.append(z.copy())
            store.event(dict(kind='image_saved',index=image_index,latent=latent)); ledger.image=None
        export=task_export(p,prepare_sha,slot,packet,values); task_interfaces(export)
        name=f'task_{slot["order"]:02d}.json.gz'; store.json(name,export)
        row.update(status='task_sealed',task_path=str(store.output/name),task_file_sha256=sha(store.output/name),task_id=export['task_id'])
        exports.append(export); store.event(dict(kind='task_sealed',row=deepcopy(row)))


def frozen(model):
    require(all(not m.training for m in model.modules()) and all(not v.requires_grad and v.device.type=='cuda' for v in model.parameters()) and
            all(b.device.type=='cuda' for b in model.buffers()),'frozen CUDA eval model')


def tensor_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous(); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
    return h.hexdigest()


def error(stage,exc): return dict(stage=stage,type=type(exc).__name__,message=str(exc)[:1500],notes=list(getattr(exc,'__notes__',[])))


def run(prepared,expected,output):
    p=verify(prepared,expected); store=Store(output); n=len(p['slots']); errors=[]; rows=[]; exports=[]
    ledger=Ledger(n,store.event); model=None; before=after=host=None; source_ok=False
    store.json('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    try:
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0'); torch.backends.cuda.matmul.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        frozen(model); before=tensor_hash(model); require(before==p['expected_model_hash'],'same original frozen tensor identity')
        transform=p0.image_transform(); ledger.sync=lambda:torch.cuda.synchronize(device); ledger.no_grad=lambda:not torch.is_grad_enabled()
        with guard(model,torch,ledger,errors):
            def encode(rgb):
                require(isinstance(rgb,np.ndarray) and rgb.dtype==np.uint8 and rgb.shape==(224,224,3),'encoder receives RGB only')
                with torch.inference_mode():
                    x=p0._image_tensor(rgb,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native pixels-only CUDA input')
                    z=model.encode({'pixels':x})['emb']
                    require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native projected float32 output')
                    return z[0,0].detach().cpu().numpy().copy()
            encode_all(p,expected,encode,ledger,store,rows,exports)
    except BaseException as exc:
        errors.insert(0,error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(error('error_journal',secondary))
    finally:
        ledger.image=None
        if rows and rows[-1]['status']=='attempted': rows[-1]['status']='failed_partial'
        if model is not None:
            try: frozen(model); after=tensor_hash(model)
            except BaseException as exc: errors.append(error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(error('final_source',exc))
        for slot in p['slots'][len(rows):]: rows.append(dict(order=slot['order'],blind_source_sha256=slot['blind_sha256'],status='blocked_not_attempted',images=[]))
        counts=ledger.counts(); complete=(not errors and not ledger.blocked and len(exports)==n and source_ok and before is not None and before==after and
                                         counts=={k:dict(attempted=4*n,completed=4*n) for k in counts})
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as exc: errors.append(error('artifact_hash',exc)); complete=False
        result=dict(schema=NAME+'.report',status='blind_tasks_encoded_and_sealed' if complete else 'partial_or_failed',prepare_sha256=expected,
            rows=rows,resources=counts,errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,with_target=False,sourcepack_admission='MAIN external attestation, not verified here',global_freshness='unknown',
            new_predict=0,new_actionencoder=0,new_AD=0,new_physics=0,training=False,P1_admission=False)
        try: store.json('report.json',result,final=True)
        except BaseException as exc:
            errors.append(error('final_report',exc)); complete=False; print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial blind encoding preserved; no retry: '+str(errors)); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('approved-whitelist','whitelist-sha','prepared','prepare-sha','output'): parser.add_argument('--'+name)
    a=parser.parse_args()
    result=prepare(a.output,approved_whitelist=a.approved_whitelist,whitelist_sha=a.whitelist_sha) if a.command=='prepare' else (
        dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots'])) if a.command=='verify' else run(a.prepared,a.prepare_sha,a.output))
    print(canonical(result).decode().strip())
