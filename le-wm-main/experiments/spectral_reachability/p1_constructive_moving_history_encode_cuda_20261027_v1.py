"""Two MAIN-approved blind moving histories -> native H2 tasks; HOST CUDA only.

Protocol test, not a claim that moving history fixes a planner. No sourcepack,
certificate, state, witness or outcome reader. Prepare/tests need no torch.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as low

NAME='p1_constructive_moving_history_encode_cuda_20261027_v1'
ROOT,core=low.ROOT,low.core
require,sha,canonical,digest,read=low.require,low.sha,low.canonical,low.digest,low.read
MODELS,POLICY=low.MODELS,low.POLICY
CAP,FLOOR,FINAL=low.CAP,low.FLOOR,low.FINAL
Store,Ledger=low.Store,low.Ledger
ROLES=['observed0','observed5','observed10','goal']


def fixed_raw():
    return np.asarray([[.1,0]]*5+[[0,.1]]*5,dtype=np.float32)


def normalization():
    return dict(mean=np.asarray([-.0087,.0068],np.float32).tolist(),
        std=np.asarray([.2019,.2002],np.float32).tolist(),source_sha256=low.NORMALIZER_SHA,
        training_provenance_verified=False,raw_semantics='relative raw already action_scale/100; no second division')


def check_packet(values):
    core.fields(values,'rgb_history raw_action_history goal_rgb','blind-only arrays')
    r,g,a=values['rgb_history'],values['goal_rgb'],values['raw_action_history']
    require(isinstance(r,np.ndarray) and r.dtype==np.uint8 and r.shape==(3,224,224,3),'three actual RGB224 history frames')
    require(isinstance(g,np.ndarray) and g.dtype==np.uint8 and g.shape==(224,224,3),'goal RGB224')
    require(isinstance(a,np.ndarray) and a.dtype==np.float32 and a.shape==(10,2) and
        np.array_equal(a,fixed_raw()),'exact actual float32 moving raw10, not zero padding')


def decode_blind(raw):
    require(len(raw)<=2*1024**2,'bounded NPZ input')
    with zipfile.ZipFile(BytesIO(raw)) as z:
        es=z.infolist();require(len(es)==3 and {e.filename for e in es}==
            {'rgb_history.npy','raw_action_history.npy','goal_rgb.npy'},'exact blind-only NPZ members')
        require(all(e.file_size<=512*1024 and not e.flag_bits&1 and e.compress_type in
            (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) for e in es) and sum(e.file_size for e in es)<1024**2,
            'bounded unencrypted expansion')
    with np.load(BytesIO(raw),allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    check_packet(values);return values


def load_blind(path,expected):
    core.hash_value(expected);path=Path(path).absolute()
    require(path.name.endswith('_blind_inputs.npz') and path.resolve()==path and
        path.stat().st_size<=2*1024**2,'explicit regular blind NPZ only')
    require(sha(path)==expected,'external blind SHA');raw=path.read_bytes()
    require(digest(raw)==expected and sha(path)==expected,'blind read TOCTOU')
    return decode_blind(raw)


def manifest_slots(m,base):
    core.fields(m,'schema approved_by purpose with_target history_steps raw_action_steps slots','MAIN manifest')
    require(m['schema']==NAME+'.approved_blind_manifest' and m['approved_by']=='MAIN' and
        m['purpose']=='calibration' and m['with_target'] is False and m['history_steps']==[0,5,10] and
        all(type(x) is int for x in m['history_steps']) and type(m['raw_action_steps']) is int and
        m['raw_action_steps']==10 and isinstance(m['slots'],list) and len(m['slots'])==2,
        'fixed two-slot actual moving-history contract')
    rows=[]
    for i,s in enumerate(m['slots']):
        core.fields(s,'order source_slot blind_npz blind_sha256','approved blind slot')
        require(type(s['order']) is int and type(s['source_slot']) is int and s['order']==s['source_slot']==i,
            'fixed literal slots0 then1; no duplicates/replacements')
        core.hash_value(s['blind_sha256'])
        require(type(s['blind_npz']) is str and Path(s['blind_npz']).is_absolute(),'MAIN absolute blind NPZ path')
        rows.append(deepcopy(s))
    require(len({s['blind_npz'] for s in rows})==len({s['blind_sha256'] for s in rows})==2,'distinct approved NPZ identities')
    return rows


def contract():
    return dict(with_target=False,roots=2,images_per_root=4,encoder_cap=8,projector_cap=8,
        image_order=ROLES,history='actual states0/5/10 with moving raw packets; never padding',
        horizon=2,packet_steps=5,raw_action_steps=10,representation='native_projected_cls192',
        metric='identity_native192_no_fit',batch=1,predict_cap=0,actionencoder_cap=0,AD_cap=0,physics_cap=0,
        training=False,no_cpu_neural_fallback=True,no_retries=True,no_dedup=True,P1_admission=False,
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),output_cap_bytes=CAP,reserve_bytes=FLOOR,
        source_admission='MAIN manifest attestation only; actual execution/replay not independently established here')


def source_files():
    return sorted(set(low.source_files()+[Path(low.__file__).resolve(),Path(__file__).resolve(),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]))


def build(approved_manifest,manifest_sha):
    core.hash_value(manifest_sha);path=Path(approved_manifest).absolute()
    require(sha(path)==manifest_sha,'MAIN external manifest SHA');m=read(path);slots=manifest_slots(m,path.parent)
    bindings={str(path):manifest_sha,**MODELS}
    for s in slots:
        packet=load_blind(s['blind_npz'],s['blind_sha256'])
        s.update(pixel_sha256=[digest(a.tobytes()) for a in [*packet['rgb_history'],packet['goal_rgb']]],
            raw_action_sha256=digest(packet['raw_action_history'].tobytes()))
        bindings[s['blind_npz']]=s['blind_sha256']
    norm=low.native_metadata();require(norm==normalization(),'original raw normalizer bytes and constants')
    bindings.update({str(f.resolve()):sha(f) for f in source_files()})
    for f,h in bindings.items():require(sha(f)==h,'prepare source/input/model TOCTOU')
    return dict(schema=NAME+'.prepare',arguments=dict(approved_manifest=str(path),manifest_sha=manifest_sha),
        slots=slots,normalization=norm,model_bindings=MODELS,policy=str(POLICY),expected_model_hash=low.MODEL_TENSOR_SHA,
        bindings_sha256=bindings,contract=contract(),scientific_pass_threshold=None,global_freshness='unknown')


def prepare(output,**arguments):
    p=build(**arguments);raw=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(raw)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),
        'exclusive small prepare +200MiB reserve')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(prepared,expected):
    core.hash_value(expected);require(sha(prepared)==expected,'external prepare SHA');p=read(prepared)
    require(p==build(**p['arguments']),'entire moving manifest/input/model/source reconstruction');return p


def task_export(p,prepare_sha,slot,packet,values):
    check_packet(packet);require(len(values)==4,'four actual encodings, no dedup')
    for z in values:low.native(z)
    require([digest(a.tobytes()) for a in [*packet['rgb_history'],packet['goal_rgb']]]==slot['pixel_sha256'] and
        digest(packet['raw_action_history'].tobytes())==slot['raw_action_sha256'],'source RGB/raw alignment')
    require(p['normalization']==normalization() and p['model_bindings']==MODELS,'original model/normalizer')
    tc=dict(schema=NAME+'.task_contract',prepare_sha256=core.hash_value(prepare_sha),
        manifest_sha256=core.hash_value(p['arguments']['manifest_sha']),with_target=False,
        source_slot=slot['source_slot'],blind_sha256=slot['blind_sha256'],observed_steps=[0,5,10],
        normalization=normalization(),model_bindings=MODELS,metric='identity_native192_no_fit',
        pixel_sha256=slot['pixel_sha256'],raw_action_sha256=slot['raw_action_sha256'])
    task=dict(environment='swm/PushT-v1',task_contract_sha256=core.digest(tc),
        model_sha256=MODELS[str(POLICY/'weights.pt')],model_config_sha256=MODELS[str(POLICY/'config.json')],
        metric_source_sha256=core.digest(dict(metric='identity_native192',rank=192,no_fit=True,with_target=False)),
        goal_source_sha256=slot['pixel_sha256'][3],representation='native_projected_cls',latent_dim=192,horizon=2,
        packet_steps=5,control_dim=2,action_mean=normalization()['mean'],action_std=normalization()['std'],
        packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only',projection=np.eye(192).tolist(),
        goal=values[3].astype(float).tolist(),root_raw_action_history=packet['raw_action_history'].astype(float).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
            action_dim=10,latent_history=[z.astype(float).tolist() for z in values[:3]],
            action_history=core.normalized_tokens(dict(action_mean=normalization()['mean'],action_std=normalization()['std']),packet['raw_action_history']).astype(float).tolist()))
    export=dict(schema=NAME+'.task_export',task=task,task_id=core.task_id(task),task_contract=tc,
        source_slot=slot['source_slot'],blind_sha256=slot['blind_sha256'],blind_source_sha256=slot['blind_sha256'],
        native_root_sha256=None,with_target=False,history_semantics='actual_moving_raw_between_observed0_5_10',
        source_admission='MAIN_manifest_only_not_independently_verified',global_freshness='unknown',P1_admission=False)
    task_interfaces(export);return export


def task_interfaces(e):
    core.fields(e,'schema task task_id task_contract source_slot blind_sha256 blind_source_sha256 native_root_sha256 with_target history_semantics source_admission global_freshness P1_admission','moving export envelope')
    require(e['schema']==NAME+'.task_export' and e['with_target'] is False and e['native_root_sha256'] is None and
        e['P1_admission'] is False and type(e['source_slot']) is int and e['source_slot'] in (0,1) and
        e['history_semantics']=='actual_moving_raw_between_observed0_5_10' and e['global_freshness']=='unknown' and
        e['source_admission']=='MAIN_manifest_only_not_independently_verified','own blind moving export only')
    tc=e['task_contract'];t=e['task']
    core.fields(tc,'schema prepare_sha256 manifest_sha256 with_target source_slot blind_sha256 observed_steps normalization model_bindings metric pixel_sha256 raw_action_sha256','moving task contract')
    for h in [e['blind_sha256'],e['blind_source_sha256'],tc['prepare_sha256'],tc['manifest_sha256'],tc['blind_sha256']]:core.hash_value(h)
    require(tc['schema']==NAME+'.task_contract' and tc['with_target'] is False and type(tc['source_slot']) is int and
        tc['source_slot']==e['source_slot'] and tc['blind_sha256']==e['blind_sha256']==e['blind_source_sha256'] and
        tc['observed_steps']==[0,5,10] and all(type(x) is int for x in tc['observed_steps']) and
        canonical(tc['normalization'])==canonical(normalization()) and tc['model_bindings']==MODELS and
        tc['metric']=='identity_native192_no_fit' and isinstance(tc['pixel_sha256'],list) and len(tc['pixel_sha256'])==4,
        'strict source/normalizer/model contract')
    for h in tc['pixel_sha256']:core.hash_value(h)
    require(tc['raw_action_sha256']==digest(fixed_raw().tobytes()),'fixed raw byte identity')
    core.validate_task(t)
    require(core.task_id(t)==e['task_id'] and t['task_contract_sha256']==core.digest(tc) and
        t['environment']=='swm/PushT-v1' and t['goal_source_sha256']==tc['pixel_sha256'][3] and
        t['model_sha256']==MODELS[str(POLICY/'weights.pt')] and t['model_config_sha256']==MODELS[str(POLICY/'config.json')] and
        t['metric_source_sha256']==core.digest(dict(metric='identity_native192',rank=192,no_fit=True,with_target=False)) and
        np.array_equal(t['projection'],np.eye(192)) and t['packet_logit_cap']==float(.75*np.sqrt(5)) and
        np.array_equal(t['action_mean'],normalization()['mean']) and np.array_equal(t['action_std'],normalization()['std']) and
        np.array_equal(t['root_raw_action_history'],fixed_raw().astype(float)) and
        np.array_equal(t['root_context']['action_history'],core.normalized_tokens(t,fixed_raw())),
        'native identityQ H2 task/seals/actual normalized history consistency')
    # Saved native latent arrays must remain exactly float32-representable.
    a=np.asarray(t['root_context']['latent_history']+[t['goal']],dtype=float)
    require(np.isfinite(a).all() and np.array_equal(a,a.astype(np.float32).astype(float)),'original native float32 latent values')
    return dict(gn_task=deepcopy(t),d2p_packet=dict(root_context=deepcopy(t['root_context']),goal=deepcopy(t['goal'])),
        q=deepcopy(t['projection']),task_id=e['task_id'])


def load_task(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external task export SHA');e=read(path)
    result=task_interfaces(e);require(sha(path)==expected,'task export read TOCTOU');return result


def encode_all(p,prepare_sha,encode,ledger,store,rows,exports):
    for slot,row in zip(p['slots'],rows,strict=True):
        row['status']='attempted';packet=load_blind(slot['blind_npz'],slot['blind_sha256']);values=[]
        for offset,a in enumerate([*packet['rgb_history'],packet['goal_rgb']]):
            index=4*slot['order']+offset;ledger.image=index
            require(digest(a.tobytes())==slot['pixel_sha256'][offset],'prepared image pixel identity')
            store.event(dict(kind='image_attempt',index=index,source_slot=slot['source_slot'],role=ROLES[offset],pixel_sha256=slot['pixel_sha256'][offset]))
            z=encode(a.copy());low.native(z)
            require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==index]==
                [('encoder',True),('projector',True)],'actual two native module returns')
            latent=store.latent(index,z);row['images'].append(latent);values.append(z.copy())
            store.event(dict(kind='image_saved',index=index,latent=latent));ledger.image=None
        e=task_export(p,prepare_sha,slot,packet,values);name=f'task_{slot["order"]:02d}.json.gz'
        store.json(name,e);path=store.output/name;h=sha(path)
        require(load_task(path,h)==task_interfaces(e),'exact export gzip readback')
        row.update(status='task_sealed',task_path=str(path),task_file_sha256=h,task_id=e['task_id'])
        exports.append(e);store.event(dict(kind='task_sealed',row=deepcopy(row)))


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];exports=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(order=s['order'],source_slot=s['source_slot'],blind_source_sha256=s['blind_sha256'],status='blocked_not_attempted',images=[]) for s in p['slots']]
    ledger=Ledger(2,store.event);complete=False
    try:
        store.json('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        low.frozen(model);before=low.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model tensor SHA')
        transform=p0.image_transform();ledger.sync=lambda:torch.cuda.synchronize(device);ledger.no_grad=lambda:not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH),low.guard(model,torch,ledger,errors):
            def encode(a):
                require(a.dtype==np.uint8 and a.shape==(224,224,3),'RGB-only model input')
                with torch.inference_mode():
                    x=p0._image_tensor(a,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA pixels batch1')
                    z=model.encode({'pixels':x})['emb']
                    require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native float32 projected CLS192')
                    return z[0,0].detach().cpu().numpy().copy()
            encode_all(p,expected,encode,ledger,store,rows,exports)
    except BaseException as exc:
        errors.insert(0,low.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(low.error('error_journal',e))
    finally:
        ledger.image=None
        for r in rows:
            if r['status']=='attempted':r['status']='failed_partial'
        if model is not None:
            try:low.frozen(model);after=low.tensor_hash(model)
            except BaseException as e:errors.append(low.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(low.error('final_source',e))
        counts=ledger.counts();complete=not errors and not ledger.blocked and len(exports)==2 and source_ok and before is not None and before==after and all(r['status']=='task_sealed' for r in rows) and counts=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try:hashes[f.name]=sha(f)
                except BaseException as e:errors.append(low.error('artifact_hash',e));complete=False
        result=dict(schema=NAME+'.report',status='moving_blind_tasks_encoded_and_sealed' if complete else 'partial_or_failed',
            prepare_sha256=expected,arguments=p['arguments'],contract=p['contract'],rows=rows,resources=counts,
            errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,with_target=False,new_predict=0,new_actionencoder=0,new_AD=0,new_physics=0,training=False,P1_admission=False)
        try:store.json('report.json',result,final=True)
        except BaseException as e:errors.append(low.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial moving encoding retained; no retry: '+str(errors));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for k in ('approved-manifest','manifest-sha','prepared','prepare-sha','output'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output,approved_manifest=a.approved_manifest,manifest_sha=a.manifest_sha)
    elif a.command=='verify':r=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
