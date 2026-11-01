"""Four paired blind image packets -> eight NEW frozen-kernel CUDA plans. MAIN only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
from io import BytesIO
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as blind
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving
from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as history

NAME='p1_moving_overlay_recompile_cuda_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/p1_moving_target_overlay_source_20261027_v1_main01/report.json'
SOURCE_SHA='85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a'
BLIND_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'
MOVING_SHA='246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'
PREREG=ROOT.parent/'docs/D2P_MOVING_OVERLAY_RECOMPILE_PROTOCOL_20261027.md'
PREREG_SHA='242e3277831ce3b5ac8db7d1da27d384e02a2485a218911bb3dbe80a0db7af95'
ARMS=blind.ARMS
CONDITIONS=('off','on')
KEYS=tuple((s,c,a) for s in (0,1) for c in CONDITIONS for a in ARMS)
CAP,FLOOR,FINAL,WALL=64*1024**2,200*1024**2,4*1024**2,900
core=blind.encode.core
require,sha,canonical,read=blind.require,blind.sha,blind.canonical,blind.read
hash_bytes=moving.low.digest


def encoder_output_shape(config):
    e=config['encoder']
    require(type(e['image_size']) is int and type(e['patch_size']) is int and
        e['image_size']==224 and e['patch_size']==14 and config['projector']['input_dim']==192,
        'frozen224/patch14/native192 checkpoint metadata, not arbitrary ViT')
    require(e['image_size']%e['patch_size']==0,'whole patch grid')
    tokens=(e['image_size']//e['patch_size'])**2+1
    require(tokens==257,'frozen16x16 patches plus CLS, not patch16/197')
    return [1,tokens,192]


def visual_bridge(pre,ein,eout,pin,pout,z,shape):
    require(list(shape)==[1,257,192] and pre.shape==(1,1,3,224,224) and
        np.array_equal(ein,pre.reshape(1,3,224,224)) and list(eout.shape)==list(shape) and
        np.array_equal(pin,eout[:,0]) and pout.shape==(1,192) and np.array_equal(z,pout[0]),
        'real native patch14/257 preprocessing/CLS/projector output bridge')


def contract():
    return dict(roots=2,conditions=list(CONDITIONS),arms=list(ARMS),plans=8,images=16,
        encoder=16,projector=16,predict=576,action_encoder=576,pred_proj=576,
        per_root_condition=blind.budgets(),AD_API_per_root_condition=blind.contract(1)['AD_API_per_root'],
        horizon=2,packet_steps=5,metric='identity_native192_no_fit',normalization=moving.normalization(),
        renderer='only with_target off/on; native default outline goal_pose is NOT constructed task goal',
        edit_goal_pose=False,edit_state=False,edit_target=False,training=False,Qfit=False,P1_admission=False,
        no_retry=True,no_dedup=True,no_CPU_neural=True,new_physics=0,
        numerics=dict(MATH=True,TF32=False,autocast=False,batch=1),
        wall_seconds=WALL,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL)


def flat(root,name):
    require(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat artifact path')
    return Path(root)/name


def packets(report,base,load):
    """Blind-only callback boundary; never decode witness, native outcome or old actions."""
    require(report['schema']=='p1_moving_target_overlay_source_20261027_v1.report' and
        report['status']=='paired_moving_overlay_recorded' and report['full_protocol'] is True and
        report['source_verified'] is True and report['errors']==[] and report['first_error'] is None and
        report['fixed_denominator']==2 and report['expected_episodes']==6 and report['expected_frames']==186,
        'complete pinned paired source report')
    c=report['contract']
    require(c['flags']==[False,True,False] and c['no_goal_pose_edit'] is True and c['warmup']==c['horizon']==10 and
        c['slots']==2 and c['source_triplets']==[0,5,10,15,20], 'original only-renderer intervention')
    require(report['counts']=={k:dict(attempted=v,completed=v) for k,v in c['caps'].items()},'full source accounting')
    require(len(report['blind_exports'])==2 and [x['source_slot'] for x in report['blind_exports']]==[0,1] and
        all(type(x['source_slot']) is int for x in report['blind_exports']),'fixed two roots, no outcome selection')
    slots=[]
    for export in report['blind_exports']:
        s=export['source_slot'];require(set(export['artifacts'])==set(CONDITIONS),'both real conditions')
        for condition in CONDITIONS:
            f=export['artifacts'][condition];name=f's{s}_{condition}_blind_inputs.npz'
            require(f['path']==name and f['fields']==['rgb_history','raw_action_history','goal_rgb'],'own literal blind NPZ')
            path=flat(base,name).absolute();a=load(path,f['sha256']);moving.check_packet(a)
            pixels=[hash_bytes(x.tobytes()) for x in (*a['rgb_history'],a['goal_rgb'])]
            require(pixels==f['rgb_history_pixel_sha256']+[f['goal_pixel_sha256']] and
                hash_bytes(a['raw_action_history'].tobytes())==f['raw_action_sha256'],'all original NPZ pixel/raw SHAs')
            slots.append(dict(order=len(slots),root_slot=s,condition=condition,blind_path=str(path),
                blind_sha256=f['sha256'],pixel_sha256=pixels,raw_sha256=f['raw_action_sha256']))
    require(len({x['blind_path'] for x in slots})==len({x['blind_sha256'] for x in slots})==4,'four distinct NPZ identities')
    return slots


def load_packet(path,expected):
    require(Path(path).is_absolute() and Path(path).resolve()==Path(path) and not Path(path).is_symlink(),'literal absolute blind file')
    require(sha(path)==expected,'blind external SHA');raw=Path(path).read_bytes()
    require(hash_bytes(raw)==expected,'blind read SHA');return moving.decode_blind(raw)


def source_files():
    return sorted(set(blind.source_files()+moving.source_files()+[PREREG,Path(history.__file__),Path(__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]))


def build():
    require(sha(blind.__file__)==BLIND_SHA and sha(moving.__file__)==MOVING_SHA,'frozen planner and packet primitives')
    require(sha(PREREG)==PREREG_SHA,'MAIN preregistration before new experiments')
    require(sha(SOURCE)==SOURCE_SHA,'MAIN exact paired source authority');r=read(SOURCE)
    slots=packets(r,SOURCE.parent,load_packet)
    require(blind.encode.native_metadata()==moving.normalization(),'original frozen model/normalizer')
    shape=encoder_output_shape(read(moving.POLICY/'config.json'))
    bindings={str(p.resolve()):sha(p) for p in source_files()}
    bindings.update(moving.MODELS);bindings[str(SOURCE)]=SOURCE_SHA
    bindings.update({s['blind_path']:s['blind_sha256'] for s in slots})
    for path,h in bindings.items():require(sha(path)==h,'final input/source/model seal')
    return dict(schema=NAME+'.prepare',slots=slots,contract=contract(),bindings=bindings,
        source_report=str(SOURCE),source_report_sha256=SOURCE_SHA,policy=str(moving.POLICY),
        expected_model_hash=moving.low.MODEL_TENSOR_SHA,encoder_output_shape=shape,preregistration_sha256=PREREG_SHA)


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(raw)<=256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive bounded prepare')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external prepare SHA');p=read(path,256*1024)
    require(p==build(),'entire own conditional packet/source/model reconstruction');return p


def task_export(p,prepare_sha,slot,packet,values):
    moving.check_packet(packet);require(len(values)==4,'own four encodings, no off/on reuse')
    for z in values:moving.low.native(z)
    require([hash_bytes(x.tobytes()) for x in (*packet['rgb_history'],packet['goal_rgb'])]==slot['pixel_sha256'],'own condition images')
    tc=dict(schema=NAME+'.task_contract',prepare_sha256=prepare_sha,source_report_sha256=SOURCE_SHA,
        root_slot=slot['root_slot'],condition=slot['condition'],with_target=slot['condition']=='on',
        blind_path=slot['blind_path'],blind_sha256=slot['blind_sha256'],pixel_sha256=slot['pixel_sha256'],
        raw_sha256=hash_bytes(packet['raw_action_history'].tobytes()),metric='identity_native192_no_fit',
        normalization=moving.normalization(),renderer_only=True,goal_pose_edited=False)
    t=dict(environment='swm/PushT-v1',task_contract_sha256=core.digest(tc),
        model_sha256=moving.MODELS[str(moving.POLICY/'weights.pt')],model_config_sha256=moving.MODELS[str(moving.POLICY/'config.json')],
        metric_source_sha256=core.digest(dict(metric='identity_native192',rank=192,no_fit=True)),
        goal_source_sha256=slot['pixel_sha256'][3],representation='native_projected_cls',latent_dim=192,horizon=2,
        packet_steps=5,control_dim=2,action_mean=moving.normalization()['mean'],action_std=moving.normalization()['std'],
        packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only',projection=np.eye(192).tolist(),
        goal=values[3].astype(float).tolist(),root_raw_action_history=packet['raw_action_history'].astype(float).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=[z.astype(float).tolist() for z in values[:3]],
            action_history=core.normalized_tokens(dict(action_mean=moving.normalization()['mean'],action_std=moving.normalization()['std']),packet['raw_action_history']).tolist()))
    e=dict(schema=NAME+'.task_export',task=t,task_contract=tc,task_id=core.task_id(t),P1_admission=False)
    task_interface(e);return e


def task_interface(e):
    core.fields(e,'schema task task_contract task_id P1_admission','own conditional export')
    t,c=e['task'],e['task_contract']
    require(e['schema']==NAME+'.task_export' and c['schema']==NAME+'.task_contract' and e['P1_admission'] is False and
        type(c['root_slot']) is int and c['root_slot'] in (0,1) and c['condition'] in CONDITIONS and
        c['with_target'] is (c['condition']=='on') and c['renderer_only'] is True and c['goal_pose_edited'] is False,
        'own genuine off/on interface; never call old off-only export loader')
    for h in (c['prepare_sha256'],c['blind_sha256'],e['task_id']):core.hash_value(h)
    require(c['source_report_sha256']==SOURCE_SHA and Path(c['blind_path']).is_absolute() and
        c['raw_sha256']==hash_bytes(moving.fixed_raw().tobytes()) and c['normalization']==moving.normalization() and
        t['task_contract_sha256']==core.digest(c) and t['goal_source_sha256']==c['pixel_sha256'][3] and
        c['metric']=='identity_native192_no_fit','original source/raw/Q/normalizer identity')
    return history.interface(dict(gn_task=deepcopy(t),d2p_packet=dict(root_context=deepcopy(t['root_context']),goal=deepcopy(t['goal'])),
        q=deepcopy(t['projection']),task_id=e['task_id']))


class Store(blind.Store):
    def __init__(self,output,check=lambda:None):
        self.output=Path(output).absolute();self.check=check
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW64MiB plus200MiB floor')
        self.output.mkdir();self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        path=flat(self.output,name);require(not path.is_symlink(),'no output symlink')
        if not final or name=='report.json':self.check()
        used=sum(f.stat().st_size for f in self.output.iterdir())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'64MiB cap/reserve/200MiB floor')
        with path.open('ab' if append else 'xb') as f:require(f.write(raw)==len(raw),'short artifact');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def array(self,name,a):
        a=np.ascontiguousarray(a);require(a.dtype in (np.float32,np.uint8) and np.isfinite(a).all(),'original finite bytes')
        stream=BytesIO();np.save(stream,a,allow_pickle=False);raw=gzip.compress(stream.getvalue(),mtime=0)
        name=name+'.npy.gz';self._write(name,raw)
        return dict(file=name,file_sha256=hash_bytes(raw),array_sha256=hash_bytes(a.tobytes()),dtype=str(a.dtype),shape=list(a.shape))


class NativeTrace:
    """Read-only hooks; original tensor/AD computation untouched, no global monkeypatch."""
    def __init__(self,model,torch,store,check,encoder_shape):
        self.model,self.torch,self.store,self.check=model,torch,store,check
        require(list(encoder_shape)==[1,257,192],'fixed checkpoint shape');self.encoder_shape=tuple(encoder_shape)
        self.entries=[];self.scope=None;self.pending={};self.hooks=[]
    def save(self,x,name):
        require(getattr(getattr(x,'device',None),'type',None)=='cuda' and x.dtype==self.torch.float32,'CUDA float32 native tensor only')
        return self.store.array(name,x.detach().cpu().contiguous().numpy().copy())
    def __enter__(self):
        def pre(kind):
            def hook(module,args):
                self.check();require(self.scope is not None and module not in self.pending,'scoped nonrecursive tensor trace')
                image='image_index' in self.scope
                require((kind in ('encoder','projector'))==image,'no visual/planner cross-phase calls')
                cap=16 if image else 576
                require(sum(e['resource']==kind for e in self.entries)<cap,'actual native predispatch cap')
                e=dict(index=len(self.entries),resource=kind,scope=deepcopy(self.scope),completed=False)
                self.entries.append(e);self.pending[module]=e
                self.store.event(dict(kind='tensor_module_attempt',entry=deepcopy(e)))
                e['inputs']=[self.save(x,f'module_{e["index"]:04}_input{j}') for j,x in enumerate(args)]
            return hook
        def post(module,args,out):
            e=self.pending.pop(module);require(out is not None,'actual module failed, no fabricated completion')
            if hasattr(out,'last_hidden_state'):out=out.last_hidden_state
            if e['resource']=='encoder':require(tuple(out.shape)==self.encoder_shape,'actual patch14 encoder257 tokens')
            e['output']=self.save(out,f'module_{e["index"]:04}_output')
            self.torch.cuda.synchronize();self.check();e['completed']=True
            self.store.event(dict(kind='tensor_module_completed',entry=deepcopy(e)))
        try:
            for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
                module=getattr(self.model,kind)
                self.hooks.append(module.register_forward_pre_hook(pre(kind)))
                self.hooks.append(module.register_forward_hook(post))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise
    def __exit__(self,typ,exc,tb):
        errors=[]
        for h in reversed(self.hooks):
            try:h.remove()
            except BaseException as e:errors.append(e)
        if errors:
            if exc is not None:exc.add_note('trace cleanup '+repr(errors))
            else:raise errors[0]
    def finish(self,scope):
        es=[e for e in self.entries if e['scope']==scope]
        require(not self.pending and all(e['completed'] for e in es),'all actual scoped module returns')
        return deepcopy(es)


@contextmanager
def traced_scope(trace,scope,receipt):
    trace.scope=scope;primary=None
    try:yield
    except BaseException as e:primary=e;raise
    finally:
        receipt['modules']=deepcopy([e for e in trace.entries if e['scope']==scope])
        try:trace.finish(scope)
        except BaseException as e:
            if primary is not None:primary.add_note('partial native trace: '+str(e))
            else:raise
        finally:trace.scope=None


def initial_rows():
    return [dict(index=j,root_slot=s,condition=c,arm=a,status='not_started',resources={},AD_API_resources={})
        for j,(s,c,a) in enumerate(KEYS)]


def execute(p,prepare_sha,store,encode,runner,rows,images,check=lambda:None,load=load_packet):
    require(len(rows)==8 and [(r['root_slot'],r['condition'],r['arm']) for r in rows]==list(KEYS),'fixed all8 denominator')
    store.put('initial_rows.json',rows)
    for slot in p['slots']:
        check();packet=load(slot['blind_path'],slot['blind_sha256']);values=[]
        for rgb in (*packet['rgb_history'],packet['goal_rgb']):
            image=dict(image_index=len(images),root_slot=slot['root_slot'],condition=slot['condition'],status='attempted')
            images.append(image);store.event(dict(kind='image_attempt',image=deepcopy(image)))
            image['pixels']=store.array(f'image_{image["image_index"]:02}_pixels',rgb)
            z=encode(rgb.copy(),image);moving.low.native(z)
            require([(e['resource'],e['completed']) for e in image['modules']]==[('encoder',True),('projector',True)],'exact own two visual module returns')
            image['latent']=store.array(f'image_{image["image_index"]:02}_latent',z)
            image['status']='encoded';values.append(z.copy());store.event(dict(kind='image_saved',image=deepcopy(image)))
        export=task_export(p,prepare_sha,slot,packet,values);i=task_interface(export)
        name=f'task_{slot["order"]:02}.json.gz';store.put(name,export)
        for arm in ARMS:
            check();row=rows[2*slot['order']+ARMS.index(arm)];row.update(status='attempted',task_id=i['task_id'],task_file=name)
            events=[]
            def sink(e):
                check();events.append(deepcopy(e));store.event(dict(plan_index=row['index'],event=e))
            result=runner(arm,i,sink,row);choice=blind.selected(arm,i,result)
            blind.exact_counts(row['resources'],blind.budgets()[arm])
            blind.exact_counts(row['AD_API_resources'],blind.contract(1)['AD_API_per_root'][arm])
            history.ledger_check(arm,events);row['history_verification']=history.history_check(arm,i,result,events)
            stem=f'plan_{row["index"]:02}';store.put(stem+'.result.json.gz',result)
            plan=dict(schema=NAME+'.plan',root_slot=slot['root_slot'],condition=slot['condition'],arm=arm,
                blind_path=slot['blind_path'],blind_sha256=slot['blind_sha256'],source_report_sha256=SOURCE_SHA,
                task_file=name,with_target=slot['condition']=='on',selection_uses_physics=False,P1_admission=False,**choice)
            plan['selection_digest']=core.digest(plan);store.put(stem+'.plan.json',plan)
            row.update(status='plan_sealed',result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_digest=plan['selection_digest'])
            store.event(dict(kind='plan_sealed',row=deepcopy(row)))


def run(prepared,expected,output):
    started=time.monotonic();rows=initial_rows();images=[];errors=[];model=None;p=None;trace=None
    before=after=host=None;source_ok=False
    def check():require(time.monotonic()-started<=WALL,'whole job900s incl verify/load/write/final; no retry')
    store=Store(output,check)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        p=verify(prepared,expected);check()
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        require(device.type=='cuda','no CPU neural fallback')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        moving.low.frozen(model);before=moving.low.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        transform=p0.image_transform();trace=NativeTrace(model,torch,store,check,p['encoder_output_shape'])
        def encode(rgb,image):
            with traced_scope(trace,dict(image_index=image['image_index']),image):
                with torch.inference_mode():
                    x=p0._image_tensor(rgb,transform=transform,device=device)
                    require(tuple(x.shape)==(1,1,3,224,224) and x.device.type=='cuda','real native pixels-only preprocessing')
                    image['preprocessed']=trace.save(x,f'image_{image["image_index"]:02}_preprocessed')
                    z=model.encode({'pixels':x})['emb']
                    require(tuple(z.shape)==(1,1,192) and z.dtype==torch.float32,'native projected CLS192')
                    return z[0,0].detach().cpu().numpy().copy()
        def runner(arm,i,sink,row):
            with traced_scope(trace,dict(plan_index=row['index']),row):
                with history.context_trace(model,sink):return blind.execute_arm(arm,i,model,device,torch,sink,row)
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False),trace:
            execute(p,expected,store,encode,runner,rows,images,check)
        check()
    except BaseException as e:
        errors.append(moving.low.error('execution',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:moving.low.frozen(model);after=moving.low.tensor_hash(model)
            except BaseException as e:errors.append(moving.low.error('final_model',e))
        try:verify(prepared,expected);check();source_ok=True
        except BaseException as e:errors.append(moving.low.error('final_source_or_wall',e))
        try:store.close()
        except BaseException as e:errors.append(moving.low.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and len(images)==16 and all(r['status']=='plan_sealed' for r in rows)
        hashes={}
        try:hashes={f.name:sha(f) for f in store.output.iterdir()}
        except BaseException as e:errors.append(moving.low.error('artifact_hash',e));complete=False
        try:check()
        except BaseException as e:errors.append(moving.low.error('final_wall',e));complete=False
        result=dict(schema=NAME+'.report',status='overlay_recompiled_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,images=images,errors=errors,
            first_error=errors[0] if errors else None,model_unchanged=before==after if before is not None else None,
            model_hash_before=before,model_hash_after=after,source_verified=source_ok,host=host,contract=contract(),
            module_ledger=trace.entries if trace else [],artifact_sha256=hashes,fixed_denominator=8,independent_tasks=2,
            commands_released=complete,source_report_sha256=SOURCE_SHA,wall_seconds=time.monotonic()-started,
            new_physics=0,training=False,Qfit=False,P1_admission=False)
        # A timed-out job still gets a truthful partial report using the final reserve.
        # Store's regular deadline guard is disabled only for this failure record.
        if not complete:store.check=lambda:None
        try:store.put('report.json',result,final=True)
        except BaseException as e:complete=False;print(canonical(dict(report_write_error=moving.low.error('report',e))).decode(),flush=True)
    require(complete,'partial evidence preserved; no retry: '+str(errors));return result


def consume(output,report_sha):
    """Stable Bohr handoff: list of exactly8 newly compiled plans, never witness selection."""
    root=Path(output).absolute();core.hash_value(report_sha);require(sha(root/'report.json')==report_sha,'external actual planner report SHA')
    r=read(root/'report.json',4*1024**2)
    require(r['schema']==NAME+'.report' and r['status']=='overlay_recompiled_plans_sealed_not_admitted' and
        r['errors']==[] and r['model_unchanged'] is True and r['source_verified'] is True and r['commands_released'] is True and
        r['fixed_denominator']==8 and len(r['images'])==16,'only complete8 global release')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['contract']==p['contract']==contract() and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'same frozen model and protocol')
    require({f.name for f in root.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'exact output artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(flat(root,name))==h,'artifact SHA')
    require(read(root/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'own execution binding')
    plain,eof=blind.inflate((root/'events.jsonl.gz').read_bytes(),128*1024**2)
    require(eof and plain.endswith(b'\n'),'complete journal');events=[blind.decode(x) for x in plain.splitlines()]
    require([(r['root_slot'],r['condition'],r['arm']) for r in r['rows']]==list(KEYS),'stable ALL8 order')
    def array(receipt):
        raw=flat(root,receipt['file']).read_bytes();require(hash_bytes(raw)==receipt['file_sha256'],'tensor file SHA')
        data,eof=blind.inflate(raw,4*1024**2);require(eof,'complete tensor gzip');stream=BytesIO(data);a=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(data) and str(a.dtype)==receipt['dtype'] and list(a.shape)==receipt['shape'] and
            np.isfinite(a).all() and hash_bytes(np.ascontiguousarray(a).tobytes())==receipt['array_sha256'],'original array SHA/dtype/shape')
        return a
    ledger=r['module_ledger']
    require([e['index'] for e in ledger]==list(range(len(ledger))) and all(e['completed'] is True for e in ledger),
        'complete native tensor ledger indices')
    require({k:sum(e['resource']==k for e in ledger) for k in ('encoder','projector','action_encoder','predictor','pred_proj')}==
        dict(encoder=16,projector=16,action_encoder=576,predictor=576,pred_proj=576),'all actual native module counts')
    attempts=[e['entry'] for e in events if e.get('kind')=='tensor_module_attempt']
    returns=[e['entry'] for e in events if e.get('kind')=='tensor_module_completed']
    require(len(attempts)==len(returns)==len(ledger) and returns==ledger and
        all(a=={k:v for k,v in e.items() if k not in ('inputs','output','completed')}|{'completed':False}
            for a,e in zip(attempts,ledger,strict=True)), 'durable attempted/completed native trace')
    for entry in ledger:
        for receipt in entry['inputs']+[entry['output']]:array(receipt)
    # Validate each true preprocessing/encoder/CLS/projector chain before task release.
    exports={}
    for slot in p['slots']:
        packet=load_packet(slot['blind_path'],slot['blind_sha256']);values=[]
        for j,rgb in enumerate((*packet['rgb_history'],packet['goal_rgb'])):
            image=r['images'][4*slot['order']+j]
            require(image['image_index']==4*slot['order']+j and image['root_slot']==slot['root_slot'] and
                image['condition']==slot['condition'] and image['status']=='encoded' and np.array_equal(array(image['pixels']),rgb),'own four visual images')
            es=image['modules'];require([(e['resource'],e['completed']) for e in es]==[('encoder',True),('projector',True)],'two native visuals')
            require(es==[e for e in ledger if e['scope']==dict(image_index=image['image_index'])],'own global image trace join')
            encoder,projector=es;pre=array(image['preprocessed']);ein=array(encoder['inputs'][0]);eout=array(encoder['output'])
            pin=array(projector['inputs'][0]);pout=array(projector['output']);z=array(image['latent'])
            visual_bridge(pre,ein,eout,pin,pout,z,p['encoder_output_shape'])
            values.append(z)
        e=task_export(p,r['prepare_sha256'],slot,packet,values)
        require(read(root/f'task_{slot["order"]:02}.json.gz')==e,'exact new encoded task readback');exports[slot['order']]=e
    plans=[]
    for row in r['rows']:
        index=row['index'];require(index==len(plans) and row['status']=='plan_sealed','complete own plan')
        slot=p['slots'][index//2];arm=ARMS[index%2];i=task_interface(exports[index//2])
        require(row['task_file']==f'task_{index//2:02}.json.gz' and row['task_id']==i['task_id'] and
            row['plan_file']==f'plan_{index:02}.plan.json' and row['result_file']==f'plan_{index:02}.result.json.gz','own canonical result identity')
        result=read(root/row['result_file'],16*1024**2);choice=blind.selected(arm,i,result)
        own=[e['event'] for e in events if e.get('plan_index')==index]
        history.ledger_check(arm,own);require(history.history_check(arm,i,result,own)==row['history_verification'],'every model context and AD/replay chronology')
        blind.exact_counts(row['resources'],blind.budgets()[arm]);blind.exact_counts(row['AD_API_resources'],blind.contract(1)['AD_API_per_root'][arm])
        modules=row['modules'];count=blind.budgets()[arm]['predict']
        require(modules==[e for e in ledger if e['scope']==dict(plan_index=index)] and
            [e['resource'] for e in modules]==['action_encoder','predictor','pred_proj']*count,'per-plan full native trace schedule')
        contexts=[e['context'] for e in own if e['kind']=='native_context_attempt']
        require(len(contexts)==count,'actual all full3 contexts')
        for j,context in enumerate(contexts):
            ae,pred,proj=modules[3*j:3*j+3]
            ain,aout=array(ae['inputs'][0]),array(ae['output'])
            pin,pa,pout=array(pred['inputs'][0]),array(pred['inputs'][1]),array(pred['output'])
            qin,qout=array(proj['inputs'][0]),array(proj['output'])
            require(ain.shape==(1,3,10) and np.array_equal(ain[0],context['action_history']) and
                pin.shape==(1,3,192) and np.array_equal(pin[0],context['latent_history']) and
                np.array_equal(pa,aout) and pout.shape==(1,3,192) and qin.shape==qout.shape==(3,192) and
                np.array_equal(qin,pout.reshape(3,192)), 'every native input/intermediate/projection tensor bridge')
        require(np.array_equal(array(modules[-1]['output'])[-1],choice['prediction']['latent_history'][-1]),
            'final selected replay equals actual native projected output')
        plan=read(root/row['plan_file']);seal=plan['selection_digest']
        require(core.digest({k:v for k,v in plan.items() if k!='selection_digest'})==seal==row['selection_digest'] and
            all(plan[k]==v for k,v in choice.items()) and plan['blind_path']==slot['blind_path'] and plan['blind_sha256']==slot['blind_sha256'] and
            (plan['root_slot'],plan['condition'],plan['arm'])==KEYS[index] and plan['source_report_sha256']==SOURCE_SHA,'exact selected new plan seal')
        commands=np.asarray(plan['commands']);require(commands.shape==(10,2) and np.isfinite(commands).all() and
            np.array_equal(commands,commands.astype(np.float32)) and (abs(commands)<=1).all(),'literal rawfloat32 commands, no rescale')
        plans.append(dict(plan,planner_report_sha256=report_sha))
    require(len(plans)==8 and sha(root/'report.json')==report_sha,'all8 final handoff SHA');return plans


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_execution')
    elif a.command=='run':result=run(a.prepared,a.prepare_sha,a.output)
    else:result=consume(a.output,a.report_sha)
    print(canonical(result).decode(),end='')


if __name__=='__main__':main()
