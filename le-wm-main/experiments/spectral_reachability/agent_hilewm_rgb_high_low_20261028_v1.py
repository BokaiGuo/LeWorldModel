"""Known moving RGB -> owned Hi encoder -> installed high CEM -> FIRST -> low.

MAIN-only CUDA integration, not official-class parity, physics or E1 admission.
Import/prepare/consumer use NumPy and saved data only. No pickle execution or
foreign model classes. Saved authority layers are lazy (stdlib pickletools).
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import io
import importlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
from experiments.spectral_reachability import agent_hilewm_solver_cuda_seam_20261028_v1 as seam
from experiments.spectral_reachability import agent_hilewm_trusted_loader_20261028_v1 as loader
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as moving

TAG='agent_hilewm_rgb_high_low_20261028_v1'
ROOT=seam.ROOT
need,encode,digest=seam.require,seam.encode,seam.digest


class SavedDependency:
    # Evidence layers transitively import stdlib pickletools; load only when
    # MAIN rebuilds their saved authority, never at import or RAM math paths.
    def __init__(self,name):self.name=name
    def __getattr__(self,key):return getattr(importlib.import_module('experiments.spectral_reachability.'+self.name),key)


paid=SavedDependency('agent_hilewm_real_high_cem_20261028_v1')
v3=SavedDependency('agent_hilewm_trusted_cuda_validation_20261028_v3')
CAP,FLOOR,WALL,RESERVE=192*1024**2,200*1024**2,900,8*1024**2
ORDER=((0,'off'),(0,'on'),(1,'off'),(1,'on'))
TEXTS=loader.TEXTS
EVAL=TEXTS/'package__code__h_le_wm__eval__hierarchical.py.source.txt'
STATS=ROOT/'.venv/lib/python3.12/site-packages/stable_pretraining/data/dataset_stats.py'
VIT=ROOT/'.venv/lib/python3.12/site-packages/transformers/models/vit/modeling_vit.py'
PINS={str(ROOT/'experiments/spectral_reachability/agent_hilewm_real_high_cem_20261028_v1.py'):'fad571c64cf3e861d8ea5c90ebd7a86c797f15ab0daf0ef24e83dd43efad737d',
    str(ROOT/'experiments/spectral_reachability/agent_hilewm_trusted_cuda_validation_20261028_v3.py'):'0b237a166e77610f52db413f58e0f8ad06dce7cde7ff9923264d5911b21f0456',
    str(Path(moving.__file__).resolve()):'ce0e5ddb9a3ffc223efd94dbbee93a97bffa44b225a3bbb28afba41075a67ab6',
    str(EVAL):'0a2c2b2688b73a388ff6011c4b32081704a7641b31efa1645a47b2f1d34eab3c',
    str(STATS):'b5063768d173792bbb38a25acf9ea2a599e3adcbea077cf9507ee0c5ca258302'}
V3_REPORT=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v3_cuda_MAIN01/report.json'
V3_RH='c6fd37e6e4540a82bb1bcb168a34cb4d222c81cf82f9cce88b1b44f557e27c20'
V3_RECEIPT=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v3_consume_MAIN01/receipt.json'
V3_CH='6151e93c40fc1c5b053b81f8ac73a8dca7644a8a037492686a50a2718e24ddc6'
CEM_REPORT=ROOT/'results/agent_hilewm_real_high_cem_20261028_v1_cuda_MAIN01/report.json'
CEM_RH='8fe58985700dcdefa79bcfd56f19f1c857781a926a31eebe5a220f83b0e97801'
CEM_RECEIPT=ROOT/'results/agent_hilewm_real_high_cem_20261028_v1_consume_MAIN01.json'
CEM_CH='7dfb4b49d552a6248da8b603e027a95c541c5db95af4fab777828ed412886e43'


def protocol():
    return dict(schema=TAG+'.protocol',order=[list(k) for k in ORDER],known_roots=2,fresh_roots=0,
        input='each original moving blind history LAST actual frame at warmup10 plus its own goal RGB',
        high=dict(batch=1,chunk=1,samples=8,iterations=3,topk=3,horizon=2,action_block=2,macro_dim=32,
            seed_base=1729,receding_horizon=1,warm_start=False,bounds=[-1.,1.],clamp=False),
        low=dict(history='single current Hi latent, zero grouped action history per official _plan_low',
            horizon=2,grouped_action_dim=10,candidates=3,candidate_recipe='all0,all-0.25,all+0.25 in MODEL normalized coordinates',
            solver='NO low CEM; fixed delivery/cost/rollout panel only; no raw controls released'),
        new_calls=dict(encoder=8,projector=8,high_cost=12,high_rollout_selected=4,
            high_predict=56,high_predict_sample_rows=392,low_cost=4,low_predict=8,low_predict_sample_rows=24,
            action_encoder=8,low_projection=8,macro_encoder=0,AD=0,physics=0),
        encoder='new owned dense fixed224 ViT equations over certified data graph; NOT original class/bytecode parity',
        preprocessing='ToDtype f32 mul(1/255), ImageNet Normalize, identity Resize224; original eval order',
        waypoint='original _plan_high reshape first receding block -> rollout -> pred[:,0,0,:], NOT terminal',
        normalizer='no raw-action transformation applied: low panel already in model grouped coordinates; raw inverse unknown',
        low_high_pairing='embedded original checkpoint only; NOT current common-controller/backbone equivalence',
        precision=dict(CUDA=True,MATH=True,TF32=False,autocast=False,grad=False),
        output_cap_bytes=CAP,floor_bytes=FLOOR,wall_seconds=WALL,final_reserve_bytes=RESERVE,
        training=False,physical_release=False,official_parity=False,common_backbone_verified=False,
        full_policy=False,scientific_admission=False,P1_admission=False)


def read(path,h=None):
    body=loader.bytes_read(path,CAP)
    need(h is None or digest(body)==h,'external JSON SHA')
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(body,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))


def sha(path):return digest(loader.bytes_read(path,loader.ARRAY_CAP))


def authorities():
    rows=[]
    for module,rp,rh,cp,ch in ((v3,V3_REPORT,V3_RH,V3_RECEIPT,V3_CH),(paid,CEM_REPORT,CEM_RH,CEM_RECEIPT,CEM_CH)):
        r=read(rp,rh);m=r['manifest_path'];h=r['manifest_sha256']
        accepted=module.consume(m,h,rp,rh)
        need(accepted==read(cp,ch),'entire actual v3/CEM strict rebuilt, not relabelled')
        rows.append(dict(manifest=m,manifest_sha256=h,report=str(rp),report_sha256=rh,receipt=str(cp),receipt_sha256=ch))
    return rows


def config_contract(saved,graph,facts):
    """Reject unsupported literal values/shapes; never supply config defaults."""
    c=facts['encoder_config'];mods={tuple(x['path']):x for x in graph['modules']}
    shapes={'.'.join(x['module_path']+[x['name']]):x['metadata']['shape'] for x in saved['nn']['entries'] if x['metadata'] is not None}
    expected=dict(image_size=224,patch_size=14,num_channels=3,hidden_size=192,num_hidden_layers=12,
        num_attention_heads=3,intermediate_size=768,hidden_act='gelu',hidden_dropout_prob=0.,attention_probs_dropout_prob=0.,
        layer_norm_eps=1e-12,qkv_bias=True,chunk_size_feed_forward=0,
        _attn_implementation_internal='sdpa',interpolate_pos_encoding=True)
    need(all(k in c and type(c[k]) is type(v) and c[k]==v for k,v in expected.items()),'literal fixed ViT config, no defaults')
    need(shapes['encoder.embeddings.position_embeddings']==[1,257,192] and
        shapes['encoder.embeddings.cls_token']==[1,1,192] and shapes['encoder.embeddings.patch_embeddings.projection.weight']==[192,3,14,14]
        and shapes['low_predictor.pos_embedding']==[1,3,192]
        and shapes['action_encoder.patch_embed.weight']==[10,10,1],'real token/low/grouped ABI')
    need(mods[('encoder','embeddings')]['literal_attributes']['mask_token'] is None,'unmasked pixel path only')
    conv=mods[('encoder','embeddings','patch_embeddings','projection')]['literal_attributes']
    for k,v in dict(kernel_size=[14,14],stride=[14,14],padding=[0,0],dilation=[1,1],groups=1,padding_mode='zeros').items():
        need(conv[k]==v,'literal patch convolution '+k)
    for i in range(12):
        p=('encoder','encoder','layer',str(i));a=mods[p+('attention','attention')]['literal_attributes']
        need(all(a[k]==v for k,v in dict(num_attention_heads=3,attention_head_size=64,all_head_size=192,
            dropout_prob=0.,scaling=.125,is_causal=False).items()),'literal ViT attention')
        need(mods[p]['literal_attributes']['chunk_size_feed_forward']==0,'no chunked MLP')
        for norm in ('layernorm_before','layernorm_after'):
            need(mods[p+(norm,)]['literal_attributes']['eps']==c['layer_norm_eps'],'literal ViT norm epsilon')
        for name,shape in [('attention.attention.query.weight',[192,192]),('attention.attention.key.weight',[192,192]),
            ('attention.attention.value.weight',[192,192]),('intermediate.dense.weight',[768,192]),('output.dense.weight',[192,768])]:
            need(shapes['.'.join(p)+'.'+name]==shape,'literal ViT parameter shape')
    ac=mods[('action_encoder','patch_embed')]['literal_attributes']
    need(all(ac[k]==v for k,v in dict(stride=[1],padding=[0],dilation=[1],groups=1,kernel_size=[1]).items()),'literal action Conv1d')
    need(mods[('encoder','layernorm')]['literal_attributes']['eps']==c['layer_norm_eps'],'literal final ViT norm')
    return dict(config=deepcopy(c),low_context=3,action_width=10,original_transformers_version=c['transformers_version'],
        native_encoder_equivalence=False,raw_normalizer_certified=False,defaults_applied=False)


def image_stats():
    tree=ast.parse(STATS.read_text());values=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='ImageNet' for x in n.targets):
            need(isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='dict'
                and not n.value.args,'literal ImageNet declaration')
            values.append({k.arg:ast.literal_eval(k.value) for k in n.value.keywords})
    need(values==[dict(mean=[.485,.456,.406],std=[.229,.224,.225])],'bound original ImageNet statistics')
    return values[0]


def preprocess(rgb,stats):
    need(rgb.dtype==np.uint8 and rgb.shape==(224,224,3),'actual RGB uint8 224 only, no implicit resize')
    x=rgb.transpose(2,0,1).astype(np.float32)*np.float32(1/255)
    x=(x-np.asarray(stats['mean'],np.float32)[:,None,None])/np.asarray(stats['std'],np.float32)[:,None,None]
    return x[None].copy()


def inputs():
    need(sha(moving.SOURCE)==moving.SOURCE_SHA,'original moving report fixed SHA')
    # This packet-only decoder never passes evaluator state/outcome to planning.
    slots=moving.packets(read(moving.SOURCE,moving.SOURCE_SHA),moving.SOURCE.parent,moving.load_packet)
    need([(s['root_slot'],s['condition']) for s in slots]==list(ORDER),'all original four, no outcome filtering')
    return slots


def closure():
    for p,h in PINS.items():need(sha(p)==h,'frozen source '+p)
    out=paid.sources();out.update(v3.closure());out.update(PINS)
    out.update({str(p.resolve()):sha(p) for p in moving.source_files()})
    for p in (VIT,ROOT/'.venv/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py',
        ROOT/'.venv/lib/python3.12/site-packages/torch/nn/functional.py',
        ROOT/'.venv/lib/python3.12/site-packages/torch/nn/attention/__init__.py',
        Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):out[str(p)]=sha(p)
    out.update({str(TEXTS/k):v for k,v in loader.TEXT_HASHES.items()})
    out[str(seam.bridge_module().POLICY)]=seam.bridge_module().OWNER_SHAS['policy']
    return out


def build():
    sources=closure();auth=authorities();saved,g,f=loader.audit();config=config_contract(saved,g,f)
    parent,_=paid.diag.upstream();slots=inputs()
    for s in slots:sources[s['blind_path']]=s['blind_sha256']
    sources[str(moving.SOURCE)]=moving.SOURCE_SHA
    need(all(sha(p)==h for p,h in sources.items()),'complete byte closure')
    return dict(schema=TAG+'.prepare',protocol=protocol(),sources=sources,authorities=auth,slots=slots,
        model=config,statistics=image_stats(),export={k:parent[k] for k in
            ('export_report','export_report_sha256','export_receipt','export_receipt_sha256')},
        scientific_admission=False)


def precision(t):
    v=dict(deterministic=t.are_deterministic_algorithms_enabled(),tf32_matmul=t.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=t.backends.cudnn.allow_tf32,autocast=t.is_autocast_enabled('cuda'),grad=t.is_grad_enabled(),
        math_sdp=t.backends.cuda.math_sdp_enabled(),flash_sdp=t.backends.cuda.flash_sdp_enabled(),
        efficient_sdp=t.backends.cuda.mem_efficient_sdp_enabled(),cudnn_sdp=t.backends.cuda.cudnn_sdp_enabled())
    need(v==precision_expected(),'actual MATH/noTF32/noautocast/inference guard');return v


def precision_expected():
    return dict(deterministic=True,tf32_matmul=False,tf32_cudnn=False,autocast=False,grad=False,
        math_sdp=True,flash_sdp=False,efficient_sdp=False,cudnn_sdp=False)


def scoped(path):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and p.parent==ROOT/'results'
        and p.name.startswith(TAG+'_'),'exclusive own-stem results path');return p


def write(path,body):
    need(not Path(path).exists() and len(body)<=CAP and shutil.disk_usage(Path(path).parent).free>=FLOOR+len(body),'exclusive file/floor')
    with Path(path).open('xb') as f:need(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())


def prepare(output):
    m=build();p=scoped(output);need(shutil.disk_usage(p.parent).free>=CAP+FLOOR,'cap+floor');p.mkdir()
    write(p/'manifest.json',encode(m));return dict(manifest=str(p/'manifest.json'),sha256=sha(p/'manifest.json'))


def verify(path,h):
    m=read(path,h);need(m==build() and sha(path)==h,'whole manifest/source/input reconstruction');return m


class Store:
    def __init__(self,path,start):
        self.path=scoped(path);self.start=start;self.used=0;self.arrays={}
        need(shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'whole cap+floor');self.path.mkdir()
    def guard(self,n=0,final=False):
        need(final or time.monotonic()-self.start<=WALL,'cooperative wall')
        need(self.used+n<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+n,'cap/reserve/floor')
    def put(self,name,body,final=False):
        need(Path(name).name==name and name not in ('','.','..'),'flat output');self.guard(len(body),final)
        write(self.path/name,body);self.used+=len(body)
    def array(self,value):
        if not isinstance(value,np.ndarray):value=value.detach().cpu().numpy()
        a=np.asarray(value);need(a.dtype in (np.dtype('<f4'),np.dtype('<i8')) and np.isfinite(a).all(),'original finite f32/i64 buffers')
        name=f'array_{len(self.arrays):05d}.npy';s=io.BytesIO();np.save(s,a,allow_pickle=False);body=s.getvalue();self.put(name,body)
        self.arrays[name]=dict(file_sha256=digest(body),file_bytes=len(body),data_sha256=digest(a.tobytes()),shape=list(a.shape),dtype=a.dtype.str)
        return name


class Trace:
    """Persist attempts before actual calls, original buffers before returns."""
    def __init__(self,t,store,events):self.t,self.store,self.events=t,store,events;self.stack=[]
    def call(self,kind,args,fn):
        self.store.guard();precision(self.t)
        event=dict(index=len(self.events),parent=self.stack[-1] if self.stack else None,kind=kind,returned=False,
            inputs=[self.store.array(a) for a in args],layouts=[paid.diag.layout(a) for a in args])
        self.events.append(event);self.stack.append(event['index'])
        value=fn();self.t.cuda.synchronize();self.store.guard()
        event.update(output=self.store.array(value),output_layout=paid.diag.layout(value),returned=True)
        self.stack.pop();return value


def owned_pixel(t,root,x,c,trace):
    """Dense, unmasked224 ViT source-layout mapping over ORIGINAL tensor nodes.

    No original foreign class import/forward, no config defaults/interpolation.
    This NEW kernel needs MAIN numerical review; not a pretrained-bytecode cert.
    """
    f=t.nn.functional;e=root.encoder
    def norm(v,node):return f.layer_norm(v,(192,),node.weight,node.bias,c['layer_norm_eps'])
    def kernel():
        p=e.embeddings.patch_embeddings.projection
        patches=trace.call('pixel.patch',[x],lambda:f.conv2d(x,p.weight,p.bias,stride=14)).flatten(2).transpose(1,2)
        z=t.cat((e.embeddings.cls_token.expand(x.shape[0],-1,-1),patches),dim=1)+e.embeddings.position_embeddings
        for i in range(c['num_hidden_layers']):
            layer=e.encoder.layer._modules[str(i)]
            def block(z=z,layer=layer):
                v=norm(z,layer.layernorm_before);att=layer.attention.attention
                def heads(v):return v.reshape(v.shape[0],v.shape[1],3,64).transpose(1,2)
                q,k,v=(heads(f.linear(v,n.weight,n.bias)) for n in (att.query,att.key,att.value))
                a=f.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False).transpose(1,2).reshape(z.shape)
                d=layer.attention.output.dense;y=z+f.linear(a,d.weight,d.bias)
                h=norm(y,layer.layernorm_after);d=layer.intermediate.dense
                h=f.gelu(f.linear(h,d.weight,d.bias),approximate='none');d=layer.output.dense
                return y+f.linear(h,d.weight,d.bias)
            z=trace.call(f'pixel.layer{i}',[z],block)
        return trace.call('pixel.finalnorm',[z],lambda:norm(z,e.layernorm))
    hidden=trace.call('encoder',[x],kernel)
    z=trace.call('projector',[hidden[:,0]],lambda:root.projector(hidden[:,0]))
    return z


class OwnedLow:
    """Official fixed single-history get_cost_low/rollout_low/predict_low subset.

    Does not edit root.get_cost (which remains high-only). New facade own ABI.
    """
    def __init__(self,t,root,trace):self.t,self.root,self.trace=t,root,trace
    def get_cost(self,info,candidates):
        need(set(info)=={'planner_level','z_hist','a_hist','z_subgoal'} and info['planner_level']=='low','original low metadata')
        t,r,trace=self.t,self.root,self.trace;b,s,steps,width=candidates.shape
        need((b,s,steps,width)==(1,3,2,10) and info['z_hist'].shape==(1,1,192)
            and info['a_hist'].shape==(1,1,10) and info['z_subgoal'].shape==(1,192),'fixed official low info and panel')
        def kernel():
            z=info['z_hist'][:,None].expand(b,s,1,192).reshape(b*s,1,192)
            a=info['a_hist'][:,None].expand(b,s,1,10).reshape(b*s,1,10);pred=[]
            for step in range(steps):
                a=t.cat((a,candidates[:,:,step].reshape(b*s,1,10)),dim=1)
                def embed():
                    n=r.action_encoder.patch_embed
                    p=t.nn.functional.conv1d(a.transpose(1,2),n.weight,n.bias)
                    return r.action_encoder.embed(p.transpose(1,2))
                emb=trace.call('action_encoder',[a],embed)
                ctx=min(3,z.shape[1],emb.shape[1]);zi,ai=z[:,-ctx:],emb[:,-ctx:]
                h=trace.call('low_predictor',[zi,ai],lambda:r.low_predictor(zi,ai))
                p=trace.call('low_pred_proj',[h.reshape(-1,192)],lambda:r.low_pred_proj(h.reshape(-1,192)))
                nxt=p.reshape(b*s,ctx,192)[:,-1:];z=t.cat((z,nxt),dim=1);pred.append(nxt[:,0])
            rollout=t.stack(pred,dim=1).reshape(b,s,steps,192)
            # Persist genuine full native rollout, not only an alleged cost.
            self.rollout=self.trace.store.array(rollout)
            return (rollout[:,:,-1]-info['z_subgoal'][:,None]).pow(2).sum(dim=-1)
        return trace.call('low_cost',[info['z_hist'],info['a_hist'],info['z_subgoal'],candidates],kernel)


def low_panel():
    return np.stack([np.full((2,10),v,np.float32) for v in (0.,-.25,.25)])[None]


def high_search(t,CEM,Box,root,z,goal,trace,row,index):
    """Actual installed solver. No actor/analytic cost or foreign policy exec."""
    b=seam.bridge_module();device=z.device
    # Tap the three actual high modules without replacing any root method.
    pending={};handles=[]
    def pre(kind):
        def hook(module,args):
            trace.store.guard();precision(t)
            e=dict(index=len(trace.events),parent=trace.stack[-1] if trace.stack else None,kind=kind,returned=False,
                inputs=[trace.store.array(a) for a in args],layouts=[paid.diag.layout(a) for a in args])
            trace.events.append(e);trace.stack.append(e['index']);pending[kind]=e
        return hook
    def post(kind):
        def hook(module,args,out):
            e=pending.pop(kind);t.cuda.synchronize();trace.store.guard()
            e.update(output=trace.store.array(out),output_layout=paid.diag.layout(out),returned=True);trace.stack.pop()
        return hook
    class RootCost:
        def parameters(self):return root.parameters()
        def get_cost(self,info,candidates):
            return trace.call('high_cost',[info['z_init'],info['z_goal'],candidates],lambda:root.get_cost(info,candidates))
    info=dict(planner_level='high',z_init=z,z_goal=goal)
    facade=b.HighBatchBridge.bind_high_model(info,RootCost(),tensor_batch=1,chunk_size=1,num_samples=8,
        abi=seam.cuda_abi(t,device),max_cost_calls=3,max_candidate_rows=24)
    class Recorder:
        output_key=TAG
        def __init__(self):self.history=[]
        def reset(self):row['cem_reset']+=1;self.history=[]
        def start_batch(self):row['cem_start']+=1
        def end_solve(self):row['cem_end']+=1
        def __call__(self,**state):
            row['cem'].append(dict(step=state['step'],arrays={k:trace.store.array(state[k]) for k in seam.KEYS}))
            self.history.append(dict(step=state['step']))
    row.update(cem=[],cem_reset=0,cem_start=0,cem_end=0)
    try:
        for k in paid.KINDS:handles.extend((getattr(root,k).register_forward_pre_hook(pre(k)),getattr(root,k).register_forward_hook(post(k))))
        solver=CEM(model=facade,batch_size=1,num_samples=8,var_scale=1.,n_steps=3,topk=3,
            device=device,seed=1729+index,callbacks=[Recorder()])
        solver.configure(action_space=Box(-1.,1.,shape=(1,32),dtype=np.float32),n_envs=1,
            config=SimpleNamespace(horizon=2,action_block=2))
        initial=t.zeros((1,2,64),device=device,dtype=t.float32);row['high_initial']=trace.store.array(initial)
        out=solver.solve(dict(facade.solver_info),init_action=initial)
        row['cem_history']=out['callbacks'][TAG]
        row['high_outputs']={k:trace.store.array(out[k] if k=='actions' else out[k][0]) for k in ('actions','mean','var')}
        row['high_outputs']['costs']=trace.store.array(np.asarray(out['costs'],np.float32))
        actions=out['actions'].to(device);macro=actions[:,:1].reshape(1,2,32)
        row['macro_sequence']=trace.store.array(macro)
        roll=trace.call('high_selected_rollout',[z,macro],lambda:root.rollout_high(z,macro))
        row['selected_rollout']=trace.store.array(roll)
        target=roll[:,0,0,:];row['high_target']=trace.store.array(target)
        return target
    finally:
        for h in handles:h.remove()
        row['bridge_receipt']=facade.receipt();row['hooks_removed']=len(handles)


def resource_counts(rows):
    events=[e for r in rows for e in r.get('events',[])];kinds=sorted({e['kind'] for e in events})
    return dict(modules={k:dict(attempted=sum(e['kind']==k for e in events),returned=sum(e['kind']==k and e['returned'] for e in events),
        input_rows=sum(e['layouts'][0]['shape'][0] for e in events if e['kind']==k),
        input_context_tokens=sum(e['layouts'][0]['shape'][0]*(e['layouts'][0]['shape'][1] if len(e['layouts'][0]['shape'])==3 else 1)
            for e in events if e['kind']==k)) for k in kinds},
        bridge_candidate_rows=sum(r.get('bridge_receipt',{}).get('candidate_rows_attempted',0) for r in rows),
        low_candidate_rows=sum(e['layouts'][3]['shape'][0]*e['layouts'][3]['shape'][1] for e in events if e['kind']=='low_cost'),
        macro_encoder=0,AD=0,physics=0,FLOPs=None,official_search_budget=False)


def run_cuda(path,h,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN only before any I/O/runtime import')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','deterministic CUBLAS')
    start=time.monotonic();m=verify(path,h);store=Store(output,start)
    report=dict(schema=TAG+'.report',manifest=str(Path(path).resolve()),manifest_sha256=h,protocol=protocol(),sources=m['sources'],
        arrays=store.arrays,rows=[dict(index=i,root_slot=s['root_slot'],condition=s['condition'],status='not_started',events=[]) for i,s in enumerate(m['slots'])],
        errors=[],completed=False,model_before=None,model_after=None,construction=None,scientific_admission=False)
    root=None
    try:
        t,CEM,Box=seam.load_runtime();t.use_deterministic_algorithms(True)
        t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        a=m['export'];root,report['construction']=loader.construct_cuda(a['export_report'],a['export_report_sha256'],
            a['export_receipt'],a['export_receipt_sha256'],main_authorized=True)
        report['model_before']=paid.diag.old.weights(root)
        need(report['model_before']==report['construction']['tensor_value_hashes'],'exact original tensors')
        device=next(root.parameters()).device;report['runtime']=dict(device=str(device),torch=str(t.__version__),cuda=t.version.cuda)
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with t.inference_mode(),t.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            report['precision_before']=precision(t)
            for row,slot in zip(report['rows'],m['slots'],strict=True):
                row['status']='attempted';trace=Trace(t,store,row['events']);packet=moving.load_packet(slot['blind_path'],slot['blind_sha256'])
                zs=[];row['pixels']=[]
                for role,rgb in (('root',packet['rgb_history'][-1]),('goal',packet['goal_rgb'])):
                    x=preprocess(rgb,m['statistics']);ref=store.array(x);row['pixels'].append(dict(role=role,preprocessed=ref))
                    z=owned_pixel(t,root,t.from_numpy(x).to(device),m['model']['config'],trace);zs.append(z)
                row['z_init'],row['z_goal']=[store.array(z) for z in zs]
                target=high_search(t,CEM,Box,root,*zs,trace,row,row['index'])
                zh=zs[0].unsqueeze(1);ah=t.zeros((1,1,10),dtype=t.float32,device=device)
                candidates=t.from_numpy(low_panel()).to(device)
                low=OwnedLow(t,root,trace);info=dict(planner_level='low',z_hist=zh,a_hist=ah,z_subgoal=target)
                value=low.get_cost(info,candidates)
                row['low']=dict(z_hist=store.array(zh),a_hist=store.array(ah),z_subgoal=store.array(target),
                    candidates=store.array(candidates),cost=store.array(value),rollout=low.rollout)
                need(not trace.stack,'all native returns');row['status']='returned'
            report['precision_after']=precision(t)
        report['model_after']=paid.diag.old.weights(root)
        need(report['model_before']==report['model_after'] and verify(path,h)==m,'source/export/weights unchanged')
        store.guard();report['completed']=True
    except BaseException as e:report['errors'].append(dict(type=type(e).__name__,message=str(e)[:3000]))
    finally:
        if root is not None and report['model_after'] is None:
            try:report['model_after']=paid.diag.old.weights(root)
            except BaseException as e:report['errors'].append(dict(type=type(e).__name__,message=str(e)[:1000]))
        report.update(resources=resource_counts(report['rows']),elapsed_seconds=time.monotonic()-start)
        store.put('report.json',encode(report),final=True)
    need(report['completed'] and not report['errors'],'partial retained, no retry/refund')
    return dict(report=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def exact(a,b,message):
    need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),message)


def check_cost(actual,terminal,goal):
    # Same independent gamma194 arithmetic bound as frozen paid CEM; never a
    # tolerance on model outputs. Kept local to avoid pickletools cold import.
    need(actual.dtype==terminal.dtype==goal.dtype==np.float32 and terminal.shape==goal.shape
        and terminal.shape[-1]==192 and actual.shape==terminal.shape[:-1],'native cost f32 ABI')
    expected=np.square(terminal.astype(float)-goal.astype(float)).sum(axis=-1)
    u=np.finfo(np.float32).eps/2;k=194
    bound=(k*u/(1-k*u))*expected+k*float(np.nextafter(np.float32(0),np.float32(1)))
    need(np.isfinite(actual).all() and np.isfinite(expected).all() and
        np.all(np.abs(actual.astype(float)-expected)<=bound) and np.all(actual[expected==0]==0),
        'terminal-goal native cost arithmetic bound, not neural parity')


def checked_layout(l,a,device):
    need(l['shape']==list(a.shape) and l['dtype']=='torch.float32' and l['device']==device
        and type(l['storage_offset']) is int and l['storage_offset']>=0
        and len(l['stride'])==a.ndim and all(type(s) is int and s>=0 for s in l['stride']),'actual native layout')
    stride=1;contiguous=True
    for size,s in zip(a.shape[::-1],l['stride'][::-1]):
        if size>1:contiguous &= s==stride;stride*=size
    need(l['contiguous'] is contiguous,'native contiguity')


def validate_events(row,arrays,device):
    """Full expected preorder hierarchy; unknown kinds or omitted calls refuse."""
    events=row['events'];cursor=0
    def event(kind,parent,nargs):
        nonlocal cursor
        need(cursor<len(events),'missing native attempt');e=events[cursor];cursor+=1
        need(e['index']==cursor-1 and e['kind']==kind and e['parent']==parent and e['returned'] is True
            and len(e['inputs'])==len(e['layouts'])==nargs,'native kind/order/parent/attempt-return')
        ins=[arrays[f] for f in e['inputs']];out=arrays[e['output']]
        need(all(a.dtype==np.float32 for a in ins+[out]),'original native float32, not coerced')
        for a,l in zip(ins,e['layouts'],strict=True):checked_layout(l,a,device)
        checked_layout(e['output_layout'],out,device)
        return e,ins,out
    pixel=[]
    for role in ('root','goal'):
        e,ins,hidden=event('encoder',None,1);_,pi,patch=event('pixel.patch',e['index'],1)
        exact(pi[0],ins[0],'pixel convolution input');need(patch.shape==(1,192,16,16),'native patch grid')
        need(hidden.shape==(1,257,192),'native real257 hidden')
        previous=None
        for i in range(12):
            _,li,lo=event(f'pixel.layer{i}',e['index'],1)
            need(li[0].shape==lo.shape==(1,257,192),'complete full-layer buffers')
            if previous is not None:exact(li[0],previous,'pixel layer assembly')
            previous=lo
        _,ni,no=event('pixel.finalnorm',e['index'],1)
        exact(ni[0],previous,'final ViT layer -> norm');exact(no,hidden,'native encoder final hidden')
        _,ji,z=event('projector',None,1);exact(ji[0],hidden[:,0],'true CLS/projector input')
        need(z.shape==(1,192),'Hi encoder/projector192')
        pixel.append((ins[0],z))
    def high(z,macro,parent,samples,steps):
        history=np.repeat(z[:,None],samples,axis=1).reshape(samples,1,192)
        actions=macro.reshape(1,samples,steps,32);pred=[]
        for i in range(steps):
            _,mi,mo=event('macro_to_condition',parent,1)
            seq=np.stack([actions[:,:,k].reshape(samples,32) for k in range(i+1)],axis=1)
            exact(mi[0],seq,'native macro sequence assembly')
            need(mo.shape==(samples,i+1,192),'complete macro condition')
            _,hi,ho=event('high_predictor',parent,2)
            exact(hi[0],history,'high history exact');exact(hi[1],mo,'macro condition delivery')
            _,pi,po=event('high_pred_proj',parent,1)
            need(ho.shape==(samples,i+1,192) and po.shape==(samples*(i+1),192),'complete high projection')
            exact(pi[0],ho.reshape(-1,192),'high predictor -> projection')
            nxt=po.reshape(samples,i+1,192)[:,-1:];history=np.concatenate((history,nxt),axis=1);pred.append(nxt[:,0])
        return np.stack(pred,axis=1).reshape(1,samples,steps,192)
    costs=[]
    for i in range(3):
        e,ins,out=event('high_cost',None,3);z,g,c=ins
        need(z.shape==g.shape==(1,8,192) and c.shape==(1,8,2,64) and out.shape==(1,8),'real high cost ABI')
        exact(z,np.repeat(pixel[0][1][:,None],8,axis=1),'encoded init -> high cost')
        exact(g,np.repeat(pixel[1][1][:,None],8,axis=1),'encoded goal -> high cost')
        rollout=high(pixel[0][1],c,e['index'],8,4);check_cost(out,rollout[:,:,-1],g)
        costs.append((ins,out))
    e,ins,out=event('high_selected_rollout',None,2)
    exact(ins[0],pixel[0][1],'selected rollout source init');exact(ins[1],arrays[row['macro_sequence']],'selected macros witness')
    rollout=high(ins[0],ins[1],e['index'],1,2);exact(out,rollout,'full selected native rollout reconstruction')
    exact(arrays[row['selected_rollout']],out,'saved selected rollout')
    target=out[:,0,0,:];exact(arrays[row['high_target']],target,'FIRST waypoint, not last/final goal')
    e,ins,cost=event('low_cost',None,4);zh,ah,goal,candidates=ins
    exact(zh,pixel[0][1][:,None],'official low uses single SAME current frame')
    exact(ah,np.zeros((1,1,10),np.float32),'official low zero action history, not moving nonzero context')
    exact(goal,target,'high FIRST target -> low z_subgoal BITWISE')
    exact(candidates,low_panel(),'fixed normalized low candidate panel')
    z=np.repeat(zh[:,None],3,axis=1).reshape(3,1,192);a=np.zeros((3,1,10),np.float32);pred=[]
    for i in range(2):
        a=np.concatenate((a,candidates[:,:,i].reshape(3,1,10)),axis=1)
        _,ai,ao=event('action_encoder',e['index'],1);exact(ai[0],a,'actual future action appended before encoder')
        need(ao.shape==(3,i+2,192),'native action encoder whole history')
        ctx=i+1;_,li,lo=event('low_predictor',e['index'],2)
        exact(li[0],z[:,-ctx:],'official low latent history alignment')
        exact(li[1],ao[:,-ctx:],'official low latest action embedding')
        _,pi,po=event('low_pred_proj',e['index'],1)
        need(lo.shape==(3,ctx,192) and po.shape==(3*ctx,192),'low full predictor/projection buffers')
        exact(pi[0],lo.reshape(-1,192),'low predictor -> projection input')
        nxt=po.reshape(3,ctx,192)[:,-1:];z=np.concatenate((z,nxt),axis=1);pred.append(nxt[:,0])
    actual=np.stack(pred,axis=1).reshape(1,3,2,192)
    exact(arrays[row['low']['rollout']],actual,'actual full low rollout')
    check_cost(cost,actual[:,:,-1],np.repeat(target[:,None],3,axis=1))
    for k,a in zip(('z_hist','a_hist','z_subgoal','candidates','cost'),ins+[cost],strict=True):
        exact(arrays[row['low'][k]],a,'low delivery/public arrays '+k)
    need(cursor==len(events),'no unknown/extra physical/neural native events')
    return pixel,costs,target


def validate_cem(row,arrays,costs,z,goal):
    need(row['status']=='returned' and (row['cem_reset'],row['cem_start'],row['cem_end'])==(1,1,1)
        and row['hooks_removed']==6 and len(row['cem'])==3 and row['cem_history']==[dict(step=i) for i in range(3)],'all CEM callbacks/cleanup')
    exact(arrays[row['high_initial']],np.zeros((1,2,64),np.float32),'full zero init no actor')
    receipt=row['bridge_receipt'];w=receipt['input_witness']
    need(receipt['schema']==seam.bridge_module().TAG and receipt['backend']=='actual_torch_CUDA_float32'
        and receipt['tensor_batch']==1 and receipt['cost_attempts']==receipt['cost_returns']==3
        and receipt['cost_failures']==0 and receipt['candidate_rows_attempted']==24 and len(receipt['calls'])==3
        and receipt['predict_API_calls'] is receipt['predictor_samples'] is None
        and receipt['scientific_admission'] is receipt['official_unadapted_reproduction'] is False,'actual full high bridge receipt')
    need(w['original_keys']==['planner_level','z_init','z_goal'] and w['original_metadata']==dict(planner_level='high')
        and w['solver_keys']==['z_init','z_goal'] and w['original_tensors']==w['solver_tensors'],'original metadata/slice interface')
    for k,a in (('z_init',z),('z_goal',goal)):seam.check_fingerprint(w['original_tensors'][k],a)
    mean=np.zeros((1,2,64),np.float32);std=np.ones_like(mean)
    for i,(e,call,(native_ins,native_cost)) in enumerate(zip(row['cem'],receipt['calls'],costs,strict=True)):
        need(e['step']==i and set(e['arrays'])==set(seam.KEYS) and call['attempt']==i+1 and call['batch']==1
            and call['candidate_rows']==8 and call['metadata']=='high' and call['returned'] is True,'complete native CEM/bridge chronology')
        x={k:arrays[f] for k,f in e['arrays'].items()}
        shapes=dict(candidates=(1,8,2,64),costs=(1,8),topk_vals=(1,3),topk_inds=(1,3),topk_candidates=(1,3,2,64),
            **{k:(1,2,64) for k in ('mean','var','prev_mean','prev_var')})
        need(all(a.shape==shapes[k] and a.dtype==(np.int64 if k=='topk_inds' else np.float32) for k,a in x.items()),'original CEM dtype/shape')
        c=x['candidates'];exact(c,native_ins[2],'CEM candidates -> original root')
        exact(x['costs'],native_cost,'original root cost -> callback')
        for k,a in (('candidates',c),('cost',native_cost)):seam.check_fingerprint(call[k],a)
        for k,a in zip(('z_init','z_goal'),native_ins[:2],strict=True):seam.check_fingerprint(call['expanded'][k],a,expanded=True)
        ix=x['topk_inds'];need(((ix>=0)&(ix<8)).all() and len(set(ix[0]))==3,'valid distinct elites')
        elites=c[np.arange(1)[:,None],ix]
        exact(elites,x['topk_candidates'],'actual CEM elite gather')
        exact(native_cost[np.arange(1)[:,None],ix],x['topk_vals'],'actual elite costs')
        exact(np.sort(native_cost,axis=1)[:,:3],x['topk_vals'],'actual smallest3 no neural tolerance')
        exact(x['prev_mean'],mean,'mean chain');exact(x['prev_var'],std,'sampleSTD chain')
        exact(c[:,0],mean,'mean candidate zero-position')
        seam.close_float(x['mean'],elites.astype(float).mean(axis=1),'CEM mean reducer only')
        seam.close_float(x['var'],elites.astype(float).std(axis=1,ddof=1),'CEM sampleSTD reducer only')
        mean,std=x['mean'],x['var']
    need(set(row['high_outputs'])=={'actions','mean','var','costs'},'complete solver output')
    for k,a in (('actions',mean),('mean',mean),('var',std)):exact(arrays[row['high_outputs'][k]],a,'full final CEM '+k)
    seam.close_float(arrays[row['high_outputs']['costs']],x['topk_vals'].astype(float).mean(axis=1),'elite logging rounding only')
    exact(arrays[row['macro_sequence']],mean[:,:1].reshape(1,2,32),'official receding1 block2 macro selection')


def consume(path,h,report_path,rh):
    m=verify(path,h);rp=Path(report_path);r=read(rp,rh)
    need(r['schema']==TAG+'.report' and r['manifest']==str(Path(path).resolve()) and r['manifest_sha256']==h
        and r['protocol']==protocol() and r['sources']==m['sources'] and r['completed'] is True and not r['errors']
        and r['scientific_admission'] is False and 0<=r['elapsed_seconds']<=WALL,'complete own-schema CUDA integration')
    need([(x['root_slot'],x['condition']) for x in r['rows']]==list(ORDER) and [x['index'] for x in r['rows']]==list(range(4)),
        'all four original root/condition rows, no filtering')
    guard=precision_expected()
    need(r['precision_before']==r['precision_after']==guard,'actual before/after precision')
    device=r['runtime']['device'];c=r['construction'];a=m['export']
    need(device.startswith('cuda:') and c['device']==device and c['torch']==r['runtime']['torch']
        and c['all_parameters_frozen'] is True and c['tensor_copy_attempts']==c['tensor_copy_returns']==425
        and r['model_before']==r['model_after']==c['tensor_value_hashes'],'original CUDA graph frozen/unchanged')
    _,spec,_,export=loader.authorities(a['export_report'],a['export_report_sha256'],a['export_receipt'],a['export_receipt_sha256'])
    expected={e['key']:x['array']['value_sha256'] for x,e in zip(export['rows'],spec['entries'],strict=True) if x['array'] is not None}
    need(r['model_before']==expected and c['export_report_sha256']==a['export_report_sha256']
        and c['export_receipt_sha256']==a['export_receipt_sha256'] and c['ir_sha256']==loader.IR_SHA,'original export/IR all425 identity')
    files=list(rp.parent.iterdir());need(all(p.is_file() and not p.is_symlink() for p in files)
        and {p.name for p in files}==set(r['arrays'])|{rp.name} and sum(p.stat().st_size for p in files)<=CAP,'whole saved census/cap')
    arrays={}
    for name,s in r['arrays'].items():
        need(Path(name).name==name and name.endswith('.npy'),'flat array path')
        body=loader.bytes_read(rp.parent/name,CAP);need(len(body)==s['file_bytes'] and digest(body)==s['file_sha256'],'entire NPY SHA')
        stream=io.BytesIO(body);v=np.load(stream,allow_pickle=False)
        need(stream.tell()==len(body) and v.dtype.str==s['dtype'] and list(v.shape)==s['shape']
            and v.dtype in (np.dtype('<f4'),np.dtype('<i8')) and np.isfinite(v).all() and digest(v.tobytes())==s['data_sha256'],'actual array ABI/bytes/no trailing')
        arrays[name]=v
    summaries=[]
    for row,slot in zip(r['rows'],m['slots'],strict=True):
        pixel,costs,target=validate_events(row,arrays,device);validate_cem(row,arrays,costs,pixel[0][1],pixel[1][1])
        packet=moving.load_packet(slot['blind_path'],slot['blind_sha256'])
        need([x['role'] for x in row['pixels']]==['root','goal'],'two genuine images not duplicate/cached old latents')
        for i,rgb in enumerate((packet['rgb_history'][-1],packet['goal_rgb'])):
            x=preprocess(rgb,m['statistics']);exact(pixel[i][0],x,'independent original RGB preprocessing')
            exact(arrays[row['pixels'][i]['preprocessed']],x,'preprocessed public buffer')
        for k,a in (('z_init',pixel[0][1]),('z_goal',pixel[1][1])):exact(arrays[row[k]],a,'same encoded input/goal delivery')
        summaries.append(dict(index=row['index'],root_slot=row['root_slot'],condition=row['condition'],
            high_target_sha256=digest(target.tobytes()),low_goal_sha256=digest(arrays[row['low']['z_subgoal']].tobytes()),
            low_costs=arrays[row['low']['cost']].tolist(),target_delivered_exact=True,
            RGB_source=deepcopy(slot),normalizer_raw_delivery=False))
    used=set()
    def refs(v):
        if isinstance(v,dict):
            for a in v.values():refs(a)
        elif isinstance(v,list):
            for a in v:refs(a)
        elif isinstance(v,str) and v.endswith('.npy'):used.add(v)
    refs(r['rows']);need(used==set(arrays),'all actual buffers referenced, none omitted/extra')
    actual=resource_counts(r['rows']);need(actual==r['resources'] and actual['bridge_candidate_rows']==96
        and actual['low_candidate_rows']==12,'full real resource accounting')
    for k,n in dict(encoder=8,projector=8,high_cost=12,high_selected_rollout=4,high_predictor=56,
        macro_to_condition=56,high_pred_proj=56,low_cost=4,action_encoder=8,low_predictor=8,low_pred_proj=8).items():
        need(actual['modules'][k]['attempted']==actual['modules'][k]['returned']==n,'all real module denominators '+k)
    need(actual['modules']['high_predictor']['input_rows']==392 and actual['modules']['low_predictor']['input_rows']==24,'nested sample axes')
    need(verify(path,h)==m and sha(rp)==rh,'final input/source/weights/report closure')
    return dict(schema=TAG+'.consume',status='RGB_high_FIRST_to_embedded_low_delivery_verified_integration_only',
        manifest_sha256=h,report_sha256=rh,rows=summaries,resources=actual,
        known_roots=2,fresh_roots=0,official_parity=False,common_backbone_verified=False,
        full_policy=False,physical_release=False,P1_admission=False,scientific_admission=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('manifest','manifest-sha256','output','report','report-sha256','receipt'):p.add_argument('--'+k)
    p.add_argument('--main-authorized',action='store_true');a=p.parse_args(argv)
    if a.command=='prepare':value=prepare(a.output)
    elif a.command=='verify':verify(a.manifest,a.manifest_sha256);value=dict(status='saved_prepare_verified_not_run_authorized')
    elif a.command=='run':value=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:
        value=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256);write(scoped(a.receipt),encode(value))
    print(encode(value).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
