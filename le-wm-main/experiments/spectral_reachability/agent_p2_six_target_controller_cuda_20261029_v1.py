"""Six fresh solver dispatches in one MAIN HOST CUDA job; no physical release."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time

import numpy as np
from experiments.spectral_reachability import agent_p2_target_controller_cuda_20261029_v1 as old
from experiments.spectral_reachability import agent_p2_arbitrary_latent_delivery_20261029_v1 as arbitrary

TAG='agent_p2_six_target_controller_cuda_20261029_v1'
ROOT=old.ROOT
tracker,core,delivery=old.tracker,old.core,old.delivery
need,sha,read,encode=old.need,old.sha,old.read,old.encode
ACTIVE=tuple(range(6))
ARMS=old.ARMS
CAP,RESERVE,ROOT_CAP,ROOT_RESERVE,FLOOR,WALL=1024*1024**2,128*1024**2,256*1024**2,32*1024**2,200*1024**2,1800
ALLOCATED,RESERVED=3*1024**3,4*1024**3
PROTOCOL_PATH=ROOT/'innerdocs/agent_p2_six_target_controller_protocol_20261029_v1.md'
PROTOCOL_SHA='306305ea86638d120760f7a497859319e3c716cee2b61ddf2522a46af18b7f1e'
REFS={
 'delivery':('agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json','82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62'),
 'old_prepare':('agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json','1a021116784dfaa5b4ce976f694fbfde377e6b282ce3553cb57d49a5e033c0aa'),
 'old_report':('agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json','b1358eb3cec42cdddb30b0d138c2bc2e68773f195aa985977bf9c090527382cb'),
 'old_receipt':('agent_p2_target_controller_cuda_20261029_v1_consume_MAIN01.json','34e0491ca05a1b381db3541997c6215c44f9927761d6123a559e604160f0be4a'),
 'arbitrary_prepare':('agent_p2_arbitrary_latent_delivery_20261029_v1_prepare_MAIN01.json','3aaffe495a3cfddda0d88403a39b1c937121942c64f79be5a574499579384ad0'),
 'arbitrary_report':('agent_p2_arbitrary_latent_delivery_20261029_v1_numerical_MAIN01/report.json','7431e2b851856fbf5e24d2db3b3762bb6298532be7316e599d6bd32de739c278'),
 'arbitrary_receipt':('agent_p2_arbitrary_latent_delivery_20261029_v1_consume_MAIN01.json','1e47155d480c23a0bda9cae50e633f0fe7ef56c992d3f489a31e255e43286ffc'),
 'arbitrary_completion':('agent_p2_arbitrary_latent_delivery_20261029_v1_numerical_MAIN01/completion.json','7d2edf45b88a85c9d8e62ea4de63333e3520c204b354ab2cc0f6e07c29839007')}
PINS={Path(old.__file__):'44b87adb98a6ac60e755eee3e5fc544a194ce27c7ddf367f7cfb834d586ad946',
 ROOT/'tests/test_agent_p2_target_controller_cuda_20261029_v1.py':'7f865986d20300b7431c03168cc53a990b253b685acc7b2085078e0b8de37254',
 ROOT/'innerdocs/agent_p2_target_controller_cuda_20261029_v1.md':'a2af60566728e6f306f403795a199cb56b1b83b188a19a5c943762bd390e5df0',
 Path(arbitrary.__file__):'709959e6b95b4cbea65d3607b5fccdc571a85f05551a3234bcd9114d5b26f600',
 ROOT/'tests/test_agent_p2_arbitrary_latent_delivery_20261029_v1.py':'df801093993b3d3464202b29c53572a1a6c6f98bb65d6c16057ecca14f51e036',
 ROOT/'innerdocs/agent_p2_arbitrary_latent_delivery_20261029_v1.md':'29048dc54d11663aa52653776dfe866e23732ba49df3fec7e31fe3f99a4090a9',
 PROTOCOL_PATH:PROTOCOL_SHA}
AXES=('predict_API','predictor_samples','AE_attempts','AE_dispatched','AE_returned','AE_captured','AE_rows','AE_tokens',
 'predictor_attempts','predictor_dispatched','predictor_returned','predictor_captured','predictor_context_tokens',
 'lowproj_attempts','lowproj_dispatched','lowproj_returned','lowproj_captured','lowproj_rows',
 'candidate_cost_calls','candidate_trajectories','mean_paid_replays','replay_trajectories','AD_primal','J','VJP','backward','FLOPs')


def refs():return {k:dict(path=str(ROOT/'results'/p),sha256=h) for k,(p,h) in REFS.items()}


def protocol():
    return dict(schema=TAG+'.protocol',approved_protocol_sha256=PROTOCOL_SHA,roots=[0,1],condition='off',
        active_slots=list(ACTIVE),arms=list(ARMS),denominator=6,missing=0,execution='one_graph_six_fresh_seed0_solvers_serial_0_to_5',
        spec=json.loads(encode(core.asdict(tracker.spec()))),planned_predict_API=72,planned_predictor_samples=324,axes=list(AXES),
        encoder=0,high=0,physics=0,SVD=0,FLOPs=None,raw_inverse=False,raw_deployment_authorized=False,
        scientific_admission=False,P1_admission=False,P2_admission=False,E1_admission=False,
        known_only=True,fresh=False,prior_physical_time=None,time_matched=False,fair_baseline=False,equal_generation_budget=False,
        output_cap=CAP,footer_reserve=RESERVE,per_root_cap=ROOT_CAP,per_root_reserve=ROOT_RESERVE,
        floor=FLOOR,whole_wall=WALL,per_slot_wall=300,cuda_allocated=ALLOCATED,cuda_reserved=RESERVED)


def merge(dst,src):
    for p,h in src.items():need(p not in dst or dst[p]==h,'consistent complete closure');dst[p]=h


def closure():
    out=old.closure()
    merge(out,arbitrary.source_seals())
    for p,h in PINS.items():need(sha(p)==h,'frozen parent/protocol');merge(out,{str(p):h})
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name;merge(out,{str(p):sha(p)})
    return out


def authenticate_arbitrary():
    """MAIN-only workflow call: pure numerical/saved consume, retained exclusive proof."""
    started=time.monotonic();rr=refs()
    need(all(sha(p)==h for p,h in PINS.items()),'source-first authentication')
    for key in ('arbitrary_prepare','arbitrary_report','arbitrary_receipt','arbitrary_completion'):
        need(sha(rr[key]['path'])==rr[key]['sha256'],'actual companion authority')
    need(shutil.disk_usage(ROOT/'results').free>=FLOOR+32*1024**2,'authentication output/floor')
    folder=Path(tempfile.mkdtemp(prefix=TAG+'_auth_',dir=ROOT/'results'))
    dest=folder/'receipt.json'
    returned=arbitrary.consume(rr['arbitrary_prepare']['path'],rr['arbitrary_prepare']['sha256'],
        Path(rr['arbitrary_report']['path']).parent,rr['arbitrary_report']['sha256'],dest)
    need(returned==dict(path=str(dest.absolute()),sha256=sha(dest)),'consume returns ref, not receipt body')
    expected=read(rr['arbitrary_receipt']['path'],rr['arbitrary_receipt']['sha256'])
    need(read(dest,returned['sha256'])==expected and returned['sha256']==rr['arbitrary_receipt']['sha256'],'exact whole persisted receipt reconstruction')
    proof=dict(schema=TAG+'.arbitrary_authentication',authorities=rr,receipt=returned,
        elapsed_seconds=time.monotonic()-started,verification_cpu_calls=expected['verification_cpu_calls'],
        sources={str(p):sha(p) for p in PINS},neural=0,physics=0,scope='companion verify+whole saved arithmetic consume+exclusive receipt fsync/hash; proof write excluded')
    need(proof['sources']=={str(p):h for p,h in PINS.items()},'authentication sources unchanged')
    tracker.durable(folder/'proof.json',encode(proof))
    return dict(path=str((folder/'proof.json').resolve()),sha256=sha(folder/'proof.json'))


def check_authentication(ref,expected):
    p=Path(ref['path']);need(p.name=='proof.json' and p.parent.parent.resolve()==ROOT/'results' and
        p.parent.name.startswith(TAG+'_auth_'),'own retained authentication directory')
    proof=read(p,ref['sha256']);need(proof['schema']==TAG+'.arbitrary_authentication' and proof['authorities']==refs(),'full authentication join')
    need(proof['sources']=={str(p):h for p,h in PINS.items()} and all(sha(p)==h for p,h in proof['sources'].items()),'auth source pins')
    need(proof['receipt']['path']==str(p.parent/'receipt.json') and proof['receipt']['sha256']==refs()['arbitrary_receipt']['sha256'] and
         read(proof['receipt']['path'],proof['receipt']['sha256'])==expected,'full authentication receipt content')
    need(math.isfinite(proof['elapsed_seconds']) and proof['elapsed_seconds']>=0 and
         proof['verification_cpu_calls']==expected['verification_cpu_calls'] and proof['neural']==proof['physics']==0,'authentication CPU fees disclosed')
    return proof


def assemble(d,cm,report,ac,ap,get_old):
    """Six-only schema; untouched old row embedded as source, no planner metadata."""
    values=old.select_payloads(d)
    need(cm['delivery_artifact']==d and len(report['rows'])==6,'entire original delivery binding')
    need(ac['schema']==arbitrary.NAME+'.receipt' and ac['complete'] is True and ac['denominator']==2 and
         ac['status']=='all2_arbitrary_FIRST_numerically_verified_NOT_tracked' and len(ac['plans'])==len(ap['slots'])==2,'actual full arbitrary delivery')
    need(ac['parent_delivery']==ap['parent_delivery']==dict(path=refs()['delivery']['path'],sha256=refs()['delivery']['sha256']),'shared original parent')
    need(ac['prepare_sha256']==refs()['arbitrary_prepare']['sha256'] and ac['report_sha256']==refs()['arbitrary_report']['sha256'] and
         ac['completion_sha256']==refs()['arbitrary_completion']['sha256'],'full actual companion authority join')
    rows=[]
    for i in ACTIVE:
        root=i//3;original=d['rows'][i]
        if i in old.ACTIVE:
            rr=report['rows'][i];need(rr['status']=='returned' and rr['slot']==i and rr['root']==root and rr['arm']==ARMS[i%3],'original actual returned slot')
            for k,v in values[i].items():
                need(delivery.same(get_old(i,k),v),'original four payload byte identity including zero sign')
            payload=original['controller_payload'];origin=dict(kind='unchanged_old_delivery_row',row=original,original_input_files=rr['input_files'])
        else:
            plan=ac['plans'][root];slot=ap['slots'][root]
            need(plan['root']==slot['root']==root and plan['condition']==slot['condition']=='off' and
                 plan['status']=='delivered' and plan['failure'] is None and plan['source_time'] is None,'fixed arbitrary FIRST not selected by result')
            need(all(plan[k]==slot[k]==original[k] for k in ('task_id',)) and
                 all(slot[k]==original[k] for k in ('task_sha256','blind_sha256')),'whole original task/source identity')
            payload=plan['controller_payload'];v=delivery.validate_payload(payload)
            for k,hk in (('z','history_sha256'),('a','action_history_sha256')):
                need(delivery.same(v[k],values[3*root][k]) and core.digest(v[k].tobytes())==plan[hk]==original[hk],'new target same byte-exact causal history')
            need(core.digest(v['target'].tobytes())==plan['target32_sha256'],'final FIRST single cast target bytes')
            need(delivery.same(v['z'],np.asarray(slot['z'],np.float32)) and delivery.same(v['a'],np.asarray(slot['a'],np.float32)),'companion prepare context')
            origin=dict(kind='arbitrary_receipt_plan',pointer=f'/plans/{root}',plan=plan,source_time=None)
        vv=delivery.validate_payload(payload)
        rows.append(dict(slot=i,root=root,condition='off',arm=ARMS[i%3],task_id=original['task_id'],
            task_sha256=original['task_sha256'],blind_sha256=original['blind_sha256'],
            source=origin,controller_payload=payload,input_sha256={k:core.digest(v.tobytes()) for k,v in vv.items()}))
    return rows


def select_payloads(m):
    need(m['schema']==TAG+'.prepare' and len(m['slots'])==6,'independent six schema')
    out={}
    for i,row in enumerate(m['slots']):
        need((row['slot'],row['root'],row['condition'],row['arm'])==(i,i//3,'off',ARMS[i%3]),'all six fixed order')
        values=delivery.validate_payload(row['controller_payload'])
        need(row['input_sha256']=={k:core.digest(v.tobytes()) for k,v in values.items()},'all actual input bits')
        out[i]=values
    for root in (0,1):
        for i in (3*root+1,3*root+2):
            for k in ('z','a'):need(delivery.same(out[3*root][k],out[i][k]),'common context no privileged witness')
    return out


def build(authentication):
    rr=refs();docs={k:read(v['path'],v['sha256']) for k,v in rr.items()}
    d,cm,r,c=(docs[k] for k in ('delivery','old_prepare','old_report','old_receipt'))
    need(d==delivery.build() and cm==old.build(rr['delivery']['path'],rr['delivery']['sha256']),'whole saved original prepare reconstructed')
    need(r['manifest_sha256']==rr['old_prepare']['sha256'] and Path(r['manifest']).resolve()==Path(rr['old_prepare']['path']),'old report authority')
    oldcheck=old.analyze_saved(cm,r,Path(rr['old_report']['path']).parent)
    oldcheck.update(report_sha256=rr['old_report']['sha256'],manifest_sha256=rr['old_prepare']['sha256'])
    need(oldcheck==c,'entire original strict receipt reconstructed; no neural execution')
    ap,ar,ac=(docs[k] for k in ('arbitrary_prepare','arbitrary_report','arbitrary_receipt'))
    need(ar['rows']==ac['plans'] and ar['prepare_sha256']==ac['prepare_sha256'] and
         Path(ar['prepared']).resolve()==Path(rr['arbitrary_prepare']['path']) and docs['arbitrary_completion']['complete'] is True and
         docs['arbitrary_completion']['report_sha256']==rr['arbitrary_report']['sha256'],'actual numerical report/complete receipt exact join')
    auth=check_authentication(authentication,ac)
    sources=dict(cm['sources']);merge(sources,ap['bindings']);merge(sources,closure())
    merge(sources,{v['path']:v['sha256'] for v in rr.values()});merge(sources,{authentication['path']:authentication['sha256'],auth['receipt']['path']:auth['receipt']['sha256']})
    def get_old(i,k):
        desc=r['rows'][i]['input_files'][k]
        need(desc['file']==f'slot{i}_input_{k}.npy' and desc['root']==i//3,'original exact file identity')
        p=Path(rr['old_report']['path']).parent/desc['file'];merge(sources,{str(p):desc['file_sha256']})
        return old.load_array(p.parent,desc)
    slots=assemble(d,cm,r,ac,ap,get_old)
    need(ap['inherited_expected_state_sha256']==arbitrary.digest(cm['expected_state']),'same425 inherited graph')
    need(all(sha(p)==h for p,h in sources.items()),'entire source/model/input closure current')
    return dict(schema=TAG+'.prepare',protocol=protocol(),authorities=rr,upstream_authentication=authentication,
        full_receipts=dict(old=c,arbitrary=ac),slots=slots,export=cm['export'],expected_state=cm['expected_state'],
        expected_construction=cm['expected_construction'],sources=sources)


def own_output(path):
    p=Path(path);need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'exclusive own result child');return p


def prepare(output):
    p=own_output(output);need(not p.exists(),'never overwrite prepare')
    auth=authenticate_arbitrary();m=build(auth);b=encode(m)
    need(len(b)<=32*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'prepare bounded metadata/floor')
    tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p),authentication=auth)


def verify(path,h):
    m=read(path,h);auth=authenticate_arbitrary()
    need(m==build(m['upstream_authentication']),'whole six prepare reconstructed')
    check_authentication(auth,m['full_receipts']['arbitrary'])
    return m,auth


class Store(tracker.Store):
    def __init__(self,path):
        self.path=own_output(path);self.start=time.monotonic();self.used=0;self.root_used={0:0,1:0};self.files={};self.journal=[]
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new output/floor')
        self.path.mkdir()

    def guard(self,n=0,root=None,footer=False,final=False):
        need(final or footer or time.monotonic()-self.start<=WALL,'whole1800 cooperative wall')
        need(self.used+n<=CAP-(0 if final else RESERVE),'whole1024MiB including128reserve')
        need(shutil.disk_usage(self.path).free>=FLOOR+n,'floor200 before write/dispatch')
        if root is not None:need(root in (0,1) and self.root_used[root]+n<=ROOT_CAP-(0 if footer else ROOT_RESERVE),'root256MiB incl32reserve')


def load_array(folder,d):
    name=d['file'];need(re.fullmatch(r'slot[0-5]_(?:input_(?:z|a|target)|array_[0-9]+)\.npy',name) is not None,'six-only flat NPY names')
    b=tracker.parity.rgb.loader.bytes_read(Path(folder)/name,ROOT_CAP)
    need(len(b)==d['file_bytes'] and hashlib.sha256(b).hexdigest()==d['file_sha256'],'NPY bytes/SHA')
    f=io.BytesIO(b);need(np.lib.format.read_magic(f)==(1,0),'NPYv1')
    shape,order,dtype=np.lib.format.read_array_header_1_0(f)
    need(list(shape)==d['shape'] and dtype.str==d['dtype'] and dtype.str in ('<f4','<f8','<i8') and
         not order and math.prod(shape)*dtype.itemsize==len(b)-f.tell(),'dtype/header/EOF')
    f.seek(0);a=np.load(f,allow_pickle=False);need(np.isfinite(a).all() and core.digest(a.tobytes())==d['data_sha256'],'data SHA/finite');return a


def accounting27(tr,arrays):
    """Named local ledger: attempts are not replaced by planned successful costs."""
    out={k:0 for k in AXES};out['FLOPs']=None
    out.update(predict_API=tr['predict_API_debited'],predictor_samples=tr['predictor_samples_debited'])
    for kind,prefix in (('action_encoder','AE'),('low_predictor','predictor'),('low_pred_proj','lowproj')):
        es=[e for e in tr['events'] if e['kind']==kind];dispatched=[e for e in es if e['dispatched']]
        out[prefix+'_attempts']=len(es);out[prefix+'_dispatched']=len(dispatched)
        out[prefix+'_returned']=sum(e['module_returned'] for e in es);out[prefix+'_captured']=sum(e['returned'] for e in es)
        rows=sum(e['shapes'][0][0] for e in dispatched)
        tokens=sum(e['shapes'][0][0]*e['shapes'][0][1] for e in dispatched if len(e['shapes'][0])==3)
        if prefix=='AE':out.update(AE_rows=rows,AE_tokens=tokens)
        elif prefix=='predictor':out['predictor_context_tokens']=tokens
        else:out['lowproj_rows']=rows
    out['candidate_cost_calls']=len(tr['searches']);out['candidate_trajectories']=sum(arrays[x['future']].shape[1] for x in tr['searches'])
    out['mean_paid_replays']=len(tr['replays']);out['replay_trajectories']=sum(arrays[x['future']].shape[1] for x in tr['replays'])
    for k in ('AD_primal','J','VJP','backward'):out[k]=tr['resources'][k]
    need(set(out)==set(AXES) and len(out)==27,'exact named27 axes')
    return out


def aggregate(rows):
    return dict(known_subtotals={k:sum(r['axes'][k] for r in rows if r['axes'][k] is not None) for k in AXES if k!='FLOPs'},
        actual={k:None if any(r['axes'][k] is None for r in rows) else sum(r['axes'][k] for r in rows) for k in AXES},
        unknown_slots={k:[r['slot'] for r in rows if r['axes'][k] is None] for k in AXES},FLOPs=None)


def initial_rows():
    return [dict(slot=i,root=i//3,arm=ARMS[i%3],condition='off',status='not_started',arrays={},local_check=None,
        debits=dict(known=True,predict_API=0,predictor_samples=0),axes={k:None if k=='FLOPs' else 0 for k in AXES}) for i in ACTIVE]


def persist_inputs(store,rows,values):
    need(tuple(values)==ACTIVE and len(rows)==6,'all six before construction')
    for i in ACTIVE:rows[i]['input_files']={k:store.array(f'slot{i}_input_{k}.npy',a,i//3) for k,a in values[i].items()}
    return store.put('input_seal.json',encode([dict(slot=i,inputs=rows[i]['input_files']) for i in ACTIVE]))


def persist_trace(store,row,tr,arrays):
    row['trace']=store.put(f"slot{row['slot']}_trace.json",encode(tr),row['root'],footer=True)
    row['arrays']={}
    for key,a in arrays.items():
        need(re.fullmatch(r'array_[0-9]+',key) is not None,'core array key')
        row['arrays'][key]=store.array(f"slot{row['slot']}_{key}.npy",a,row['root'])


def require_host(tr):
    need(tr['backend']=='host_cuda_installed_CEM' and tr.get('host_tensor_seam_checked') is True,'HOST only; fixture never promoted')


def execute_slots(store,rows,values,invoke,snapshot,expected_state):
    need(all(set(rows[i].get('input_files',{}))=={'z','a','target'} for i in ACTIVE),'all18 input arrays sealed first')
    for i in ACTIVE:
        store.guard();row=rows[i];started=time.monotonic();row['status']='attempted'
        row['dispatch_model_before']=snapshot();need(row['dispatch_model_before']==expected_state,'slot425 before')
        row['axes']={k:None for k in AXES};row['debits']=dict(known=False,predict_API=None,predictor_samples=None)
        tr,arrays=invoke({k:v.copy() for k,v in values[i].items()})
        row['axes']=accounting27(tr,arrays);row['debits']=dict(known=True,predict_API=tr['predict_API_debited'],predictor_samples=tr['predictor_samples_debited'])
        row['status']='partial';persist_trace(store,row,tr,arrays)
        require_host(tr)
        row['local_check']=core.validate_saved(tr,arrays)
        row['dispatch_model_after']=snapshot();need(row['dispatch_model_after']==row['dispatch_model_before']==expected_state,'slot425 after')
        row['capture']=dict(saved_array_count=len(arrays),saved_payload_bytes=tr['array_payload_bytes'],capture_sync_wall_seconds=tr['capture_sync_wall_seconds'],
            all_cpu_transfer_calls=None,core_wall_seconds=tr['whole_core_wall_seconds'],host_entry_wall_seconds=tr['host_entry_through_postguard_wall_seconds'])
        row['wall_seconds']=time.monotonic()-started;need(row['wall_seconds']<=300,'outer slot300 incl guards/persistence')
        row['status']='returned'


def run_cuda(manifest,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before IO/torch')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS pinned workspace')
    store=Store(output);caps=tracker.GPUResourceGuard(store);graph=t=None;precision=handler=None
    r=dict(schema=TAG+'.report',manifest=str(Path(manifest).resolve()),manifest_sha256=mh,protocol=protocol(),
        completed=False,errors=[],rows=initial_rows(),files=store.files,write_journal=store.journal,
        input_seal=None,construction=None,construction_file=None,model_before=None,model_after=None,sources_after=None,
        runtime_authentication=None,raw_inverse=False,scientific_admission=False)
    def timeout(signum,frame):raise TimeoutError('whole1800 alarm; not CUDA kernel preemption')
    try:
        need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'standalone no previous alarm')
        handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,WALL)
        m,r['runtime_authentication']=verify(manifest,mh);store.put('dispatch_manifest.json',encode(m))
        values=select_payloads(m);r['input_seal']=persist_inputs(store,r['rows'],values)
        import torch as t
        need(Path(t.__file__).resolve()==ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py' and t.cuda.is_available(),'exact venv HOST CUDA only')
        from torch.utils._python_dispatch import TorchDispatchMode
        precision=(t.are_deterministic_algorithms_enabled(),t.backends.cuda.matmul.allow_tf32,t.backends.cudnn.allow_tf32)
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        t.cuda.set_device(0);t.cuda.reset_peak_memory_stats(0)
        def gpu():caps.observe(int(t.cuda.max_memory_allocated(0)),int(t.cuda.max_memory_reserved(0)))
        class Bounds(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                gpu();v=func(*args,**({} if kwargs is None else kwargs));gpu();return v
        with t.enable_grad(),t.autocast('cuda',enabled=False),Bounds():
            e=m['export'];graph,r['construction']=tracker.parity.rgb.loader.construct_cuda(e['report'],e['report_sha256'],e['receipt'],e['receipt_sha256'],main_authorized=True)
            tracker.validate_construction(r['construction'],m)
            r['construction_file']=store.put('construction.json',encode(r['construction']))
            def snapshot():return tracker.state_hashes(graph,t)
            r['model_before']=snapshot();need(r['model_before']==m['expected_state'],'all425 before')
            def invoke(p):
                need(set(p)=={'z','a','target'},'three-key boundary, no witness/warmstart')
                xs={k:t.from_numpy(v.copy()).to('cuda:0') for k,v in p.items()}
                return core.track_cuda(graph,xs['z'],xs['a'],xs['target'],spec=tracker.spec(),expected_state=m['expected_state'],main_authorized=True)
            execute_slots(store,r['rows'],values,invoke,snapshot,m['expected_state']);gpu()
        r['model_after']=tracker.state_hashes(graph,t);need(r['model_after']==r['model_before']==m['expected_state'],'all425 after')
        r['sources_after']={p:sha(p) for p in m['sources']};need(r['sources_after']==m['sources'],'full source/input closure after')
        store.guard();r['completed']=True
    except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)[:3000]))
    finally:
        if handler is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
        if graph is not None and r['model_after'] is None:
            try:r['model_after']=tracker.state_hashes(graph,t)
            except BaseException as ex:r['errors'].append(dict(stage='final_model',message=str(ex)[:1000]))
        if 'm' in locals() and r['sources_after'] is None:
            try:r['sources_after']={p:sha(p) for p in m['sources']}
            except BaseException as ex:r['errors'].append(dict(stage='final_sources',message=str(ex)[:1000]))
        if precision is not None:
            t.use_deterministic_algorithms(precision[0]);t.backends.cuda.matmul.allow_tf32=precision[1];t.backends.cudnn.allow_tf32=precision[2]
        r['accounting27']=aggregate(r['rows']);r.update(old.totals(r['rows']))
        r['side_axes']=dict(encoder=0,high=0,physics=0,SVD=0,model_state_copies=0 if r['construction'] is None else r['construction']['tensor_copy_returns'],
            original_Hi_scaler_certified=False,prior_physical_time=None,independent_gradient_recompute=False)
        r['cuda_resources']=caps.snapshot();r['whole_wall_seconds_before_report']=time.monotonic()-store.start
        r['file_bytes_before_report']=store.used;r['root_file_bytes']=store.root_used
        store.put('report.json',encode(r),final=True)
    report=store.path/'report.json';rh=sha(report);elapsed=time.monotonic()-store.start
    done=dict(schema=TAG+'.completion',report_sha256=rh,complete=r['completed'] and r['errors']==[] and elapsed<=WALL,
        whole_wall_through_report_fsync_hash=elapsed,boundary='Store start before verify through report fsync/hash; completion acknowledgement excluded')
    store.put('completion.json',encode(done),final=True)
    return dict(report=str(report.resolve()),sha256=rh,completed=done['complete'])


def validate_files(r,folder):
    folder=Path(folder);files=r['files']
    need({p.name for p in folder.iterdir()}==set(files)|{'report.json','completion.json'},'all output file census')
    for name,d in files.items():
        p=folder/name;need(Path(name).name==name and d['file']==name and p.stat().st_size==d['file_bytes'] and sha(p)==d['file_sha256'],'file bytes/SHA')
    need(sum(d['file_bytes'] for d in files.values())==r['file_bytes_before_report'] and sum(p.stat().st_size for p in folder.iterdir())<=CAP,'whole output1024MiB')
    need(len(r['write_journal'])==len(files) and {e['name']:e for e in r['write_journal']}==
         {n:dict(name=n,root=d['root'],bytes=d['file_bytes'],returned=True) for n,d in files.items()},'all write attempts/returns')
    for root in (0,1):need(sum(d['file_bytes'] for d in files.values() if d['root']==root)==r['root_file_bytes'][str(root)]<=ROOT_CAP,'root256 census')


def analyze_saved(m,r,folder):
    need(r['schema']==TAG+'.report' and r['completed'] is True and r['errors']==[],'all six complete HOST slots')
    need(r['protocol']==protocol() and r['raw_inverse'] is False and r['scientific_admission'] is False,'no scientific/raw release')
    tracker.validate_construction(r['construction'],m)
    need(r['model_before']==r['model_after']==m['expected_state'] and r['sources_after']==m['sources'],'all425/source guards')
    check_authentication(r['runtime_authentication'],m['full_receipts']['arbitrary'])
    validate_files(r,folder);files=r['files'];referenced=set()
    def member(d):
        need({k:d[k] for k in ('file','file_bytes','file_sha256','root')}==files[d['file']],'file descriptor inventory');referenced.add(d['file'])
    for name,value in (('dispatch_manifest.json',m),('construction.json',r['construction'])):
        d=files[name];member(d);need(read(Path(folder)/name,d['file_sha256'])==value,'exact durable '+name)
    need(r['construction_file']==files['construction.json'] and r['input_seal']==files['input_seal.json'],'bound construction/input seal')
    member(r['input_seal']);need(read(Path(folder)/'input_seal.json',r['input_seal']['file_sha256'])==
        [dict(slot=i,inputs=r['rows'][i]['input_files']) for i in ACTIVE],'entire input seal')
    values=select_payloads(m);need(len(r['rows'])==6,'six denominator')
    checks=[];journal=[e['name'] for e in r['write_journal']];positions={n:i for i,n in enumerate(journal)}
    need(positions['dispatch_manifest.json']<positions['input_seal.json']<positions['construction.json'],'input seal before construction')
    last=positions['construction.json']
    for i,row in enumerate(r['rows']):
        need((row['slot'],row['root'],row['condition'],row['arm'])==(i,i//3,'off',ARMS[i%3]),'six fixed order')
        need(row['status']=='returned' and 0<=row['wall_seconds']<=300,'all six newly returned')
        need(row['dispatch_model_before']==row['dispatch_model_after']==m['expected_state'],'slot425 guards')
        need(set(row['input_files'])=={'z','a','target'},'onlypayload persisted')
        for k,v in values[i].items():
            d=row['input_files'][k];member(d)
            need(d['file']==f'slot{i}_input_{k}.npy' and d['root']==i//3 and positions[d['file']]<positions['input_seal.json'],'every input presealed')
            need(delivery.same(v,load_array(folder,d)),'exact delivered bytes')
        td=row['trace'];member(td);need(td['file']==f'slot{i}_trace.json' and td['root']==i//3 and positions[td['file']]>last,'all six serial fresh traces')
        tr=read(Path(folder)/td['file'],td['file_sha256']);arrays={}
        require_host(tr)
        need(tr['spec']==protocol()['spec'] and tr['model_before']==tr['model_after']==m['expected_state'],'per-slot original spec/state')
        need(all(m['sources'].get(p)==h for p,h in tr['sources_before'].items()),'core full source closure presealed')
        for key,d in row['arrays'].items():
            member(d);need(d['file']==f'slot{i}_{key}.npy' and d['root']==i//3 and positions[d['file']]>positions[td['file']],'no cross-slot or old file reuse')
            arrays[key]=load_array(folder,d)
        last=max(positions[d['file']] for d in row['arrays'].values())
        for k,v in values[i].items():need(delivery.same(v,arrays[tr['inputs'][k]]),'only exact payload entered core')
        lc=core.validate_saved(tr,arrays);need(lc==row['local_check'],'complete local_check reconstructed')
        need(row['axes']==accounting27(tr,arrays) and tr['predict_API_debited']==12 and tr['predictor_samples_debited']==54,'27 axes and full mean replay debits')
        need(row['debits']==dict(known=True,predict_API=12,predictor_samples=54),'actual12/54 not borrowed')
        capture=dict(saved_array_count=len(arrays),saved_payload_bytes=tr['array_payload_bytes'],capture_sync_wall_seconds=tr['capture_sync_wall_seconds'],
            all_cpu_transfer_calls=None,core_wall_seconds=tr['whole_core_wall_seconds'],host_entry_wall_seconds=tr['host_entry_through_postguard_wall_seconds'])
        need(row['capture']==capture and 0<=capture['capture_sync_wall_seconds']<=capture['core_wall_seconds']<=300 and
             capture['core_wall_seconds']<=capture['host_entry_wall_seconds']<=row['wall_seconds'],'actual capture/time scopes')
        checks.append(dict(slot=i,root=i//3,arm=ARMS[i%3],local_check=lc))
    need(referenced==set(files),'no unconsumed files')
    need(r['accounting27']==aggregate(r['rows']) and r['accounting27']['actual']['predict_API']==72 and
         r['accounting27']['actual']['predictor_samples']==324,'all six fresh72/324')
    need(old.totals(r['rows'])=={k:r[k] for k in old.totals(r['rows'])},'aggregate debits not planned substitution')
    need(r['side_axes']==dict(encoder=0,high=0,physics=0,SVD=0,model_state_copies=425,original_Hi_scaler_certified=False,
         prior_physical_time=None,independent_gradient_recompute=False),'remaining construction/no-neural-encoder axes')
    gpu=r['cuda_resources'];need(gpu['dispatch_guard_calls']>0 and 0<gpu['max_allocated']<=ALLOCATED and 0<gpu['max_reserved']<=RESERVED and
         gpu['allocated_cap']==ALLOCATED and gpu['reserved_cap']==RESERVED,'actual CUDA bounds')
    need(0<=r['whole_wall_seconds_before_report']<=WALL,'whole1800 before footer')
    return dict(schema=TAG+'.receipt',status='all6_new_target_controller_saved_verified',denominator=dict(total=6,returned=6,missing=0),
        local_checks=checks,accounting27=r['accounting27'],predict_API=72,predictor_samples=324,side_axes=r['side_axes'],
        fair_baseline=False,equal_generation_budget=False,time_matched=False,FLOPs=None,raw_inverse=False,scientific_admission=False,
        authority_import_is_not_independent_gradient=True)


def consume(report,rh,output):
    r=read(report,rh);m,authentication=verify(r['manifest'],r['manifest_sha256'])
    folder=Path(report).parent;c=read(folder/'completion.json',sha(folder/'completion.json'))
    need(c['schema']==TAG+'.completion' and c['report_sha256']==rh and c['complete'] is True and
        r['whole_wall_seconds_before_report']<=c['whole_wall_through_report_fsync_hash']<=WALL and
        c['boundary']=='Store start before verify through report fsync/hash; completion acknowledgement excluded','actual inclusive complete job')
    out=analyze_saved(m,r,folder);out.update(report_sha256=rh,manifest_sha256=r['manifest_sha256'],completion_sha256=sha(folder/'completion.json'),
        saved_consume_authentication=authentication,upstream_arbitrary_receipt=m['authorities']['arbitrary_receipt'])
    p=own_output(output);b=encode(out);need(len(b)<=32*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'receipt bounded metadata/floor')
    tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('protocol','prepare','verify','run','consume'))
    for k in ('manifest','report','output'):p.add_argument('--'+k,type=Path)
    for k in ('manifest','report'):p.add_argument('--'+k+'-sha256')
    p.add_argument('--main-authorized',action='store_true');a=p.parse_args(argv)
    if a.command=='protocol':out=protocol()
    elif a.command=='prepare':out=prepare(a.output)
    elif a.command=='verify':_,auth=verify(a.manifest,a.manifest_sha256);out=dict(verified=True,authentication=auth)
    elif a.command=='run':out=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:out=consume(a.report,a.report_sha256,a.output)
    print(encode(out).decode())


if __name__=='__main__':main()
