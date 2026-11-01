"""Fixed archive/nativeoff/nativeon action discrimination, MAIN HOST CUDA only."""
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
import signal
import time
import zlib

import numpy as np
from experiments.spectral_reachability import p1_archive_target_overlay_fidelity_cuda_20261027_v1 as old

NAME='agent_archive_action_discrimination_20261028_v1'
ROOT,enc,fidelity=old.ROOT,old.enc,old.fidelity
EPISODES,CONDITIONS=old.EPISODES,old.CONDITIONS
KINDS=('H1_10_to_15','teacher_forced_15_to_20')
ARMS=('factual','zero_outgoing','reverse_time_outgoing','swap_xy_outgoing')
ORDER=('action_encoder','predict','predictor','pred_proj')
EXPECTED=dict(encoder=0,projector=0,predict=48,predictor=48,action_encoder=48,pred_proj=48)
OLD_SOURCE_SHA='f536cf5fe9baa4c652b17888f6ff0112bccc5aa241c567581a034b5ec8ccc31c'
OLD_REPORT_SHA='d8cbd4b6e8ff1497421e6cdaf8644f4b39ed2cd2b695bb10cd15b57fbae121f0'
OLD_PREP_SHA='f45d89d60e6da7e6ef97b1bc33ae8c3d1f335ddc92405a6e73873fde5ee2f25f'
OLD_INDEPENDENT_SHA='d1c93f3c4f79d1906a6e8fe7f1e0d2c1fe8c260e051b2f7179f39e7d5268766f'
OLD_REPORT=ROOT/'results/p1_archive_target_overlay_fidelity_cuda_20261027_v1_cuda_main01/report.json'
OLD_PREP=ROOT/'results/p1_archive_target_overlay_fidelity_cuda_20261027_v1_prepare_main01.json'
OLD_INDEPENDENT=ROOT/'results/p1_archive_target_overlay_fidelity_independent_analysis_main01.json'
CAP,FLOOR,FINAL=32*1024**2,200*1024**2,2*1024**2
WALL,PREP_CAP=600,2*1024**2
REQUEST_KEYS=old.REQUEST_KEYS|{'arm'}
require,sha,canonical,bits=old.require,old.sha,old.canonical,old.bits


def digest(raw):return hashlib.sha256(raw).hexdigest()


def read(path,expected=None,cap=CAP):
    p=Path(path)
    require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)) and p.stat().st_size<=cap,'bounded nonsymlink JSON')
    before=sha(p);require(expected is None or before==expected,'actual external SHA')
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    value=json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda x:require(False,'nonfinite JSON'))
    require(sha(p)==before,'JSON byte stability');return value


def contract():
    return dict(schema=NAME,episodes=list(EPISODES),conditions=list(CONDITIONS),kinds=list(KINDS),arms=list(ARMS),
        base_requests=12,request_denominator=48,batch=1,counts=EXPECTED,
        API_samples={k:v for k,v in EXPECTED.items()},input_tokens_per_call=3,projected_token_samples=144,
        order='episode/condition/window/arm; no fresh H2 or dedup',
        outgoing='change only final raw5x2: factual / literal float32 zero / reverse five time rows / exchange x,y columns',
        history='original factual full3 latent context and first two raw5 packets bitwise unchanged',
        normalization='original float32 mean/std; rebuild all3 tokens once per request; no extra /100/clip/tanh',
        sealing='all48 full3 outputs fsynced and journal seal before reading labels for scoring; predictor whitelist excludes label/goal',
        labels='same-condition saved actual successor15 or20, not a physical counterfactual ground truth',
        factual_gate='all12 factual full3 outputs must reproduce old saved outputs bitwise, no tolerance or retry',
        scoring='native192 float64 MSE; each nonfactual minus factual, output sensitivity, archive-relative contrasts; no winner selection',
        inherited=dict(encoder=30,projector=30,predict=24,action_encoder=24,predictor=24,pred_proj=24),
        inherited_scope='old known calibration; never added to new API counts; no training-lineage/freshness certification',
        new_encoder=0,AD=0,physics=0,training=False,Q_fit=False,CPU_neural=False,no_retry=True,
        TF32=False,SDP='MATH',model_frozen=True,training_lineage_verified=False,
        physical_counterfactual_verified=False,independent_confirmation=False,P1_admission=False,scientific_winner=None,
        wall_seconds=WALL,allocated_bytes=3*1024**3,reserved_bytes=4*1024**3,
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=PREP_CAP,
        storage='single-member gzip of original full NPY bytes + full native module input/output tensor leaves')


def gunzip(raw,limit):
    dec=zlib.decompressobj(31);out=dec.decompress(raw,limit+1)
    require(len(out)<=limit and dec.eof and not dec.unused_data and not dec.unconsumed_tail,'single complete bounded gzip, no trailing/member')
    return out


class Arrays:
    def __init__(self,base,inventory):self.base=Path(base);self.inventory=inventory;self.used=set()
    def load(self,rec):
        name=rec['file'];require(Path(name).name==name and name.endswith('.npy.gz'),'flat native NPY gzip')
        path=self.base/name
        require(self.inventory.get(name)==rec['sha256'] and sha(path)==rec['sha256'] and path.stat().st_size<=8*1024**2,'array inventory/file SHA')
        raw=gunzip(path.read_bytes(),8*1024**2)
        require(digest(raw)==rec['npy_sha256'],'original full NPY SHA')
        stream=BytesIO(raw);a=np.load(stream,allow_pickle=False,max_header_size=1024)
        require(stream.tell()==len(raw) and a.dtype in (np.dtype('float32'),np.dtype('uint8')) and
                list(a.shape)==rec['shape'] and a.dtype.str==rec['dtype'] and np.isfinite(a).all(),'native array ABI/no trailing data')
        require(old.array_id(a)==dict(shape=rec['shape'],dtype=rec['dtype'],sha256=rec['raw_sha256']),'raw tensor byte SHA')
        self.used.add(name);return a
    def tree(self,node):
        if isinstance(node,dict):
            if set(node)=={'tensor'}:return self.load(node['tensor'])
            return {k:self.tree(v) for k,v in node.items()}
        if isinstance(node,list):return [self.tree(v) for v in node]
        require(node is None or type(node) in (str,bool,int,float),'native IO tree');return node


def validate_trace(entries,arrays,requests,images=None):
    """Saved actual call graph; both API output and native submodule tensor edges checked."""
    image_count=0 if images is None else len(images)
    require(len(entries)==2*image_count+4*len(requests),'complete native ledger denominator')
    pos=0
    if images is not None:
        for index,row in enumerate(images):
            group=entries[pos:pos+2];pos+=2
            require([(e['resource'],e['phase'],e['sample']) for e in group]==[(k,'image',index) for k in ('encoder','projector')],'original encoder ordering')
            for e in group:require(e['index']==entries.index(e) and e['completed'] is True,'actual complete encoder entry')
            a,b=(arrays.tree(e) for e in group)
            require(set(a['output'])=={'last_hidden_state'} and
                    bits(a['output']['last_hidden_state'][:,0],b['inputs'][0]),'encoder CLS->projector tensor edge')
            require(bits(b['output'].reshape(192),arrays.load(row['latent'])),'original192 projected encoding identity')
    for index,row in enumerate(requests):
        group=entries[pos:pos+4];pos+=4
        require([(e['resource'],e['phase'],e['sample']) for e in group]==[(k,'prediction',index) for k in ORDER],'fixed prediction module order')
        require(all(e['completed'] is True and e['index']==pos-4+j for j,e in enumerate(group)),'actual complete module indices')
        a,p,n,q=(arrays.tree(e) for e in group)
        require(all(e['kwargs']=={} for e in group),'original empty native kwargs')
        z=arrays.load(row['inputs']['latent_history']);tokens=arrays.load(row['inputs']['tokens']);out=arrays.load(row['output'])
        require(bits(a['inputs'][0],tokens[None]) and bits(p['inputs'][0],z[None]) and
                bits(p['inputs'][1],a['output']),'actual API context/action inputs')
        require(bits(n['inputs'][0],p['inputs'][0]) and bits(n['inputs'][1],p['inputs'][1]),'predict->predictor tensor inputs')
        require(bits(q['inputs'][0],n['output'].reshape(3,192)),'predictor->pred_proj full tensor')
        require(bits(q['output'],out) and bits(p['output'],out[None]),'whole full3 output/API/projector equivalence')


def validate_events(events,entries):
    attempts=[e['entry'] for e in events if e.get('kind')=='native_module_attempt']
    inputs=[e['entry'] for e in events if e.get('kind')=='native_inputs_saved']
    returns=[e['entry'] for e in events if e.get('kind')=='native_module_return']
    require(len(attempts)==len(inputs)==len(returns)==len(entries) and
            sorted(returns,key=lambda e:e['index'])==entries,'all native durable attempts/inputs/returns')
    for a,b,c in zip(attempts,inputs,entries,strict=True):
        require(a=={k:c[k] for k in ('index','phase','sample','resource')}|{'completed':False},'actual predispatch attempt')
        require(b=={k:v for k,v in c.items() if k!='output'}|{'completed':False},'actual full saved inputs before return')
    stack=[]
    for e in events:
        if e.get('kind')=='native_module_attempt':stack.append(e['entry']['index'])
        elif e.get('kind')=='native_module_return':require(stack and stack.pop()==e['entry']['index'],'nested API/forward return chronology')
    require(not stack,'balanced native call stack')


def validate_old():
    """Full saved provenance/source/30 encodes/24 requests/NN leaves, no old execution."""
    require(sha(old.__file__)==OLD_SOURCE_SHA,'frozen upstream source')
    r=read(OLD_REPORT,OLD_REPORT_SHA);p,data=old.verify(OLD_PREP,OLD_PREP_SHA)
    require(r['prepared']==str(OLD_PREP) and r['prepare_sha256']==OLD_PREP_SHA and
            r['schema']==old.NAME+'.report' and r['status']=='overlay_factual_fidelity_saved_not_admitted' and
            r['contract']==old.contract() and r['errors']==r['forbidden_AD_attempts']==[] and r['first_error'] is None and
            r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['model_unchanged'] is True and
            r['source_verified'] is True and r['P1_admission'] is False,'complete actual factual source')
    require(r['resources']=={k:dict(attempted=v,completed=v) for k,v in old.contract()['counts'].items()},'all inherited resource counts')
    inv=r['artifact_sha256'];base=OLD_REPORT.parent
    require({x.name for x in base.iterdir()}==set(inv)|{'report.json'},'original exact full output census')
    require(all(sha(base/name)==h for name,h in inv.items()),'all old artifact bytes')
    require(read(base/'binding.json')==dict(prepared=str(OLD_PREP),prepare_sha256=OLD_PREP_SHA) and read(base/'input_receipts.json')==p,'fixed original input/prepare binding')
    arrays=Arrays(base,inv);require(len(r['images'])==30 and len(r['requests'])==24,'original full denominators')
    images,rows=old.census();latents={};raws={}
    for source,d in zip(EPISODES,data,strict=True):
        require(d['episode_ordinal']==source,'original episode alignment');raws[source]=d['raw_actions']
        rawrec=next({'file':n,'sha256':h} for n,h in inv.items() if n==f'episode_{source}.raw20.npy.gz')
        raw=np.load(BytesIO(gunzip((base/rawrec['file']).read_bytes(),1024)),allow_pickle=False)
        require(bits(raw,raws[source]),'saved literal archive raw20')
        arrays.used.add(rawrec['file'])
    for want,row in zip(images,r['images'],strict=True):
        require({k:row[k] for k in want}==dict(want,status='encoded'),'all original image identities')
        ep,c,s=row['episode_ordinal'],row['condition'],row['step']
        payload=data[EPISODES.index(ep)]['frames'][c][old.STEPS.index(s)]
        require(bits(arrays.load(row['pixels']),payload),'original30 pixels against producer archive/native source')
        z=arrays.load(row['latent']);enc.native(z);latents[ep,c,s]=z
    outputs=[];prefix={}
    for want,row in zip(rows,r['requests'],strict=True):
        require({k:row[k] for k in want}==dict(want,status='saved'),'all original request identities')
        ep,c,k=row['episode_ordinal'],row['condition'],row['kind'];zs=[latents[ep,c,s] for s in old.STEPS]
        request=old.prediction_request(ep,c,k,zs,raws[ep],prefix.get((ep,c)))
        require(row['identity']=={a:request[a] for a in ('episode_ordinal','condition','kind')},'request explicit identity')
        for field in ('latent_history','raw_packets','tokens'):require(bits(arrays.load(row['inputs'][field]),request[field]),'original full input reconstruction')
        out=arrays.load(row['output']);require(out.shape==(3,192),'original full3 output')
        if k==old.KINDS[1]:
            first=r['requests'][row['index']-1];require(bits(out,arrays.load(first['output'])),'old H1 and first H2 exact')
            prefix[ep,c]=out[-1]
        label_step=15 if k in old.KINDS[:2] else 20;target=latents[ep,c,label_step]
        outputs.append(dict(episode_ordinal=ep,condition=c,kind=k,actual_successor_step=label_step,
            MSE=old.mse(out[-1],target),persistence_last_input_MSE=old.mse(request['latent_history'][-1],target),
            persistence_origin10_MSE=old.mse(latents[ep,c,10],target)))
    analysis=read(base/'analysis.json');require(analysis['scores']==outputs,'all old24 scalar MSE independently recomputed')
    require(analysis['paired_latent_distances']==[dict(episode_ordinal=ep,step=s,
        nativeoff_nativeon_MSE=old.mse(latents[ep,'nativeoff',s],latents[ep,'nativeon',s]))
        for ep in EPISODES for s in old.STEPS],'all original paired latent distances')
    validate_trace(r['module_ledger'],arrays,r['requests'],r['images'])
    require(arrays.used=={n for n in inv if n.endswith('.npy.gz')},'every original NPY tensor referenced and decoded')
    events=[json.loads(x) for x in gunzip((base/'events.jsonl.gz').read_bytes(),32*1024**2).splitlines()]
    validate_events(events,r['module_ledger'])
    require([e['receipt'] for e in events if e.get('kind')=='image_saved']==r['images'] and
            [e['receipt'] for e in events if e.get('kind')=='prediction_saved']==r['requests'] and
            events[-1]==dict(kind='all24_predictions_durable_before_scoring'),'original scheduler seal/no missing labels')
    independent=read(OLD_INDEPENDENT,OLD_INDEPENDENT_SHA)
    # The MAIN receipt records516 IO arrays; two separately saved literal raw20s
    # make518 NPY files here. Both raw20s were independently decoded above.
    io_arrays=arrays.used-{f'episode_{ep}.raw20.npy.gz' for ep in EPISODES}
    require(independent['report_sha']==OLD_REPORT_SHA and independent['artifact_count']==len(inv) and
            independent['array_count']==len(io_arrays)==516 and len(arrays.used)==518 and independent['module_entries']==len(r['module_ledger']) and
            independent['journal_events']==len(events),'existing independent MAIN receipt full source association')
    bindings=dict(p['bindings'])
    bindings.update({str(base/n):h for n,h in inv.items()})
    bindings.update({str(OLD_REPORT):OLD_REPORT_SHA,str(OLD_PREP):OLD_PREP_SHA,str(OLD_INDEPENDENT):OLD_INDEPENDENT_SHA,str(Path(old.__file__)):OLD_SOURCE_SHA})
    return p,r,bindings


def normalize(raw,metadata):
    require(raw.dtype==np.float32 and raw.shape==(3,5,2) and np.isfinite(raw).all(),'full literal raw3 packets')
    mean=np.asarray(metadata['mean'],np.float32);std=np.asarray(metadata['std'],np.float32)
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0),'original finite mean/std')
    return ((raw-mean)/std).reshape(3,10).astype(np.float32)


def mutate(base,arm,metadata):
    require(set(base)==old.REQUEST_KEYS and base['kind'] in KINDS and arm in ARMS,'teacher H1 inputs only, no target')
    raw=base['raw_packets'].copy();history=base['latent_history'].copy()
    if arm=='zero_outgoing':raw[-1]=np.float32(0.)
    elif arm=='reverse_time_outgoing':raw[-1]=base['raw_packets'][-1,::-1,:].copy()
    elif arm=='swap_xy_outgoing':raw[-1]=base['raw_packets'][-1,:,::-1].copy()
    tokens=normalize(raw,metadata)
    require(bits(raw[:2],base['raw_packets'][:2]) and bits(history,base['latent_history']) and
            bits(tokens[:2],base['tokens'][:2]),'history raw10/latents/normalized history exact')
    if arm=='factual':require(bits(tokens,base['tokens']) and bits(raw,base['raw_packets']),'factual full input bytes exact')
    return dict(base,arm=arm,latent_history=history,raw_packets=raw,tokens=tokens)


def material():
    p,r,bindings=validate_old()
    bases=[dict(base_index=i,old_index=row['index'],episode_ordinal=row['episode_ordinal'],condition=row['condition'],
        kind=row['kind'],inputs=row['inputs'],factual_output=row['output'],successor_step=15 if row['kind']==KINDS[0] else 20)
        for i,row in enumerate(x for x in r['requests'] if x['kind'] in KINDS)]
    require(len(bases)==12 and [(b['episode_ordinal'],b['condition'],b['kind']) for b in bases]==
            [(ep,c,k) for ep in EPISODES for c in CONDITIONS for k in KINDS],'exact twelve base authority')
    require(p['normalization']==enc.native_metadata(),'same original normalizer source/float32 values')
    label_refs=[dict(episode_ordinal=x['episode_ordinal'],condition=x['condition'],step=x['step'],latent=x['latent'])
                for x in r['images'] if x['step'] in (15,20)]
    for path in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bindings[str(path)]=sha(path)
    requests=[];arrays=Arrays(OLD_REPORT.parent,r['artifact_sha256'])
    for b in bases:
        base={k:b[k] for k in ('episode_ordinal','condition','kind')}
        base.update({k:arrays.load(rec) for k,rec in b['inputs'].items()})
        for arm in ARMS:
            req=mutate(base,arm,p['normalization'])
            requests.append(dict(index=len(requests),base_index=b['base_index'],old_index=b['old_index'],
                identity={k:req[k] for k in ('episode_ordinal','condition','kind','arm')},
                inputs={k:old.array_id(req[k]) for k in ('latent_history','raw_packets','tokens')}))
    return dict(schema=NAME+'.prepare',contract=contract(),bindings=bindings,bases=bases,request_receipts=requests,
        label_refs=label_refs,upstream_inventory=r['artifact_sha256'],old_report=str(OLD_REPORT),old_report_sha256=OLD_REPORT_SHA,
        old_prepare=str(OLD_PREP),old_prepare_sha256=OLD_PREP_SHA,old_independent=str(OLD_INDEPENDENT),old_independent_sha256=OLD_INDEPENDENT_SHA,
        normalization=p['normalization'],policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        inherited_resources=r['resources'],P1_admission=False,training_lineage_verified=False,selection=False)


def write_prepare(path):
    p=material();raw=canonical(p);require(len(raw)<=PREP_CAP,'explicit 2MiB prepare cap, no rounding/dropping bindings')
    path=Path(path).absolute();require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and
        not any(x.is_symlink() for x in (path,*path.parents)) and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'new own prepare/floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(status='prepared_no_neural_execution',prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    p=read(path,expected,PREP_CAP);require(p==material(),'full prepared source/old-artifact/request recomputation');return p


class Store(old.sealed.Store):
    def __init__(self,output):
        path=Path(output).absolute()
        require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not any(x.is_symlink() for x in (path,*path.parents)),'own nonsymlink output')
        require(shutil.disk_usage(path.parent).free>=FLOOR+CAP,'new32MiB +200MiB floor')
        self.started=time.monotonic();self.tail=None;self.event_index=0
        super().__init__(path)
    def _write(self,name,raw,append=False,final=False):
        require(time.monotonic()-self.started<WALL+5 if final else time.monotonic()-self.started<WALL,'wall600s (+5s final failure preservation only)')
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat nonsymlink artifact')
        used=sum(x.stat().st_size for x in self.output.iterdir())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'32MiB incl2MiB final reserve/200MiB floor')
        with (self.output/name).open('ab' if append else 'xb') as f:require(f.write(raw)==len(raw),'full durable write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_DIRECTORY|os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def event(self,value):
        e=dict(index=self.event_index,previous_sha256=self.tail,value=deepcopy(value));e['event_sha256']=digest(canonical(e))
        super().event(e);self.tail=e['event_sha256'];self.event_index+=1


class Trace(old.NativeTrace):
    def begin(self,kind,args,kwargs):
        require(self.active is not None and self.active[0]=='prediction' and
                type(self.active[1]) is int and 0<=self.active[1]<48 and kind in ORDER,'no encoder/projector/unscoped call')
        require(sum(e['resource']==kind for e in self.entries)<48,'actual module predispatch48 cap')
        return super().begin(kind,args,kwargs)
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in EXPECTED}


def census():
    return [dict(index=i,base_index=i//4,episode_ordinal=ep,condition=c,kind=k,arm=arm,status='not_started')
        for i,(ep,c,k,arm) in enumerate((ep,c,k,arm) for ep in EPISODES for c in CONDITIONS for k in KINDS for arm in ARMS)]


def requests(p):
    arrays=Arrays(Path(p['old_report']).parent,p['upstream_inventory'])
    for b in p['bases']:
        base={k:b[k] for k in ('episode_ordinal','condition','kind')}
        base.update({k:arrays.load(rec) for k,rec in b['inputs'].items()})
        for arm in ARMS:yield mutate(base,arm,p['normalization'])


def labels(p):
    arrays=Arrays(Path(p['old_report']).parent,p['upstream_inventory'])
    return {(r['episode_ordinal'],r['condition'],r['step']):arrays.load(r['latent']) for r in p['label_refs']}


def scores(outputs,label_values):
    require(len(outputs)==48 and len(label_values)==12,'complete denominators before scores')
    rows=[];groups=[]
    for index,out in enumerate(outputs):
        meta=out['identity'];require(meta=={k:census()[index][k] for k in ('episode_ordinal','condition','kind','arm')},'fixed scoring identities')
        ep,c,k=meta['episode_ordinal'],meta['condition'],meta['kind'];step=15 if k==KINDS[0] else 20
        target=label_values[ep,c,step];enc.native(target)
        value=out['value'];require(value.dtype==np.float32 and value.shape==(3,192) and np.isfinite(value).all(),'full3 predictions before scoring')
        rows.append(dict(**meta,actual_successor_step=step,MSE=old.mse(value[-1],target),
            target_role='same-condition factual observed successor, NOT physical counterfactual',
            persistence_MSE=old.mse(out['last_input'],target)))
    for i in range(12):
        base=rows[4*i:4*i+4];f=base[0];fout=outputs[4*i]['value']
        for j in range(1,4):
            r=base[j];groups.append(dict(episode_ordinal=f['episode_ordinal'],condition=f['condition'],kind=f['kind'],arm=r['arm'],
                nonfactual_minus_factual_MSE=r['MSE']-f['MSE'],factual_MSE=f['MSE'],nonfactual_MSE=r['MSE'],
                full3_output_change_MSE=old.mse(outputs[4*i+j]['value'],fout),
                successor_output_change_MSE=old.mse(outputs[4*i+j]['value'][-1],fout[-1])))
    condition_contrasts=[]
    for ep in EPISODES:
        for kind in KINDS:
            for arm in ARMS[1:]:
                selected={g['condition']:g for g in groups if g['episode_ordinal']==ep and g['kind']==kind and g['arm']==arm}
                for c in CONDITIONS[1:]:
                    a,b=selected[c],selected['archive']
                    condition_contrasts.append(dict(episode_ordinal=ep,kind=kind,arm=arm,condition=c,reference='archive',
                        native_minus_archive_discrimination_gap=a['nonfactual_minus_factual_MSE']-b['nonfactual_minus_factual_MSE']))
    return dict(rows=rows,contrasts=groups,render_relative_contrasts=condition_contrasts,
        label_semantics='successor discrimination only; no physical counterfactual or control benefit',
        P1_admission=False,training_lineage_verified=False,scientific_winner=None)


def execute(p,predict,store,rows,label_loader):
    require(len(rows)==48,'all48 rows retained')
    outputs=[];old_arrays=Arrays(Path(p['old_report']).parent,p['upstream_inventory'])
    for index,request in enumerate(requests(p)):
        row=rows[index];require(set(request)==REQUEST_KEYS,'no target/label in predictor request')
        row['status']='attempted';row['identity']={k:request[k] for k in ('episode_ordinal','condition','kind','arm')}
        row['inputs']={k:old.save_array(store,f'request_{index:02d}.{k}',request[k]) for k in ('latent_history','raw_packets','tokens')}
        store.event(dict(kind='prediction_attempt',receipt=deepcopy(row)))
        value=predict(deepcopy(request),index)
        require(value.dtype==np.float32 and value.shape==(3,192) and np.isfinite(value).all(),'full finite native3x192')
        row['output']=old.save_array(store,f'request_{index:02d}.prediction',value)
        if request['arm']=='factual':
            reference=old_arrays.load(p['bases'][index//4]['factual_output'])
            require(bits(reference,value),'all12 factual full3 must exactly reproduce original outputs')
            row['old_full3_bitwise_equal']=True
        else:row['old_full3_bitwise_equal']=None
        row['status']='saved'
        store.event(dict(kind='prediction_saved',receipt=deepcopy(row)))
        outputs.append(dict(identity=deepcopy(row['identity']),value=value.copy(),last_input=request['latent_history'][-1].copy()))
    store.event(dict(kind='all48_predictions_sealed_before_labels',outputs=[r['output'] for r in rows]))
    label_values=label_loader(p)  # first label use for scoring, after all48 fsync/seal
    store.event(dict(kind='labels_loaded_for_scoring',keys=[list(x) for x in label_values]))
    result=scores(outputs,label_values);store.put('analysis.json',result);return result


def run(prepared,expected,output,confirm=False):
    require(confirm,'MAIN explicit HOST CUDA authorization required')
    p=verify(prepared,expected);store=Store(output);rows=census();errors=[];forbidden=[]
    model=trace=before=after=host=None;source_ok=False;limits=fidelity.Limits()
    require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'no preexisting wall timer')
    previous_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*args):raise TimeoutError('600s wall, no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(WALL)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('input_receipts.json',p)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        limits.free=lambda:shutil.disk_usage(store.output).free
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model hash')
        trace=Trace(model,store,lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        def predict(request,index):
            require(set(request)==REQUEST_KEYS,'label-free backend whitelist')
            limits.check();trace.active=('prediction',index,request)
            with torch.inference_mode():
                z=torch.from_numpy(request['latent_history'].copy()).to(device)[None]
                tokens=torch.from_numpy(request['tokens'].copy()).to(device)[None]
                value=model.predict(z,model.action_encoder(tokens))
                require(value.device.type=='cuda' and value.dtype==torch.float32 and tuple(value.shape)==(1,3,192),'actual CUDA full3 output')
                result=value[0].detach().cpu().numpy().copy()
            trace.finish('prediction',index);limits.check();return result
        def ad_event(e):forbidden.append(deepcopy(e));store.event(e)
        with fidelity.old.no_ad(torch,ad_event,errors),sdpa_kernel(SDPBackend.MATH),trace:
            execute(p,predict,store,rows,labels)
        require(trace.counts()=={k:dict(attempted=v,completed=v) for k,v in EXPECTED.items()} and not forbidden,'all actual new48 modules, zero encoder/AD')
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[-1]))
        except BaseException as secondary:errors.append(enc.error('journal_error',secondary))
    finally:
        signal.alarm(0)
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        try:
            if model is not None:enc.frozen(model);after=enc.tensor_hash(model)
            require(model is None or after==before==p['expected_model_hash'],'frozen model after/expected')
        except BaseException as exc:errors.append(enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(enc.error('final_source',exc))
        try:store.close()
        except BaseException as exc:errors.append(enc.error('final_journal',exc))
        complete=not errors and source_ok and before==after==p['expected_model_hash'] and all(r['status']=='saved' for r in rows)
        elapsed=time.monotonic()-store.started
        if elapsed>=WALL:complete=False;errors.append(dict(phase='wall',type='TimeoutError',message='600s exceeded'))
        r=dict(schema=NAME+'.report',status='all48_action_discrimination_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=contract(),requests=rows,
            resources=trace.counts() if trace else {k:dict(attempted=0,completed=0) for k in EXPECTED},
            module_ledger=deepcopy(trace.entries) if trace else [],forbidden_AD_attempts=forbidden,
            errors=errors,first_error=errors[0] if errors else None,source_verified=source_ok,
            artifact_sha256={f.name:sha(f) for f in store.output.iterdir()},journal_chain_tail=store.tail,journal_events=store.event_index,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,limits=limits.receipt(),wall_seconds=elapsed,AD=0,physics=0,training=False,P1_admission=False,
            training_lineage_verified=False,scientific_winner=None)
        store.put('report.json',r,final=True);signal.signal(signal.SIGALRM,previous_alarm)
    require(complete,'failure preserved, no retry: '+str(errors));return r


def consume(prepared,expected,report,report_sha):
    p=verify(prepared,expected);path=Path(report);r=read(path,report_sha)
    require(path.name=='report.json' and r['schema']==NAME+'.report' and r['contract']==contract() and
        r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==expected and
        r['status']=='all48_action_discrimination_saved_not_admitted' and r['errors']==r['forbidden_AD_attempts']==[] and r['first_error'] is None,'complete own48 authority')
    require(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['model_unchanged'] is True and r['source_verified'] is True,'model and source frozen')
    require(r['AD']==r['physics']==0 and r['training'] is r['P1_admission'] is r['training_lineage_verified'] is False and r['scientific_winner'] is None,'no admission/control claims')
    require(r['resources']=={k:dict(attempted=v,completed=v) for k,v in EXPECTED.items()},'new actual API/module/sample counts')
    require(0<=r['wall_seconds']<WALL and r['limits']['elapsed_seconds']<WALL and
        r['limits']['peak_allocated_bytes']<=3*1024**3 and r['limits']['peak_reserved_bytes']<=4*1024**3 and
        r['limits']['minimum_observed_free_bytes']>=FLOOR,'actual runtime resource caps')
    inv=r['artifact_sha256'];require(set(inv)|{'report.json'}=={x.name for x in path.parent.iterdir()},'exact output census')
    require(sum(x.stat().st_size for x in path.parent.iterdir())<=CAP and all(sha(path.parent/n)==h for n,h in inv.items()),'all output SHA/32MiB cap')
    require(read(path.parent/'binding.json')==dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected) and read(path.parent/'input_receipts.json')==p,'saved own actual execution inputs')
    arrays=Arrays(path.parent,inv);expected_rows=census();outputs=[];old_arrays=Arrays(OLD_REPORT.parent,p['upstream_inventory'])
    require(len(r['requests'])==48,'all48 denominator')
    for row,want,request in zip(r['requests'],expected_rows,requests(p),strict=True):
        require({k:row[k] for k in want}==dict(want,status='saved') and row['identity']=={k:request[k] for k in ('episode_ordinal','condition','kind','arm')},'all48 unchanged row identities')
        for field in ('latent_history','raw_packets','tokens'):require(bits(arrays.load(row['inputs'][field]),request[field]),'own literal mutated packets/normalizer/history')
        out=arrays.load(row['output']);require(out.shape==(3,192),'full saved prediction')
        if request['arm']=='factual':require(row['old_full3_bitwise_equal'] is True and bits(out,old_arrays.load(p['bases'][row['base_index']]['factual_output'])),'all12 factual bitwise')
        else:require(row['old_full3_bitwise_equal'] is None,'no counterfactual old-output equivalence claim')
        outputs.append(dict(identity=row['identity'],value=out,last_input=request['latent_history'][-1]))
    validate_trace(r['module_ledger'],arrays,r['requests'])
    require(arrays.used=={n for n in inv if n.endswith('.npy.gz')},'all saved native IO accounted')
    raw=gunzip((path.parent/'events.jsonl.gz').read_bytes(),32*1024**2)
    require(raw.endswith(b'\n'),'complete event lines');events=[];tail=None
    for i,line in enumerate(raw.splitlines()):
        e=json.loads(line);require(e['index']==i and e['previous_sha256']==tail and e['event_sha256']==digest(canonical({k:v for k,v in e.items() if k!='event_sha256'})),'own durable event chain')
        tail=e['event_sha256'];events.append(e['value'])
    require(tail==r['journal_chain_tail'] and len(events)==r['journal_events'],'complete own journal tail')
    validate_events(events,r['module_ledger'])
    saved=[e['receipt'] for e in events if e.get('kind')=='prediction_saved'];require(saved==r['requests'],'all48 durable prediction receipts')
    attempts=[e['receipt'] for e in events if e.get('kind')=='prediction_attempt']
    require(len(attempts)==48,'exact48 scheduler attempts')
    for before,after in zip(attempts,r['requests'],strict=True):
        require(before=={k:v for k,v in after.items() if k not in ('output','old_full3_bitwise_equal')}|{'status':'attempted'},'exact predispatch request receipts')
    allowed={'prediction_attempt','prediction_saved','native_module_attempt','native_inputs_saved','native_module_return',
             'all48_predictions_sealed_before_labels','labels_loaded_for_scoring'}
    require(all(e.get('kind') in allowed for e in events),'no hidden or forbidden event types')
    seal=[i for i,e in enumerate(events) if e.get('kind')=='all48_predictions_sealed_before_labels']
    label=[i for i,e in enumerate(events) if e.get('kind')=='labels_loaded_for_scoring']
    require(len(seal)==len(label)==1 and seal[0]<label[0] and
            events[seal[0]]['outputs']==[r['output'] for r in r['requests']] and
            max(i for i,e in enumerate(events) if e.get('kind') in ('native_module_return','prediction_saved'))<seal[0], 'all predictions before labels/no postscore model')
    label_values=labels(p);require(events[label[0]]['keys']==[list(k) for k in label_values],'same condition successor labels')
    actual=scores(outputs,label_values);require(read(path.parent/'analysis.json')==actual,'all48 MSE/36 gap/24 render contrasts independent recomputation')
    require(sha(path)==report_sha,'final report unchanged')
    return dict(schema=NAME+'.consume',status='all48_action_discrimination_saved_verified_not_admitted',report_sha256=report_sha,
        prepare_sha256=expected,resources=r['resources'],analysis=actual,factual_full3_exact=12,
        inherited_resources=p['inherited_resources'],artifact_count=len(inv),module_entries=len(r['module_ledger']),journal_events=len(events),
        P1_admission=False,training_lineage_verified=False,physical_counterfactual_verified=False,scientific_winner=None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for name in ('prepared','prepare-sha','output','report','report-sha'):parser.add_argument('--'+name)
    parser.add_argument('--confirm-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':require(a.prepared,'new own prepare path');result=write_prepare(a.prepared)
    else:
        require(a.prepared and a.prepare_sha,'explicit prepare/SHA')
        if a.command=='verify':verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_neural_execution')
        elif a.command=='run':require(a.output,'new output');result=run(a.prepared,a.prepare_sha,a.output,a.confirm_host_cuda)
        else:
            require(a.report and a.report_sha and a.output,'actual report/SHA and new receipt output')
            result=consume(a.prepared,a.prepare_sha,a.report,a.report_sha)
            target=Path(a.output).absolute();raw=canonical(result)
            require(target.parent==ROOT/'results' and target.name.startswith(NAME+'_') and not target.exists() and
                not any(x.is_symlink() for x in (target,*target.parents)) and len(raw)<=PREP_CAP and
                shutil.disk_usage(target.parent).free>=FLOOR+len(raw),'exclusive bounded own strict receipt/floor')
            with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'],P1_admission=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
