"""Saved own-action H1/H2 localization; MAIN-only CUDA, no control or new labels."""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2 as core

NAME='p1_moving_teacher_horizon_localization_cuda_20261027_v1'
ROOT=core.ROOT
CORE_SHA='492ff318c83a312be5f5f6efd85e49ad06eb56457d5c0e92c3b92ee617a38a1b'
TEACHER=ROOT/'results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01/report.json'
TEACHER_SHA='7323347a9e74c8f1751550642503426d07ddce55549ddaef4331f3c362567128'
STAGES=('prefix_H1','open_H2','teacher_H1')
RESOURCES=core.RESOURCES
CAP,FLOOR,FINAL,PREP_CAP,SEMANTIC_CAP=core.CAP,core.FLOOR,core.FINAL,core.PREP_CAP,core.SEMANTIC_CAP
require,canonical,digest,sha=core.require,core.canonical,core.digest,core.sha
Store=core.Store


def contract():
    c=core.contract()
    c.update(horizon=2,stages=list(STAGES),resources={k:36 for k in RESOURCES},
        packet_steps=5,raw_plan_steps=10,forwards_per_episode=3,
        label_access='score labels only after all36 predictions sealed; actual prefix is teacher conditioning input',
        internal_request_schema=core.NAME+'.forward_input',
        science='same fixed12 executed paths; H1 already exists, H2 sensitivity not causal attribution or P1')
    return c


def reseal(r):
    r=deepcopy(r);r['input_digest']=digest({k:v for k,v in r.items() if k!='input_digest'})
    core.validate_request(r);return r


def root_request(suffix,task,warm,prefix):
    r=deepcopy(suffix)
    r.update(latents=deepcopy(task['root_context']['latent_history']),
        history=core.normalized(warm,r['mean'],r['std']).tolist(),raw5=np.asarray(prefix,np.float32).tolist())
    require(r['history']==task['root_context']['action_history'],'actual warmup full2 equals source action tokens')
    return reseal(r)


def open_request(root,suffix,predicted_prefix):
    require(root['identity']==suffix['identity'],'own episode no cross-context join')
    z=core.literal(root['latents'],(3,192));predicted_prefix=core.literal(predicted_prefix,(192,))
    r=deepcopy(suffix)
    # Never take the actual prefix latent into this branch; only last predicted successor shifts in.
    r['latents']=np.concatenate((z[1:],predicted_prefix[None])).tolist()
    outgoing=core.validate_request(root)
    require(np.array_equal(outgoing[1:],r['history']),'same executed prefix/raw chronology in both branches')
    return reseal(r)


def source_census(events,rows,bind,base):
    """Pure saved PNG/guard chronology. Raw step is NOT model step."""
    from PIL import Image
    total=0
    for row in rows:
        ep=row['episode_id'];scoped=[e for e in events if e.get('episode_id')==ep]
        obs=[e['receipt'] for e in scoped if e['kind']=='fresh_observation']
        expected=[(phase,s) for phase in ('warmup','plan') for s in range(11)]
        require([(o['phase'],o['step']) for o in obs]==expected,'all22 actual observations per episode')
        guards=[(e['phase'],e['step']) for e in scoped if e['kind']=='guard_passed']
        require(guards==expected,'all22 original freshRGB guards')
        for o in obs:
            f=bind(core.saved.local(base,o['path']),o['file_sha256'])
            with Image.open(f) as im:a=np.asarray(im).copy()
            require(a.dtype==np.uint8 and a.shape==(224,224,3) and
                core.saved.sha_bytes(a.tobytes())==o['pixel_sha256'],'actual saved PNG pixels, no substitute')
            core.saved.array(o['state'],(7,));total+=1
        for kind,field in (('warmup_return','raw'),('control_return','command')):
            core.raw_rows(scoped,kind,range(1,11),field)
        for cp in row['checkpoints']:
            require(cp in obs and cp['phase']=='plan' and cp['step'] in (0,5,10),'checkpoint owns fresh observation')
        require([cp['step'] for cp in row['checkpoints']]==[0,5,10],'actual packet endpoints0/5/10')
    require(total==264,'fixed all264 original saved PNGs')
    return dict(episodes=12,PNG=264,raw_steps_per_episode=20,packet_steps=5,plan_packet_endpoints=[0,5,10],
        maximum_model_horizon=2,intermediate_raw_frames_are_not_model_steps=True)


def build(**arguments):
    require(sha(core.__file__)==CORE_SHA,'frozen teacher core authority')
    p=core.build(**arguments)  # Strict original own context/receipt/source/model joins, no terminal decode.
    bindings=p['bindings']
    def bind(path,h=None):
        f=Path(path).resolve();value=sha(f) if h is None else h
        require(sha(f)==value and (str(f) not in bindings or bindings[str(f)]==value),'nonconflicting own source binding')
        bindings[str(f)]=value;return f
    bind(TEACHER,TEACHER_SHA)  # Hash reference only: no teacher outcomes become inputs.
    a=p['arguments'];base=Path(a['report_path']).parent
    live=core.saved.parse(core.saved.read_bytes(a['report_path'],a['report_sha']))
    old=core.saved.parse(core.saved.read_bytes(a['live_prepared'],a['live_prepare_sha']))
    events=core.saved.journal((base/'journal.jsonl').read_bytes(),live['journal_chain_tail'])
    census=source_census(events,live['rows'],bind,base)
    tasks={}
    for layer in old['layers']:
        tasks[layer['source_slot']]=core.moving.load_task(layer['task_export'],layer['task_sha256'])['gn_task']
    roots=[]
    for r,row in zip(p['inputs'],live['rows'],strict=True):
        scoped=[e for e in events if e.get('episode_id')==row['episode_id']]
        warm=core.raw_rows(scoped,'warmup_return',range(1,11),'raw')
        raw=core.raw_rows(scoped,'control_return',range(1,11),'command')
        require(core.saved.key(r['identity'])==core.saved.key(row),'own actual root identity')
        roots.append(root_request(r,tasks[row['source_slot']],warm,raw[:5]))
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bind(f)
    for f,h in bindings.items():require(sha(f)==h,'final all source seals')
    p.update(schema=NAME+'.prepare',root_inputs=roots,contract=contract(),source_census=census,
        reference_teacher_report=dict(path=str(TEACHER),sha256=TEACHER_SHA))
    return p


def encode_prepare(p):
    semantic=canonical(p);require(len(semantic)<=SEMANTIC_CAP,'semantic1MiB')
    e=dict(schema=NAME+'.prepare_envelope',codec='single-gzip+base64',semantic_bytes=len(semantic),
        semantic_sha256=core.saved.sha_bytes(semantic),payload=base64.b64encode(gzip.compress(semantic,mtime=0)).decode())
    raw=json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(len(raw)<=PREP_CAP,'disk prepare256KiB');return raw


def decode_prepare(raw):
    require(type(raw) is bytes and len(raw)<=PREP_CAP,'disk prepare256KiB')
    e=core.saved.parse(raw);core.fields(e,'schema codec semantic_bytes semantic_sha256 payload')
    require(e['schema']==NAME+'.prepare_envelope' and e['codec']=='single-gzip+base64' and
        type(e['semantic_bytes']) is int and 0<e['semantic_bytes']<=SEMANTIC_CAP,'own bounded envelope')
    packed=base64.b64decode(e['payload'],validate=True)
    require(base64.b64encode(packed).decode()==e['payload'],'canonical base64')
    d=zlib.decompressobj(31);semantic=d.decompress(packed,SEMANTIC_CAP+1)
    require(len(semantic)<=SEMANTIC_CAP and d.eof and not d.unused_data and not d.unconsumed_tail and
        len(semantic)==e['semantic_bytes'] and core.saved.sha_bytes(semantic)==e['semantic_sha256'],'single bounded gzip/semanticSHA')
    p=core.saved.parse(semantic)
    require(canonical(p)==semantic and p['schema']==NAME+'.prepare','original semantic canonical');return p


def prepare(output):
    raw=encode_prepare(build());path=Path(output)
    require(not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive prepare/floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path.resolve()),prepare_sha256=sha(path),bytes=len(raw))


def verify(prepared,expected):
    p=decode_prepare(core.saved.read_bytes(prepared,expected,PREP_CAP))
    require(p==build(**p['arguments']),'complete saved reconstruction');return p


class Ledger(core.Ledger):
    def begin(self,kind):
        self.check();n=len(self.entries)
        require(n<144 and self.current==n//4 and kind==RESOURCES[n%4],'fixed36 API/module order no retry')
        require(not self.entries or self.entries[-1]['completed'] or kind=='predictor','actual prior operation return')
        e=dict(index=n,episode_index=self.current,resource=kind,completed=False);self.entries.append(e)
        self.store.event(dict(kind='operation_attempt',entry=e));return e


def initial_rows(p):
    return [dict(identity=deepcopy(r['identity']),status='not_started',
        predictions={s:None for s in STAGES},labels=None,metrics=None) for r in p['inputs']]


def score(prefix_pred,open_pred,teacher_pred,prefix,label,goal,root):
    f=lambda a,b:float(np.mean((a.astype(float)-b.astype(float))**2))
    for a in (prefix_pred,open_pred,teacher_pred,prefix,label,root):core.saved.array(a,(192,),np.dtype('float32'))
    e=teacher_pred.astype(float)-label.astype(float);d=open_pred.astype(float)-teacher_pred.astype(float)
    return dict(prefix_H1=core.metrics(prefix_pred,prefix,goal,root),
        open_H2=core.metrics(open_pred,label,goal,prefix),teacher_H1=core.metrics(teacher_pred,label,goal,prefix),
        propagation_gap_mse=f(open_pred,teacher_pred),
        open_minus_teacher_factual_mse=f(open_pred,label)-f(teacher_pred,label),
        propagation_cross_term=float(2*np.mean(e*d)))


def execute(p,store,ledger,forward,rows,check=lambda:None,label_loader=core.receipt_array):
    require(len(p['inputs'])==len(p['root_inputs'])==len(p['evaluators'])==len(rows)==12,'fixed12 not subset')
    store.put('initial_rows.json',rows);store.event(dict(kind='run_started',contract=p['contract']));all_inputs=[]
    for i,(root,suffix,ev,row) in enumerate(zip(p['root_inputs'],p['inputs'],p['evaluators'],rows,strict=True)):
        require(root['identity']==suffix['identity']==ev['identity']==row['identity'] and root['identity']['index']==i,'full own identity')
        row['status']='attempted';predictions={}
        for stage in STAGES:
            check();n=3*i+STAGES.index(stage)
            r=root if stage=='prefix_H1' else (suffix if stage=='teacher_H1' else open_request(root,suffix,predictions['prefix_H1']))
            core.validate_request(r);ledger.current=n
            receipt=dict(file=f'input_{n:02d}.json',sha256=store.put(f'input_{n:02d}.json',r),stage=stage,episode_index=i)
            all_inputs.append(receipt);store.event(dict(kind='prediction_attempt',forward_index=n,stage=stage,identity=row['identity'],input_digest=r['input_digest']))
            z=forward(deepcopy(r));core.saved.array(z,(192,),np.dtype('float32'))
            require(len(ledger.entries)==4*(n+1) and all(e['completed'] for e in ledger.entries),'one actual H1 forward per stage')
            predictions[stage]=z.copy();row['predictions'][stage]=store.array(f'prediction_{n:02d}.npy',z)
            store.event(dict(kind='prediction_saved',forward_index=n,stage=stage,identity=row['identity'],receipt=row['predictions'][stage]))
        row['status']='predicted'
    ledger.current=None
    require(not ledger.blocked and ledger.counts()=={k:dict(attempted=36,completed=36) for k in RESOURCES},'actual36 before labels')
    seal=dict(schema=NAME+'.predictions_seal',rows=deepcopy(rows),forward_inputs=all_inputs,
        resources=ledger.counts(),ledger=deepcopy(ledger.entries))
    seal_sha=store.put('predictions_seal.json',seal);store.event(dict(kind='all_predictions_sealed',sha256=seal_sha))
    # Prefix is already legitimate conditioning input in teacher branch; scoring label loads begin only here.
    for i,(ev,row) in enumerate(zip(p['evaluators'],rows,strict=True)):
        check();require(ev['identity']==row['identity'],'own evaluator identity')
        prefix=label_loader(ev['prefix']);label=label_loader(ev['label'])
        row['labels']=dict(prefix=store.array(f'label_prefix_{i:02d}.npy',prefix),terminal=store.array(f'label_terminal_{i:02d}.npy',label))
        # Forward outputs are immutable original saved arrays, not a second model call.
        z=[load_array(store.output,row['predictions'][s]) for s in STAGES]
        row['metrics']=score(*z,prefix,label,ev['goal'],core.literal(p['root_inputs'][i]['latents'][-1],(192,)))
        row['status']='scored';store.event(dict(kind='episode_scored',receipt=deepcopy(row)))
    return seal_sha


def load_array(root,receipt):
    raw=core.saved.read_bytes(core.saved.local(root,receipt['file']),receipt['sha256'],1024**2)
    f=BytesIO(raw);a=np.load(f,allow_pickle=False)
    require(f.tell()==len(raw) and str(a.dtype)==receipt['dtype'] and list(a.shape)==receipt['shape'] and
        a.dtype==np.float32 and np.isfinite(a).all(),'exact saved native NPY');return a


def run(prepared,expected,output):
    started=time.monotonic();p=verify(prepared,expected);store=Store(output);rows=initial_rows(p);errors=[]
    model=None;before=after=host=None;source_ok=False;seal=None;ledger=Ledger(store)
    def check():
        require(time.monotonic()-started<300,'whole300s including verify/load/final')
        if model is not None:require(torch.cuda.memory_allocated(device)<=3*1024**3 and torch.cuda.memory_reserved(device)<=4*1024**3,'CUDA3/4GiB')
        require(sum(f.stat().st_size for f in store.output.iterdir() if f.is_file())<CAP-FINAL and
            shutil.disk_usage(store.output).free>=FLOOR,'whole8MiB/floor')
    ledger.check=check
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));check()
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');check()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        core.moving.low.frozen(model);before=core.moving.low.tensor_hash(model);require(before==p['expected_model_hash'],'frozen original model');check()
        with core.no_ad(torch,ledger,errors),torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH),core.Native(model,torch,device,ledger,store) as forward:
            seal=execute(p,store,ledger,forward,rows,check)
    except BaseException as e:
        errors.append(core.error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(core.error('error_journal',secondary))
    finally:
        ledger.current=None
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:core.moving.low.frozen(model);after=core.moving.low.tensor_hash(model)
            except BaseException as e:errors.append(core.error('final_model',e))
        try:verify(prepared,expected);source_ok=True;check()
        except BaseException as e:errors.append(core.error('final_source_or_limits',e))
        complete=not errors and not ledger.blocked and source_ok and before is not None and before==after and seal is not None and all(r['status']=='scored' for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:files=[];errors.append(core.error('final_listing',e));complete=False
        for f in files:
            try:hashes[f.name]=sha(f)
            except BaseException as e:errors.append(core.error('final_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='horizon_localization_recorded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,
            resources=ledger.counts(),ledger=ledger.entries,blocked=ledger.blocked,predictions_seal_sha256=seal,artifact_sha256=hashes,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,source_verified=source_ok,
            contract=contract(),fixed_denominator=12,prediction_denominator=36,independent_tasks=2,
            training=False,selection=False,P1_admission=False,P4_admission=False,new_AD=0,new_encoder=0,new_physics=0,host=host,wall_seconds=time.monotonic()-started)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(core.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial evidence retained no retry: '+str(errors));return report


def consume(output,report_sha):
    root=Path(output).resolve();r=core.saved.parse(core.saved.read_bytes(root/'report.json',report_sha));p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='horizon_localization_recorded_not_admitted' and not r['errors'] and not r['blocked'] and
        r['source_verified'] is True and r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['contract']==contract() and r['fixed_denominator']==12 and r['prediction_denominator']==36 and len(r['rows'])==12 and
        r['resources']=={k:dict(attempted=36,completed=36) for k in RESOURCES},'complete actual localization')
    require({f.name for f in root.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'exact full artifact set')
    for name,h in r['artifact_sha256'].items():require(sha(core.saved.local(root,name))==h,'all original artifact SHA')
    seal=core.saved.parse(core.saved.read_bytes(root/'predictions_seal.json',r['predictions_seal_sha256']))
    require(seal['schema']==NAME+'.predictions_seal' and seal['resources']==r['resources'] and seal['ledger']==r['ledger'] and
        len(seal['rows'])==12 and len(seal['forward_inputs'])==36,'whole36 prediction seal')
    raw=(root/'events.jsonl').read_bytes();events=core.saved.journal(raw,core.saved.parse(raw.splitlines()[-1])['event_sha256'])
    seals=[e for e in events if e['kind']=='all_predictions_sealed'];scored=[e for e in events if e['kind']=='episode_scored']
    require(len(seals)==1 and seals[0]['sha256']==r['predictions_seal_sha256'] and len(scored)==12 and all(e['index']>seals[0]['index'] for e in scored),'all36 seal before any scoring')
    saved_predictions=[e for e in events if e['kind']=='prediction_saved']
    attempts=[e['entry'] for e in events if e['kind']=='operation_attempt']
    returns=[e['entry'] for e in events if e['kind']=='operation_return']
    require(len(saved_predictions)==36 and [e['forward_index'] for e in saved_predictions]==list(range(36)) and
        all(e['index']<seals[0]['index'] for e in saved_predictions),'all36 actually saved before global seal')
    require(len(attempts)==len(returns)==144 and
        [(e['index'],e['episode_index'],e['resource']) for e in attempts]==
        [(e['index'],e['episode_index'],e['resource']) for e in r['ledger']] and
        sorted(returns,key=lambda e:e['index'])==r['ledger'],'actual durable module attempts/returns equal ledger')
    requests=[]
    for i,(row,ev) in enumerate(zip(r['rows'],p['evaluators'],strict=True)):
        require(row['identity']==ev['identity']==p['inputs'][i]['identity']==p['root_inputs'][i]['identity'] and row['status']=='scored' and
            scored[i]['receipt']==row and seal['rows'][i]['status']=='predicted' and seal['rows'][i]['predictions']==row['predictions'],'own fixed row/source/score')
        z=[load_array(root,row['predictions'][s]) for s in STAGES]
        prefix=core.receipt_array(ev['prefix']);label=core.receipt_array(ev['label'])
        require(np.array_equal(prefix,load_array(root,row['labels']['prefix'])) and np.array_equal(label,load_array(root,row['labels']['terminal'])) and
            row['metrics']==score(*z,prefix,label,ev['goal'],core.literal(p['root_inputs'][i]['latents'][-1],(192,))),'own actual labels/exact metrics')
        expect=[p['root_inputs'][i],open_request(p['root_inputs'][i],p['inputs'][i],z[0]),p['inputs'][i]]
        for j,req in enumerate(expect):
            n=3*i+j;entry=seal['forward_inputs'][n]
            require(entry['stage']==STAGES[j] and entry['episode_index']==i and entry['file']==f'input_{n:02d}.json','actual fixed36 request identity')
            event=saved_predictions[n]
            require(event['stage']==STAGES[j] and event['identity']==row['identity'] and event['receipt']==row['predictions'][STAGES[j]],'own durable saved prediction event')
            saved_req=core.saved.parse(core.saved.read_bytes(core.saved.local(root,entry['file']),entry['sha256']))
            require(saved_req==req,'no actual prefix leak into open branch; same action/context chronology')
            require(np.array_equal(load_array(root,r['ledger'][4*n+3]['output'])[-1],z[j]),'own final native projected token')
            requests.append(req)
    check_trace(r['ledger'],requests,lambda e:load_array(root,e))
    return dict(schema=NAME+'.consume',report_sha256=report_sha,rows=deepcopy(r['rows']),resources=r['resources'],
        fixed_denominator=12,prediction_denominator=36,independent_tasks=2,P1_admission=False,P4_admission=False)


def check_trace(entries,requests,load):
    require(len(entries)==144 and len(requests)==36,'full36 native trace')
    for i,request in enumerate(requests):
        es=entries[4*i:4*i+4]
        require(all(e['index']==4*i+j and e['episode_index']==i and e['resource']==RESOURCES[j] and e['completed'] is True for j,e in enumerate(es)),'actual ordered trace36')
        ae,api,pred,proj=es
        require(len(ae['inputs'])==len(proj['inputs'])==1 and len(pred['inputs'])==2,'module arity')
        a,emb=load(ae['inputs'][0]),load(ae['output']);z,b,out=load(pred['inputs'][0]),load(pred['inputs'][1]),load(pred['output'])
        pin,pout=load(proj['inputs'][0]),load(proj['output'])
        require(a.shape==(1,3,10) and np.array_equal(a[0],core.validate_request(request)) and emb.ndim==3 and emb.shape[:2]==(1,3) and
            z.shape==(1,3,192) and np.array_equal(z[0],request['latents']) and np.array_equal(b,emb) and
            out.shape==(1,3,192) and pin.shape==pout.shape==(3,192) and np.array_equal(pin,out.reshape(3,192)),'complete real native module bridges')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':value=prepare(a.output)
    elif a.command=='verify':value=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='consume':value=consume(a.output,a.report_sha)
    else:value=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(value).decode(),end='')


if __name__=='__main__':main()
