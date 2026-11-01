"""Saved-only all24 compiler validation; no model/physics runtime import."""
import argparse
from collections import Counter
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import os
import numpy as np
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as packet

NAME='capability_suffix_factorial_cuda_MAIN_20261027_v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate key');out[k]=v
        return out
    def constant(v):raise ValueError('nonfinite JSON '+v)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=constant)


def raw_action(params,scale):
    p=np.asarray(params,np.float32)
    require(p.shape==(10,) and np.isfinite(p).all() and scale in (1.,.1),'own finite parameters/scale')
    return scale*np.tanh(p).reshape(5,2)


def normalized(raw):
    mean=np.tile(np.asarray([-.0087,.0068],np.float32),5)
    std=np.tile(np.asarray([.2019,.2002],np.float32),5)
    return (np.asarray(raw,np.float32).reshape(10)-mean)/std


def check_result(case,result):
    goal=np.asarray(case['goal'],np.float64);n=np.asarray(result['nominal'],np.float64);j=np.asarray(result['jacobian'],np.float64)
    require(goal.shape==n.shape==(192,) and j.shape==(192,10) and all(np.isfinite(x).all() for x in (goal,n,j)),'actual native arrays')
    bank=packet.spectral_bank(j,n,goal,np.eye(192))
    require(result['bank']==bank and len(result['candidates'])==10,'full saved spectral bank')
    for i,c in enumerate(result['candidates']):
        require(c['index']==i and c['kind']==bank['kinds'][i] and
                np.array_equal(np.asarray(c['params'],np.float32),np.asarray(bank['params'][i],np.float32)),'fixed candidate params/order')
        require(np.array_equal(np.asarray(c['raw'],np.float32),raw_action(c['params'],case['scale'])),'own raw mapping, no normalizer swap')
        pred=np.asarray(c['predicted'],np.float64)
        require(pred.shape==(192,) and np.isfinite(pred).all() and c['Q']==float(np.mean((pred-goal)**2)),'original nonlinear Q')
    winner=min(result['candidates'],key=lambda c:(c['Q'],c['index']))
    require(result['winner']==winner and result['replay_exact'] is True,'stable true nonlinear winner')
    return winner


def consume(directory,expected):
    d=Path(directory).resolve();require(sha(d/'report.json')==expected,'external compiler report SHA')
    r=read(d/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='all24_plans_sealed' and r['errors']==[] and
            r['predict_API']==312 and r['P1_admission'] is False,'complete raw compiler')
    require(r['model_hash_before']==r['model_hash_after']=='4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546','unchanged original model')
    require(set(x.name for x in d.iterdir())==set(r['artifact_sha256'])|{'report.json'},'artifact census')
    for name,h in r['artifact_sha256'].items():require(Path(name).name==name and sha(d/name)==h,'artifact SHA')
    require(sha(r['prepared'])==r['prepare_sha256'],'external prepare SHA');p=read(r['prepared'])
    require(read(d/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual execution binding')
    require(p['contract']==r['contract'] and len(p['cases'])==len(r['rows'])==24 and r['commands_released'] is False,'fixed contract/raw release stage')
    for path,h in p['bindings'].items():require(sha(path)==h,'frozen input/source SHA')
    ledger=r['ledger'];require(len(ledger)==936 and [e['index'] for e in ledger]==list(range(936)) and all(e['completed'] is True for e in ledger),'936 actual complete module entries')
    require(Counter(e['resource'] for e in ledger)==Counter(dict(action_encoder=312,predictor=312,pred_proj=312)),'separate native counts')
    def array(receipt):
        name=receipt['file'];require(Path(name).name==name,'flat tensor receipt');path=d/name
        require(sha(path)==receipt['file_sha256'],'tensor file SHA')
        content=gzip.decompress(path.read_bytes());require(len(content)<4*1024**2,'bounded tensor')
        f=BytesIO(content);a=np.load(f,allow_pickle=False)
        require(f.tell()==len(content) and a.dtype==np.float32 and np.isfinite(a).all() and list(a.shape)==receipt['shape'] and receipt['dtype']=='float32' and hashlib.sha256(a.tobytes()).hexdigest()==receipt['array_sha256'],'native original float32 tensor')
        return a
    events=[json.loads(x) for x in gzip.decompress((d/'events.jsonl.gz').read_bytes()).splitlines()]
    attempts=[e['entry'] for e in events if e.get('kind')=='tensor_module_attempt']
    returns=[e['entry'] for e in events if e.get('kind')=='tensor_module_completed']
    require(len(attempts)==len(returns)==936 and sorted(returns,key=lambda x:x['index'])==ledger,'all durable module attempts/returns')
    require(all(a=={k:v for k,v in b.items() if k not in ('inputs','output','completed')}|{'completed':False} for a,b in zip(attempts,ledger,strict=True)),'attempt identities')
    require(len(r['operations'])==72 and Counter(x['kind'] for x in r['operations'])==Counter(dict(jacobian=24,svd=48)) and all(x['completed'] is True for x in r['operations']),'actual math census')
    require([x['entry'] for x in events if x.get('kind')=='math_attempt']==[{k:v for k,v in x.items() if k!='completed'}|{'completed':False} for x in r['operations']] and [x['entry'] for x in events if x.get('kind')=='math_return_or_failed']==r['operations'],'durable math attempts/returns')
    plans=[]
    for i,(case,row) in enumerate(zip(p['cases'],r['rows'],strict=True)):
        require(case['index']==row['index']==i and case['anchor']==i//3 and case['variant']==('open_full','teacher_full','teacher_small')[i%3] and case['scale']==(.1 if i%3==2 else 1.) and row['status']=='sealed','fixed24 ordering/scale')
        require(read(d/f'case_{i:02}.json')==row,'durable case seal')
        result=row['result'];winner=check_result(case,result)
        calls=[np.zeros(10,np.float32),np.zeros(10,np.float32)]+[np.asarray(x['params'],np.float32) for x in result['candidates']]+[np.asarray(winner['params'],np.float32)]
        entries=[e for e in ledger if e['scope']==dict(plan_index=i)]
        require(len(entries)==39,'13 complete native forwards per case')
        for call,params in enumerate(calls):
            ae,pred,proj=entries[3*call:3*call+3]
            require([e['resource'] for e in (ae,pred,proj)]==['action_encoder','predictor','pred_proj'],'native call sequence')
            ain,aout=array(ae['inputs'][0]),array(ae['output'])
            hin,pa=[array(x) for x in pred['inputs']]
            pout,pi,po=array(pred['output']),array(proj['inputs'][0]),array(proj['output'])
            outgoing=np.concatenate([np.asarray(case['context']['action_history'],np.float32),normalized(raw_action(params,case['scale']))[None]],axis=0)
            require(ain.shape==(1,3,10) and aout.shape==hin.shape==pa.shape==pout.shape==(1,3,192) and pi.shape==po.shape==(3,192),'complete module tensor shapes')
            require(np.array_equal(ain[0],outgoing) and np.array_equal(hin[0],np.asarray(case['context']['latent_history'],np.float32)) and np.array_equal(aout,pa) and np.array_equal(pout.reshape(3,192),pi),'exact context/action/predictor bridges')
            if call==0:target=result['nominal']
            elif 2<=call<12:target=result['candidates'][call-2]['predicted']
            elif call==12:target=winner['predicted']
            else:target=None
            if target is not None:require(np.array_equal(po[-1],np.asarray(target,np.float32)),'native final predicted output')
        plans.append(dict(index=i,anchor=case['anchor'],variant=case['variant'],scale=case['scale'],root_slot=case['root_slot'],condition=case['condition'],arm=case['arm'],
            selection_digest=case['selection_digest'],task_id=case['task_id'],original_plan=case['original_plan'],prefix=case['prefix'],
            suffix=winner['raw'],commands=case['prefix']+winner['raw'],predicted_Q=winner['Q'],winner_index=winner['index']))
    for path,h in p['bindings'].items():require(sha(path)==h,'final input seal')
    require(sha(d/'report.json')==expected,'final report seal')
    return dict(schema='consume_suffix_factorial_MAIN_20261027_v1.report',status='all24_saved_verified',source_report=str(d/'report.json'),source_report_sha256=expected,
        prepare_sha256=r['prepare_sha256'],plans=plans,branches=24,known_roots=2,fresh_roots=0,module_entries=936,
        saved_verification_SVD=48,new_neural_calls=0,new_physics_calls=0,commands_released=True,P1_admission=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('input','report-sha','output'):parser.add_argument('--'+k,required=True)
    a=parser.parse_args();result=consume(a.input,a.report_sha)
    with Path(a.output).open('xb') as f:f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode());f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'],sha256=sha(a.output),plans=24,module_entries=936)))
