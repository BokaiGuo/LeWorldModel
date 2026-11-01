"""E3 K1/K3 RAM numerical dispatcher and bounded transport rehearsal, NOT CUDA runner."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as resolver
from experiments.spectral_reachability import agent_downstream_resource_contract_20261029_v1 as resources

NAME='agent_e3_gd_numerical_preparation_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
MiB=1024**2
RESOLVER_SHA='221786d1cf9438431cd12e30f9da287bf4ed3f1c750c680074814887d7297e34'
ACTUAL_PREPARE=ROOT/'results/agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json'
ACTUAL_PREPARE_SHA='b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26'
CAP=512*MiB
RESERVE=64*MiB
WALL=1800
need=resolver.require


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(raw(x)).hexdigest()


def validate_body(body):
    need(type(body) is dict and set(body)==set(resolver.PLANNER_FIELDS),'planner body ONLY exact whitelist')
    need(body['schema']==resolver.NAME+'.task' and body['model_sha256']==resolver.MODEL_SHA
        and body['root_sha256'] in resolver.KNOWN_ROOTS,'original task/model/root identity')
    K=body['horizon_packets'];packet_budget(K)
    need(body['domain']==resolver.domain_spec(body['domain']['name']),'exact original domain')
    resolver.cp5.context(body['root_context'])
    resolver.array(body['goal'],(192,),f32=True)
    need(np.array_equal(resolver.array(body['projection'],(192,192),f32=False),np.eye(192)),'identity192 Q')
    resolver.array(body['action_mean'],(2,),f32=True)
    need((resolver.array(body['action_std'],(2,),f32=True)>0).all(),'positive original normalizer')
    need(type(body['feedback_boundary']) is int and body['feedback_boundary']==5
        and type(body['task_deadline_control']) is int and body['task_deadline_control']==20,'cp5 input, control20 deadline')
    resolver.cp5.hash_value(body['identity_sha256']);return deepcopy(body)


def load_planner_inputs(path=ACTUAL_PREPARE,expected=ACTUAL_PREPARE_SHA):
    """SHA-authenticated accepted resolver output, NOT recursive upstream strict."""
    p=resolver.absolute(str(path));need(p.is_file() and p.stat().st_size<=16*MiB,'bounded resolver output')
    b=p.read_bytes();need(hashlib.sha256(b).hexdigest()==expected,'resolver prepare SHA')
    d=resolver.cp5.decode(b)
    need(d['schema']==resolver.NAME+'.prepare' and d['ready']==16 and d['blocked']==0
        and d['planner_allowlist']==list(resolver.PLANNER_FIELDS),'complete16 resolver authority')
    need(len(d['cells'])==16,'fixed full denominator')
    bodies=[]
    for row,expected_cell in zip(d['cells'],resolver.roster()):
        need(all(row[k]==expected_cell[k] for k in ('index','cell_id','root','goal_index','horizon_packets','domain'))
            and row['status']=='input_ready_NOT_run_authorized','exact ordered cells')
        task=row['task'];body=validate_body(task['body'])
        need(task['task_sha256']==resolver.digest(body) and body['root_sha256']==resolver.KNOWN_ROOTS[row['root']]
            and body['horizon_packets']==row['horizon_packets'] and body['domain']['name']==row['domain'],'task identity join')
        bodies.append(body)
    # Never return cells/evaluator/native labels to planner callbacks.
    return bodies


def optimize_body(body,backend_factory,tape):
    t=validate_body(body)
    backend=backend_factory(deepcopy(t))
    return optimize_node(t['root_context'],t['goal'],t['horizon_packets'],t['domain']['name'],backend,tape,
        normalizer=(t['action_mean'],t['action_std']))


def packet_budget(K):
    need(type(K) is int and K in (1,3),'literal K1/K3')
    return dict(candidate_forwards=33,replays=3,AD_forwards=32,predict=68*K,reverse=32)


def axes27(K):
    """Successful numerical node EXPECTATION, not observed production accounting."""
    n=packet_budget(K)['predict'];a=dict.fromkeys(resources.P3_AXES,0)
    a.update(predict_API=n,predict_samples=n,predict_context_tokens=3*n,
        action_encoder_calls=n,action_encoder_samples=n,action_encoder_context_tokens=3*n,
        predictor_calls=n,predictor_samples=n,predictor_context_tokens=3*n,
        pred_proj_calls=n,pred_proj_rows=3*n,grad_API=32,VJP_directions=32,raw_backward_callbacks=32,
        replay_calls=3,metric_calls=36)
    return a


def transport_budget(K):
    """Conservative new transport envelope; size guards enforce it before publication.

    No gzip: no compression assumptions or per-record flush hidden overhead.
    Per packet at most32 float32 arrays, each<=768 elements (native ABI inventory
    still needs a production adapter); descriptors are charged in JSON events.
    """
    n=packet_budget(K)['predict']*3
    arrays=32*n+3*32*4
    # Every NPY independently charged 4096 header/padding + data maximum.
    per_array=4096+768*4
    events=4*n+3*256+16
    event_cap=16*1024
    total=arrays*per_array+events*(event_cap+1)+16*MiB+RESERVE
    return dict(K=K,predict_packets=n,array_files=arrays,array_bytes_cap=arrays*per_array,
        npy_header_each_cap=4096,largest_array_elements=768,event_count_cap=events,
        event_payload_each_cap=event_cap,event_bytes_cap=events*(event_cap+1),
        metadata_cap=16*MiB,failure_and_completion_reserve=RESERVE,total_bytes_cap=total,
        job_cap=CAP,fits=total<=CAP,compression='none',sync_flush_overhead=0,
        encoder_transport_included=False,physics_transport_included=False,
        native_array_inventory_certified=False)


def proof():
    return dict(schema=NAME+'.proof',roster=resolver.roster(),full=resolver.complete_path_budget(),
        node27={str(K):axes27(K) for K in (1,3)},transport={str(K):transport_budget(K) for K in (1,3)},
        wall=dict(job_cap_seconds=WALL,all16_job_cap_sum_seconds=16*WALL,
            boundary='before verification through report fsync/hash and separate completion',
            predicted_seconds=None,actual_seconds=None,latency_guarantee=False),
        GPU=dict(allocated_cap=3*1024**3,reserved_cap=4*1024**3,measured_peak=None,
            activation_and_workspace_bound=None,fit_proven=False,
            reason='K3 retained autograd graph and native CUDA workspaces require MAIN measurement'),
        extra_axes=dict(scalar_reverse=1536,AD_primal_packets=3072,scalar_loss_backward=0,
            engine_backward=None,FLOPs=None),
        new_calls=dict(neural=0,physics=0,encoder=0),production_ready=False,
        full_upstream_rebuilt_here=False,scientific_admission=False)


def project(x,K,domain):
    need(isinstance(x,np.ndarray) and x.dtype.kind in 'fi','numeric array proposal')
    x=resolver.parameters(x.tolist(),K).copy()
    if domain=='Dx':x[:,:,1]=0.
    for packet in x:
        norm=np.linalg.norm(packet.reshape(10))
        if norm>resolver.PACKET_CAP:packet*=resolver.PACKET_CAP/norm
    # Original projected_logits cast/inward algorithm, generalized packet count.
    x=x.astype(np.float32)
    for packet in x:
        if np.linalg.norm(packet.astype(float).reshape(10))>resolver.PACKET_CAP:
            packet*=np.float32(1-4*np.finfo(np.float32).eps)
    resolver.mask_chart(x.tolist(),K,domain)
    return x


class Tape:
    """RAM or real disk single-LF hashchain; independent array files, no inline blobs."""
    def __init__(self,path=None,clock=time.monotonic,started=None):
        self.path=Path(path) if path is not None else None
        self.clock=clock;self.started=clock() if started is None else started
        self.events=[];self.attempted={'predict':0,'reverse':0};self.completed=dict(self.attempted)
        self.bytes=0;self.array_count=0
        if self.path is not None:self.path.mkdir(exist_ok=False)
    def check(self):
        elapsed=self.clock()-self.started
        need(0<=elapsed<=WALL,'inclusive stage wall exceeded')
        need(self.bytes<=CAP-RESERVE,'working bytes leave failure reserve')
    def put(self,value):
        self.check();a=np.asarray(value)
        need(a.dtype==np.dtype('float32') and a.size<=768 and np.isfinite(a).all(),'bounded finite native f32 array')
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);b=buf.getvalue()
        need(len(b)<=4096+a.nbytes,'NPY header cap')
        need(self.array_count<transport_budget(3)['array_files'] and self.bytes+len(b)<=CAP-RESERVE,'array count/byte write-before cap')
        desc=dict(file=f'a{self.array_count:06}.npy',sha256=hashlib.sha256(b).hexdigest(),shape=list(a.shape),dtype='<f4')
        self.array_count+=1
        if self.path is not None:self._write(self.path/desc['file'],b)
        self.bytes+=len(b);self.check();return desc
    def _write(self,path,b):
        with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    def emit(self,kind,**fields):
        self.check();e=dict(index=len(self.events),previous=self.events[-1]['sha256'] if self.events else None,kind=kind,**fields)
        e['sha256']=digest(e);b=raw(e)
        need(len(b)<=16*1024,'event payload cap')
        need(len(self.events)<transport_budget(3)['event_count_cap'] and self.bytes+len(b)+1<=CAP-RESERVE,'event count/byte write-before cap')
        if self.path is not None:
            with (self.path/'events.jsonl').open('ab') as f:f.write(b+b'\n');f.flush();os.fsync(f.fileno())
        self.events.append(e);self.bytes+=len(b)+1;self.check()
    def call(self,kind,callback):
        self.check();self.attempted[kind]+=1;self.emit('start',resource=kind)
        try:value=callback()
        except Exception as e:self.emit('failed',resource=kind,error=type(e).__name__);raise
        self.completed[kind]+=1;self.emit('end',resource=kind);return value


def read_tape(path):
    path=Path(path);events=[];previous=None;stack=[]
    for i,line in enumerate((path/'events.jsonl').read_bytes().splitlines()):
        need(0<len(line)<=16384 and i<transport_budget(3)['event_count_cap'],'bounded nonblank events')
        e=json.loads(line);signature=e.pop('sha256')
        need(e['index']==i and e['previous']==previous and digest(e)==signature,'exact event chain')
        if e['kind']=='start':stack.append(e['resource'])
        elif e['kind'] in ('end','failed'):
            need(stack and stack.pop()==e['resource'],'nested callback lifecycle')
        for d in e.get('arrays',[]):
            need(set(d)=={'file','sha256','shape','dtype'} and d['dtype']=='<f4'
                and type(d['shape']) is list and all(type(v) is int and v>=0 for v in d['shape'])
                and Path(d['file']).name==d['file'],'array descriptor ABI')
            p=path/d['file'];need(p.is_file() and not p.is_symlink() and p.stat().st_size<=7168,'bounded regular array')
            b=p.read_bytes();need(hashlib.sha256(b).hexdigest()==d['sha256'],'array bytes SHA')
            a=np.load(io.BytesIO(b),allow_pickle=False)
            need(a.dtype==np.float32 and a.size<=768 and list(a.shape)==d['shape'] and np.isfinite(a).all(),'saved array ABI')
        previous=signature;e['sha256']=signature;events.append(e)
    need(not stack,'unreturned callback remains partial, not complete tape')
    return events


def optimize_node(context,goal,K,domain,backend,tape,*,normalizer=None):
    """Executable callback numerical kernel. RAM backend ONLY; no CUDA entry point.

    backend.packet returns context (latent3, history2x10), IO dict; backend.reverse
    invokes supplied replay exactly once then returns fresh raw gradient+scalar.
    """
    need(getattr(backend,'kind',None)=='RAM_FIXTURE','production backend not authorized')
    packet_budget(K);goal=resolver.array(goal,(192,),f32=False)
    resolver.cp5.context(context)
    if normalizer is None:normalizer=(np.full(2,.01,np.float32).tolist(),np.full(2,.2,np.float32).tolist())
    mean,std=(resolver.array(v,(2,),f32=True).astype(np.float32) for v in normalizer)
    need((std>0).all(),'positive once-normalizer')
    def forward(actions,mode):
        c=deepcopy(context)
        for k in range(K):
            def execute():
                nc,arrays=backend.packet(deepcopy(c),actions[5*k:5*k+5].copy())
                resolver.cp5.context(nc)
                resolver.array(nc['latent_history'],(3,192),f32=True)
                resolver.array(nc['action_history'],(2,10),f32=True)
                prior_z=np.asarray(c['latent_history'],np.float32)
                prior_h=np.asarray(c['action_history'],np.float32)
                token=((actions[5*k:5*k+5]-mean)/std).reshape(1,10)
                need(np.array_equal(np.asarray(nc['latent_history'],np.float32)[:2],prior_z[1:])
                    and np.array_equal(np.asarray(nc['action_history'],np.float32),np.concatenate([prior_h,token])[1:]),
                    'full rolling context and original once-normalized raw packet')
                need(type(arrays) is dict and 0<len(arrays)<=32,'complete bounded fixture IO inventory')
                ds=[tape.put(v) for v in arrays.values()]
                tape.emit('packet',mode=mode,packet=k,keys=list(arrays),arrays=ds)
                return nc
            c=tape.call('predict',execute)
        return c
    def evaluate(x,mode):
        actions=resolver.mask_chart(x.tolist(),K,domain);c=forward(actions,mode)
        cost=float(np.mean((np.asarray(c['latent_history'][-1],float)-goal)**2))
        tape.emit('candidate',mode=mode,cost=cost,arrays=[tape.put(actions)])
        return dict(logits=x.copy(),actions=actions,context=c,cost=cost)
    x=np.zeros((K,5,2));candidates=[evaluate(x,'iterate0')]
    for it in range(32):
        current=candidates[-1];count=tape.attempted['predict']
        def reverse():
            def primal(a):
                need(np.asarray(a).dtype==np.float32 and np.array_equal(a,current['actions']),'AD raw point exact')
                return forward(a,'AD')
            result=backend.reverse(current['actions'].copy(),goal.copy(),primal)
            need(tape.attempted['predict']-count==K,'one complete fresh AD primal')
            return result
        fresh,gradient,loss=tape.call('reverse',reverse)
        need(raw(fresh)==raw(current['context']),'AD fresh point exact context')
        need(isinstance(gradient,np.ndarray) and gradient.dtype==np.float32,'native raw f32 gradient array')
        g=resolver.array(gradient.tolist(),(5*K,2),f32=True)
        need(np.asarray(loss).shape==() and np.asarray(loss).dtype==np.float32 and np.isfinite(loss),'native scalar loss')
        expected=np.mean((np.asarray(fresh['latent_history'][-1],np.float32)-goal.astype(np.float32))**2,dtype=np.float32)
        need(loss==expected,'native scalar objective')
        tape.emit('gradient',iteration=it,arrays=[tape.put(g),tape.put(np.asarray(loss))])
        chain=1-current['actions'].astype(float)**2
        objective=resolver.mask_gradient((g.astype(float)*chain).reshape(K,5,2).tolist(),K,domain)
        x=project(current['logits']-objective,K,domain)
        candidates.append(evaluate(x,f'iterate{it+1}'))
    winner=min(range(33),key=lambda i:(candidates[i]['cost'],i))
    tape.emit('selection',winner=winner)
    for i in range(3):
        fresh=evaluate(candidates[winner]['logits'],f'replay{i}')
        need(raw(fresh['context'])==raw(candidates[winner]['context']) and fresh['cost']==candidates[winner]['cost'],'independent winner replay exact')
    n=68*K
    need(tape.attempted==tape.completed==dict(predict=n,reverse=32),'actual no-refund complete node census')
    return dict(winner=winner,candidate_costs=[c['cost'] for c in candidates],
        commands=candidates[winner]['actions'].tolist(),context=candidates[winner]['context'],
        attempted=deepcopy(tape.attempted),completed=deepcopy(tape.completed),scientific_admission=False)


def build():
    need(resolver.sha(resolver.__file__)==RESOLVER_SHA,'frozen resolver source')
    files=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        Path(resolver.__file__).resolve(),Path(resources.__file__).resolve()]
    return dict(proof=proof(),sources={str(p):resolver.sha(p) for p in files},
        resolver_authority=dict(path=str(ACTUAL_PREPARE),sha256=ACTUAL_PREPARE_SHA),
        resolver_authority_read_here=False)


def write_new(path,value):
    p=resolver.absolute(str(path));b=raw(value);need(len(b)<=16*MiB,'bounded proof metadata')
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return resolver.sha(p)


def verify(path,expected):
    p=resolver.absolute(str(path));need(p.stat().st_size<=16*MiB and resolver.sha(p)==expected,'saved proof SHA')
    need(raw(json.loads(p.read_bytes()))==raw(build()),'whole own proof rebuild')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('inspect','inspect-inputs','verify'))
    p.add_argument('--saved');p.add_argument('--sha');a=p.parse_args()
    if a.command=='verify':verify(a.saved,a.sha);print('verified own preparation proof only')
    elif a.command=='inspect-inputs':
        bodies=load_planner_inputs()
        print(raw(dict(schema=NAME+'.input_inspection',cells=16,
            bodies=[dict(sha256=resolver.digest(b),root_sha256=b['root_sha256'],K=b['horizon_packets'],domain=b['domain']['name']) for b in bodies],
            new_neural_calls=0,new_physics_calls=0,full_upstream_rebuilt_here=False)).decode())
    else:print(raw(build()).decode())


if __name__=='__main__':main()
