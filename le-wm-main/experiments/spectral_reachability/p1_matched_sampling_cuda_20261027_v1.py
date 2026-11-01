"""Fixed small forward-only P1 controls, NOT admission or a default/SOTA protocol.

MAIN alone may run CUDA. Prepare reads cached input/task/provenance/accounting,
never direct trajectories or witness outcomes. No simulator or image forwards.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np

from experiments.spectral_reachability import p1_two_stage_witness_20261027_v1 as bridge

screen, core, io = bridge.screen, bridge.core, bridge.io
require = core.require
NAME = 'p1_matched_sampling_cuda_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
LIMIT, FLOOR, FINAL = 8*1024**2, 200*1024**2, 512*1024
METHODS = ('cem', 'iid_random')
# Predeclared engineering choices, not selected using direct/witness gains.
PROTOCOL = dict(methods=list(METHODS), batches=6, population=10, candidates=60,
    zero='only candidate 0; remaining 59 stochastic nonzero samples; no mean injection',
    cem=dict(initial_mean=0., initial_sigma=.5, sigma_floor=.05, elite=3,
             ddof=0, smoothing=0, update='executed feasible logits; current batch stable(cost,index) elites',
             updates=6, final_update_used_for_sampling=False,
             proposal='independent diagonal Gaussian then per-packet radial projection'),
    iid_random='independent uniform-volume 10D logit balls per packet; Gaussian direction and U**(1/10) radius',
    seeds=dict(base=20261027, root_order='ascending stable root_sha256', root_stride=1000,
               method_offsets=dict(cem=0, iid_random=100)),
    rng='NumPy Generator(PCG64); one stream per root/method; zero consumes no draw',
    domain='original independent packet logit balls .75*sqrt(5); float32 NumPy tanh raw[-1,1]',
    objective='original same task Q/goal terminal mean-square cost; full H2 from full root context',
    feedback='open_loop_model_only', selection='best of all60; first index wins exact ties',
    verification='new H2 nominal + new H2 whole replay + new H2 segmented replay',
    match='126 predict/action_encoder per root/arm ONLY; NOT AD/FLOPs/wall-time matched',
    comparison='fixed local small-budget control, NOT paper default protocol or SOTA',
    P1_admission=False, physical_future_labels_read=False)


def budgets(method):
    require(method in METHODS, 'known method')
    zero = dict(jacobian=0, vjp_directions=0, svd=0, linear_solve=0, encoder=0, projector=0)
    return dict(sampling=dict(zero, predict=120, action_encoder=120, generation=6,
                    selection=60, cem_update=6 if method=='cem' else 0),
                winner=dict(zero, predict=6, action_encoder=6, generation=0, selection=0, cem_update=0))


def seed_for(root_index, method):
    require(type(root_index) is int and 0<=root_index<2 and method in METHODS, 'fixed two root seed table')
    return PROTOCOL['seeds']['base']+1000*root_index+PROTOCOL['seeds']['method_offsets'][method]


def population(method, rng, mean, sigma, batch, cap):
    require(method in METHODS and type(batch) is int and 0<=batch<6, 'fixed generation')
    n=9 if batch==0 else 10
    if method=='cem':
        values=rng.normal(mean, sigma, size=(n,2,10))
    else:
        direction=rng.normal(size=(n,2,10))
        norm=np.linalg.norm(direction,axis=-1,keepdims=True)
        require((norm>0).all(), 'finite nonzero random directions; no hidden retries')
        values=direction/norm*(cap*rng.random((n,2,1))**.1)
    values=[screen.projected_logits(x,cap) for x in values]
    require(all(np.any(x!=0) for x in values), '59 genuinely nonzero stochastic slots; no hidden retries')
    return ([np.zeros((2,10))] if batch==0 else [])+values


def cem_update(values, costs):
    """Repo P2 custom CEM empirical elite mean/std rule, now elite3/logit balls.

    Deliberately NOT stable_worldmodel default (which uses torch std ddof1),
    nor old P2 raw-box/repeated-mean-injection protocol. No softmax weights.
    """
    a=core.array(values,(10,2,10)); c=core.array(costs,(10,))
    ids=sorted(range(10),key=lambda i:(c[i],i))[:3]
    elite=a[ids]
    return elite.mean(axis=0),np.maximum(elite.std(axis=0,ddof=0),.05),ids


def counts(events, method):
    return {phase:{r:dict(attempted=sum(e['samples'] for e in events if e['phase']==phase and e['resource']==r),
        completed=sum(e['samples'] for e in events if e['phase']==phase and e['resource']==r and e['completed']))
        for r in caps} for phase,caps in budgets(method).items()}


class Meter(screen.Meter):
    def __init__(self,method,sink):
        self.method=method
        super().__init__({r:sum(c[r] for c in budgets(method).values()) for r in budgets(method)['sampling']},sink)
        self.phase='sampling'

    def emit(self,kind,**values):
        self.sink(dict(kind=kind,phase=self.phase,evaluation_id=self.group,**deepcopy(values)))

    def begin(self,resource):
        cap=budgets(self.method)[self.phase].get(resource,0)
        used=sum(e['samples'] for e in self.extended if e['phase']==self.phase and e['resource']==resource)
        require(used<cap,'predispatch '+self.phase+' '+resource+' cap')
        e=dict(index=len(self.extended),phase=self.phase,evaluation_id=self.group,
               resource=resource,samples=1,completed=False)
        self.extended.append(e);idx=None
        if resource in core.RESOURCES:
            idx=len(self.events);self.events.append(dict(e,index=idx))
        self.emit('operation_attempted',event=e)
        return e,idx


def run_arm(task,method,seed,backend,sink):
    """Only backend.guard/rollout needed; injected array backends are mocks."""
    core.validate_task(task)
    require(task['packet_logit_cap']==float(.75*np.sqrt(5)), 'original P1 scale')
    require(np.array_equal(task['projection'],np.eye(192)), 'same original identity Q')
    meter=Meter(method,sink);t=deepcopy(task);best=None
    result=dict(schema=NAME+'.arm',status='started',task_id=core.task_id(t),method=method,seed=seed,
                candidates=[],winner=None,witness=None,error=None,secondary_errors=[],P1_admission=False)
    start=time.monotonic()
    try:
        meter.emit('arm_started',task_id=result['task_id'],method=method,seed=seed,budget=budgets(method))
        rng=np.random.Generator(np.random.PCG64(seed))
        mean=np.zeros((2,10));sigma=np.full((2,10),.5)
        with backend.guard(meter):
            for batch in range(6):
                meter.group=f'batch_{batch}'
                values=meter.call('generation',lambda:population(method,rng,mean,sigma,batch,t['packet_logit_cap']))
                meter.emit('population_frozen',batch=batch,logits=[v.tolist() for v in values],mean=mean.tolist(),sigma=sigma.tolist())
                costs=[]
                for slot,a in enumerate(values):
                    index=10*batch+slot;meter.group=f'candidate_{index}'
                    raw=np.tanh(a.astype(np.float32)).reshape(10,2)
                    paths=backend.rollout(t,raw,meter,'candidate')
                    require(len(paths)==2,'full H2 candidate')
                    require(raw.tolist()==bridge.edge(t,t['root_context'],a[0],paths[0])+
                            bridge.edge(t,paths[0],a[1],paths[1]),'exact candidate actions/context chronology')
                    c=dict(index=index,batch=batch,slot=slot,logits=a.tolist(),commands=raw.tolist(),
                           contexts=paths,cost=core.model_cost(t,paths[-1]['latent_history'][-1]))
                    # Full lossless contexts durable before selection or elite updates.
                    meter.emit('candidate_complete',candidate=c)
                    def select():
                        return c if best is None or c['cost']<best['cost'] else best
                    best=meter.call('selection',select);costs.append(c['cost'])
                    result['candidates'].append({k:deepcopy(c[k]) for k in ('index','batch','slot','logits','cost')})
                    meter.emit('incumbent',index=best['index'])
                if method=='cem':
                    meter.group=f'update_{batch}'
                    mean,sigma,ids=meter.call('cem_update',lambda:cem_update(values,costs))
                    meter.emit('elite_update',batch=batch,elite_slots=ids,mean=mean.tolist(),sigma=sigma.tolist())
            result['winner']=deepcopy(best)
            meter.phase='winner';meter.group='independent_winner'
            w=screen.evaluate(t,best['logits'],backend,meter)
            require(w['contexts']==best['contexts'] and w['commands']==best['commands'] and w['cost']==best['cost'],
                    'paid independent nominal/whole/segmented winner exact')
            result['witness']=w
        work=counts(meter.extended,method)
        require(all(work[p][r]==dict(attempted=n,completed=n) for p,caps in budgets(method).items() for r,n in caps.items()),
                'exact complete per-phase budget, no refunded work')
        result['status']='sampling_completed_not_admitted'
    except BaseException as exc:
        result['status']='failed';result['error']=io.error('arm',exc)
    result.update(resources=counts(meter.extended,method),wall_seconds=time.monotonic()-start)
    try:meter.emit('arm_finished',status=result['status'],error=result['error'],resources=result['resources'])
    except BaseException as exc:
        err=io.error('arm_finished',exc)
        if result['error'] is None:result.update(error=err,status='failed')
        else:result['secondary_errors'].append(err)
    return result


class Store:
    """Lossless streaming gzip JSONL; fsync EACH event, even attempted entries.

    Z_SYNC_FLUSH makes a partial stream incrementally decodable after failure.
    All output bytes share 8MiB cap; 512KiB withheld for finalization. No closure
    snapshots or lossy float formatting. Quota failure never reduces candidates.
    """
    def __init__(self,output):
        self.output=Path(output).absolute();self.streams={};self.broken=set()
        require(not self.output.exists(),'NEW output only')
        require(shutil.disk_usage(self.output.parent).free>=FLOOR+LIMIT,'8MiB output plus 200MiB reserve')
        self.output.mkdir()

    def write(self,name,raw,*,append=False,final=False):
        require(Path(name).name==name and name not in ('.','..'),'flat local artifact')
        used=sum(p.stat().st_size for p in self.output.iterdir())
        require(used+len(raw)<=LIMIT-(0 if final else FINAL),'global compact output quota')
        require(shutil.disk_usage(self.output).free>=FLOOR+len(raw),'200MiB floor before every write')
        with (self.output/name).open('ab' if append else 'xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())

    def put(self,name,value,*,final=False):
        self.write(name,(core.canonical(value)+'\n').encode(),final=final)

    def sink(self,name):
        require(name.endswith('.jsonl.gz') and name not in self.streams,'new compact journal')
        comp=zlib.compressobj(6,zlib.DEFLATED,31);self.streams[name]=comp
        self.write(name,b'')
        def emit(value):
            require(name not in self.broken,'failed journal cannot resume')
            raw=(core.canonical(value)+'\n').encode()
            try:self.write(name,comp.compress(raw)+comp.flush(zlib.Z_SYNC_FLUSH),append=True)
            except BaseException:
                # Compressor state advanced but bytes may not be durable. Do
                # NOT append a misleading footer from that advanced state.
                self.broken.add(name)
                raise
        emit(dict(kind='journal_created',lossless=True))
        return emit

    def close(self):
        errors=[]
        for name in list(self.streams):
            comp=self.streams.pop(name)
            if name in self.broken:
                errors.append(dict(stage='journal_finish',message='partial stream; prior write failed',journal=name))
                continue
            try:self.write(name,comp.flush(zlib.Z_FINISH),append=True,final=True)
            except BaseException as exc:errors.append(io.error('journal_finish',exc))
        return errors


def source_files():
    files=set(io.source_files()+[Path(__file__).resolve(),Path(bridge.__file__).resolve(),
        Path(screen.__file__).resolve(),Path(core.__file__).resolve(),Path(bridge.upstream.__file__).resolve(),
        Path(bridge.source.__file__).resolve(),
        ROOT/'experiments/spectral_reachability/p1_conditional_source_metadata_20261027_v1.py',
        ROOT/'experiments/spectral_reachability/p2_full20_sampling_20261027_v2.py',
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')])
    return sorted(files)


def collect(direct_prepared,direct_prepare_sha,task_directory):
    # Calls ONLY upstream task/input collector, NOT witness run/verify/results.
    inputs=bridge.collect(direct_prepared,direct_prepare_sha,task_directory)
    inputs['tasks'].sort(key=lambda x:x['root_sha256'])
    require(len(inputs['tasks'])==2 and len({x['root_sha256'] for x in inputs['tasks']})==2,'two stable actual roots')
    return inputs


def prepare(direct_prepared,direct_prepare_sha,task_directory,output):
    inputs=collect(direct_prepared,direct_prepare_sha,task_directory)
    p=dict(schema=NAME+'.protocol',algorithm=deepcopy(PROTOCOL),inputs=inputs,
        budgets={m:budgets(m) for m in METHODS},seeds=[{m:seed_for(i,m) for m in METHODS} for i in range(2)],
        sources_sha256={str(f):io.sha(f) for f in source_files()},output_limit_bytes=LIMIT,reserve_bytes=FLOOR,
        sealing='hash manifest; no large closure copy; external prepare SHA required')
    s=Store(output);s.put('protocol.json',p)
    r=dict(schema=NAME+'.prepare',status='prepared_not_executed',protocol_sha256=io.sha(s.output/'protocol.json'))
    s.put('prepare.json',r,final=True)
    return r


def verify(folder,expected_sha):
    folder=Path(folder).absolute()
    require(io.sha(folder/'prepare.json')==expected_sha,'external prepare SHA')
    r=io.read(folder/'prepare.json');require(r['schema']==NAME+'.prepare','prepare schema')
    require(io.sha(folder/'protocol.json')==r['protocol_sha256'],'protocol SHA')
    p=io.read(folder/'protocol.json')
    require(p['schema']==NAME+'.protocol' and p['algorithm']==PROTOCOL and p['budgets']=={m:budgets(m) for m in METHODS} and
            p['seeds']==[{m:seed_for(i,m) for m in METHODS} for i in range(2)] and
            p['output_limit_bytes']==LIMIT and p['reserve_bytes']==FLOOR,'unchanged predeclared protocol')
    require(set(p['sources_sha256'])=={str(f) for f in source_files()},'complete source manifest')
    for f,h in {**p['sources_sha256'],**p['inputs']['bindings'],**p['inputs']['model_bindings']}.items():
        require(io.sha(f)==h,'source/input/model drift')
    old=p['inputs'];require(collect(old['direct_prepared'],old['direct_prepare_sha256'],old['task_directory'])==old,'actual task reconstruction')
    return p


def run(folder,expected_sha,output,*,authorize_host_cuda=False):
    require(authorize_host_cuda is True,'explicit MAIN host CUDA authorization')
    p=verify(folder,expected_sha);store=Store(output)
    errors=[];arms=[];model=None;before=after=None;torch=None;source_ok=False
    start=time.monotonic()
    try:
        store.put('launch.json',dict(prepare_sha256=expected_sha,prepared=str(Path(folder).absolute()),algorithm=PROTOCOL))
        import torch as runtime
        torch=runtime
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False
        model=load_compatible_pretrained(p['inputs']['policy'],device=device).eval().requires_grad_(False)
        io.frozen(model);before=io.tensor_hash(model)
        with sdpa_kernel(SDPBackend.MATH),torch.inference_mode():
            for i,item in enumerate(p['inputs']['tasks']):
                for method in METHODS:
                    name=f'root_{i}_{method}'
                    sink=store.sink(name+'.jsonl.gz')
                    result=run_arm(item['task'],method,seed_for(i,method),screen.NativeCuda(model,device,item['task']),sink)
                    row=dict(root_sha256=item['root_sha256'],task_id=item['task_id'],method=method,
                             status=result['status'],resources=result['resources'],error=result['error'])
                    arms.append(row) # Retain partial status BEFORE report I/O.
                    store.put(name+'.json',result)
                    row.update(report_file=name+'.json',report_sha256=io.sha(store.output/(name+'.json')),
                               journal_file=name+'.jsonl.gz')
                    require(result['status']=='sampling_completed_not_admitted','failed arm; no hidden retry or budget reduction')
    except BaseException as exc:errors.append(io.error('execution',exc))
    finally:
        if torch is not None:
            try:torch.cuda.synchronize()
            except BaseException as exc:errors.append(io.error('synchronize',exc))
        if model is not None:
            try:io.frozen(model);after=io.tensor_hash(model)
            except BaseException as exc:errors.append(io.error('final_model',exc))
        try:verify(folder,expected_sha);source_ok=True
        except BaseException as exc:errors.append(io.error('final_source',exc))
        if before is not None and before!=after:errors.append(dict(stage='model_identity',message='model changed or unavailable'))
        errors.extend(store.close())
        for row in arms:
            if 'journal_file' in row:
                try:row['journal_sha256']=io.sha(store.output/row['journal_file'])
                except BaseException as exc:errors.append(io.error('journal_hash',exc))
        totals={r:dict(attempted=sum(a['resources'][phase][r]['attempted'] for a in arms for phase in ('sampling','winner')),
                       completed=sum(a['resources'][phase][r]['completed'] for a in arms for phase in ('sampling','winner')))
                for r in budgets('cem')['sampling']}
        complete=not errors and len(arms)==4 and before is not None and before==after and source_ok
        result=dict(schema=NAME+'.report',status='cuda_completed_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected_sha,arms=arms,resources=totals,errors=errors,
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,inherited=p['inputs']['inherited'],source_purpose=p['inputs']['source_purpose'],
            history_semantics=p['inputs']['history_semantics'],new_image_calls=0,new_physical_steps=0,
            wall_seconds_including_io_and_finalization=time.monotonic()-start,compute_match=PROTOCOL['match'],
            P1_admission=False,model_task_eligible=False,physical_future_labels_read=False,
            missing=['independent threshold authority','fresh confirmation / observed transition history',
                     'direct stall evidence not established here','physical consequences','total compute matching'])
        try:store.put('report.json',result,final=True)
        except BaseException as exc:
            errors.append(io.error('final_report',exc));result['status']='partial_or_failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('direct-prepared','task-directory','prepared','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--direct-prepare-sha');parser.add_argument('--prepare-sha')
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args()
    if a.command=='prepare':
        prepare(a.direct_prepared,a.direct_prepare_sha,a.task_directory,a.output);print(io.sha(a.output/'prepare.json'))
    elif a.command=='verify':print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        r=run(a.prepared,a.prepare_sha,a.output,authorize_host_cuda=a.authorize_host_cuda)
        print(core.canonical(r))
        if r['status']!='cuda_completed_not_admitted':raise SystemExit(2)


if __name__=='__main__':main()
