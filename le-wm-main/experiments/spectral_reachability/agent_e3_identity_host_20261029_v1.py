"""Candidate E3 same-environment HOST with identity journal; not full strict."""
import base64
import argparse
import hashlib
import os
from copy import deepcopy
from pathlib import Path
import time
import traceback

from experiments.spectral_reachability import agent_e3_integrated_host_saved_20261029_v1 as prior
from experiments.spectral_reachability import agent_e3_encoder_identity_20261029_v1 as identity
from experiments.spectral_reachability import agent_e3_identity_capacity_20261029_v1 as capacity

NAME='agent_e3_identity_host_20261029_v1'
launch=prior.launch
need=launch.need
MAX_IDENTITY_RECORDS=9
MAX_IDENTITY_BYTES=MAX_IDENTITY_RECORDS*16385


def authority(path,sha):
    binding=prior._authority(path,sha)
    sources=dict(binding['prepared']['bindings_sha256'])
    launch.host.source.merge(sources,binding['additional_sources'])
    required=[str(Path(m.__file__).resolve()) for m in (identity,capacity)]+[str(Path(__file__).resolve())]
    launch.episode.old.guard.validate_sources(sources,required)
    return binding


def execute_episode(env,args,body,first5,binding,torch,device,store,meter,row):
    """Original same-env dispatch with bound encoder; explicit cleanup."""
    live=launch.reset_identity.old.live
    bridge=live.capture.CaptureStore(store);kw=dict(args)
    kw.update(row=row,meter=meter,store=bridge,packet_ops=live.region)
    def encode(actual_env,observation,boundary):
        need(actual_env is env,'same environment for encoding')
        frame=bridge.descriptor(observation);previous=meter.region;meter.region='encode'
        try:
            check=lambda:store.check('encoding_arrays',0)
            latent,_=binding.encode(torch,frame,boundary,check)
            rgb=live.encoding.u.read_bytes(store.output/frame['file'],live.capture.pub.PNG_CAP)
            return dict(latent=latent,rgb_bytes=rgb,input_sha=frame['sha256'],episode_id=row['episode_id'],boundary=boundary)
        finally:meter.region=previous
    def plan(task):
        previous=meter.region;meter.region='compile'
        try:
            adapter=live.nodes.NodeBridge(store,task['feedback_boundary']);record=dict(forwards=[],AD=[],result=None)
            backend=live.nodes.bridge.host.HostBackend(binding.model,device,torch,adapter,task,record)
            with backend.observe():record['result']=live.nodes.node.solve(task,backend,adapter.event)
            adapter.finish(record);return record['result']
        finally:meter.region=previous
    primary=None
    try:
        setup=live.setup.setup(env,**kw)
        if setup is not None:
            live.dispatcher.dispatch(env,body,setup,first5,encode,plan,bridge,meter,row,packet_fn=live.region.packet)
        return row
    except BaseException as exc:
        primary=exc;detail=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        row['status']='failed';row['error']=detail
        try:bridge.emit('episode_failure',error=deepcopy(detail))
        except BaseException as secondary:exc.add_note('failure event: '+repr(secondary))
        raise
    finally:
        errors=[];closed=False
        try:bridge.emit('cleanup_intent',episode_id=row['episode_id'])
        except BaseException as exc:errors.append(exc)
        try:env.close();closed=True
        except BaseException as exc:errors.append(exc)
        try:bridge.emit('cleanup_return' if closed else 'cleanup_failure',episode_id=row['episode_id'])
        except BaseException as exc:errors.append(exc)
        if errors:
            row['cleanup_errors']=[dict(type=type(e).__name__,message=str(e)) for e in errors];row['status']='failed'
        try:live.terminal.write(store,row)
        except BaseException as exc:errors.append(exc)
        if errors:
            if primary is None:raise errors[0]
            for exc in errors:primary.add_note('cleanup or terminal persistence failure: '+repr(exc))


def guarded_episode(binding,loaded,torch,device,store,row):
    reset=launch.reset_identity;old=reset.old;prepared=binding['prepared']
    sources=dict(prepared['bindings_sha256']);launch.host.source.merge(sources,binding['additional_sources'])
    modules=(old,old.authority,old.guard,old.live,old.setup_authority,old.live.setup,old.live.region,
             old.live.capture,old.live.dispatcher,old.live.encoding,old.live.nodes,old.native,reset.native_meter,reset,identity)
    required=[str(Path(m.__file__).resolve()) for m in modules]+[str(Path(__file__).resolve())]
    need(all(p in sources for p in required),'all identity episode sources pinned')
    meter=reset.native_meter.Meter(store,row['episode_id'])
    def record(stage,value):
        data=launch.host.raw(value);need(0<len(data)<=16*1024**2,'bounded guard snapshot')
        sha=hashlib.sha256(data).hexdigest();parts=[data[i:i+6000] for i in range(0,len(data),6000)]
        for i,part in enumerate(parts):
            reset.log(store,'record','guard_snapshot_chunk',store.index,dict(stage=stage,index=i,total=len(parts),
                      sha256=sha,data=base64.b64encode(part).decode('ascii')))
        reset.log(store,'record',stage,store.index,dict(sha256=sha,total=len(parts),bytes=len(data)))
    def load():
        result=old.authority.load(*binding['args'])
        actual=old.setup_authority.resolve(binding['resolver']['path'],binding['resolver']['sha256'],result['cell'],result['body'])
        launch.host.source.merge(sources,actual.pop('source_bindings'))
        actual['ops']=reset.Ops(old.native,store,row['episode_id']);result['setup_kwargs']=actual;return result
    def snap():return old.guard.snapshot(loaded.model,torch,device,sources,required)
    def validate(a,b):return old.guard.validate_pair(a,b,prepared['expected_model_hash'])
    def create():return reset.TracedEnv(reset.native_meter.native_factory(meter),store,row['episode_id'])
    def execute(env,task):return execute_episode(env,task['setup_kwargs'],task['body'],task['first5'],loaded,torch,device,store,meter,row)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
        return old._sequence(load,snap,create,execute,validate,record)


def host_perform(binding,store):
    launch._record(store,'identity_host_started',dict(schema=NAME,max_records=MAX_IDENTITY_RECORDS,
                                                      max_event_bytes=MAX_IDENTITY_BYTES))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from experiments.gpu_preflight import require_cuda_allocation
    gpu=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    launch._record(store,'launcher_gpu_preflight',{'preflight':gpu})
    store.check('physical_events',MAX_IDENTITY_BYTES)
    loaded,capacity_proof=capacity.load_bound(binding['prepared'],torch,device,store,binding['prepare_sha'])
    launch._record(store,'identity_capacity_preflight',capacity_proof)
    row=dict(episode_id='cell%02d'%binding['cell'],warmup_returned=0,control_returned=0,
             executed_raw=[],PNG=0,checkpoints=[],status='started')
    sources=dict(binding['prepared']['bindings_sha256']);launch.host.source.merge(sources,binding['additional_sources'])
    launch.module_census.record(store,'before',launch.module_census.snapshot(sources));primary=None
    try:result=guarded_episode(binding,loaded,torch,device,store,row)
    except BaseException as exc:primary=exc;raise
    finally:
        try:launch.module_census.record(store,'after',launch.module_census.snapshot(sources))
        except BaseException as exc:
            if primary is None:raise
            primary.add_note('post-cleanup module census: '+repr(exc))
    need(launch.host.sha(binding['manifest_path'])==binding['manifest_sha'],'authority unchanged')
    return result


def run(authority_path,authority_sha,output,authorization):
    need(authorization==launch.AUTH,'explicit HOST CUDA authorization')
    started=time.monotonic()
    return launch._run_lifecycle(output,started,lambda:authority(authority_path,authority_sha),host_perform)


def verify_capacity(bound,proof,journal):
    need(bound['index']<proof['index'],'capacity proof after binding')
    lines=journal.splitlines(keepends=True);need(0<=bound['index']<len(lines),'binding journal index')
    line=lines[bound['index']];detail=proof['detail']
    need(detail==dict(parameter_count=len(bound['detail']['parameters']),binding_event_bytes=len(line),
         max_event_bytes=capacity.MAX_EVENT,max_identity_records=capacity.MAX_RECORDS,
         max_identity_bytes=capacity.MAX_BYTES,later_capacity_reserved=False,model_or_physics_executed=False)
         and len(line)<=capacity.MAX_EVENT,'exact bound event capacity proof')
    return True


def inspect(output,report_sha,completion_sha,authority_path,authority_sha):
    base=prior.inspect(output,report_sha,completion_sha,authority_path,authority_sha)
    binding=authority(authority_path,authority_sha)
    report=launch.pub.base.read_bytes(Path(output)/'report.json',launch.pub.FINAL_CAP)
    journal=launch.pub.base.read_bytes(Path(output)/'events.jsonl',launch.pub.base.CAP)
    need(launch.pub.base.fsha(report)==report_sha and
         launch.pub.base.fsha(journal)==launch.pub.base.decode(report)['journal_sha256'],'stable identity host journal')
    events=[launch.pub.base.decode(line) for line in journal.splitlines()]
    starts=[e for e in events if e['origin']=='physical' and e['resource']=='identity_host_started']
    need(len(starts)==1 and starts[0]['detail']==dict(schema=NAME,max_records=MAX_IDENTITY_RECORDS,
         max_event_bytes=MAX_IDENTITY_BYTES),'new identity HOST marker; reject old logs')
    bound=[e for e in events if e['origin']=='physical' and e['resource']=='encoder_identity_bind']
    loaders=[e for e in events if e['origin']=='physical' and e['resource']=='loader_receipt']
    need((not loaders or starts[0]['index']<loaders[0]['index']) and
         (not bound or (len(bound)==1 and loaders and loaders[0]['index']<bound[0]['index'])),
         'identity host loader and binding chronology')
    if not bound:
        need(base['status']=='partial','missing identity binding only acceptable as partial')
        return dict(schema=NAME+'.inspect',base=base,identity=None,full_saved_strict=False,
                    scientific_admission=False,missing=['identity binding not reached']+base['missing'])
    proofs=[e for e in events if e['origin']=='physical' and e['resource']=='identity_capacity_preflight']
    if proofs:
        need(len(proofs)==1,'unique capacity proof');verify_capacity(bound[0],proofs[0],journal)
    else:need(base['status']=='partial','successful output requires capacity proof')
    sources=dict(binding['prepared']['bindings_sha256']);launch.host.source.merge(sources,binding['additional_sources'])
    recorded=identity.consume(output,report_sha,completion_sha,binding['prepared'],binding['prepare_sha'],sources)
    physical_boundaries=[x['boundary'] for x in base['encoding_ledger']['boundaries']]
    identity_boundaries=[x['boundary'] for x in recorded['encodes']]
    if recorded['pending'] is not None:identity_boundaries.append(recorded['pending']['boundary'])
    if base['status']!='partial':
        need(identity_boundaries==physical_boundaries and recorded['pending'] is None and
             all(x['outcome']=='return' for x in recorded['encodes']), 'every successful encoder bound')
    else:
        need(identity_boundaries==physical_boundaries[:len(identity_boundaries)], 'partial identity boundary prefix')
    return dict(schema=NAME+'.inspect',base=base,identity=recorded,capacity_preflight_verified=bool(proofs),full_saved_strict=False,
                scientific_admission=False,missing=base['missing'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','inspect'):
        p=sub.add_parser(name)
        for option in ('authority','authority-sha256','output'):p.add_argument('--'+option,required=True)
        if name=='run':p.add_argument('--authorization',required=True)
        else:
            p.add_argument('--report-sha256',required=True);p.add_argument('--completion-sha256',required=True)
    a=parser.parse_args()
    if a.command=='run':result=run(a.authority,a.authority_sha256,a.output,a.authorization)
    else:result=inspect(a.output,a.report_sha256,a.completion_sha256,a.authority,a.authority_sha256)
    print(launch.host.raw(result).decode())


if __name__=='__main__':main()
