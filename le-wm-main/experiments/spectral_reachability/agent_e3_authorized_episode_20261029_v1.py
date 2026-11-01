"""Authority/model guarded composition of the E3 episode component.

Not a production launcher: setup input provenance, full source closure,
publication and independent physical/encoder semantics remain to be joined.
"""
import hashlib
import base64
import traceback
from pathlib import Path
from experiments.spectral_reachability import agent_e3_control5_authority_join_20261029_v1 as authority
from experiments.spectral_reachability import agent_e3_model_source_guard_20261029_v1 as guard
from experiments.spectral_reachability import agent_e3_live_episode_join_20261029_v1 as live
from experiments.spectral_reachability import agent_e3_setup_authority_20261029_v1 as setup_authority
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as native

AUTH='MAIN_E3_KNOWN16_CONTINUOUS_HOSTCUDA_ONLY'
need=guard.need


def _sequence(load_authority, take_snapshot, create_env, execute, validate, record):
    """Ordering primitive; injectable only as an explicit fixture seam."""
    binding=load_authority()
    before=take_snapshot();validate(before,before)
    record('guard_before',before)
    primary=None
    try:
        env=create_env()
        return execute(env,binding)
    except BaseException as exc:
        primary=exc
        raise
    finally:
        try:
            after=take_snapshot();validate(before,after);record('guard_after',after)
        except BaseException as exc:
            try:record('guard_failure',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            except BaseException as secondary:exc.add_note('guard failure record: '+repr(secondary))
            if primary is None:raise
            primary.add_note('post-cleanup model/source guard failure: '+repr(exc))


def run(authority_args, resolver_path, resolver_sha, model, torch, device, store, meter, row, additional_sources, authorization):
    """Guarded call; no prepare/build, receipt or full semantic certification."""
    need(authorization==AUTH and type(authority_args) is tuple and len(authority_args)==9,
         'explicit authorization and control5 pins')
    need(type(additional_sources) is dict,'typed source bindings')
    prepared=authority.host.read(authority_args[0],authority_args[1]);bindings=dict(prepared['bindings_sha256'])
    need(prepared['actual_resolver']==dict(path=resolver_path,sha256=resolver_sha),
         'control5 preparation binds exact resolver authority')
    authority.host.source.merge(bindings,additional_sources)
    modules=(authority,guard,live,setup_authority,live.setup,live.region,live.capture,live.dispatcher,live.encoding,live.nodes,native)
    required=[str(Path(m.__file__).resolve()) for m in modules]
    need(all(p in bindings for p in required),'all directly used modules source pinned')
    need(type(row) is dict and type(row.get('episode_id')) is str and row['episode_id'],'episode identity')
    def record(kind,value):
        data=authority.host.raw(value);need(0<len(data)<=16*1024**2,'bounded complete guard snapshot')
        sha=hashlib.sha256(data).hexdigest();parts=[data[i:i+6000] for i in range(0,len(data),6000)]
        for i,part in enumerate(parts):
            store.event('physical','record','guard_snapshot_chunk',store.index,
                        dict(stage=kind,index=i,total=len(parts),sha256=sha,data=base64.b64encode(part).decode('ascii')))
        store.event('physical','record',kind,store.index,dict(sha256=sha,total=len(parts),bytes=len(data)))
    def load():
        result=authority.load(*authority_args)
        actual=setup_authority.resolve(resolver_path,resolver_sha,result['cell'],result['body'])
        authority.host.source.merge(bindings,actual.pop('source_bindings'))
        result['setup_kwargs']=actual
        return result
    def snap():return guard.snapshot(model,torch,device,bindings,required)
    def validate(a,b):return guard.validate_pair(a,b,prepared['expected_model_hash'])
    def execute(env,binding):
        return live.execute(env,binding['setup_kwargs'],binding['body'],binding['first5'],model,torch,device,store,meter,row)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
        return _sequence(load,snap,lambda:native.native_factory(meter),execute,validate,record)
