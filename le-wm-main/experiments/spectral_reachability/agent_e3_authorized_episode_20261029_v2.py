"""E3 guarded lifecycle with own 1800s native meter; not an experiment strict."""
import base64
import hashlib
from pathlib import Path
from experiments.spectral_reachability import agent_e3_authorized_episode_20261029_v1 as old
from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as native_meter

AUTH=old.AUTH


def run(authority_args,resolver_path,resolver_sha,model,torch,device,store,row,additional_sources,authorization):
    """Caller supplies a verified-start Store and authorized loaded model.

    This component does not implement prepare/CLI or full semantic consume.
    The Store clock MUST have started before authority verification.
    """
    old.need(authorization==AUTH and type(authority_args) is tuple and len(authority_args)==9
             and type(additional_sources) is dict and type(row) is dict
             and type(row.get('episode_id')) is str and bool(row['episode_id']),'authorized fixed input')
    prepared=old.authority.host.read(authority_args[0],authority_args[1]);bindings=dict(prepared['bindings_sha256'])
    old.need(prepared['actual_resolver']==dict(path=resolver_path,sha256=resolver_sha),'exact resolver authority')
    old.authority.host.source.merge(bindings,additional_sources)
    modules=(old,old.authority,old.guard,old.live,old.setup_authority,old.live.setup,old.live.region,
             old.live.capture,old.live.dispatcher,old.live.encoding,old.live.nodes,old.native,native_meter)
    required=[str(Path(m.__file__).resolve()) for m in modules]+[str(Path(__file__).resolve())]
    old.need(all(p in bindings for p in required),'all directly used modules source pinned')
    meter=native_meter.Meter(store,row['episode_id'])
    def record(stage,value):
        data=old.authority.host.raw(value);old.need(0<len(data)<=16*1024**2,'bounded guard snapshot')
        sha=hashlib.sha256(data).hexdigest();parts=[data[i:i+6000] for i in range(0,len(data),6000)]
        for i,part in enumerate(parts):
            store.event('physical','record','guard_snapshot_chunk',store.index,
                dict(stage=stage,index=i,total=len(parts),sha256=sha,data=base64.b64encode(part).decode('ascii')))
        store.event('physical','record',stage,store.index,dict(sha256=sha,total=len(parts),bytes=len(data)))
    def load():
        result=old.authority.load(*authority_args)
        actual=old.setup_authority.resolve(resolver_path,resolver_sha,result['cell'],result['body'])
        old.authority.host.source.merge(bindings,actual.pop('source_bindings'))
        result['setup_kwargs']=actual;return result
    def snap():return old.guard.snapshot(model,torch,device,bindings,required)
    def validate(a,b):return old.guard.validate_pair(a,b,prepared['expected_model_hash'])
    def execute(env,binding):
        return old.live.execute(env,binding['setup_kwargs'],binding['body'],binding['first5'],model,torch,device,store,meter,row)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
        return old._sequence(load,snap,lambda:native_meter.native_factory(meter),execute,validate,record)
