"""Write-before identity binding capacity preflight; not a production runner."""
from experiments.spectral_reachability import agent_e3_encoder_identity_20261029_v1 as identity

need=identity.need
MAX_EVENT=16385
MAX_RECORDS=9
MAX_BYTES=MAX_EVENT*MAX_RECORDS


def preflight(model,device,model_sha,prepare_sha,store):
    """Measure the actual parameter names/instances without writing a binding.

    The 64-byte synthetic tag has the exact encoded size of the eventual tag.
    No result proves that future whole-job traffic fits its original limits.
    """
    need(getattr(device,'type',None)=='cuda' and type(model_sha) is str and len(model_sha)==64
         and type(prepare_sha) is str and len(prepare_sha)==64,'identity preflight inputs')
    modules={n:getattr(model,n) for n in ('encoder','projector')}
    need(modules['encoder'] is not modules['projector'],'distinct module objects')
    params=identity._inventory(model)
    need(all(p['device']==str(device) for p in params),'exact parameter device')
    detail=dict(schema=identity.NAME+'.binding',tag='0'*64,prepare_sha256=prepare_sha,model_sha256=model_sha,
                model_instance=hex(id(model)),device=str(device),modules={n:hex(id(v)) for n,v in modules.items()},
                parameters=params)
    u=identity.launch.pub.base
    unsigned=dict(index=store.index,previous=store.previous,origin='physical',kind='record',
                  resource='encoder_identity_bind',token=store.index,detail=detail)
    event=dict(unsigned,sha256=u.digest(unsigned));size=len(u.raw(event)+b'\n')
    need(size<=MAX_EVENT,'actual identity bind exceeds per-event cap')
    store.check('physical_events',MAX_BYTES)
    return dict(parameter_count=len(params),binding_event_bytes=size,max_event_bytes=MAX_EVENT,
                max_identity_records=MAX_RECORDS,max_identity_bytes=MAX_BYTES,
                later_capacity_reserved=False,model_or_physics_executed=False)


def load_bound(prepared,torch,device,store,prepare_sha):
    """Original loader -> exact preflight -> bound identical returned object."""
    model,model_sha=identity.launch.host.primitives.native.loader.loaded_model(prepared,torch,device,
                       identity.launch.loader_bridge.LoaderBridge(store),prepare_sha)
    need(model_sha==prepared['expected_model_hash'],'original loaded model hash')
    result=preflight(model,device,model_sha,prepare_sha,store)
    return identity.Binding(model,device,model_sha,prepare_sha,store),result
