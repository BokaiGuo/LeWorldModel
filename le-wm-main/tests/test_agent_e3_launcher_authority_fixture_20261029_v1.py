"""Real saved authority plus synthetic loader/partial lifecycle, never NN/physics."""
from pathlib import Path
import hashlib
import time
import pytest
from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
from experiments.spectral_reachability import agent_e3_continuous_saved_consumer_20261029_v1 as saved


def actual_authority(tmp_path):
    root=launch.host.ROOT;results=root/'results'
    paths=dict(prepare=results/'agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json',
        report=results/'agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01/report.json',
        completion=results/'agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01/completion.json',
        receipt=results/'agent_e3_gd_host_cuda_20261029_v1_consume_cell00_MAIN01.json')
    resolver=results/'agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json'
    if not all(p.is_file() for p in [*paths.values(),resolver]):pytest.skip('actual MAIN saved inputs unavailable')
    pins=dict(prepare=launch.CONTROL5_PREPARE_SHA,
        report='e6d465652eb993e31d19ce11e5e0f494cb18bf54e65d4d9f0cac983a733e7385',
        completion='ca309a42a5bf7ab8344b8c1672b66829d3b8bc3b4a82fac3650f04ce5d360b1b',
        receipt='9b5b8751e38138b6f127d05de331e54bd355075501cdaf6914fb1fb3e63b945a')
    modules=(launch,launch.episode,launch.loader_bridge,launch.pub,launch.pub.base,launch.host,
             launch.reset_identity,launch.module_census)
    extra={str(Path(m.__file__).resolve()):launch.host.sha(m.__file__) for m in modules}
    manifest=dict(schema=launch.NAME+'.authority',cell=0,
        control5={k:dict(path=str(v),sha256=pins[k]) for k,v in paths.items()},
        resolver=dict(path=str(resolver),sha256=launch.RESOLVER_SHA),additional_sources=extra)
    data=launch.host.raw(manifest);path=tmp_path/'authority.json';path.write_bytes(data)
    return path,hashlib.sha256(data).hexdigest(),data


def synthetic_loader(binding,store):
    p=binding['prepared'];loader=launch.host.primitives.native.loader
    files=[]
    for name in ('config.json','weights.pt'):
        target=str(Path(p['policy'])/name);h=p['bindings_sha256'][target]
        files.append(dict(path=target,sha256=h,bytes=1,hash_buffer_sha256=h,deserialize_buffer_sha256=h,same_buffer=True))
    bridge=launch.loader_bridge.LoaderBridge(store)
    for f in files:
        bridge.event(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']))
        bridge.event(dict(kind='consumed_buffer_frozen',receipt=f))
    receipt=dict(schema=loader.NAME+'.loader_receipt',prepare_sha256=binding['prepare_sha'],
        adapter_source_sha256=loader.sha(loader.__file__),
        legacy_loader_source_sha256=loader.native.SOURCES['experiments/checkpoint_compat.py'],
        files=files,device='cuda:0',remapped_keys=[],missing=[],unexpected=[],
        tensor_hash_before=p['expected_model_hash'],frozen_eval=True)
    bridge.put('loader.json',receipt)
    launch._record(store,'fixture_only',{'synthetic_loader':True,'no_nn_or_physics':True})


@pytest.mark.parametrize('mode',['loader_failure','nested_failure','fake_cleanup'])
def test_actual_saved_authority_synthetic_loader_failure_chain(tmp_path,mode):
    path,manifest_sha,data=actual_authority(tmp_path);called=[]
    def perform(binding,store):
        called.append('synthetic fixture loader: no model loaded or deserialized')
        synthetic_loader(binding,store)
        if mode=='fake_cleanup':
            launch._record(store,'native_cleanup_intent',{'episode_id':'cell00'})
            launch._record(store,'native_cleanup_return',{'episode_id':'cell00'})
        if mode=='nested_failure':
            from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as native_meter
            from test_agent_e3_native_meter_20261029_v1 import Base
            meter=native_meter.Meter(store,'cell00')
            cls=native_meter.native.metered_class(Base,meter)
            env=native_meter.Env(meter.call('constructor',lambda:cls(True)),meter)
            env.reset();meter.region='warmup'
            try:
                env.step(1)
            finally:
                launch._record(store,'native_cleanup_intent',{'episode_id':'cell00'})
                env.close()
                launch._record(store,'native_cleanup_return',{'episode_id':'cell00'})
        raise RuntimeError('synthetic failure before environment')
    result=launch._run_lifecycle(tmp_path/'job',time.monotonic(),
                                 lambda:launch.read_authority(str(path),manifest_sha),perform)
    assert called and not result['complete']
    seal=result['publication']
    if mode=='fake_cleanup':
        with pytest.raises(ValueError,match='requires metered close'):
            saved.consume(tmp_path/'job',seal['report_sha256'],seal['completion_sha256'],str(path),manifest_sha)
        return
    audit=saved.consume(tmp_path/'job',seal['report_sha256'],seal['completion_sha256'],str(path),manifest_sha)
    assert audit['status']=='partial' and not audit['full_saved_strict']
    if mode=='nested_failure':
        assert audit['cleanup_return_observed'] and audit['error']['message']=='physics payload'
        counts=audit['meter']['counts']
        assert counts['public_step']==counts['physics_step']==dict(attempted=1,completed=0,failed=1)
        assert counts['close']==dict(attempted=1,completed=1,failed=0)
        assert not audit['meter']['ledger_complete']
    events=[launch.pub.base.decode(line) for line in (tmp_path/'job/events.jsonl').read_bytes().splitlines()]
    binding=launch.read_authority(str(path),manifest_sha)
    launch.loader_bridge.verify_events(events,binding['prepared'],binding['prepare_sha'],launch.host.primitives.native.loader)
    bad=bytearray(data);bad[-2]^=1;path.write_bytes(bad)
    with pytest.raises(ValueError,match='SHA'):launch.read_authority(str(path),manifest_sha)
