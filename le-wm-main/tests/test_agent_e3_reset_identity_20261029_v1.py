from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as m
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub


class Polygon:
    def __init__(self,xy):self.xy=xy
    def get_vertices(self):return [SimpleNamespace(x=x,y=y) for x,y in self.xy]


class Env:
    def __init__(self,fail=False):self.unwrapped=self;self.closed=False;self.fail=fail
    def reset(self,*,seed,options=None):
        if self.fail:raise RuntimeError('reset failure')
        goal=np.array([1,2,150,160,.4,0,0],dtype=np.float64)
        if options is None:return {'state':goal.copy()},{'goal_state':goal.copy()}
        return {'state':options['state'].copy()},{'goal_state':options['goal_state'].copy()}
    def close(self):self.closed=True


def fixture(tmp_path,fail=False):
    store=pub.Store(tmp_path/'job');raw=Env(fail);env=m.TracedEnv(raw,store,'e')
    return store,raw,env,m.Source(m.old.native.source,store,'e')


def test_actual_original_sample_abi_disk_and_tamper(tmp_path):
    store,raw,env,source=fixture(tmp_path)
    recipe=dict(position_jitter=5.,angle_offset_range=[-.2,.2],edge_fraction_range=[.2,.8],pusher_surface_offset=2.)
    plan=dict(recipe=recipe);slot=dict(root_seed=11);shapes=[Polygon([(0.,0.),(3.,0.),(3.,2.),(0.,2.)])]
    _,info=env.reset(seed=23);goal=m.old.native.source.vector(info['goal_state'])
    placed=source.sample_state(goal,shapes,slot,0,0,plan)
    obs,_=env.reset(seed=0,options=dict(state=placed.copy(),goal_state=goal.copy()))
    assert np.array_equal(obs['state'],placed)
    env.close();assert raw.closed
    seal=store.publish('completed');assert pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    setup=dict(ref=dict(slot=dict(reset_seed=23,root_seed=11)),source_plan=plan)
    result=m.verify_events(events,'e',setup)
    assert result['reset_completed']==2 and result['sample_completed']==1 and result['reset_state']==m.vector_bytes(placed)
    bad=deepcopy(events);record=next(e for e in bad if e['resource']=='reset_identity' and e['kind']=='intent' and e['detail']['ordinal']==1)
    record['detail']['options']['state']='00'*56
    with pytest.raises(ValueError):m.verify_events(bad,'e',setup)
    bad=deepcopy(events);record=next(e for e in bad if e['resource']=='sample_identity' and e['kind']=='return')
    record['detail']['selected']['index']=1
    with pytest.raises(ValueError):m.verify_events(bad,'e',setup)


def test_failed_reset_is_recorded_and_cleanup_still_runs(tmp_path):
    store,raw,env,_=fixture(tmp_path,True)
    try:
        with pytest.raises(RuntimeError,match='reset failure'):env.reset(seed=23)
    finally:env.close()
    assert raw.closed
    seal=store.publish('partial');assert not pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    assert [e['kind'] for e in events]==['intent','failure']
    assert events[-1]['detail']['error']['message']=='reset failure'
    trace=m.inspect_partial(events)
    assert trace['calls'][0]['status']=='failure' and trace['calls'][0]['result']['error']['message']=='reset failure'
    with pytest.raises(ValueError):m.verify_events(events,'e',{})


def test_real_setup_abi_traced_early_stop_and_cleanup(tmp_path):
    from test_agent_e3_feedback_setup_20261029_v2 import case
    from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v2 as setup
    from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as capture
    raw,kw,_,row=case(stop=1)
    raw.block.shapes=[Polygon([(0.,0.),(4.,0.),(4.,3.),(0.,3.)])]
    closed=[];raw.close=lambda:closed.append(True)
    original_ops=kw['ops'];original_ops.source.sample_state=m.old.native.source.sample_state
    original_capture=original_ops.capture
    original_ops.capture=lambda env,*args:original_capture(env.env,*args)
    plan=dict(recipe=dict(position_jitter=5.,angle_offset_range=[-.2,.2],
                          edge_fraction_range=[.2,.8],pusher_surface_offset=2.))
    store=pub.Store(tmp_path/'job');env=m.TracedEnv(raw,store,row['episode_id'])
    kw.update(source_plan=plan,ops=m.Ops(original_ops,store,row['episode_id']),store=capture.CaptureStore(store))
    try:
        assert setup.setup(env,**kw) is None
    finally:env.close()
    assert closed==[True] and row['status']=='native_stopped_warmup'
    seal=store.publish('completed');assert pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    assert m.verify_events(events,row['episode_id'],dict(ref=kw['ref'],source_plan=plan))['reset_completed']==2


def test_sample_exception_preserved_no_retry(tmp_path):
    store,raw,env,source=fixture(tmp_path)
    _,info=env.reset(seed=23)
    plan=dict(recipe=dict(position_jitter=1.,angle_offset_range=[0.,0.],
                          edge_fraction_range=[.2,.8],pusher_surface_offset=2.))
    try:
        with pytest.raises(ValueError):
            source.sample_state(info['goal_state'],[Polygon([(0.,0.),(0.,0.),(0.,0.)])],dict(root_seed=11),0,0,plan)
    finally:env.close()
    assert raw.closed
    seal=store.publish('partial');assert not pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    assert [e['kind'] for e in events if e['resource']=='sample_identity']==['intent','failure']
    trace=m.inspect_partial(events)
    assert [call['status'] for call in trace['calls']]==['return','failure'] and trace['pending'] is None


def test_full_geometry_matches_pinned_source_and_rejects_splice(tmp_path):
    path=Path(__file__).resolve().parents[1]/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    polygons=m.official_polygons(path,m.ENV_SHA)
    store,raw,env,source=fixture(tmp_path)
    _,info=env.reset(seed=23)
    recipe=dict(position_jitter=5.,angle_offset_range=[-.2,.2],edge_fraction_range=[.2,.8],pusher_surface_offset=2.)
    source.sample_state(info['goal_state'],[Polygon(p) for p in polygons],dict(root_seed=11),0,0,dict(recipe=recipe))
    env.close();seal=store.publish('completed')
    assert pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    result=m.verify_geometry(events,{str(path):m.ENV_SHA})
    assert result['polygon_count']==2 and result['vertex_read_attempted']==result['vertex_read_returned']==2
    bad=deepcopy(events);end=next(e for e in bad if e['resource']=='sample_identity' and e['kind']=='return')
    end['detail']['polygons'][1]['bits']='00'*64
    with pytest.raises(ValueError):m.verify_geometry(bad,{str(path):m.ENV_SHA})


def test_extra_geometry_read_failure_accounted_and_cleanup(tmp_path):
    class Broken:
        def get_vertices(self):raise RuntimeError('vertices unavailable')
    store,raw,env,source=fixture(tmp_path)
    _,info=env.reset(seed=23)
    plan=dict(recipe=dict(position_jitter=1.,angle_offset_range=[0.,0.],
                          edge_fraction_range=[.2,.8],pusher_surface_offset=2.))
    try:
        with pytest.raises(RuntimeError,match='vertices unavailable'):
            source.sample_state(info['goal_state'],[Broken()],dict(root_seed=11),0,0,plan)
    finally:env.close()
    assert raw.closed
    seal=store.publish('partial');assert not pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    reads=[e for e in events if e['resource']=='geometry_vertex_read']
    assert [e['kind'] for e in reads]==['intent','failure'] and reads[-1]['detail']['error']['message']=='vertices unavailable'
