from copy import deepcopy

import pytest
from experiments.spectral_reachability import agent_p3_future_cp10_native_transcript_20261029_v1 as m
import test_agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as fixture


@pytest.fixture
def transcript(tmp_path,monkeypatch):
    p,s,r,_=fixture.completed_disk(tmp_path,monkeypatch)
    events=[m.native.controls.native.decode((s.path/f'event_{i:04}.json').read_bytes()) for i in range(r['event_count'])]
    load=lambda e:m.native.bridge.rgb.load_array(e,lambda n:(s.path/n).read_bytes())
    return p,r,events,load


def test_real_temp_store_full_native_to_own_projection(transcript):
    p,r,e,load=transcript;v=m.validate_native(p,r,e,load)
    assert v['native_module_count']==10 and v['event_count']==52
    assert len(v['projection']['signals'])==7 and len(v['projection']['endpoints'])==5
    assert not v['full_saved_consumer'] and not v['execution_authorized']
    assert all(x['identity']['namespace']==m.projection.NAME for x in v['projection']['endpoints'])


@pytest.mark.parametrize('kind',['event','guard','missing','count','scope','latent'])
def test_transcript_mutations_fail(transcript,kind):
    p,r,e,load=transcript;p=deepcopy(p);r=deepcopy(r);e=deepcopy(e)
    if kind=='event':e[1]['kind']='module_return'
    elif kind=='guard':r['guards'][0]['completed']=False
    elif kind=='missing':r['rows'].pop()
    elif kind=='count':r['counts']['encoder']['completed']-=1
    elif kind=='scope':r['rows'][0]['modules'][0]['scope']['row']=1
    elif kind=='latent':
        orig=load
        def load(desc):
            x=orig(desc).copy()
            if desc==r['rows'][0]['latent']:x[0]+=1
            return x
    with pytest.raises((ValueError,KeyError,AssertionError)):m.validate_native(p,r,e,load)
