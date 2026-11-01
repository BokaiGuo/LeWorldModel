import copy
import pytest
from experiments.spectral_reachability import p2_h15_physical_adapter_20261029_v1 as m


@pytest.fixture(scope='module')
def prepared(): return m.build()


def test_actual_two_root_plans_bound_to_saved_winners(prepared):
    assert len(prepared['plans'])==2
    assert [p['root_slot'] for p in prepared['plans']]==[0,1]
    assert all(len(p['commands'])==15 for p in prepared['plans'])
    assert prepared['contract']['caps']['public_step']==50
    assert prepared['contract']['caps']['physics_step']==508
    assert prepared['physical_release'] is False


def test_mutated_action_and_wrong_root_rejected(prepared):
    altered=copy.deepcopy(prepared);altered['plans'][0]['commands'][0][0]=0.9
    with pytest.raises(ValueError):m.validate(altered,prepared)
    altered=copy.deepcopy(prepared);altered['plans'].reverse()
    with pytest.raises(ValueError):m.validate(altered,prepared)


def test_no_environment_without_authority(tmp_path):
    calls=[]
    with pytest.raises(ValueError):
        m.run(None,None,tmp_path/'run',lambda:calls.append(1),authorization='wrong')
    assert not calls


def test_fixture_native_stop_is_not_padded(tmp_path,prepared,monkeypatch):
    from contextlib import nullcontext
    p=copy.deepcopy(prepared)
    data={'roots':[{'slot':p['source_slots'][i],'warm_trace':p['warm_traces'][i]} for i in (0,1)],'source_plan':{}}
    class Env:
        def __init__(self):
            self.unwrapped=self;self.render_action=False;self.render_size=224;self.render_mode='rgb_array'
        def close(self):pass
    monkeypatch.setattr(m,'verify',lambda *args:p)
    monkeypatch.setattr(m.physical,'source_material',lambda:data)
    monkeypatch.setattr(m.physical.native.allmode,'root_metadata',lambda _:p['source_roots'])
    monkeypatch.setattr(m.physical,'validate_env',lambda _:None)
    monkeypatch.setattr(m.physical,'native_hooks',lambda *args:nullcontext())
    monkeypatch.setattr(m.physical,'episode',lambda *args:(_ for _ in ()).throw(m.physical.native.allmode.NativeStop('fixture early terminal')))
    output=tmp_path/(m.NAME+'_fixture')
    result=m.run('/tmp/prepare','0'*64,str(output),Env,authorization=m.AUTH)
    assert result['status']=='two_root_native_recorded_not_admitted'
    assert [r['status'] for r in result['rows']]==['native_stopped','native_stopped']
    assert all(r['own_returned']==0 for r in result['rows'])


@pytest.mark.parametrize('early',[False,True])
def test_fixture_complete_and_saved_audit(tmp_path,monkeypatch,early):
    from test_agent_p2_intersection_physical_20261029_v2 import fixture
    from test_agent_p2_common_resource_physical_20261029_v1 import Env, fake_place
    old,data=fixture()
    p={'plans':[copy.deepcopy(old['plans'][0]),copy.deepcopy(old['plans'][3])],
       'source_roots':old['source_roots'],'source_slots':old['source_slots'],'warm_traces':old['warm_traces']}
    for plan in p['plans']:plan['arm']='h15_sra'
    monkeypatch.setattr(m,'verify',lambda *args:p)
    monkeypatch.setattr(m.physical,'source_material',lambda:data)
    monkeypatch.setattr(m.physical,'place',fake_place)
    monkeypatch.setattr(m.physical,'validate_env',lambda _:None)
    path=tmp_path/(m.NAME+'_complete')
    report=m.run('/tmp/prepare','0'*64,str(path),lambda:Env(fault='truncate',at=15) if early else Env(),authorization=m.AUTH)
    assert report['status']=='two_root_native_recorded_not_admitted'
    checked=m.consume(path,m.physical.sha(path/'report.json'),'/tmp/prepare','0'*64)
    assert checked['fixed_denominator']==2 and all(r['own_returned']==(5 if early else 15) for r in checked['rows'])
    if early:assert all(r['status']=='native_stopped' and 15 in r['missing_checkpoints'] for r in checked['rows'])
    png=next(path.glob('*.png'))
    png.write_bytes(png.read_bytes()+b'tamper')
    with pytest.raises(ValueError):m.consume(path,m.physical.sha(path/'report.json'),'/tmp/prepare','0'*64)
