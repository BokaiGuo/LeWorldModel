import copy
import pytest
from experiments.spectral_reachability import p2_h15_gradient_matched_physical_20261029_v1 as m


def test_plan_binds_exact_delivered_actions_and_identity():
    task={'binding':{'task_id':'t','task_sha256':'s'},'root_context':[1],'goal':[2]}
    native={'task_id':'t','task_sha256':'s','root_context':[1],'goal':[2]}
    mapped={'physical_root_sha256':'r'};root={'root_sha256':'r'}
    slot={'root':0,'status':'complete','report_sha256':'f'*64}
    saved={'result':{'solution':{'logits':[0.0]*30,'actions':[[0.0,0.0]]*15,
        'history':[{}]*4,'evaluation_calls':17,'linearization_calls':4}}}
    plan=m.make_plan(0,task,native,mapped,root,slot,saved)
    assert plan['commands']==saved['result']['solution']['actions']
    bad=copy.deepcopy(task);bad['goal']=[3]
    with pytest.raises(ValueError):m.make_plan(0,bad,native,mapped,root,slot,saved)
    bad=copy.deepcopy(saved);bad['result']['solution']['actions'][0][0]=2
    with pytest.raises(ValueError):m.make_plan(0,task,native,mapped,root,slot,bad)
    bad=copy.deepcopy(saved);bad['result']['solution']['logits'][0]=.1
    with pytest.raises(ValueError):m.make_plan(0,task,native,mapped,root,slot,bad)
    with pytest.raises(ValueError):m.make_plan(1,task,native,mapped,root,slot,saved)


def test_no_physics_before_authorization(tmp_path):
    called=[]
    with pytest.raises(ValueError):m.run(None,None,tmp_path/'x',lambda:called.append(True),authorization='wrong')
    assert not called


def test_fixed_two_root_contract():
    assert m.contract()['caps']['public_step']==50
    assert m.contract()['caps']['physics_step']==508
    assert m.contract()['order']==[[0,'off','gradient_matched'],[1,'off','gradient_matched']]


def test_failed_host_consumer_refuses_before_physical_sources(monkeypatch):
    def failed(*args):
        raise ValueError('upstream partial')
    monkeypatch.setattr(m.host,'consume',failed)
    monkeypatch.setattr(m.physical,'read',lambda *args:pytest.fail('physical source must not be read'))
    with pytest.raises(ValueError,match='upstream partial'):
        m.build('a'*64)


def test_incomplete_host_cannot_authorize_prepare(monkeypatch):
    monkeypatch.setattr(m.host,'consume',lambda *args:dict(status='partial',resources=[]))
    monkeypatch.setattr(m.physical,'read',lambda *args:pytest.fail('physical source must not be read'))
    with pytest.raises(ValueError,match='strict HOST consume'):
        m.build('a'*64)


@pytest.mark.parametrize('early',[False,True])
def test_fixture_complete_saved_consume_and_tamper(tmp_path,monkeypatch,early):
    from test_agent_p2_intersection_physical_20261029_v2 import fixture
    from test_agent_p2_common_resource_physical_20261029_v1 import Env, fake_place
    old,data=fixture()
    p={'plans':[copy.deepcopy(old['plans'][0]),copy.deepcopy(old['plans'][3])],
       'source_roots':old['source_roots'],'source_slots':old['source_slots'],
       'warm_traces':old['warm_traces'],'host_report_sha256':'a'*64}
    for plan in p['plans']:plan['arm']='gradient_matched'
    monkeypatch.setattr(m,'verify',lambda *args:p)
    monkeypatch.setattr(m.physical,'source_material',lambda:data)
    monkeypatch.setattr(m.physical,'place',fake_place)
    monkeypatch.setattr(m.physical,'validate_env',lambda _:None)
    path=tmp_path/(m.NAME+'_fixture')
    report=m.run('/tmp/prepare','0'*64,str(path),lambda:Env(fault='truncate',at=15) if early else Env(),authorization=m.AUTH)
    assert report['status']=='two_root_native_recorded_not_admitted'
    receipt=m.consume(path,m.physical.sha(path/'report.json'),'/tmp/prepare','0'*64)
    assert all(r['own_returned']==(5 if early else 15) for r in receipt['rows'])
    if early:assert all(15 in r['missing_checkpoints'] for r in receipt['rows'])
    png=next(path.glob('*.png'));png.write_bytes(png.read_bytes()+b'tamper')
    with pytest.raises(ValueError):m.consume(path,m.physical.sha(path/'report.json'),'/tmp/prepare','0'*64)
