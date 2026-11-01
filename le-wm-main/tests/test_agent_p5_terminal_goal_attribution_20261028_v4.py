"""v4 order regression, including direct real saved prepare."""
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v4 as m

ROOT=m.ROOT
REAL=(ROOT/'results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json',ROOT/'results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json',ROOT/'results/agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json',ROOT/'results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json',ROOT/'results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json',ROOT/'results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json',ROOT/'results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json',ROOT/'results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json',ROOT/'results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json')

def test_direct_real_nine_file_prepare_and_sha():
    assert all(p.is_file() for p in REAL)
    value=m.prepare(*REAL)
    assert value['schema']==m.TAG and value['authority_order']==list(m.ORDER)
    assert len(value['bindings'])>=9 and value['authority']['paths']['physical_report']==str(REAL[4].resolve())
    assert value['authority']['paths']['numeric_report']==str(REAL[7].resolve())

def test_public_order_is_not_internal_order():
    assert m.ORDER[3:6]==('physical_manifest','physical_report','physical_receipt')
    assert m.ORDER[6:]==('numeric_manifest','numeric_report','numeric_receipt')

def test_translation_calls_delegate_goal_numeric_physical(monkeypatch,tmp_path):
    seen=[]
    def fake(*args): seen.append(args); return dict(schema='x',protocol={},authority={'paths':{}},bindings={})
    monkeypatch.setattr(m.impl,'prepare',fake); monkeypatch.setattr(m.impl,'sha',lambda p:'a'*64)
    monkeypatch.setattr(m,'__file__',str(tmp_path/'v4.py'))
    m.prepare(*[tmp_path/x for x in ('g0','g1','g2','p0','p1','p2','n0','n1','n2')])
    assert seen[0][3:6]==(Path(tmp_path/'n0'),Path(tmp_path/'n1'),Path(tmp_path/'n2'))
    assert seen[0][6:]==(Path(tmp_path/'p0'),Path(tmp_path/'p1'),Path(tmp_path/'p2'))

def test_wrong_arity_and_tamper_fail(tmp_path):
    with pytest.raises(ValueError,match=r'goal\(3\)'):
        m.prepare(*([tmp_path/'x']*8))
    with pytest.raises((ValueError,KeyError)):
        m.verify({'schema':m.TAG,'protocol':m.protocol(),'authority_order':list(m.ORDER),'bindings':{},'authority':{}})

def test_v2_v3_untouched():
    assert m.impl.sha(ROOT/'experiments/spectral_reachability/agent_p5_terminal_goal_attribution_20261028_v2.py')=='849d7647110b663414d7ac989ce923887c4fec556021f701408e8c4ca9c42a25'
    assert m.impl.sha(ROOT/'experiments/spectral_reachability/agent_p5_terminal_goal_attribution_20261028_v3.py')=='aa252fc8de5b103ed62ce4f08494c866ff6a18b5ca12c3770b74f71ee2d90d72'
