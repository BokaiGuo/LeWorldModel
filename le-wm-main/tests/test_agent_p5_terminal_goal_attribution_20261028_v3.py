"""v3 authority-order and delegation tests; no model/CUDA/physics."""
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v3 as m

def test_single_public_order():
    assert m.ORDER==('goal_prepare','goal_report','goal_receipt','physical_manifest','physical_report','physical_receipt','numeric_manifest','numeric_report','numeric_receipt')
    assert m.protocol()['authority_order']==list(m.ORDER)

def test_real_paths_are_translated_goal_physical_numeric(monkeypatch,tmp_path):
    paths=[tmp_path/x for x in ('g0','g1','g2','p0','p1','p2','n0','n1','n2')]
    seen=[]
    def fake(*args): seen.append(args); return dict(schema='v2',protocol={},authority={'paths':{'goal_prepare':str(paths[0]),'goal_report':str(paths[1]),'goal_receipt':str(paths[2]),'phys_manifest':str(paths[3]),'phys_report':str(paths[4]),'phys_receipt':str(paths[5]),'num_manifest':str(paths[6]),'num_report':str(paths[7]),'num_receipt':str(paths[8])}},bindings={})
    monkeypatch.setattr(m.impl,'prepare',fake); monkeypatch.setattr(m.impl,'sha',lambda p:'a'*64)
    monkeypatch.setattr(m,'__file__',str(tmp_path/'v3.py'))
    out=m.prepare(*paths)
    assert seen==[tuple(paths)]
    assert out['schema']==m.TAG and out['authority_order']==list(m.ORDER)

def test_wrong_arity_fail_closed(tmp_path):
    with pytest.raises(ValueError,match=r'goal\(3\)'):
        m.prepare(*([tmp_path/'x']*8))

def test_verify_rejects_wrong_order_or_schema():
    base={'schema':m.TAG,'protocol':m.protocol(),'authority_order':list(m.ORDER),'bindings':{},'authority':{'paths':{}}}
    with pytest.raises((ValueError,KeyError)): m.verify(base)
    wrong=dict(base); wrong['authority_order']=list(reversed(m.ORDER))
    with pytest.raises(ValueError,match='v3 manifest/order'): m.verify(wrong)

def test_v2_frozen_source_bound():
    assert m.impl.sha(m.ROOT/'experiments/spectral_reachability/agent_p5_terminal_goal_attribution_20261028_v2.py')=='849d7647110b663414d7ac989ce923887c4fec556021f701408e8c4ca9c42a25'
    assert (m.ROOT/'innerdocs'/(m.TAG+'.md')).is_file()

def test_cli_requires_nine_ordered_paths(tmp_path):
    with pytest.raises(ValueError,match='authority order'):
        m.main(['prepare','--manifest',str(tmp_path/'out')])
