"""RAM/saved arithmetic only; no torch, model, physics or network."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_safeguard_factorial_20261028_v1 as m
import test_agent_p2_contextual_midfeedback_recompile_20261028_v2 as prior


def item(i,q):
    commands=np.full((5,2),np.float32(i*.1)).astype(float).tolist()
    return dict(index=i,commands=commands,raw_bits=m.v2.suffix_bits(commands),prediction=dict(cost=q))


def incumbent(q):
    x=item(0,q);del x['index'];return x


def test_guard_is_the_only_factor_on_identical_four_proposals():
    xs=[item(i,float(i)) for i in range(1,5)]
    assert m.choose(incumbent(.5),xs,False)['selected_index']==1
    assert m.choose(incumbent(.5),xs,True)['selected_index']==0
    assert m.choose(incumbent(2),xs,False)['selected_index']==m.choose(incumbent(2),xs,True)['selected_index']==1


def test_stable_ties_and_no_favorable_subset_or_outcome_abi():
    xs=[item(i,1.) for i in range(1,5)]
    assert m.choose(incumbent(1),xs,True)['selected_index']==0
    assert m.choose(incumbent(1),xs,False)['selected_index']==1
    xs[3]['prediction']=None
    assert m.choose(incumbent(1),xs,True) is None
    with pytest.raises(ValueError):m.choose(incumbent(1),xs[:3],False)
    xs[0]['native_success']=True
    with pytest.raises(ValueError):m.choose(incumbent(1),xs,False)


def test_anchor_only_changes_same_formula_not_goal_alpha_or_old_incumbent():
    b=prior.branch();saved=deepcopy(b);j=np.zeros((192,10));j[:10]=np.eye(10);z=np.ones(192)*.2
    old=m.proposals(b,'incumbent',z,j)
    assert old['logits']==m.v2.proposals(b,z,j)['logits'][1:]
    zero=m.proposals(b,'zero',z,j);local=deepcopy(b);local['incumbent_logits']=[0.]*10
    assert zero['logits']==m.v2.proposals(local,z,j)['logits'][1:]
    assert b==saved and m.contract()['alphas']==[1.,.5,.25,.125]
    for p in (old,zero):
        assert len(p['logits'])==4
        assert all(m.v2.legality(b,u)['legal'] for u in p['logits'])


def test_zero_gradient_no_dedup_and_exact_chart_not_clipping():
    b=prior.branch();p=m.proposals(b,'zero',b['goal'],np.zeros((192,10)))
    assert len(p['logits'])==4 and p['logits']==[[0.]*10]*4
    b['incumbent_logits'][0]=10
    with pytest.raises(ValueError):m.proposals(b,'incumbent',b['goal'],np.zeros((192,10)))


def test_saved_prediction_join_requires_context_goal_and_exact_repeated_outputs():
    b,s,r=prior.execute();e=r['evaluations'][0]
    c=m.candidate(b,e['commands'],e['context'],dict(test='RAM'))
    assert m.find_prediction(b,e['commands'],[c])['cost']==e['cost']
    assert m.find_prediction(b,np.zeros((5,2),np.float32).tolist(),[c]) is None
    for key,val in (('context_sha256','0'*64),('goal',[99.]*192),('cost',999.)):
        bad=deepcopy(c);bad[key]=val
        with pytest.raises(ValueError):m.find_prediction(b,e['commands'],[bad])
    bad=deepcopy(c);bad['context']['latent_history'][0][0]+=.1
    with pytest.raises(ValueError):m.find_prediction(b,e['commands'],[c,bad])


def test_physical_match_is_exact_raw_and_never_reads_outcomes():
    raw=np.zeros((10,2),np.float32).tolist()
    class NoOutcome(dict):
        def __getitem__(self,k):
            assert k in ('commands','authority');return super().__getitem__(k)
    assert m.physical_matches(raw,[NoOutcome(commands=raw,authority=dict(receipt='saved'))])==[dict(receipt='saved')]
    other=deepcopy(raw);other[0][0]=float(np.float32(1e-4))
    assert m.physical_matches(other,[dict(commands=raw,authority={})])==[]


def test_no_run_or_scientific_greenlight_or_equal_flops():
    c=m.contract()
    assert c['branches']==8 and c['cells']==32 and c['proposal_slots']==64
    assert c['new_jacobian']==c['new_encoder']==c['new_physics']==0
    assert c['run_available'] is c['P1_admission'] is c['P2_admission'] is c['physical_release'] is False
    assert c['total_FLOPs'] is None and c['equal_FLOPs'] is False


@pytest.fixture(scope='module')
def saved_table():
    root=m.ROOT/'results';outputs=[root/f'agent_p2_contextual_midfeedback_recompile_20261028_v{v}_cuda_MAIN01' for v in (1,2)]
    if not all((x/'report.json').exists() for x in outputs):pytest.skip('actual sealed CUDA reports unavailable; no fabricated seam')
    reports=[m.v2.physical.read(x/'report.json') for x in outputs]
    assert [m.sha(x/'report.json') for x in outputs]==list(m.REPORTS)
    p1,p2=[m.v2.physical.read(r['prepared']) for r in reports]
    assert len(p1['branches'])==len(p2['branches'])==8
    result=[]
    for i,b in enumerate(p2['branches']):
        result.append(m.branch_table(i,b,reports[0]['rows'][i],reports[1]['rows'][i],p2['plans'][i],
            lambda e:m.v1.load_npy(outputs[0],e),lambda e:m.v1.load_npy(outputs[1],e),[]))
    return result


def test_actual_saved_full8_all32_cells_all64_proposals(saved_table):
    assert [r['index'] for r in saved_table]==list(range(8))
    assert len({r['root_slot'] for r in saved_table})==2
    assert sum(len(a['cells']) for r in saved_table for a in r['controlled_axes'])==32
    assert sum(len(a['candidates']) for r in saved_table for a in r['controlled_axes'])==64
    for r in saved_table:
        assert len(r['legacy_confounded_reconstruction'])==4
        assert all(a['cells'][j]['selection'] is not None for a in r['controlled_axes'] if a['anchor']=='incumbent' for j in (0,1))


def test_actual_saved_missing_requests_are_exact_full_native_inputs(saved_table):
    qs=[q for r in saved_table for q in r['missing_requests']]
    assert len(qs)==32 and {q['branch'] for q in qs}==set(range(8))
    assert all(q['anchor']=='zero' and q['domain']['legal'] for q in qs)
    for q in qs:
        assert np.array_equal(np.asarray(q['commands']),m.v2.chart(q['logits']))
        assert q['raw_bits']==m.v2.suffix_bits(q['commands'])
        assert np.asarray(q['native_input']['action_history']).shape==(3,10)
        assert np.asarray(q['native_input']['latent_history']).shape==(3,192)


def test_external_authority_rejects_before_native_consume(tmp_path,monkeypatch):
    (tmp_path/'report.json').write_text('{}')
    monkeypatch.setattr(m.v1,'consume_compile',lambda *a:pytest.fail('untrusted report reached native consume'))
    with pytest.raises(ValueError):m.authority(tmp_path,1)


def test_own_prepare_hash_rejects_before_rebuild(tmp_path,monkeypatch):
    f=tmp_path/'fake.json';f.write_text('{}')
    monkeypatch.setattr(m,'build',lambda **kw:pytest.fail('bad own SHA reached rebuild'))
    with pytest.raises(ValueError):m.verify(f,'0'*64)
