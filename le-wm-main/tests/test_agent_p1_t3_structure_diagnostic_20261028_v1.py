"""Synthetic exact arithmetic only; no production closure/model/physics execution."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import itertools
import json

import pytest

from experiments.spectral_reachability import agent_p1_t3_structure_diagnostic_20261028_v1 as m


def packed(x):
    x = F(x)
    return dict(numerator=str(x.numerator), denominator=str(x.denominator))


def row(pred=1, terms=(-2, 0, 0), i=0, j=1):
    pred = F(pred); terms = tuple(map(F, terms)); obs = pred+sum(terms)
    exact = dict(predicted_margin=packed(pred), observed_margin=packed(obs),
                 observed_minus_predicted_margin=packed(obs-pred),
                 identity_residual=packed(0), factorization_residual=packed(0))
    exact.update({k:packed(v) for k,v in zip(m.TERMS, terms, strict=True)})
    return dict(i=i,j=j,anchor_index=i//5,missing_indices=[],exact=exact,
                float64=dict(predicted_margin=float(pred),observed_margin=float(obs)),
                model_to_observed=dict(classification='discordant' if pred*obs<0 else
                    'both_tie' if pred==obs==0 else 'left_tie_only' if pred==0 else
                    'right_tie_only' if obs==0 else 'concordant',
                    delta_a=float(pred),delta_b=float(obs)))


def snapshot():
    pairs=[]
    for a in range(8):
        for i,j in itertools.combinations(range(a*5,a*5+5),2):
            r=row(i=i,j=j)
            if 12 in (i,j):
                r.update(missing_indices=[12],exact=None,float64=None,
                         model_to_observed=dict(classification='unknown',delta_a=1.0,delta_b=None))
            pairs.append(r)
    return dict(schema=m.UPSTREAM_NAME+'.analysis',status='saved_T3_T4_exact_pair_algebra_not_admitted',
        fixed_slots=40,observed_latent_slots=39,missing_indices=[12],fixed_pairs=80,
        known_roots=2,fresh_roots=0,P1_admission=False,fresh_confirmation=False,
        protocol=dict(MSE_divisor=192,metric_operator='identity192',actual_absolute_step=20,
                      teacher_context_absolute_step=15),pairs=pairs)


def test_common_only_crossing_and_all_fixed_subsets():
    out=m.analyze_pair(row())
    assert len(out['counterfactuals'])==8
    assert [x['retained_mask'] for x in out['counterfactuals']]==list(range(8))
    b=out['contributions'][m.TERMS[0]]
    assert b['alone_strict_crossing'] and b['necessary_for_full_strict_crossing']
    assert b['oriented_progress']==packed(2)
    assert out['counterfactuals'][0]['margin']==packed(1)
    assert out['counterfactuals'][7]['margin']==packed(-1)


def test_relative_response_only_crossing():
    out=m.analyze_pair(row(1,(0,-3,1)))
    assert not out['contributions'][m.TERMS[0]]['necessary_for_full_strict_crossing']
    assert out['contributions'][m.TERMS[1]]['necessary_for_full_strict_crossing']
    assert out['contributions'][m.TERMS[2]]['direction']=='opposes_crossing'
    assert out['counterfactuals'][6]['strict_crossing']


def test_cancellation_no_fake_unique_cause():
    out=m.analyze_pair(row(1,(-3,3,0)))
    assert out['cancellation']['cancelled_l1_mass']==packed(6)
    assert out['cancellation']['opposing_mass']==packed(3)
    assert out['contributions'][m.TERMS[0]]['alone_strict_crossing']
    assert not out['full_exact']['strict_crossing']
    assert not any(v['necessary_for_full_strict_crossing'] for v in out['contributions'].values())


def test_joint_crossing_not_individually_sufficient():
    out=m.analyze_pair(row(1,(-F(3,5),-F(3,5),0)))
    assert out['full_exact']['strict_crossing']
    for key in m.TERMS[:2]:
        assert not out['contributions'][key]['alone_strict_crossing']
        assert out['contributions'][key]['necessary_for_full_strict_crossing']


@pytest.mark.parametrize('pred,terms',[(1,(-1,0,0)),(-1,(1,0,0)),(0,(-1,0,0)),(0,(0,0,0))])
def test_ties_are_not_strict_crossings(pred,terms):
    out=m.analyze_pair(row(pred,terms))
    assert not out['full_exact']['strict_crossing']
    if pred==0:
        assert out['margin_threshold'] is None
        assert all(v['oriented_progress'] is None for v in out['contributions'].values())
    else:
        assert out['full_exact']['classification']=='right_tie_only'


def test_negative_pred_margin_orientation():
    out=m.analyze_pair(row(-1,(2,0,0)))
    assert out['contributions'][m.TERMS[0]]['oriented_progress']==packed(2)
    assert out['full_exact']['strict_crossing']


def test_quadratic_positive_can_itself_cross_negative_margin():
    out=m.analyze_pair(row(-1,(0,0,2)))
    assert out['contributions'][m.TERMS[2]]['alone_strict_crossing']
    assert out['contributions'][m.TERMS[2]]['direction']=='toward_crossing'


def test_negative_quadratic_rejected_even_with_exact_identity():
    with pytest.raises(ValueError,match='nonnegative'):
        m.analyze_pair(row(1,(0,0,-2)))


def test_all_zero_correction_cancellation_fraction_null():
    out=m.analyze_pair(row(1,(0,0,0)))
    assert out['cancellation']['cancelled_l1_fraction'] is None
    assert out['cancellation']['cancelled_l1_mass']==packed(0)


def test_masks_have_exact_full_and_empty_endpoints_for_fraction_inputs():
    out=m.analyze_pair(row(F(4,41),(-F(40,41),F(123,37),F(1,11))))
    assert out['counterfactuals'][0]['margin']==packed(F(4,41))
    assert out['counterfactuals'][7]['margin']==out['full_exact']['observed_margin']
    for cf in out['counterfactuals']:
        assert len(cf['retained_terms'])+len(cf['removed_terms'])==3
        assert set(cf['retained_terms']).isdisjoint(cf['removed_terms'])


def test_unknown_and_fixed_denominator():
    out=m.assemble(snapshot())
    assert len(out['pairs'])==80
    assert out['summary']['available']==76 and out['summary']['unknown']==4
    assert [(p['i'],p['j']) for p in out['pairs'] if not p['available']]==[(10,12),(11,12),(12,13),(12,14)]
    assert all(p['counterfactuals'] is None for p in out['pairs'] if not p['available'])
    assert all(x['fixed_pairs']==80 and x['unknown']==4 for x in out['summary']['counterfactuals'])


@pytest.mark.parametrize('change', ['order','missing','anchor','identity','negative_quadratic','residual','duplicate','protocol','admission'])
def test_fail_closed_structure(change):
    s=snapshot()
    if change=='order': s['pairs'][0],s['pairs'][1]=s['pairs'][1],s['pairs'][0]
    if change=='missing': s['pairs'][0]['missing_indices']=[12]
    if change=='anchor': s['pairs'][0]['anchor_index']=1
    if change=='identity': s['pairs'][0]['exact']['observed_margin']=packed(99)
    if change=='negative_quadratic': s['pairs'][0]['exact'][m.TERMS[2]]=packed(-1)
    if change=='residual': s['pairs'][0]['exact']['identity_residual']=packed(1)
    if change=='duplicate': s['pairs'][1]=deepcopy(s['pairs'][0])
    if change=='protocol': s['protocol']['MSE_divisor']=192**2
    if change=='admission': s['P1_admission']=True
    with pytest.raises(ValueError): m.assemble(s)


def test_float64_signs_retained_not_snapped_to_exact():
    r=row();r['float64']['observed_margin']=0.0
    r['model_to_observed'].update(delta_b=0.0,classification='right_tie_only')
    out=m.analyze_pair(r)
    assert out['original_float64']['classification']=='right_tie_only'
    assert out['full_exact']['classification']=='discordant'
    assert out['original_float64']['signs_equal_exact'] is False


def test_no_physical_label_dependence():
    r=row();a=m.analyze_pair(r)
    r.update(model_to_physical={'fake':'changed'},physical_lower_is_better_deltas={'x':999})
    assert m.analyze_pair(r)==a


@pytest.mark.parametrize('bad',[{'numerator':'1','denominator':'0'}, {'numerator':'2','denominator':'2'},
                                 {'numerator':'01','denominator':'1'}, {'numerator':1,'denominator':'1'}])
def test_fraction_encoding_fail_closed(bad):
    r=row();r['exact']['predicted_margin']=bad
    with pytest.raises(ValueError): m.analyze_pair(r)


def test_read_sha_and_duplicate_keys(tmp_path):
    p=tmp_path/'input.json';p.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError,match='duplicate'): m.read(p,hashlib.sha256(p.read_bytes()).hexdigest())
    p.write_bytes(b'{}')
    with pytest.raises(ValueError,match='SHA'):m.read(p,'0'*64)


def test_pinned_snapshot_and_frozen_sources(tmp_path,monkeypatch):
    p=tmp_path/'snapshot.json';value=snapshot()
    f=tmp_path/'frozen.py';f.write_bytes(b'frozen')
    value['source_files']={str(f):hashlib.sha256(f.read_bytes()).hexdigest()}
    p.write_text(json.dumps(value))
    monkeypatch.setattr(m,'SOURCE',p)
    monkeypatch.setattr(m,'SOURCE_SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    monkeypatch.setattr(m,'FROZEN_SOURCES',value['source_files'])
    assert m.load_source()==value
    f.write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen source'):m.load_source()


def test_exclusive_writer_and_full_own_verify(tmp_path,monkeypatch):
    root=tmp_path; (root/'results').mkdir();monkeypatch.setattr(m,'ROOT',root)
    value=m.assemble(snapshot());p=root/'results'/f'{m.NAME}_fixture.json'
    receipt=m.write(p,value)
    assert m.verify(p,receipt['analysis_sha256'],value)['verified']
    with pytest.raises(FileExistsError):m.write(p,value)
    changed=deepcopy(value);changed['fixed_pairs']=79
    with pytest.raises(ValueError):m.verify(p,receipt['analysis_sha256'],changed)
    with pytest.raises(ValueError):m.write(root/'results'/'other.json',value)


def test_build_compact_authority_and_explicit_closure_boundary(monkeypatch):
    value=snapshot()
    value['authority']=dict(full_saved_native_rebuilt=True,closure_digest='fixture')
    monkeypatch.setattr(m,'load_source',lambda:deepcopy(value))
    out=m.build()
    assert out['authority']['input_full_saved_native_rebuilt'] is True
    assert out['authority']['upstream_closure_rebuilt_by_this_consumer'] is False
    assert len(out['source_files'])==3
    assert out['protocol']['choose_best_mask'] is False
    assert len(m.canonical(out))<m.CAP
