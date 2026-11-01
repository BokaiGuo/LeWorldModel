"""No torch/physics: all42 numerical callback library and saved authority seams."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
import test_agent_p2_three_program_20261028_v1 as fixture
from experiments.spectral_reachability import agent_p2_three_program_model_mechanism_20261029_v1 as m


@pytest.fixture(scope='module')
def data():
    slots=[fixture.slot(i) for i in (0,1)];backend=fixture.Numeric();result=m.planner.plan(slots[0],backend)
    p=dict(source_report_sha256='1'*64,expected_model_hash='2'*64)
    rows=[]
    for s in slots:
        rows.append(dict(index=s['root_slot'],status='sealed',result=deepcopy(result),
            plans=[m.planner.model_plan(p,s,result,a) for a in m.planner.ARMS]))
    return slots,rows


def test_all_denominators_full_library_no_selection_and_unknown_physical(data):
    slots,rows=data;out=m.compute(slots,rows)
    assert out['arm_candidate_denominator']==168 and out['paired_candidate_denominator']==84
    assert len(out['paired_candidates'])==84 and len(out['all_unordered_model_rank_pairs'])==1722
    assert [(r['root_slot'],r['index']) for r in out['paired_candidates']]==[(s,i) for s in (0,1) for i in range(42)]
    assert [(r['root_slot'],r['left_index'],r['right_index']) for r in out['all_unordered_model_rank_pairs']]==[
        (s,i,j) for s in (0,1) for i in range(42) for j in range(i+1,42)]
    for root in out['roots']:
        assert sum(root['counts']['rank_categories'].values())==861
        assert root['counts']['terminal_worse']==0
        assert sum(root['counts'][k] for k in ('terminal_improve','terminal_tie','terminal_worse'))==42
        for selected in root['source_sealed_winners']:
            original=rows[root['root_slot']]['result']['winners'][selected['arm']]
            assert selected['index']==original['index'] and selected['model_Q']==original['cost']
            assert selected['new_selection'] is False and selected['native_success'] is None
    for r in out['paired_candidates']:
        assert r['native_success'] is r['physical_metrics'] is r['actual_latent_Q'] is r['physical_non_greedy_capability'] is None
        assert r['telescoping_exact'] and len(r['cp5_recompile']['all5_Q'])==len(r['cp10_recompile']['all5_Q'])==5
        d2=r['cp5_recompile']['selected_delta'];d3=r['cp10_recompile']['selected_delta'];total=r['composed_minus_joint']
        value=lambda d:m.Fraction(int(d['numerator']),int(d['denominator']))
        assert value(d2)+value(d3)==value(total)
    assert 'torch' not in sys.modules


def test_zero_and_small_margin_signs_no_epsilon():
    a=1.;b=float(np.nextafter(a,np.inf));d=m.exact_delta(a,b)
    assert d['sign']==-1 and d['value']<0 and m.exact_delta(a,a)['sign']==0
    assert m.exact_delta(-0.,0.)['sign']==0
    for v in (True,float('nan'),float('inf')):
        with pytest.raises(ValueError):m.sign(v)


def test_protocol_cannot_claim_independent_84_roots_or_physical_mechanism():
    p=m.protocol();assert p['distinct_known_roots']==2 and p['independent_confirmation_samples']==0 and p['new_model_calls']==p['new_physics_calls']==0
    assert p['no_new_selection'] and p['no_physical_inputs'] and p['all_ties_retained']
    assert not any(p[k] for k in ('fresh','P1_admission','P2_mechanism','physical_release','factorial_causality'))
    assert p['non_greedy_physical_capability'] is p['native_success'] is p['actual_latent_Q'] is None


def test_all_ties_preserved_not_reclassified_as_reversal(data):
    slots,rows=data;rows=deepcopy(rows)
    # Full original finite library, but all contexts zero and all guarded scores tied.
    for row in rows:
        initial=slots[row['index']]['packet']['root_context']
        value=m.q(initial,slots[row['index']]['packet']['goal'])
        r=row['result']
        for arm in m.planner.ARMS:
            for c in r['candidates'][arm]:
                c['contexts']=[deepcopy(initial)]*3;c['cost']=value
                if arm==m.planner.ARMS[1]:
                    for label,size in (('edge2',2),('edge3',1)):
                        e=c[label];e['input_context']=deepcopy(initial);e['selected_index']=0
                        e['selected_contexts']=[deepcopy(initial)]*size;e['incumbent_cost']=e['selected_cost']=value
                        for v in e['evaluations']:v.update(contexts=[deepcopy(initial)]*size,cost=value)
            w=r['winners'][arm];w.update(index=0,cost=value,contexts=[deepcopy(initial)]*3)
        # Do not inject any physical label to decide what is a tie.
    out=m.compute(slots,rows)
    assert all(r['counts']['rank_categories']['both_tie']==861 for r in out['roots'])
    assert all(r['counts']['terminal_tie']==42 for r in out['roots'])


@pytest.mark.parametrize('change',['root_subset','slot_order','missing_candidate','missing_pair_arm','wrong_index','prefix_context','raw_prefix',
    'goal','Qcost','edge2context','edge3context','incumbent_path','localcost','localwinner','missing_evaluation','sealedwinner'])
def test_no_subsets_or_context_cost_selector_splices(data,change):
    slots,rows=data;slots=deepcopy(slots);rows=deepcopy(rows)
    j=rows[0]['result']['candidates'][m.planner.ARMS[0]][1]
    c=rows[0]['result']['candidates'][m.planner.ARMS[1]][1]
    if change=='root_subset':slots.pop();rows.pop()
    elif change=='slot_order':slots.reverse()
    elif change=='missing_candidate':rows[0]['result']['candidates'][m.planner.ARMS[1]].pop()
    elif change=='missing_pair_arm':rows[0]['result']['candidates'].pop(m.planner.ARMS[1])
    elif change=='wrong_index':c['index']=0
    elif change=='prefix_context':c['contexts'][0]['latent_history'][0][0]=3
    elif change=='raw_prefix':c['logits'][0]+=.01
    elif change=='goal':slots[0]['packet']['goal'][0]+=1
    elif change=='Qcost':j['cost']+=.1
    elif change=='edge2context':c['edge2']['input_context']['latent_history'][0][0]=3
    elif change=='edge3context':c['edge3']['input_context']['latent_history'][0][0]=3
    elif change=='incumbent_path':c['edge2']['evaluations'][0]['contexts'][0]['latent_history'][0][0]=3
    elif change=='localcost':c['edge3']['evaluations'][0]['cost']+=1
    elif change=='localwinner':c['edge2']['selected_index']=(c['edge2']['selected_index']+1)%5
    elif change=='missing_evaluation':c['edge3']['evaluations'].pop()
    else:rows[0]['result']['winners'][m.planner.ARMS[0]]['index']=99
    with pytest.raises(ValueError):m.compute(slots,rows)


def test_real_frozen_planner_pins_and_no_report_open():
    for rel,h in m.PINS.items():assert m.sha(m.ROOT/rel)==h
    assert m.PREP_SHA=='86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461'
    assert 'torch' not in sys.modules


@pytest.fixture
def authority(data,tmp_path,monkeypatch):
    slots,rows=data;output=tmp_path/'planner';output.mkdir();receipt=tmp_path/'strict.json';prep=tmp_path/'prepare.json'
    p=dict(slots=slots,bindings_sha256={});prep.write_bytes(m.raw(p));prepare_sha=m.sha(prep)
    artifacts={}
    for i,row in enumerate(rows):
        c=m.planner.zlib.compressobj(6,m.planner.zlib.DEFLATED,31);path=output/f'root{i}.json.gz'
        path.write_bytes(c.compress(m.raw(row))+c.flush());artifacts[path.name]=m.sha(path)
    source=dict(prepare_sha256=prepare_sha,prepared=str(prep),status='all4_full15_plans_sealed_not_admitted',artifact_sha256=artifacts)
    (output/'report.json').write_bytes(m.raw(source));report_sha=m.sha(output/'report.json')
    accepted=dict(P2_mechanism=False,plans=[{'RAM_ONLY':i} for i in range(4)])
    receipt.write_bytes(m.raw(accepted));receipt_sha=m.sha(receipt);calls=[]
    proxy=NS(**vars(m.planner))
    def consume(path,h):calls.append((str(path),h));assert h==report_sha;return deepcopy(accepted)
    proxy.consume=consume;monkeypatch.setattr(m,'planner',proxy)
    monkeypatch.setattr(m,'REPORT_SHA',report_sha);monkeypatch.setattr(m,'PREP_SHA',prepare_sha)
    return output,receipt,receipt_sha,calls


def test_saved_authority_calls_full_consumer_and_compares_entire_receipt(authority):
    output,receipt,h,calls=authority;p,rows,bindings=m.material(output,receipt,h)
    assert len(calls)==1 and len(rows)==2 and bindings[str(receipt)]==h
    assert str(output/'root0.json.gz') in bindings and str(Path(__file__).resolve()) in bindings
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('change',['receipt_hash','receipt_payload','partial_status','prepare_sha','dependency_sha'])
def test_saved_authority_rejects_weak_or_partial_authority(authority,change,monkeypatch):
    output,receipt,h,_=authority
    if change=='receipt_hash':receipt.write_bytes(receipt.read_bytes()+b' ')
    elif change=='receipt_payload':
        v=json.loads(receipt.read_bytes());v['invented']=True;receipt.write_bytes(m.raw(v));h=m.sha(receipt)
    elif change=='dependency_sha':monkeypatch.setattr(m,'PINS',{next(iter(m.PINS)):'0'*64})
    else:
        p=output/'report.json';v=json.loads(p.read_bytes())
        if change=='partial_status':v['status']='failed_partial'
        else:v['prepare_sha256']='e'*64
        p.write_bytes(m.raw(v));monkeypatch.setattr(m,'REPORT_SHA',m.sha(p))
    with pytest.raises(ValueError):m.material(output,receipt,h)


def test_complete_own_analysis_saved_verify_rebuild_and_coherent_report_tamper(authority,tmp_path,monkeypatch):
    output,receipt,h,_=authority
    report=m.build(output,receipt,h);path=tmp_path/'analysis.json';path.write_bytes(m.raw(report))
    assert m.verify(path,m.sha(path))['paired_candidates']==84
    report['model_analysis']['paired_candidates'][0]['prefix_Q']+=1
    path.write_bytes(m.raw(report))
    with pytest.raises(ValueError,match='entire all42'):m.verify(path,m.sha(path))


def test_writer_only_new_own_results_scope(tmp_path):
    with pytest.raises(ValueError):m.write(tmp_path/(m.NAME+'_not_results.json'),dict(status='RAM_ONLY'))
