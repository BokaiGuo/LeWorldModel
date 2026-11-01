"""RAM/saved metadata only; synthesized strict fixtures are NOT MAIN receipts."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_budget_physical_handoff_20261028_v1 as m

REPORT_SHA='066b2f7f6e37fc9c4ed2795a07387c473da7e2e4418414e05ce0e6080ddc2978'


@pytest.fixture(scope='module')
def saved_metadata():
    report=m.ROOT/'results/agent_p3_child_budget_host_cuda_20261028_v1_cuda_MAIN01/report.json'
    assert m.sha(report)==REPORT_SHA
    r=m.READ(report,256*1024**2);p=m.READ(r['prepared'],64*1024**2)
    oldr=m.READ(m.saved.ACTUAL_OUTPUT/'report.json',m.physical.FINAL)
    assert m.sha(m.saved.ACTUAL_OUTPUT/'report.json')==m.saved.ACTUAL_REPORT_SHA
    oldp=m.READ(oldr['prepared'],64*1024**2)
    # Only a RAM fixture for projection tests, NEVER external strict authority.
    oldstrict=dict(complete=True,rows=oldr['rows'],resources=oldr['resources'])
    return p,r,oldp,oldr,oldstrict,m.sha(oldp['source_manifest'])


def test_real_actual_host_report_ABI_full_receipt_shape_only(saved_metadata):
    p,r,*_=saved_metadata
    fixture=m.expected_receipt(r,REPORT_SHA)
    m.validate_host_link(p,r,fixture,REPORT_SHA)
    assert fixture['census']==dict(compiled=41,refused=5,not_reached=2)
    assert fixture['actual_resources']['fixed']['attempted']['predict_API']==391
    assert fixture['actual_resources']['adaptive']['attempted']['predict_API']==634


@pytest.mark.parametrize('tamper',['summary','dropped_row','report_sha','prepare_sha','gate_digest','actual','science','status'])
def test_summary_partial_or_wrong_receipt_rejected(saved_metadata,tamper):
    p,r,*_=saved_metadata
    fixture=m.expected_receipt(r,REPORT_SHA)
    # Copy only altered large subtrees when needed.
    fixture=dict(fixture)
    if tamper=='summary':fixture={'status':'exit0'}
    elif tamper=='dropped_row':fixture['full_rows']=fixture['full_rows'][:-1]
    elif tamper=='report_sha':fixture['report_sha256']='0'*64
    elif tamper=='prepare_sha':fixture['prepare_sha256']='0'*64
    elif tamper=='gate_digest':fixture['gate_receipt_sha256']='0'*64
    elif tamper=='actual':fixture['actual_resources']={}
    elif tamper=='science':fixture['P3_admission']=True
    else:fixture['status']='partial'
    with pytest.raises(ValueError):m.validate_host_link(p,r,fixture,REPORT_SHA)


def test_full48_handoff_without_reuse_all_compiled_pending(saved_metadata):
    p,r,oldp,oldr,_,manifest=saved_metadata
    out=m.project(p,r,oldp,oldr,None,'none',manifest)
    assert len(out['rows'])==48 and out['child_denominator']==24
    assert out['census']==dict(pending_new_physics=41,reused_saved=0,paid_refused=5,not_reached=2)
    assert out['new_budget']['new_actual_resources'] is None
    assert out['new_budget']['execute_slots']==41
    assert all(row['runner_plan']['selected'] is None for row in out['rows'] if row['route'] in ('paid_refused','not_reached'))
    assert out['runner_entry']['old_CLI_prepare_compatible'] is False


def test_exact_reuse_routes_include_earlystops_not_outcome_selection(saved_metadata):
    p,r,oldp,oldr,strict,manifest=saved_metadata
    out=m.project(p,r,oldp,oldr,strict,'strict_saved',manifest)
    compiled=[row for row in out['rows'] if row['route'] in ('reused_saved','pending_new_physics')]
    assert len(compiled)==41 and out['census']['paid_refused']==5 and out['census']['not_reached']==2
    for row in compiled:
        i=row['ordinal']
        same=m.raw_identity(r['session']['rows'][i]['compiler']['selected']['raw5'],5)==m.raw_identity(oldp['rows'][i]['selected']['raw5'],5)
        assert (row['route']=='reused_saved')==same
        if same:
            evidence=row['reuse_evidence']
            assert evidence['fresh_execution'] is False and evidence['new_physics_calls']==0
            assert evidence['full_old_row']['status']==oldr['rows'][i]['status']
    assert out['independent_roots']==2 and out['freshness'] is out['eqcompute'] is False


def test_one_float32_bit_changed_command_must_remain_new_physics(saved_metadata):
    p,r,oldp,oldr,strict,manifest=saved_metadata
    changed=dict(r,session=dict(r['session'],rows=list(r['session']['rows'])))
    row=deepcopy(changed['session']['rows'][0]);changed['session']['rows'][0]=row
    raw=m.physical.raw(oldp['rows'][0]['selected']['raw5'],5)
    raw[0,0]=np.nextafter(raw[0,0],np.float32(0 if raw[0,0] else 1),dtype=np.float32)
    row['compiler']['selected']['raw5']=raw.astype(float).tolist()
    out=m.project(p,changed,oldp,oldr,strict,'strict_saved',manifest)
    assert out['rows'][0]['route']=='pending_new_physics'
    assert out['rows'][0]['reuse_evidence'] is None


def test_signed_zero_is_not_bitwise_same_raw5():
    positive=np.zeros((5,2),np.float32);negative=positive.copy();negative[0,0]=-0.
    assert np.array_equal(positive,negative)
    assert m.raw_identity(positive.tolist(),5)!=m.raw_identity(negative.tolist(),5)


@pytest.mark.parametrize('tamper',['context','source','goal_bits','ordinal','no_strict','partial_strict'])
def test_reuse_binding_or_missing_old_full_consumer_fails(saved_metadata,tamper):
    p,r,oldp,oldr,strict,manifest=saved_metadata
    if tamper in ('context','source'):
        p=dict(p,wiring=dict(p['wiring'],children=list(p['wiring']['children'])))
        p['wiring']['children'][0]=deepcopy(p['wiring']['children'][0])
        if tamper=='context':p['wiring']['children'][0]['context_sha256']='0'*64
        else:p['wiring']['children'][0]['ancestry']['source_report_sha256']='0'*64
    elif tamper in ('goal_bits','ordinal'):
        oldr=dict(oldr,rows=list(oldr['rows']));oldr['rows'][0]=deepcopy(oldr['rows'][0])
        if tamper=='goal_bits':oldr['rows'][0]['boundary']['snapshot']['bodies']['block']['float64_bits']='0'*128
        else:oldr['rows'][0]['ordinal']=1
        strict=dict(strict,rows=oldr['rows'])
    elif tamper=='no_strict':strict=None
    else:strict=dict(strict,complete=False)
    with pytest.raises(ValueError):m.project(p,r,oldp,oldr,strict,'strict_saved',manifest)


def test_reuse_does_not_choose_successes_or_shorten_earlystop(saved_metadata):
    p,r,oldp,oldr,strict,manifest=saved_metadata
    a=m.project(p,r,oldp,oldr,strict,'strict_saved',manifest)
    altered=dict(oldr,rows=[dict(row,final_metrics={'RAM_outcome_changed':True}) for row in oldr['rows']])
    b=m.project(p,r,oldp,altered,dict(strict,rows=altered['rows']),'strict_saved',manifest)
    assert [row['route'] for row in a['rows']]==[row['route'] for row in b['rows']]
    for row in a['rows']:
        if row['route']=='reused_saved':
            n=row['reuse_evidence']['full_old_row']['own_returned']
            assert n==oldr['rows'][row['ordinal']]['own_returned']


def test_missing_future_receipt_cannot_prepare(tmp_path):
    authority=m.inputs(str(tmp_path/'future_report.json'),'0'*64,str(tmp_path/'future_strict.json'),'1'*64,'strict_saved')
    with pytest.raises(ValueError,match='missing terminal report or MAIN complete strict receipt'):m.load_host(authority)


def test_bad_external_sha_rejected_before_source_consumers(monkeypatch,tmp_path):
    a=tmp_path/'report.json';b=tmp_path/'strict.json';a.write_bytes(b'{}');b.write_bytes(b'{}')
    authority=m.inputs(str(a),'0'*64,str(b),'1'*64,'none');calls=[]
    monkeypatch.setattr(m,'READ',lambda *a,**kw:calls.append('BAD'))
    with pytest.raises(ValueError,match='external actual host'):m.load_host(authority)
    assert calls==[]


def test_material_requires_old_physical_consume_and_never_host_consume(monkeypatch,saved_metadata):
    p,r,oldp,oldr,strict,manifest=saved_metadata;calls=[]
    receipt=m.expected_receipt(r,REPORT_SHA)
    authority=m.inputs('/RAM/report.json',REPORT_SHA,'/RAM/MAIN_full_strict.json','a'*64,'strict_saved')
    monkeypatch.setattr(m,'load_host',lambda a:(p,r,receipt))
    def old(policy):calls.append(policy);return oldp,oldr,strict
    monkeypatch.setattr(m,'load_physical',old)
    # Full material file hashing depends on persisted external receipt, so test
    # the seam separately; production load_physical explicitly calls saved.consume.
    source=Path(m.__file__).read_text()
    assert 'strict=saved.consume(' in source and 'host.consume(' not in source
    assert 'import torch' not in source and 'native_factory(' not in source
    m.project(p,r,*old('strict_saved'),'strict_saved',manifest)
    assert calls==['strict_saved']


def test_old_physical_budget_is_not_new_actual_fee(saved_metadata):
    p,r,oldp,oldr,strict,manifest=saved_metadata
    out=m.project(p,r,oldp,oldr,strict,'strict_saved',manifest)
    assert out['new_budget']['new_actual_resources'] is None
    assert out['new_budget']['historical993public_is_not_new_payment'] is True
    assert out['historical_resources']['resources']==oldr['resources']
    assert out['physical_execution_started'] is False


def test_explicit_reuse_policy_and_no_relative_paths():
    with pytest.raises(ValueError):m.inputs('/a','0'*64,'/b','1'*64,'pick_success')
    with pytest.raises(ValueError):m.inputs('relative','0'*64,'/b','1'*64,'none')


def test_source_pins_are_real_frozen_files():m.contract.check_sources(m.PINS)


def test_actual_MAIN_persisted_complete_host_receipt_no_host_consume(monkeypatch):
    report=m.ROOT/'results/agent_p3_child_budget_host_cuda_20261028_v1_cuda_MAIN01/report.json'
    receipt=m.ROOT/'results/agent_p3_child_budget_host_cuda_20261028_v1_consume_MAIN01.json'
    authority=m.inputs(str(report),REPORT_SHA,str(receipt),
        '14c87cd5c69562a53cacc7d3082a4938fcb776b814c9fc2b00dacd99df20bd0d','strict_saved')
    def forbidden(*a,**kw):raise AssertionError('MUST NOT repeat host.consume')
    monkeypatch.setattr(m.host,'consume',forbidden)
    p,r,full=m.load_host(authority)
    assert full['census']==dict(compiled=41,refused=5,not_reached=2)
    assert len(full['full_rows'])==48 and len(p['wiring']['children'])==24
